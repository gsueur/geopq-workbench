//! geopq-cli: headless batch converter for scripted pipelines (cron jobs,
//! monthly refreshes) that don't want a GUI. Wraps the same import +
//! optimize machinery the app's Import/Export dialogs use: no window, no
//! GPU, just a source path in and an optimized GeoParquet out.

use std::path::{Path, PathBuf};

use clap::Parser;
use geopq_workbench::data::import::ImportFormat;
use geopq_workbench::data::info::fmt_bytes;
use geopq_workbench::data::optimize::{self, Codec, GpVersion, OptimizeOptions};
use geopq_workbench::data::partition::PartitionBy;
use geopq_workbench::data::source::Source;
use geopq_workbench::data::{geojson, gpkg, shp};

/// Convert a vector source (File Geodatabase, GeoPackage, Shapefile,
/// GeoJSON, or an existing GeoParquet file) into an optimized GeoParquet:
/// spatially sorted, tuned row groups, covering bbox column.
#[derive(Parser)]
#[command(name = "geopq-cli", version, about)]
struct Cli {
    /// Source path: a .gdb/.gpkg/.shp/.geojson file, or an existing
    /// .parquet file (skipped straight to the optimize step).
    #[arg(long)]
    input: PathBuf,

    /// Layer/table name, for multi-layer sources (.gdb, .gpkg). Required
    /// when the source has more than one; omit otherwise. See --list-layers.
    #[arg(long)]
    layer: Option<String>,

    /// Output .parquet path. Ignored with --list-layers.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Output flavor.
    #[arg(long, value_enum, default_value = "wkb")]
    format: FormatArg,

    /// Row cap per row group.
    #[arg(long, default_value_t = 65_536)]
    row_group_size: usize,

    /// Byte cap per row group (MiB), whichever limit hits first.
    #[arg(long, default_value_t = 16)]
    row_group_mib: usize,

    /// Compression codec.
    #[arg(long, value_enum, default_value = "zstd")]
    compression: CompressionArg,

    /// Skip the Hilbert spatial sort (on by default — it's what makes
    /// row-group bboxes prunable instead of all covering the whole extent).
    #[arg(long)]
    no_hilbert: bool,

    /// Skip the bbox covering column (on by default; needed for spatial
    /// pruning under GeoParquet 1.1 readers).
    #[arg(long)]
    no_covering: bool,

    /// Add an H3 cell column at this resolution (0-15).
    #[arg(long)]
    h3: Option<u8>,

    /// List the layers/tables in --input and exit, instead of converting.
    #[arg(long)]
    list_layers: bool,
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum FormatArg {
    Wkb,
    Geoarrow,
    Native2,
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum CompressionArg {
    Zstd,
    Snappy,
    None,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    if cli.list_layers {
        return print_layers(&cli.input);
    }
    let output = cli
        .output
        .as_deref()
        .ok_or("--output is required (or pass --list-layers)")?;

    let is_parquet = cli
        .input
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"));

    // Stage 1: raw import to an intermediate GeoParquet, unless the input
    // is already one (then the optimize pass alone is the whole job — a
    // convenient way to re-optimize a file too).
    let (intermediate, is_temp): (PathBuf, bool) = if is_parquet {
        (cli.input.clone(), false)
    } else {
        let fmt = ImportFormat::from_path(&cli.input)
            .ok_or_else(|| format!("unsupported input: {}", cli.input.display()))?;
        let tmp = std::env::temp_dir()
            .join(format!("geopq-cli-import-{}.parquet", std::process::id()));
        import(fmt, &cli.input, cli.layer.as_deref(), &tmp)?;
        (tmp, true)
    };

    let opts = OptimizeOptions {
        version: match cli.format {
            FormatArg::Wkb => GpVersion::V1_1,
            FormatArg::Geoarrow => GpVersion::V1_1GeoArrow,
            FormatArg::Native2 => GpVersion::V2_0,
        },
        row_group_size: cli.row_group_size,
        row_group_bytes: cli.row_group_mib << 20,
        codec: match cli.compression {
            CompressionArg::Zstd => Codec::Zstd,
            CompressionArg::Snappy => Codec::Snappy,
            CompressionArg::None => Codec::Uncompressed,
        },
        hilbert_sort: !cli.no_hilbert,
        covering: !cli.no_covering,
        h3_resolution: cli.h3,
        partition: PartitionBy::None,
        ..Default::default()
    };

    let result = optimize::optimize(
        &Source::Local(intermediate.clone()),
        output,
        &opts,
        None,
        None,
        &|frac, stage| eprint!("\r{stage}: {:>5.1}%   ", frac * 100.0),
    );
    eprintln!();

    if is_temp {
        let _ = std::fs::remove_file(&intermediate);
    }

    let report = result?;
    println!(
        "{} rows | row groups {} -> {} | {} -> {}",
        report.rows,
        report.rg_before,
        report.rg_after,
        fmt_bytes(report.size_before),
        fmt_bytes(report.size_after),
    );
    Ok(())
}

fn import(fmt: ImportFormat, input: &Path, layer: Option<&str>, dst: &Path) -> Result<(), String> {
    match fmt {
        ImportFormat::Gpkg => {
            let tables = gpkg::list_tables(input)?;
            let t = pick(&tables, layer, |t| t.name.clone())?;
            gpkg::convert(input, &t, dst, &|_| {})?;
            Ok(())
        }
        ImportFormat::Shapefile => shp::convert(input, dst, &|_| {}).map(|_| ()),
        ImportFormat::GeoJson => geojson::convert(input, dst, &|_| {}).map(|_| ()),
        ImportFormat::Gdb => {
            #[cfg(feature = "gdal-import")]
            {
                use geopq_workbench::data::gdb;
                let layers = gdb::list_layers(input)?;
                let l = pick(&layers, layer, |l| l.name.clone())?;
                gdb::convert(input, &l, dst, &|_| {})?;
                Ok(())
            }
            #[cfg(not(feature = "gdal-import"))]
            {
                let _ = (input, layer, dst);
                Err("this build has no File Geodatabase support \
                     (rebuild with --features gdal-import)"
                    .to_string())
            }
        }
    }
}

fn print_layers(input: &Path) -> Result<(), String> {
    let fmt = ImportFormat::from_path(input)
        .ok_or_else(|| format!("unsupported input: {}", input.display()))?;
    match fmt {
        ImportFormat::Gpkg => {
            for t in gpkg::list_tables(input)? {
                println!("{}\t{} rows\t{}", t.name, t.rows, t.srs_name);
            }
        }
        ImportFormat::Gdb => {
            #[cfg(feature = "gdal-import")]
            {
                for l in geopq_workbench::data::gdb::list_layers(input)? {
                    println!("{}\t{} rows\t{}", l.name, l.rows, l.srs_name);
                }
            }
            #[cfg(not(feature = "gdal-import"))]
            return Err("this build has no File Geodatabase support \
                         (rebuild with --features gdal-import)"
                .to_string());
        }
        ImportFormat::Shapefile | ImportFormat::GeoJson => {
            println!("(single layer — no --layer needed)");
        }
    }
    Ok(())
}

/// Resolve `want` (a `--layer` name) against a multi-layer source's
/// entries, or the sole entry when there's exactly one and none was asked
/// for. Errors list what's actually available.
fn pick<T: Clone>(
    items: &[T],
    want: Option<&str>,
    name: impl Fn(&T) -> String,
) -> Result<T, String> {
    let names = || items.iter().map(&name).collect::<Vec<_>>().join(", ");
    match want {
        Some(n) => items
            .iter()
            .find(|t| name(t).eq_ignore_ascii_case(n))
            .cloned()
            .ok_or_else(|| format!("layer '{n}' not found; available: {}", names())),
        None if items.len() == 1 => Ok(items[0].clone()),
        None if items.is_empty() => Err("source has no layers".to_string()),
        None => Err(format!("multiple layers, pick one with --layer: {}", names())),
    }
}
