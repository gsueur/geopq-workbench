//! geopq-cli: headless batch converter for scripted pipelines (cron jobs,
//! monthly refreshes) that don't want a GUI. Wraps the same import +
//! optimize machinery the app's Import/Export dialogs use: no window, no
//! GPU, just a source path in and an optimized GeoParquet out.

use std::path::{Path, PathBuf};

use clap::Parser;
use geopq_workbench::data::import::ImportFormat;
use geopq_workbench::data::info::fmt_bytes;
use geopq_workbench::data::optimize::{self, Codec, GpVersion, OptimizeOptions, OptimizeReport};
use geopq_workbench::data::partition::PartitionBy;
use geopq_workbench::data::source::Source;
use geopq_workbench::data::{geojson, gpkg, settings, shp};

/// Convert a vector source (GeoPackage, Shapefile, GeoJSON, or an
/// existing GeoParquet file) into an optimized GeoParquet: spatially
/// sorted, tuned row groups, covering bbox column.
#[derive(Parser)]
#[command(name = "geopq-cli", version)]
struct Cli {
    /// Source path: a .gpkg/.shp/.geojson file, or an existing .parquet
    /// file (skipped straight to the optimize step).
    #[arg(long)]
    input: PathBuf,

    /// Table name, for multi-table sources (.gpkg). Required when the
    /// source has more than one; omit otherwise. See --list-layers.
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
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=15))]
    h3: Option<u8>,

    /// Partition output into hive directories by these output column
    /// names (comma-separated, e.g. `ISO3` or `ISO3,STATUS_YR`). With
    /// this set, --output is a directory: parts land at
    /// <output>/<field>=<value>/part-0.parquet. Mutually exclusive with
    /// --partition-h3.
    #[arg(long, value_delimiter = ',')]
    partition_by: Vec<String>,

    /// Partition output into adaptive H3 cells instead of fields: cells
    /// over --partition-h3-target-rows split into children until
    /// balanced or the max resolution is reached. Mutually exclusive
    /// with --partition-by.
    #[arg(long)]
    partition_h3: bool,

    /// Row target per adaptive-H3 partition (only with --partition-h3).
    #[arg(long, default_value_t = 100_000)]
    partition_h3_target_rows: usize,

    /// Finest resolution the adaptive-H3 split may reach (0-15).
    /// Defaults to --h3 when that is set, otherwise 10.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=15))]
    partition_h3_max_res: Option<u8>,

    /// Order the output coarse to fine and write the COGP v0.1 level
    /// metadata (experimental). Implies the Hilbert sort and, on the
    /// 1.1 WKB flavor, the covering column; cannot be combined with
    /// partitioning. Level GSDs and thinning factors come from the
    /// `cogp` block of ~/.geopq-workbench.json, same as the GUI's.
    #[arg(long)]
    cogp: bool,

    /// List the tables in --input and exit, instead of converting.
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

    // Every flag combination is settled before the import runs: an
    // unrunnable request should cost a millisecond, not the minutes it
    // takes to write the intermediate first.
    let opts = options(&cli)?;

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
        let r = import(fmt, &cli.input, cli.layer.as_deref(), &tmp);
        if r.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        r?;
        (tmp, true)
    };

    let progress = Progress::default();
    let result = optimize::optimize(
        &Source::Local(intermediate.clone()),
        output,
        &opts,
        None,
        None,
        &|frac, stage| progress.report(frac, stage),
        // Nothing cancels a CLI run but the signal that ends it.
        &std::sync::atomic::AtomicBool::new(false),
    );
    progress.finish();

    if is_temp {
        let _ = std::fs::remove_file(&intermediate);
    }

    println!("{}", summary(&result?));
    Ok(())
}

/// The `OptimizeOptions` the flags add up to, or why they don't.
fn options(cli: &Cli) -> Result<OptimizeOptions, String> {
    if !cli.partition_by.is_empty() && cli.partition_h3 {
        return Err("--partition-by and --partition-h3 are mutually exclusive".into());
    }
    // Single-layer formats have nothing to pick from, so --layer there is
    // a typo (usually the wrong --input), not a no-op.
    if cli.layer.is_some()
        && let Some(fmt) = ImportFormat::from_path(&cli.input)
        && !matches!(fmt, ImportFormat::Gpkg)
    {
        return Err(format!(
            "--layer applies to multi-layer sources; {} holds a single layer",
            cli.input.display()
        ));
    }
    let partition = if cli.partition_h3 {
        // The adaptive split needs a floor to stop at. --h3 is the natural
        // one when the run already asks for a cell column at that
        // resolution; 10 (~0.015 km² cells) otherwise.
        let max_res = cli.partition_h3_max_res.or(cli.h3).unwrap_or(10);
        if cli.partition_h3_target_rows == 0 {
            return Err("--partition-h3-target-rows must be positive".into());
        }
        PartitionBy::AdaptiveH3 { target_rows: cli.partition_h3_target_rows, max_res }
    } else if !cli.partition_by.is_empty() {
        PartitionBy::Fields(cli.partition_by.clone())
    } else {
        PartitionBy::None
    };

    let version = match cli.format {
        FormatArg::Wkb => GpVersion::V1_1,
        FormatArg::Geoarrow => GpVersion::V1_1GeoArrow,
        FormatArg::Native2 => GpVersion::V2_0,
    };

    // COGP decides the physical layout, so it takes over what it needs
    // and refuses what it cannot share the file with — the same rules the
    // Export dialog applies when the ordering radio is switched to it.
    let cogp = if cli.cogp {
        if partition != PartitionBy::None {
            return Err("--cogp cannot be combined with partitioning: \
                        levels and hive parts both own the file layout"
                .into());
        }
        if cli.no_hilbert {
            return Err("--cogp implies the Hilbert sort; drop --no-hilbert".into());
        }
        if cli.no_covering && version == GpVersion::V1_1 {
            return Err("--cogp on the wkb flavor implies the covering column; \
                        drop --no-covering"
                .into());
        }
        // Refuse rather than silently substitute the reference GSDs:
        // a pyramid written to levels the user did not configure is
        // wrong in a way nothing downstream can detect.
        Some(
            settings::cogp_settings()
                .map_err(|e| format!("--cogp: {e}"))?
                .clone(),
        )
    } else {
        None
    };

    if cli.row_group_size == 0 {
        return Err("--row-group-size must be positive".into());
    }
    if cli.row_group_mib == 0 {
        return Err("--row-group-mib must be positive".into());
    }

    Ok(OptimizeOptions {
        version,
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
        partition,
        cogp,
        ..Default::default()
    })
}

/// The one line stdout gets on success, so a shell pipeline has something
/// stable to read while the progress noise goes to stderr.
fn summary(r: &OptimizeReport) -> String {
    let mut out = format!(
        "{} rows | row groups {} -> {} | {} -> {} | {} file{}",
        r.rows,
        r.rg_before,
        r.rg_after,
        fmt_bytes(r.size_before),
        fmt_bytes(r.size_after),
        r.files,
        if r.files == 1 { "" } else { "s" },
    );
    if !r.cogp_levels.is_empty() {
        out.push_str(&format!(" | {} COGP levels", r.cogp_levels.len()));
    }
    out
}

/// Progress on stderr, rewritten in place. The optimizer calls back far
/// more often than a terminal can usefully repaint (and far more often
/// than a redirected log wants a line), so a repaint only happens when
/// the stage changes or the percentage moves by a tenth of a point.
#[derive(Default)]
struct Progress {
    last: std::cell::RefCell<(String, i32)>,
}

impl Progress {
    fn report(&self, frac: f32, stage: &str) {
        let tenths = (frac.clamp(0.0, 1.0) * 1000.0) as i32;
        let mut last = self.last.borrow_mut();
        if last.0 == stage && last.1 == tenths {
            return;
        }
        if last.0 != stage {
            last.0 = stage.to_string();
        }
        last.1 = tenths;
        eprint!("\r{stage}: {:>5.1}%   ", tenths as f32 / 10.0);
    }

    /// Close the in-place line, once, and only if something was drawn on it.
    fn finish(&self) {
        if !self.last.borrow().0.is_empty() {
            eprintln!();
        }
    }
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
        ImportFormat::Shapefile | ImportFormat::GeoJson => {
            // Prose, not data: stdout carries the tab-separated layer
            // list, and a shell pipeline must not have to filter this
            // sentence back out of it.
            eprintln!("(single layer — no --layer needed)");
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
