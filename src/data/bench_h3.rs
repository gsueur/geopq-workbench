//! "H3 partitions, measured": the benchmark behind the talk of that name.
//!
//! Writes the same layer several ways (one Hilbert-sorted file, adaptive
//! H3, fixed-resolution H3, hive by town) and then reads each one back
//! through the app's own remote path — a local HTTP server with range
//! support, opened as `{prefix}/collection.json` — recording what each
//! viewport actually costs: parts opened, row groups selected, rows
//! decoded, bytes pulled, wall time.
//!
//! Everything here is `#[ignore]`d: it wants a 500 MB input and writes
//! several gigabytes. `bench/h3_partitions.sh` runs the whole thing;
//! the write-up lives in _WIKI/reference/h3-partition-benchmark.md.

#![allow(clippy::type_complexity)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::crs::{Crs, DisplayCrs};
use super::loader::{
    build_planned_for_test, open_source_with_view_for_test, plan_viewport_for_test,
};
use super::net::{self, Channel};
use super::optimize::{optimize, OptimizeOptions};
use super::partition::PartitionBy;
use super::source::Source;
use super::store::FeatureStore;

/// EPSG of the input (MassGIS state plane metres).
const DATA_EPSG: u32 = 26986;

/// Viewport pixel width the planner is told about.
const VIEWPORT_PX: u32 = 1600;

/// Runs kept per measurement, after one warm-up. The median of these is
/// what the report prints; five is enough that one scheduler hiccup on a
/// laptop with a background daemon on it does not become the number.
fn runs() -> usize {
    std::env::var("GEOPQ_BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

fn input() -> Option<PathBuf> {
    let p = PathBuf::from(
        std::env::var("GEOPQ_BENCH_INPUT").unwrap_or_else(|_| {
            "/Users/guillaume/3_APPS/GEOPQ_WORKBENCH/DATA/\
             L3_TAXPAR_POLY_ASSESS_WEST_optimized.parquet"
                .to_string()
        }),
    );
    p.exists().then_some(p)
}

fn bench_root() -> PathBuf {
    PathBuf::from(
        std::env::var("GEOPQ_BENCH_DIR")
            .unwrap_or_else(|_| "/Users/guillaume/3_APPS/GEOPQ_WORKBENCH/DATA/bench".to_string()),
    )
}

/// The five viewports, in EPSG:26986 metres. Picked from the data itself
/// (densest 1 km cell of centroids, then widening around it), so each one
/// is a place a user would actually be looking at.
fn viewports() -> Vec<(&'static str, [f64; 4])> {
    vec![
        // Boston, the densest square kilometre in the file (6.4k centroids).
        ("a_city_1km", [235_000.0, 899_000.0, 236_000.0, 900_000.0]),
        // Newton: a dense inner suburb, 5 km.
        ("b_suburb_5km", [220_000.0, 895_000.0, 225_000.0, 900_000.0]),
        // Berkshire hilltowns, 20 km of genuinely rural coverage.
        ("c_rural_20km", [60_000.0, 880_000.0, 80_000.0, 900_000.0]),
        // Worcester County scale, 60 km.
        ("d_county_60km", [110_000.0, 860_000.0, 170_000.0, 920_000.0]),
        // The whole state.
        ("e_state", [33_000.0, 777_000.0, 331_000.0, 960_000.0]),
    ]
}

/// The variants, in report order.
fn variants() -> Vec<(&'static str, OptimizeOptions)> {
    let base = || OptimizeOptions {
        // Defaults already: zstd, Hilbert sort, covering column, GeoParquet
        // 1.1, 65536 rows / 16 MiB per row group. Spelled out because a
        // benchmark that silently inherits a changed default is worthless.
        row_group_size: 65_536,
        row_group_bytes: 16 << 20,
        codec: super::optimize::Codec::Zstd,
        hilbert_sort: true,
        covering: true,
        ..Default::default()
    };
    vec![
        ("single", base()),
        (
            "h3_adaptive",
            OptimizeOptions {
                partition: PartitionBy::AdaptiveH3 { target_rows: 250_000, max_res: 10 },
                ..base()
            },
        ),
        // Fixed resolution is adaptive with a target no bucket can meet:
        // every cell splits until it hits `max_res`, so the output is
        // exactly the res-N tiling of the data. The optimizer has no
        // separate fixed mode; this is the same code path with the
        // splitting rule pinned.
        (
            "h3_fixed_r5",
            OptimizeOptions {
                partition: PartitionBy::AdaptiveH3 { target_rows: 0, max_res: 5 },
                ..base()
            },
        ),
        (
            "h3_fixed_r6",
            OptimizeOptions {
                partition: PartitionBy::AdaptiveH3 { target_rows: 0, max_res: 6 },
                ..base()
            },
        ),
        (
            "admin",
            OptimizeOptions {
                partition: PartitionBy::Fields(vec!["TOWN_ID".into()]),
                ..base()
            },
        ),
    ]
}

// ---------------------------------------------------------------------
// A range-request HTTP server that streams from disk.
// ---------------------------------------------------------------------

/// Counters for what the server actually put on the wire, split by the
/// kind of object: the STAC manifest is fetched by a code path that does
/// not go through `net::record`, so the client-side counters cannot see
/// it and the server has to.
#[derive(Default)]
pub struct Served {
    pub json_bytes: AtomicU64,
    pub json_reqs: AtomicU64,
    pub parquet_bytes: AtomicU64,
    pub parquet_reqs: AtomicU64,
}

impl Served {
    fn snap(&self) -> (u64, u64, u64, u64) {
        (
            self.json_bytes.load(Ordering::SeqCst),
            self.json_reqs.load(Ordering::SeqCst),
            self.parquet_bytes.load(Ordering::SeqCst),
            self.parquet_reqs.load(Ordering::SeqCst),
        )
    }
}

/// Serve `root` over HTTP/1.1 with `Range` support, streaming from the
/// file rather than reading it whole: the parts here are hundreds of
/// megabytes, and a server that slurps each one per request would be
/// measuring its own `read()` instead of the reader's request pattern.
fn spawn_server(root: PathBuf) -> (String, Arc<Served>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = Arc::new(Served::default());
    let s2 = served.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { continue };
            let (root, served) = (root.clone(), s2.clone());
            std::thread::spawn(move || {
                let _ = serve_one(&mut conn, &root, &served);
            });
        }
    });
    (format!("http://127.0.0.1:{port}"), served)
}

fn serve_one(
    conn: &mut std::net::TcpStream,
    root: &Path,
    served: &Served,
) -> std::io::Result<()> {
    let mut head = Vec::new();
    let mut b = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") && head.len() < 8192 {
        match conn.read(&mut b) {
            Ok(1) => head.push(b[0]),
            _ => return Ok(()),
        }
    }
    let text = String::from_utf8_lossy(&head);
    let is_head = text.starts_with("HEAD");
    let rel = text
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("")
        .trim_start_matches('/')
        .to_string();
    // Percent-decoding: hive values are percent-encoded on the way out
    // (`partition::encode_hive_component`), so the request path for
    // `h3=8a2a...` arrives exactly as written, but a value with an
    // escaped byte would not.
    let rel = percent_decode(&rel);
    let path = root.join(&rel);
    let Ok(meta) = std::fs::metadata(&path) else {
        return write!(
            conn,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    };
    if !meta.is_file() {
        return write!(
            conn,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    }
    let len = meta.len();
    if is_head {
        return write!(
            conn,
            "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nAccept-Ranges: bytes\r\n\
             Connection: close\r\n\r\n"
        );
    }
    let range = text
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().strip_prefix("bytes="))
        .and_then(|v| v.split_once('-'))
        .map(|(a, b)| {
            let start: u64 = a.parse().unwrap_or(0);
            let end: u64 = b.parse().unwrap_or(len.saturating_sub(1));
            (start.min(len.saturating_sub(1)), end.min(len.saturating_sub(1)))
        });
    let (start, end) = range.unwrap_or((0, len.saturating_sub(1)));
    let n = end + 1 - start;
    let (status, extra) = match range {
        Some(_) => (
            "206 Partial Content",
            format!("Content-Range: bytes {start}-{end}/{len}\r\n"),
        ),
        None => ("200 OK", String::new()),
    };
    write!(
        conn,
        "HTTP/1.1 {status}\r\nContent-Length: {n}\r\n{extra}Accept-Ranges: bytes\r\n\
         Connection: close\r\n\r\n"
    )?;
    let json = rel.ends_with(".json");
    if json {
        served.json_reqs.fetch_add(1, Ordering::SeqCst);
    } else {
        served.parquet_reqs.fetch_add(1, Ordering::SeqCst);
    }
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::File::open(&path)?;
    f.seek(SeekFrom::Start(start))?;
    let mut pos = start;
    let mut buf = vec![0u8; 256 * 1024];
    while pos <= end {
        let take = ((end - pos + 1) as usize).min(buf.len());
        let got = f.read(&mut buf[..take])?;
        if got == 0 || conn.write_all(&buf[..got]).is_err() {
            break;
        }
        if json {
            served.json_bytes.fetch_add(got as u64, Ordering::SeqCst);
        } else {
            served.parquet_bytes.fetch_add(got as u64, Ordering::SeqCst);
        }
        pos += got as u64;
    }
    Ok(())
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------
// Writing the variants.
// ---------------------------------------------------------------------

/// `cargo test --release -p geopq-workbench bench_write_variants -- --ignored --nocapture`
#[test]
#[ignore = "writes several GB from a 500 MB input"]
fn bench_write_variants() {
    let Some(src) = input() else {
        eprintln!("input missing; set GEOPQ_BENCH_INPUT");
        return;
    };
    let root = bench_root();
    std::fs::create_dir_all(&root).unwrap();
    let only = std::env::var("GEOPQ_BENCH_ONLY").ok();
    let crs = Crs::from_epsg(DATA_EPSG).unwrap();

    for (name, opts) in variants() {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let partitioned = !matches!(opts.partition, PartitionBy::None);
        let dir = root.join(name);
        let dst = if partitioned { dir.clone() } else { dir.join("data.parquet") };
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        let t0 = Instant::now();
        let rep = optimize(
            &Source::Local(src.clone()),
            &dst,
            &opts,
            Some(DATA_EPSG),
            None,
            &|_, _| {},
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let write_s = t0.elapsed().as_secs_f64();
        // The reader opens an https prefix through the collection at it,
        // so every variant needs one — the optimizer's own writer, which
        // reads each part's footer for its bbox and row count.
        let stac_t = Instant::now();
        let col = super::stac::write_for_output(&dst, name, &crs)
            .unwrap_or_else(|e| panic!("{name} stac: {e}"));
        let bytes = dir_bytes(&dir);
        eprintln!(
            "VARIANT\t{name}\tfiles={}\trows={}\tbytes={}\trg={}\twrite_s={:.1}\tstac_s={:.1}\tcollection={}",
            rep.files,
            rep.rows,
            bytes,
            rep.rg_after,
            write_s,
            stac_t.elapsed().as_secs_f64(),
            col.display(),
        );
    }
}

fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            total += dir_bytes(&p);
        } else if let Ok(m) = std::fs::metadata(&p) {
            total += m.len();
        }
    }
    total
}

// ---------------------------------------------------------------------
// Measuring.
// ---------------------------------------------------------------------

struct Run {
    parts: usize,
    groups: usize,
    rows: usize,
    footer_bytes: u64,
    manifest_bytes: u64,
    data_bytes: u64,
    open_ms: f64,
    load_ms: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Which fragments the planned groups live in. `rg_offset` is unique per
/// fragment, so it doubles as the part identity.
fn parts_touched(store: &FeatureStore, groups: &[u32]) -> usize {
    let mut seen: Vec<usize> = groups
        .iter()
        .map(|&g| store.frag_of_group(g as usize).rg_offset)
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

fn one_run(root: &Path, variant: &str, rect: [f64; 4], crs: &Crs, display: &DisplayCrs) -> Run {
    // A fresh port per run: the STAC part list is cached on disk under
    // its collection URL, and a repeat on the same URL would report a
    // manifest fetch that never happened. Every run here is a cold open.
    let (base, served) = spawn_server(root.to_path_buf());
    let url = format!("{base}/{variant}/collection.json");
    let source = Source::Stac { url: url.clone(), name: variant.to_string() };

    // STAC item bboxes are WGS84 by spec; the part-level prune needs the
    // viewport in those coordinates.
    let geo_rect = super::stac::wgs84_bbox(rect, crs).expect("viewport reprojects");

    let net0 = net::totals(Channel::Data);
    let srv0 = served.snap();
    let t0 = Instant::now();
    let (store, store_crs, _info, rg) =
        open_source_with_view_for_test(&source, Some(geo_rect)).expect("open");
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let net1 = net::totals(Channel::Data);
    let srv1 = served.snap();

    let boxes = rg.map(|(_, b)| b).unwrap_or_default();
    if std::env::var("GEOPQ_BENCH_DEBUG").is_ok() {
        eprintln!(
            "DEBUG {variant}: frags={} rg={} boxes={} covering={} wkb={} polygons_only={} rows={}",
            store.fragments.len(),
            store.rg_starts().len() - 1,
            boxes.len(),
            store.covering.is_some(),
            store.encoding.is_wkb(),
            store.polygons_only,
            store.total_rows(),
        );
    }
    let sel = plan_viewport_for_test(
        &store,
        (!boxes.is_empty()).then_some(boxes.as_slice()),
        Some(rect),
    );
    let groups: Vec<u32> = sel.iter().map(|s| s.group()).collect();
    if std::env::var("GEOPQ_BENCH_DEBUG").is_ok() {
        eprintln!("DEBUG {variant}: sel={:?}", &sel[..sel.len().min(6)]);
    }

    let t1 = Instant::now();
    let rows = build_planned_for_test(&store, &store_crs, display, sel).expect("build");
    let load_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let net2 = net::totals(Channel::Data);
    let srv2 = served.snap();

    Run {
        parts: parts_touched(&store, &groups),
        groups: groups.len(),
        rows,
        footer_bytes: net1.0 - net0.0,
        // The manifest is fetched by `repo::get_json`, which does not go
        // through the byte counters; the server sees it.
        manifest_bytes: (srv1.0 - srv0.0) + (srv2.0 - srv1.0),
        data_bytes: net2.0 - net1.0,
        open_ms,
        load_ms,
    }
}

/// `cargo test --release -p geopq-workbench bench_measure -- --ignored --nocapture`
#[test]
#[ignore = "reads the variants written by bench_write_variants"]
fn bench_measure() {
    let root = bench_root();
    if !root.exists() {
        eprintln!("{} missing; run bench_write_variants first", root.display());
        return;
    }
    let crs = Crs::from_epsg(DATA_EPSG).unwrap();
    let display = DisplayCrs::hobo_dyer();
    let mut names: Vec<String> = variants().iter().map(|(n, _)| n.to_string()).collect();
    // Written by `bench_finish_kdtree`, not by `variants()`.
    names.push("kdtree".to_string());
    names.push("kdtree_auto512".to_string());

    eprintln!(
        "MEASURE\tviewport\tvariant\tparts\tgroups\trows\tmanifest_B\tfooter_B\tdata_B\topen_ms\tload_ms"
    );
    let only_vp = std::env::var("GEOPQ_BENCH_VIEWPORT").ok();
    let only = std::env::var("GEOPQ_BENCH_ONLY").ok();
    for (vp_name, rect) in viewports() {
        if only_vp.as_deref().is_some_and(|v| v != vp_name) {
            continue;
        }
        for name in &names {
            if only.as_deref().is_some_and(|o| o != name.as_str()) {
                continue;
            }
            if !root.join(name).join("collection.json").exists() {
                eprintln!("skip {name}: no collection.json");
                continue;
            }
            // Warm-up, then RUNS measured. The page cache and the process
            // are warm from here on; what varies is request count and CPU.
            let _ = one_run(&root, name, rect, &crs, &display);
            let n = runs();
            let mut runs: Vec<Run> = Vec::with_capacity(n);
            for _ in 0..n {
                runs.push(one_run(&root, name, rect, &crs, &display));
            }
            let r = &runs[0];
            eprintln!(
                "MEASURE\t{vp_name}\t{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.0}\t{:.0}",
                r.parts,
                r.groups,
                r.rows,
                median(runs.iter().map(|x| x.manifest_bytes as f64).collect()),
                median(runs.iter().map(|x| x.footer_bytes as f64).collect()),
                median(runs.iter().map(|x| x.data_bytes as f64).collect()),
                median(runs.iter().map(|x| x.open_ms).collect()),
                median(runs.iter().map(|x| x.load_ms).collect()),
            );
        }
    }
    let _ = VIEWPORT_PX;
}

// ---------------------------------------------------------------------
// SQL: hive equality pushdown vs a bbox predicate.
// ---------------------------------------------------------------------

/// The H3 cell of the viewport-(a) centre that the given variant actually
/// stores, found by walking the res-10 cell's parents until one names a
/// directory that exists.
fn cell_for(root: &Path, variant: &str, lon: f64, lat: f64) -> Option<String> {
    use h3o::{LatLng, Resolution};
    let fine = LatLng::new(lat, lon).ok()?.to_cell(Resolution::Ten);
    let dir = root.join(variant);
    for r in (0..=10u8).rev() {
        let res = Resolution::try_from(r).ok()?;
        let c = fine.parent(res)?;
        let name = format!("h3={c}");
        if dir.join(&name).is_dir() {
            return Some(c.to_string());
        }
    }
    None
}

/// `cargo test --release -p geopq-workbench bench_sql_pushdown -- --ignored --nocapture`
#[test]
#[ignore = "reads the variants written by bench_write_variants"]
fn bench_sql_pushdown() {
    use datafusion::prelude::SessionContext;

    let root = bench_root();
    if !root.exists() {
        eprintln!("{} missing; run bench_write_variants first", root.display());
        return;
    }
    let crs = Crs::from_epsg(DATA_EPSG).unwrap();
    let rect = viewports()[0].1;
    let geo_rect = super::stac::wgs84_bbox(rect, &crs).expect("viewport reprojects");
    let (lon, lat) = ((geo_rect[0] + geo_rect[2]) / 2.0, (geo_rect[1] + geo_rect[3]) / 2.0);

    eprintln!("SQL\tvariant\tquery\tcell\tparts_in_store\tparquet_reqs\tdata_B\tms\tcount");
    for variant in ["h3_adaptive", "h3_fixed_r5", "h3_fixed_r6"] {
        let Some(cell) = cell_for(&root, variant, lon, lat) else {
            eprintln!("skip {variant}: no cell for the viewport centre");
            continue;
        };
        let queries = vec![
            (
                "hive_eq",
                format!("SELECT count(*) FROM layer WHERE h3 = '{cell}'"),
            ),
            (
                "st_intersects",
                format!(
                    "SELECT count(*) FROM layer WHERE st_intersects(geometry, \
                     st_makeenvelope({}, {}, {}, {}))",
                    rect[0], rect[1], rect[2], rect[3]
                ),
            ),
        ];
        for (label, sql) in queries {
            let (base, served) = spawn_server(root.to_path_buf());
            let url = format!("{base}/{variant}/collection.json");
            let source = Source::Stac { url, name: variant.to_string() };
            // Opened at viewport (a), like the layer the SQL console
            // runs against: a collection opened with no view at all is
            // truncated at STAC_PART_CAP, and counting rows out of a
            // truncated store would be measuring the cap. What is under
            // test here is what the *predicate* prunes once the store is
            // open, which is the same question either way.
            let (store, _c, _i, rg) =
                open_source_with_view_for_test(&source, Some(geo_rect)).expect("open");
            let parts_in_store = store.fragments.len();
            let store = Arc::new(store);
            let table = super::super::sql::table::LayerTable::new(
                store.clone(),
                rg.map(|(_, b)| Arc::new(b)),
            );
            let ctx = SessionContext::new();
            crate::sql::udf::register_all(&ctx);
            ctx.register_table("layer", Arc::new(table)).unwrap();
            let srv0 = served.snap();
            let net0 = net::totals(Channel::Data);
            let t = Instant::now();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let batches = rt.block_on(async { ctx.sql(&sql).await?.collect().await });
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let net1 = net::totals(Channel::Data);
            let srv1 = served.snap();
            let count = batches
                .as_ref()
                .ok()
                .and_then(|b| b.first().cloned())
                .map(|b| {
                    use arrow::array::AsArray;
                    b.column(0)
                        .as_primitive::<arrow::datatypes::Int64Type>()
                        .value(0)
                })
                .unwrap_or(-1);
            eprintln!(
                "SQL\t{variant}\t{label}\t{cell}\t{parts_in_store}\t{}\t{}\t{ms:.0}\t{count}",
                srv1.3 - srv0.3,
                net1.0 - net0.0,
            );
            if let Err(e) = batches {
                eprintln!("  query failed: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------
// KD-tree, via geoparquet-io.
// ---------------------------------------------------------------------

/// Re-optimize a `gpio partition kdtree` tree with the same file-internal
/// options as every other variant. The scheme under test is the split;
/// leaving gpio's own codec, row-group sizing and (absent) covering
/// column in place would make the row read a comparison of writers.
///
/// `GEOPQ_BENCH_KDTREE_RAW=... GEOPQ_BENCH_KDTREE_OUT=... cargo test
/// --release bench_finish_kdtree -- --ignored --nocapture`
#[test]
#[ignore = "needs a gpio kdtree tree; see bench/h3_partitions.sh"]
fn bench_finish_kdtree() {
    let (Ok(raw), Ok(out)) = (
        std::env::var("GEOPQ_BENCH_KDTREE_RAW"),
        std::env::var("GEOPQ_BENCH_KDTREE_OUT"),
    ) else {
        eprintln!("set GEOPQ_BENCH_KDTREE_RAW and GEOPQ_BENCH_KDTREE_OUT");
        return;
    };
    let (raw, out) = (PathBuf::from(raw), PathBuf::from(out));
    let opts = variants()
        .into_iter()
        .find(|(n, _)| *n == "single")
        .map(|(_, o)| o)
        .unwrap();
    let mut parts = Vec::new();
    collect_parquet(&raw, &mut parts);
    parts.sort();
    assert!(!parts.is_empty(), "no parquet under {}", raw.display());
    let t0 = Instant::now();
    let mut rows = 0u64;
    for p in &parts {
        // gpio writes a flat tree of numbered files, not hive
        // directories: a KD-tree cell id is not a queryable key the way
        // an H3 cell or a town id is, so the layout is preserved as it
        // came and the parts keep their own names.
        let rel = p.strip_prefix(&raw).unwrap();
        let dst = out.join(rel);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        let rep = optimize(
            &Source::Local(p.clone()),
            &dst,
            &opts,
            Some(DATA_EPSG),
            None,
            &|_, _| {},
        )
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        rows += rep.rows;
    }
    let crs = Crs::from_epsg(DATA_EPSG).unwrap();
    let col = super::stac::write_for_output(&out, "kdtree", &crs).unwrap();
    eprintln!(
        "VARIANT\tkdtree\tfiles={}\trows={rows}\tbytes={}\twrite_s={:.1}\tcollection={}",
        parts.len(),
        dir_bytes(&out),
        t0.elapsed().as_secs_f64(),
        col.display(),
    );
}

fn collect_parquet(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_parquet(&p, out);
        } else if p.extension().is_some_and(|x| x == "parquet") {
            out.push(p);
        }
    }
}

#[test]
#[ignore = "diagnostic"]
fn bench_debug_dir() {
    let root = bench_root();
    for v in ["single", "h3_adaptive", "h3_fixed_r5", "admin"] {
        let d = root.join(v);
        match super::loader::open_source_for_test(&Source::Dir(d.clone())) {
            Ok((store, _c, _i, rg)) => eprintln!(
                "DIR {v}: frags={} rg={} boxes={:?} covering={}",
                store.fragments.len(),
                store.rg_starts().len() - 1,
                rg.as_ref().map(|(s, b)| (s.clone(), b.len())),
                store.covering.is_some(),
            ),
            Err(e) => eprintln!("DIR {v}: {e}"),
        }
    }
}
