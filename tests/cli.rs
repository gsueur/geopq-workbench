//! End-to-end tests for the `geopq-cli` binary: run the real executable
//! the way a pipeline would, then open what it wrote through the library
//! the GUI reads with. A conversion that exits 0 but produces a file this
//! app cannot open is the failure mode worth guarding, so every case
//! asserts on the output, not just on the exit status.
//!
//! The fixtures are generated, not committed (`testdata/regenerate.sh`);
//! cases whose fixture is missing skip, matching the in-crate tests.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use geopq_workbench::data::info::FileInfo;
use geopq_workbench::data::loader::open_store;
use geopq_workbench::data::source::Source;

const CLI: &str = env!("CARGO_BIN_EXE_geopq-cli");

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

/// A scratch directory of this test's own, removed on drop so a failing
/// assertion still leaves the next run a clean slate.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("geopq-cli-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        Self(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(CLI).args(args).output().expect("geopq-cli runs")
}

/// stdout of a run that must have succeeded, with the stderr progress
/// echoed into the failure message when it did not.
fn stdout_ok(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "geopq-cli {args:?} failed ({}):\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// Open a written output the way the app does, and return its metadata.
fn open(source: Source) -> FileInfo {
    let (_store, _crs, info, _rg) = open_store(&source).expect("output opens");
    info
}

/// Every output this CLI writes must declare a covering bbox column:
/// without it a reader has no way to select features spatially, which is
/// the entire point of the optimize pass.
fn assert_covering(info: &FileInfo) {
    assert!(
        info.geo.covering.is_some(),
        "no covering column declared; geo metadata: {:?}",
        info.geo.raw_geo_json
    );
}

/// A tiny two-feature GeoJSON, so the import path is exercised without a
/// generated fixture.
fn write_geojson(dst: &Path) {
    std::fs::write(
        dst,
        r#"{"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{"region":"north"},
           "geometry":{"type":"Point","coordinates":[2.35,48.85]}},
          {"type":"Feature","properties":{"region":"south"},
           "geometry":{"type":"Point","coordinates":[5.37,43.30]}}
        ]}"#,
    )
    .expect("write geojson");
}

#[test]
fn optimizes_a_parquet_in_place_of_an_import() {
    let src = testdata("polygons_5k_l93.parquet");
    if !src.exists() {
        eprintln!("fixture missing, skipping");
        return;
    }
    let scratch = Scratch::new("plain");
    let dst = scratch.path("plain.parquet");

    let summary = stdout_ok(&[
        "--input",
        src.to_str().unwrap(),
        "--output",
        dst.to_str().unwrap(),
    ]);

    // One parseable line, whether or not the run partitioned.
    assert_eq!(summary.lines().count(), 1, "summary: {summary:?}");
    assert!(summary.starts_with("5000 rows |"), "summary: {summary:?}");
    assert!(summary.contains("| 1 file"), "summary: {summary:?}");

    let info = open(Source::Local(dst));
    assert_eq!(info.rows, 5000);
    assert_eq!(info.files, 1);
    assert_covering(&info);
}

#[test]
fn partitions_adaptively_by_h3() {
    let src = testdata("polygons_5k_l93.parquet");
    if !src.exists() {
        eprintln!("fixture missing, skipping");
        return;
    }
    let scratch = Scratch::new("h3");
    let dst = scratch.path("by_h3");

    let summary = stdout_ok(&[
        "--input",
        src.to_str().unwrap(),
        "--output",
        dst.to_str().unwrap(),
        "--partition-h3",
        "--h3",
        "6",
        "--partition-h3-target-rows",
        "2000",
    ]);
    assert!(summary.starts_with("5000 rows |"), "summary: {summary:?}");

    // Adaptive H3 names its directories `h3=<cell>`, and 5000 features
    // across a 500 km box cannot fit one 2000-row partition.
    let dirs: Vec<String> = std::fs::read_dir(&dst)
        .expect("output directory")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(dirs.len() > 1, "expected several H3 partitions, got {dirs:?}");
    assert!(dirs.iter().all(|d| d.starts_with("h3=")), "{dirs:?}");

    let info = open(Source::Dir(dst));
    assert_eq!(info.rows, 5000, "every feature lands in exactly one part");
    assert_eq!(info.files, dirs.len());
    assert_covering(&info);
}

#[test]
fn partitions_into_hive_directories_by_field() {
    let src = testdata("polygons_5k_l93.parquet");
    if !src.exists() {
        eprintln!("fixture missing, skipping");
        return;
    }
    let scratch = Scratch::new("hive");
    let dst = scratch.path("by_score");

    // `score` is the fixture's low-cardinality column (0-100).
    let summary = stdout_ok(&[
        "--input",
        src.to_str().unwrap(),
        "--output",
        dst.to_str().unwrap(),
        "--partition-by",
        "score",
    ]);
    assert!(summary.starts_with("5000 rows |"), "summary: {summary:?}");

    let dirs: Vec<String> = std::fs::read_dir(&dst)
        .expect("output directory")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(dirs.len() > 1, "expected one directory per score, got {dirs:?}");
    assert!(dirs.iter().all(|d| d.starts_with("score=")), "{dirs:?}");

    let info = open(Source::Dir(dst));
    assert_eq!(info.rows, 5000);
    assert_eq!(info.files, dirs.len());
    assert_covering(&info);
    // The hive key comes back as a virtual column, not a stored one.
    assert!(
        info.columns.iter().any(|c| c.name == "score"),
        "hive key missing from the reopened dataset"
    );
}

#[test]
fn imports_a_geojson_and_lists_its_single_layer() {
    let scratch = Scratch::new("geojson");
    let src = scratch.path("points.geojson");
    write_geojson(&src);

    // The note goes to stderr: stdout is the machine-readable layer list
    // and a shell pipeline should not have to filter prose out of it.
    let out = run(&["--input", src.to_str().unwrap(), "--list-layers"]);
    assert!(out.status.success());
    let listed = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(listed.contains("single layer"), "listing: {listed:?}");
    assert!(String::from_utf8_lossy(&out.stdout).trim().is_empty());

    let dst = scratch.path("points.parquet");
    let summary = stdout_ok(&[
        "--input",
        src.to_str().unwrap(),
        "--output",
        dst.to_str().unwrap(),
        "--format",
        "geoarrow",
    ]);
    assert!(summary.starts_with("2 rows |"), "summary: {summary:?}");

    let info = open(Source::Local(dst));
    assert_eq!(info.rows, 2);
    assert_covering(&info);
}

#[test]
fn lists_and_converts_a_geopackage_table() {
    let src = testdata("places.gpkg");
    if !src.exists() {
        eprintln!("fixture missing, skipping");
        return;
    }
    let listed = stdout_ok(&["--input", src.to_str().unwrap(), "--list-layers"]);
    // One tab-separated `name<TAB>N rows<TAB>srs` line per feature table.
    assert!(!listed.trim().is_empty(), "a GeoPackage lists its tables");
    for line in listed.lines() {
        assert_eq!(line.split('\t').count(), 3, "line: {line:?}");
    }
    assert!(listed.contains("places"), "listing: {listed:?}");

    let scratch = Scratch::new("gpkg");

    // Naming the table and letting the sole table be picked must agree.
    for layer in [None, Some("places")] {
        let dst = scratch.path(match layer {
            None => "auto.parquet",
            Some(_) => "named.parquet",
        });
        let mut args = vec![
            "--input",
            src.to_str().unwrap(),
            "--output",
            dst.to_str().unwrap(),
        ];
        if let Some(l) = layer {
            args.extend_from_slice(&["--layer", l]);
        }
        let summary = stdout_ok(&args);
        assert!(summary.starts_with("200 rows |"), "summary: {summary:?}");
        assert_covering(&open(Source::Local(dst)));
    }

    // A name that is not there says what is, rather than failing blankly.
    let out = run(&[
        "--input",
        src.to_str().unwrap(),
        "--output",
        scratch.path("never.parquet").to_str().unwrap(),
        "--layer",
        "no_such_table",
    ]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("available: places"), "stderr: {err:?}");
}

#[test]
fn refuses_impossible_flag_combinations_before_doing_any_work() {
    let src = testdata("polygons_5k_l93.parquet");
    if !src.exists() {
        eprintln!("fixture missing, skipping");
        return;
    }
    let scratch = Scratch::new("errors");

    for (args, expect) in [
        (
            vec!["--partition-by", "score", "--partition-h3"],
            "mutually exclusive",
        ),
        (vec!["--cogp", "--partition-by", "score"], "--cogp cannot"),
        (vec!["--cogp", "--no-hilbert"], "implies the Hilbert sort"),
        // H3 tops out at 15. This used to be checked deep inside the
        // write pass, after the import had already run.
        (vec!["--h3", "20"], "20"),
        (vec!["--partition-h3", "--partition-h3-max-res", "16"], "16"),
    ] {
        let dst = scratch.path("never-written");
        let mut full = vec![
            "--input",
            src.to_str().unwrap(),
            "--output",
            dst.to_str().unwrap(),
        ];
        full.extend_from_slice(&args);
        let out = run(&full);
        assert!(!out.status.success(), "{args:?} should have failed");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(expect), "{args:?}: stderr was {err:?}");
        assert!(!dst.exists(), "{args:?} wrote an output anyway");
    }

    // --layer names a table in a multi-layer source; on a single-layer
    // one it was accepted and then ignored, so a mistyped --input read
    // as a successful conversion of the wrong file.
    let gj = scratch.path("one.geojson");
    std::fs::write(
        &gj,
        r#"{"type":"FeatureCollection","features":[
             {"type":"Feature","properties":{},
              "geometry":{"type":"Point","coordinates":[1,2]}}]}"#,
    )
    .unwrap();
    let dst = scratch.path("never-written-layer");
    let out = run(&[
        "--input",
        gj.to_str().unwrap(),
        "--output",
        dst.to_str().unwrap(),
        "--layer",
        "places",
    ]);
    assert!(!out.status.success(), "--layer on a geojson should fail");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("single layer"), "stderr was {err:?}");
    assert!(!dst.exists());

    // --list-layers keeps stdout for the layer list; the "nothing to
    // pick" note is prose and belongs on stderr.
    let out = run(&["--input", gj.to_str().unwrap(), "--list-layers"]);
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        "stdout was {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("single layer"));

    // A missing input is an error too, not a silently empty output.
    let out = run(&[
        "--input",
        scratch.path("absent.geojson").to_str().unwrap(),
        "--output",
        scratch.path("out.parquet").to_str().unwrap(),
    ]);
    assert!(!out.status.success());
}
