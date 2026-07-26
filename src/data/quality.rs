//! Footer-only file quality analysis: is this file shaped so the
//! viewport machinery (row-group pruning, per-feature selection,
//! refinement) can work, and how close is it to display best practice?
//!
//! See docs/OPEN_POLICY.md. The report states facts about the file;
//! the open policy (Indexed vs Direct mode, the optimize offer) is
//! decided by the app from the report plus the row count.

use super::geoarrow::GeomEncoding;
use super::info::GeoParquetInfo;

/// C2 threshold on [`overlap_frac`]: above this, row-group bboxes
/// overlap so much that viewport pruning degenerates and whole-group
/// refinement selects most of the file. Matches
/// `RgBboxes::poorly_clustered` (measured reference points: Hilbert
/// ~13–25%, attribute-ordered ~35%, spatially random ~100%).
pub const OVERLAP_FRAC_MAX: f64 = 0.30;
/// C2 absolute floor: a spatially sorted file forms a chain of bboxes
/// where each touches ~2 neighbors, independent of group count. Small
/// files would fail the fraction test on adjacency alone (4 sorted
/// groups measure 50%), so an average overlap up to this many boxes
/// always passes.
pub const CHAIN_OVERLAP_MIN: f64 = 2.0;
/// C3: the row group is the refine/decode unit; groups larger than this
/// make pruning coarse and single-group decodes chunky.
pub const RG_ROWS_MAX: u64 = 512_000;
/// C3: the same limit in bytes. Rows measure a row group only when
/// features are small; a few hundred administrative boundaries can fill
/// hundreds of megabytes, and that group is one indivisible decode
/// whatever its row count says.
pub const RG_BYTES_MAX: u64 = 128 << 20;
/// C3 advisory: below this, footer size and per-group overhead dominate.
pub const RG_ROWS_MIN: u64 = 16_000;
/// C3 advisory: and only when the groups are small in bytes too. Few
/// rows of heavy geometry make a substantial row group, and calling
/// that "overhead dominated" would push the user to undo a split that
/// is doing exactly what it should.
pub const RG_BYTES_MIN: u64 = 4 << 20;
/// Hard ceiling for the quality-gate "Load all" path: above either, an
/// unoptimized file can only be Optimized (an honest refusal beats an
/// out-of-memory tessellation). Provisional values, to be calibrated.
pub const DIRECT_MAX_ROWS: u64 = 12_000_000;
pub const DIRECT_MAX_GEOM_BYTES: u64 = 8 << 30;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Pass,
    /// Suboptimal but workable; never blocks the Indexed verdict.
    Warn,
    Fail,
}

#[derive(Clone, Debug)]
pub struct Check {
    /// Stable code (C1…C7) referenced by docs/OPEN_POLICY.md.
    pub code: &'static str,
    pub title: &'static str,
    pub status: Status,
    /// A gating check that fails flips the verdict to non-indexable.
    pub gating: bool,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct QualityReport {
    pub checks: Vec<Check>,
    /// Every gating check passed (Warn allowed): pruning + refinement
    /// will work on this file.
    pub indexable: bool,
    /// Uncompressed bytes of the geometry column's leaves — the decode
    /// size proxy used by the Direct-mode ceiling and dialog estimate.
    pub geom_bytes: u64,
    /// C2's measured overlap fraction, when boxes exist.
    #[allow(dead_code)]
    pub overlap_frac: Option<f64>,
}

impl QualityReport {
    /// Gating failures only, for the dialog summary.
    pub fn failures(&self) -> impl Iterator<Item = &Check> {
        self.checks
            .iter()
            .filter(|c| c.gating && c.status == Status::Fail)
    }
}

/// Everything the analyzer needs, all derived from the parquet footer
/// and geo metadata (no data pages).
pub struct QualityInput<'a> {
    pub rows: u64,
    pub row_groups: usize,
    pub rg_rows_max: u64,
    /// Uncompressed bytes of the largest row group.
    pub rg_bytes_max: u64,
    /// Merged per-row-group bboxes with their source label, as resolved
    /// by the loader (native geo stats, covering stats, coordinate
    /// stats, or the per-file bbox fallback of multi-file datasets).
    pub boxes: Option<(&'a str, &'a [[f64; 4]])>,
    pub encoding: GeomEncoding,
    /// Geometry synthesized from x/y coordinate columns.
    pub xy_synthesized: bool,
    /// Every column chunk carries a page index (offset index offset).
    pub page_index: bool,
    /// Compression of the geometry column, as shown in the info panel.
    pub geom_compression: Option<&'a str>,
    pub geo: &'a GeoParquetInfo,
    pub geom_bytes: u64,
}

/// Average pairwise-overlap fraction of the boxes: 0 = disjoint,
/// 1 = every box intersects every other. Same formula as
/// `RgBboxes::overlap_frac` over `bbox_overlap_metric`.
fn overlap_frac(boxes: &[[f64; 4]]) -> f64 {
    super::loader::bbox_overlap_metric(boxes) / (boxes.len().max(2) - 1) as f64
}

pub fn analyze(inp: &QualityInput) -> QualityReport {
    let mut checks = Vec::with_capacity(7);

    // C1 — row-group bboxes available from metadata (gating).
    let frac = inp.boxes.map(|(_, b)| overlap_frac(b));
    let file_level = inp
        .boxes
        .is_some_and(|(src, _)| src.contains("file-level"));
    checks.push(match (inp.boxes, file_level) {
        (Some((src, b)), false) => Check {
            code: "C1",
            title: "spatial index",
            status: Status::Pass,
            gating: true,
            detail: format!("{} row-group bboxes from {src}", b.len()),
        },
        (Some((_, b)), true) => Check {
            code: "C1",
            title: "spatial index",
            status: Status::Warn,
            gating: true,
            detail: format!(
                "only file-level bboxes ({} groups): pruning works per file, \
                 not per row group",
                b.len()
            ),
        },
        (None, _) => Check {
            code: "C1",
            title: "spatial index",
            status: Status::Fail,
            gating: true,
            detail: "no row-group bboxes: no covering column, no native geo \
                     statistics, and WKB column statistics are unusable"
                .into(),
        },
    });

    // C2 — spatial clustering (gating). Only measurable with boxes.
    // Pass on the fraction OR the absolute chain floor: sorted files
    // overlap ~2 neighbors per box regardless of group count.
    checks.push(match inp.boxes {
        Some((_, b)) => {
            let n = b.len();
            let avg = super::loader::bbox_overlap_metric(b);
            let allowed = (OVERLAP_FRAC_MAX * (n.max(2) - 1) as f64).max(CHAIN_OVERLAP_MIN);
            let f = frac.unwrap_or(0.0);
            if avg <= allowed {
                Check {
                    code: "C2",
                    title: "spatial ordering",
                    status: Status::Pass,
                    gating: true,
                    detail: format!(
                        "each row-group bbox overlaps ×{avg:.1} others of {n} \
                         ({:.0}% of possible)",
                        f * 100.0
                    ),
                }
            } else {
                Check {
                    code: "C2",
                    title: "spatial ordering",
                    status: Status::Fail,
                    gating: true,
                    detail: format!(
                        "not spatially sorted: each row-group bbox overlaps \
                         ×{avg:.1} others of {n} ({:.0}% of possible) — most \
                         viewports touch most row groups",
                        f * 100.0
                    ),
                }
            }
        }
        None => Check {
            code: "C2",
            title: "spatial ordering",
            status: Status::Fail,
            gating: true,
            detail: "not measurable without row-group bboxes (C1)".into(),
        },
    });

    // C3 — row-group granularity (gating).
    let mean_rows = inp.rows / (inp.row_groups.max(1) as u64);
    checks.push(if inp.rg_rows_max > RG_ROWS_MAX {
        Check {
            code: "C3",
            title: "row-group size",
            status: Status::Fail,
            gating: true,
            detail: format!(
                "largest row group has {} rows (max {}): pruning and \
                 refinement decode in units too big for the row budget",
                inp.rg_rows_max, RG_ROWS_MAX
            ),
        }
    } else if inp.rg_bytes_max > RG_BYTES_MAX {
        Check {
            code: "C3",
            title: "row-group size",
            status: Status::Fail,
            gating: true,
            detail: format!(
                "largest row group holds {} of geometry and attributes \
                 (max {}) in {} rows: heavy features make a row group \
                 that must be fetched and decoded whole, however few \
                 rows it counts",
                super::info::fmt_bytes(inp.rg_bytes_max),
                super::info::fmt_bytes(RG_BYTES_MAX),
                inp.rg_rows_max
            ),
        }
    } else if mean_rows < RG_ROWS_MIN
        && inp.row_groups > 1
        && inp.rg_bytes_max < RG_BYTES_MIN
    {
        Check {
            code: "C3",
            title: "row-group size",
            status: Status::Warn,
            gating: true,
            detail: format!(
                "row groups average {mean_rows} rows and none exceeds {}: \
                 footer and per-group overhead dominate",
                super::info::fmt_bytes(inp.rg_bytes_max)
            ),
        }
    } else {
        Check {
            code: "C3",
            title: "row-group size",
            status: Status::Pass,
            gating: true,
            detail: format!(
                "{} groups, largest {} rows / {}",
                inp.row_groups,
                inp.rg_rows_max,
                super::info::fmt_bytes(inp.rg_bytes_max)
            ),
        }
    });

    // C4 — geometry encoding (advisory).
    checks.push(if inp.xy_synthesized {
        Check {
            code: "C4",
            title: "encoding",
            status: Status::Pass,
            gating: false,
            detail: "x/y coordinate columns: columnar decode, free statistics"
                .into(),
        }
    } else if inp.encoding.is_wkb() {
        Check {
            code: "C4",
            title: "encoding",
            status: Status::Warn,
            gating: false,
            detail: "WKB parses feature by feature; GeoArrow coordinate \
                     arrays decode faster"
                .into(),
        }
    } else {
        Check {
            code: "C4",
            title: "encoding",
            status: Status::Pass,
            gating: false,
            detail: "GeoArrow coordinate arrays".into(),
        }
    });

    // C5 — page index (advisory).
    checks.push(Check {
        code: "C5",
        title: "page index",
        status: if inp.page_index { Status::Pass } else { Status::Warn },
        gating: false,
        detail: if inp.page_index {
            "present on every column chunk".into()
        } else {
            "absent: sub-row-group pruning unavailable to readers".into()
        },
    });

    // C6 — compression (advisory).
    let comp = inp.geom_compression.unwrap_or("?");
    let comp_upper = comp.to_ascii_uppercase();
    let uncompressed = comp_upper.contains("UNCOMPRESSED");
    let zstd = comp_upper.contains("ZSTD");
    checks.push(Check {
        code: "C6",
        title: "compression",
        status: if uncompressed { Status::Warn } else { Status::Pass },
        gating: false,
        detail: if uncompressed {
            "geometry column is uncompressed".into()
        } else if zstd {
            format!("geometry column: {comp}")
        } else {
            format!(
                "geometry column: {comp}; zstd recommended for distribution"
            )
        },
    });

    // C7 — metadata hygiene (advisory).
    let mut missing: Vec<&str> = Vec::new();
    if inp.geo.version_label.starts_with("none") {
        missing.push("geo metadata (CRS assumed)");
    }
    if inp.geo.geometry_types.is_empty() && !inp.xy_synthesized {
        missing.push("declared geometry_types");
    }
    if inp.geo.bbox.is_none() {
        missing.push("file bbox");
    }
    checks.push(Check {
        code: "C7",
        title: "metadata",
        status: if missing.is_empty() { Status::Pass } else { Status::Warn },
        gating: false,
        detail: if missing.is_empty() {
            "geo metadata, geometry types, CRS and bbox all declared".into()
        } else {
            format!("missing: {}", missing.join(", "))
        },
    });

    let indexable = !checks
        .iter()
        .any(|c| c.gating && c.status == Status::Fail);
    QualityReport {
        checks,
        indexable,
        geom_bytes: inp.geom_bytes,
        overlap_frac: frac,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo_ok() -> GeoParquetInfo {
        GeoParquetInfo {
            version_label: "GeoParquet 1.1.0".into(),
            geometry_types: vec!["MultiPolygon".into()],
            bbox: Some([-79.6, 45.0, -57.1, 62.4]),
            ..Default::default()
        }
    }

    /// Disjoint boxes along a line: overlap 0.
    fn sorted_boxes(n: usize) -> Vec<[f64; 4]> {
        (0..n)
            .map(|i| {
                let x = i as f64 * 2.0;
                [x, 0.0, x + 1.0, 1.0]
            })
            .collect()
    }

    fn input<'a>(
        boxes: Option<(&'a str, &'a [[f64; 4]])>,
        rows: u64,
        row_groups: usize,
        rg_rows_max: u64,
        geo: &'a GeoParquetInfo,
    ) -> QualityInput<'a> {
        QualityInput {
            rows,
            row_groups,
            rg_rows_max,
            // Test files are sized by rows unless a case says
            // otherwise; small enough that the byte branches stay out of
            // the way.
            rg_bytes_max: 1 << 20,
            boxes,
            encoding: GeomEncoding::MultiPolygon,
            xy_synthesized: false,
            page_index: true,
            geom_compression: Some("ZSTD(ZstdLevel(3))"),
            geo,
            geom_bytes: 1 << 30,
        }
    }

    fn status(r: &QualityReport, code: &str) -> Status {
        r.checks.iter().find(|c| c.code == code).unwrap().status
    }

    #[test]
    fn few_rows_of_heavy_geometry_are_not_overhead() {
        // Byte-capped splitting of administrative boundaries gives
        // groups of a few dozen features weighing tens of megabytes.
        // Warning about per-group overhead there would argue for undoing
        // the split that made the file prunable in the first place.
        let geo = geo_ok();
        let boxes = sorted_boxes(6);
        let mut inp = input(Some(("bbox", &boxes)), 218, 6, 61, &geo);
        inp.rg_bytes_max = 33 << 20;
        assert_eq!(status(&analyze(&inp), "C3"), Status::Pass);
        // Genuinely tiny groups still warn.
        inp.rg_bytes_max = 200 << 10;
        assert_eq!(status(&analyze(&inp), "C3"), Status::Warn);
    }

    #[test]
    fn heavy_row_groups_fail_however_few_rows_they_hold() {
        // The case that motivated the check: 218 administrative
        // boundaries, one row group, 162 MB. By rows it is trivially
        // small; as a decode unit it is the whole file.
        let geo = geo_ok();
        let boxes = sorted_boxes(1);
        let mut inp = input(Some(("bbox", &boxes)), 218, 1, 218, &geo);
        inp.rg_bytes_max = 162 << 20;
        let r = analyze(&inp);
        assert_eq!(status(&r, "C3"), Status::Fail);
        let d = &r.checks.iter().find(|c| c.code == "C3").unwrap().detail;
        assert!(d.contains("162.0 MB"), "{d}");
        assert!(!r.indexable, "a gating fail must block the fast path");

        // Same shape under the limit passes, and says both measures.
        inp.rg_bytes_max = 40 << 20;
        let r = analyze(&inp);
        assert_eq!(status(&r, "C3"), Status::Pass);
        let d = &r.checks.iter().find(|c| c.code == "C3").unwrap().detail;
        assert!(d.contains("218 rows") && d.contains("40.0 MB"), "{d}");
    }

    #[test]
    fn ideal_file_is_indexable() {
        let geo = geo_ok();
        let boxes = sorted_boxes(68);
        let r = analyze(&input(
            Some(("covering column statistics", &boxes)),
            4_452_455,
            68,
            65_536,
            &geo,
        ));
        assert!(r.indexable);
        assert!(r.checks.iter().all(|c| c.status == Status::Pass));
    }

    #[test]
    fn no_bboxes_fails_c1_and_c2() {
        let geo = geo_ok();
        let mut inp = input(None, 4_452_455, 37, 121_000, &geo);
        inp.encoding = GeomEncoding::Wkb;
        let r = analyze(&inp);
        assert!(!r.indexable);
        assert_eq!(status(&r, "C1"), Status::Fail);
        assert_eq!(status(&r, "C2"), Status::Fail);
        assert_eq!(status(&r, "C3"), Status::Pass);
        assert_eq!(status(&r, "C4"), Status::Warn);
        assert_eq!(r.overlap_frac, None);
    }

    #[test]
    fn unsorted_boxes_fail_c2() {
        let geo = geo_ok();
        // Every group spans the whole extent: overlap fraction 100%.
        let boxes = vec![[-79.6, 45.0, -57.1, 62.4]; 37];
        let r = analyze(&input(
            Some(("computed at load", &boxes)),
            4_452_455,
            37,
            121_000,
            &geo,
        ));
        assert!(!r.indexable);
        assert_eq!(status(&r, "C1"), Status::Pass);
        assert_eq!(status(&r, "C2"), Status::Fail);
        assert!(r.overlap_frac.unwrap() > 0.9);
    }

    #[test]
    fn oversized_row_groups_fail_c3() {
        let geo = geo_ok();
        let boxes = sorted_boxes(4);
        let r = analyze(&input(
            Some(("covering column statistics", &boxes)),
            5_000_000,
            4,
            1_250_000,
            &geo,
        ));
        assert!(!r.indexable);
        assert_eq!(status(&r, "C3"), Status::Fail);
    }

    #[test]
    fn warns_do_not_gate() {
        let geo = GeoParquetInfo::default(); // C7 warns: everything missing
        let boxes = sorted_boxes(200);
        let mut inp = input(
            Some(("coordinate column statistics (GeoArrow)", &boxes)),
            1_000_000,
            200,
            5_000,
            &geo,
        );
        inp.page_index = false; // C5 warn
        inp.geom_compression = Some("UNCOMPRESSED"); // C6 warn
        let r = analyze(&inp); // C3 warns: tiny groups
        assert!(r.indexable);
        assert_eq!(status(&r, "C3"), Status::Warn);
        assert_eq!(status(&r, "C5"), Status::Warn);
        assert_eq!(status(&r, "C6"), Status::Warn);
        assert_eq!(status(&r, "C7"), Status::Warn);
    }

    #[test]
    fn spatially_partitioned_file_bboxes_pass_the_chain_floor() {
        let geo = geo_ok();
        // Two files × 3 groups each: identical boxes within a file,
        // disjoint across files (spatially partitioned dataset). Each box
        // overlaps exactly its 2 file-mates — within the chain floor, so
        // file-granularity pruning counts as workable.
        let mut boxes = vec![[0.0, 0.0, 1.0, 1.0]; 3];
        boxes.extend(vec![[2.0, 0.0, 3.0, 1.0]; 3]);
        let r = analyze(&input(
            Some(("file-level geo bbox", &boxes)),
            2_000_000,
            6,
            400_000,
            &geo,
        ));
        assert_eq!(status(&r, "C1"), Status::Warn);
        assert_eq!(status(&r, "C2"), Status::Pass);
        assert!(r.indexable, "C1 warn does not gate");
    }

    /// The chain floor must not let a small unsorted file through: 5
    /// mutually overlapping groups exceed both the floor and the fraction.
    #[test]
    fn small_unsorted_file_still_fails_c2() {
        let geo = geo_ok();
        let boxes = vec![[0.0, 0.0, 10.0, 10.0]; 5];
        let r = analyze(&input(
            Some(("computed at load", &boxes)),
            3_000_000,
            5,
            512_000,
            &geo,
        ));
        assert_eq!(status(&r, "C2"), Status::Fail);
        assert!(!r.indexable);
    }

    #[test]
    fn single_group_small_file_passes() {
        let geo = geo_ok();
        let boxes = sorted_boxes(1);
        let r = analyze(&input(
            Some(("covering column statistics", &boxes)),
            100_000,
            1,
            100_000,
            &geo,
        ));
        assert!(r.indexable);
    }
}
