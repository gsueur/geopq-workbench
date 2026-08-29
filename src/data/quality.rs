//! Footer-only file quality analysis: is this file shaped so the
//! viewport machinery (row-group pruning, per-feature selection,
//! refinement) can work, and how close is it to display best practice?
//!
//! See docs/OPEN_POLICY.md. The report states facts about the file;
//! the open policy (Indexed vs Direct mode, the optimize offer) is
//! decided by the app from the report plus the row count.

use super::cogp::LevelRun;
use super::geoarrow::GeomEncoding;
use super::info::GeoParquetInfo;

/// C2 threshold on [`Clustering::frac`]: above this, row-group bboxes
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
/// C2: average geometry bytes per feature above which features are big
/// enough that their bounding boxes overlap however the file is sorted,
/// and "sort it" is the wrong advice.
pub const SPRAWL_BYTES: u64 = 8 << 10;
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

/// COGP facts for C8, as resolved by the loader from the footer.
#[derive(Clone, Debug)]
pub struct CogpQuality {
    pub version: String,
    /// Each level's last row group and row count, coarse to fine. C2
    /// reads them to measure clustering inside a level instead of across
    /// the file; see the check for why that is the only honest measure
    /// of a COGP layout.
    pub levels: Vec<LevelRun>,
    /// Rows in the whole file, against which level 0's prefix is the
    /// share of the dataset a first paint costs.
    pub total_rows: u64,
    /// Which bbox statistics the prefix is prunable by.
    pub pruning: &'static str,
    /// Native 2.0 statistics rather than the 1.1 covering the published
    /// profile names.
    pub extension_2_0: bool,
}

impl CogpQuality {
    fn runs(&self) -> Vec<LevelRun> {
        self.levels.clone()
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Row groups a viewport covering the whole file bbox reads at the
    /// coarsest level — the quality metric SPEC §8 suggests, which for a
    /// whole-file viewport is simply level 0's prefix length.
    pub fn level0_groups(&self) -> usize {
        self.levels.first().map_or(0, |l| l.row_group_end + 1)
    }

    pub fn level0_rows(&self) -> u64 {
        self.levels.first().map_or(0, |l| l.rows)
    }
}

#[derive(Clone, Debug)]
pub struct Check {
    /// Stable code (C1…C9) referenced by docs/OPEN_POLICY.md.
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
    /// COGP levels: None when the file carries no `cogp` block, Err when
    /// it carries one that does not validate.
    pub cogp: Option<Result<CogpQuality, String>>,
}

/// C2's clustering measurement over one run of row-group boxes.
#[derive(Clone, Copy, Debug)]
pub struct Clustering {
    /// Boxes measured.
    pub n: usize,
    /// Average number of *other* boxes each one intersects.
    pub avg: f64,
    /// What the chain rule allows for `n` boxes.
    allowed: f64,
    /// Rows in the run, when it is a COGP level.
    rows: Option<u64>,
}

impl Clustering {
    fn measure(boxes: &[[f64; 4]], rows: Option<u64>) -> Self {
        let n = boxes.len();
        Clustering {
            n,
            avg: super::loader::bbox_overlap_metric(boxes),
            allowed: (OVERLAP_FRAC_MAX * (n.max(2) - 1) as f64).max(CHAIN_OVERLAP_MIN),
            rows,
        }
    }

    /// A run small enough to decode whole cannot be the trap C2 exists
    /// to catch: however badly its boxes overlap, every viewport in it
    /// costs at most the whole run, and that is within budget.
    ///
    /// This is the per-level form of the policy's existing early pass
    /// ("small files cannot fall into the permanent-preview trap"), and
    /// it is what a COGP file needs: its coarse levels hold a handful of
    /// groups of widely spread features, so they overlap each other more
    /// than a chain does — the reference converter's own output measures
    /// ×2.4 over a 5-group level — while being a few tens of thousands
    /// of rows that are read whole and never previewed.
    fn bounded(&self) -> bool {
        self.rows.is_some_and(|r| r <= super::loader::MAX_BUILD_ROWS)
    }

    /// Overlap as a fraction of the possible overlaps: 0 = disjoint,
    /// 1 = every box intersects every other.
    pub fn frac(&self) -> f64 {
        self.avg / (self.n.max(2) - 1) as f64
    }

    pub fn passes(&self) -> bool {
        self.avg <= self.allowed || self.bounded()
    }

    /// How far past what the chain rule allows this run sits. `allowed`
    /// is at least [`CHAIN_OVERLAP_MIN`], so this never divides by zero.
    fn ratio(&self) -> f64 {
        self.avg / self.allowed
    }

    /// The run C2 judges the file on: normally the whole file, and on a
    /// COGP layout the worst level measured inside itself, with that
    /// level's index.
    ///
    /// A COGP file is ordered by (level, spatial curve). Coarse levels
    /// hold few features spread over the whole dataset, so their row
    /// groups' bboxes span the whole extent and every finer group sits
    /// inside them — by construction, not by bad sorting. Measured as
    /// one sequence, a correct 55-group file reads ×36.5 overlaps (68%)
    /// and fails the gate. What the profile promises, and what pruning
    /// within a chosen prefix actually depends on, is tight clustering
    /// *inside* a level.
    pub fn worst(boxes: &[[f64; 4]], levels: Option<&[LevelRun]>) -> (Option<usize>, Self) {
        let whole = || (None, Self::measure(boxes, None));
        if boxes.is_empty() {
            return whole();
        }
        let Some(levels) = levels else { return whole() };
        let ends: Vec<usize> = levels.iter().map(|l| l.row_group_end).collect();
        let Some(ranges) = super::cogp::level_ranges(&ends, boxes.len()) else {
            return whole();
        };
        let measured: Vec<(usize, Self)> = ranges
            .into_iter()
            .enumerate()
            .map(|(k, (start, end))| {
                (k, Self::measure(&boxes[start..=end], Some(levels[k].rows)))
            })
            .collect();
        // A level that fails decides the check, so it must also be the
        // one reported; only when every level passes does the loudest
        // one stand in as "the worst".
        let pick = |c: &&(usize, Self)| c.1.ratio();
        measured
            .iter()
            .filter(|c| !c.1.passes())
            .max_by(|a, b| pick(a).total_cmp(&pick(b)))
            .or_else(|| measured.iter().max_by(|a, b| pick(a).total_cmp(&pick(b))))
            .map(|&(k, m)| (Some(k), m))
            .unwrap_or_else(whole)
    }
}

/// Why C2 sees the overlap it sees. The metric cannot tell a badly
/// sorted file from one whose features genuinely sprawl (country
/// outlines with overseas territories, long rivers, dateline crossers),
/// and telling that second user to sort a sorted file sends them after a
/// fix that does not exist. Average geometry size separates the two well
/// enough to name the likely one.
fn overlap_cause(inp: &QualityInput) -> String {
    let per_feature = inp.geom_bytes / inp.rows.max(1);
    if per_feature >= SPRAWL_BYTES {
        format!(
            "features average {} of geometry, so their bounding boxes may \
             overlap whatever the row order",
            super::info::fmt_bytes(per_feature)
        )
    } else {
        "the rows are probably not spatially sorted".to_string()
    }
}

pub fn analyze(inp: &QualityInput) -> QualityReport {
    let mut checks = Vec::with_capacity(8);

    // C1 — row-group bboxes available from metadata (gating).
    // COGP levels, when the file carries valid ones, decide how C2
    // measures — see `Clustering::worst`.
    let runs = match &inp.cogp {
        Some(Ok(c)) => Some(c.runs()),
        _ => None,
    };
    let c2 = inp.boxes.map(|(_, b)| Clustering::worst(b, runs.as_deref()));
    let frac = c2.map(|(_, m)| m.frac());
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
    // overlap ~2 neighbors per box regardless of group count. On a COGP
    // file the run measured is one level, not the file.
    checks.push(match c2 {
        Some((Some(k), m)) => {
            let head = format!(
                "within COGP levels: worst level {k} overlaps ×{:.1} of {} \
                 ({:.0}% of possible)",
                m.avg,
                m.n,
                m.frac() * 100.0
            );
            Check {
                code: "C2",
                title: "spatial ordering",
                status: if m.passes() { Status::Pass } else { Status::Fail },
                gating: true,
                detail: if m.avg <= m.allowed {
                    format!(
                        "{head}; levels each span the whole extent by design, so \
                         the file as a whole is not the measure"
                    )
                } else if m.passes() {
                    format!(
                        "{head}, but it holds {} rows — small enough to decode \
                         whole, so no viewport in it can be stuck previewing",
                        m.rows.unwrap_or(0)
                    )
                } else {
                    format!(
                        "{head}: most viewports touch most of that level's row \
                         groups — {}",
                        overlap_cause(inp)
                    )
                },
            }
        }
        Some((None, m)) => Check {
            code: "C2",
            title: "spatial ordering",
            status: if m.passes() { Status::Pass } else { Status::Fail },
            gating: true,
            detail: if m.passes() {
                format!(
                    "each row-group bbox overlaps ×{:.1} others of {} \
                     ({:.0}% of possible)",
                    m.avg,
                    m.n,
                    m.frac() * 100.0
                )
            } else {
                format!(
                    "each row-group bbox overlaps ×{:.1} others of {} \
                     ({:.0}% of possible): most viewports touch most row \
                     groups — {}",
                    m.avg,
                    m.n,
                    m.frac() * 100.0,
                    overlap_cause(inp)
                )
            },
        },
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

    // C8 — cloud-optimized levels (advisory, never gating).
    //
    // Absence is not a fault: COGP is one way to lay a file out, not a
    // requirement, and a well-sorted 1.1 file with a covering column is
    // already fast here. A `cogp` block that does not validate is worth
    // a warning, because it means a producer meant to write one.
    checks.push(match &inp.cogp {
        None => Check {
            code: "C8",
            title: "cloud-optimized levels",
            status: Status::Pass,
            gating: false,
            detail: "no COGP levels (optional)".into(),
        },
        Some(Err(e)) => Check {
            code: "C8",
            title: "cloud-optimized levels",
            status: Status::Warn,
            gating: false,
            detail: format!("`cogp` metadata present but not usable: {e}"),
        },
        Some(Ok(c)) => Check {
            code: "C8",
            title: "cloud-optimized levels",
            status: Status::Pass,
            gating: false,
            detail: format!(
                "COGP {}{}: {} levels, prunable by {}; a whole-file viewport \
                 reads {} of {} row groups at the coarsest level ({:.0}% of rows)",
                c.version,
                if c.extension_2_0 { " (2.0 extension)" } else { "" },
                c.level_count(),
                c.pruning,
                c.level0_groups(),
                inp.row_groups,
                100.0 * c.level0_rows() as f64 / c.total_rows.max(1) as f64,
            ),
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

/// Pyramid facts for C9, as resolved by the loader from
/// `h3-pyramid.json` and one listing of the root.
#[derive(Clone, Debug)]
pub struct PyramidQuality {
    /// The descriptor in one line (`PyramidState::info_line`).
    pub summary: String,
    /// Files the descriptor's cell lists name.
    pub listed: usize,
    /// …of those, the ones the root does not hold.
    pub missing: Vec<String>,
    /// The root serves no listing (an HTTPS prefix), so the descriptor
    /// is taken at its word rather than checked.
    pub unlisted: bool,
}

/// C9 — H3 pyramid (advisory, never gating).
///
/// Absence is not a fault: a pyramid is one way to publish a layer, and
/// the plain partitioned datasets this app has always read are not worse
/// files for lacking one. A descriptor that is there but does not hold
/// up is worth a warning, because a producer meant to write one and a
/// reader will silently fall back to reading every file under the root.
pub fn pyramid_check(p: Option<&Result<PyramidQuality, String>>) -> Check {
    const TITLE: &str = "H3 pyramid";
    match p {
        None => Check {
            code: "C9",
            title: TITLE,
            status: Status::Pass,
            gating: false,
            detail: "no pyramid (optional)".into(),
        },
        Some(Err(e)) => Check {
            code: "C9",
            title: TITLE,
            status: Status::Warn,
            gating: false,
            detail: format!("{e}; opened as a plain partitioned dataset"),
        },
        Some(Ok(q)) if !q.missing.is_empty() => Check {
            code: "C9",
            title: TITLE,
            status: Status::Warn,
            gating: false,
            detail: format!(
                "{}; {} of {} listed files are missing (e.g. {})",
                q.summary,
                q.missing.len(),
                q.listed,
                q.missing
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        },
        Some(Ok(q)) => Check {
            code: "C9",
            title: TITLE,
            status: Status::Pass,
            gating: false,
            detail: if q.unlisted {
                format!(
                    "{}; {} files listed, not verified (an https prefix serves no listing)",
                    q.summary, q.listed
                )
            } else {
                format!("{}; all {} listed files present", q.summary, q.listed)
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// C9 is advisory in every direction: no pyramid is not a fault, a
    /// descriptor that does not hold is a warning, and a listed file
    /// that is not there is a warning the user can act on.
    #[test]
    fn c9_reports_the_pyramid_without_ever_gating() {
        let ok = |missing: Vec<String>, unlisted: bool| {
            Ok(PyramidQuality {
                summary: "leaf r8, overviews r5..r7 (dissolve), 64 px/cell".into(),
                listed: 700,
                missing,
                unlisted,
            })
        };
        let c = pyramid_check(None);
        assert_eq!((c.code, c.status, c.gating), ("C9", Status::Pass, false));
        assert_eq!(c.detail, "no pyramid (optional)");

        let c = pyramid_check(Some(&ok(Vec::new(), false)));
        assert_eq!(c.status, Status::Pass);
        assert!(c.detail.ends_with("all 700 listed files present"), "{}", c.detail);

        let c = pyramid_check(Some(&ok(Vec::new(), true)));
        assert_eq!(c.status, Status::Pass);
        assert!(c.detail.contains("not verified"), "{}", c.detail);

        let c = pyramid_check(Some(&ok(vec!["r8/abc.parquet".into()], false)));
        assert_eq!(c.status, Status::Warn);
        assert!(c.detail.contains("1 of 700 listed files are missing"), "{}", c.detail);
        assert!(!c.gating);

        let c = pyramid_check(Some(&Err("pyramid: resolution 16 out of range".into())));
        assert_eq!(c.status, Status::Warn);
        assert!(c.detail.contains("out of range"), "{}", c.detail);
        assert!(!c.gating);
    }

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
            cogp: None,
        }
    }

    fn status(r: &QualityReport, code: &str) -> Status {
        r.checks.iter().find(|c| c.code == code).unwrap().status
    }

    #[test]
    fn c2_names_sprawl_rather_than_blaming_the_sort() {
        // Administrative boundaries: Hilbert-sorted, yet every group's
        // bbox overlaps every other, because a country's bbox spans its
        // overseas territories. Telling the user to sort a sorted file
        // sends them after a fix that does not exist.
        let geo = geo_ok();
        let boxes = vec![[-180.0, -60.0, 180.0, 75.0]; 6];
        let mut inp = input(Some(("bbox", &boxes)), 218, 6, 61, &geo);
        inp.geom_bytes = 218 * (400 << 10); // ~400 KB per country
        let r = analyze(&inp);
        assert_eq!(status(&r, "C2"), Status::Fail);
        let d = &r.checks.iter().find(|c| c.code == "C2").unwrap().detail;
        assert!(d.contains("whatever the row order"), "{d}");
        assert!(!d.contains("not spatially sorted"), "{d}");

        // Small features overlapping that much really is a sort problem.
        inp.geom_bytes = 218 * 200;
        let r = analyze(&inp);
        let d = &r.checks.iter().find(|c| c.code == "C2").unwrap().detail;
        assert!(d.contains("not spatially sorted"), "{d}");
        // Either way the measurement itself is stated.
        assert!(d.contains("of possible"), "{d}");
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

    fn cogp(extension_2_0: bool) -> CogpQuality {
        CogpQuality {
            version: "0.1.0".into(),
            levels: vec![
                LevelRun { row_group_end: 0, rows: 40_000 },
                LevelRun { row_group_end: 3, rows: 160_000 },
                LevelRun { row_group_end: 12, rows: 800_000 },
                LevelRun { row_group_end: 67, rows: 3_000_000 },
            ],
            total_rows: 4_000_000,
            pruning: if extension_2_0 {
                "native geospatial statistics"
            } else {
                "covering column statistics"
            },
            extension_2_0,
        }
    }

    /// C8 is advisory in every direction: a file without COGP levels is
    /// not penalised, and a file whose `cogp` block is broken warns
    /// rather than failing — the layout is still readable GeoParquet.
    #[test]
    fn c8_grades_cogp_levels_without_ever_gating() {
        let geo = geo_ok();
        let boxes = sorted_boxes(68);
        let base = |c: Option<Result<CogpQuality, String>>| {
            let mut inp = input(
                Some(("covering column statistics", &boxes)),
                4_000_000,
                68,
                65_536,
                &geo,
            );
            inp.cogp = c;
            analyze(&inp)
        };
        let detail = |r: &QualityReport| {
            r.checks.iter().find(|c| c.code == "C8").unwrap().detail.clone()
        };

        // Absent: optional, so it passes and says so.
        let r = base(None);
        assert_eq!(status(&r, "C8"), Status::Pass);
        assert!(detail(&r).contains("optional"), "{}", detail(&r));
        assert!(r.indexable);

        // Present and valid: the spec's suggested quality metrics — the
        // coarsest level's prefix length and its share of the rows.
        let r = base(Some(Ok(cogp(false))));
        assert_eq!(status(&r, "C8"), Status::Pass);
        let d = detail(&r);
        assert!(d.contains("COGP 0.1.0:"), "{d}");
        assert!(d.contains("4 levels"), "{d}");
        assert!(d.contains("1 of 68 row groups"), "{d}");
        assert!(d.contains("1% of rows"), "{d}");
        assert!(d.contains("covering column statistics"), "{d}");

        // The 2.0 form is labelled, so nobody reads it as conformance
        // with the published 1.1 profile.
        let d = detail(&base(Some(Ok(cogp(true)))));
        assert!(d.contains("(2.0 extension)"), "{d}");
        assert!(d.contains("native geospatial statistics"), "{d}");

        // Present and broken: a warning, never a failure.
        let r = base(Some(Err("gsd 250 does not decrease from 100".into())));
        assert_eq!(status(&r, "C8"), Status::Warn);
        assert!(detail(&r).contains("does not decrease"), "{}", detail(&r));
        assert!(r.indexable, "C8 never gates");
    }

    /// A COGP layout the naive metric condemns and the per-level one
    /// clears — the bug this check was rewritten for.
    ///
    /// The geometry is what a real conversion produces: coarse levels
    /// hold a handful of widely spread features, so their single row
    /// group's bbox is the whole dataset and every finer group sits
    /// inside it. Twenty such levels plus one dense, tightly chained
    /// level of 35 groups measures ×33.6 of 55 (62%) across the file —
    /// the shape Guillaume hit at ×36.5 of 55 (68%) — while every level
    /// is exactly as clustered as it should be.
    #[test]
    fn c2_measures_clustering_inside_cogp_levels() {
        let geo = geo_ok();
        const COARSE: usize = 20;
        const FINE: usize = 35;
        let extent = [0.0, 0.0, 100.0, 100.0];
        let mut boxes = vec![extent; COARSE];
        let w = 100.0 / FINE as f64;
        boxes.extend((0..FINE).map(|i| {
            let x = i as f64 * w;
            [x, 0.0, x + w, 100.0]
        }));
        // The fine level is deliberately over the build budget, so it
        // passes on its clustering and not on its size.
        let fine_rows = super::super::loader::MAX_BUILD_ROWS + 100_000;
        let mut levels: Vec<LevelRun> = (0..COARSE)
            .map(|k| LevelRun { row_group_end: k, rows: 500 })
            .collect();
        levels.push(LevelRun { row_group_end: COARSE + FINE - 1, rows: fine_rows });
        let total_rows = COARSE as u64 * 500 + fine_rows;

        let cogp = |levels: Vec<LevelRun>| CogpQuality {
            version: "0.1.0".into(),
            levels,
            total_rows,
            pruning: "covering column statistics",
            extension_2_0: false,
        };
        let run = |boxes: &[[f64; 4]], c: Option<CogpQuality>| {
            let mut inp = input(
                Some(("covering column statistics", boxes)),
                total_rows,
                boxes.len(),
                65_536,
                &geo,
            );
            inp.cogp = c.map(Ok);
            analyze(&inp)
        };
        let detail = |r: &QualityReport| {
            r.checks.iter().find(|c| c.code == "C2").unwrap().detail.clone()
        };

        // Measured as one file it looks like an unsorted export…
        let naive = run(&boxes, None);
        assert_eq!(status(&naive, "C2"), Status::Fail);
        assert!(!naive.indexable, "this is the gate the user hit");
        let d = detail(&naive);
        assert!(d.contains("of possible") && d.contains("not spatially sorted"), "{d}");
        assert!(naive.overlap_frac.unwrap() > 0.5, "{:?}", naive.overlap_frac);

        // …and measured inside its levels it is exactly what COGP asks
        // for. The worst level is the dense one, which is also the only
        // one big enough to matter.
        let r = run(&boxes, Some(cogp(levels.clone())));
        assert_eq!(status(&r, "C2"), Status::Pass);
        assert!(r.indexable, "a well-formed COGP file must pass the gate");
        let d = detail(&r);
        assert!(d.starts_with(&format!("within COGP levels: worst level {COARSE} ")), "{d}");
        assert!(d.contains("of possible"), "{d}");
        assert!(r.overlap_frac.unwrap() < 0.1, "{:?}", r.overlap_frac);

        // Not a rubber stamp: a level that really cannot be pruned, and
        // is too big to read whole, still fails.
        let mut sprawling = vec![extent; COARSE];
        sprawling.extend(vec![extent; FINE]);
        let r = run(&sprawling, Some(cogp(levels.clone())));
        assert_eq!(status(&r, "C2"), Status::Fail);
        assert!(!r.indexable);
        assert!(detail(&r).contains(&format!("worst level {COARSE}")), "{}", detail(&r));

        // …but the same shape within the build budget passes on size:
        // reading the level whole is bounded work, so no viewport in it
        // can be stuck previewing.
        let mut small = levels;
        small[COARSE] = LevelRun {
            row_group_end: COARSE + FINE - 1,
            rows: super::super::loader::MAX_BUILD_ROWS,
        };
        let r = run(&sprawling, Some(cogp(small)));
        assert_eq!(status(&r, "C2"), Status::Pass);
        assert!(detail(&r).contains("small enough to decode whole"), "{}", detail(&r));
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
