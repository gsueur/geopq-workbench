//! Cloud Optimized GeoParquet Profile (COGP) v0.1: a `cogp` file-level
//! metadata block naming, per rendering scale, the prefix of row groups
//! a reader needs.
//!
//! The profile reorders features coarse-to-fine across row groups and
//! never simplifies or duplicates them, so a prefix is not a preview —
//! it is exactly the features that are independently meaningful at that
//! scale. That is why the loader treats a prefix that fits the build
//! budget as an ordinary exact load, not as an approximation to badge.
//!
//! Spec: <https://github.com/Kanahiro/cloud-optimized-geoparquet>
//! (SPEC.md v0.1.0, CC BY 4.0).

use serde::{Deserialize, Serialize};

/// Parquet file-level key-value metadata key (spec §6).
pub const KEY: &str = "cogp";

/// Profile version this build writes.
pub const VERSION: &str = "0.1.0";

/// COGP major version this reader implements. A file declaring a higher
/// major version is rejected: per SPEC §6.1 a major bump may change what
/// the fields mean, and guessing is worse than reading the file plainly.
pub const SUPPORTED_MAJOR: u64 = 0;

/// The row-group bbox statistics a file offers, which is what makes a
/// prefix prunable to a viewport.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pruning {
    /// GeoParquet 1.1 `covering.bbox` whose four columns carry row-group
    /// min/max statistics — what SPEC §5.1 requires.
    Covering,
    /// Native Parquet geospatial statistics on the geometry column, as
    /// GeoParquet 2.0 writers emit (this workbench's own included).
    ///
    /// The spec is written against 1.1 and predates them, but the
    /// requirement it actually states is that a reader can prune row
    /// groups by bbox, and these answer it. A 2.0 file carrying them is
    /// read as COGP and labelled as the extension it is, so nobody
    /// mistakes it for something the published profile covers.
    NativeStats,
}

impl Pruning {
    pub fn label(self) -> &'static str {
        match self {
            Pruning::Covering => "covering column statistics",
            Pruning::NativeStats => "native geospatial statistics",
        }
    }
}

/// One rendering detail level: row groups `0..=row_group_end` are the
/// prefix to read for a viewport whose ground sample distance is `gsd`
/// metres per pixel or coarser.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    pub row_group_end: usize,
    /// Metres of ground per pixel, whatever the CRS's own units.
    pub gsd: f64,
}

/// A level as the quality checks measure it: where it ends, and how big
/// it is. Clustering is judged inside a level, and a level small enough
/// to decode whole is judged not at all.
#[derive(Clone, Copy, Debug)]
pub struct LevelRun {
    pub row_group_end: usize,
    pub rows: u64,
}

#[derive(Clone, Debug)]
pub struct CogpLevels {
    pub version: String,
    /// Coarse to fine: `row_group_end` strictly increasing, `gsd`
    /// strictly decreasing, the last entry covering every row group.
    pub levels: Vec<Level>,
    pub pruning: Pruning,
}

impl CogpLevels {
    /// Index of the level to read at `target_gsd` metres per pixel: the
    /// finest level whose `gsd` is still coarser than or equal to the
    /// target (SPEC §7.1). A target coarser than the coarsest level
    /// selects level 0 — there is nothing coarser to give it.
    pub fn level_for_gsd(&self, target_gsd: f64) -> usize {
        self.levels
            .iter()
            .rposition(|l| l.gsd >= target_gsd)
            .unwrap_or(0)
    }

    /// Last row group needed at `target_gsd`.
    pub fn row_group_end_for_gsd(&self, target_gsd: f64) -> usize {
        self.levels[self.level_for_gsd(target_gsd)].row_group_end
    }

    /// Each level's last row group and row count, coarse to fine, from
    /// the file's per-row-group row counts.
    pub fn runs(&self, rg_rows: &[u64]) -> Vec<LevelRun> {
        let mut out = Vec::with_capacity(self.levels.len());
        let mut start = 0usize;
        for l in &self.levels {
            let rows = rg_rows
                .get(start..=l.row_group_end)
                .map_or(0, |r| r.iter().sum());
            out.push(LevelRun { row_group_end: l.row_group_end, rows });
            start = l.row_group_end + 1;
        }
        out
    }

    /// One line for the file info panel.
    pub fn summary(&self) -> String {
        let gsds = self
            .levels
            .iter()
            .map(|l| {
                if l.gsd >= 10.0 {
                    format!("{:.0}", l.gsd)
                } else {
                    format!("{:.1}", l.gsd)
                }
            })
            .collect::<Vec<_>>()
            .join("/");
        let prefixes = self
            .levels
            .iter()
            .map(|l| (l.row_group_end + 1).to_string())
            .collect::<Vec<_>>()
            .join("/");
        let form = match self.pruning {
            Pruning::Covering => String::new(),
            Pruning::NativeStats => " (2.0 extension)".to_string(),
        };
        format!(
            "COGP {}{}: {} levels, gsd {} m, prefix row groups {}",
            self.version,
            form,
            self.levels.len(),
            gsds,
            prefixes
        )
    }
}

/// Parse and structurally validate a `cogp` metadata value against SPEC
/// §5.3 and §6. `pruning` is the bbox-statistics source the file offers,
/// None when it offers none.
///
/// Unknown fields are ignored on purpose (SPEC §6.1): a file written
/// against a later minor version has to stay readable here.
pub fn parse(
    json: &str,
    row_groups: usize,
    pruning: Option<Pruning>,
) -> Result<CogpLevels, String> {
    let raw = Cogp::parse(json)?;

    // §6.1: a major bump may redefine what the existing fields mean, so
    // an unsupported one is not "read it anyway" — it is "this is not a
    // profile this build can claim to implement".
    let major: u64 = raw
        .version
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("version '{}' is not MAJOR.MINOR.PATCH", raw.version))?;
    if major != SUPPORTED_MAJOR {
        return Err(format!(
            "version {}: this reader implements COGP {SUPPORTED_MAJOR}.x",
            raw.version
        ));
    }

    // §5.3 lives in `Cogp::validate`, and only there: the writer and the
    // reader must not be able to disagree about what conforms.
    raw.validate(row_groups)?;

    // The levels are only useful if a viewport can prune within a
    // prefix, which is why the spec requires bbox statistics as part of
    // the profile rather than as advice.
    let pruning = pruning.ok_or(
        "no row-group bbox statistics: neither a declared covering.bbox \
         nor native geospatial statistics",
    )?;

    Ok(CogpLevels {
        version: raw.version,
        levels: raw.levels,
        pruning,
    })
}

/// Inclusive row-group index range of each level, coarse to fine, from
/// the levels' `row_group_end` values.
///
/// None when the ends do not describe exactly `row_groups` groups — a
/// caller holding boxes that the levels do not account for has nothing
/// level-shaped to measure and must fall back to the whole file.
pub fn level_ranges(level_ends: &[usize], row_groups: usize) -> Option<Vec<(usize, usize)>> {
    if row_groups == 0 || level_ends.last() != Some(&(row_groups - 1)) {
        return None;
    }
    let mut out = Vec::with_capacity(level_ends.len());
    let mut start = 0usize;
    for &end in level_ends {
        if end < start {
            return None;
        }
        out.push((start, end));
        start = end + 1;
    }
    Some(out)
}

/// The `cogp` metadata object as written: the serialisable twin of
/// [`CogpLevels`], which is what the reader derives after validation.
///
/// Unknown fields are ignored on the way in, as §6.1 requires of readers:
/// a file written against a later minor version must still parse here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cogp {
    pub version: String,
    pub levels: Vec<Level>,
}

impl Cogp {
    /// Levels coarse to fine, tagged with the version this build writes.
    pub fn new(levels: Vec<Level>) -> Self {
        Self {
            version: VERSION.to_string(),
            levels,
        }
    }

    pub fn to_json(&self) -> String {
        // The struct has no unrepresentable shape (f64 gsd is finite by
        // construction — `validate` rejects anything else), so this cannot
        // fail; a literal keeps the signature infallible for callers.
        serde_json::to_string(self).unwrap_or_else(|_| r#"{"version":"0.1.0","levels":[]}"#.into())
    }

    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("cogp metadata is not valid JSON: {e}"))
    }

    /// Structural conformance (spec §5.3). Says nothing about whether the
    /// features in a level are genuinely meaningful at its `gsd` — §8 is
    /// explicit that no validator can check that.
    pub fn validate(&self, num_row_groups: usize) -> Result<(), String> {
        if self.levels.is_empty() {
            return Err("cogp: the levels list must not be empty".into());
        }
        let mut prev_end: Option<usize> = None;
        let mut prev_gsd: Option<f64> = None;
        for (i, l) in self.levels.iter().enumerate() {
            if l.row_group_end >= num_row_groups {
                return Err(format!(
                    "cogp: level {i} ends at row group {} but the file has {num_row_groups}",
                    l.row_group_end
                ));
            }
            if let Some(p) = prev_end
                && l.row_group_end <= p
            {
                return Err(format!(
                    "cogp: row_group_end must strictly increase ({p} then {})",
                    l.row_group_end
                ));
            }
            if !(l.gsd.is_finite() && l.gsd > 0.0) {
                return Err(format!("cogp: level {i} has a non-positive gsd {}", l.gsd));
            }
            if let Some(p) = prev_gsd
                && l.gsd >= p
            {
                return Err(format!("cogp: gsd must strictly decrease ({p} then {})", l.gsd));
            }
            prev_end = Some(l.row_group_end);
            prev_gsd = Some(l.gsd);
        }
        // §5.3: the levels must cover every row group in the file.
        if prev_end != Some(num_row_groups - 1) {
            return Err(format!(
                "cogp: last row_group_end is {:?}, expected {}",
                prev_end,
                num_row_groups - 1
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(levels: &str) -> String {
        format!(r#"{{"version":"0.1.0","levels":{levels}}}"#)
    }

    fn valid() -> CogpLevels {
        parse(
            &meta(
                r#"[{"row_group_end":0,"gsd":1000},
                    {"row_group_end":3,"gsd":250},
                    {"row_group_end":12,"gsd":60}]"#,
            ),
            13,
            Some(Pruning::Covering),
        )
        .unwrap()
    }

    #[test]
    fn a_conforming_block_parses() {
        let c = valid();
        assert_eq!(c.version, "0.1.0");
        assert_eq!(c.levels.len(), 3);
        assert_eq!(c.levels[1], Level { row_group_end: 3, gsd: 250.0 });
        assert_eq!(c.pruning, Pruning::Covering);
        assert!(c.summary().starts_with("COGP 0.1.0: 3 levels"), "{}", c.summary());
        assert!(c.summary().contains("gsd 1000/250/60 m"), "{}", c.summary());
        assert!(c.summary().ends_with("prefix row groups 1/4/13"), "{}", c.summary());
    }

    /// Unknown fields are the forward-compatibility contract (SPEC §6.1).
    #[test]
    fn unknown_fields_are_ignored() {
        let json = r#"{"version":"0.1.9","levels":[{"row_group_end":2,"gsd":10,
                       "features":99}],"tiling":"whatever"}"#;
        assert!(parse(json, 3, Some(Pruning::Covering)).is_ok());
    }

    #[test]
    fn a_newer_major_version_is_refused() {
        let json = r#"{"version":"1.0.0","levels":[{"row_group_end":0,"gsd":10}]}"#;
        let e = parse(json, 1, Some(Pruning::Covering)).unwrap_err();
        assert!(e.contains("COGP 0.x"), "{e}");
    }

    /// The structural rules of §5.3 are `Cogp::validate`'s, which is why
    /// these expect its wording: one implementation, one set of messages.
    #[test]
    fn structural_rules_are_enforced() {
        let cases: Vec<(&str, usize, &str)> = vec![
            (r#"[]"#, 1, "must not be empty"),
            // row_group_end must strictly increase…
            (
                r#"[{"row_group_end":3,"gsd":100},{"row_group_end":3,"gsd":50}]"#,
                4,
                "must strictly increase",
            ),
            // …gsd must strictly decrease…
            (
                r#"[{"row_group_end":1,"gsd":100},{"row_group_end":3,"gsd":100}]"#,
                4,
                "must strictly decrease",
            ),
            // …gsd must be positive…
            (r#"[{"row_group_end":0,"gsd":0}]"#, 1, "non-positive gsd"),
            // …and the last level must cover the file.
            (r#"[{"row_group_end":0,"gsd":100}]"#, 4, "last row_group_end"),
            (r#"[{"row_group_end":9,"gsd":100}]"#, 4, "but the file has"),
        ];
        for (levels, groups, want) in cases {
            let e = parse(&meta(levels), groups, Some(Pruning::Covering))
                .unwrap_err();
            assert!(e.contains(want), "{levels}: {e}");
        }
    }

    /// Levels with no way to prune inside them are not a usable profile.
    #[test]
    fn no_bbox_statistics_is_invalid() {
        let e = parse(&meta(r#"[{"row_group_end":0,"gsd":10}]"#), 1, None).unwrap_err();
        assert!(e.contains("bbox statistics"), "{e}");
    }

    /// Native 2.0 statistics stand in for the covering column, and the
    /// summary says so.
    #[test]
    fn native_statistics_are_labelled_as_an_extension() {
        let c = parse(&meta(r#"[{"row_group_end":0,"gsd":10}]"#), 1, Some(Pruning::NativeStats))
            .unwrap();
        assert!(c.summary().contains("(2.0 extension)"), "{}", c.summary());
    }

    /// SPEC §7.1: the last level whose gsd is still ≥ the target, and
    /// level 0 when the view is coarser than anything the file has.
    #[test]
    fn level_selection_follows_the_spec() {
        let c = valid();
        assert_eq!(c.level_for_gsd(5000.0), 0, "coarser than the file");
        assert_eq!(c.level_for_gsd(1000.0), 0, "exactly the coarsest gsd");
        assert_eq!(c.level_for_gsd(300.0), 0);
        assert_eq!(c.level_for_gsd(250.0), 1);
        assert_eq!(c.level_for_gsd(61.0), 1);
        assert_eq!(c.level_for_gsd(60.0), 2);
        assert_eq!(c.level_for_gsd(0.5), 2, "finer than the file");
        assert_eq!(c.row_group_end_for_gsd(300.0), 0);
        assert_eq!(c.row_group_end_for_gsd(60.0), 12);
    }

    #[test]
    fn level_ranges_partition_the_row_groups() {
        assert_eq!(
            level_ranges(&[0, 3, 12], 13),
            Some(vec![(0, 0), (1, 3), (4, 12)])
        );
        // Ends that do not account for the file describe nothing usable.
        assert_eq!(level_ranges(&[0, 3], 13), None);
        assert_eq!(level_ranges(&[0, 3, 12], 0), None);
    }

    /// Each level's run owns only its own row groups: the coarsest
    /// level's is the prefix a first paint costs.
    #[test]
    fn runs_count_each_level_alone() {
        let c = valid();
        let rg_rows: Vec<u64> = vec![100; 13];
        let runs = c.runs(&rg_rows);
        assert_eq!(runs.iter().map(|r| r.rows).collect::<Vec<_>>(), vec![100, 300, 900]);
        assert_eq!(runs.iter().map(|r| r.row_group_end).collect::<Vec<_>>(), vec![0, 3, 12]);
    }

    // --- writer-side type ---

    fn lv(row_group_end: usize, gsd: f64) -> Level {
        Level { row_group_end, gsd }
    }

    #[test]
    fn spec_example_round_trips() {
        let json = r#"{"version":"0.1.0","levels":[
            {"row_group_end":0,"gsd":1000},
            {"row_group_end":3,"gsd":500},
            {"row_group_end":12,"gsd":100}]}"#;
        let c = Cogp::parse(json).unwrap();
        assert_eq!(c.levels.len(), 3);
        c.validate(13).unwrap();
        // And back out through our own serializer unchanged.
        assert_eq!(Cogp::parse(&c.to_json()).unwrap(), c);
    }

    #[test]
    fn cogp_type_unknown_fields_are_ignored() {
        let json = r#"{"version":"0.2.0","levels":[{"row_group_end":0,"gsd":10,"extra":7}],
                       "future":"whatever"}"#;
        Cogp::parse(json).unwrap().validate(1).unwrap();
    }

    #[test]
    fn cogp_type_structural_rules_are_enforced() {
        assert!(Cogp::new(vec![]).validate(1).is_err());
        // row_group_end must strictly increase…
        assert!(Cogp::new(vec![lv(2, 10.0), lv(2, 5.0)]).validate(3).is_err());
        // …gsd must strictly decrease…
        assert!(Cogp::new(vec![lv(0, 10.0), lv(1, 10.0)]).validate(2).is_err());
        // …and stay positive…
        assert!(Cogp::new(vec![lv(0, 0.0)]).validate(1).is_err());
        // …and the last level must own the last row group.
        assert!(Cogp::new(vec![lv(0, 10.0)]).validate(3).is_err());
        assert!(Cogp::new(vec![lv(0, 10.0), lv(2, 5.0)]).validate(3).is_ok());
    }
}
