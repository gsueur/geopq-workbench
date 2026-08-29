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

use serde_json::Value;

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    pub row_group_end: usize,
    /// Metres of ground per pixel, whatever the CRS's own units.
    pub gsd: f64,
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

    /// Rows in the coarsest level's prefix.
    pub fn level0_rows(&self, rg_rows: &[u64]) -> u64 {
        let end = self.levels[0].row_group_end;
        rg_rows.iter().take(end + 1).sum()
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
    let v: Value = serde_json::from_str(json).map_err(|e| format!("not JSON: {e}"))?;
    let obj = v.as_object().ok_or("not a JSON object")?;

    let version = obj
        .get("version")
        .and_then(Value::as_str)
        .ok_or("no `version` string")?
        .to_string();
    let major: u64 = version
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("version '{version}' is not MAJOR.MINOR.PATCH"))?;
    if major != SUPPORTED_MAJOR {
        return Err(format!(
            "version {version}: this reader implements COGP {SUPPORTED_MAJOR}.x"
        ));
    }

    let raw = obj
        .get("levels")
        .and_then(Value::as_array)
        .ok_or("no `levels` array")?;
    if raw.is_empty() {
        return Err("`levels` is empty".into());
    }
    let mut levels: Vec<Level> = Vec::with_capacity(raw.len());
    for (i, l) in raw.iter().enumerate() {
        let end = l
            .get("row_group_end")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("level {i}: `row_group_end` is not a non-negative integer"))?
            as usize;
        let gsd = l
            .get("gsd")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("level {i}: `gsd` is not a number"))?;
        if !(gsd.is_finite() && gsd > 0.0) {
            return Err(format!("level {i}: gsd {gsd} is not positive"));
        }
        if end >= row_groups {
            return Err(format!(
                "level {i}: row_group_end {end} is outside the file's {row_groups} row groups"
            ));
        }
        if let Some(prev) = levels.last() {
            if end <= prev.row_group_end {
                return Err(format!(
                    "level {i}: row_group_end {end} does not increase on {}",
                    prev.row_group_end
                ));
            }
            if gsd >= prev.gsd {
                return Err(format!(
                    "level {i}: gsd {gsd} does not decrease from {}",
                    prev.gsd
                ));
            }
        }
        levels.push(Level { row_group_end: end, gsd });
    }
    let last = levels.last().unwrap().row_group_end;
    if last + 1 != row_groups {
        return Err(format!(
            "the last level ends at row group {last}, not the file's last ({})",
            row_groups.saturating_sub(1)
        ));
    }

    // The levels are only useful if a viewport can prune within a
    // prefix, which is why the spec requires bbox statistics as part of
    // the profile rather than as advice.
    let pruning = pruning.ok_or(
        "no row-group bbox statistics: neither a declared covering.bbox \
         nor native geospatial statistics",
    )?;

    Ok(CogpLevels { version, levels, pruning })
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

    #[test]
    fn structural_rules_are_enforced() {
        let cases: Vec<(&str, usize, &str)> = vec![
            (r#"[]"#, 1, "empty"),
            // row_group_end must strictly increase…
            (
                r#"[{"row_group_end":3,"gsd":100},{"row_group_end":3,"gsd":50}]"#,
                4,
                "does not increase",
            ),
            // …gsd must strictly decrease…
            (
                r#"[{"row_group_end":1,"gsd":100},{"row_group_end":3,"gsd":100}]"#,
                4,
                "does not decrease",
            ),
            // …gsd must be positive…
            (r#"[{"row_group_end":0,"gsd":0}]"#, 1, "not positive"),
            // …and the last level must cover the file.
            (r#"[{"row_group_end":0,"gsd":100}]"#, 4, "last level ends"),
            (r#"[{"row_group_end":9,"gsd":100}]"#, 4, "outside the file"),
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
    fn level0_rows_counts_the_prefix() {
        let c = valid();
        let rg_rows: Vec<u64> = vec![100; 13];
        assert_eq!(c.level0_rows(&rg_rows), 100);
    }
}
