//! The Cloud Optimized GeoParquet Profile (COGP) file metadata key.
//!
//! COGP v0.1 (<https://github.com/Kanahiro/cloud-optimized-geoparquet>) is a
//! layout convention on top of GeoParquet: features are ordered coarse to
//! fine, every level ends on a row-group boundary, and one file-level
//! key-value entry names those boundaries. This module is only that entry —
//! the struct, its JSON form, and the structural checks of spec §5.3 — so
//! that the writer and any reader agree on one definition of the key
//! instead of two hand-rolled ones that drift.

use serde::{Deserialize, Serialize};

/// Parquet file-level key-value metadata key (spec §6).
pub const KEY: &str = "cogp";

/// Profile version this build writes.
pub const VERSION: &str = "0.1.0";

/// One rendering detail level: the last row group it owns, and the ground
/// sample distance below which features are deferred to a finer level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    /// Inclusive, zero-based index of the row group ending this level.
    pub row_group_end: usize,
    /// Approximate smallest independently meaningful ground distance, in
    /// metres — always metres, whatever units the CRS measures in.
    pub gsd: f64,
}

/// The `cogp` metadata object.
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
    fn unknown_fields_are_ignored() {
        let json = r#"{"version":"0.2.0","levels":[{"row_group_end":0,"gsd":10,"extra":7}],
                       "future":"whatever"}"#;
        Cogp::parse(json).unwrap().validate(1).unwrap();
    }

    #[test]
    fn structural_rules_are_enforced() {
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
