//! The H3 pyramid descriptor: `h3-pyramid.json` at the root of a layer
//! published as `r<res>/<cell>.parquet` files, plus the per-file
//! `geopq:pyramid` key an overview file carries so it says what it is
//! when opened alone.
//!
//! The pyramid gives vector data what a COG gets from its overviews: a
//! leaf level holding the source features partitioned by H3 cell, and
//! coarser levels holding derived features (simplified, pruned or
//! dissolved) a reader picks by zoom. Discovery needs no manifest beyond
//! this file: viewport -> ground sample distance -> resolution ->
//! `polygon_to_cells` -> file names; the cell lists here only spare the
//! reader from probing for files that do not exist.
//!
//! Shared by the writer (`optimize.rs`) and the reader (`loader.rs`) so
//! there is exactly one definition of the layout. Design notes:
//! `_WIKI/concepts/h3-pyramid.md`.

use h3o::{CellIndex, Resolution};
use serde::{Deserialize, Serialize};

/// Root descriptor file name.
pub const DESCRIPTOR: &str = "h3-pyramid.json";
/// Descriptor version this build writes.
pub const VERSION: &str = "0.1.0";
/// Parquet key-value entry an overview file carries.
pub const FILE_KEY: &str = "geopq:pyramid";
/// Default pixels per cell edge used to pick a level for a viewport.
pub const DEFAULT_PIXELS_PER_CELL: f64 = 64.0;
/// Name of the leaf part holding null-geometry rows.
pub const NULL_PART: &str = "__HIVE_DEFAULT_PARTITION__";

/// How an overview level was derived from the next finer one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Same features, fewer vertices (Douglas-Peucker at the level's tolerance).
    Simplify,
    /// Fewer features: largest first, or one in x.
    Prune,
    /// Union of the features by child cell, with summary attributes.
    Dissolve,
}

impl Method {
    pub fn label(self) -> &'static str {
        match self {
            Method::Simplify => "simplify",
            Method::Prune => "prune",
            Method::Dissolve => "dissolve",
        }
    }
}

/// One level of the pyramid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    pub res: u8,
    /// None on the leaf level (source features), Some on overviews.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<Method>,
    /// Cells present at this level, as H3 index hex strings. The leaf
    /// level lists adaptive children at their own resolution under
    /// their own `Level` entry, so every entry here is at `res`.
    #[serde(default)]
    pub cells: Vec<String>,
    /// Rows across the level's files, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
}

/// Leaf-level layout parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Leaf {
    /// Reference resolution: the coarsest resolution holding source features.
    pub res: u8,
    /// Finest resolution adaptive splitting may reach; equals `res` when off.
    pub adaptive_max_res: u8,
    /// Row target a cell must exceed to be split.
    pub target_rows: u64,
    /// Whether a `r<res>/__HIVE_DEFAULT_PARTITION__.parquet` part exists.
    #[serde(default)]
    pub null_part: bool,
}

/// `h3-pyramid.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Descriptor {
    pub version: String,
    pub leaf: Leaf,
    /// Coarse to fine, overviews first, then the leaf resolutions
    /// (reference res and any adaptive children resolutions).
    pub levels: Vec<Level>,
    /// Pixels per cell edge readers use to pick a level.
    #[serde(default = "default_ppc")]
    pub pixels_per_cell: f64,
    /// PROJJSON of the source CRS, or null when undefined.
    #[serde(default)]
    pub crs: serde_json::Value,
    /// Data extent in the source CRS, [xmin, ymin, xmax, ymax].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
    /// Source rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    /// Free-form method parameters as written (tolerances, keep
    /// fractions, dissolve attributes), for humans and for re-runs.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub methods: serde_json::Value,
}

fn default_ppc() -> f64 {
    DEFAULT_PIXELS_PER_CELL
}

/// The `geopq:pyramid` entry of one overview file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FileMeta {
    pub res: u8,
    pub method: Method,
    pub source_res: u8,
    pub derived: bool,
}

/// Relative path of a level's file for a cell.
pub fn part_path(res: u8, cell: &str) -> String {
    format!("r{res}/{cell}.parquet")
}

/// Metres of ground one pixel covers at which a level's cells are drawn
/// `pixels_per_cell` wide: `edge_length(res) / pixels_per_cell`.
pub fn gsd_for_res(res: Resolution, pixels_per_cell: f64) -> f64 {
    res.edge_length_m() / pixels_per_cell
}

impl Descriptor {
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("{DESCRIPTOR} is not valid JSON: {e}"))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Structural checks: version major 0, resolutions in range and
    /// strictly increasing, leaf entries carry no method and overview
    /// entries carry one, the reference resolution is listed, adaptive
    /// range sane, cell ids parse at their level's resolution.
    pub fn validate(&self) -> Result<(), String> {
        let major = self.version.split('.').next().and_then(|m| m.parse::<u64>().ok());
        if major != Some(0) {
            return Err(format!("pyramid: unsupported version {}", self.version));
        }
        if self.levels.is_empty() {
            return Err("pyramid: the levels list must not be empty".into());
        }
        if self.leaf.adaptive_max_res < self.leaf.res || self.leaf.adaptive_max_res > 15 {
            return Err(format!(
                "pyramid: adaptive_max_res {} must lie in {}..=15",
                self.leaf.adaptive_max_res, self.leaf.res
            ));
        }
        if !(self.pixels_per_cell.is_finite() && self.pixels_per_cell > 0.0) {
            return Err("pyramid: pixels_per_cell must be positive".into());
        }
        let mut prev: Option<u8> = None;
        let mut has_reference = false;
        for l in &self.levels {
            if l.res > 15 {
                return Err(format!("pyramid: resolution {} out of range", l.res));
            }
            if let Some(p) = prev
                && l.res <= p
            {
                return Err(format!("pyramid: resolutions must strictly increase ({p} then {})", l.res));
            }
            let is_leaf = l.res >= self.leaf.res;
            match (is_leaf, l.method) {
                (true, Some(m)) => {
                    return Err(format!(
                        "pyramid: leaf level r{} must not declare a method ({})",
                        l.res,
                        m.label()
                    ));
                }
                (false, None) => {
                    return Err(format!("pyramid: overview level r{} declares no method", l.res));
                }
                _ => {}
            }
            if is_leaf && l.res > self.leaf.adaptive_max_res {
                return Err(format!(
                    "pyramid: level r{} is finer than adaptive_max_res {}",
                    l.res, self.leaf.adaptive_max_res
                ));
            }
            has_reference |= l.res == self.leaf.res;
            let want = Resolution::try_from(l.res).map_err(|e| e.to_string())?;
            for c in &l.cells {
                let idx: CellIndex = c
                    .parse()
                    .map_err(|e| format!("pyramid: bad cell id {c:?} at r{}: {e}", l.res))?;
                if idx.resolution() != want {
                    return Err(format!(
                        "pyramid: cell {c} is r{} but listed at r{}",
                        u8::from(idx.resolution()),
                        l.res
                    ));
                }
            }
            prev = Some(l.res);
        }
        if !has_reference {
            return Err(format!("pyramid: reference resolution r{} is not listed", self.leaf.res));
        }
        Ok(())
    }

    /// The level a viewport at `gsd` metres per pixel should read: the
    /// finest level whose cells are still at least `pixels_per_cell`
    /// wide on screen, i.e. `gsd_for_res(res) >= gsd`; a viewport
    /// coarser than the coarsest level gets the coarsest one.
    ///
    /// Only one resolution per pyramid "band" is returned: overview
    /// levels are single resolutions, and the leaf band (reference res
    /// plus adaptive children) is reported as the reference res, since
    /// a leaf read must consider every leaf resolution for the cells it
    /// touches.
    pub fn res_for_gsd(&self, gsd: f64) -> u8 {
        let mut chosen = self.levels[0].res;
        for l in &self.levels {
            let res = l.res.min(self.leaf.res);
            let Ok(r) = Resolution::try_from(res) else { continue };
            if gsd_for_res(r, self.pixels_per_cell) >= gsd {
                chosen = res;
            } else {
                break;
            }
        }
        chosen.min(self.leaf.res)
    }

    /// Overview levels, coarse to fine.
    pub fn overviews(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter().filter(|l| l.res < self.leaf.res)
    }

    /// Leaf levels (reference res and adaptive children), coarse to fine.
    pub fn leaves(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter().filter(|l| l.res >= self.leaf.res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> Descriptor {
        Descriptor {
            version: VERSION.into(),
            leaf: Leaf { res: 8, adaptive_max_res: 9, target_rows: 250_000, null_part: true },
            levels: vec![
                Level { res: 5, method: Some(Method::Dissolve), cells: vec!["852a3067fffffff".into()], rows: None },
                Level { res: 6, method: Some(Method::Dissolve), cells: vec![], rows: None },
                Level { res: 7, method: Some(Method::Dissolve), cells: vec![], rows: None },
                Level { res: 8, method: None, cells: vec![], rows: Some(2_500_000) },
                Level { res: 9, method: None, cells: vec![], rows: Some(57_399) },
            ],
            pixels_per_cell: DEFAULT_PIXELS_PER_CELL,
            crs: serde_json::Value::Null,
            bbox: Some([0.0, 0.0, 1.0, 1.0]),
            rows: Some(2_557_399),
            methods: serde_json::Value::Null,
        }
    }

    #[test]
    fn descriptor_round_trips_and_validates() {
        let d = desc();
        d.validate().unwrap();
        let back = Descriptor::parse(&d.to_json()).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn unknown_fields_are_ignored_and_defaults_fill_in() {
        let json = r#"{"version":"0.2.0","leaf":{"res":8,"adaptive_max_res":8,"target_rows":1},
            "levels":[{"res":8}],"future":1}"#;
        let d = Descriptor::parse(json).unwrap();
        d.validate().unwrap();
        assert_eq!(d.pixels_per_cell, DEFAULT_PIXELS_PER_CELL);
        assert!(!d.leaf.null_part);
    }

    #[test]
    fn structural_rules_are_enforced() {
        let mut d = desc();
        d.version = "1.0.0".into();
        assert!(d.validate().is_err());

        let mut d = desc();
        d.levels[3].method = Some(Method::Prune); // leaf with a method
        assert!(d.validate().is_err());

        let mut d = desc();
        d.levels[0].method = None; // overview without a method
        assert!(d.validate().is_err());

        let mut d = desc();
        d.levels.swap(0, 1); // not increasing
        assert!(d.validate().is_err());

        let mut d = desc();
        d.levels[0].cells = vec!["862a30667ffffff".into()]; // an r6 cell listed at r5
        assert!(d.validate().is_err());

        let mut d = desc();
        d.levels.retain(|l| l.res != 8); // reference res missing
        assert!(d.validate().is_err());

        let mut d = desc();
        d.leaf.adaptive_max_res = 7;
        assert!(d.validate().is_err());
    }

    #[test]
    fn level_choice_follows_pixels_per_cell() {
        let d = desc();
        // r8 edge ~461 m: at 64 px per cell that is ~7.2 m/px; anything
        // finer than that reads the leaf.
        assert_eq!(d.res_for_gsd(1.0), 8);
        assert_eq!(d.res_for_gsd(7.0), 8);
        // r7 edge ~1.2 km -> ~19 m/px; r6 ~3.2 km -> ~50 m/px; r5 ~8.5 km -> ~133 m/px.
        assert_eq!(d.res_for_gsd(15.0), 7);
        assert_eq!(d.res_for_gsd(40.0), 6);
        assert_eq!(d.res_for_gsd(100.0), 5);
        // Coarser than the coarsest level still gets the coarsest level.
        assert_eq!(d.res_for_gsd(10_000.0), 5);
        // Adaptive children never come back as a level choice.
        assert_eq!(d.res_for_gsd(0.01), 8);
    }

    #[test]
    fn paths_and_file_meta() {
        assert_eq!(part_path(6, "862a30667ffffff"), "r6/862a30667ffffff.parquet");
        let m = FileMeta { res: 6, method: Method::Dissolve, source_res: 7, derived: true };
        let back: FileMeta = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
    }
}
