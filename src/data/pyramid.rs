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
/// Virtual partition column a reader exposes for a part's cell id. The
/// files are not hive-named (`r8/<cell>.parquet`, not `h3=<cell>/…`),
/// but the column they stand for is the same one the adaptive-H3 writer
/// spells in paths, so it keeps the name SQL already knows.
pub const CELL_COLUMN: &str = "h3";

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

    /// Every part path the descriptor names, coarse to fine and in the
    /// cell order each level lists, with the leaf null part last.
    ///
    /// Discovery does not need this — a reader derives file names from
    /// cell ids — but a caller that wants the whole dataset (a copy, an
    /// upload, a completeness check) would otherwise have to re-implement
    /// the layout, and there is exactly one place the layout is defined.
    pub fn files(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .levels
            .iter()
            .flat_map(|l| l.cells.iter().map(|c| part_path(l.res, c)))
            .collect();
        if self.leaf.null_part {
            out.push(part_path(self.leaf.res, NULL_PART));
        }
        out
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

// ---------------------------------------------------------------------
// Reader side: what an opened pyramid keeps, and how one viewport turns
// into the exact set of part files to read.
// ---------------------------------------------------------------------

/// Cells one viewport pass may enumerate before the descriptor's own
/// lists get a chance to cut them down. A whole-world rect at r10 covers
/// 33 billion cells and the tiler would spend the machine proving it.
/// Over this the planner reads the level's listed cells instead, which is
/// a superset of what the viewport wants and is bounded by the file.
pub const MAX_VIEWPORT_CELLS: usize = 100_000;

/// Part files one level of a pyramid layer may open for a viewport.
/// The cell list is exact — no probing for files that do not exist — so
/// this is a build-cost ceiling, not the STAC guess `STAC_PART_CAP` is:
/// past it the planner takes the next coarser level, which covers the
/// same ground in fewer, cheaper files.
pub const MAX_LEVEL_PARTS: usize = 256;

/// One level's cells, parsed once at open.
#[derive(Clone, Debug)]
pub struct LevelCells {
    pub res: u8,
    /// None on the leaf band, Some on overviews.
    pub method: Option<Method>,
    pub cells: std::collections::HashSet<CellIndex>,
}

/// The parts one viewport needs at one level: never a mix of levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelPlan {
    /// Level read, as a resolution. On the leaf band this is the
    /// reference resolution even when adaptive children are included.
    pub res: u8,
    /// Relative paths (`r<res>/<cell>.parquet`), coarse res first then
    /// cell id, so a plan is reproducible and diffable.
    pub parts: Vec<String>,
    /// The level the viewport's ground scale asked for, when the part
    /// count made the planner settle for a coarser one.
    pub coarsened_from: Option<u8>,
}

/// An opened pyramid: the descriptor, the root its part names hang off,
/// the per-level cell sets, and which level the layer is reading.
#[derive(Clone, Debug)]
pub struct PyramidState {
    pub descriptor: Descriptor,
    /// Display form of the root the `r<res>/<cell>.parquet` names are
    /// relative to (a directory path, an `s3://` prefix, or a URL).
    pub root: String,
    /// Resolution of the level currently open. Levels are never mixed,
    /// so this is one value, not a set.
    pub active_res: u8,
    levels: Vec<LevelCells>,
    /// Files the descriptor lists that the root does not hold, from the
    /// one listing taken at open. A plan never names them, so a gap in
    /// the tree costs the cells that are missing and nothing else — and
    /// re-planning the same viewport keeps giving the same answer
    /// instead of chasing files that will not appear.
    absent: std::collections::HashSet<String>,
}

impl PyramidState {
    /// Validate a descriptor and parse its cell lists. The active level
    /// starts at the reference resolution; `plan` moves it.
    pub fn new(descriptor: Descriptor, root: impl Into<String>) -> Result<Self, String> {
        descriptor.validate()?;
        let levels = descriptor
            .levels
            .iter()
            .map(|l| {
                let cells = l
                    .cells
                    .iter()
                    .map(|c| c.parse::<CellIndex>().map_err(|e| format!("pyramid: bad cell id {c:?}: {e}")))
                    .collect::<Result<_, _>>()?;
                Ok(LevelCells { res: l.res, method: l.method, cells })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let active_res = descriptor.leaf.res;
        Ok(Self {
            descriptor,
            root: root.into(),
            active_res,
            levels,
            absent: std::collections::HashSet::new(),
        })
    }

    /// Record which of the descriptor's files the root does not hold.
    /// `have` is the root's listing; None means it serves none, in which
    /// case the descriptor is taken at its word.
    pub fn mark_absent(&mut self, have: Option<&std::collections::HashSet<String>>) {
        self.absent = match have {
            Some(have) => self.all_parts().into_iter().filter(|p| !have.contains(p)).collect(),
            None => std::collections::HashSet::new(),
        };
    }

    /// Files the descriptor lists that are not there.
    pub fn absent(&self) -> &std::collections::HashSet<String> {
        &self.absent
    }

    /// Resolutions of the levels a `res_for_gsd` answer may name, coarse
    /// to fine: every overview, then the reference resolution. Adaptive
    /// child resolutions are not levels of their own — they are part of
    /// the leaf band.
    pub fn bands(&self) -> Vec<u8> {
        let mut out: Vec<u8> = self.descriptor.overviews().map(|l| l.res).collect();
        out.push(self.descriptor.leaf.res);
        out
    }

    /// The overview method of the level currently open, or None on the
    /// leaf band (where the features are the source's own).
    pub fn active_method(&self) -> Option<Method> {
        self.levels
            .iter()
            .find(|l| l.res == self.active_res)
            .and_then(|l| l.method)
    }

    /// An overview is derived data, so a layer showing one must say so.
    pub fn is_overview(&self) -> bool {
        self.active_res < self.descriptor.leaf.res
    }

    /// Layer badge while an overview is on screen; None on the leaf,
    /// where nothing is approximated (OPEN_POLICY invariant 1).
    pub fn badge(&self) -> Option<String> {
        let method = self.active_method()?;
        self.is_overview()
            .then(|| format!("overview r{} ({})", self.active_res, method.label()))
    }

    /// The info-panel line: what this pyramid is, in one sentence.
    pub fn info_line(&self) -> String {
        let leaf = &self.descriptor.leaf;
        let mut s = format!("leaf r{}", leaf.res);
        if leaf.adaptive_max_res > leaf.res {
            s.push_str(&format!(" (adaptive to r{})", leaf.adaptive_max_res));
        }
        let ov: Vec<&Level> = self.descriptor.overviews().collect();
        match (ov.first(), ov.last()) {
            (Some(first), Some(last)) => {
                let mut methods: Vec<&str> =
                    ov.iter().filter_map(|l| l.method).map(Method::label).collect();
                methods.dedup();
                let range = if first.res == last.res {
                    format!("r{}", first.res)
                } else {
                    format!("r{}..r{}", first.res, last.res)
                };
                s.push_str(&format!(", overviews {range} ({})", methods.join("+")));
            }
            _ => s.push_str(", no overviews"),
        }
        s.push_str(&format!(", {} px/cell", fmt_ppc(self.descriptor.pixels_per_cell)));
        s
    }

    /// The null-geometry part, when the leaf level has one. It draws
    /// nothing, so viewport plans leave it out; SQL and the scorecard
    /// still count it as a leaf file.
    pub fn null_part(&self) -> Option<String> {
        self.descriptor
            .leaf
            .null_part
            .then(|| part_path(self.descriptor.leaf.res, NULL_PART))
    }

    /// Every leaf file the descriptor lists: the reference resolution,
    /// the adaptive children, and the null part.
    pub fn leaf_parts(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .descriptor
            .leaves()
            .flat_map(|l| l.cells.iter().map(move |c| part_path(l.res, &c.to_string())))
            .collect();
        out.extend(self.null_part());
        out
    }

    /// Every file the descriptor lists, all levels.
    pub fn all_parts(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .descriptor
            .levels
            .iter()
            .flat_map(|l| l.cells.iter().map(move |c| part_path(l.res, &c.to_string())))
            .collect();
        out.extend(self.null_part());
        out
    }

    /// Parts of one level covering `rect` (WGS84 lon/lat), or of the
    /// whole level when `rect` is None.
    ///
    /// On the leaf band this is the reference-resolution cells the
    /// viewport wants plus any adaptive children the descriptor lists at
    /// a finer resolution whose reference-resolution ancestor is one of
    /// them — the writer replaces a dense cell by its children, so the
    /// parent it stands for need not be listed at all.
    pub fn parts_for(&self, res: u8, rect: Option<[f64; 4]>) -> Vec<String> {
        let leaf_band = res >= self.descriptor.leaf.res;
        let want = rect.and_then(|r| cells_for_rect(res, r));
        let keep = |c: CellIndex, at: u8| -> bool {
            let Some(want) = &want else { return true };
            if at == res {
                return want.contains(&c);
            }
            // An adaptive child belongs to the viewport exactly when the
            // reference-resolution cell it was split out of does.
            Resolution::try_from(res)
                .ok()
                .and_then(|r| c.parent(r))
                .is_some_and(|p| want.contains(&p))
        };
        let mut out: Vec<String> = self
            .levels
            .iter()
            .filter(|l| if leaf_band { l.res >= res } else { l.res == res })
            .flat_map(|l| {
                l.cells
                    .iter()
                    .filter(|c| keep(**c, l.res))
                    .map(|c| part_path(l.res, &c.to_string()))
                    .filter(|p| !self.absent.contains(p))
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Which level to read for a viewport, and its parts.
    ///
    /// `gsd` (metres per pixel at the view centre) picks the level; the
    /// part count then decides whether it is affordable. Because the
    /// cell list is exact, a pyramid opens every cell the viewport needs
    /// rather than the best `max_parts` of them the way a STAC
    /// collection does — but a level over the cap is the wrong level,
    /// not a truncated one, so the planner steps to the next coarser
    /// one, which covers the same ground in fewer files.
    pub fn plan(&self, gsd: Option<f64>, rect: Option<[f64; 4]>, max_parts: usize) -> LevelPlan {
        let bands = self.bands();
        let want = gsd.map_or(self.descriptor.leaf.res, |g| self.descriptor.res_for_gsd(g));
        let start = bands.iter().position(|&r| r == want).unwrap_or(bands.len() - 1);
        let mut plan = LevelPlan { res: want, parts: Vec::new(), coarsened_from: None };
        for (i, &res) in bands[..=start].iter().enumerate().rev() {
            let parts = self.parts_for(res, rect);
            let fits = parts.len() <= max_parts || i == 0;
            plan = LevelPlan {
                res,
                parts,
                coarsened_from: (res != want).then_some(want),
            };
            if fits {
                break;
            }
        }
        plan
    }

    /// The same state reading a different level.
    pub fn with_active(&self, res: u8) -> Self {
        let mut out = self.clone();
        out.active_res = res;
        out
    }
}

/// What a layer must say about the pyramid content it is showing:
/// the active level when the layer was opened from a descriptor, or
/// what a single file says about itself when it was not. None on the
/// leaf level and on ordinary files, where nothing is derived.
pub fn layer_badge(state: Option<&PyramidState>, file: Option<&FileMeta>) -> Option<String> {
    if let Some(state) = state {
        return state.badge();
    }
    let m = file.filter(|m| m.derived)?;
    Some(format!("overview r{} ({})", m.res, m.method.label()))
}

/// Does a part's cell answer to `h3 = <filter>`?
///
/// Equality, plus the one relation the layout adds: an adaptive child
/// lives inside the reference-resolution cell it was split out of, so a
/// query naming that cell means the children too. Anything that does not
/// parse as a cell simply does not match.
pub fn cell_matches(part_cell: &str, filter: &str) -> bool {
    if part_cell == filter {
        return true;
    }
    let (Ok(part), Ok(want)) = (part_cell.parse::<CellIndex>(), filter.parse::<CellIndex>()) else {
        return false;
    };
    part.parent(want.resolution()) == Some(want)
}

/// `64` rather than `64.0`, so the info line reads like a setting.
fn fmt_ppc(ppc: f64) -> String {
    if (ppc - ppc.round()).abs() < 1e-9 {
        format!("{}", ppc.round() as i64)
    } else {
        format!("{ppc:.1}")
    }
}

/// Cells of resolution `res` covering a WGS84 lon/lat rect, padded by
/// one ring so a cell whose centre sits just outside the view but whose
/// area reaches into it still comes along.
///
/// None when the rect is unusable (empty, or off the globe) or when it
/// would enumerate more than [`MAX_VIEWPORT_CELLS`] cells — a view that
/// wide has no business being answered cell by cell, and the caller
/// reads the level's listed cells instead.
pub fn cells_for_rect(res: u8, rect: [f64; 4]) -> Option<std::collections::HashSet<CellIndex>> {
    use h3o::geom::{ContainmentMode, TilerBuilder};

    let res = Resolution::try_from(res).ok()?;
    let x0 = rect[0].max(-180.0);
    let y0 = rect[1].max(-90.0);
    let x1 = rect[2].min(180.0);
    let y1 = rect[3].min(90.0);
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) || x1 < x0 || y1 < y0
    {
        return None;
    }
    let ring = geo::LineString::from(vec![
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (x0, y0),
    ]);
    let mut tiler = TilerBuilder::new(res)
        // Covers, not ContainsCentroid: a cell overlapping the view has
        // features in it whether or not its centre is on screen, and a
        // view smaller than one cell must still name that cell.
        .containment_mode(ContainmentMode::Covers)
        .build();
    tiler.add(geo::Polygon::new(ring, Vec::new())).ok()?;
    if tiler.coverage_size_hint() > MAX_VIEWPORT_CELLS {
        return None;
    }
    let mut out: std::collections::HashSet<CellIndex> = std::collections::HashSet::new();
    for cell in tiler.into_coverage() {
        out.extend(cell.grid_disk::<Vec<_>>(1));
        if out.len() > MAX_VIEWPORT_CELLS {
            return None;
        }
    }
    Some(out)
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


    /// A small pyramid around one point: r5 and r6 dissolve overviews, an
    /// r7 leaf whose central cell the writer split into r8 children.
    /// Real cell ids throughout, so parent/child relations hold.
    fn paris() -> (PyramidState, CellIndex, Vec<CellIndex>) {
        let centre = h3o::LatLng::new(48.85, 2.35)
            .unwrap()
            .to_cell(Resolution::Seven);
        let disk: Vec<CellIndex> = centre.grid_disk(1);
        // The writer replaces one dense leaf cell by its children: the
        // parent is then absent from r7 and present as r8 cells.
        let split = disk[1];
        let children: Vec<CellIndex> = split.children(Resolution::Eight).collect();
        let leaf: Vec<CellIndex> = disk.iter().copied().filter(|c| *c != split).collect();
        let up = |res: Resolution| -> Vec<String> {
            let mut v: Vec<String> = disk
                .iter()
                .filter_map(|c| c.parent(res))
                .map(|c| c.to_string())
                .collect();
            v.sort();
            v.dedup();
            v
        };
        let level = |res: u8, method: Option<Method>, cells: Vec<String>| Level {
            res,
            method,
            cells,
            rows: None,
        };
        let d = Descriptor {
            version: VERSION.into(),
            leaf: Leaf { res: 7, adaptive_max_res: 8, target_rows: 1000, null_part: true },
            levels: vec![
                level(5, Some(Method::Dissolve), up(Resolution::Five)),
                level(6, Some(Method::Dissolve), up(Resolution::Six)),
                level(7, None, leaf.iter().map(CellIndex::to_string).collect()),
                level(8, None, children.iter().map(CellIndex::to_string).collect()),
            ],
            pixels_per_cell: DEFAULT_PIXELS_PER_CELL,
            crs: serde_json::Value::Null,
            bbox: None,
            rows: None,
            methods: serde_json::Value::Null,
        };
        (PyramidState::new(d, "/tmp/pyr").unwrap(), split, children)
    }

    /// A rect names the cells it touches, not only the ones centred in
    /// it: a viewport smaller than one cell still has to read that cell.
    #[test]
    fn a_viewport_names_the_cells_it_touches() {
        let centre = h3o::LatLng::new(48.85, 2.35)
            .unwrap()
            .to_cell(Resolution::Seven);
        let cells = cells_for_rect(7, [2.349, 48.849, 2.351, 48.851]).unwrap();
        assert!(cells.contains(&centre), "the cell under the viewport");
        // The one-ring pad: a cell whose centre is off screen but whose
        // area reaches in comes along.
        for n in centre.grid_disk::<Vec<_>>(1) {
            assert!(cells.contains(&n), "{n} is one ring out");
        }
        // Somewhere else entirely.
        assert!(!cells_for_rect(7, [-70.0, 45.0, -69.99, 45.01])
            .unwrap()
            .contains(&centre));
    }

    /// The leaf band is the reference resolution plus the children the
    /// writer split out of it, and nothing else.
    #[test]
    fn the_leaf_band_carries_its_adaptive_children() {
        let (p, split, children) = paris();
        let rect = [2.349, 48.849, 2.351, 48.851];
        let parts = p.parts_for(7, Some(rect));
        assert!(parts.iter().all(|s| s.ends_with(".parquet")));
        // Six r7 cells (the disk minus the split one) and the split
        // cell's seven r8 children.
        assert_eq!(parts.iter().filter(|s| s.starts_with("r7/")).count(), 6);
        assert_eq!(parts.iter().filter(|s| s.starts_with("r8/")).count(), children.len());
        assert!(!parts.contains(&part_path(7, &split.to_string())), "the split cell has no file");
        for c in &children {
            assert!(parts.contains(&part_path(8, &c.to_string())), "{c} is wanted through its parent");
        }
        // Sorted, so a plan is reproducible.
        let mut sorted = parts.clone();
        sorted.sort();
        assert_eq!(parts, sorted);
    }

    /// An overview level reads that level's cells alone — levels are
    /// never mixed.
    #[test]
    fn an_overview_reads_one_level() {
        let (p, _, _) = paris();
        let parts = p.parts_for(6, Some([2.349, 48.849, 2.351, 48.851]));
        assert!(!parts.is_empty());
        assert!(parts.iter().all(|s| s.starts_with("r6/")), "{parts:?}");
    }

    #[test]
    fn a_rect_outside_every_cell_opens_nothing() {
        let (p, _, _) = paris();
        for res in [5, 6, 7] {
            assert!(p.parts_for(res, Some([-70.0, 45.0, -69.9, 45.1])).is_empty());
        }
    }

    /// The ground scale picks the level, and the part count vetoes it: a
    /// level that would open more files than the budget allows is the
    /// wrong level, so the planner takes the next coarser one.
    #[test]
    fn the_plan_follows_the_scale_then_the_part_budget() {
        let (p, _, _) = paris();
        let rect = Some([2.349, 48.849, 2.351, 48.851]);
        // r7 edge ~1.2 km, 64 px per cell -> ~19 m/px; r6 ~50 m/px.
        assert_eq!(p.plan(Some(1.0), rect, 256).res, 7);
        assert_eq!(p.plan(Some(40.0), rect, 256).res, 6);
        assert_eq!(p.plan(Some(200.0), rect, 256).res, 5);
        // No viewport at all: the leaf band over every cell it lists.
        let all = p.plan(None, None, 256);
        assert_eq!(all.res, 7);
        assert_eq!(all.parts.len(), 6 + 7);
        // A budget that the leaf band busts but r6 fits: the plan
        // coarsens by one level and says where it came from.
        let room = p.parts_for(6, rect).len();
        assert!(room < p.parts_for(7, rect).len(), "the leaf band is the wider one");
        let tight = p.plan(Some(1.0), rect, room);
        assert_eq!(tight.res, 6);
        assert_eq!(tight.coarsened_from, Some(7));
        // The coarsest level is taken however wide it is: there is
        // nothing above it to fall back to.
        let floor = p.plan(Some(1.0), rect, 0);
        assert_eq!(floor.res, 5);
    }

    #[test]
    fn an_overview_badges_and_the_leaf_does_not() {
        let (p, _, _) = paris();
        assert_eq!(p.badge(), None, "the leaf level is the source's own features");
        assert_eq!(p.with_active(6).badge().as_deref(), Some("overview r6 (dissolve)"));
        assert_eq!(p.with_active(5).badge().as_deref(), Some("overview r5 (dissolve)"));
        assert!(p.with_active(6).is_overview());
        assert!(!p.is_overview());
    }

    #[test]
    fn the_info_line_says_what_the_pyramid_is() {
        let (p, _, _) = paris();
        assert_eq!(
            p.info_line(),
            "leaf r7 (adaptive to r8), overviews r5..r6 (dissolve), 64 px/cell"
        );
        let mut d = p.descriptor.clone();
        d.leaf.adaptive_max_res = 7;
        d.levels.retain(|l| l.res != 8);
        d.levels.retain(|l| l.res != 5);
        let flat = PyramidState::new(d, "/tmp/pyr").unwrap();
        assert_eq!(flat.info_line(), "leaf r7, overviews r6 (dissolve), 64 px/cell");
    }

    /// The leaf file list is what SQL and the scorecard count: every
    /// reference-resolution cell, every adaptive child, and the null part.
    #[test]
    fn leaf_parts_include_the_null_part() {
        let (p, _, children) = paris();
        let leaves = p.leaf_parts();
        assert_eq!(leaves.len(), 6 + children.len() + 1);
        assert!(leaves.contains(&part_path(7, NULL_PART)));
        assert_eq!(p.all_parts().len(), leaves.len() + p.descriptor.levels[0].cells.len()
            + p.descriptor.levels[1].cells.len());
    }

    /// `h3 = '<cell>'` names a leaf file, and the children that cell was
    /// split into.
    #[test]
    fn a_cell_filter_reaches_its_adaptive_children() {
        let (_, split, children) = paris();
        let split = split.to_string();
        assert!(cell_matches(&split, &split));
        for c in &children {
            assert!(cell_matches(&c.to_string(), &split));
            assert!(!cell_matches(&split, &c.to_string()), "a parent is not one of its children");
        }
        assert!(!cell_matches("not a cell", &split));
    }

    #[test]
    fn files_lists_every_part_once() {
        let mut d = desc();
        d.levels[1].cells = vec!["862a30667ffffff".into()];
        d.levels[3].cells = vec!["882a3066d1fffff".into(), "882a3066d3fffff".into()];
        d.levels[4].cells = vec!["892a3066d13ffff".into()];
        let files = d.files();
        assert_eq!(
            files,
            vec![
                "r5/852a3067fffffff.parquet",
                "r6/862a30667ffffff.parquet",
                "r8/882a3066d1fffff.parquet",
                "r8/882a3066d3fffff.parquet",
                "r9/892a3066d13ffff.parquet",
                "r8/__HIVE_DEFAULT_PARTITION__.parquet",
            ]
        );
        let uniq: std::collections::HashSet<&String> = files.iter().collect();
        assert_eq!(uniq.len(), files.len(), "no path is listed twice");

        // No null part means no null entry, and an empty pyramid lists nothing.
        d.leaf.null_part = false;
        assert_eq!(d.files().len(), 5);
        for l in &mut d.levels {
            l.cells.clear();
        }
        assert!(d.files().is_empty());
    }

    #[test]
    fn paths_and_file_meta() {
        assert_eq!(part_path(6, "862a30667ffffff"), "r6/862a30667ffffff.parquet");
        let m = FileMeta { res: 6, method: Method::Dissolve, source_res: 7, derived: true };
        let back: FileMeta = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
    }
}
