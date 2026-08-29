use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use geo::MapCoordsInPlace;
use geo_traits::to_geo::ToGeoGeometry;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::PageIndexPolicy;
use rayon::prelude::*;
use rstar::RTree;
use serde_json::Value;

use super::crs::{BulkTransformer, Crs, DisplayCrs};
use super::geoarrow::{GeomCol, GeomEncoding};
use super::geometry::{FeatureRef, MeshBuilder};
use super::info::{summarize_geo_meta, ColumnInfo, FileInfo};
use super::layer::{GroupLoad, LayerGeometry, LoadStats, PickItem, RgBboxes, VectorLayer};
use super::pyramid::{self, PyramidState};
use super::source::Source;
use super::store::{CoveringCol, FeatureStore};

const BATCH_SIZE: usize = 64 * 1024;
/// Geometry bytes a single read batch should carry. With one batch in
/// flight per worker, this times the core count is the decode footprint,
/// whatever the dataset's rows weigh.
const BATCH_TARGET_BYTES: u64 = 16 * 1024 * 1024;

/// Error string of a user-cancelled load (the app treats it quietly).
pub const CANCELLED: &str = "load cancelled";
/// An append that found nothing to do. Not an error: the viewport simply
/// holds no part files the layer has not already opened.
pub const NOTHING_TO_APPEND: &str = "no parts to add";

pub enum LoadMsg {
    Progress {
        job: u64,
        frac: f32,
        stage: String,
    },
    Loaded {
        job: u64,
        layer: Box<VectorLayer>,
        /// Auto-selected display projection: (display, geometry already
        /// built in it). false = the app must run a projection rebuild.
        adopt_display: Option<(DisplayCrs, bool)>,
    },
    /// Where this layer will land, known from metadata alone and sent
    /// before a byte of geometry is read.
    ///
    /// The camera used to sit on the whole world until the build finished,
    /// which meant the basemap had nothing to fetch for however long that
    /// took, then everything at once. Framing early lets the tiles load
    /// alongside the data. The exact bounds still refine the fit when the
    /// layer itself arrives.
    Framed {
        job: u64,
        /// The auto-selected projection `world` is expressed in, when the
        /// app is not already on it.
        display: Option<DisplayCrs>,
        world: [f64; 4],
    },
    /// Geometry rebuilt for a new display projection (replaces all sections).
    Rebuilt {
        layer_id: u64,
        generation: u64,
        geometry: LayerGeometry,
        stats_build_ms: u64,
        bad_geoms: usize,
    },
    /// Full drop-and-reload restricted to the viewport: replaces the
    /// sections AND the per-group decode state.
    Reloaded {
        layer_id: u64,
        generation: u64,
        geometry: LayerGeometry,
        loaded: Vec<GroupLoad>,
        rows: usize,
        bad_geoms: usize,
        build_ms: u64,
    },
    /// A row append ended without a result (error or user cancel).
    AppendEnded { layer_id: u64, error: String },
    /// Extra part files of a multi-part collection were opened and the
    /// layer's store has grown to include them.
    ///
    /// Always precedes the `Appended` messages that carry their geometry:
    /// those address row groups by global index, which only exist once the
    /// bigger store is in place. Appending fragments is index-stable, so
    /// the geometry and decode state already on the layer stay valid.
    PartsOpened {
        layer_id: u64,
        generation: u64,
        store: Arc<FeatureStore>,
        /// Row-group boxes for the appended groups only, in order.
        added_boxes: Vec<[f64; 4]>,
        /// How many row groups the store gained.
        added_groups: usize,
        /// Part file names, for the status line.
        names: Vec<String>,
    },
    /// Exact viewport selection is still too large for a safe refinement.
    /// This is not an error: retry after the camera moves to a tighter view.
    RefineDeferred {
        layer_id: u64,
        at_least_rows: u64,
        /// Set when geometry bytes, not the row count, stopped it.
        geom_bytes: Option<u64>,
    },
    /// A projection/filter rebuild ended without geometry (error or
    /// cancel). The layer keeps its previous sections; the app clears its
    /// rebuilding flag only when `generation` is still the layer's current
    /// one (a superseded rebuild must not unmask the newer in-flight one).
    RebuildFailed {
        layer_id: u64,
        generation: u64,
        error: String,
    },
    /// Additional rows loaded for an existing layer (new section).
    /// Appends stream in batches — the first lands fast so refinement
    /// is visible while the rest decodes; `done` marks the last one.
    Appended {
        layer_id: u64,
        generation: u64,
        geometry: LayerGeometry,
        rows: usize,
        /// New decode state per touched row group.
        loaded: Vec<(u32, GroupLoad)>,
        done: bool,
    },
    Failed {
        job: u64,
        source: String,
        error: String,
    },
    /// A non-indexable file too big for a full build opened under
    /// `LoadMode::Auto`: the app must ask the user (Optimize / load all /
    /// cancel) before anything is decoded (docs/OPEN_POLICY.md). Carries
    /// the opened store so the chosen path resumes without reopening.
    QualityGate {
        job: u64,
        layer_id: u64,
        opened: Box<OpenedStore>,
        color: eframe::egui::Color32,
        auto_project: bool,
    },
}

/// A store opened up to (and including) metadata analysis, before any
/// data pages are read — the resume point after the quality gate.
pub struct OpenedStore {
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
    pub info: FileInfo,
    pub rg_meta: Option<(String, Vec<[f64; 4]>)>,
}

pub struct LoaderHandle {
    pub tx: Sender<LoadMsg>,
    pub egui_ctx: eframe::egui::Context,
}

impl LoaderHandle {
    fn send(&self, msg: LoadMsg) {
        let _ = self.tx.send(msg);
        self.egui_ctx.request_repaint();
    }
}

/// Compute a conservative data-CRS bbox for a world-space viewport rect by
/// sampling a grid of points through the inverse projection chain.
pub fn viewport_to_data_bbox(
    view_world: [f64; 4],
    display: &DisplayCrs,
    data_crs: &Crs,
) -> Option<[f64; 4]> {
    use super::crs::transform_point;
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut any = false;
    const N: usize = 8;
    for i in 0..=N {
        for j in 0..=N {
            let w = [
                view_world[0] + (view_world[2] - view_world[0]) * i as f64 / N as f64,
                view_world[1] + (view_world[3] - view_world[1]) * j as f64 / N as f64,
            ];
            let (px, py) = display.projected_from_world(w);
            if !px.is_finite() || !py.is_finite() {
                continue;
            }
            if let Ok((x, y)) = transform_point(&display.crs, data_crs, px, py) {
                if x.is_finite() && y.is_finite() {
                    b[0] = b[0].min(x);
                    b[1] = b[1].min(y);
                    b[2] = b[2].max(x);
                    b[3] = b[3].max(y);
                    any = true;
                }
            }
        }
    }
    if !any {
        return None;
    }
    // Safety margin for projection nonlinearity between samples.
    let (mx, my) = ((b[2] - b[0]) * 0.05, (b[3] - b[1]) * 0.05);
    Some([b[0] - mx, b[1] - my, b[2] + mx, b[3] + my])
}

/// Inverse of [`viewport_to_data_bbox`]: a data-CRS bbox expressed in world
/// (display) coordinates, by sampling a grid of points through the
/// transform. None when nothing transforms (e.g. outside projection domain).
pub fn data_bbox_to_world(
    bbox: [f64; 4],
    data_crs: &Crs,
    display: &DisplayCrs,
) -> Option<[f64; 4]> {
    use super::crs::transform_point;
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut any = false;
    const N: usize = 8;
    for i in 0..=N {
        for j in 0..=N {
            let x = bbox[0] + (bbox[2] - bbox[0]) * i as f64 / N as f64;
            let y = bbox[1] + (bbox[3] - bbox[1]) * j as f64 / N as f64;
            let Ok((px, py)) = transform_point(data_crs, &display.crs, x, y) else {
                continue;
            };
            if !px.is_finite() || !py.is_finite() {
                continue;
            }
            let w = display.world_from_projected(px, py);
            if w[0].is_finite() && w[1].is_finite() {
                b[0] = b[0].min(w[0]);
                b[1] = b[1].min(w[1]);
                b[2] = b[2].max(w[0]);
                b[3] = b[3].max(w[1]);
                any = true;
            }
        }
    }
    any.then_some(b)
}

/// Never-intersecting placeholder for a row group that contributed no
/// decodable geometry: keeps computed per-group bbox vectors index-aligned
/// with the file's row groups, and fails every intersection test.
pub(crate) const EMPTY_BBOX: [f64; 4] =
    [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

/// Union of a set of bboxes, ignoring empty (inverted) sentinels.
/// Ground sample distance of the current view: metres of ground per
/// screen pixel, from the viewport's width in the data CRS and its width
/// in physical pixels.
///
/// Geographic data converts at the viewport's centre latitude, because
/// "metres of ground" for a rectangle on a globe is a function of where
/// it sits. Projected CRSs are read as metres — the ones this matters
/// for (COGP level choice) are, and being a few percent off on a
/// foot-based grid picks the same level either way.
pub fn view_gsd(rect: [f64; 4], view_px: f64, crs: &Crs) -> Option<f64> {
    let width = rect[2] - rect[0];
    if !(width.is_finite() && width > 0.0 && view_px.is_finite() && view_px >= 1.0) {
        return None;
    }
    let per_px = width / view_px;
    Some(if crs.is_latlong {
        let lat = ((rect[1] + rect[3]) * 0.5).clamp(-89.9, 89.9);
        per_px * 111_320.0 * lat.to_radians().cos().max(1e-6)
    } else {
        per_px
    })
}

/// Last row group a COGP layer needs for this viewport: the finest level
/// whose gsd still covers the view (SPEC §7.1). None for a layer with no
/// COGP levels, or a view whose ground scale cannot be worked out — in
/// both cases planning proceeds over every row group, as before.
///
/// This is not pruning. Row groups past the prefix hold features that
/// are not independently meaningful at this scale; skipping them is the
/// point of the layout, not a compromise, which is why a prefix that
/// fits the build budget loads as exact geometry with no badge.
pub fn cogp_prefix_end(
    store: &FeatureStore,
    rect: Option<[f64; 4]>,
    view_px: f64,
    crs: &Crs,
) -> Option<u32> {
    let levels = store.cogp.as_ref()?;
    let gsd = view_gsd(rect?, view_px, crs)?;
    Some(levels.row_group_end_for_gsd(gsd) as u32)
}

pub(crate) fn union_of(boxes: &[[f64; 4]]) -> Option<[f64; 4]> {
    boxes
        .iter()
        .filter(|b| b[0] <= b[2] && b[1] <= b[3])
        .copied()
        .reduce(|a, b| {
            [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
        })
}

/// Row groups whose bbox intersects the given data-CRS rect.
pub fn intersecting_rgs(boxes: &[[f64; 4]], rect: [f64; 4]) -> Vec<u32> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| b[0] <= rect[2] && b[2] >= rect[0] && b[1] <= rect[3] && b[3] >= rect[1])
        .map(|(i, _)| i as u32)
        .collect()
}

/// What to decode from one row group.
#[derive(Clone, Debug)]
pub enum GroupSel {
    /// The whole group.
    All(u32),
    /// Only features whose covering bbox intersects the rect (data CRS);
    /// resolved to row ranges by a bbox-leaf scan in the worker. Falls back
    /// to the whole group when the file has no covering column.
    Rect(u32, [f64; 4]),
    /// A rect selection whose covering/x-y scan has already been resolved.
    /// Used by refinement so its budget is based on the exact visible rows
    /// and the expensive selection scan is not repeated during decoding.
    ResolvedRect {
        group: u32,
        rect: [f64; 4],
        ranges: Vec<(u32, u32)>,
    },
    /// Explicit group-relative [start, end) row ranges (rebuilds of
    /// partially loaded groups, complement appends).
    Ranges(u32, Vec<(u32, u32)>),
    /// Decimated preview: every `stride`-th row, optionally rect-filtered
    /// first. Used when a full decode would exceed the row budget.
    Preview {
        group: u32,
        rect: Option<[f64; 4]>,
        stride: u32,
    },
    /// Every feature drawn from its covering bbox, no geometry read.
    /// Chosen over `Preview` for polygon coverages that carry a covering
    /// column (see `GroupLoad::Boxes`).
    Boxes {
        group: u32,
        rect: Option<[f64; 4]>,
    },
    /// Boxes for exactly these group-relative rows.
    ///
    /// A group refined for one viewport holds real geometry for those
    /// rows and nothing for the rest, which on a box layer would punch a
    /// row-group-shaped hole in the coverage as soon as the camera moved
    /// on. The complement is drawn as boxes instead, so the map stays
    /// complete and the refined part simply gets its true outlines.
    BoxRanges { group: u32, ranges: Vec<(u32, u32)> },
}

impl GroupSel {
    fn group(&self) -> u32 {
        match self {
            GroupSel::All(g) | GroupSel::Rect(g, _) | GroupSel::Ranges(g, _) => *g,
            GroupSel::Preview { group, .. }
            | GroupSel::Boxes { group, .. }
            | GroupSel::BoxRanges { group, .. }
            | GroupSel::ResolvedRect { group, .. } => *group,
        }
    }
}

/// Row budget for one build: selections above it decode a decimated
/// preview instead (every Nth row), refined with real rows on zoom-in.
pub const MAX_BUILD_ROWS: u64 = 2_500_000;
/// The same budget in geometry bytes, because rows do not measure the
/// work. 2.5M parcels carry ~1 GB of WKB and build fine; 2.4M land-cover
/// polygons carry 7 GB and take tens of gigabytes of RAM to tessellate,
/// while sitting *under* the row budget and looking harmless.
///
/// Calibrated against the mesh, which is what actually occupies memory.
/// Measured on CORINE: 0.18 GB of WKB becomes 0.65 GB of mesh (240 MB of
/// fills, 413 MB of line segments across the LOD stack) and peaks at
/// 1.45 GB of RSS, so a build costs about 8× the geometry bytes it
/// reads. The earlier 1 GB budget therefore authorized an 8 GB peak,
/// which is more than many machines can give and, on one without swap,
/// takes the machine down rather than the app.
pub const MAX_BUILD_GEOM_BYTES: u64 = 256 << 20;
/// Preview decimation targets roughly this many features.
const PREVIEW_TARGET_ROWS: u64 = 1_200_000;
/// …and roughly this many geometry bytes: ≈0.5 GB of mesh, ≈1 GB of peak
/// at the measured expansion. Quartered from 512 MB, which costs the
/// MassGIS parcels stride levels at world zoom (1/3 → 1/8) and takes
/// CORINE from 1/15 to 1/57 — except that CORINE now draws boxes
/// instead, keeping every feature. Zooming in refines to real geometry
/// under the same budget either way, so what this really sets is how
/// coarse the whole-dataset view is, not what the user works with.
const PREVIEW_TARGET_BYTES: u64 = 128 << 20;

/// What a selection that exceeds the build budget falls back to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OverBudget {
    /// Every feature, drawn from its covering bbox.
    Boxes,
    /// One feature in `stride`, drawn from its geometry.
    Stride,
}

/// Which fallback a store can offer.
///
/// Boxes need a covering column to read and polygons to stand in for.
/// Given both they are the better answer: a stride preview throws away
/// most of the features, which on data that tiles the plane reads as
/// holes, while boxes keep every feature at a resolution the screen can
/// show. Without a covering column there is nothing to draw but the
/// geometry, and for lines a bounding box is not a stand-in for anything.
fn over_budget_plan(covering: bool, polygons_only: bool) -> OverBudget {
    if covering && polygons_only {
        OverBudget::Boxes
    } else {
        OverBudget::Stride
    }
}

/// Decimation stride for a candidate selection, or None when it fits
/// both budgets. Rows and geometry bytes are separate limits because
/// neither predicts the other: a million points are 2.5M rows of nothing,
/// a hundred thousand land-cover polygons are gigabytes.
fn preview_stride(rows: u64, geom_bytes: u64) -> Option<u32> {
    if rows <= MAX_BUILD_ROWS && geom_bytes <= MAX_BUILD_GEOM_BYTES {
        return None;
    }
    let by_rows = rows.div_ceil(PREVIEW_TARGET_ROWS);
    let by_bytes = geom_bytes.div_ceil(PREVIEW_TARGET_BYTES);
    Some(by_rows.max(by_bytes).max(2).min(u32::MAX as u64) as u32)
}

/// Coalesce sorted row indices into [start, end) ranges.
fn rows_to_ranges(rows: impl Iterator<Item = u32>) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for r in rows {
        match out.last_mut() {
            Some((_, end)) if *end == r => *end = r + 1,
            _ => out.push((r, r + 1)),
        }
    }
    out
}

/// Rows of `0..len` not covered by `ranges` (sorted, non-overlapping).
pub fn complement_ranges(ranges: &[(u32, u32)], len: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(ranges.len() + 1);
    let mut pos = 0u32;
    for &(start, end) in ranges {
        if start > pos {
            out.push((pos, start));
        }
        pos = pos.max(end);
    }
    if pos < len {
        out.push((pos, len));
    }
    out
}

/// Scan a group's covering bbox leaves and return the row ranges of
/// features intersecting `rect`. None when the file has no usable covering
/// column (caller falls back to the whole group).
pub(crate) fn covering_select(
    store: &FeatureStore,
    group: u32,
    rect: [f64; 4],
) -> Result<Option<Vec<(u32, u32)>>, String> {
    use arrow::array::{Array, Float64Array, StructArray};
    let Some(cov) = &store.covering else {
        // x/y point stores: the coordinate columns themselves are an
        // exact covering — scan them for the in-rect row ranges. This is
        // what keeps lat-ordered global grids from decoding a whole
        // world-wide strip per row group.
        if let Some((xi, yi)) = store.xy_geom {
            return xy_select(store, group, rect, xi, yi).map(Some);
        }
        // WKB (incl. 2.0 native GEOMETRY) without a covering column:
        // scan the geometry column's envelopes byte-wise. Reads the same
        // column the decode would, but only matching rows become
        // geometry — the win is decode/tessellation work and memory.
        if store.encoding.is_wkb() {
            return wkb_envelope_select(store, group, rect).map(Some);
        }
        return Ok(None);
    };
    let reader = store.reader_for_group(group as usize, BATCH_SIZE, None, Some(&[cov.root]))?;
    let mut rows: Vec<u32> = Vec::new();
    let mut row = 0u32;
    for res in reader {
        let batch = res.map_err(|e| format!("covering scan error: {e}"))?;
        let st = batch
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or("covering column is not a struct")?;
        let leaf = |name: &str| -> Result<Float64Array, String> {
            let col = st
                .column_by_name(name)
                .ok_or_else(|| format!("covering child '{name}' missing"))?;
            arrow::compute::cast(col, &arrow::datatypes::DataType::Float64)
                .map_err(|e| format!("covering cast: {e}"))
                .map(|a| a.as_any().downcast_ref::<Float64Array>().unwrap().clone())
        };
        let (xmin, ymin, xmax, ymax) = (
            leaf(&cov.children[0])?,
            leaf(&cov.children[1])?,
            leaf(&cov.children[2])?,
            leaf(&cov.children[3])?,
        );
        for i in 0..batch.num_rows() {
            if !st.is_null(i)
                && !xmin.is_null(i)
                && xmin.value(i) <= rect[2]
                && xmax.value(i) >= rect[0]
                && ymin.value(i) <= rect[3]
                && ymax.value(i) >= rect[1]
            {
                rows.push(row + i as u32);
            }
        }
        row += batch.num_rows() as u32;
    }
    Ok(Some(rows_to_ranges(rows.into_iter())))
}

/// In-rect row ranges of one group by scanning the WKB envelopes (byte
/// parse, no geometry allocation). Null and unparseable geometries are
/// not selected.
fn wkb_envelope_select(
    store: &FeatureStore,
    group: u32,
    rect: [f64; 4],
) -> Result<Vec<(u32, u32)>, String> {
    let reader =
        store.reader_for_group(group as usize, BATCH_SIZE, None, Some(&[store.geom_col]))?;
    let mut rows: Vec<u32> = Vec::new();
    let mut row = 0u32;
    for res in reader {
        let batch = res.map_err(|e| format!("geometry scan error: {e}"))?;
        let col = BinCol::new(batch.column(0).as_ref())
            .ok_or("geometry column is not binary")?;
        for i in 0..batch.num_rows() {
            if let Some(buf) = col.value(i) {
                let mut env =
                    [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                if grow_wkb_envelope(buf, &mut env).is_some()
                    && env[0] <= rect[2]
                    && env[2] >= rect[0]
                    && env[1] <= rect[3]
                    && env[3] >= rect[1]
                {
                    rows.push(row + i as u32);
                }
            }
        }
        row += batch.num_rows() as u32;
    }
    Ok(rows_to_ranges(rows.into_iter()))
}

/// In-rect row ranges of one group of an x/y point store, by scanning
/// the two coordinate columns (the same bytes the decode would read, but
/// only matching rows become geometry).
fn xy_select(
    store: &FeatureStore,
    group: u32,
    rect: [f64; 4],
    xi: usize,
    yi: usize,
) -> Result<Vec<(u32, u32)>, String> {
    use arrow::array::Float64Array;
    let cols = if xi < yi { [xi, yi] } else { [yi, xi] };
    let (xpos, ypos) = if xi < yi { (0, 1) } else { (1, 0) };
    let reader = store.reader_for_group(group as usize, BATCH_SIZE, None, Some(&cols))?;
    let mut rows: Vec<u32> = Vec::new();
    let mut row = 0u32;
    for res in reader {
        let batch = res.map_err(|e| format!("coordinate scan error: {e}"))?;
        let as_f64 = |i: usize| -> Result<Float64Array, String> {
            arrow::compute::cast(batch.column(i), &DataType::Float64)
                .map_err(|e| format!("coordinate cast: {e}"))
                .map(|a| a.as_any().downcast_ref::<Float64Array>().unwrap().clone())
        };
        let (xs, ys) = (as_f64(xpos)?, as_f64(ypos)?);
        for i in 0..batch.num_rows() {
            if !xs.is_null(i)
                && !ys.is_null(i)
                && xs.value(i) >= rect[0]
                && xs.value(i) <= rect[2]
                && ys.value(i) >= rect[1]
                && ys.value(i) <= rect[3]
            {
                rows.push(row + i as u32);
            }
        }
        row += batch.num_rows() as u32;
    }
    Ok(rows_to_ranges(rows.into_iter()))
}

/// Load a GeoParquet file in a background thread. Only geometry-derived data
/// stays in memory; attributes are re-read lazily via `FeatureStore`.
///
/// `view_world`: current viewport in world space. When the file carries
/// per-row-group bboxes in its metadata, row groups outside the viewport
/// are pruned and stream in later on demand.
#[allow(clippy::too_many_arguments)]
pub fn spawn_load(
    handle: LoaderHandle,
    job: u64,
    layer_id: u64,
    source: Source,
    display: DisplayCrs,
    color: eframe::egui::Color32,
    view_world: [f64; 4],
    // Viewport width in physical pixels: with `view_world` it gives the
    // ground scale a COGP layer picks its level from.
    view_px: f64,
    auto_project: bool,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
) {
    use std::sync::atomic::Ordering;
    std::thread::spawn(move || {
        let cancelled = || cancel.load(Ordering::Relaxed);
        let t0 = Instant::now();
        // Resolve URLs / S3 credentials (content-length probe, presign)
        // off the UI thread.
        let label = source.label();
        handle.send(LoadMsg::Progress {
            job,
            frac: 0.0,
            stage: "resolving source".into(),
        });
        let source = match source.resolve() {
            Ok(s) => s,
            Err(e) => {
                handle.send(LoadMsg::Failed {
                    job,
                    source: label,
                    error: e,
                });
                return;
            }
        };
        handle.send(LoadMsg::Progress {
            job,
            frac: 0.02,
            stage: "reading file metadata".into(),
        });
        if cancelled() {
            handle.send(LoadMsg::Failed { job, source: label, error: CANCELLED.into() });
            return;
        }
        // Part pruning happens at open, before any CRS is known. STAC
        // item bboxes are WGS84 lon/lat by spec, and so is an H3 cell;
        // the pixel width comes along because a pyramid picks its level
        // from the ground scale, not from where the camera is.
        let view = viewport_to_data_bbox(view_world, &display, &Crs::wgs84())
            .map(|rect| ViewHint { rect, view_px });
        match open_store_with_view(&source, view) {
            Ok((store, crs, info, rg_meta)) => {
                if cancelled() {
                    handle.send(LoadMsg::Failed {
                        job,
                        source: source.label(),
                        error: CANCELLED.into(),
                    });
                    return;
                }
                let store = Arc::new(store);
                // Non-indexable file too big for a full build: hand the
                // opened store to the app for the quality-gate dialog (or
                // an instant resume when the file has a remembered answer)
                // instead of silently previewing.
                let gate = info.quality.as_ref().is_some_and(|q| !q.indexable)
                    && store.total_rows() > MAX_BUILD_ROWS;
                let opened = OpenedStore {
                    store,
                    crs,
                    info,
                    rg_meta,
                };
                if gate {
                    handle.send(LoadMsg::QualityGate {
                        job,
                        layer_id,
                        opened: Box::new(opened),
                        color,
                        auto_project,
                    });
                    return;
                }
                build_opened(
                    &handle,
                    job,
                    layer_id,
                    opened,
                    display,
                    color,
                    view_world,
                    view_px,
                    auto_project,
                    &cancel,
                    style,
                    false,
                    t0,
                );
            }
            Err(e) => handle.send(LoadMsg::Failed {
                job,
                source: source.label(),
                error: e,
            }),
        }
    });
}

/// Resume a load the quality gate paused: the user chose "load all"
/// (`direct` true) or the file's remembered answer applied.
#[allow(clippy::too_many_arguments)]
pub fn spawn_load_gated(
    handle: LoaderHandle,
    job: u64,
    layer_id: u64,
    opened: OpenedStore,
    display: DisplayCrs,
    color: eframe::egui::Color32,
    view_world: [f64; 4],
    view_px: f64,
    auto_project: bool,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
    direct: bool,
) {
    std::thread::spawn(move || {
        build_opened(
            &handle,
            job,
            layer_id,
            opened,
            display,
            color,
            view_world,
            view_px,
            auto_project,
            &cancel,
            style,
            direct,
            Instant::now(),
        );
    });
}

/// Plan + build + send for an opened store: the shared tail of first
/// loads, whichever side of the quality gate they took.
#[allow(clippy::too_many_arguments)]
fn build_opened(
    handle: &LoaderHandle,
    job: u64,
    layer_id: u64,
    opened: OpenedStore,
    display: DisplayCrs,
    color: eframe::egui::Color32,
    view_world: [f64; 4],
    view_px: f64,
    auto_project: bool,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
    direct: bool,
    t0: Instant,
) {
    let OpenedStore {
        store,
        crs,
        info,
        rg_meta,
    } = opened;
    log::debug!("build_opened: job {job}, layer {layer_id}, direct {direct}");
    let n_rg = store.rg_starts().len().saturating_sub(1);
    // The pruning rect is in the DATA CRS, so it stays valid
    // across a display switch — compute it with the display the
    // viewport coordinates are expressed in.
    let rect = viewport_to_data_bbox(view_world, &display, &crs);
    // Auto projection: pick a best-fit display before building
    // when the extent is known from metadata (projected data CRS
    // needs no extent at all).
    let mut display = display;
    let mut adopt_display: Option<(DisplayCrs, bool)> = None;
    if auto_project {
        let mut bbox = rg_meta
            .as_ref()
            .filter(|_| crs.is_latlong)
            .and_then(|(_, boxes)| union_of(boxes));
        // No metadata extent (raw WKB files): probe it with an
        // envelope-only scan so the single build happens directly in
        // the adopted projection. The post-build fallback below would
        // otherwise tessellate and upload the whole layer twice.
        if bbox.is_none() && crs.is_latlong && store.encoding.is_wkb() {
            handle.send(LoadMsg::Progress {
                job,
                frac: 0.04,
                stage: "scanning extent".into(),
            });
            bbox = scan_wkb_extent(&store, cancel);
        }
        if let Some(d) = DisplayCrs::auto_for(&crs, bbox) {
            log::info!("auto projection: {}", d.name);
            display = d.clone();
            adopt_display = Some((d, true));
        }
    }
    // Frame the view now that the projection is settled. The row-group
    // boxes give the extent for free, so the map can point at the right
    // place — and the basemap start downloading for it — while the
    // geometry is still being read.
    if let Some(world) = rg_meta
        .as_ref()
        .and_then(|(_, boxes)| union_of(boxes))
        .and_then(|b| data_bbox_to_world(b, &crs, &display))
    {
        handle.send(LoadMsg::Framed {
            job,
            display: adopt_display.as_ref().map(|(d, _)| d.clone()),
            world,
        });
    }
    let sel = if direct {
        // Direct mode: every group in full — no viewport planning, no
        // preview fallback (docs/OPEN_POLICY.md).
        (0..n_rg as u32).map(GroupSel::All).collect()
    } else {
        // Prune only with metadata-sourced boxes: a pruned first load
        // never sees the skipped row groups, so computed boxes can't
        // drive it.
        plan_viewport_selection(
            &store,
            &store.source.label(),
            rg_meta.as_ref().map(|(_, b)| b.as_slice()),
            rect,
            cogp_prefix_end(&store, rect, view_px, &crs),
            "initial load",
        )
    };
    let box_layer = sel.iter().any(|s| matches!(s, GroupSel::Boxes { .. }));
    let build_t0 = Instant::now();
    // No style asked for (a plain open, not a restored context): let the
    // schema speak. A column named for a published nomenclature gets
    // that nomenclature's colours, binned in this very build — the same
    // style applied afterwards would cost a full rebuild.
    let style = style.or_else(|| {
        let sb = crate::data::colormap::schema_style(&store.style_columns())?;
        log::info!("auto colour map on column {}", sb.column);
        Some(sb)
    });
    let style_sel = style.as_ref().and_then(|sb| resolve_style(&store, sb));
    match build_geometry(
        &store,
        &crs,
        &display,
        Some((handle, job)),
        sel,
        Some(cancel),
        style_sel.as_ref(),
    ) {
        Ok((geometry, rows, bad, rg_computed, resolved)) => {
            // Extent only known post-build (no metadata bboxes):
            // suggest the auto projection; the app rebuilds.
            if auto_project && adopt_display.is_none() && crs.is_latlong {
                if let Some(d) = DisplayCrs::auto_for(&crs, union_of(&rg_computed)) {
                    log::info!("auto projection (post-build): {}", d.name);
                    adopt_display = Some((d, false));
                }
            }
            let mut loaded = vec![GroupLoad::None; n_rg];
            for (g, st) in resolved {
                loaded[g as usize] = st;
            }
            let rg_bboxes = rg_meta
                .or_else(|| {
                    // At least one real (non-sentinel) box.
                    rg_computed
                        .iter()
                        .any(|b| b[0] <= b[2])
                        .then(|| ("computed at load".to_string(), rg_computed))
                })
                .map(|(source, boxes)| {
                    // Measured the way C2 measures it: on a COGP layer the
                    // worst level, inside itself. A file-wide average
                    // there reports a correct layout as poorly clustered
                    // (see `quality::Clustering::worst`), and this is what
                    // the info panel and the "consider Export…" hint read.
                    let starts = store.rg_starts();
                    let rg_rows: Vec<u64> =
                        starts.windows(2).map(|w| w[1] - w[0]).collect();
                    RgBboxes::new(
                        source,
                        boxes,
                        store.cogp.as_ref().map(|c| c.runs(&rg_rows)).as_deref(),
                    )
                });
            let stats = LoadStats {
                read_ms: 0,
                build_ms: build_t0.elapsed().as_millis() as u64,
                rows,
                bad_geoms: bad,
            };
            let name = store.source.name();
            log::info!(
                "loaded {name}: {rows} features in {} ms",
                t0.elapsed().as_millis()
            );
            // The geometry is binned for `style`, so the layer must carry
            // it: the app overwrites this for a restored context, which
            // passed the very same style in.
            let mut layer_style = super::layer::LayerStyle::new(color);
            layer_style.style_by = style_sel.is_some().then_some(style).flatten();
            if layer_style
                .style_by
                .as_ref()
                .is_some_and(|sb| sb.mode.is_color_map())
            {
                layer_style.adopt_palette();
            }
            let layer = VectorLayer {
                id: layer_id,
                generation: 0,
                name,
                store,
                crs,
                sections: vec![geometry],
                box_layer,
                draw_gen: 0,
                style: layer_style,
                feature_count: rows,
                stats,
                info,
                rg_bboxes,
                loaded,
                filter: None,
                mode: if direct {
                    super::layer::LayerMode::Direct
                } else {
                    super::layer::LayerMode::Indexed
                },
            };
            handle.send(LoadMsg::Loaded {
                job,
                layer: Box::new(layer),
                adopt_display,
            });
        }
        Err(e) => handle.send(LoadMsg::Failed {
            job,
            source: store.source.label(),
            error: e,
        }),
    }
}

/// Rebuild geometry for an existing layer under a new display projection,
/// re-streaming exactly the loaded rows (consolidates any appended
/// sections into one).
/// The selections a rebuild runs for each group's current decode state.
///
/// `box_gaps`: the layer draws its unrefined data from covering boxes, so
/// every group must contribute *something* — a group that contributes
/// nothing is a hole in the coverage that no camera move can fill.
fn rebuild_selection(loaded: &[GroupLoad], starts: &[u64], box_gaps: bool) -> Vec<GroupSel> {
    loaded
        .iter()
        .enumerate()
        .flat_map(|(g, st)| {
            let group = g as u32;
            let rows = (starts[g + 1] - starts[g]) as u32;
            match st {
                GroupLoad::None => Vec::new(),
                GroupLoad::Full => vec![GroupSel::All(group)],
                // Empty ranges: either a layer-filter group with no
                // matching rows, or a group whose bbox met the viewport
                // while none of its features did. The second is routine —
                // row-group boxes are coarse — and on a box layer dropping
                // it takes the group's boxes with it, leaving a
                // row-group-shaped hole that only a full reload refills.
                GroupLoad::Rows { ranges, .. } if ranges.is_empty() => {
                    if box_gaps {
                        vec![GroupSel::Boxes { group, rect: None }]
                    } else {
                        Vec::new()
                    }
                }
                GroupLoad::Rows { ranges, .. } => {
                    let mut out = vec![GroupSel::Ranges(group, ranges.clone())];
                    // The rest of the group as boxes: without it the group
                    // shows geometry for one old viewport and nothing
                    // anywhere else.
                    if box_gaps {
                        let gaps = complement_ranges(ranges, rows);
                        if !gaps.is_empty() {
                            out.push(GroupSel::BoxRanges { group, ranges: gaps });
                        }
                    }
                    out
                }
                GroupLoad::Preview { stride, rect } => vec![GroupSel::Preview {
                    group,
                    rect: *rect,
                    stride: *stride,
                }],
                GroupLoad::Boxes { rect } => vec![GroupSel::Boxes { group, rect: *rect }],
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_rebuild(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    loaded: Vec<GroupLoad>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
    // Draw boxes for the rows a partly refined group did not decode, so
    // a box layer keeps complete coverage as the camera moves on.
    box_gaps: bool,
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        log::debug!("spawn_rebuild: layer {layer_id}, generation {generation}");
        let starts = store.rg_starts().to_vec();
        let sel: Vec<GroupSel> = rebuild_selection(&loaded, &starts, box_gaps);
        let style_sel = style.as_ref().and_then(|sb| resolve_style(&store, sb));
        match build_geometry(&store, &crs, &display, None, sel, Some(&cancel), style_sel.as_ref())
        {
            Ok((geometry, _rows, bad, _rg, _resolved)) => handle.send(LoadMsg::Rebuilt {
                layer_id,
                generation,
                geometry,
                stats_build_ms: t0.elapsed().as_millis() as u64,
                bad_geoms: bad,
            }),
            Err(e) if e == CANCELLED => handle.send(LoadMsg::RebuildFailed {
                layer_id,
                generation,
                error: CANCELLED.into(),
            }),
            Err(e) => handle.send(LoadMsg::RebuildFailed {
                layer_id,
                generation,
                error: format!("{}: projection rebuild failed: {e}", store.source.label()),
            }),
        }
    });
}

/// Load additional rows for an existing layer (viewport refinement or
/// "Load all"). `GroupSel::Ranges` here must be the complement of what is
/// already loaded — the group is marked `Full` afterwards.
///
/// With `refinement_budget`, rect jobs are first resolved against the real
/// covering/x-y values in this worker. This avoids both the old bbox-area
/// guess and a second covering scan during geometry decoding. `None` is used
/// by the explicit "Load all" action, which must not be silently budgeted.
#[allow(clippy::too_many_arguments)]
pub fn spawn_append(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    jobs: Vec<GroupSel>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
    refinement_budget: Option<(u64, u64)>,
) {
    std::thread::spawn(move || {
        run_append(
            &handle, layer_id, generation, &store, &crs, &display, jobs, &cancel, style,
            refinement_budget,
        );
    });
}

/// Resolve, budget and stream an append for `jobs` against `store`.
/// Shared by row refinement and by opening new part files.
#[allow(clippy::too_many_arguments)]
fn run_append(
    handle: &LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: &Arc<FeatureStore>,
    crs: &Crs,
    display: &DisplayCrs,
    jobs: Vec<GroupSel>,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
    refinement_budget: Option<(u64, u64)>,
) {
    {
        let jobs = if let Some((budget, geom_budget)) = refinement_budget {
            match prepare_refinement_jobs(store, jobs, budget, geom_budget, cancel) {
                Ok(RefinePlan::Ready(jobs)) => jobs,
                Ok(RefinePlan::Deferred { rows: at_least_rows, geom_bytes }) => {
                    match geom_bytes {
                        Some(b) => log::debug!(
                            "{}: refining that viewport would decode {} of geometry \
                             ({at_least_rows} rows) — zoom in further",
                            store.source.label(),
                            crate::data::info::fmt_bytes(b),
                        ),
                        None => log::debug!(
                            "{}: exact refinement exceeds {budget} rows (at least {at_least_rows}) — zoom in further",
                            store.source.label()
                        ),
                    }
                    handle.send(LoadMsg::RefineDeferred {
                        layer_id,
                        at_least_rows,
                        geom_bytes,
                    });
                    return;
                }
                Err(e) if e == CANCELLED => {
                    handle.send(LoadMsg::AppendEnded {
                        layer_id,
                        error: CANCELLED.into(),
                    });
                    return;
                }
                Err(e) => {
                    handle.send(LoadMsg::AppendEnded {
                        layer_id,
                        error: format!("{}: viewport selection failed: {e}", store.source.label()),
                    });
                    return;
                }
            }
        } else {
            jobs
        };
        let style_sel = style.as_ref().and_then(|sb| resolve_style(store, sb));
        // Stream the append: content appears within ~a second instead of
        // after the whole (up to budget-sized) build.
        let batches = append_batches(store, jobs);
        let last = batches.len().saturating_sub(1);
        for (bi, batch) in batches.into_iter().enumerate() {
            match build_geometry(
                store,
                crs,
                display,
                None,
                batch,
                Some(cancel),
                style_sel.as_ref(),
            ) {
                Ok((geometry, rows, _bad, _rg, resolved)) => handle.send(LoadMsg::Appended {
                    layer_id,
                    generation,
                    geometry,
                    rows,
                    loaded: resolved,
                    done: bi == last,
                }),
                Err(e) if e == CANCELLED => {
                    handle.send(LoadMsg::AppendEnded {
                        layer_id,
                        error: CANCELLED.into(),
                    });
                    return;
                }
                Err(e) => {
                    handle.send(LoadMsg::AppendEnded {
                        layer_id,
                        error: format!("{}: row append failed: {e}", store.source.label()),
                    });
                    return;
                }
            }
        }
    }
}

/// Part files one pan may add. Each costs a length probe plus a footer
/// fetch, so this bounds the round-trips a single camera move can spend
/// while still letting a sustained pan pull a collection in steadily.
pub const PART_APPEND_PER_PASS: usize = 8;

/// Total parts a layer may hold. Footers are small next to geometry, but
/// they are not free, and a pan across a large collection would otherwise
/// accumulate every one of them.
pub const PART_TOTAL_CAP: usize = 256;

/// Parts the viewport wants that the layer does not have, best first.
///
/// "Best" is intersection area with the viewport: a pan should pull in
/// what it is heading into before what it is only clipping the corner of.
/// Parts with no bbox are skipped rather than guessed at — an unbounded
/// item would sort as if it covered everything and crowd out the rest.
fn parts_to_add(
    parts: Vec<crate::data::repo::StacPart>,
    have: &std::collections::HashSet<String>,
    rect: [f64; 4],
    room: usize,
) -> Vec<crate::data::repo::StacPart> {
    let mut scored: Vec<(f64, crate::data::repo::StacPart)> = parts
        .into_iter()
        .filter(|p| !have.contains(&p.url))
        .filter_map(|p| {
            let a = overlap_area(p.bbox, rect);
            (a > 0.0).then_some((a, p))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(room);
    scored.into_iter().map(|(_, p)| p).collect()
}

/// A part opened for append: where it reads from, its short label, its
/// parsed footer, and the hive `key=value` segments of its path under
/// the collection (empty for a part addressed by its own Item).
type AppendPart = (Source, String, FileOpen, Vec<(String, Option<String>)>);

/// Open the part files of a STAC collection that the viewport wants and
/// the layer does not have yet, then build their geometry.
///
/// This is what makes a collection pannable. Parts are chosen at open from
/// the viewport, and without this the layer stayed frozen at that choice:
/// panning somewhere else showed nothing, and the collection could only
/// report the parts it was refusing to fetch.
///
/// Sends `PartsOpened` first (the store grows), then streams `Appended`
/// for the new row groups exactly as a row append does.
#[allow(clippy::too_many_arguments)]
pub fn spawn_part_append(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    // Viewport in WGS84 lon/lat: STAC item bboxes are WGS84 by spec.
    rect: [f64; 4],
    box_layer: bool,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
    refinement_budget: Option<(u64, u64)>,
) {
    use std::sync::atomic::Ordering;
    std::thread::spawn(move || {
        let ended = |error: &str| {
            handle.send(LoadMsg::AppendEnded {
                layer_id,
                error: error.into(),
            });
        };
        let Some(collection) = store.stac_collection().map(str::to_string) else {
            ended(NOTHING_TO_APPEND);
            return;
        };
        if store.fragments.len() >= PART_TOTAL_CAP {
            ended(NOTHING_TO_APPEND);
            return;
        }
        // Disk-cached after the first call, so panning does not re-read
        // the collection document.
        let parts = match crate::data::repo::fetch_stac_parts(&collection) {
            Ok(p) => p,
            Err(e) => return ended(&format!("{collection}: {e}")),
        };
        let have = store.part_urls();
        let room = PART_APPEND_PER_PASS.min(PART_TOTAL_CAP - store.fragments.len());
        let wanted = parts_to_add(parts, &have, rect, room);
        if wanted.is_empty() {
            ended(NOTHING_TO_APPEND);
            return;
        }
        if cancel.load(Ordering::Relaxed) {
            ended(CANCELLED);
            return;
        }

        // Resolve and read footers in parallel: each part is a round trip.
        let opened: Vec<AppendPart> = {
            use rayon::prelude::*;
            let r: Result<Vec<_>, String> = wanted
                .par_iter()
                .map(|p| {
                    let short = p
                        .rel
                        .clone()
                        .unwrap_or_else(|| p.url.rsplit('/').next().unwrap_or(&p.url).to_string());
                    let src = Source::Remote {
                        url: p.url.clone(),
                        len: 0,
                    }
                    .resolve()
                    .map_err(|e| format!("{short}: {e}"))?;
                    let f = open_file(&src).map_err(|e| format!("{short}: {e}"))?;
                    let hive = p
                        .rel
                        .as_deref()
                        .map(|rel| super::store::hive_segments(std::path::Path::new(rel)))
                        .unwrap_or_default();
                    Ok((src, short, f, hive))
                })
                .collect();
            match r {
                Ok(v) => v,
                Err(e) => return ended(&e),
            }
        };

        // A part that does not match the dataset is dropped, not fatal:
        // one odd item must not stop a collection from being pannable.
        let base = &store;
        let mut frags: Vec<(super::store::Fragment, Vec<u64>)> = Vec::new();
        let mut added_boxes: Vec<[f64; 4]> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut boxes_ok = true;
        for (src, short, f, hive) in opened {
            if !f.crs.same_as(&crs) || f.encoding != base.encoding || f.geom_col != base.geom_col {
                log::warn!("{short}: does not match the collection's schema; skipped");
                continue;
            }
            match (&f.rg_boxes, f.info.geo.bbox) {
                (Some((_, b)), _) => added_boxes.extend_from_slice(b),
                (None, Some(b)) => added_boxes.extend(std::iter::repeat_n(b, f.rg_rows.len())),
                (None, None) => boxes_ok = false,
            }
            // Partition values of a part panned into, keyed by the
            // columns the layer already has: leaving them null would make
            // the hive column lie about the parts that arrived last.
            let part_values: Vec<Option<String>> = base
                .part_cols
                .iter()
                .map(|c| {
                    hive.iter()
                        .find(|(k, _)| k == c)
                        .and_then(|(_, v)| v.clone())
                })
                .collect();
            frags.push((
                super::store::Fragment {
                    source: src,
                    meta: f.meta.clone(),
                    part_values,
                    rg_offset: 0,
                    row_offset: 0,
                },
                f.rg_rows.clone(),
            ));
            names.push(short);
        }
        if frags.is_empty() || !boxes_ok {
            ended(NOTHING_TO_APPEND);
            return;
        }
        let first_new = base.rg_starts().len() - 1;
        let added_groups: usize = frags.iter().map(|(_, r)| r.len()).sum();
        let grown = Arc::new(base.with_fragments_appended(frags));
        if grown.total_rows() >= u32::MAX as u64 {
            ended("collection exceeds the maximum supported row count");
            return;
        }
        log::info!(
            "{}: +{} part(s), {} row groups",
            grown.source.label(),
            names.len(),
            added_groups
        );
        handle.send(LoadMsg::PartsOpened {
            layer_id,
            generation,
            store: grown.clone(),
            added_boxes,
            added_groups,
            names,
        });

        // The new groups only. Boxes when the layer draws boxes, otherwise
        // the same viewport-bounded selection a refinement would make.
        let jobs: Vec<GroupSel> = (first_new..first_new + added_groups)
            .map(|g| {
                if box_layer {
                    GroupSel::Boxes {
                        group: g as u32,
                        rect: None,
                    }
                } else {
                    GroupSel::All(g as u32)
                }
            })
            .collect();
        run_append(
            &handle, layer_id, generation, &grown, &crs, &display, jobs, &cancel, style,
            refinement_budget,
        );
    });
}

/// Streamed-append batch sizes: the first batch lands fast so the user
/// sees refinement content almost immediately; later batches amortize
/// per-batch overhead.
const APPEND_FIRST_BATCH_ROWS: u64 = 80_000;
const APPEND_BATCH_ROWS: u64 = 250_000;

/// Split resolved append jobs into row-bounded batches, order preserved.
/// Splitting never goes below job (row group) granularity.
fn append_batches(store: &FeatureStore, jobs: Vec<GroupSel>) -> Vec<Vec<GroupSel>> {
    append_batches_with(store, jobs, APPEND_FIRST_BATCH_ROWS, APPEND_BATCH_ROWS)
}

fn append_batches_with(
    store: &FeatureStore,
    jobs: Vec<GroupSel>,
    first_target: u64,
    later_target: u64,
) -> Vec<Vec<GroupSel>> {
    let starts = store.rg_starts();
    let rows_of = |j: &GroupSel| -> u64 {
        let g = j.group() as usize;
        let group_rows = starts[g + 1] - starts[g];
        match j {
            GroupSel::Ranges(_, r) | GroupSel::ResolvedRect { ranges: r, .. } => {
                r.iter().map(|&(s, e)| (e - s) as u64).sum()
            }
            GroupSel::Preview { stride, .. } => group_rows.div_ceil((*stride).max(2) as u64),
            _ => group_rows,
        }
    };
    let mut out: Vec<Vec<GroupSel>> = Vec::new();
    let mut cur: Vec<GroupSel> = Vec::new();
    let mut cur_rows = 0u64;
    for j in jobs {
        cur_rows += rows_of(&j);
        cur.push(j);
        let target = if out.is_empty() { first_target } else { later_target };
        if cur_rows >= target {
            out.push(std::mem::take(&mut cur));
            cur_rows = 0;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

enum RefinePlan {
    Ready(Vec<GroupSel>),
    /// Over budget: the rows the selection had reached, and the geometry
    /// bytes they carry when bytes are what stopped it. A land-cover
    /// viewport blows the byte budget at a row count that reads as
    /// trivial, so saying "too many rows" would name the wrong thing.
    Deferred { rows: u64, geom_bytes: Option<u64> },
}

/// Resolve viewport rects and enforce the refinement budget using the exact
/// number of selected features. The previous area-ratio estimate could stay
/// above the budget indefinitely for clustered or overlapping row groups,
/// leaving the every-Nth-row preview visible even at street-level zooms.
fn prepare_refinement_jobs(
    store: &FeatureStore,
    jobs: Vec<GroupSel>,
    budget: u64,
    geom_budget: u64,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<RefinePlan, String> {
    let starts = store.rg_starts();
    let mut rows = 0u64;
    // Geometry bytes alongside rows, for the same reason the initial plan
    // counts them: CORINE is 2.38M rows, comfortably under the row
    // budget, and 7.7 GB of geometry. Refining it on a row count alone
    // decodes the whole file the moment the camera settles, whatever the
    // initial load carefully avoided.
    let mut geom_bytes = 0u64;
    let mut resolved = Vec::with_capacity(jobs.len());

    // Resolve every rect selection first, in parallel. Each one is a
    // read of that group's bbox column, which on a remote source is a
    // round trip: twenty groups resolved one after another put several
    // seconds between the camera settling and the first byte of geometry
    // being asked for. The budget below still walks them in order, so
    // which groups make the cut does not depend on who answered first.
    /// A group's resolved row ranges, or None when the store has no
    /// selector and the whole group must be read.
    type GroupRanges = Option<Vec<(u32, u32)>>;
    let resolved_rects: std::collections::HashMap<u32, GroupRanges> = {
        use rayon::prelude::*;
        let rects: Vec<(u32, [f64; 4])> = jobs
            .iter()
            .filter_map(|j| match j {
                GroupSel::Rect(g, r) => Some((*g, *r)),
                _ => None,
            })
            .collect();
        let out: Vec<Result<(u32, GroupRanges), String>> = rects
            .into_par_iter()
            .map(|(g, r)| {
                if cancel.load(Ordering::Relaxed) {
                    return Err(CANCELLED.to_string());
                }
                covering_select(store, g, r).map(|ranges| (g, ranges))
            })
            .collect();
        let mut map = std::collections::HashMap::new();
        for r in out {
            let (g, ranges) = r?;
            map.insert(g, ranges);
        }
        map
    };

    for job in jobs {
        if cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        let group = job.group();
        let group_rows = starts[group as usize + 1] - starts[group as usize];
        let (count, job) = match job {
            GroupSel::Rect(group, rect) => match resolved_rects
                .get(&group)
                .cloned()
                .unwrap_or_else(|| covering_select(store, group, rect).ok().flatten())
            {
                Some(ranges) => {
                    let count = ranges.iter().map(|&(s, e)| (e - s) as u64).sum();
                    if ranges.as_slice() == [(0, group_rows as u32)] {
                        (count, GroupSel::All(group))
                    } else {
                        (
                            count,
                            GroupSel::ResolvedRect {
                                group,
                                rect,
                                ranges,
                            },
                        )
                    }
                }
                // No covering/x-y selector: a rect decode reads the whole
                // group, so count and represent it honestly.
                None => (group_rows, GroupSel::All(group)),
            },
            GroupSel::ResolvedRect {
                group,
                rect,
                ranges,
            } => {
                let count = ranges.iter().map(|&(s, e)| (e - s) as u64).sum();
                (
                    count,
                    GroupSel::ResolvedRect {
                        group,
                        rect,
                        ranges,
                    },
                )
            }
            GroupSel::Ranges(group, ranges) => {
                let count = ranges.iter().map(|&(s, e)| (e - s) as u64).sum();
                (count, GroupSel::Ranges(group, ranges))
            }
            GroupSel::All(group) => (group_rows, GroupSel::All(group)),
            preview @ GroupSel::Preview { stride, .. } => {
                (group_rows.div_ceil(stride.max(2) as u64), preview)
            }
            // Boxes read four doubles a row and tessellate two triangles;
            // against a refine budget meant for geometry, that is not
            // work worth counting.
            boxes @ (GroupSel::Boxes { .. } | GroupSel::BoxRanges { .. }) => (0, boxes),
        };
        rows = rows.saturating_add(count);
        // The group's bytes in proportion to the rows this job selects.
        geom_bytes = geom_bytes.saturating_add(
            store
                .rg_geom_bytes(group)
                .saturating_mul(count)
                .checked_div(group_rows.max(1))
                .unwrap_or(0),
        );
        if rows > budget || geom_bytes > geom_budget {
            return Ok(RefinePlan::Deferred {
                rows,
                geom_bytes: (geom_bytes > geom_budget).then_some(geom_bytes),
            });
        }
        resolved.push(job);
    }
    Ok(RefinePlan::Ready(resolved))
}

/// Parse file metadata: geometry column, CRS, row-group layout. Reads no data.
pub type StoreOpen = (
    FeatureStore,
    Crs,
    FileInfo,
    Option<(String, Vec<[f64; 4]>)>,
);

/// One file's parsed metadata (no data read).
struct FileOpen {
    meta: ArrowReaderMetadata,
    schema: arrow::datatypes::SchemaRef,
    crs: Crs,
    encoding: GeomEncoding,
    geom_col: usize,
    covering: Option<CoveringCol>,
    rg_rows: Vec<u64>,
    rg_boxes: Option<(String, Vec<[f64; 4]>)>,
    info: FileInfo,
    /// Point geometry synthesized from these coordinate columns (x, y)
    /// when the file has no geometry column at all.
    xy: Option<(usize, usize)>,
    /// Every column chunk carries a page index.
    page_index: bool,
    /// Uncompressed bytes of the geometry leaves (decode-size proxy).
    geom_bytes: u64,
    /// WKB primary superseded by a GeoArrow sibling (`geom_col` points at
    /// the sibling; this is the primary's index, hidden from attribute UIs).
    hidden_wkb: Option<usize>,
    /// `edges: spherical` metadata, or a GEOGRAPHY logical type.
    spherical_edges: bool,
    /// The geometry column holds base64 text, not WKB bytes: `schema`
    /// already says Binary and the store decodes on read.
    base64_wkb: bool,
    /// Parsed `cogp` block, or why it was rejected. A rejected block
    /// leaves the file perfectly openable — it is simply not COGP.
    cogp: Option<Result<super::cogp::CogpLevels, String>>,
}

/// Quality analysis over an opened file / merged dataset (footer facts
/// only). `boxes` are the merged per-row-group bboxes.
#[allow(clippy::too_many_arguments)]
fn quality_report(
    info: &FileInfo,
    boxes: Option<&(String, Vec<[f64; 4]>)>,
    encoding: GeomEncoding,
    xy_synthesized: bool,
    page_index: bool,
    geom_bytes: u64,
    cogp: Option<Result<super::quality::CogpQuality, String>>,
) -> super::quality::QualityReport {
    super::quality::analyze(&super::quality::QualityInput {
        rows: info.rows,
        row_groups: info.row_groups,
        rg_rows_max: info.rg_rows_max,
        rg_bytes_max: info.rg_bytes_max,
        boxes: boxes.map(|(s, b)| (s.as_str(), b.as_slice())),
        encoding,
        xy_synthesized,
        page_index,
        geom_compression: info
            .columns
            .iter()
            .find(|c| c.is_geometry)
            .map(|c| c.compression.as_str()),
        geo: &info.geo,
        geom_bytes,
        cogp,
    })
}

/// Load planning shared by first loads and viewport reloads: which row
/// groups intersect the viewport and how each is read — whole, per-feature
/// rect selection (covering/xy), or a decimated preview when the candidate
/// rows exceed the build budget.
fn plan_viewport_selection(
    store: &FeatureStore,
    label: &str,
    boxes: Option<&[[f64; 4]]>,
    rect: Option<[f64; 4]>,
    cogp_end: Option<u32>,
    caller: &str,
) -> Vec<GroupSel> {
    log::debug!("plan_viewport_selection from {caller}");
    let n_rg = store.rg_starts().len().saturating_sub(1);
    if n_rg == 0 {
        return Vec::new();
    }
    // COGP levels first, before any budget fallback: the prefix is the
    // set of features that are independently meaningful at this scale,
    // so a prefix that fits the budget is an exact load. Falling back to
    // boxes or a stride over the whole file would trade real geometry
    // for approximated geometry the user then sees badged as such.
    let last_group = cogp_end.unwrap_or(u32::MAX).min(n_rg as u32 - 1);
    if let Some(levels) = store.cogp.as_ref().filter(|_| cogp_end.is_some()) {
        log::info!(
            "{label}: COGP level ending at row group {last_group} of {n_rg} \
             ({} levels available)",
            levels.levels.len()
        );
    }
    // A dataset that will be drawn from covering boxes loads all of them,
    // whatever the camera is looking at. Boxes are four doubles a feature
    // and no geometry read at all, and pruning them to the opening
    // viewport is how a layer opened while the camera sits on another
    // continent ends up with nothing: every group outside the viewport
    // stays unloaded, and zooming towards it finds no boxes to show.
    // Complete coverage from the first frame is the property that makes
    // this display mode worth having.
    let prefix_rows = store.rg_starts()[last_group as usize + 1];
    let all_boxes = matches!(
        over_budget_plan(store.covering.is_some(), store.polygons_only),
        OverBudget::Boxes
    ) && preview_stride(
        prefix_rows,
        (0..=last_group).map(|g| store.rg_geom_bytes(g)).sum(),
    )
    .is_some();
    if all_boxes {
        log::info!(
            "{label}: {prefix_rows} rows of polygons with a covering column: \
             drawing every feature from its bounding box"
        );
        return (0..=last_group)
            .map(|group| GroupSel::Boxes { group, rect: None })
            .collect();
    }
    let mut groups: Vec<u32> = match (boxes, rect) {
        (Some(b), Some(r)) => intersecting_rgs(b, r),
        _ => (0..=last_group).collect(),
    };
    groups.retain(|&g| g <= last_group);
    if groups.len() < n_rg {
        log::info!("{label}: row-group pruning {n_rg} -> {} groups", groups.len());
    }
    // Per-feature covering selection: only when the viewport doesn't
    // already cover the whole data extent.
    let use_rect = (store.covering.is_some()
        || store.xy_geom.is_some()
        || store.encoding.is_wkb())
        && match (boxes, rect) {
            (Some(bs), Some(r)) => {
                let mut u = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for b in bs {
                    u = [u[0].min(b[0]), u[1].min(b[1]), u[2].max(b[2]), u[3].max(b[3])];
                }
                !(r[0] <= u[0] && r[1] <= u[1] && r[2] >= u[2] && r[3] >= u[3])
            }
            _ => false,
        };
    let mut sel: Vec<GroupSel> = groups
        .iter()
        .map(|&g| {
            if use_rect {
                GroupSel::Rect(g, rect.unwrap())
            } else {
                GroupSel::All(g)
            }
        })
        .collect();
    // Row budget: a selection that could decode more rows than the budget
    // becomes a decimated preview (an upper bound — rect selections may
    // resolve smaller, but a preview that refines on zoom beats an
    // out-of-memory tessellation).
    let est: u64 = groups
        .iter()
        .map(|&g| store.rg_starts()[g as usize + 1] - store.rg_starts()[g as usize])
        .sum();
    let bytes: u64 = groups.iter().map(|&g| store.rg_geom_bytes(g)).sum();
    if let Some(stride) = preview_stride(est, bytes) {
        let boxes_ok = matches!(
            over_budget_plan(store.covering.is_some(), store.polygons_only),
            OverBudget::Boxes
        );
        if boxes_ok {
            log::info!(
                "{label}: {est} candidate rows / {} of geometry exceed the budget — \
                 drawing all features from their bounding boxes",
                crate::data::info::fmt_bytes(bytes)
            );
        } else {
            log::info!(
                "{label}: {est} candidate rows / {} of geometry exceed the budget — \
                 preview at 1/{stride}",
                crate::data::info::fmt_bytes(bytes)
            );
        }
        sel = sel
            .into_iter()
            .map(|s| match (s, boxes_ok) {
                (GroupSel::All(g), true) => GroupSel::Boxes { group: g, rect: None },
                (GroupSel::Rect(g, r), true) => GroupSel::Boxes { group: g, rect: Some(r) },
                (GroupSel::All(g), false) => {
                    GroupSel::Preview { group: g, rect: None, stride }
                }
                (GroupSel::Rect(g, r), false) => {
                    GroupSel::Preview { group: g, rect: Some(r), stride }
                }
                (other, _) => other,
            })
            .collect();
    }
    sel
}

/// Drop-and-reload: run the initial-load planning against the current
/// viewport on an already open store and rebuild the layer from scratch —
/// the memory-relief counterpart of refinement. Everything outside the
/// view is simply not decoded.
#[allow(clippy::too_many_arguments)]
pub fn spawn_reload(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    boxes: Option<Vec<[f64; 4]>>,
    display: DisplayCrs,
    view_world: [f64; 4],
    view_px: f64,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let n_rg = store.rg_starts().len().saturating_sub(1);
        let rect = viewport_to_data_bbox(view_world, &display, &crs);
        log::debug!("spawn_reload: layer {layer_id}, generation {generation}");
        let sel = plan_viewport_selection(
            &store,
            &store.source.label(),
            boxes.as_deref(),
            rect,
            cogp_prefix_end(&store, rect, view_px, &crs),
            "viewport reload",
        );
        let style_sel = style.as_ref().and_then(|sb| resolve_style(&store, sb));
        match build_geometry(
            &store,
            &crs,
            &display,
            None,
            sel,
            Some(&cancel),
            style_sel.as_ref(),
        ) {
            Ok((geometry, rows, bad, _boxes, resolved)) => {
                let mut loaded = vec![GroupLoad::None; n_rg];
                for (g, st) in resolved {
                    loaded[g as usize] = st;
                }
                log::info!(
                    "reloaded {}: {rows} features in {} ms",
                    store.source.name(),
                    t0.elapsed().as_millis()
                );
                handle.send(LoadMsg::Reloaded {
                    layer_id,
                    generation,
                    geometry,
                    loaded,
                    rows,
                    bad_geoms: bad,
                    build_ms: t0.elapsed().as_millis() as u64,
                });
            }
            Err(error) => handle.send(LoadMsg::RebuildFailed {
                layer_id,
                generation,
                error,
            }),
        }
    });
}

// ---------------------------------------------------------------------
// H3 pyramid: detection, level choice, and opening one level's parts.
// Layout and descriptor: `super::pyramid`; design notes:
// `_WIKI/concepts/h3-pyramid.md`.
// ---------------------------------------------------------------------

/// The viewport as the store opener sees it, before any data CRS is
/// known: WGS84 lon/lat plus its width in physical pixels.
///
/// STAC part pruning only ever needed the rect. A pyramid needs the
/// ground scale as well, because the level it reads is a function of
/// metres per pixel, not of where the camera is.
#[derive(Clone, Copy, Debug)]
pub struct ViewHint {
    pub rect: [f64; 4],
    pub view_px: f64,
}

impl ViewHint {
    /// Metres of ground per pixel at the view centre.
    fn gsd(&self) -> Option<f64> {
        view_gsd(self.rect, self.view_px, &Crs::wgs84())
    }
}

/// Where a pyramid's files live: the three prefixes a dataset opens
/// from, each able to read one small document and address one part.
enum PyramidRoot {
    Dir(std::path::PathBuf),
    S3 {
        bucket: String,
        prefix: String,
        profile: Option<String>,
        endpoint: Option<String>,
    },
    /// An HTTPS prefix, with its trailing slash.
    Https(String),
}

/// The root a source's parts hang off, when the source is a prefix that
/// could hold a pyramid. A single file cannot, and neither can a glob or
/// a fixed part list: both name files, not a root.
fn pyramid_root(source: &Source) -> Option<PyramidRoot> {
    match source {
        Source::Dir(dir) => Some(PyramidRoot::Dir(dir.clone())),
        Source::S3 { uri, profile, endpoint, .. } if source.is_s3_prefix() => {
            let rest = uri.strip_prefix("s3://").unwrap_or(uri);
            if rest.contains('*') {
                return None;
            }
            let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
            Some(PyramidRoot::S3 {
                bucket: bucket.to_string(),
                prefix: prefix.to_string(),
                profile: profile.clone(),
                endpoint: endpoint.clone(),
            })
        }
        // An https prefix is routed as its `collection.json`; the
        // pyramid sits beside that document, and takes precedence over
        // it when both are there (the writer leaves both).
        Source::Stac { url, .. } => url
            .strip_suffix("collection.json")
            .map(|root| PyramidRoot::Https(root.to_string())),
        _ => None,
    }
}

/// GET one small document; Ok(None) on 404.
fn http_text(url: &str) -> Result<Option<String>, String> {
    let res = crate::data::source::http_agent()
        .get(url)
        .call()
        .map_err(|e| format!("cannot reach {url}: {e}"))?;
    match res.status().as_u16() {
        200 => res
            .into_body()
            .read_to_string()
            .map(Some)
            .map_err(|e| format!("read {url}: {e}")),
        404 => Ok(None),
        s => Err(format!("{url}: HTTP {s}")),
    }
}

impl PyramidRoot {
    fn label(&self) -> String {
        match self {
            PyramidRoot::Dir(d) => d.display().to_string(),
            PyramidRoot::S3 { bucket, prefix, .. } => format!("s3://{bucket}/{prefix}"),
            PyramidRoot::Https(u) => u.clone(),
        }
    }

    /// `h3-pyramid.json`, or None when there is none to read.
    fn read_descriptor(&self) -> Result<Option<String>, String> {
        match self {
            PyramidRoot::Dir(d) => match std::fs::read_to_string(d.join(pyramid::DESCRIPTOR)) {
                Ok(s) => Ok(Some(s)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(format!("{}: {e}", pyramid::DESCRIPTOR)),
            },
            PyramidRoot::S3 { bucket, prefix, profile, endpoint } => {
                crate::data::source::aws::fetch_small(
                    &format!("s3://{bucket}/{prefix}{}", pyramid::DESCRIPTOR),
                    profile.as_deref(),
                    endpoint.as_deref(),
                )
            }
            PyramidRoot::Https(root) => http_text(&format!("{root}{}", pyramid::DESCRIPTOR)),
        }
    }

    /// The source of one part, by its path relative to the root.
    fn child(&self, rel: &str) -> Result<Source, String> {
        match self {
            PyramidRoot::Dir(d) => Ok(Source::Local(d.join(rel))),
            PyramidRoot::S3 { bucket, prefix, profile, endpoint } => Source::S3 {
                uri: format!("s3://{bucket}/{prefix}{rel}"),
                profile: profile.clone(),
                endpoint: endpoint.clone(),
                url: String::new(),
                len: 0,
            }
            .resolve(),
            PyramidRoot::Https(root) => {
                Source::Remote { url: format!("{root}{rel}"), len: 0 }.resolve()
            }
        }
    }

    /// Parquet files under the root, relative to it — one listing, used
    /// both to drop cells whose file is not there and to answer C9.
    /// None when the root cannot be listed: an HTTPS prefix serves no
    /// index, so the descriptor is taken at its word.
    fn list(&self) -> Option<std::collections::HashSet<String>> {
        match self {
            // The pyramid's own layout, not the generic dataset walk:
            // the null part is spelled `__HIVE_DEFAULT_PARTITION__` and
            // `list_dataset_files` skips leading underscores as sidecar
            // names. Here that name is data.
            PyramidRoot::Dir(d) => {
                let mut out = std::collections::HashSet::new();
                for level in std::fs::read_dir(d).ok()?.filter_map(Result::ok) {
                    let dir = level.file_name().to_string_lossy().into_owned();
                    if !dir.starts_with('r') || !dir[1..].chars().all(|c| c.is_ascii_digit()) {
                        continue;
                    }
                    for e in std::fs::read_dir(level.path()).ok()?.filter_map(Result::ok) {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if name.ends_with(".parquet") {
                            out.insert(format!("{dir}/{name}"));
                        }
                    }
                }
                Some(out)
            }
            PyramidRoot::S3 { bucket, prefix, profile, endpoint } => {
                let listed = crate::data::source::aws::list_prefix(
                    &format!("s3://{bucket}/{prefix}"),
                    profile.as_deref(),
                    endpoint.as_deref(),
                )
                .ok()?;
                Some(
                    listed
                        .iter()
                        .filter_map(|(k, _)| k.strip_prefix(prefix.as_str()))
                        .map(str::to_string)
                        .collect(),
                )
            }
            PyramidRoot::Https(_) => None,
        }
    }
}

/// Look for a pyramid at a dataset root: one small read, before any
/// parquet footer.
///
/// A descriptor that does not parse or validate is not fatal. The tree
/// underneath is still ordinary parquet and still opens as a plain
/// partitioned dataset; the error travels to C9 instead, where the user
/// can see it.
fn find_pyramid(root: &PyramidRoot) -> Option<Result<PyramidState, String>> {
    match root.read_descriptor() {
        Ok(None) => None,
        Ok(Some(text)) => Some(
            pyramid::Descriptor::parse(&text)
                .and_then(|d| PyramidState::new(d, root.label())),
        ),
        Err(e) => Some(Err(e)),
    }
}

/// Hive value of a part: its file name is the cell id, and the null part
/// keeps hive's own spelling for "no value".
fn pyramid_hive(rel: &str) -> Vec<(String, Option<String>)> {
    let stem = rel
        .rsplit('/')
        .next()
        .and_then(|f| f.strip_suffix(".parquet"))
        .unwrap_or_default();
    let value = (stem != pyramid::NULL_PART).then(|| stem.to_string());
    vec![(pyramid::CELL_COLUMN.to_string(), value)]
}

/// Open exactly the parts one level of a pyramid needs for a viewport.
///
/// Never a mix of levels: an overview holds derived features and the
/// leaf holds the source's own, and drawing both would double the map.
/// Each part's cell id becomes the virtual `h3` partition column, so
/// picking, the attribute table and SQL can all say which cell a row
/// came from without the file name being parsed again downstream.
fn open_pyramid_store(
    source: &Source,
    root: &PyramidRoot,
    state: &PyramidState,
    view: Option<ViewHint>,
    listing: Option<&std::collections::HashSet<String>>,
) -> Result<StoreOpen, String> {
    let mut plan = state.plan(
        view.and_then(|v| v.gsd()),
        view.map(|v| v.rect),
        pyramid::MAX_LEVEL_PARTS,
    );
    // The descriptor lists cells, not files. Where the root can be
    // listed we believe the listing: a cell whose file is missing would
    // otherwise fail the whole open, and C9 reports the same gap.
    if let Some(have) = listing {
        plan.parts.retain(|p| have.contains(p));
    }
    // The null part draws nothing, so it stays out of viewport plans;
    // SQL and the scorecard still count it (see `leaf_parts`).
    if plan.parts.is_empty() {
        return Err(format!(
            "no cells of this pyramid are inside the current view (level r{})",
            plan.res
        ));
    }
    if let Some(from) = plan.coarsened_from {
        log::info!(
            "{}: level r{from} would open more than {} parts here; reading r{} instead",
            root.label(),
            pyramid::MAX_LEVEL_PARTS,
            plan.res
        );
    }
    log::info!(
        "{}: H3 pyramid level r{}, {} part file(s)",
        root.label(),
        plan.res,
        plan.parts.len()
    );
    let files: Vec<(Source, String)> = {
        use rayon::prelude::*;
        plan.parts
            .par_iter()
            .map(|rel| {
                root.child(rel)
                    .map(|src| (src, rel.clone()))
                    .map_err(|e| format!("{rel}: {e}"))
            })
            .collect::<Result<_, _>>()?
    };
    let hive: Vec<Vec<(String, Option<String>)>> =
        plan.parts.iter().map(|rel| pyramid_hive(rel)).collect();
    let mut opened = open_multi_store(source, files, hive)?;
    opened.2.pyramid = Some(state.info_line());
    opened.0.pyramid = Some(state.with_active(plan.res));
    Ok(opened)
}

/// C9's facts: what the descriptor says, and whether the files it names
/// are there.
fn pyramid_quality(
    found: Option<&Result<PyramidState, String>>,
    listing: Option<&std::collections::HashSet<String>>,
) -> Option<Result<super::quality::PyramidQuality, String>> {
    match found? {
        Err(e) => Some(Err(e.clone())),
        Ok(state) => {
            let listed = state.all_parts();
            let missing: Vec<String> = match listing {
                Some(have) => listed.iter().filter(|p| !have.contains(*p)).cloned().collect(),
                None => Vec::new(),
            };
            Some(Ok(super::quality::PyramidQuality {
                summary: state.info_line(),
                listed: listed.len(),
                missing,
                unlisted: listing.is_none(),
            }))
        }
    }
}

/// Un-gated open with no viewport: what the in-crate tests and the
/// `geopq-cli` integration tests read a written file back with. The app
/// itself always goes through `open_store_with_view`.
pub fn open_store(source: &Source) -> Result<StoreOpen, String> {
    open_store_with_view(source, None)
}

/// Open a dataset for a viewport.
///
/// An H3 pyramid is looked for first, because it changes what "open"
/// means: one level's cells instead of every file under the root. The
/// look costs one small read of `h3-pyramid.json`, and a descriptor that
/// does not validate simply falls through to the plain multi-part path —
/// the tree underneath is still ordinary parquet — with the error
/// carried to the scorecard as C9.
fn open_store_with_view(
    source: &Source,
    view: Option<ViewHint>,
) -> Result<StoreOpen, String> {
    let root = pyramid_root(source);
    let found = root.as_ref().and_then(find_pyramid);
    // One listing, shared by the plan (cells whose file is absent) and
    // by C9 (which files the descriptor promised).
    let listing = found
        .as_ref()
        .is_some_and(Result::is_ok)
        .then(|| root.as_ref().and_then(PyramidRoot::list))
        .flatten();
    let mut opened = match (&root, &found) {
        (Some(r), Some(Ok(state))) => {
            open_pyramid_store(source, r, state, view, listing.as_ref())?
        }
        _ => {
            if let Some(Err(e)) = &found {
                log::warn!("{}: {e}; opening as a plain partitioned dataset", source.label());
            }
            open_plain_store(source, view.map(|v| v.rect))?
        }
    };
    if let Some(q) = opened.2.quality.as_mut() {
        q.checks
            .push(super::quality::pyramid_check(pyramid_quality(found.as_ref(), listing.as_ref()).as_ref()));
    }
    Ok(opened)
}

/// `stac_rect`: current viewport in WGS84 lon/lat, for part-level pruning
/// of STAC collections (their item bboxes are WGS84 by spec).
fn open_plain_store(
    source: &Source,
    stac_rect: Option<[f64; 4]>,
) -> Result<StoreOpen, String> {
    if let Source::Dir(dir) = source {
        return open_dir_store(source, dir);
    }
    if source.is_s3_prefix() {
        return open_s3_prefix_store(source);
    }
    if let Source::Multi { urls, .. } = source {
        return open_multi_remote_store(source, urls);
    }
    if let Source::Stac { url, .. } = source {
        return open_stac_store(source, url, stac_rect);
    }
    let mut f = open_file(source)?;
    let cogp_quality = f.cogp.as_ref().map(|r| {
        r.as_ref()
            .map(|c| super::quality::CogpQuality {
                version: c.version.clone(),
                levels: c.runs(&f.rg_rows),
                total_rows: f.info.rows,
                pruning: c.pruning.label(),
                extension_2_0: c.pruning == super::cogp::Pruning::NativeStats,
            })
            .map_err(String::clone)
    });
    f.info.quality = Some(quality_report(
        &f.info,
        f.rg_boxes.as_ref(),
        f.encoding,
        f.xy.is_some(),
        f.page_index,
        f.geom_bytes,
        cogp_quality,
    ));
    let mut store = FeatureStore::new(
        source.clone(),
        f.meta,
        f.geom_col,
        f.schema,
        f.covering,
        f.encoding,
        f.rg_rows,
        f.xy,
    );
    store.hidden_wkb = f.hidden_wkb;
    store.spherical_edges = f.spherical_edges;
    store.base64_wkb = f.base64_wkb;
    store.cogp = f.cogp.clone().and_then(Result::ok);
    // Declared types only: an undeclared file stays out of the bbox
    // path rather than guessing from the first feature it decodes.
    store.polygons_only = !f.info.geo.geometry_types.is_empty()
        && f.info
            .geo
            .geometry_types
            .iter()
            .all(|t| t.trim_end_matches(" Z").ends_with("Polygon"));
    Ok((store, f.crs, f.info, f.rg_boxes))
}

/// All parquet files under a dataset directory, in stable path order.
/// Hidden and sidecar entries (`.`/`_` prefixes, e.g. `_metadata`,
/// `_SUCCESS`) are skipped.
fn list_dataset_files(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        entries.sort();
        for p in entries {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if name.starts_with('.') || name.starts_with('_') {
                continue;
            }
            if p.is_dir() {
                walk(&p, out)?;
            } else if matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("parquet" | "geoparquet" | "pq")
            ) {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    Ok(out)
}

/// Open a directory of (possibly hive-partitioned) GeoParquet files as one
/// multi-fragment store. Hive `key=value` path segments become virtual
/// partition columns.
fn open_dir_store(source: &Source, dir: &std::path::Path) -> Result<StoreOpen, String> {
    use super::store::hive_segments;

    let paths = list_dataset_files(dir)?;
    if paths.is_empty() {
        return Err(format!("no .parquet files under {}", dir.display()));
    }
    let hive: Vec<Vec<(String, Option<String>)>> = paths
        .iter()
        .map(|p| hive_segments(p.strip_prefix(dir).unwrap_or(p)))
        .collect();
    let files: Vec<(Source, String)> = paths
        .into_iter()
        .map(|p| {
            let short = p
                .strip_prefix(dir)
                .unwrap_or(&p)
                .display()
                .to_string();
            (Source::Local(p), short)
        })
        .collect();
    open_multi_store(source, files, hive)
}

/// Part files a STAC load opens up front: each costs a content-length
/// probe plus a footer fetch, so a world view of a 512-part collection
/// opens the parts covering most of it and picks the rest up while
/// panning rather than stalling on half a terabyte of metadata.
pub const STAC_PART_CAP: usize = 16;

/// Area of the intersection of a part's bbox with the viewport, 0 when
/// they miss or the part has no bbox.
fn overlap_area(bbox: Option<[f64; 4]>, r: [f64; 4]) -> f64 {
    let Some(b) = bbox else { return 0.0 };
    let w = (b[2].min(r[2]) - b[0].max(r[0])).max(0.0);
    let h = (b[3].min(r[3]) - b[1].max(r[1])).max(0.0);
    w * h
}

/// Open a fixed set of remote parquet parts (repository "all states"
/// loads) as one multi-fragment layer. Hive `key=value` URL path
/// segments become partition columns. Parts whose probe fails are
/// dropped with a warning — a theme absent from one region (404) must
/// not sink the other 50 — but schema mismatches among the surviving
/// parts still fail the open loudly.
fn open_multi_remote_store(source: &Source, urls: &[String]) -> Result<StoreOpen, String> {
    use super::store::hive_segments;

    if urls.len() > PREFIX_PART_CAP {
        return Err(format!(
            "{} part files (cap {PREFIX_PART_CAP} per load)",
            urls.len()
        ));
    }
    // URL path portion, for hive segments and short labels.
    let url_path = |u: &str| -> String {
        let rest = u.split_once("://").map(|(_, r)| r).unwrap_or(u);
        rest.split_once('/').map(|(_, p)| p.to_string()).unwrap_or_default()
    };
    let resolved: Vec<(String, Result<Source, String>)> = {
        use rayon::prelude::*;
        urls.par_iter()
            .map(|u| (u.clone(), Source::Remote { url: u.clone(), len: 0 }.resolve()))
            .collect()
    };
    let mut files = Vec::new();
    let mut hive = Vec::new();
    let mut dropped = 0usize;
    for (u, r) in resolved {
        let path = url_path(&u);
        let short = path
            .split('/')
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        match r {
            Ok(src) => {
                hive.push(hive_segments(std::path::Path::new(&path)));
                files.push((src, short));
            }
            Err(e) => {
                dropped += 1;
                log::warn!("{u}: {e} (part skipped)");
            }
        }
    }
    if files.is_empty() {
        return Err(format!(
            "none of the {} parts could be opened (see log)",
            urls.len()
        ));
    }
    if dropped > 0 {
        log::info!(
            "{}: {dropped} of {} parts unavailable, loading {}",
            source.label(),
            urls.len(),
            files.len()
        );
    }
    open_multi_store(source, files, hive)
}

/// Cap on part files an S3 prefix dataset opens. Higher than the STAC
/// cap: hive layouts legitimately fan out (50 US states and then some),
/// footers open in parallel, and prefixes/globs are how users narrow
/// further — but still bounded: every part costs a footer round-trip.
pub const PREFIX_PART_CAP: usize = 64;

/// Match one path segment against a `*` glob (no `/` crossing).
fn seg_match(pat: &str, s: &str) -> bool {
    let (pb, sb) = (pat.as_bytes(), s.as_bytes());
    let (mut pi, mut si) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while si < sb.len() {
        if pi < pb.len() && (pb[pi] == sb[si]) {
            pi += 1;
            si += 1;
        } else if pi < pb.len() && pb[pi] == b'*' {
            star = Some(pi);
            mark = si;
            pi += 1;
        } else if let Some(st) = star {
            pi = st + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    while pi < pb.len() && pb[pi] == b'*' {
        pi += 1;
    }
    pi == pb.len()
}

/// Match a full key against a glob pattern, segment by segment: `*`
/// matches within one path segment, never across `/`.
fn glob_match(pat: &str, key: &str) -> bool {
    let ps: Vec<&str> = pat.split('/').collect();
    let ks: Vec<&str> = key.split('/').collect();
    ps.len() == ks.len() && ps.iter().zip(&ks).all(|(p, k)| seg_match(p, k))
}

/// Open `s3://bucket/prefix/` (or a `*` glob like
/// `s3://bucket/d/state=*/roads.parquet`) as one multi-fragment remote
/// dataset: list the objects under the literal prefix, keep the
/// matching parquet parts, and turn hive `key=value` path segments
/// into virtual partition columns — the remote twin of
/// `open_dir_store`.
fn open_s3_prefix_store(source: &Source) -> Result<StoreOpen, String> {
    use super::store::hive_segments;

    let Source::S3 { uri, profile, endpoint, .. } = source else {
        return Err("not an S3 prefix".into());
    };
    let rest = uri.strip_prefix("s3://").unwrap_or(uri);
    let (bucket, keypat) = rest.split_once('/').unwrap_or((rest, ""));
    // With a glob, list the literal prefix before the first `*` (cut
    // at the last `/` so the listing prefix is a whole path).
    let (prefix, glob) = match keypat.find('*') {
        Some(star) => {
            let lit = &keypat[..star];
            let cut = lit.rfind('/').map(|i| i + 1).unwrap_or(0);
            (&keypat[..cut], Some(keypat))
        }
        None => (keypat, None),
    };
    let list_uri = format!("s3://{bucket}/{prefix}");
    let listed = crate::data::source::aws::list_prefix(
        &list_uri,
        profile.as_deref(),
        endpoint.as_deref(),
    )?;
    let keys: Vec<&str> = listed
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| {
            let rel = k.strip_prefix(prefix).unwrap_or(k);
            matches!(
                std::path::Path::new(k).extension().and_then(|e| e.to_str()),
                Some("parquet" | "geoparquet" | "pq")
            ) && !rel
                .split('/')
                .any(|seg| seg.starts_with('.') || seg.starts_with('_'))
                && glob.is_none_or(|g| glob_match(g, k))
        })
        .collect();
    if keys.is_empty() {
        return Err(match glob {
            Some(g) => format!("no .parquet objects matching s3://{bucket}/{g}"),
            None => format!("no .parquet objects under {uri}"),
        });
    }
    if keys.len() > PREFIX_PART_CAP {
        return Err(format!(
            "{} parquet files under {uri} (cap {PREFIX_PART_CAP} per load) — \
             open a deeper prefix (e.g. one hive partition)",
            keys.len()
        ));
    }

    let hive: Vec<Vec<(String, Option<String>)>> = keys
        .iter()
        .map(|k| {
            let rel = k.strip_prefix(prefix).unwrap_or(k);
            hive_segments(std::path::Path::new(rel))
        })
        .collect();
    // Resolve (presign + length probe) every part in parallel.
    let files: Vec<(Source, String)> = {
        use rayon::prelude::*;
        keys.par_iter()
            .map(|k| {
                let short = k.strip_prefix(prefix).unwrap_or(k).to_string();
                Source::S3 {
                    uri: format!("s3://{bucket}/{k}"),
                    profile: profile.clone(),
                    endpoint: endpoint.clone(),
                    url: String::new(),
                    len: 0,
                }
                .resolve()
                .map(|src| (src, short.clone()))
                .map_err(|e| format!("{short}: {e}"))
            })
            .collect::<Result<_, _>>()?
    };
    log::info!("{uri}: opening {} part files", files.len());
    open_multi_store(source, files, hive)
}

/// Open a STAC collection as one multi-fragment remote store: resolve
/// its parts (Items, or the collection's own parquet assets), keep the
/// ones whose bbox intersects the viewport, cap.
fn open_stac_store(
    source: &Source,
    collection_url: &str,
    rect: Option<[f64; 4]>,
) -> Result<StoreOpen, String> {
    use super::store::hive_segments;

    let parts = crate::data::repo::fetch_stac_parts(collection_url)?;
    let total = parts.len();
    let keep: Vec<_> = parts
        .into_iter()
        .filter(|p| match (rect, p.bbox) {
            (Some(r), Some(b)) => b[0] <= r[2] && b[2] >= r[0] && b[1] <= r[3] && b[3] >= r[1],
            _ => true,
        })
        .collect();
    if keep.is_empty() {
        return Err("no parts of this collection intersect the current view".into());
    }
    // Over the cap, open the parts covering most of the view and let
    // panning bring in the rest (see `spawn_part_append`). Refusing the
    // load was the old behaviour and it left the collection unopenable
    // from any view wide enough to want it.
    let mut keep = keep;
    if keep.len() > STAC_PART_CAP {
        if let Some(r) = rect {
            keep.sort_by(|a, b| {
                overlap_area(b.bbox, r)
                    .partial_cmp(&overlap_area(a.bbox, r))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        log::info!(
            "{}: {} of {total} parts intersect; opening {STAC_PART_CAP}, \
             the rest load as you pan",
            source.label(),
            keep.len()
        );
        keep.truncate(STAC_PART_CAP);
    }
    if keep.len() < total {
        log::info!(
            "{}: part pruning {total} -> {} files",
            source.label(),
            keep.len()
        );
    }
    // Hive `key=value` segments of the parts that live under the
    // collection — a dataset published as one collection.json over a
    // partitioned tree is the https twin of `open_s3_prefix_store`, and
    // its partition columns have to survive the trip. Parts addressed by
    // their own Items carry no such path (`rel` is None) and keep the
    // empty segments they always had.
    let hive: Vec<Vec<(String, Option<String>)>> = keep
        .iter()
        .map(|p| match &p.rel {
            Some(rel) => hive_segments(std::path::Path::new(rel)),
            None => Vec::new(),
        })
        .collect();
    // Resolve (length probe) every part in parallel.
    let files: Vec<(Source, String)> = {
        use rayon::prelude::*;
        keep.into_par_iter()
            .map(|p| {
                let short = p
                    .rel
                    .clone()
                    .unwrap_or_else(|| p.url.rsplit('/').next().unwrap_or(&p.url).to_string());
                Source::Remote { url: p.url, len: 0 }
                    .resolve()
                    .map(|src| (src, short.clone()))
                    .map_err(|e| format!("{short}: {e}"))
            })
            .collect::<Result<_, _>>()?
    };
    let mut opened = open_multi_store(source, files, hive)?;
    // The parts are plain object-store URLs, so the info built from the
    // first one looked for a credit beside a parquet file in a bucket and
    // found none. What licenses this data is the collection that lists it.
    // Only as a fallback: a part that credits itself is more specific than
    // anything the collection can say about the set.
    if opened.2.attribution.is_none() {
        opened.2.attribution = crate::data::attribution::find(source, &[]);
    }
    Ok(opened)
}

/// Open a set of same-schema parquet files as one multi-fragment store.
/// All files must share the schema, CRS, geometry column and encoding;
/// `hive` carries each file's `key=value` path segments (empty when the
/// dataset has none).
fn open_multi_store(
    source: &Source,
    files: Vec<(Source, String)>,
    hive: Vec<Vec<(String, Option<String>)>>,
) -> Result<StoreOpen, String> {
    use super::store::Fragment;

    // Hive keys, in first-appearance order across all paths.
    let mut part_cols: Vec<String> = Vec::new();
    for segs in &hive {
        for (k, _) in segs {
            if !part_cols.iter().any(|c| c == k) {
                part_cols.push(k.clone());
            }
        }
    }

    // Open every part's footer in parallel: remote parts spend their
    // time in network round-trips (a HEAD probe plus footer ranges
    // each), so this is the wall-clock win for prefix/STAC datasets.
    // The merge below stays sequential and order-stable.
    let opened: Vec<FileOpen> = {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|(src, short)| open_file(src).map_err(|e| format!("{short}: {e}")))
            .collect::<Result<_, _>>()?
    };
    // A hive key shadowed by a real column stays path-only.
    part_cols.retain(|k| {
        !opened[0]
            .schema
            .fields()
            .iter()
            .any(|f| f.name().eq_ignore_ascii_case(k))
    });

    let mut frags: Vec<(Fragment, Vec<u64>)> = Vec::with_capacity(files.len());
    let mut boxes: Option<Vec<[f64; 4]>> = Some(Vec::new());
    let mut box_source: Option<String> = None;
    let mut info = opened[0].info.clone();
    info.files = files.len();
    let mut total_rows = 0u64;
    let mut page_index = true;
    let mut geom_bytes = 0u64;

    for (i, ((src, short), f)) in files.iter().zip(&opened).enumerate() {
        if i > 0 {
            let first = &opened[0];
            schema_compatible(&first.schema, &f.schema)
                .map_err(|e| format!("{short}: {e}"))?;
            if !f.crs.same_as(&first.crs) {
                return Err(format!(
                    "{short}: CRS '{}' differs from the dataset's '{}'",
                    f.crs.name, first.crs.name
                ));
            }
            if f.encoding != first.encoding || f.geom_col != first.geom_col {
                return Err(format!(
                    "{short}: geometry column/encoding differs from the dataset's"
                ));
            }
            info.file_size += f.info.file_size;
            info.rows += f.info.rows;
            info.row_groups += f.info.row_groups;
            info.rg_rows_min = info.rg_rows_min.min(f.info.rg_rows_min);
            info.rg_rows_max = info.rg_rows_max.max(f.info.rg_rows_max);
            info.rg_bytes_max = info.rg_bytes_max.max(f.info.rg_bytes_max);
            info.compressed_bytes += f.info.compressed_bytes;
            info.uncompressed_bytes += f.info.uncompressed_bytes;
        }
        total_rows += f.rg_rows.iter().sum::<u64>();
        page_index &= f.page_index;
        geom_bytes += f.geom_bytes;

        // Per-fragment row-group boxes; a file-level bbox from the geo
        // metadata is a valid (coarse) fallback for all its groups.
        if let Some(all) = &mut boxes {
            match (&f.rg_boxes, f.info.geo.bbox) {
                (Some((src_label, b)), _) => {
                    all.extend_from_slice(b);
                    if box_source.is_none() {
                        box_source = Some(src_label.clone());
                    }
                }
                (None, Some(b)) => {
                    all.extend(std::iter::repeat_n(b, f.rg_rows.len()));
                    if box_source.is_none() {
                        box_source = Some("file-level geo bbox".into());
                    }
                }
                // A part that describes no extent at all. Adaptive H3
                // writes exactly one of these: the
                // `h3=__HIVE_DEFAULT_PARTITION__` file holding the
                // null-geometry rows, whose covering column is all
                // nulls and whose `geo` metadata therefore carries no
                // bbox. Voiding the whole dataset's row-group boxes
                // over it costs every other part its pruning: one row
                // with no shape turns a 700-file H3 tree into a tree
                // with no spatial index. Its groups get an empty box
                // instead. They hold nothing drawable, so nothing is
                // lost by never selecting them, and every other part
                // keeps its bbox.
                (None, None) => {
                    all.extend(std::iter::repeat_n(
                        [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
                        f.rg_rows.len(),
                    ));
                    if box_source.is_none() {
                        box_source = Some("mixed (a part describes no extent)".into());
                    }
                }
            }
        }

        let part_values: Vec<Option<String>> = part_cols
            .iter()
            .map(|k| {
                hive[i]
                    .iter()
                    .find(|(sk, _)| sk == k)
                    .and_then(|(_, v)| v.clone())
            })
            .collect();
        frags.push((
            Fragment {
                source: src.clone(),
                meta: f.meta.clone(),
                part_values,
                rg_offset: 0,
                row_offset: 0,
            },
            f.rg_rows.clone(),
        ));
    }
    if total_rows >= u32::MAX as u64 {
        return Err(format!(
            "dataset has {total_rows} rows; max supported is {}",
            u32::MAX - 1
        ));
    }
    let first = opened
        .into_iter()
        .next()
        .expect("multi store has at least one file");
    for k in &part_cols {
        info.columns.push(ColumnInfo {
            name: k.clone(),
            arrow_type: "Utf8".into(),
            compression: "(hive path)".into(),
            logical: None,
            is_geometry: false,
        });
    }

    let rg_boxes = boxes
        .filter(|b| !b.is_empty())
        .map(|b| (box_source.unwrap_or_else(|| "mixed".into()), b));
    // COGP row-group indices are per file, and this store renumbers them
    // across fragments: a per-part `cogp` block says nothing about the
    // dataset, so it is not carried over.
    info.geo.cogp = None;
    info.quality = Some(quality_report(
        &info,
        rg_boxes.as_ref(),
        first.encoding,
        first.xy.is_some(),
        page_index,
        geom_bytes,
        None,
    ));
    let mut store = FeatureStore::from_fragments(
        source.clone(),
        frags,
        part_cols,
        first.geom_col,
        first.schema,
        first.covering,
        first.encoding,
        first.xy,
    );
    store.hidden_wkb = first.hidden_wkb;
    store.spherical_edges = first.spherical_edges;
    store.base64_wkb = first.base64_wkb;
    Ok((store, first.crs, info, rg_boxes))
}

/// Field-by-field schema equality (names and types; nullability may vary).
fn schema_compatible(
    base: &arrow::datatypes::SchemaRef,
    other: &arrow::datatypes::SchemaRef,
) -> Result<(), String> {
    if base.fields().len() != other.fields().len() {
        return Err(format!(
            "schema has {} columns, dataset has {}",
            other.fields().len(),
            base.fields().len()
        ));
    }
    for (a, b) in base.fields().iter().zip(other.fields()) {
        if a.name() != b.name() || a.data_type() != b.data_type() {
            return Err(format!(
                "column '{}: {}' does not match the dataset's '{}: {}'",
                b.name(),
                b.data_type(),
                a.name(),
                a.data_type()
            ));
        }
    }
    Ok(())
}

fn open_file(source: &Source) -> Result<FileOpen, String> {
    // Load the footer exactly once (with the page index when present);
    // every reader over this file reuses it.
    let reader = source.open()?;
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional);
    let arrow_meta = ArrowReaderMetadata::load(&reader, options)
        .map_err(|e| format!("not a parquet file: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(reader, arrow_meta.clone());

    // GeoParquet "geo" key-value metadata.
    let kv = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    let geo_meta: Option<Value> = kv
        .iter()
        .find(|kv| kv.key == "geo")
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| serde_json::from_str(v).ok());

    let schema = builder.schema().clone();

    let mut encoding = GeomEncoding::Wkb;
    let mut xy: Option<(usize, usize)> = None;
    let mut spherical_edges = false;
    // Geometry spelled as base64 text (Spark/Sedona exports): the store
    // decodes it to WKB bytes on read.
    let mut base64_wkb = false;
    let (geom_name, crs) = match &geo_meta {
        Some(meta) => {
            let primary = meta
                .get("primary_column")
                .and_then(Value::as_str)
                .unwrap_or("geometry")
                .to_string();
            let col_meta = meta.get("columns").and_then(|c| c.get(&primary));
            if let Some(cm) = col_meta {
                let enc = cm.get("encoding").and_then(Value::as_str).unwrap_or("WKB");
                encoding = GeomEncoding::parse(enc)
                    .ok_or_else(|| format!("geometry encoding '{enc}' not supported"))?;
                spherical_edges =
                    cm.get("edges").and_then(Value::as_str) == Some("spherical");
            }
            // A `geopq:crs` extension (our shapefile importer, for ESRI
            // .prj files without an EPSG identity) beats a spec-level
            // `crs: null`: the spec value stays honest for other
            // readers, the proj4 string positions the data correctly.
            let vendor = col_meta
                .and_then(|c| c.get("geopq:crs"))
                .filter(|_| {
                    col_meta.and_then(|c| c.get("crs")) == Some(&Value::Null)
                })
                .and_then(|v| {
                    let p4 = v.get("proj4")?.as_str()?;
                    let name = v.get("name").and_then(Value::as_str).unwrap_or("from .prj");
                    Crs::from_proj4(p4, None, name).ok()
                });
            let crs = match vendor {
                Some(c) => c,
                None => Crs::from_geoparquet_crs(col_meta.and_then(|c| c.get("crs")))?,
            };
            (primary, crs)
        }
        None if native_geometry_column(&builder).is_some() => {
            // Native GEOMETRY/GEOGRAPHY logical type without `geo`
            // metadata (2.0 writers may omit it): the column and its CRS
            // come from the logical type itself. GEOGRAPHY means edges
            // are great-circle arcs.
            let (name, crs_str, geography) = native_geometry_column(&builder).unwrap();
            spherical_edges = geography;
            (name, crs_from_type_string(crs_str.as_deref())?)
        }
        None => {
            // Not GeoParquet-tagged: find a WKB column, assume CRS84.
            //
            // The name list is only a ranking, never a promise, and "the
            // first binary column" is a coin flip: a thumbnail, a hash or
            // a protobuf blob is binary too, and adopting one produces a
            // layer of garbage instead of an honest refusal. Every
            // candidate is probed against real values before it is taken.
            let candidates = wkb_candidates(&schema);
            let found = candidates
                .into_iter()
                .find(|&(i, b64)| probe_wkb_column(source, &arrow_meta, i, b64));
            match found {
                Some((i, b64)) => {
                    base64_wkb = b64;
                    (schema.field(i).name().clone(), Crs::wgs84())
                }
                None => {
                    // Last resort: coordinate columns → synthesized points.
                    let (xi, yi) = xy_columns(&schema).ok_or(
                        "no 'geo' metadata, no column whose values read as WKB \
                         geometry, and no lon/lat or x/y coordinate columns found",
                    )?;
                    xy = Some((xi, yi));
                    encoding = GeomEncoding::Point;
                    let mut crs = Crs::wgs84();
                    crs.name = format!(
                        "assumed CRS84 (points from {}/{} columns)",
                        schema.field(xi).name(),
                        schema.field(yi).name()
                    );
                    (String::new(), crs)
                }
            }
        }
    };

    let geom_col = match xy {
        // The store appends the virtual geometry right after the base
        // fields; this index is recomputed there.
        Some(_) => schema.fields().len(),
        None => schema
            .index_of(&geom_name)
            .map_err(|_| format!("geometry column '{geom_name}' not found in schema"))?,
    };

    // A WKB primary with a GeoArrow sibling column (the optimizer's 2.0
    // flavor export): decode from the coordinate arrays instead, and hide
    // the redundant WKB blob from attribute UIs. The `geo` metadata keeps
    // declaring the primary, so metadata lookups below stay on its name.
    let primary_name = geom_name.clone();
    let mut hidden_wkb: Option<usize> = None;
    let (geom_name, geom_col, encoding) = match aux_geoarrow_column(&schema, &geom_name) {
        Some((i, enc)) if encoding.is_wkb() && xy.is_none() => {
            hidden_wkb = Some(geom_col);
            (schema.field(i).name().clone(), i, enc)
        }
        _ => (geom_name, geom_col, encoding),
    };

    let total_rows = builder.metadata().file_metadata().num_rows();
    if total_rows >= u32::MAX as i64 {
        return Err(format!("file has {total_rows} rows; max supported is {}", u32::MAX - 1));
    }
    let rg_rows: Vec<u64> = builder
        .metadata()
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as u64)
        .collect();

    // ---- file info for the UI panel ----
    let meta = builder.metadata();
    let fmd = meta.file_metadata();
    let pq_columns = builder.parquet_schema().columns();
    // Arrow root index != parquet leaf index once nested columns exist:
    // resolve each root to its first leaf by path root name.
    let leaf_of_root = |name: &str| -> Option<usize> {
        pq_columns
            .iter()
            .position(|c| c.path().parts().first().map(String::as_str) == Some(name))
    };
    let geom_leaf = leaf_of_root(&geom_name);
    // The native GEOMETRY logical type lives on the declared primary even
    // when display decodes from a GeoArrow sibling.
    let native_probe = hidden_wkb.unwrap_or(geom_col);
    let mut has_native_geometry = false;
    let columns: Vec<ColumnInfo> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let leaf = leaf_of_root(field.name());
            let logical = leaf.and_then(|l| {
                pq_columns[l].logical_type_ref().map(|lt| format!("{lt:?}"))
            });
            if i == native_probe {
                if let Some(l) = &logical {
                    if l.starts_with("Geometry") || l.starts_with("Geography") {
                        has_native_geometry = true;
                    }
                }
            }
            let compression = leaf
                .and_then(|l| meta.row_groups().first().and_then(|rg| rg.columns().get(l)))
                .map(|c| format!("{}", c.compression()))
                .unwrap_or_else(|| "?".into());
            ColumnInfo {
                name: field.name().clone(),
                arrow_type: format!("{}", field.data_type()),
                compression,
                logical,
                is_geometry: i == geom_col,
            }
        })
        .collect();
    let (compressed_bytes, uncompressed_bytes) = meta.row_groups().iter().fold((0u64, 0u64), |acc, rg| {
        (
            acc.0 + rg.compressed_size().max(0) as u64,
            acc.1 + rg.total_byte_size().max(0) as u64,
        )
    });
    // Facts for the quality report: page-index presence and the
    // uncompressed size of the geometry leaves (all leaves under the
    // geometry root; for x/y files, the two coordinate columns).
    let page_index = meta
        .row_groups()
        .iter()
        .all(|rg| rg.columns().iter().all(|c| c.offset_index_offset().is_some()));
    let geom_leaves: Vec<usize> = match xy {
        Some((xi, yi)) => [xi, yi]
            .iter()
            .filter_map(|&i| leaf_of_root(schema.field(i).name()))
            .collect(),
        None => pq_columns
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.path().parts().first().map(String::as_str) == Some(geom_name.as_str())
            })
            .map(|(i, _)| i)
            .collect(),
    };
    let geom_bytes: u64 = meta
        .row_groups()
        .iter()
        .flat_map(|rg| geom_leaves.iter().map(|&l| rg.column(l).uncompressed_size().max(0) as u64))
        .sum();
    let info = FileInfo {
        file_size: source.size(),
        parquet_format_version: fmd.version(),
        created_by: fmd.created_by().map(String::from),
        rows: total_rows as u64,
        row_groups: rg_rows.len(),
        rg_rows_min: rg_rows.iter().copied().min().unwrap_or(0),
        rg_rows_max: rg_rows.iter().copied().max().unwrap_or(0),
        rg_bytes_max: meta
            .row_groups()
            .iter()
            .map(|rg| rg.total_byte_size().max(0) as u64)
            .max()
            .unwrap_or(0),
        compressed_bytes,
        uncompressed_bytes,
        columns,
        geo: summarize_geo_meta(geo_meta.as_ref(), &primary_name, &crs.name, has_native_geometry),
        files: 1,
        attribution: super::attribution::find(source, &kv),
        quality: None,
        // Set by the pyramid open path, which is the only thing that
        // knows the root a part belongs to.
        pyramid: None,
        pyramid_file: None,
    };

    let rg_boxes = match xy {
        // Ordinary min/max statistics of the coordinate columns give a
        // bbox per row group for free.
        Some((xi, yi)) => xy_rg_boxes(
            &builder,
            leaf_of_root(schema.field(xi).name()),
            leaf_of_root(schema.field(yi).name()),
        ),
        None => rg_bboxes_from_metadata(
            &builder,
            geo_meta.as_ref(),
            geom_leaf,
            &geom_name,
            encoding,
            crs.is_latlong,
        ),
    };
    let covering = covering_column(geo_meta.as_ref(), &primary_name, &schema);

    // COGP: a prefix of row groups per rendering scale (docs at
    // src/data/cogp.rs). Validated here, from the footer, and a block
    // that fails validation is recorded and otherwise ignored — the file
    // is still an ordinary GeoParquet and must still open.
    let cogp = kv
        .iter()
        .find(|kv| kv.key == "cogp")
        .and_then(|kv| kv.value.clone())
        .map(|v| {
            super::cogp::parse(
                &v,
                rg_rows.len(),
                cogp_pruning_signal(
                    &builder,
                    geo_meta.as_ref(),
                    &primary_name,
                    geom_leaf,
                    encoding,
                    crs.is_latlong,
                ),
            )
        });

    let mut info = info;
    // `geopq:pyramid`: an overview file says what it is even when it is
    // opened on its own, away from the descriptor that would otherwise
    // badge it. Derived geometry is never on screen unmarked
    // (docs/OPEN_POLICY.md invariant 1), and one file out of a pyramid
    // is exactly the case where the layer has no other way to know.
    info.pyramid_file = kv
        .iter()
        .find(|kv| kv.key == pyramid::FILE_KEY)
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| match serde_json::from_str::<pyramid::FileMeta>(v) {
            Ok(m) => Some(m),
            Err(e) => {
                log::warn!("{}: unusable {}: {e}", source.name(), pyramid::FILE_KEY);
                None
            }
        });
    if hidden_wkb.is_some() {
        info.geo.encoding = format!("{} + GeoArrow column (used for display)", info.geo.encoding);
    }
    if base64_wkb {
        info.geo.encoding = "WKB, base64 text (decoded on read)".into();
    }
    info.geo.cogp = match &cogp {
        Some(Ok(c)) => Some(c.summary()),
        Some(Err(e)) => Some(format!("`cogp` metadata present but not usable: {e}")),
        None => None,
    };
    if let Some((xi, yi)) = xy {
        info.geo.version_label = "none (points synthesized from coordinate columns)".into();
        info.geo.primary_column = "geometry (virtual)".into();
        info.geo.encoding = format!(
            "x/y columns: {}, {}",
            schema.field(xi).name(),
            schema.field(yi).name()
        );
        info.columns.push(ColumnInfo {
            name: "geometry".into(),
            arrow_type: "Struct(x, y)".into(),
            compression: "(virtual)".into(),
            logical: None,
            is_geometry: true,
        });
    }

    // The store hands out decoded WKB bytes for a base64 column, so the
    // schema the rest of the app plans against must already say so. The
    // info panel's column list was built above from the file's own types,
    // which is what that panel is for.
    let schema = if base64_wkb {
        let mut fields: Vec<arrow::datatypes::Field> =
            schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields[geom_col] = arrow::datatypes::Field::new(
            fields[geom_col].name(),
            DataType::Binary,
            true,
        );
        std::sync::Arc::new(arrow::datatypes::Schema::new_with_metadata(
            fields,
            schema.metadata().clone(),
        ))
    } else {
        schema
    };

    Ok(FileOpen {
        meta: arrow_meta,
        schema,
        crs,
        encoding,
        geom_col,
        covering,
        rg_rows,
        rg_boxes,
        info,
        xy,
        page_index,
        geom_bytes,
        hidden_wkb,
        spherical_edges,
        base64_wkb,
        cogp,
    })
}

/// GeoArrow sibling of a WKB primary (`{primary}_geoarrow*`), as written
/// by the optimizer's 2.0 flavor export: (schema index, encoding). Strict
/// structural match against our canonical layout — the extension name is
/// written for interop but layouts we can't decode must not be adopted.
fn aux_geoarrow_column(
    schema: &arrow::datatypes::SchemaRef,
    primary: &str,
) -> Option<(usize, GeomEncoding)> {
    let prefix = format!("{primary}_geoarrow");
    schema.fields().iter().enumerate().find_map(|(i, f)| {
        if !f.name().starts_with(prefix.as_str()) {
            return None;
        }
        let enc = [
            GeomEncoding::Point,
            GeomEncoding::LineString,
            GeomEncoding::Polygon,
            GeomEncoding::MultiPoint,
            GeomEncoding::MultiLineString,
            GeomEncoding::MultiPolygon,
        ]
        .into_iter()
        .find(|&e| super::geoarrow::data_type(e) == *f.data_type())?;
        Some((i, enc))
    })
}

/// First column carrying a native GEOMETRY/GEOGRAPHY logical type:
/// (root name, crs string from the type, is GEOGRAPHY).
fn native_geometry_column(
    builder: &ParquetRecordBatchReaderBuilder<super::source::SourceReader>,
) -> Option<(String, Option<String>, bool)> {
    use parquet::basic::LogicalType;
    builder.parquet_schema().columns().iter().find_map(|c| {
        let (crs, geography) = match c.logical_type_ref() {
            Some(LogicalType::Geometry { crs }) => (crs.clone(), false),
            Some(LogicalType::Geography { crs, .. }) => (crs.clone(), true),
            _ => return None,
        };
        Some((c.path().parts().first()?.clone(), crs, geography))
    })
}

/// CRS recorded in a GEOMETRY/GEOGRAPHY logical type. The parquet spec
/// leaves it a free-form string: PROJJSON, "EPSG:nnnn" and the CRS84
/// spellings are recognized; anything else renders as CRS84 with an
/// honest name.
fn crs_from_type_string(s: Option<&str>) -> Result<Crs, String> {
    let Some(t) = s.map(str::trim) else {
        return Ok(Crs::wgs84()); // absent = CRS84 per spec
    };
    if t.is_empty()
        || t.eq_ignore_ascii_case("OGC:CRS84")
        || t.eq_ignore_ascii_case("CRS84")
        || t.eq_ignore_ascii_case("EPSG:4326")
    {
        return Ok(Crs::wgs84());
    }
    // The string may itself be JSON: a PROJJSON object, or a JSON-quoted
    // plain string (writers differ).
    if t.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            return Crs::from_geoparquet_crs(Some(&v));
        }
    }
    if t.starts_with('"') {
        if let Ok(Value::String(inner)) = serde_json::from_str::<Value>(t) {
            return crs_from_type_string(Some(&inner));
        }
    }
    if let Some(code) = t
        .strip_prefix("EPSG:")
        .or_else(|| t.strip_prefix("epsg:"))
        .and_then(|c| c.parse::<u32>().ok())
    {
        return Crs::from_epsg(code);
    }
    let mut crs = Crs::wgs84();
    crs.name = format!(
        "unknown CRS '{}' (rendered as CRS84)",
        t.chars().take(40).collect::<String>()
    );
    crs.epsg = None;
    Ok(crs)
}

/// Column names that conventionally hold WKB geometry, matched
/// case-insensitively when a file carries no `geo` metadata at all.
const WKB_COLUMN_NAMES: [&str; 6] =
    ["geometry", "geom", "wkb_geometry", "geometry_wkb", "geom_wkb", "wkb"];

/// Values sampled before an untagged column is accepted as geometry.
const WKB_PROBE_VALUES: usize = 16;

/// Does `buf` open with a plausible WKB header — a byte-order flag of 0
/// or 1 followed by a geometry type code in the ISO/EWKB set?
///
/// EWKB puts its Z/M/SRID flags in the top three bits of the code and ISO
/// WKB adds 1000/2000/3000 for Z/M/ZM, so the discriminant is what is left
/// modulo 1000: 1..=7, Point through GeometryCollection.
fn is_wkb_header(buf: &[u8]) -> bool {
    if buf.len() < 5 {
        return false;
    }
    let code = match buf[0] {
        0 => u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]),
        1 => u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]),
        _ => return false,
    } & !(0x8000_0000 | 0x4000_0000 | 0x2000_0000);
    code / 1000 <= 3 && (1..=7).contains(&(code % 1000))
}

/// Geometry-column candidates for a file with no `geo` metadata, best
/// first: named binary columns, then any other binary column, then named
/// text columns (which can only be base64 WKB). The bool is "decode
/// base64 first".
///
/// Text is ranked last and only by name because an unnamed string column
/// that happens to base64-decode is far likelier to be an opaque blob
/// than a geometry, and unlike binary there is no cheap type signal.
fn wkb_candidates(schema: &arrow::datatypes::SchemaRef) -> Vec<(usize, bool)> {
    let named = |f: &arrow::datatypes::Field| {
        WKB_COLUMN_NAMES.contains(&f.name().to_ascii_lowercase().as_str())
    };
    let binary = |f: &arrow::datatypes::Field| {
        matches!(
            f.data_type(),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        )
    };
    let text = |f: &arrow::datatypes::Field| {
        matches!(
            f.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    };
    let mut out: Vec<(usize, bool)> = Vec::new();
    let mut push = |i: usize, b64: bool| {
        if !out.iter().any(|o| o.0 == i) {
            out.push((i, b64));
        }
    };
    for (i, f) in schema.fields().iter().enumerate() {
        if binary(f) && named(f) {
            push(i, false);
        }
    }
    for (i, f) in schema.fields().iter().enumerate() {
        if binary(f) {
            push(i, false);
        }
    }
    for (i, f) in schema.fields().iter().enumerate() {
        if text(f) && named(f) {
            push(i, true);
        }
    }
    out
}

/// Read up to [`WKB_PROBE_VALUES`] non-null values of one column and test
/// every one against [`is_wkb_header`], base64-decoding first when asked.
/// A column with nothing to sample fails: nothing was seen, nothing is
/// proven.
///
/// This is the only point in the opener that touches a data page, and it
/// costs one page of one column. That is the price of not handing the
/// user a map of a thumbnail column.
fn probe_wkb_column(
    source: &Source,
    meta: &ArrowReaderMetadata,
    root: usize,
    base64: bool,
) -> bool {
    let Ok(reader) = source.open() else {
        return false;
    };
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(reader, meta.clone());
    let mask = ProjectionMask::roots(builder.parquet_schema(), [root]);
    let Ok(mut rdr) = builder
        .with_projection(mask)
        .with_batch_size(WKB_PROBE_VALUES)
        .build()
    else {
        return false;
    };
    let want = if base64 { DataType::Utf8 } else { DataType::Binary };
    let mut seen = 0usize;
    while seen < WKB_PROBE_VALUES {
        let Some(Ok(batch)) = rdr.next() else {
            break;
        };
        let Ok(col) = arrow::compute::cast(batch.column(0), &want) else {
            return false;
        };
        // One closure over both shapes: the base64 case decodes, the
        // binary case is already bytes.
        let value = |i: usize| -> Option<Option<Vec<u8>>> {
            if base64 {
                let a = col.as_any().downcast_ref::<arrow::array::StringArray>()?;
                if !a.is_valid(i) {
                    return Some(None);
                }
                // A named text column whose values are not base64 at all
                // is not geometry: fail the probe, do not skip the value.
                Some(Some(super::store::decode_base64_wkb(a.value(i))?))
            } else {
                let a = col.as_any().downcast_ref::<BinaryArray>()?;
                Some(a.is_valid(i).then(|| a.value(i).to_vec()))
            }
        };
        for i in 0..batch.num_rows() {
            match value(i) {
                Some(Some(bytes)) => {
                    if !is_wkb_header(&bytes) {
                        return false;
                    }
                    seen += 1;
                    if seen >= WKB_PROBE_VALUES {
                        break;
                    }
                }
                // Null: neither evidence for nor against.
                Some(None) => {}
                None => return false,
            }
        }
    }
    seen > 0
}

/// Coordinate-column pair (x/lon, y/lat) guessed from names, for files
/// with no geometry column at all. Both must be floating point.
fn xy_columns(schema: &arrow::datatypes::SchemaRef) -> Option<(usize, usize)> {
    let find = |names: &[&str]| {
        schema.fields().iter().position(|f| {
            names.contains(&f.name().to_ascii_lowercase().as_str())
                && matches!(f.data_type(), DataType::Float64 | DataType::Float32)
        })
    };
    let x = find(&["lon", "longitude", "long", "lng", "x"])?;
    let y = find(&["lat", "latitude", "y"])?;
    (x != y).then_some((x, y))
}

/// Per-row-group bboxes from the coordinate columns' ordinary min/max
/// statistics.
fn xy_rg_boxes(
    builder: &ParquetRecordBatchReaderBuilder<super::source::SourceReader>,
    x_leaf: Option<usize>,
    y_leaf: Option<usize>,
) -> Option<(String, Vec<[f64; 4]>)> {
    let (xi, yi) = (x_leaf?, y_leaf?);
    let stat = |rg: &parquet::file::metadata::RowGroupMetaData,
                idx: usize,
                want_max: bool|
     -> Option<f64> {
        use parquet::file::statistics::Statistics;
        match rg.columns().get(idx)?.statistics()? {
            Statistics::Double(s) => Some(*if want_max { s.max_opt() } else { s.min_opt() }?),
            Statistics::Float(s) => {
                Some(*if want_max { s.max_opt() } else { s.min_opt() }? as f64)
            }
            _ => None,
        }
    };
    let boxes: Option<Vec<[f64; 4]>> = builder
        .metadata()
        .row_groups()
        .iter()
        .map(|rg| {
            Some([
                stat(rg, xi, false)?,
                stat(rg, yi, false)?,
                stat(rg, xi, true)?,
                stat(rg, yi, true)?,
            ])
        })
        .collect();
    boxes
        .filter(|b| !b.is_empty())
        .map(|b| ("coordinate column statistics (x/y)".into(), b))
}

/// Resolve the GeoParquet 1.1 covering bbox column: a single root struct
/// whose four children hold xmin/ymin/xmax/ymax. Non-canonical layouts
/// (paths deeper than root.child or spread over several roots) are ignored.
fn covering_column(
    geo_meta: Option<&Value>,
    primary: &str,
    schema: &arrow::datatypes::SchemaRef,
) -> Option<CoveringCol> {
    let bbox = geo_meta?
        .get("columns")?
        .get(primary)?
        .get("covering")?
        .get("bbox")?;
    let path = |part: &str| -> Option<(String, String)> {
        let arr = bbox.get(part)?.as_array()?;
        match arr.as_slice() {
            [root, child] => Some((root.as_str()?.to_string(), child.as_str()?.to_string())),
            _ => None,
        }
    };
    let (r0, xmin) = path("xmin")?;
    let (r1, ymin) = path("ymin")?;
    let (r2, xmax) = path("xmax")?;
    let (r3, ymax) = path("ymax")?;
    if r0 != r1 || r0 != r2 || r0 != r3 {
        return None;
    }
    Some(CoveringCol {
        root: schema.index_of(&r0).ok()?,
        children: [xmin, ymin, xmax, ymax],
    })
}

/// Extract per-row-group bboxes from file metadata, best source first:
/// parquet native geospatial statistics (GeoParquet 2.0), then GeoParquet
/// 1.1 `covering` bbox-column statistics. Returns (source label, boxes).
pub(crate) fn rg_bboxes_from_metadata(
    builder: &ParquetRecordBatchReaderBuilder<super::source::SourceReader>,
    geo_meta: Option<&Value>,
    geom_leaf: Option<usize>,
    primary: &str,
    encoding: GeomEncoding,
    is_latlong: bool,
) -> Option<(String, Vec<[f64; 4]>)> {
    let meta = builder.metadata();

    // Statistics helper over any leaf column.
    let stat_f64 = |rg: &parquet::file::metadata::RowGroupMetaData,
                    idx: usize,
                    want_max: bool|
     -> Option<f64> {
        use parquet::file::statistics::Statistics;
        let st = rg.columns().get(idx)?.statistics()?;
        match st {
            Statistics::Double(s) => {
                let v = if want_max { s.max_opt() } else { s.min_opt() }?;
                Some(*v)
            }
            Statistics::Float(s) => {
                let v = if want_max { s.max_opt() } else { s.min_opt() }?;
                Some(*v as f64)
            }
            _ => None,
        }
    };

    // 1. Native geospatial statistics on the geometry column chunks
    // (leaf-indexed, not arrow-root-indexed).
    let native: Option<Vec<[f64; 4]>> = geom_leaf.and_then(|leaf| {
        meta.row_groups()
            .iter()
            .map(|rg| {
                let stats = rg.columns().get(leaf)?.geo_statistics()?;
                let b = stats.bounding_box()?;
                normalize_geo_stat_bbox(
                    [b.get_xmin(), b.get_ymin(), b.get_xmax(), b.get_ymax()],
                    is_latlong,
                )
            })
            .collect()
    });
    if let Some(boxes) = native {
        if !boxes.is_empty() {
            return Some(("parquet geospatial statistics".into(), boxes));
        }
    }

    // 2. GeoParquet 1.1 covering bbox columns: column-chunk min/max stats.
    let covering_boxes = || -> Option<Vec<[f64; 4]>> {
        let covering = geo_meta?
            .get("columns")?
            .get(primary)?
            .get("covering")?
            .get("bbox")?;
        let leaf_path = |part: &str| -> Option<usize> {
            let arr = covering.get(part)?.as_array()?;
            let path: Vec<&str> = arr.iter().filter_map(Value::as_str).collect();
            let dotted = path.join(".");
            builder
                .parquet_schema()
                .columns()
                .iter()
                .position(|c| c.path().string() == dotted)
        };
        let (xmin_i, ymin_i, xmax_i, ymax_i) = (
            leaf_path("xmin")?,
            leaf_path("ymin")?,
            leaf_path("xmax")?,
            leaf_path("ymax")?,
        );
        meta.row_groups()
            .iter()
            .map(|rg| {
                Some([
                    stat_f64(rg, xmin_i, false)?,
                    stat_f64(rg, ymin_i, false)?,
                    stat_f64(rg, xmax_i, true)?,
                    stat_f64(rg, ymax_i, true)?,
                ])
            })
            .collect()
    };
    if let Some(boxes) = covering_boxes().filter(|b| !b.is_empty()) {
        return Some(("covering column statistics (GeoParquet 1.1)".into(), boxes));
    }

    // 3. GeoArrow encodings: the x/y coordinate leaves carry ordinary
    // min/max statistics — a bbox per row group for free.
    if encoding.is_wkb() {
        return None;
    }
    let coord_leaf = |axis: &str| -> Option<usize> {
        builder.parquet_schema().columns().iter().position(|c| {
            let parts = c.path().parts();
            parts.first().map(String::as_str) == Some(primary)
                && parts.last().map(String::as_str) == Some(axis)
        })
    };
    let (x_i, y_i) = (coord_leaf("x")?, coord_leaf("y")?);
    let boxes: Option<Vec<[f64; 4]>> = meta
        .row_groups()
        .iter()
        .map(|rg| {
            Some([
                stat_f64(rg, x_i, false)?,
                stat_f64(rg, y_i, false)?,
                stat_f64(rg, x_i, true)?,
                stat_f64(rg, y_i, true)?,
            ])
        })
        .collect();
    boxes
        .filter(|b| !b.is_empty())
        .map(|b| ("coordinate column statistics (GeoArrow)".into(), b))
}

/// Which row-group bbox statistics this file offers, as COGP requires
/// (SPEC §5.1). Runs the loader's own bbox sources one at a time so the
/// answer names the source that exists rather than the one that won:
/// `geom_leaf` None disables the native branch, `geo_meta` None the
/// covering one, and a WKB `encoding` the GeoArrow coordinate leaves.
///
/// Covering wins a tie because it is what the published profile asks
/// for, so a file carrying both is plain COGP rather than an extension.
fn cogp_pruning_signal(
    builder: &ParquetRecordBatchReaderBuilder<super::source::SourceReader>,
    geo_meta: Option<&Value>,
    primary: &str,
    geom_leaf: Option<usize>,
    encoding: GeomEncoding,
    is_latlong: bool,
) -> Option<super::cogp::Pruning> {
    use super::cogp::Pruning;
    if rg_bboxes_from_metadata(builder, geo_meta, None, primary, GeomEncoding::Wkb, is_latlong)
        .is_some()
    {
        return Some(Pruning::Covering);
    }
    if rg_bboxes_from_metadata(builder, None, geom_leaf, primary, GeomEncoding::Wkb, is_latlong)
        .is_some()
    {
        return Some(Pruning::NativeStats);
    }
    // Last, because it is the weakest claim of the three: the coordinate
    // leaves are a decode format's side effect rather than a declared
    // spatial index. They still bound every row group, which is all the
    // levels ask of them.
    rg_bboxes_from_metadata(builder, None, None, primary, encoding, is_latlong)
        .map(|_| Pruning::GeoArrowLeaves)
}

/// Parquet geospatial statistics allow xmin > xmax for row groups that
/// wrap the antimeridian. Taken verbatim such a box never intersects any
/// query rect (silently un-prunable groups); widen it to the full
/// longitude span instead — conservative: the group is never pruned and
/// unions stay sane. A wrapped box on a non-geographic CRS has no defined
/// world span, so the whole native-stats source is rejected (None) and
/// the caller falls back to the next bbox source.
fn normalize_geo_stat_bbox(b: [f64; 4], is_latlong: bool) -> Option<[f64; 4]> {
    if b[0] <= b[2] {
        return Some(b);
    }
    is_latlong.then_some([-180.0, b[1], 180.0, b[3]])
}

/// Average number of other boxes each box intersects.
pub(crate) fn bbox_overlap_metric(boxes: &[[f64; 4]]) -> f64 {
    let n = boxes.len().min(4096);
    if n < 2 {
        return 0.0;
    }
    let mut total = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (&boxes[i], &boxes[j]);
            if a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1] {
                total += 2;
            }
        }
    }
    total as f64 / n as f64
}

/// Resolved data-driven styling for the build: which store column feeds
/// the bins and how values map to them.
#[derive(Clone)]
pub struct StyleSel {
    /// Store-schema index of the value column.
    pub col: usize,
    pub binning: Binning,
    /// Normalize by polygon area (data-CRS units) before binning:
    /// bin = f(value / area). Graduated styling only.
    pub per_area: bool,
    /// Build polygon outlines at all. A colour map defines fills and
    /// nothing else, and outlines are where a polygon mesh's memory
    /// goes: on CORINE, 413 MB of a 653 MB mesh was line segments across
    /// ten LOD levels, for borders the style never draws.
    pub outlines: bool,
}

#[derive(Clone)]
pub enum Binning {
    /// Ascending break values; bin = number of breaks ≤ value.
    Breaks(Vec<f64>),
    Categorical { map: std::collections::HashMap<String, u8> },
}

/// Resolve a layer's `style_by` against its store (None when the column
/// vanished or is a geometry).
pub fn resolve_style(
    store: &FeatureStore,
    sb: &crate::data::layer::StyleBy,
) -> Option<StyleSel> {
    use crate::data::layer::{StyleMode, STYLE_BINS};
    let col = store
        .schema
        .fields()
        .iter()
        .position(|f| f.name().eq_ignore_ascii_case(&sb.column))?;
    if col == store.geom_col || col >= store.first_part_index() {
        // Partition columns aren't readable through the group readers.
        return None;
    }
    let binning = match &sb.mode {
        StyleMode::Graduated { breaks, .. } => Binning::Breaks(breaks.clone()),
        StyleMode::Categorical { values, .. } => Binning::Categorical {
            map: values
                .iter()
                .take(STYLE_BINS - 1)
                .enumerate()
                .map(|(i, v)| (v.clone(), i as u8))
                .collect(),
        },
    };
    let per_area =
        sb.per_area && matches!(sb.mode, crate::data::layer::StyleMode::Graduated { .. });
    Some(StyleSel {
        col,
        binning,
        per_area,
        outlines: !sb.mode.is_color_map(),
    })
}

/// Sample up to `cap` values of a column from the already-loaded rows of
/// a layer (classification must never fetch the whole dataset). Blocking —
/// run off the UI thread.
pub fn sample_loaded_values(
    store: &FeatureStore,
    loaded: &[GroupLoad],
    col: usize,
    cap: usize,
    per_area: bool,
    latlong: bool,
    groups: Option<&[u32]>,
) -> Result<Vec<f64>, String> {
    let starts = store.rg_starts();
    // Optional spatial restriction: only sample these row groups (the
    // ones intersecting the viewport for "reclassify from viewport").
    let gset: Option<std::collections::HashSet<u32>> =
        groups.map(|g| g.iter().copied().collect());
    let allowed = |g: usize| gset.as_ref().is_none_or(|s| s.contains(&(g as u32)));
    // Rect-filtered previews: reproduce the load's exact selection (same
    // covering scan, same decimation) so sampling never fetches rows that
    // were never loaded.
    let mut preview_rows: std::collections::HashMap<usize, Vec<u32>> = Default::default();
    for (g, st) in loaded.iter().enumerate() {
        if !allowed(g) {
            continue;
        }
        if let GroupLoad::Preview { stride, rect: Some(r) } = st {
            let group_rows = (starts[g + 1] - starts[g]) as u32;
            let ranges = covering_select(store, g as u32, *r)?
                .unwrap_or_else(|| vec![(0, group_rows)]);
            let sampled: Vec<u32> = ranges
                .iter()
                .flat_map(|&(s, e)| s..e)
                .step_by((*stride).max(1) as usize)
                .collect();
            preview_rows.insert(g, sampled);
        }
    }
    let total: u64 = loaded
        .iter()
        .enumerate()
        .filter(|(g, _)| allowed(*g))
        .map(|(g, st)| match st {
            GroupLoad::Full => starts[g + 1] - starts[g],
            GroupLoad::Rows { ranges, .. } => {
                ranges.iter().map(|&(s, e)| (e - s) as u64).sum()
            }
            GroupLoad::Preview { rect: Some(_), .. } => preview_rows[&g].len() as u64,
            GroupLoad::Preview { stride, rect: None } => {
                (starts[g + 1] - starts[g]).div_ceil(*stride as u64)
            }
            // Boxes hold every feature of the group; a rect-filtered one
            // holds fewer, and sampling a superset of what is visible is
            // what the unfiltered preview case does too.
            GroupLoad::Boxes { .. } => starts[g + 1] - starts[g],
            GroupLoad::None => 0,
        })
        .sum();
    if total == 0 {
        return Err("no rows loaded yet".into());
    }
    let stride = (total / cap.max(1) as u64).max(1) as usize;
    let mut rows: Vec<u32> = Vec::with_capacity(cap + 1);
    let mut c = 0usize; // running loaded-row counter across groups
    for (g, st) in loaded.iter().enumerate() {
        if !allowed(g) {
            continue;
        }
        let start = starts[g] as u32;
        let mut push_span = |s: u32, e: u32, c: &mut usize, step_by: usize| {
            let mut i = s as usize;
            // Align to the global stride phase.
            let phase = (*c) % stride;
            if phase != 0 {
                i += (stride - phase) * step_by;
            }
            while i < e as usize {
                rows.push(start + i as u32);
                i += stride * step_by;
            }
            *c += ((e - s) as usize).div_ceil(step_by);
        };
        match st {
            GroupLoad::Full => {
                push_span(0, (starts[g + 1] - starts[g]) as u32, &mut c, 1)
            }
            GroupLoad::Rows { ranges, .. } => {
                for &(s, e) in ranges {
                    push_span(s, e, &mut c, 1);
                }
            }
            GroupLoad::Preview { rect: Some(_), .. } => {
                // Every global-stride-th of the group's loaded rows.
                let list = &preview_rows[&g];
                let mut i = (stride - (c % stride)) % stride;
                while i < list.len() {
                    rows.push(start + list[i]);
                    i += stride;
                }
                c += list.len();
            }
            GroupLoad::Preview { stride: ps, rect: None } => {
                push_span(0, (starts[g + 1] - starts[g]) as u32, &mut c, *ps as usize)
            }
            GroupLoad::Boxes { .. } => {
                push_span(0, (starts[g + 1] - starts[g]) as u32, &mut c, 1)
            }
            GroupLoad::None => {}
        }
    }
    rows.sort_unstable();
    rows.dedup();
    if rows.is_empty() {
        return Err("no rows loaded yet".into());
    }
    let batches = store.fetch(&rows, Some(&[col]))?;
    // Normalization needs each sampled feature's area too: fetch the
    // geometries for the same rows (aligned with the value order).
    let areas: Option<Vec<f64>> = if per_area {
        let geoms = store.fetch_geoms(&rows)?;
        Some(
            geoms
                .iter()
                .map(|(_, g)| {
                    // Measured exactly as the draw path measures, or the
                    // breaks would be fitted to one scale and the map
                    // coloured on another.
                    g.as_ref().map_or(0.0, |g| ground_area(g, latlong))
                })
                .collect(),
        )
    } else {
        None
    };
    let mut out = Vec::with_capacity(rows.len());
    let mut k = 0usize; // running row index across batches
    for b in &batches {
        let vals = arrow::compute::cast(b.column(0), &DataType::Float64)
            .map_err(|e| format!("value cast: {e}"))?;
        let vals = vals
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        for i in 0..vals.len() {
            let null = arrow::array::Array::is_null(vals, i);
            let v = vals.value(i);
            match &areas {
                Some(a) => {
                    let area = a.get(k).copied().unwrap_or(0.0);
                    if !null && v.is_finite() && area > 0.0 {
                        out.push(v / area);
                    }
                }
                None => {
                    if !null && v.is_finite() {
                        out.push(v);
                    }
                }
            }
            k += 1;
        }
    }
    Ok(out)
}

/// Per-batch style-bin source: precomputed bins, or raw values plus
/// breaks when bins depend on each feature's area (normalize by area).
pub(crate) enum RowBins {
    Pre(Vec<u8>),
    PerArea {
        vals: Vec<f64>,
        breaks: Vec<f64>,
        /// Data CRS is geographic, so the shoelace area is in degrees²
        /// and needs the latitude correction before it means anything.
        latlong: bool,
    },
}

/// Ground area of a feature for per-area normalization.
///
/// Projected data is already in ground units and is measured as it is —
/// the mapping agency picked that CRS. Geographic data is projected onto
/// an equal-area CRS first, because a shoelace over degrees measures
/// nothing comparable between latitudes.
pub(crate) fn ground_area(g: &geo_types::Geometry<f64>, latlong: bool) -> f64 {
    use geo::Area;
    if !latlong {
        return g.unsigned_area();
    }
    let mut g = g.clone();
    let dst = crate::data::crs::equal_area_measure();
    let src = crate::data::crs::wgs84_cached();
    let failed = std::cell::Cell::new(false);
    use geo::MapCoordsInPlace;
    g.map_coords_in_place(
        |c| match crate::data::crs::transform_point(src, dst, c.x, c.y) {
            Ok((x, y)) => geo_types::Coord { x, y },
            Err(_) => {
                // Outside the projection's domain: no usable area, and
                // norm_bin sends a zero area to bin 0 like a null.
                failed.set(true);
                c
            }
        },
    );
    if failed.get() { 0.0 } else { g.unsigned_area() }
}

/// Same, for coordinates already split into x/y slices (the GeoArrow
/// bulk path): returns the scale factor to apply to a shoelace area
/// computed at `(lon, lat)`, or None when the data is already projected.
pub(crate) fn equal_area_projector(
    latlong: bool,
) -> Option<impl Fn(f64, f64) -> Option<(f64, f64)>> {
    if !latlong {
        return None;
    }
    let dst = crate::data::crs::equal_area_measure();
    let src = crate::data::crs::wgs84_cached();
    Some(move |x: f64, y: f64| crate::data::crs::transform_point(src, dst, x, y).ok())
}

/// Bin for an area-normalized value. Nulls, non-finite values and
/// degenerate areas land in bin 0 like nulls do in the plain path.
pub(crate) fn norm_bin(v: f64, area: f64, breaks: &[f64]) -> u8 {
    use crate::data::layer::STYLE_BINS;
    if !v.is_finite() || !(area > 0.0) {
        return 0;
    }
    let x = v / area;
    (breaks.partition_point(|b| x >= *b) as u8).min((STYLE_BINS - 1) as u8)
}

/// Raw numeric per row (NaN for nulls / uncastable columns).
pub(crate) fn batch_values(arr: &arrow::array::ArrayRef) -> Vec<f64> {
    let n = arr.len();
    let vals = arrow::compute::cast(arr, &DataType::Float64).ok();
    let vals = vals
        .as_ref()
        .and_then(|a| a.as_any().downcast_ref::<arrow::array::Float64Array>());
    (0..n)
        .map(|i| match vals {
            Some(v) if !v.is_null(i) => v.value(i),
            _ => f64::NAN,
        })
        .collect()
}

/// Per-row style bins for one batch's value column.
pub(crate) fn batch_bins(arr: &arrow::array::ArrayRef, binning: &Binning) -> Vec<u8> {
    use crate::data::layer::STYLE_BINS;
    let n = arr.len();
    match binning {
        Binning::Breaks(breaks) => {
            let vals = arrow::compute::cast(arr, &DataType::Float64).ok();
            let vals = vals
                .as_ref()
                .and_then(|a| a.as_any().downcast_ref::<arrow::array::Float64Array>());
            (0..n)
                .map(|i| match vals {
                    // A non-finite value would otherwise sort above every
                    // break and land in the top class; the area-normalized
                    // path already sends it to bin 0, and the two must
                    // agree or toggling "normalize by area" would flip a
                    // row from brightest to darkest.
                    Some(v) if !v.is_null(i) && v.value(i).is_finite() => {
                        let x = v.value(i);
                        (breaks.partition_point(|b| x >= *b) as u8)
                            .min((STYLE_BINS - 1) as u8)
                    }
                    _ => 0,
                })
                .collect()
        }
        Binning::Categorical { map } => {
            let vals = arrow::compute::cast(arr, &DataType::Utf8).ok();
            let vals = vals
                .as_ref()
                .and_then(|a| a.as_any().downcast_ref::<arrow::array::StringArray>());
            (0..n)
                .map(|i| match vals {
                    Some(v) if !v.is_null(i) => map
                        .get(v.value(i))
                        .copied()
                        .unwrap_or((STYLE_BINS - 1) as u8),
                    _ => (STYLE_BINS - 1) as u8,
                })
                .collect()
        }
    }
}

// (geometry, rows, bad, computed rg boxes, decode state per selected group).
// State mapping: All → Full, Rect → Rows (or Full when everything matched /
// no covering column), Ranges → Full (append-complement semantics; rebuilds
// ignore it).
type BuildOutput = (
    LayerGeometry,
    usize,
    usize,
    Vec<[f64; 4]>,
    Vec<(u32, GroupLoad)>,
);

/// Maps batch-local row offsets to global file row indices, correct under
/// row selections.
#[derive(Clone)]
enum RowMap {
    /// Contiguous rows from this global start.
    Contiguous(u64),
    /// Selected global rows of the group; the batch starts at `offset`.
    Sparse(Arc<Vec<u32>>, usize),
}

impl RowMap {
    fn global(&self, i: usize) -> u64 {
        match self {
            RowMap::Contiguous(start) => start + i as u64,
            RowMap::Sparse(rows, offset) => rows[offset + i] as u64,
        }
    }
}

/// Stream the selected rows batch by batch, tessellating in parallel (rayon
/// par_bridge). Peak memory is a handful of in-flight batches, not the file.
/// Only the geometry column is decoded. Global row indices stay correct
/// under any selection, so picking and attributes line up.
fn build_geometry(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
    progress: Option<(&LoaderHandle, u64)>,
    sel: Vec<GroupSel>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
    style: Option<&StyleSel>,
) -> Result<BuildOutput, String> {
    let cancelled = || {
        cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    };
    // Projected columns: geometry (+ the styling value column), sorted —
    // remember where each lands in the batches.
    let mut proj: Vec<usize> = vec![store.geom_col];
    if let Some(st) = style {
        if st.col != store.geom_col {
            proj.push(st.col);
        }
    }
    proj.sort_unstable();
    let geom_pos = proj.binary_search(&store.geom_col).unwrap();
    let style_pos = style.map(|st| proj.binary_search(&st.col).unwrap());
    // Second projection for box selections: the covering column stands in
    // for the geometry, so the geometry column is never touched. Four
    // doubles a row against, on land cover, three kilobytes.
    let covering = store.covering.clone();
    let mut box_proj: Vec<usize> = covering.iter().map(|c| c.root).collect();
    if let Some(st) =
        style.filter(|st| covering.as_ref().is_some_and(|c| st.col != c.root))
    {
        box_proj.push(st.col);
    }
    box_proj.sort_unstable();
    let box_pos = covering
        .as_ref()
        .map(|c| box_proj.binary_search(&c.root).unwrap());
    let box_style_pos = style
        .zip(covering.as_ref())
        .map(|(st, _)| box_proj.binary_search(&st.col).unwrap());
    // One chunk grid for the whole build: the key is a cell index, so
    // two grids in one mesh would collide. A build carrying any box
    // group takes the coarse grid — that is the build that would
    // otherwise open tens of thousands of GPU buffers.
    let cell = if sel
        .iter()
        .any(|s| matches!(s, GroupSel::Boxes { .. } | GroupSel::BoxRanges { .. }))
    {
        crate::data::geometry::BOX_CHUNK_WORLD
    } else {
        crate::data::geometry::CHUNK_WORLD
    };
    // A fill-only palette needs no outlines built (boxes never carry any).
    let outlines = style.is_none_or(|st| st.outlines);
    let rg_starts = store.rg_starts();
    let proj_ref: &[usize] = &proj;
    let box_proj_ref: &[usize] = &box_proj;
    let stream_error: Mutex<Option<String>> = Mutex::new(None);
    let resolved: Mutex<Vec<(u32, GroupLoad)>> = Mutex::new(Vec::with_capacity(sel.len()));

    // Upper bound (Rect selections resolve smaller).
    let total: u64 = sel
        .iter()
        .map(|s| {
            let g = s.group() as usize;
            rg_starts[g + 1] - rg_starts[g]
        })
        .sum::<u64>()
        .max(1);
    let done = AtomicUsize::new(0);

    let encoding = store.encoding;
    // Spherical edges only mean something on a geographic CRS.
    let spherical = store.spherical_edges && crs.is_latlong;
    let err_ref = &stream_error;
    let resolved_ref = &resolved;
    let starts = rg_starts;
    // One task per group: group reads run concurrently (essential for
    // remote sources, where each group is a series of range requests) and
    // the flattened batches then tessellate in parallel.
    let per_group = move |job: GroupSel| {
        let g = job.group();
        let start = starts[g as usize];
        let group_rows = (starts[g as usize + 1] - start) as u32;
        // Cancelled work resolves to an empty selection rather than
        // returning early: every path has to yield the same iterator
        // type, and an empty one reads nothing.
        let stop = cancelled();
        // Resolve the job to optional group-relative ranges + final state.
        let (ranges, state): (Option<Vec<(u32, u32)>>, GroupLoad) = if stop {
            (Some(Vec::new()), GroupLoad::None)
        } else {
            match &job {
                GroupSel::All(_) => (None, GroupLoad::Full),
                GroupSel::Ranges(_, r) => (Some(r.clone()), GroupLoad::Full),
                GroupSel::ResolvedRect { rect, ranges, .. } => (
                    Some(ranges.clone()),
                    GroupLoad::Rows {
                        ranges: ranges.clone(),
                        rect: *rect,
                    },
                ),
                GroupSel::Rect(_, rect) => {
                    match covering_select(store, g, *rect) {
                        Ok(Some(r)) if r == [(0, group_rows)] => (None, GroupLoad::Full),
                        Ok(Some(r)) => (
                            Some(r.clone()),
                            GroupLoad::Rows {
                                ranges: r,
                                rect: *rect,
                            },
                        ),
                        Ok(None) => (None, GroupLoad::Full),
                        Err(e) => {
                            *err_ref.lock().unwrap() = Some(e);
                            (Some(vec![]), GroupLoad::None)
                        }
                    }
                }
                GroupSel::Preview { rect, stride, .. } => {
                    // Optional rect filter first, then every stride-th row of
                    // the selection as explicit 1-row ranges (the reader skips
                    // decoding the rest).
                    let base: Option<Vec<(u32, u32)>> = match rect {
                        Some(r) => match covering_select(store, g, *r) {
                            Ok(v) => v,
                            Err(e) => {
                                *err_ref.lock().unwrap() = Some(e);
                                Some(vec![])
                            }
                        },
                        None => None,
                    };
                    let stride = (*stride).max(2) as usize;
                    let sampled: Vec<u32> = match &base {
                        Some(rs) => rs
                            .iter()
                            .flat_map(|&(s, e)| s..e)
                            .step_by(stride)
                            .collect(),
                        None => (0..group_rows).step_by(stride).collect(),
                    };
                    (
                        Some(rows_to_ranges(sampled.into_iter())),
                        GroupLoad::Preview { stride: stride as u32, rect: *rect },
                    )
                }
                GroupSel::Boxes { rect, .. } => {
                    // The row selection a rect load would make; only the
                    // columns read differ.
                    let ranges = match rect {
                        Some(r) => match covering_select(store, g, *r) {
                            Ok(v) => v,
                            Err(e) => {
                                *err_ref.lock().unwrap() = Some(e);
                                Some(vec![])
                            }
                        },
                        None => None,
                    };
                    (ranges, GroupLoad::Boxes { rect: *rect })
                }
                // Gap filler beside a refined group: the group's own state
                // belongs to the geometry selection, so this one records
                // nothing.
                GroupSel::BoxRanges { ranges, .. } => (Some(ranges.clone()), GroupLoad::None),
            }
        };
        let boxes = matches!(job, GroupSel::Boxes { .. } | GroupSel::BoxRanges { .. });
        // A gap filler must not overwrite the state its group already has.
        let record = !matches!(job, GroupSel::BoxRanges { .. });
        if !stop && record {
            resolved_ref.lock().unwrap().push((g, state));
        }
        // Global rows of the selection, for sparse batches.
        let sparse: Option<Arc<Vec<u32>>> = ranges.as_ref().map(|rs| {
            Arc::new(
                rs.iter()
                    .flat_map(|&(s, e)| (start as u32 + s)..(start as u32 + e))
                    .collect::<Vec<u32>>(),
            )
        });
        let reader = if sparse.as_ref().is_some_and(|s| s.is_empty()) {
            None
        } else {
            // Batch by bytes, not by rows: this group's own footer says
            // how heavy its rows are.
            let batch_rows = if boxes {
                BATCH_SIZE
            } else {
                store.batch_rows(g as usize, BATCH_TARGET_BYTES, BATCH_SIZE)
            };
            match store.reader_for_group(
                g as usize,
                batch_rows,
                ranges.as_deref(),
                Some(if boxes { box_proj_ref } else { proj_ref }),
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    *err_ref.lock().unwrap() = Some(e);
                    None
                }
            }
        };
        // Yielded lazily: the worker tessellates each batch and drops it
        // before pulling the next, so a group's decoded rows are never
        // all alive at once. Collecting them here would put a whole row
        // group per worker in memory, which on heavy geometry is
        // hundreds of megabytes per core.
        let err_ref = err_ref;
        let cancel = cancel;
        let mut consumed = 0usize;
        reader.into_iter().flatten().map_while(move |res| {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                return None;
            }
            match res {
                Ok(batch) => {
                    let map = match &sparse {
                        None => RowMap::Contiguous(start + consumed as u64),
                        Some(rows) => RowMap::Sparse(rows.clone(), consumed),
                    };
                    consumed += batch.num_rows();
                    Some((map, batch, boxes))
                }
                Err(e) => {
                    *err_ref.lock().unwrap() = Some(e.to_string());
                    None
                }
            }
        })
    };

    let (builder, items, rows, bad, rg_boxes) = sel
        .into_par_iter()
        // flat_map_iter keeps each group's batches on the worker that
        // read them and pulls them one at a time; flat_map would collect
        // the group first.
        .flat_map_iter(per_group)
        // One accumulator per worker rather than one mesh per batch: the
        // merge tree used to copy every vertex through log(batches)
        // levels, doubling peak memory on a heavy layer.
        .fold(
            || {
                (
                    MeshBuilder::new(cell, outlines),
                    Vec::new(),
                    0usize,
                    0usize,
                    Default::default(),
                )
            },
            |(mut mb, mut items, rows_acc, mut bad, mut rg_boxes): (
                MeshBuilder,
                Vec<PickItem>,
                usize,
                usize,
                std::collections::HashMap<u32, [f64; 4]>,
            ),
             (map, batch, boxes)| {
                let tr = BulkTransformer::new(crs, display);
                let rows = if boxes {
                    process_box_batch(
                        &batch, &map, &tr, display, &mut mb, &mut items, &mut bad,
                        rg_starts, &mut rg_boxes, box_pos.unwrap_or(0),
                        covering.as_ref().map(|c| &c.children),
                        style.map(|st| (box_style_pos.unwrap(), &st.binning, st.per_area)),
                        crs.is_latlong,
                    )
                } else {
                    process_batch(
                        &batch, &map, encoding, &tr, display, &mut mb, &mut items, &mut bad,
                        rg_starts, &mut rg_boxes, geom_pos,
                        style.map(|st| (style_pos.unwrap(), &st.binning, st.per_area)),
                        crs.is_latlong,
                        spherical,
                    )
                };
                if let Some((handle, job)) = progress {
                    let d = done.fetch_add(rows, Ordering::Relaxed) + rows;
                    // The parallel decode+tessellate pass is ~70% of a
                    // load; chunking and the pick index follow
                    // single-threaded.
                    handle.send(LoadMsg::Progress {
                        job,
                        frac: 0.70 * (d as f32 / total as f32).min(1.0),
                        stage: "decoding & tessellating".into(),
                    });
                }
                (mb, items, rows_acc + rows, bad, rg_boxes)
            },
        )
        .reduce(
            || {
                (
                    MeshBuilder::new(cell, outlines),
                    Vec::new(),
                    0usize,
                    0usize,
                    Default::default(),
                )
            },
            |(mut mb1, mut it1, r1, b1, mut g1), (mb2, mut it2, r2, b2, g2)| {
                mb1.merge_into(mb2);
                it1.append(&mut it2);
                for (k, b) in g2 {
                    g1.entry(k)
                        .and_modify(|a: &mut [f64; 4]| {
                            a[0] = a[0].min(b[0]);
                            a[1] = a[1].min(b[1]);
                            a[2] = a[2].max(b[2]);
                            a[3] = a[3].max(b[3]);
                        })
                        .or_insert(b);
                }
                (mb1, it1, r1 + r2, b1 + b2, g1)
            },
        );

    if cancelled() {
        return Err(CANCELLED.into());
    }
    if let Some(e) = stream_error.lock().unwrap().take() {
        return Err(format!("parquet decode error: {e}"));
    }

    let bounds = if builder.bounds[0].is_finite() {
        builder.bounds
    } else {
        [0.0, 0.0, 1.0, 1.0]
    };
    let kind = builder.kind;
    if builder.fill_errors > 0 {
        log::warn!(
            "{}: {} polygons failed tessellation (rendered outline-only)",
            store.source.label(),
            builder.fill_errors
        );
    }
    let bad = bad + builder.fill_errors;
    if let Some((handle, job)) = progress {
        handle.send(LoadMsg::Progress {
            job,
            frac: 0.78,
            stage: "chunking meshes".into(),
        });
    }
    let chunks = builder.finish();
    if let Some((handle, job)) = progress {
        handle.send(LoadMsg::Progress {
            job,
            frac: 0.90,
            stage: "building pick index".into(),
        });
    }
    let rtree = RTree::bulk_load(items);

    // Index-aligned with the file's row groups: a group whose selected
    // rows yielded no decodable geometry keeps a sentinel box, so
    // consumers can index the vector by global group id.
    let rg_vec: Vec<[f64; 4]> = (0..rg_starts.len().saturating_sub(1))
        .map(|i| rg_boxes.get(&(i as u32)).copied().unwrap_or(EMPTY_BBOX))
        .collect();

    Ok((
        LayerGeometry {
            chunks: Arc::new(chunks),
            rtree: Arc::new(rtree),
            bounds_world: bounds,
            kind,
        },
        rows,
        bad,
        rg_vec,
        resolved.into_inner().unwrap(),
    ))
}

fn rg_of(row: u64, rg_starts: &[u64]) -> u32 {
    match rg_starts.binary_search(&row) {
        Ok(i) => i as u32,
        Err(i) => (i - 1) as u32,
    }
}

fn grow_rg_box(
    rg_boxes: &mut std::collections::HashMap<u32, [f64; 4]>,
    rg: u32,
    b: [f64; 4],
) {
    rg_boxes
        .entry(rg)
        .and_modify(|a| {
            a[0] = a[0].min(b[0]);
            a[1] = a[1].min(b[1]);
            a[2] = a[2].max(b[2]);
            a[3] = a[3].max(b[3]);
        })
        .or_insert(b);
}

#[allow(clippy::too_many_arguments)]
fn process_batch(
    batch: &RecordBatch,
    map: &RowMap,
    encoding: GeomEncoding,
    tr: &BulkTransformer,
    display: &DisplayCrs,
    mb: &mut MeshBuilder,
    items: &mut Vec<PickItem>,
    bad: &mut usize,
    rg_starts: &[u64],
    rg_boxes: &mut std::collections::HashMap<u32, [f64; 4]>,
    geom_pos: usize,
    style: Option<(usize, &Binning, bool)>,
    crs_latlong: bool,
    spherical: bool,
) -> usize {
    let col = batch.column(geom_pos);
    let Some(get) = GeomCol::new(col.as_ref(), encoding) else {
        *bad += batch.num_rows();
        return batch.num_rows();
    };
    // Data-driven styling: per-row bin from the value column; the mesh
    // builder keys chunks by (cell, bin). Area-normalized bins can only
    // be computed once the feature's geometry is decoded.
    let bins: Option<RowBins> = style.map(|(pos, binning, per_area)| match binning {
        Binning::Breaks(breaks) if per_area => RowBins::PerArea {
            vals: batch_values(batch.column(pos)),
            breaks: breaks.clone(),
            latlong: crs_latlong,
        },
        _ => RowBins::Pre(batch_bins(batch.column(pos), binning)),
    });

    // GeoArrow: bulk path — one linear reprojection pass over the whole
    // coordinate buffer, features emitted straight from the arrow offsets.
    // Spherical-edges data skips it: densification needs per-feature
    // geometry (the per-row path below handles GeoArrow too).
    if let (GeomCol::Ga(ga), false) = (&get, spherical) {
        return super::geoarrow::emit_bulk(
            ga,
            batch.num_rows(),
            &|i| map.global(i),
            tr,
            display,
            mb,
            items,
            bad,
            &mut |global, b| grow_rg_box(rg_boxes, rg_of(global, rg_starts), b),
            bins.as_ref(),
        );
    }

    for row in 0..batch.num_rows() {
        if get.is_null(row) {
            continue;
        }
        if let Some(RowBins::Pre(b)) = &bins {
            mb.bin = b[row];
        }
        let global = map.global(row);
        let fref = FeatureRef {
            index: global as u32,
        };

        // Fast path: 2D point (WKB parse or GeoArrow coordinate read), no
        // per-feature geo allocation.
        if let Some((x, y)) = get.point2(row) {
            if let Some(RowBins::PerArea { vals, breaks, .. }) = &bins {
                // Points have no area; norm_bin sends them to bin 0.
                mb.bin = norm_bin(vals[row], 1.0, breaks);
            }
            grow_rg_box(rg_boxes, rg_of(global, rg_starts), [x, y, x, y]);
            let (mut px, mut py) = (x, y);
            if !tr.apply(&mut px, &mut py) {
                *bad += 1;
                continue;
            }
            let w = display.world_from_projected(px, py);
            let g = geo_types::Geometry::Point(geo_types::Point::new(w[0], w[1]));
            mb.add(&g, fref);
            continue;
        }

        match get.geometry(row) {
            Some(mut geom) => {
                if let Some(RowBins::PerArea {
                    vals,
                    breaks,
                    latlong,
                }) = &bins
                {
                    mb.bin = norm_bin(vals[row], ground_area(&geom, *latlong), breaks);
                }
                if spherical {
                    densify_spherical(&mut geom, SPHERICAL_MAX_SEG_DEG);
                }
                {
                    use geo::BoundingRect;
                    if let Some(r) = geom.bounding_rect() {
                        let (min, max) = (r.min(), r.max());
                        if min.x.is_finite() && max.y.is_finite() {
                            grow_rg_box(
                                rg_boxes,
                                rg_of(global, rg_starts),
                                [min.x, min.y, max.x, max.y],
                            );
                        }
                    }
                }
                let failed = std::cell::Cell::new(false);
                geom.map_coords_in_place(|c| {
                    let (mut x, mut y) = (c.x, c.y);
                    if !tr.apply(&mut x, &mut y) {
                        failed.set(true);
                    }
                    let w = display.world_from_projected(x, y);
                    geo_types::Coord { x: w[0], y: w[1] }
                });
                if failed.get() {
                    *bad += 1;
                    continue;
                }
                if let Some(added) = mb.add(&geom, fref) {
                    if added.needs_rtree {
                        items.push(PickItem {
                            bbox: added.bbox,
                            feature: fref,
                        });
                    }
                }
            }
            None => *bad += 1,
        }
    }
    batch.num_rows()
}

/// Batch path for `GroupSel::Boxes`: every feature drawn as its covering
/// rectangle, from four doubles a row.
///
/// The rectangle is projected by its corners. On a feature small enough
/// for its box to stand in for it that is exact to well under a pixel;
/// on a large one it is not, which is the other half of why boxes are a
/// wide-zoom state that refinement replaces.
#[allow(clippy::too_many_arguments)]
fn process_box_batch(
    batch: &RecordBatch,
    map: &RowMap,
    tr: &BulkTransformer,
    display: &DisplayCrs,
    mb: &mut MeshBuilder,
    items: &mut Vec<PickItem>,
    bad: &mut usize,
    rg_starts: &[u64],
    rg_boxes: &mut std::collections::HashMap<u32, [f64; 4]>,
    box_pos: usize,
    children: Option<&[String; 4]>,
    style: Option<(usize, &Binning, bool)>,
    crs_latlong: bool,
) -> usize {
    use arrow::array::{Array, Float64Array, StructArray};
    let n = batch.num_rows();
    let Some(children) = children else {
        *bad += n;
        return n;
    };
    let Some(st) = batch.column(box_pos).as_any().downcast_ref::<StructArray>() else {
        *bad += n;
        return n;
    };
    let leaf = |name: &str| -> Option<Float64Array> {
        let col = st.column_by_name(name)?;
        arrow::compute::cast(col, &arrow::datatypes::DataType::Float64)
            .ok()
            .map(|a| a.as_any().downcast_ref::<Float64Array>().unwrap().clone())
    };
    let (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) = (
        leaf(&children[0]),
        leaf(&children[1]),
        leaf(&children[2]),
        leaf(&children[3]),
    ) else {
        *bad += n;
        return n;
    };
    let bins: Option<RowBins> = style.map(|(pos, binning, per_area)| match binning {
        Binning::Breaks(breaks) if per_area => RowBins::PerArea {
            vals: batch_values(batch.column(pos)),
            breaks: breaks.clone(),
            latlong: crs_latlong,
        },
        _ => RowBins::Pre(batch_bins(batch.column(pos), binning)),
    });
    for row in 0..n {
        // A feature with no geometry has no box either, and that is the
        // file stating a fact rather than failing: the geometry path
        // skips null geometries silently and this one must agree, or a
        // dataset with a single empty row reports a decode error it does
        // not have. Present-but-unusable values below are a real fault.
        if st.is_null(row)
            || xmin.is_null(row)
            || ymin.is_null(row)
            || xmax.is_null(row)
            || ymax.is_null(row)
        {
            continue;
        }
        let (x0, y0, x1, y1) = (
            xmin.value(row),
            ymin.value(row),
            xmax.value(row),
            ymax.value(row),
        );
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            *bad += 1;
            continue;
        }
        let global = map.global(row);
        let fref = FeatureRef { index: global as u32 };
        grow_rg_box(rg_boxes, rg_of(global, rg_starts), [x0, y0, x1, y1]);
        match &bins {
            Some(RowBins::Pre(b)) => mb.bin = b[row],
            // Area normalization falls back to the box's own area, which
            // is what there is to work with before the geometry is read.
            Some(RowBins::PerArea { vals, breaks, latlong }) => {
                let poly = geo_types::Geometry::Polygon(geo_types::Polygon::new(
                    geo_types::LineString::from(vec![
                        (x0, y0),
                        (x1, y0),
                        (x1, y1),
                        (x0, y1),
                        (x0, y0),
                    ]),
                    Vec::new(),
                ));
                mb.bin = norm_bin(vals[row], ground_area(&poly, *latlong), breaks);
            }
            None => {}
        }
        // Project the two corners only. The box is axis-aligned in the
        // data CRS and stays a box on screen at the scale it is used;
        // carrying it through geo/lyon instead costs microseconds and
        // kilobytes of churn per feature, times millions of features.
        let (mut px0, mut py0, mut px1, mut py1) = (x0, y0, x1, y1);
        if !tr.apply(&mut px0, &mut py0) || !tr.apply(&mut px1, &mut py1) {
            *bad += 1;
            continue;
        }
        let w0 = display.world_from_projected(px0, py0);
        let w1 = display.world_from_projected(px1, py1);
        let rect = [
            w0[0].min(w1[0]),
            w0[1].min(w1[1]),
            w0[0].max(w1[0]),
            w0[1].max(w1[1]),
        ];
        if !rect.iter().all(|v| v.is_finite()) {
            *bad += 1;
            continue;
        }
        let added = mb.add_rect(rect);
        items.push(PickItem {
            bbox: added.bbox,
            feature: fref,
        });
    }
    n
}

/// Accessor over the three arrow binary array flavors.
pub enum BinCol<'a> {
    Bin(&'a BinaryArray),
    Large(&'a LargeBinaryArray),
    View(&'a BinaryViewArray),
}

impl<'a> BinCol<'a> {
    pub fn new(col: &'a dyn Array) -> Option<Self> {
        match col.data_type() {
            DataType::Binary => Some(Self::Bin(col.as_any().downcast_ref()?)),
            DataType::LargeBinary => Some(Self::Large(col.as_any().downcast_ref()?)),
            DataType::BinaryView => Some(Self::View(col.as_any().downcast_ref()?)),
            _ => None,
        }
    }

    pub fn value(&self, i: usize) -> Option<&'a [u8]> {
        match self {
            Self::Bin(a) => (!a.is_null(i)).then(|| a.value(i)),
            Self::Large(a) => (!a.is_null(i)).then(|| a.value(i)),
            Self::View(a) => (!a.is_null(i)).then(|| a.value(i)),
        }
    }
}

/// Grow `b` by one WKB value's envelope: a direct byte scan with no
/// geo-types allocation. Handles both endiannesses, ISO Z/M/ZM type
/// codes, EWKB flag bits (+SRID), and nested multis/collections.
/// Returns None on malformed input (the caller skips the feature).
pub(crate) fn grow_wkb_envelope(buf: &[u8], b: &mut [f64; 4]) -> Option<()> {
    fn geom(buf: &[u8], pos: &mut usize, b: &mut [f64; 4], depth: u8) -> Option<()> {
        if depth > 8 {
            return None;
        }
        let le = match *buf.get(*pos)? {
            0 => false,
            1 => true,
            _ => return None,
        };
        *pos += 1;
        let read_u32 = |p: &mut usize| -> Option<u32> {
            let s: [u8; 4] = buf.get(*p..*p + 4)?.try_into().ok()?;
            *p += 4;
            Some(if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
        };
        let mut ty = read_u32(pos)?;
        let (mut z, mut m) = (ty & 0x8000_0000 != 0, ty & 0x4000_0000 != 0);
        if ty & 0x2000_0000 != 0 {
            *pos += 4; // EWKB SRID
        }
        ty &= 0x0FFF_FFFF;
        match (ty / 1000) % 10 {
            1 => z = true,
            2 => m = true,
            3 => (z, m) = (true, true),
            _ => {}
        }
        let dims = 2 + z as usize + m as usize;
        let mut coords = |n: usize, pos: &mut usize| -> Option<()> {
            let bytes = n.checked_mul(dims * 8)?;
            let end = pos.checked_add(bytes).filter(|&e| e <= buf.len())?;
            for c in buf[*pos..end].chunks_exact(dims * 8) {
                let f = |s: &[u8]| -> f64 {
                    let a: [u8; 8] = s.try_into().unwrap();
                    if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }
                };
                let (x, y) = (f(&c[0..8]), f(&c[8..16]));
                if x.is_finite() && y.is_finite() {
                    b[0] = b[0].min(x);
                    b[1] = b[1].min(y);
                    b[2] = b[2].max(x);
                    b[3] = b[3].max(y);
                }
            }
            *pos = end;
            Some(())
        };
        match ty % 1000 {
            1 => coords(1, pos),
            2 => {
                let n = read_u32(pos)? as usize;
                coords(n, pos)
            }
            3 => {
                let rings = read_u32(pos)? as usize;
                for _ in 0..rings {
                    let n = read_u32(pos)? as usize;
                    coords(n, pos)?;
                }
                Some(())
            }
            4..=7 => {
                let n = read_u32(pos)? as usize;
                for _ in 0..n {
                    geom(buf, pos, b, depth + 1)?;
                }
                Some(())
            }
            _ => None,
        }
    }
    let mut pos = 0;
    geom(buf, &mut pos, b, 0)
}

/// Whole-dataset extent from a geometry-only WKB scan: no tessellation,
/// no per-feature allocations, rayon over row groups. Cheap enough to
/// run before the first build so the display projection can be adopted
/// up front — a post-build adoption would tessellate, upload and hold
/// the whole layer twice (the 15 GB peak on large unindexed files).
fn scan_wkb_extent(
    store: &FeatureStore,
    cancel: &std::sync::atomic::AtomicBool,
) -> Option<[f64; 4]> {
    use rayon::prelude::*;
    let n_rg = store.rg_starts().len().saturating_sub(1);
    let boxes: Vec<[f64; 4]> = (0..n_rg)
        .into_par_iter()
        .map(|g| {
            let mut b = EMPTY_BBOX;
            if cancel.load(Ordering::Relaxed) {
                return b;
            }
            let Ok(reader) = store.reader_for_group(g, 4096, None, Some(&[store.geom_col]))
            else {
                return b;
            };
            for batch in reader {
                let Ok(batch) = batch else { break };
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let Some(col) = BinCol::new(batch.column(0).as_ref()) else {
                    break;
                };
                for i in 0..batch.num_rows() {
                    if let Some(buf) = col.value(i) {
                        let _ = grow_wkb_envelope(buf, &mut b);
                    }
                }
            }
            b
        })
        .collect();
    union_of(&boxes)
}

/// Decode WKB into geo-types (drops Z/M).
/// Max great-circle arc per segment for `edges: spherical` data, in
/// degrees. At 1° the chord-vs-arc deviation is negligible at any zoom
/// where the curve is distinguishable.
const SPHERICAL_MAX_SEG_DEG: f64 = 1.0;

/// Densify long segments along great circles: spherical edges project
/// as curves, not straight chords. Only segments spanning more than
/// `max_deg` of arc gain vertices (slerp on the unit sphere).
pub(crate) fn densify_spherical(g: &mut geo_types::Geometry<f64>, max_deg: f64) {
    use geo_types::Geometry::*;
    match g {
        Point(_) | MultiPoint(_) | Line(_) | Rect(_) | Triangle(_) => {}
        LineString(ls) => densify_ls(ls, max_deg),
        MultiLineString(mls) => mls.0.iter_mut().for_each(|l| densify_ls(l, max_deg)),
        Polygon(p) => densify_poly(p, max_deg),
        MultiPolygon(mp) => mp.0.iter_mut().for_each(|p| densify_poly(p, max_deg)),
        GeometryCollection(gc) => {
            gc.0.iter_mut().for_each(|g| densify_spherical(g, max_deg))
        }
    }
}

fn densify_poly(p: &mut geo_types::Polygon<f64>, max_deg: f64) {
    p.exterior_mut(|e| densify_ls(e, max_deg));
    p.interiors_mut(|ints| ints.iter_mut().for_each(|l| densify_ls(l, max_deg)));
}

fn densify_ls(ls: &mut geo_types::LineString<f64>, max_deg: f64) {
    let pts = &ls.0;
    if pts.len() < 2 {
        return;
    }
    let mut out: Vec<geo_types::Coord<f64>> = Vec::with_capacity(pts.len());
    out.push(pts[0]);
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (va, vb) = (sphere_unit(a), sphere_unit(b));
        let dot = (va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]).clamp(-1.0, 1.0);
        let omega = dot.acos();
        let arc_deg = omega.to_degrees();
        if arc_deg > max_deg && omega.sin() > 1e-9 {
            let n = (arc_deg / max_deg).ceil() as usize;
            let so = omega.sin();
            for k in 1..n {
                let t = k as f64 / n as f64;
                let (s1, s2) = (((1.0 - t) * omega).sin() / so, (t * omega).sin() / so);
                out.push(sphere_lonlat([
                    va[0] * s1 + vb[0] * s2,
                    va[1] * s1 + vb[1] * s2,
                    va[2] * s1 + vb[2] * s2,
                ]));
            }
        }
        out.push(b);
    }
    if out.len() > pts.len() {
        ls.0 = out;
    }
}

fn sphere_unit(c: geo_types::Coord<f64>) -> [f64; 3] {
    let (lon, lat) = (c.x.to_radians(), c.y.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

fn sphere_lonlat(v: [f64; 3]) -> geo_types::Coord<f64> {
    geo_types::Coord {
        x: v[1].atan2(v[0]).to_degrees(),
        y: v[2].atan2((v[0] * v[0] + v[1] * v[1]).sqrt()).to_degrees(),
    }
}

pub fn decode_wkb(buf: &[u8]) -> Option<geo_types::Geometry<f64>> {
    let wkb = wkb::reader::read_wkb(buf).ok()?;
    wkb.try_to_geometry()
}

/// Fast parse of a plain 2D (or 2D+SRID) WKB point. Returns None for
/// anything else, falling back to the generic decoder.
pub(crate) fn parse_wkb_point_2d(buf: &[u8]) -> Option<(f64, f64)> {
    if buf.len() < 21 {
        return None;
    }
    let le = match buf[0] {
        0 => false,
        1 => true,
        _ => return None,
    };
    let read_u32 = |b: &[u8]| -> u32 {
        let arr: [u8; 4] = b[..4].try_into().unwrap();
        if le {
            u32::from_le_bytes(arr)
        } else {
            u32::from_be_bytes(arr)
        }
    };
    let read_f64 = |b: &[u8]| -> f64 {
        let arr: [u8; 8] = b[..8].try_into().unwrap();
        if le {
            f64::from_le_bytes(arr)
        } else {
            f64::from_be_bytes(arr)
        }
    };
    let ty = read_u32(&buf[1..5]);
    let mut off = 5;
    // EWKB SRID flag
    if ty & 0x2000_0000 != 0 {
        off += 4;
    }
    let base = ty & 0x0FFF_FFFF & !0x2000_0000;
    if base != 1 {
        return None; // not a plain 2D point (Z/M points go through the generic path)
    }
    if buf.len() < off + 16 {
        return None;
    }
    let x = read_f64(&buf[off..]);
    let y = read_f64(&buf[off + 8..]);
    Some((x, y))
}

/// Test-only re-exports for headless benchmarks.
#[cfg(test)]
pub fn open_store_for_test(path: &std::path::PathBuf) -> Result<StoreOpen, String> {
    open_store(&Source::Local(path.clone()))
}

#[cfg(test)]
pub fn open_source_for_test(source: &Source) -> Result<StoreOpen, String> {
    open_store(source)
}


/// All groups of a store, unselected.
#[cfg(test)]
fn all_groups(store: &FeatureStore) -> Vec<GroupSel> {
    (0..store.rg_starts().len().saturating_sub(1) as u32)
        .map(GroupSel::All)
        .collect()
}

#[cfg(test)]
pub fn build_geometry_for_test_full(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
) -> Result<BuildOutput, String> {
    build_geometry(store, crs, display, None, all_groups(store), None, None)
}

#[cfg(test)]
pub fn build_geometry_styled_for_test(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
    style: &StyleSel,
) -> Result<BuildOutput, String> {
    build_geometry(store, crs, display, None, all_groups(store), None, Some(style))
}

#[cfg(test)]
pub fn build_geometry_for_test(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
) -> Result<(super::layer::LayerGeometry, usize, usize), String> {
    build_geometry(store, crs, display, None, all_groups(store), None, None)
        .map(|(g, r, b, _, _)| (g, r, b))
}

/// The first `groups` row groups, from their covering boxes or from
/// their geometry. Bounded on purpose: the geometry side of a land-cover
/// file is gigabytes, and a test that quietly took every group would be
/// the very thing this path exists to avoid.
#[cfg(test)]
pub fn build_selection_for_test(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
    groups: usize,
    boxes: bool,
    style: Option<&StyleSel>,
) -> Result<(super::layer::LayerGeometry, usize, usize), String> {
    let sel: Vec<GroupSel> = (0..groups as u32)
        .map(|g| {
            if boxes {
                GroupSel::Boxes { group: g, rect: None }
            } else {
                GroupSel::All(g)
            }
        })
        .collect();
    build_geometry(store, crs, display, None, sel, None, style).map(|(g, r, b, _, _)| (g, r, b))
}

#[cfg(test)]
mod bbox_transform_tests {
    use super::*;

    /// data → world → data must come back around the original bbox
    /// (both directions sample-and-inflate, so containment, not equality).
    #[test]
    fn data_bbox_world_roundtrip() {
        let l93 = Crs::from_epsg(2154).unwrap();
        let display = DisplayCrs::hobo_dyer();
        // A ~20 km box near Toulouse in Lambert-93.
        let data = [570_000.0, 6_270_000.0, 590_000.0, 6_290_000.0];
        let world = data_bbox_to_world(data, &l93, &display).expect("transforms");
        assert!(world[0] < world[2] && world[1] < world[3]);
        let back = viewport_to_data_bbox(world, &display, &l93).expect("back");
        assert!(
            back[0] <= data[0] && back[1] <= data[1] && back[2] >= data[2] && back[3] >= data[3],
            "roundtrip must cover the original: {back:?} vs {data:?}"
        );
        // And not explode: within ~3x the original span.
        assert!((back[2] - back[0]) < 3.0 * (data[2] - data[0]), "{back:?}");
    }
}

#[cfg(test)]
mod rg_bbox_tests {
    use super::*;

    #[test]
    fn computed_rg_bboxes_cover_data() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/points_1m_wgs84.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, rg_meta) = open_store(&Source::Local(path)).unwrap();
        // DuckDB spatial output: no covering, no native geo stats expected.
        assert!(store.covering.is_none());
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (_geom, rows, _bad, computed, resolved) =
            build_geometry(&store, &crs, &display, None, all_groups(&store), None, None).unwrap();
        assert!(resolved.iter().all(|(_, st)| st.is_full()));
        assert_eq!(rows, 1_000_000);
        let n_rg = store.rg_starts().len() - 1;
        let boxes = match rg_meta {
            Some((_, b)) => b,
            None => computed,
        };
        assert_eq!(boxes.len(), n_rg, "one bbox per row group");
        // Data is uniform in [-5,10]x[41,51]: every rg bbox ≈ full extent
        // and the overlap metric flags it as unclustered.
        for b in &boxes {
            assert!(b[0] >= -5.1 && b[2] <= 10.1 && b[1] >= 40.9 && b[3] <= 51.1, "{b:?}");
            assert!(b[2] - b[0] > 10.0, "unclustered data: near-full extent {b:?}");
        }
        let overlap = bbox_overlap_metric(&boxes);
        assert!(
            overlap > (n_rg - 1) as f64 * 0.9,
            "all boxes overlap: {overlap} vs {n_rg}"
        );
    }
}

#[cfg(test)]
mod sentinel_bbox_tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// A row group whose rows are all null geometry must keep its slot in
    /// the computed bbox vector (sentinel), so later groups' boxes stay
    /// addressable by global group id, and sentinel boxes must be inert
    /// for intersection and union.
    #[test]
    fn computed_bboxes_stay_index_aligned_over_empty_groups() {
        let dir = std::env::temp_dir().join("geopq_sentinel_bbox");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nulls.parquet");

        // 3 groups of 4 rows; group 1 is all-null geometry.
        let wkbs: Vec<Option<Vec<u8>>> = (0..12)
            .map(|i| match i / 4 {
                0 => Some(wkb_point(i as f64, 0.0)),
                1 => None,
                _ => Some(wkb_point(10.0 + (i - 8) as f64, 10.0)),
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, true),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(wkbs.into_iter().collect::<BinaryArray>()),
                Arc::new(Int64Array::from((0..12).collect::<Vec<i64>>())),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let (store, crs, _info, rg_meta) = open_store(&Source::Local(path)).unwrap();
        assert!(rg_meta.is_none(), "plain WKB file has no metadata bboxes");
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (_g, rows, _bad, boxes, _res) =
            build_geometry(&store, &crs, &display, None, all_groups(&store), None, None)
                .unwrap();
        assert_eq!(rows, 12);
        assert_eq!(boxes.len(), 3, "one slot per row group");
        assert_eq!(boxes[1], EMPTY_BBOX, "empty group keeps a sentinel");
        // Real boxes stay at their own group's index.
        assert!(boxes[0][0] >= -0.1 && boxes[0][2] <= 3.1 && boxes[0][3] <= 0.1, "{:?}", boxes[0]);
        assert!(boxes[2][0] >= 9.9 && boxes[2][1] >= 9.9, "{:?}", boxes[2]);
        // Sentinels never intersect and never contribute to unions.
        assert_eq!(intersecting_rgs(&boxes, [-1.0, -1.0, 20.0, 20.0]), vec![0, 2]);
        let u = union_of(&boxes).unwrap();
        assert!(u[0] >= -0.1 && u[1] >= -0.1 && u[2] <= 13.1 && u[3] <= 10.1, "{u:?}");
        assert_eq!(union_of(&[EMPTY_BBOX]), None);
    }
}

#[cfg(test)]
mod geo_stat_normalize_tests {
    use super::*;

    #[test]
    fn antimeridian_wraparound_widens_geographic_boxes() {
        // Ordinary box passes through untouched.
        assert_eq!(
            normalize_geo_stat_bbox([-10.0, 40.0, 5.0, 50.0], true),
            Some([-10.0, 40.0, 5.0, 50.0])
        );
        // Wraparound (xmin > xmax): widened to the full longitude span —
        // still intersects viewports on both sides of the antimeridian.
        let b = normalize_geo_stat_bbox([170.0, -20.0, -170.0, -10.0], true).unwrap();
        assert_eq!(b, [-180.0, -20.0, 180.0, -10.0]);
        assert_eq!(intersecting_rgs(&[b], [175.0, -15.0, 179.0, -12.0]), vec![0]);
        assert_eq!(intersecting_rgs(&[b], [-179.0, -15.0, -175.0, -12.0]), vec![0]);
        // Projected CRS: a wrapped box has no defined span — rejected.
        assert_eq!(normalize_geo_stat_bbox([5.0, 0.0, 1.0, 2.0], false), None);
    }
}

#[cfg(test)]
mod tess_validation {
    use super::*;
    use crate::data::geometry::{FeatureRef, MeshBuilder};

    /// Ground-truth check: tessellated triangle area must match the true
    /// polygon area for real-world parcel data.
    /// Run: GEOPQ_BENCH_FILE=... cargo test --release tessellation_area -- --ignored --nocapture
    #[test]
    #[ignore]
    fn tessellation_area_matches_geometry_area() {
        use geo::Area;
        let Ok(path) = std::env::var("GEOPQ_BENCH_FILE") else {
            return;
        };
        let (store, _crs, _info, _rg) =
            open_store(&Source::Local(path.clone().into())).unwrap();
        let total = store.total_rows() as u32;
        let step = (total / 20_000).max(1);
        let rows: Vec<u32> = (0..total).step_by(step as usize).collect();
        let geoms = store.fetch_geoms(&rows).unwrap();

        let mut checked = 0usize;
        let mut mismatches: Vec<(u32, f64, f64)> = Vec::new();
        for (row, geom) in geoms {
            let Some(geom) = geom else { continue };
            let true_area = match &geom {
                geo_types::Geometry::Polygon(_) | geo_types::Geometry::MultiPolygon(_) => {
                    geom.unsigned_area()
                }
                _ => continue,
            };
            if true_area <= 0.0 {
                continue;
            }
            // Tessellate through the production path (scaled so chunk-local
            // magnitudes match real usage: parcels are ~1e-6 world units).
            let scale = 2.5e-8; // meters -> world units (1 / earth circumference)
            use geo::MapCoordsInPlace;
            let mut g = geom.clone();
            g.map_coords_in_place(|c| geo_types::Coord {
                x: 0.5 + c.x * scale,
                y: 0.3 + c.y * scale,
            });
            let mut mb = MeshBuilder::default();
            mb.add(&g, FeatureRef { index: row });
            let chunks = mb.finish();
            let mut tri_area = 0.0f64;
            for c in &chunks {
                for tri in c.fill_indices.chunks_exact(3) {
                    let a = c.fill_vertices[tri[0] as usize];
                    let b = c.fill_vertices[tri[1] as usize];
                    let d = c.fill_vertices[tri[2] as usize];
                    tri_area += 0.5
                        * ((b[0] - a[0]) as f64 * (d[1] - a[1]) as f64
                            - (d[0] - a[0]) as f64 * (b[1] - a[1]) as f64)
                            .abs();
                }
            }
            let expect = true_area * scale * scale;
            checked += 1;
            let rel = if expect > 0.0 {
                (tri_area - expect).abs() / expect
            } else {
                1.0
            };
            if rel > 0.02 {
                mismatches.push((row, expect, tri_area));
            }
        }
        eprintln!(
            "checked {checked} polygons: {} area mismatches (>2%)",
            mismatches.len()
        );
        for (row, e, t) in mismatches.iter().take(10) {
            eprintln!("  row {row}: expected {e:.3e}, tessellated {t:.3e} ({:.1}%)", t / e * 100.0);
        }
        assert!(
            (mismatches.len() as f64) < checked as f64 * 0.005,
            "{} of {checked} polygons tessellate with wrong area",
            mismatches.len()
        );
    }
}

#[cfg(test)]
mod pruning_tests {
    use super::*;

    /// Manual harness: build geometry over the first `GEOPQ_GROUPS` row
    /// groups of `GEOPQ_FILE` and report rows and wall time. Peak memory
    /// is sampled from outside by the caller.
    ///
    /// The point is that peak footprint must be set by how many workers
    /// are decoding, not by how many row groups the file has. Run it at
    /// several group counts: a flat peak means the pipeline is bounded,
    /// a rising one means something accumulates.
    #[test]
    #[ignore = "needs a local file; run manually under a memory guard"]
    fn bounded_build_over_n_groups() {
        let Ok(file) = std::env::var("GEOPQ_FILE") else {
            eprintln!("set GEOPQ_FILE");
            return;
        };
        let n: usize = std::env::var("GEOPQ_GROUPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let path = std::path::PathBuf::from(&file);
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        let total = store.rg_starts().len() - 1;
        let n = n.min(total);
        let bytes: u64 = (0..n).map(|g| store.rg_geom_bytes(g as u32)).sum();
        eprintln!(
            "{n}/{total} row groups, {:.2} GB uncompressed geometry",
            bytes as f64 / 1e9
        );
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        // GEOPQ_BOXES=1 draws every feature from its covering box instead
        // of reading geometry: the same rows, the other representation.
        let boxes = std::env::var("GEOPQ_BOXES").is_ok_and(|v| v == "1");
        // GEOPQ_STYLE=1 applies the schema's colour map, as an open does.
        let style_by = std::env::var("GEOPQ_STYLE")
            .is_ok_and(|v| v == "1")
            .then(|| crate::data::colormap::schema_style(&store.style_columns()))
            .flatten();
        let style_sel = style_by.as_ref().and_then(|sb| resolve_style(&store, sb));
        eprintln!(
            "style: {}",
            style_sel.as_ref().map_or("none".to_string(), |st| format!(
                "col {} outlines {}",
                st.col, st.outlines
            ))
        );
        let jobs: Vec<GroupSel> = (0..n as u32)
            .map(|g| {
                if boxes {
                    GroupSel::Boxes { group: g, rect: None }
                } else {
                    GroupSel::All(g)
                }
            })
            .collect();
        eprintln!("source: {}", if boxes { "covering boxes" } else { "geometry" });
        // What the planner would decide for a whole-file selection.
        let all_bytes: u64 = (0..total).map(|g| store.rg_geom_bytes(g as u32)).sum();
        eprintln!(
            "planner: covering={} polygons_only={} over_budget={:?}",
            store.covering.is_some(),
            store.polygons_only,
            preview_stride(store.total_rows(), all_bytes),
        );
        let t0 = Instant::now();
        let (geometry, rows, bad, _boxes, _resolved) =
            build_geometry(&store, &crs, &display, None, jobs, None, style_sel.as_ref())
                .unwrap();
        eprintln!("bad geometries: {bad}");
        let ms = t0.elapsed().as_millis();
        let (mut fills, mut lines, mut points) = (0usize, 0usize, 0usize);
        for c in geometry.chunks.iter() {
            let (f, l, p) = c.heap_bytes();
            fills += f;
            lines += l;
            points += p;
        }
        eprintln!(
            "built {rows} rows, {} chunks in {ms} ms",
            geometry.chunks.len()
        );
        eprintln!(
            "mesh: fills {:.0} MB, lines {:.0} MB, points {:.0} MB, total {:.2} GB \
             ({:.1}× the {:.2} GB of source geometry)",
            fills as f64 / 1e6,
            lines as f64 / 1e6,
            points as f64 / 1e6,
            (fills + lines + points) as f64 / 1e9,
            (fills + lines + points) as f64 / bytes.max(1) as f64,
            bytes as f64 / 1e9,
        );
    }

    /// A box build keeps every feature, reads no geometry, and costs a
    /// fraction of the mesh the same rows would tessellate into.
    #[test]
    fn boxes_keep_every_feature_for_a_fraction_of_the_mesh() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.covering.is_some(), "fixture must carry a covering column");
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let groups: Vec<u32> = (0..2.min(store.rg_starts().len() as u32 - 1)).collect();
        let expected: u64 = groups
            .iter()
            .map(|&g| store.rg_starts()[g as usize + 1] - store.rg_starts()[g as usize])
            .sum();

        let boxes: Vec<GroupSel> = groups
            .iter()
            .map(|&g| GroupSel::Boxes { group: g, rect: None })
            .collect();
        let (box_geom, box_rows, bad, _rg, resolved) =
            build_geometry(&store, &crs, &display, None, boxes, None, None).unwrap();
        assert_eq!(box_rows as u64, expected, "every row is represented");
        assert_eq!(bad, 0, "no feature should fail from its own bbox");
        for (_, st) in &resolved {
            assert!(matches!(st, GroupLoad::Boxes { .. }));
            // Boxes are an approximation, so the viewport is never
            // satisfied by them and refinement always has work to do.
            assert!(!st.is_full());
            assert!(!st.covers([0.0, 0.0, 1.0, 1.0]));
        }
        // One pick entry per feature: picking still finds every row.
        assert_eq!(box_geom.rtree.size() as u64, expected);
        // Two triangles a feature, no outlines.
        let (bf, bl, _) = box_geom
            .chunks
            .iter()
            .map(|c| c.heap_bytes())
            .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        assert_eq!(bl, 0, "boxes carry no line segments");
        let quads: usize = box_geom.chunks.iter().map(|c| c.fill_indices.len()).sum();
        assert_eq!(quads as u64, expected * 6);

        let full: Vec<GroupSel> = groups.iter().copied().map(GroupSel::All).collect();
        let (geom, rows, _, _, _) =
            build_geometry(&store, &crs, &display, None, full, None, None).unwrap();
        assert_eq!(rows as u64, expected);
        let (gf, gl, _) = geom
            .chunks
            .iter()
            .map(|c| c.heap_bytes())
            .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        assert!(
            (bf + bl) * 4 < gf + gl,
            "boxes should be far cheaper: {} vs {} bytes",
            bf + bl,
            gf + gl
        );
    }

    /// Box mode draws every selected feature at its covering bbox, with
    /// no geometry read at all, and keeps the pick index exact.
    #[test]
    fn boxes_draw_every_feature_without_reading_geometry() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.covering.is_some(), "fixture must carry a covering column");
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let groups = [0u32, 1];
        let rows_expected: u64 = groups
            .iter()
            .map(|&g| store.rg_starts()[g as usize + 1] - store.rg_starts()[g as usize])
            .sum();

        let jobs: Vec<GroupSel> = groups
            .iter()
            .map(|&g| GroupSel::Boxes { group: g, rect: None })
            .collect();
        let (geom, rows, bad, _boxes, resolved) =
            build_geometry(&store, &crs, &display, None, jobs, None, None).unwrap();
        assert_eq!(rows as u64, rows_expected, "every row is represented");
        assert_eq!(bad, 0);
        for (_, st) in &resolved {
            assert!(
                matches!(st, GroupLoad::Boxes { rect: None }),
                "expected a box state, got {st:?}"
            );
            // A box state never satisfies a viewport, so zooming refines it.
            assert!(!st.covers([0.0, 0.0, 1.0, 1.0]));
        }

        // Two triangles a feature and not one line segment: the outline of
        // a box is not worth storing, and its absence is what makes the
        // mesh two orders of magnitude smaller than the geometry's.
        let tris: usize = geom.chunks.iter().map(|c| c.fill_indices.len() / 3).sum();
        assert_eq!(tris, 2 * rows_expected as usize);
        let segs: usize = geom
            .chunks
            .iter()
            .flat_map(|c| c.lines.iter())
            .map(|l| l.segments.len())
            .sum();
        assert_eq!(segs, 0, "boxes carry no outline");

        // Each feature is indexed at its own covering bbox: picking works
        // exactly as it does on real geometry.
        assert_eq!(geom.rtree.size(), rows_expected as usize);
        let sample: Vec<u32> = (0..rows_expected.min(200) as u32).collect();
        let want: std::collections::HashSet<u32> = sample.iter().copied().collect();
        let mut seen = 0usize;
        for item in geom.rtree.iter() {
            if want.contains(&item.feature.index) {
                seen += 1;
                assert!(item.bbox[0] <= item.bbox[2] && item.bbox[1] <= item.bbox[3]);
            }
        }
        assert_eq!(seen, sample.len(), "every sampled row is in the pick index");
    }

    /// A box layer covers its whole extent from the first frame, even
    /// when it opens while the camera is somewhere else entirely.
    ///
    /// Loading a second layer does not move the camera, so a European
    /// dataset opened over Massachusetts would otherwise select no row
    /// groups at all, and zooming towards Europe would find nothing to
    /// draw until the viewport got small enough for real geometry.
    #[test]
    fn a_box_layer_loads_every_group_wherever_the_camera_is() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, _crs, _info, rg) = open_store(&Source::Local(path)).unwrap();
        let (_, boxes) = rg.expect("row-group boxes");
        let n_rg = store.rg_starts().len() - 1;
        // A viewport on the far side of the planet from the data.
        let elsewhere = [-1.0e7, -1.0e7, -9.9e6, -9.9e6];
        assert!(
            intersecting_rgs(&boxes, elsewhere).is_empty(),
            "the fixture must not intersect this rect"
        );
        let sel = plan_viewport_selection(
            &store,
            "t",
            Some(&boxes),
            Some(elsewhere),
            None,
            "test",
        );
        assert_eq!(sel.len(), n_rg, "every group is planned, not just visible ones");
        assert!(sel.iter().all(|s| matches!(s, GroupSel::Boxes { .. })));
    }

    /// On a box layer every group must draw something, whatever its
    /// decode state. A group that contributes nothing is a hole the size
    /// of a row group, and no camera move fills it: zooming out does not
    /// re-plan a loaded layer.
    #[test]
    fn every_group_of_a_box_layer_draws_something() {
        use super::rebuild_selection;
        let starts = [0u64, 100, 200, 300, 400, 500];
        let loaded = vec![
            // Refined for some viewport.
            GroupLoad::Rows { ranges: vec![(0, 10)], rect: [0.0; 4] },
            // Its bbox met the viewport, none of its features did. This
            // is the state that used to vanish.
            GroupLoad::Rows { ranges: Vec::new(), rect: [0.0; 4] },
            GroupLoad::Boxes { rect: None },
            GroupLoad::Full,
            GroupLoad::None,
        ];
        let sel = rebuild_selection(&loaded, &starts, true);
        for g in [0u32, 1, 2, 3] {
            assert!(
                sel.iter().any(|s| s.group() == g),
                "group {g} contributes nothing: {sel:?}"
            );
        }
        // The partly refined group brings its geometry and its gaps.
        assert!(sel.iter().any(|s| matches!(s, GroupSel::Ranges(0, _))));
        assert!(sel.iter().any(|s| matches!(s, GroupSel::BoxRanges { group: 0, .. })));
        // The empty one falls back to boxes over the whole group.
        assert!(sel
            .iter()
            .any(|s| matches!(s, GroupSel::Boxes { group: 1, rect: None })));
        // A group that was never loaded stays absent — there is nothing
        // to show and nothing was dropped.
        assert!(!sel.iter().any(|s| s.group() == 4));

        // Without boxes to fall back on, an empty selection is still
        // empty: an ordinary indexed layer must not gain phantom boxes.
        let plain = rebuild_selection(&loaded, &starts, false);
        assert!(!plain.iter().any(|s| s.group() == 1));
        assert!(!plain.iter().any(|s| matches!(s, GroupSel::BoxRanges { .. })));
    }

    /// A null geometry is not a decode failure. Real files carry rows
    /// with no shape (MassGIS parcels has exactly one), and the geometry
    /// path skips them silently — the box path has to agree, or opening
    /// such a file reports an error that does not exist.
    #[test]
    fn a_row_without_geometry_is_not_a_bad_geometry() {
        use arrow::array::{ArrayRef, Float64Array, StructArray};
        use arrow::datatypes::{DataType, Field};
        // Three features; the middle one has no box, as a null geometry
        // leaves behind.
        let leaf = |v: [Option<f64>; 3]| Arc::new(Float64Array::from(v.to_vec())) as ArrayRef;
        let fields: Vec<Field> = ["xmin", "ymin", "xmax", "ymax"]
            .iter()
            .map(|n| Field::new(*n, DataType::Float64, true))
            .collect();
        let cols: Vec<ArrayRef> = vec![
            leaf([Some(0.0), None, Some(2.0)]),
            leaf([Some(0.0), None, Some(2.0)]),
            leaf([Some(1.0), None, Some(3.0)]),
            leaf([Some(1.0), None, Some(3.0)]),
        ];
        let st = StructArray::new(fields.clone().into(), cols, None);
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![Field::new(
            "bbox",
            DataType::Struct(fields.into()),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(st) as ArrayRef]).unwrap();

        let crs = Crs::wgs84();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let tr = BulkTransformer::new(&crs, &display);
        let mut mb = MeshBuilder::new(crate::data::geometry::BOX_CHUNK_WORLD, false);
        let mut items: Vec<PickItem> = Vec::new();
        let mut bad = 0usize;
        let mut rg_boxes: std::collections::HashMap<u32, [f64; 4]> = Default::default();
        let children = [
            "xmin".to_string(),
            "ymin".to_string(),
            "xmax".to_string(),
            "ymax".to_string(),
        ];
        let rows = process_box_batch(
            &batch,
            &RowMap::Contiguous(0),
            &tr,
            &display,
            &mut mb,
            &mut items,
            &mut bad,
            &[0, 3],
            &mut rg_boxes,
            0,
            Some(&children),
            None,
            true,
        );
        assert_eq!(rows, 3, "every row is accounted for");
        assert_eq!(bad, 0, "a missing box is not a failure");
        assert_eq!(items.len(), 2, "the two real features are drawn and pickable");
    }

    /// A fill-only palette builds no outlines, and that is most of the
    /// mesh: the same features, same fills, without the LOD stack.
    #[test]
    fn a_fill_only_palette_builds_no_outlines() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/polygons_5k_l93.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let col = store
            .style_columns()
            .into_iter()
            .find(|(_, numeric)| !*numeric)
            .map(|(n, _)| n)
            .expect("a text column to style by");
        let segs = |sel: &StyleSel| {
            let (g, _, _) =
                build_selection_for_test(&store, &crs, &display, 1, false, Some(sel)).unwrap();
            let s: usize = g
                .chunks
                .iter()
                .flat_map(|c| c.lines.iter())
                .map(|l| l.segments.len())
                .sum();
            let f: usize = g.chunks.iter().map(|c| c.fill_indices.len()).sum();
            (s, f)
        };
        let base = resolve_style(
            &store,
            &crate::data::layer::StyleBy {
                column: col.clone(),
                ramp: crate::data::layer::Ramp::Viridis,
                mode: crate::data::layer::StyleMode::Categorical {
                    values: vec!["x".to_string()],
                    colors: None,
                    labels: None,
                },
                hidden_bins: 0,
                per_area: false,
                classified_rows: None,
                width_px: None,
            },
        )
        .expect("styleable");
        assert!(base.outlines, "a generic palette keeps its outlines");
        let mut palette = base.clone();
        palette.outlines = false;

        let (with_segs, with_fills) = segs(&base);
        let (without_segs, without_fills) = segs(&palette);
        assert!(with_segs > 0, "the ordinary path outlines its polygons");
        assert_eq!(without_segs, 0, "a fill palette builds no segments");
        assert_eq!(with_fills, without_fills, "same fills either way");
    }

    /// A partly refined group keeps complete coverage: real rows where it
    /// was refined, boxes everywhere else in the same group.
    ///
    /// Without the gap fill, consolidating drops the boxes of any group
    /// that has been refined once, and the map shows geometry inside the
    /// old refine rect and a row-group-shaped hole around it.
    #[test]
    fn a_partly_refined_group_still_covers_its_gaps() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let starts = store.rg_starts();
        let group_rows = (starts[1] - starts[0]) as u32;
        // Group 0 refined for one viewport: the first tenth of its rows.
        let refined = vec![(0u32, group_rows / 10)];
        let gaps = complement_ranges(&refined, group_rows);

        let (geom, rows, _bad, _boxes, _resolved) = build_geometry(
            &store,
            &crs,
            &display,
            None,
            vec![
                GroupSel::Ranges(0, refined.clone()),
                GroupSel::BoxRanges { group: 0, ranges: gaps },
            ],
            None,
            None,
        )
        .unwrap();
        assert_eq!(rows as u64, group_rows as u64, "every row of the group is on the map");
        // The refined tenth brought outlines; the rest is boxes, which
        // carry none, so both representations are present at once.
        let segs: usize = geom
            .chunks
            .iter()
            .flat_map(|c| c.lines.iter())
            .map(|l| l.segments.len())
            .sum();
        assert!(segs > 0, "the refined rows keep their real outlines");
        let tris: usize = geom.chunks.iter().map(|c| c.fill_indices.len() / 3).sum();
        let gap_rows = group_rows as usize - refined[0].1 as usize;
        assert!(
            tris >= 2 * gap_rows,
            "at least two triangles per boxed row: {tris} for {gap_rows} gap rows"
        );
    }

    /// A consolidating rebuild mixes box groups and refined groups in one
    /// build, and each row appears exactly once.
    ///
    /// This is what stops boxes showing under the real geometry that
    /// replaced them: an append only adds sections, so the layer has to be
    /// rebuilt from the per-group states once a group leaves box display.
    #[test]
    fn a_mixed_rebuild_draws_each_row_once() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let starts = store.rg_starts();
        let expected = starts[2] - starts[0];
        // Group 0 refined to real rows, group 1 still on boxes: exactly
        // the state a layer is in mid-zoom.
        let sel = vec![
            GroupSel::All(0),
            GroupSel::Boxes { group: 1, rect: None },
        ];
        let (geom, rows, _bad, _boxes, resolved) =
            build_geometry(&store, &crs, &display, None, sel, None, None).unwrap();
        assert_eq!(rows as u64, expected, "each row built once, not twice");
        let mut states: Vec<_> = resolved.iter().map(|(g, st)| (*g, st)).collect();
        states.sort_by_key(|(g, _)| *g);
        assert!(matches!(states[0].1, GroupLoad::Full));
        assert!(matches!(states[1].1, GroupLoad::Boxes { .. }));
        // The refined group brought outlines, the box group did not, so
        // the mesh carries both kinds and neither is duplicated.
        let segs: usize = geom
            .chunks
            .iter()
            .flat_map(|c| c.lines.iter())
            .map(|l| l.segments.len())
            .sum();
        assert!(segs > 0, "the refined group keeps its real outlines");
    }

    /// The fallback a store can offer when a selection is over budget.
    #[test]
    fn boxes_need_a_covering_column_and_polygons() {
        use super::{over_budget_plan, OverBudget};
        assert_eq!(over_budget_plan(true, true), OverBudget::Boxes);
        // No covering column: nothing to draw but the geometry.
        assert_eq!(over_budget_plan(false, true), OverBudget::Stride);
        // Lines and points: a bounding box is not the feature. A road's
        // box is a rectangle the road does not follow.
        assert_eq!(over_budget_plan(true, false), OverBudget::Stride);
        assert_eq!(over_budget_plan(false, false), OverBudget::Stride);
    }

    /// Manual: what refinement decides for a street-level viewport.
    /// `GEOPQ_FILE=... GEOPQ_KM=8 cargo test --release refine_at_a_small_viewport
    /// -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a local file"]
    fn refine_at_a_small_viewport() {
        let Ok(file) = std::env::var("GEOPQ_FILE") else {
            eprintln!("set GEOPQ_FILE");
            return;
        };
        let km: f64 = std::env::var("GEOPQ_KM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8.0);
        let (store, _crs, _info, rg) =
            open_store(&Source::Local(std::path::PathBuf::from(&file))).unwrap();
        let (_, boxes) = rg.expect("row-group boxes");
        // A viewport of `km` across, centred on a populated row group.
        let b = boxes[boxes.len() / 2];
        let (cx, cy) = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5);
        let half = km * 1000.0 / 2.0;
        let rect = [cx - half, cy - half, cx + half, cy + half];
        let groups = intersecting_rgs(&boxes, rect);
        eprintln!("{km} km viewport touches {} of {} row groups", groups.len(), boxes.len());

        // What the app now asks for: this viewport's rows, per group.
        let jobs: Vec<GroupSel> = groups.iter().map(|&g| GroupSel::Rect(g, rect)).collect();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        match prepare_refinement_jobs(&store, jobs, MAX_BUILD_ROWS, MAX_BUILD_GEOM_BYTES, &cancel).unwrap() {
            RefinePlan::Ready(sel) => {
                let rows: u64 = sel
                    .iter()
                    .map(|s| match s {
                        GroupSel::ResolvedRect { ranges, .. } => {
                            ranges.iter().map(|&(a, b)| (b - a) as u64).sum()
                        }
                        _ => 0,
                    })
                    .sum();
                eprintln!("READY: {} selections, {rows} rows", sel.len());
            }
            RefinePlan::Deferred { rows, geom_bytes } => {
                panic!("DEFERRED at {km} km: {rows} rows, {geom_bytes:?} bytes");
            }
        }
    }

    /// Read batches are bounded by bytes, whatever a row weighs: one
    /// batch per worker is the decode footprint, and a fixed row count
    /// makes it a property of the dataset rather than of the machine.
    #[test]
    fn read_batches_are_sized_in_bytes() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, _crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        let n_rg = store.rg_starts().len() - 1;
        for g in 0..n_rg {
            let rows = store.rg_starts()[g + 1] - store.rg_starts()[g];
            let per_row = (store.rg_geom_bytes(g as u32) / rows).max(1);
            let n = store.batch_rows(g, BATCH_TARGET_BYTES, BATCH_SIZE) as u64;
            assert!(n >= 1, "group {g} would read nothing");
            // Within one row of the budget, or capped by the row ceiling.
            assert!(
                n == BATCH_SIZE as u64 || (n - 1) * per_row <= BATCH_TARGET_BYTES,
                "group {g}: {n} rows × {per_row} B over budget"
            );
            // A budget smaller than one row still reads that row: a
            // single 200k-vertex polygon must not stall the decode.
            assert_eq!(store.batch_rows(g, 1, BATCH_SIZE), 1);
        }
    }

    /// End-to-end pruning against the Hilbert-sorted covering fixture:
    /// covering stats must be detected, and a small-viewport load must
    /// decode a small fraction of the row groups and rows.
    /// The map must be framed from metadata before any geometry is read.
    ///
    /// Until it is, the camera sits on the whole world, where the basemap
    /// has nothing to fetch — so the tiles could only start once the build
    /// finished. `Framed` has to reach the app strictly before `Loaded`,
    /// and has to point at the same place the layer eventually lands.
    /// Appending fragments must not disturb a single existing index.
    ///
    /// The whole lazy-part scheme rests on this: a layer keeps the geometry
    /// and per-row-group decode state it already built, and the new parts
    /// simply extend the global row-group space. If offsets shifted, every
    /// `GroupLoad` on the layer would silently point at the wrong group.
    fn part(url: &str, bbox: Option<[f64; 4]>) -> crate::data::repo::StacPart {
        crate::data::repo::StacPart {
            url: url.into(),
            bbox,
            rows: 1,
            rel: None,
        }
    }

    /// Panning must find the parts it is moving into, skip the ones it
    /// already has, and take the most useful ones first.
    /// End to end against a real collection: a view that wants more parts
    /// than the cap opens anyway, and panning elsewhere finds parts it does
    /// not have. Overture's buildings collection is 512 items, which is the
    /// case that used to refuse to open at all.
    ///
    ///   cargo test --bin geopq-workbench overture_collection -- --ignored --nocapture
    #[test]
    #[ignore = "hits the network: Overture's STAC catalog"]
    fn overture_collection_opens_capped_and_grows() {
        let url = "https://stac.overturemaps.org/2026-07-22.0/buildings/building/collection.json";
        let src = Source::Stac {
            url: url.into(),
            name: "building".into(),
        };
        // Western Europe: far more than the cap intersects.
        let europe = [-5.0, 42.0, 15.0, 55.0];
        let (store, _crs, info, _rg) = match open_stac_store(&src, url, Some(europe)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        eprintln!(
            "opened {} parts, {} rows",
            store.fragments.len(),
            store.total_rows()
        );
        assert!(
            store.fragments.len() <= STAC_PART_CAP,
            "opened {} parts, cap is {STAC_PART_CAP}",
            store.fragments.len()
        );
        assert!(store.fragments.len() > 1, "expected a multi-part store");
        assert_eq!(store.stac_collection(), Some(url));
        assert_eq!(store.part_urls().len(), store.fragments.len());
        assert!(info.files >= store.fragments.len());
        // The parts are bare object-store URLs with no credit of their
        // own; the collection is what says who owns this and under what
        // terms, and ODbL makes the credit a licence condition.
        let a = info.attribution.expect("credit from the STAC collection");
        eprintln!("credit: {}", a.credit);
        assert!(a.credit.contains("Overture Maps Foundation"), "{}", a.credit);
        assert!(a.credit.contains("ODbL-1.0"), "{}", a.credit);

        // Pan to California: parts the store does not hold.
        let parts = crate::data::repo::fetch_stac_parts(url).expect("part list");
        let total = parts.len();
        let california = [-122.0, 36.0, -118.0, 39.0];
        let add = parts_to_add(parts, &store.part_urls(), california, PART_APPEND_PER_PASS);
        eprintln!("{total} parts total; panning west finds {} to add", add.len());
        assert!(!add.is_empty(), "panning off the opened parts must find more");
        assert!(add.len() <= PART_APPEND_PER_PASS);
        for p in &add {
            assert!(!store.part_urls().contains(&p.url), "offered an open part");
        }
    }

    #[test]
    fn part_selection_prefers_the_biggest_overlap() {
        let have: std::collections::HashSet<String> =
            ["https://x/have.parquet".to_string()].into_iter().collect();
        let parts = vec![
            // Already open: never offered again, however good the overlap.
            part("https://x/have.parquet", Some([0.0, 0.0, 10.0, 10.0])),
            // Clips one corner.
            part("https://x/corner.parquet", Some([9.0, 9.0, 20.0, 20.0])),
            // Covers the viewport.
            part("https://x/centre.parquet", Some([0.0, 0.0, 10.0, 10.0])),
            // Misses entirely.
            part("https://x/far.parquet", Some([50.0, 50.0, 60.0, 60.0])),
            // No bbox: unusable for a viewport decision.
            part("https://x/nobbox.parquet", None),
        ];
        let got = parts_to_add(parts, &have, [0.0, 0.0, 10.0, 10.0], 8);
        let urls: Vec<&str> = got.iter().map(|p| p.url.as_str()).collect();
        assert_eq!(urls, vec!["https://x/centre.parquet", "https://x/corner.parquet"]);
    }

    /// One pan opens a bounded number of parts, keeping the best.
    #[test]
    fn part_selection_respects_the_per_pass_room() {
        let parts: Vec<_> = (0..20)
            .map(|i| {
                let w = 1.0 + i as f64;
                part(&format!("https://x/{i}.parquet"), Some([0.0, 0.0, w, w]))
            })
            .collect();
        let got = parts_to_add(parts, &Default::default(), [0.0, 0.0, 100.0, 100.0], 3);
        assert_eq!(got.len(), 3);
        // Largest overlap first: the last three indices, descending.
        let urls: Vec<&str> = got.iter().map(|p| p.url.as_str()).collect();
        assert_eq!(
            urls,
            vec!["https://x/19.parquet", "https://x/18.parquet", "https://x/17.parquet"]
        );
    }

    /// A viewport with nothing new in it must produce no work at all,
    /// which is what stops a stationary camera re-probing every frame.
    #[test]
    fn part_selection_returns_nothing_when_all_are_open() {
        let parts = vec![
            part("https://x/a.parquet", Some([0.0, 0.0, 10.0, 10.0])),
            part("https://x/b.parquet", Some([0.0, 0.0, 10.0, 10.0])),
        ];
        let have: std::collections::HashSet<String> = parts
            .iter()
            .map(|p| p.url.clone())
            .collect();
        assert!(parts_to_add(parts, &have, [0.0, 0.0, 10.0, 10.0], 8).is_empty());
    }

    /// Touching bboxes are not overlapping ones: a part that only shares
    /// an edge with the viewport contributes nothing to it.
    #[test]
    fn part_selection_ignores_edge_contact() {
        let parts = vec![part("https://x/edge.parquet", Some([10.0, 0.0, 20.0, 10.0]))];
        assert!(parts_to_add(parts, &Default::default(), [0.0, 0.0, 10.0, 10.0], 8).is_empty());
    }

    #[test]
    fn appending_parts_leaves_existing_indices_alone() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let source = Source::Local(path.clone());
        let (base, _crs, _info, _rg) = open_store(&source).unwrap();
        let n_rg = base.rg_starts().len() - 1;
        let base_rows = base.total_rows();
        let before: Vec<u64> = base.rg_starts().to_vec();
        let row_10 = base.fetch_row(10).expect("row 10 of the base store");

        // The same file again, as a second fragment.
        let f = open_file(&source).unwrap();
        let grown = base.with_fragments_appended(vec![(
            crate::data::store::Fragment {
                source: source.clone(),
                meta: f.meta.clone(),
                part_values: Vec::new(),
                rg_offset: 0,
                row_offset: 0,
            },
            f.rg_rows.clone(),
        )]);

        assert_eq!(grown.rg_starts().len() - 1, n_rg * 2);
        assert_eq!(grown.total_rows(), base_rows * 2);
        // Every pre-existing boundary is byte-for-byte what it was.
        assert_eq!(&grown.rg_starts()[..=n_rg], &before[..]);
        let new_frag = grown.frag_of_group(n_rg);
        assert_eq!(new_frag.rg_offset, n_rg);
        assert_eq!(new_frag.row_offset, base_rows);
        assert_eq!(grown.frag_of_group(0).rg_offset, 0);

        // Arithmetic is not enough: the reads have to land in the right
        // file. Row 10 must still be row 10, and its twin in the appended
        // fragment must read identically.
        let same = grown.fetch_row(10).expect("row 10 of the grown store");
        assert_eq!(format!("{same:?}"), format!("{row_10:?}"));
        let twin = grown
            .fetch_row(base_rows as u32 + 10)
            .expect("the appended fragment's row 10");
        assert_eq!(format!("{twin:?}"), format!("{row_10:?}"));
    }

    /// A store that did not come from a collection has nothing to append,
    /// and one that did knows which parts it already holds.
    #[test]
    fn only_stac_stores_offer_parts() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, _, _, _) = open_store(&Source::Local(path)).unwrap();
        assert_eq!(store.stac_collection(), None);
        // A local fragment has no URL, so it can never look "already open".
        assert!(store.part_urls().is_empty());
    }

    #[test]
    fn framing_arrives_before_the_geometry() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let source = Source::Local(path);
        let (store, crs, info, rg_meta) = open_store(&source).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = LoaderHandle {
            tx,
            egui_ctx: eframe::egui::Context::default(),
        };
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        build_opened(
            &handle,
            1,
            1,
            OpenedStore {
                store: Arc::new(store),
                crs,
                info,
                rg_meta,
            },
            display,
            eframe::egui::Color32::RED,
            [0.0, 0.0, 1.0, 1.0],
            1024.0,
            true,
            &Arc::new(std::sync::atomic::AtomicBool::new(false)),
            None,
            false,
            Instant::now(),
        );
        drop(handle);

        let mut framed: Option<[f64; 4]> = None;
        let mut framed_display: Option<DisplayCrs> = None;
        let mut layer_bounds: Option<[f64; 4]> = None;
        for msg in rx {
            match msg {
                LoadMsg::Framed { display, world, .. } => {
                    assert!(layer_bounds.is_none(), "Framed arrived after Loaded");
                    framed_display = display;
                    framed = Some(world);
                }
                LoadMsg::Loaded { layer, .. } => layer_bounds = Some(layer.bounds_world()),
                _ => {}
            }
        }
        let framed = framed.expect("a file with covering stats must frame early");
        let actual = layer_bounds.expect("the layer still loads");
        assert!(
            framed_display.is_some(),
            "Massachusetts parcels adopt their own projected CRS"
        );
        // Same place: the metadata extent should contain the built one, up
        // to a small slack for the box-corner sampling.
        let slack = (actual[2] - actual[0]).max(actual[3] - actual[1]) * 0.02;
        assert!(
            framed[0] <= actual[0] + slack
                && framed[1] <= actual[1] + slack
                && framed[2] >= actual[2] - slack
                && framed[3] >= actual[3] - slack,
            "framed {framed:?} does not cover built {actual:?}"
        );
        // And it must be a real framing, not the whole world.
        assert!(framed[2] - framed[0] < 0.05, "framed span too wide: {framed:?}");
    }

    #[test]
    fn covering_stats_prune_row_groups() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _info, rg_meta) = open_store(&Source::Local(path)).unwrap();
        assert_eq!(crs.epsg, Some(26986));
        let (source, boxes) = rg_meta.expect("covering stats detected");
        assert!(source.contains("covering"), "{source}");
        let n_rg = store.rg_starts().len() - 1;
        assert_eq!(boxes.len(), n_rg);

        // Hilbert order => clearly better than all-overlapping (which would
        // be ~n_rg - 1). Coarse 131k-row groups still overlap somewhat.
        let overlap = bbox_overlap_metric(&boxes);
        assert!(
            overlap < (n_rg - 1) as f64 * 0.6,
            "hilbert file should be clustered: ×{overlap:.1} of {n_rg}"
        );

        // A ~10 km viewport around Boston (EPSG:26986 meters).
        let rect = [230_000.0, 895_000.0, 240_000.0, 905_000.0];
        let sel = intersecting_rgs(&boxes, rect);
        assert!(
            !sel.is_empty() && sel.len() <= n_rg / 2,
            "expected strong pruning: {} of {n_rg}",
            sel.len()
        );

        // Pruned load decodes only those groups' rows.
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let jobs: Vec<GroupSel> = sel.iter().map(|&g| GroupSel::All(g)).collect();
        let (geometry, rows, _bad, _rg, _resolved) =
            build_geometry(&store, &crs, &display, None, jobs, None, None).unwrap();
        let expected: u64 = sel
            .iter()
            .map(|&g| store.rg_starts()[g as usize + 1] - store.rg_starts()[g as usize])
            .sum();
        assert_eq!(rows as u64, expected);
        assert!(rows < 1_886_414 / 2);
        assert!(!geometry.chunks.is_empty());

        // Global row indices are preserved: pick items reference rows
        // belonging to the selected groups only.
        let starts = store.rg_starts();
        for item in geometry.rtree.iter().take(5000) {
            let row = item.feature.index as u64;
            let g = match starts.binary_search(&row) {
                Ok(i) => i,
                Err(i) => i - 1,
            } as u32;
            assert!(sel.contains(&g), "row {row} outside selected groups");
        }

        // --- per-feature covering selection on the same viewport ---
        let cov = store.covering.as_ref().expect("covering column resolved");
        assert_eq!(cov.children, ["xmin", "ymin", "xmax", "ymax"].map(String::from));
        let jobs: Vec<GroupSel> = sel.iter().map(|&g| GroupSel::Rect(g, rect)).collect();
        let (geometry_f, rows_f, _bad, _rg, resolved) =
            build_geometry(&store, &crs, &display, None, jobs, None, None).unwrap();
        // Dense-urban viewport: still expect a substantial decode cut vs
        // the 4 whole groups (independently verified via DuckDB: 163,151
        // features intersect this rect).
        assert!(
            rows_f > 0 && (rows_f as u64) < expected / 2,
            "feature selection should cut decode below whole groups: {rows_f} vs {expected}"
        );
        assert!(!geometry_f.chunks.is_empty());
        // Resolved states are viewport-filtered rows for the same rect.
        for (g, st) in &resolved {
            assert!(sel.contains(g));
            match st {
                GroupLoad::Rows { rect: r, .. } => {
                    // Empty ranges are valid: the group's coarse bbox
                    // intersects the viewport but no feature does.
                    assert_eq!(*r, rect);
                }
                GroupLoad::Full => {} // a group fully inside the viewport
                GroupLoad::None => panic!("group {g} unresolved"),
                GroupLoad::Preview { .. } => panic!("group {g} unexpectedly previewed"),
                GroupLoad::Boxes { .. } => panic!("group {g} unexpectedly boxed"),
            }
        }
        // Every decoded feature actually intersects the viewport rect:
        // check via the geometry fetched from the store.
        let mut picked: Vec<u32> = geometry_f
            .rtree
            .iter()
            .map(|i| i.feature.index)
            .take(2000)
            .collect();
        picked.sort_unstable();
        picked.dedup();
        let geoms = store.fetch_geoms(&picked).unwrap();
        for (row, geom) in geoms {
            use geo::BoundingRect;
            let geom = geom.expect("non-null");
            let b = geom.bounding_rect().unwrap();
            assert!(
                b.min().x <= rect[2]
                    && b.max().x >= rect[0]
                    && b.min().y <= rect[3]
                    && b.max().y >= rect[1],
                "row {row} outside viewport rect"
            );
        }

        // Complement ranges rebuild the full group without overlap.
        let (g0, st0) = resolved
            .iter()
            .find(|(_, st)| matches!(st, GroupLoad::Rows { .. }))
            .expect("at least one partially selected group");
        let GroupLoad::Rows { ranges, .. } = st0 else { unreachable!() };
        let n = (store.rg_starts()[*g0 as usize + 1] - store.rg_starts()[*g0 as usize]) as u32;
        let comp = complement_ranges(ranges, n);
        let covered: u32 = ranges.iter().chain(&comp).map(|(s, e)| e - s).sum();
        assert_eq!(covered, n, "ranges + complement must tile the group");
        let (_gc, rows_c, _bad, _rg, resolved_c) = build_geometry(
            &store,
            &crs,
            &display,
            None,
            vec![GroupSel::Ranges(*g0, comp.clone())],
            None,
            None,
        )
        .unwrap();
        let selected: u32 = ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(rows_c as u32, n - selected);
        assert!(matches!(resolved_c[0].1, GroupLoad::Full));
    }

    /// Remote loading over HTTP range requests must produce exactly the
    /// same result as the local path, while downloading only a fraction of
    /// the file (footer + covering column + selected geometry rows).
    #[test]
    fn remote_range_requests_match_local() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let file_len = std::fs::metadata(&path).unwrap().len();
        let server = crate::data::source::testserver::spawn(path.clone());
        let remote = Source::remote(&server.url).unwrap();
        assert_eq!(remote.size(), file_len);
        let local = Source::Local(path);

        // Identical metadata read.
        let (store_r, crs_r, info_r, rg_meta_r) = open_store(&remote).unwrap();
        let (store_l, crs_l, _info, rg_meta_l) = open_store(&local).unwrap();
        assert_eq!(crs_r.epsg, crs_l.epsg);
        assert_eq!(info_r.rows, 1_886_414);
        let (_, boxes_r) = rg_meta_r.expect("covering stats over http");
        let (_, boxes_l) = rg_meta_l.unwrap();
        assert_eq!(boxes_r, boxes_l);
        assert!(store_r.covering.is_some());

        // Same viewport-selected build, local vs remote.
        let rect = [230_000.0, 895_000.0, 240_000.0, 905_000.0];
        let sel = intersecting_rgs(&boxes_r, rect);
        assert!(!sel.is_empty());
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let jobs = |_: ()| -> Vec<GroupSel> {
            sel.iter().map(|&g| GroupSel::Rect(g, rect)).collect()
        };
        let (_g, rows_r, bad_r, _rg, _res) =
            build_geometry(&store_r, &crs_r, &display, None, jobs(()), None, None).unwrap();
        let (_g, rows_l, bad_l, _rg, _res) =
            build_geometry(&store_l, &crs_l, &display, None, jobs(()), None, None).unwrap();
        assert_eq!((rows_r, bad_r), (rows_l, bad_l));
        assert!(rows_r > 0);

        // Lazy attribute fetch over http.
        let row = store_r.rg_starts()[sel[0] as usize] as u32;
        let batch = store_r.fetch_row(row).unwrap();
        assert_eq!(batch.num_rows(), 1);

        // The point of range requests: a small fraction of the file moved.
        let served = server.bytes_served.load(std::sync::atomic::Ordering::SeqCst);
        let requests = server.requests.load(std::sync::atomic::Ordering::SeqCst);
        // The network readout counts what the server actually sent. The
        // two sides are measured independently — the server tallies what
        // it wrote, the app tallies what each request returned — so
        // agreement means the status bar is reporting real traffic and
        // not an estimate.
        {
            use crate::data::net::{self, Channel};
            // Per source, not per process: the counters are global and
            // other tests fetch over http at the same time, but a URL
            // belongs to exactly one of them.
            let (by_src, reqs_src) =
                net::for_source(&server.url).expect("attributed to its file");
            assert_eq!(
                by_src, served,
                "the readout must account for every byte the server sent"
            );
            // The server also sees a HEAD (no body) and a 404 sidecar
            // probe, neither of which moves data.
            assert!(
                reqs_src <= requests,
                "{reqs_src} counted against {requests} served"
            );
            assert!(net::rate(Channel::Data) > 0.0, "a live transfer reads a rate");
        }
        eprintln!(
            "remote load: {} of {} bytes ({:.1}%), {} requests, {} rows",
            served,
            file_len,
            served as f64 / file_len as f64 * 100.0,
            requests,
            rows_r
        );
        assert!(
            served < file_len / 2,
            "expected partial download: {served} of {file_len}"
        );
    }

    /// Remote pick-latency probe, opt-in: what a map click costs on a
    /// remote layer (candidate geometry fetch + full-row attribute fetch).
    /// GEOPQ_PROBE_URI=s3://bucket/key [GEOPQ_PROBE_PROFILE=name] \
    ///   cargo test --release remote_pick_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn remote_pick_probe() {
        let Ok(uri) = std::env::var("GEOPQ_PROBE_URI") else {
            eprintln!("set GEOPQ_PROBE_URI");
            return;
        };
        let src = if uri.starts_with("s3://") {
            Source::S3 {
                uri,
                profile: std::env::var("GEOPQ_PROBE_PROFILE").ok(),
                endpoint: None,
                url: String::new(),
                len: 0,
            }
        } else {
            Source::Remote { url: uri, len: 0 }
        };
        let t0 = std::time::Instant::now();
        let src = src.resolve().unwrap();
        eprintln!(
            "resolve: {} ms, {} MB",
            t0.elapsed().as_millis(),
            src.size() >> 20
        );
        let t0 = std::time::Instant::now();
        let (store, _crs, info, _rg) = open_store(&src).unwrap();
        let meta = store.fragments[0].meta.metadata();
        eprintln!(
            "open: {} ms; {} rows, {} groups ({}-{} rows), {} cols, page index: column={} offset={}",
            t0.elapsed().as_millis(),
            store.total_rows(),
            info.row_groups,
            info.rg_rows_min,
            info.rg_rows_max,
            info.columns.len(),
            meta.column_index().is_some(),
            meta.offset_index().is_some(),
        );
        let mid = (store.total_rows() / 2) as u32;
        // Polygon pick path: one batched geometry read for candidates.
        for n in [1u32, 50, 512] {
            let rows: Vec<u32> = (mid..mid + n).collect();
            let t0 = std::time::Instant::now();
            let geoms = store.fetch_geoms(&rows).unwrap();
            eprintln!(
                "fetch_geoms({n} rows): {} ms ({} geoms)",
                t0.elapsed().as_millis(),
                geoms.len()
            );
        }
        // Info panel: capped column fetch of one row (what a click costs).
        let total = store.schema.fields().len();
        let cap = 256.min(total);
        let mut cols: Vec<usize> = (0..cap).collect();
        if store.geom_col >= cap {
            cols.push(store.geom_col);
        }
        let t0 = std::time::Instant::now();
        let row = store.fetch(&[mid], Some(&cols)).unwrap();
        eprintln!(
            "fetch row, capped ({} of {total} cols): {} ms",
            row[0].num_columns(),
            t0.elapsed().as_millis()
        );
    }

    /// Local full-load benchmark, opt-in:
    /// cargo test --release local_load_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn local_load_bench() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        for run in 0..3 {
            let t0 = std::time::Instant::now();
            let (store, crs, info, _) = open_store(&Source::Local(path.clone())).unwrap();
            let open_ms = t0.elapsed().as_millis();
            let t1 = std::time::Instant::now();
            let (geometry, rows, bad, _, _) =
                build_geometry(&store, &crs, &display, None, all_groups(&store), None, None).unwrap();
            eprintln!(
                "run {run}: open {open_ms} ms, build {} ms, {rows} rows ({bad} bad), {} chunks, {} rgs",
                t1.elapsed().as_millis(),
                geometry.chunks.len(),
                info.row_groups,
            );
        }
    }

    /// Live remote benchmark, opt-in:
    /// GEOPQ_REMOTE_URL=https://... cargo test --release remote_live -- --ignored --nocapture
    #[test]
    #[ignore]
    fn remote_live() {
        let Ok(url) = std::env::var("GEOPQ_REMOTE_URL") else {
            return;
        };
        let _ = env_logger::try_init();
        let t0 = std::time::Instant::now();
        let source = if url.starts_with("s3://") {
            Source::S3 {
                uri: url,
                profile: std::env::var("GEOPQ_AWS_PROFILE").ok(),
                endpoint: std::env::var("GEOPQ_S3_ENDPOINT").ok(),
                url: String::new(),
                len: 0,
            }
            .resolve()
            .unwrap()
        } else {
            Source::remote(&url).unwrap()
        };
        eprintln!("size: {} bytes", source.size());
        let (store, crs, info, rg_meta) = open_store(&source).unwrap();
        eprintln!(
            "opened in {} ms: {} rows, {} row groups, {} — covering: {}",
            t0.elapsed().as_millis(),
            info.rows,
            info.row_groups,
            info.geo.version_label,
            store.covering.is_some(),
        );
        let Some((meta_src, boxes)) = rg_meta else {
            eprintln!("no metadata bboxes — no pruning possible");
            return;
        };
        eprintln!("rg bboxes: {} ({meta_src})", boxes.len());

        // Viewport: GEOPQ_VIEWPORT="xmin,ymin,xmax,ymax" (data CRS), or a
        // small default around the first row group's center.
        let rect: [f64; 4] = match std::env::var("GEOPQ_VIEWPORT") {
            Ok(v) => {
                let p: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                assert_eq!(p.len(), 4, "GEOPQ_VIEWPORT needs 4 numbers");
                [p[0], p[1], p[2], p[3]]
            }
            Err(_) => {
                let b = boxes[0];
                let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
                let u = boxes.iter().fold(b, |a, x| {
                    [a[0].min(x[0]), a[1].min(x[1]), a[2].max(x[2]), a[3].max(x[3])]
                });
                let (dx, dy) = ((u[2] - u[0]) * 0.01, (u[3] - u[1]) * 0.01);
                [cx - dx, cy - dy, cx + dx, cy + dy]
            }
        };
        let sel = intersecting_rgs(&boxes, rect);
        eprintln!("viewport {rect:?}: {} of {} row groups", sel.len(), boxes.len());

        let t1 = std::time::Instant::now();
        let jobs: Vec<GroupSel> = sel.iter().map(|&g| GroupSel::Rect(g, rect)).collect();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (geometry, rows, bad, _rg, _res) =
            build_geometry(&store, &crs, &display, None, jobs, None, None).unwrap();
        eprintln!(
            "loaded {rows} features ({bad} bad) in {} ms, {} chunks",
            t1.elapsed().as_millis(),
            geometry.chunks.len()
        );
        assert!(!geometry.chunks.is_empty() || rows == 0);
    }

    #[test]
    fn range_helpers() {
        assert_eq!(complement_ranges(&[], 10), vec![(0, 10)]);
        assert_eq!(complement_ranges(&[(0, 10)], 10), vec![]);
        assert_eq!(
            complement_ranges(&[(2, 4), (7, 8)], 10),
            vec![(0, 2), (4, 7), (8, 10)]
        );
        assert_eq!(
            rows_to_ranges([1u32, 2, 3, 7, 9, 10].into_iter()),
            vec![(1, 4), (7, 8), (9, 11)]
        );
    }
}

#[cfg(test)]
mod hive_tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// One hive fragment: `n` points clustered at (cx, cy), ids from `id0`,
    /// small row groups, file-level geo bbox (no covering column).
    fn write_part(path: &std::path::Path, n: usize, id0: i64, cx: f64, cy: f64) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "bbox": [cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0],
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let wkbs: Vec<Vec<u8>> = (0..n)
            .map(|i| wkb_point(cx + (i % 10) as f64 * 0.01, cy + (i / 10) as f64 * 0.01))
            .collect();
        let ids: Vec<i64> = (0..n as i64).map(|i| id0 + i).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(128)).build();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// Streamed appends: small first batch, later ones at the full
    /// target, order preserved, every job delivered exactly once.
    #[test]
    fn append_batches_small_first_then_full() {
        let path = std::env::temp_dir().join("geopq_append_batches.parquet");
        // 600 rows in 128-row groups -> groups of 128,128,128,128,88.
        write_part(&path, 600, 0, 2.0, 48.0);
        let (store, _crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        let jobs: Vec<GroupSel> = (0..5).map(GroupSel::All).collect();
        let batches = super::append_batches_with(&store, jobs, 100, 256);
        let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
        // First batch closes at >=100 rows (one group), later at >=256
        // (two groups each), remainder flushed.
        assert_eq!(sizes, vec![1, 2, 2]);
        let order: Vec<u32> = batches.iter().flatten().map(|j| j.group()).collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    /// The byte-scan envelope must agree with the full geo-types decode
    /// for every geometry family the writer produces.
    #[test]
    fn wkb_envelope_matches_decoded_bounds() {
        use geo::BoundingRect;
        let geoms: Vec<geo_types::Geometry<f64>> = vec![
            geo_types::Point::new(2.5, 41.0).into(),
            geo_types::LineString::from(vec![(0.0, 0.0), (3.0, -1.0), (2.0, 5.0)]).into(),
            geo_types::Polygon::new(
                geo_types::LineString::from(vec![
                    (0.0, 0.0),
                    (10.0, 0.0),
                    (10.0, 8.0),
                    (0.0, 8.0),
                    (0.0, 0.0),
                ]),
                vec![geo_types::LineString::from(vec![
                    (2.0, 2.0),
                    (4.0, 2.0),
                    (4.0, 4.0),
                    (2.0, 2.0),
                ])],
            )
            .into(),
            geo_types::MultiPolygon(vec![
                geo_types::Polygon::new(
                    geo_types::LineString::from(vec![
                        (-70.0, 45.0),
                        (-69.0, 45.0),
                        (-69.0, 46.0),
                        (-70.0, 45.0),
                    ]),
                    vec![],
                ),
                geo_types::Polygon::new(
                    geo_types::LineString::from(vec![
                        (-60.0, 50.0),
                        (-59.0, 50.0),
                        (-59.0, 51.0),
                        (-60.0, 50.0),
                    ]),
                    vec![],
                ),
            ])
            .into(),
            geo_types::Geometry::GeometryCollection(geo_types::GeometryCollection(vec![
                geo_types::Point::new(-1.0, -2.0).into(),
                geo_types::LineString::from(vec![(5.0, 5.0), (6.0, 7.0)]).into(),
            ])),
        ];
        for g in geoms {
            let mut buf = Vec::new();
            wkb::writer::write_geometry(&mut buf, &g, &wkb::writer::WriteOptions::default())
                .unwrap();
            let mut env = EMPTY_BBOX;
            grow_wkb_envelope(&buf, &mut env).expect("scan succeeds");
            let r = g.bounding_rect().unwrap();
            let want = [r.min().x, r.min().y, r.max().x, r.max().y];
            assert_eq!(env, want, "envelope mismatch for {g:?}");
        }
        // Malformed input must not panic or grow the box.
        let mut env = EMPTY_BBOX;
        assert!(grow_wkb_envelope(&[1u8, 2, 0], &mut env).is_none());
        assert_eq!(env, EMPTY_BBOX);
    }

    /// A DuckDB-style export (WKB, geo 1.0 metadata, no covering column)
    /// must come out of open_store with a failing quality verdict: C1 has
    /// no bbox source and WKB stats are unusable (docs/OPEN_POLICY.md).
    #[test]
    fn wkb_without_covering_is_not_indexable() {
        use crate::data::quality::Status;
        let path = std::env::temp_dir().join("geopq_quality_wkb.parquet");
        write_part(&path, 500, 0, 2.0, 48.0);
        let (_store, _crs, info, rg_meta) = open_store(&Source::Local(path)).unwrap();
        assert!(rg_meta.is_none(), "WKB without covering has no rg boxes");
        let q = info.quality.expect("quality report attached");
        assert!(!q.indexable);
        let check = |code: &str| q.checks.iter().find(|c| c.code == code).unwrap().status;
        assert_eq!(check("C1"), Status::Fail);
        assert_eq!(check("C2"), Status::Fail);
        assert_eq!(check("C4"), Status::Warn, "WKB encoding is advisory only");
        assert!(q.geom_bytes > 0, "geometry decode-size proxy measured");
    }

    /// A repository "all states" load: fixed remote part set, hive
    /// columns from the URL paths, missing parts (404) dropped.
    #[test]
    fn multi_remote_all_states_load() {
        let root = std::env::temp_dir().join("geopq_multi_remote");
        let _ = std::fs::remove_dir_all(&root);
        write_part(&root.join("country=US/state=east/aeroways.parquet"), 300, 0, 10.0, 45.0);
        write_part(&root.join("country=US/state=west/aeroways.parquet"), 200, 300, -10.0, 45.0);
        let base = crate::data::source::testserver::spawn_dir(root.clone());

        let src = Source::Multi {
            name: "US aeroways (all)".into(),
            urls: vec![
                format!("{base}/country=US/state=east/aeroways.parquet"),
                format!("{base}/country=US/state=west/aeroways.parquet"),
                // A state without the theme: 404 must be skipped, not fatal.
                format!("{base}/country=US/state=mid/aeroways.parquet"),
            ],
        };
        let (store, crs, info, _rg_meta) = open_store(&src).unwrap();
        assert!(crs.is_latlong);
        assert_eq!(store.fragments.len(), 2, "missing part dropped");
        assert_eq!(store.total_rows(), 500);
        assert_eq!(store.part_cols, vec!["country".to_string(), "state".to_string()]);
        assert_eq!(info.files, 2);
    }

    #[test]
    fn glob_matches_segment_wise() {
        assert!(glob_match("d/state=*/roads.parquet", "d/state=MA/roads.parquet"));
        assert!(glob_match("d/*/*/aeroways.parquet", "d/country=CA/state=BC/aeroways.parquet"));
        assert!(glob_match("d/part-*.parquet", "d/part-00001-abc.parquet"));
        // `*` never crosses a `/`.
        assert!(!glob_match("d/*/roads.parquet", "d/a/b/roads.parquet"));
        assert!(!glob_match("d/state=*/roads.parquet", "d/state=MA/rails.parquet"));
        // Multiple stars in one segment backtrack correctly.
        assert!(glob_match("*-x-*.parquet", "part-x-1.parquet"));
        assert!(!glob_match("*-x-*.parquet", "part-y-1.parquet"));
    }

    /// End-to-end S3 prefix open against a fake S3 endpoint: listing XML
    /// (URL-encoded keys, sidecar noise), per-part HEAD probes, ranged
    /// GETs, hive partition columns from the key paths.
    #[test]
    fn s3_prefix_opens_as_hive_dataset() {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;

        let root = std::env::temp_dir().join("geopq_s3_prefix_open");
        let _ = std::fs::remove_dir_all(&root);
        write_part(&root.join("state=east/part-0.parquet"), 300, 0, 10.0, 45.0);
        write_part(&root.join("state=west/part-0.parquet"), 200, 300, -10.0, 45.0);

        let listing = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>bucket</Name><Prefix>data/</Prefix><KeyCount>4</KeyCount><MaxKeys>1000</MaxKeys>
  <IsTruncated>false</IsTruncated>
  <Contents><Key>data/_manifest.json</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>"a"</ETag><Size>10</Size></Contents>
  <Contents><Key>data/_tmp/part-9.parquet</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>"b"</ETag><Size>10</Size></Contents>
  <Contents><Key>data/state%3Deast/part-0.parquet</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>"c"</ETag><Size>1</Size></Contents>
  <Contents><Key>data/state%3Dwest/part-0.parquet</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>"d"</ETag><Size>1</Size></Contents>
</ListBucketResult>"#;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let (root_srv, listing_srv) = (root.clone(), listing.to_string());
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                let (root, listing) = (root_srv.clone(), listing_srv.clone());
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut b = [0u8; 1];
                    while !buf.ends_with(b"\r\n\r\n") && buf.len() < 8192 {
                        match conn.read(&mut b) {
                            Ok(1) => buf.push(b[0]),
                            _ => return,
                        }
                    }
                    let text = String::from_utf8_lossy(&buf);
                    let line1 = text.lines().next().unwrap_or_default().to_string();
                    let target = line1.split_whitespace().nth(1).unwrap_or_default();
                    let (path, query) = target.split_once('?').unwrap_or((target, ""));
                    // Percent-decode the request path.
                    let mut decoded = Vec::new();
                    let pb = path.as_bytes();
                    let mut i = 0;
                    while i < pb.len() {
                        if pb[i] == b'%' && i + 2 < pb.len() {
                            decoded.push(
                                u8::from_str_radix(&path[i + 1..i + 3], 16).unwrap_or(b'%'),
                            );
                            i += 3;
                        } else {
                            decoded.push(pb[i]);
                            i += 1;
                        }
                    }
                    let path = String::from_utf8_lossy(&decoded).into_owned();
                    if query.contains("list-type=2") {
                        let _ = write!(
                            conn,
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{listing}",
                            listing.len()
                        );
                        return;
                    }
                    let Some(rel) = path.strip_prefix("/bucket/data/") else {
                        let _ = write!(conn, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                        return;
                    };
                    let file = root.join(rel);
                    let Ok(data) = std::fs::read(&file) else {
                        let _ = write!(conn, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                        return;
                    };
                    let len = data.len() as u64;
                    if line1.starts_with("HEAD") {
                        let _ = write!(
                            conn,
                            "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
                        );
                        return;
                    }
                    let range = text
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                        .and_then(|l| l.split_once('='))
                        .and_then(|(_, v)| v.trim().split_once('-'))
                        .map(|(a, b)| {
                            (
                                a.parse::<u64>().unwrap_or(0),
                                b.parse::<u64>().unwrap_or(len - 1).min(len - 1),
                            )
                        });
                    match range {
                        Some((s, e)) => {
                            let body = &data[s as usize..=e as usize];
                            let _ = write!(
                                conn,
                                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {s}-{e}/{len}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = conn.write_all(body);
                        }
                        None => {
                            let _ = write!(
                                conn,
                                "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
                            );
                            let _ = conn.write_all(&data);
                        }
                    }
                });
            }
        });

        // A glob pattern selects a single part across partitions.
        let glob_src = Source::S3 {
            uri: "s3://bucket/data/state=*/part-0.parquet".into(),
            profile: None,
            endpoint: Some(endpoint.clone()),
            url: String::new(),
            len: 0,
        };
        assert!(glob_src.is_s3_prefix());
        let (gstore, ..) = open_store(&glob_src).unwrap();
        assert_eq!(gstore.fragments.len(), 2);
        assert_eq!(gstore.total_rows(), 500);

        // A glob matching nothing fails with the pattern in the message.
        let miss = Source::S3 {
            uri: "s3://bucket/data/state=*/nope-*.parquet".into(),
            profile: None,
            endpoint: Some(endpoint.clone()),
            url: String::new(),
            len: 0,
        };
        let err = match open_store(&miss) {
            Err(e) => e,
            Ok(_) => panic!("glob matching nothing must fail"),
        };
        assert!(err.contains("nope-"), "{err}");

        let src = Source::S3 {
            uri: "s3://bucket/data/".into(),
            profile: None,
            endpoint: Some(endpoint),
            url: String::new(),
            len: 0,
        };
        assert!(src.is_s3_prefix());
        let (store, crs, info, _rg_meta) = open_store(&src).unwrap();
        assert!(crs.is_latlong);
        assert_eq!(store.fragments.len(), 2, "sidecar keys filtered out");
        assert_eq!(store.total_rows(), 500);
        assert_eq!(store.part_cols, vec!["state".to_string()]);
        assert_eq!(info.files, 2);

        // Partition values decoded from the URL-encoded listing keys.
        let state_idx = store.schema.index_of("state").unwrap();
        let batches = store.fetch(&[0u32, 499], Some(&[state_idx])).unwrap();
        let states: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let st = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                (0..b.num_rows()).map(|i| st.value(i).to_string()).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(states, vec!["east".to_string(), "west".to_string()]);
    }

    #[test]
    fn hive_dataset_loads_as_single_layer() {
        let root = std::env::temp_dir().join("geopq_hive_open");
        let _ = std::fs::remove_dir_all(&root);
        // Path sort order: __HIVE... < east < west, so global rows follow.
        write_part(
            &root.join("state=__HIVE_DEFAULT_PARTITION__/part-0.parquet"),
            100, 0, 0.0, 0.0,
        );
        write_part(&root.join("state=east/part-0.parquet"), 300, 100, 10.0, 45.0);
        write_part(&root.join("state=west/part-0.parquet"), 200, 400, -10.0, 45.0);
        // Sidecars and hidden files must be ignored.
        std::fs::write(root.join("_SUCCESS"), b"").unwrap();
        std::fs::write(root.join(".hidden.parquet"), b"junk").unwrap();

        let (store, crs, info, rg_meta) = open_store(&Source::Dir(root.clone())).unwrap();
        assert!(crs.is_latlong);
        assert_eq!(store.fragments.len(), 3);
        assert_eq!(store.total_rows(), 600);
        assert_eq!(store.part_cols, vec!["state".to_string()]);
        assert_eq!(info.files, 3);
        let state_idx = store.schema.index_of("state").unwrap();
        assert_eq!(store.schema.field(state_idx).data_type(), &DataType::Utf8);

        // File-level geo bboxes back the per-group pruning boxes.
        let (label, boxes) = rg_meta.expect("bbox fallback from geo metadata");
        assert_eq!(boxes.len(), store.rg_starts().len() - 1);
        assert!(label.contains("bbox"), "{label}");

        // Fetches cross file boundaries; partition values are injected,
        // with NULL for the Hive default partition.
        let id_idx = store.schema.index_of("id").unwrap();
        let rows = [0u32, 99, 100, 399, 400, 599];
        let batches = store.fetch(&rows, Some(&[id_idx, state_idx])).unwrap();
        let (mut ids, mut states): (Vec<i64>, Vec<Option<String>>) = (vec![], vec![]);
        for b in &batches {
            let idc = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let st = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                ids.push(idc.value(i));
                states.push((!st.is_null(i)).then(|| st.value(i).to_string()));
            }
        }
        assert_eq!(ids, vec![0, 99, 100, 399, 400, 599]);
        assert_eq!(
            states,
            vec![
                None,
                None,
                Some("east".into()),
                Some("east".into()),
                Some("west".into()),
                Some("west".into()),
            ]
        );

        // Full-row fetch (attribute panel) exposes the partition value.
        let row = store.fetch_row(150).unwrap();
        let st = row
            .column(row.schema().index_of("state").unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .clone();
        assert_eq!(st.value(0), "east");

        // Geometry decodes across the whole global row space.
        let geoms = store.fetch_geoms(&rows).unwrap();
        assert!(geoms.iter().all(|(_, g)| g.is_some()));

        // SQL: the partition column is queryable, filterable and groupable.
        use crate::sql::engine::{run_query_for_test, SqlLayer};
        let layers = vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: Some(Arc::new(boxes)),
        }];
        let count = |q: &str| -> i64 {
            let out = run_query_for_test(q, &layers).unwrap();
            out.batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap()
                .value(0)
        };
        assert_eq!(count("select count(*) from t"), 600);
        assert_eq!(count("select count(*) from t where state = 'east'"), 300);
        assert_eq!(count("select count(*) from t where state is null"), 100);
        assert_eq!(count("select count(distinct id) from t"), 600);
        // Spatial pushdown through the file-level bbox fallback.
        assert_eq!(
            count(
                "select count(*) from t \
                 where st_intersects(geometry, st_makeenvelope(5, 40, 15, 50))"
            ),
            300
        );
        // Partition-column-only projection (no file columns read at all).
        let out = run_query_for_test("select state from t", &layers).unwrap();
        assert_eq!(out.total_rows, 600);
        let out =
            run_query_for_test("select state, count(*) c from t group by state", &layers).unwrap();
        assert_eq!(out.total_rows, 3);
    }

    /// One part of a partitioned dataset with a covering column: `n`
    /// points at (cx, cy) when `extent` is true, one null-geometry row
    /// with an all-null covering and no `bbox` in `geo` when it is not.
    /// The second shape is what adaptive H3 writes into
    /// `h3=__HIVE_DEFAULT_PARTITION__`, and it is the case under test.
    fn write_covering_part(path: &std::path::Path, n: usize, cx: f64, cy: f64, extent: bool) {
        use arrow::array::{ArrayRef, Float64Array, StructArray};
        use arrow::datatypes::Fields;

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut col = serde_json::json!({
            "encoding": "WKB",
            "geometry_types": ["Point"],
            "covering": {"bbox": {
                "xmin": ["bbox", "xmin"],
                "ymin": ["bbox", "ymin"],
                "xmax": ["bbox", "xmax"],
                "ymax": ["bbox", "ymax"],
            }},
        });
        // A part with no features has no extent to declare, so it
        // declares none. Inventing one here would test the wrong file.
        if extent {
            col["bbox"] = serde_json::json!([cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0]);
        }
        let geo = serde_json::json!({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": col},
        });
        let bbox_fields: Fields = vec![
            Field::new("xmin", DataType::Float64, true),
            Field::new("ymin", DataType::Float64, true),
            Field::new("xmax", DataType::Float64, true),
            Field::new("ymax", DataType::Float64, true),
        ]
        .into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, true),
            Field::new("id", DataType::Int64, false),
            Field::new("bbox", DataType::Struct(bbox_fields.clone()), true),
        ]));

        let (geoms, xs, ys): (Vec<Option<Vec<u8>>>, Vec<Option<f64>>, Vec<Option<f64>>) = if extent
        {
            (0..n)
                .map(|i| {
                    let (x, y) = (cx + (i % 10) as f64 * 0.01, cy + (i / 10) as f64 * 0.01);
                    (Some(wkb_point(x, y)), Some(x), Some(y))
                })
                .collect()
        } else {
            // One row, no shape: null geometry, null covering.
            (vec![None], vec![None], vec![None])
        };
        let rows = geoms.len();
        let coord = |v: &[Option<f64>]| Arc::new(Float64Array::from(v.to_vec())) as ArrayRef;
        let bbox = StructArray::try_new(
            bbox_fields,
            vec![coord(&xs), coord(&ys), coord(&xs), coord(&ys)],
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter(geoms.iter().map(Option::as_deref))),
                Arc::new(Int64Array::from((0..rows as i64).collect::<Vec<_>>())),
                Arc::new(bbox),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(128)).build();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// A partition holding only null geometries describes no extent: its
    /// covering column has no min/max statistics and its `geo` metadata
    /// carries no bbox. That part must not cost every *other* part its
    /// row-group bounding boxes, which is what dropping the dataset's
    /// boxes on the first undescribed part did: one shapeless row left a
    /// 696-file H3 tree with no spatial index at all, and a 1 km viewport
    /// over Boston decoded 305,737 rows instead of 6,684.
    #[test]
    fn a_part_with_no_extent_does_not_void_the_dataset_index() {
        let root = std::env::temp_dir().join(format!("geopq_no_extent_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Sorted first, so the dataset's schema and covering come from
        // the part that has both.
        write_covering_part(&root.join("a_real.parquet"), 200, 10.0, 45.0, true);
        write_covering_part(&root.join("b_no_extent.parquet"), 0, 0.0, 0.0, false);

        let (store, _crs, _info, rg_meta) = open_store(&Source::Dir(root.clone())).unwrap();
        assert_eq!(store.fragments.len(), 2);
        assert_eq!(store.total_rows(), 201);
        assert!(store.covering.is_some(), "the covering column survives");

        let (label, boxes) = rg_meta.expect("one undescribed part must not void the boxes");
        let n_rg = store.rg_starts().len() - 1;
        assert_eq!(boxes.len(), n_rg, "one box per row group");
        assert!(label.contains("covering"), "{label}");

        // Groups belong to the fragment whose rg_offset covers them.
        let null_part = store.fragments.last().unwrap().rg_offset;
        let (real, null): (Vec<u32>, Vec<u32>) = (0..n_rg as u32)
            .partition(|&g| store.frag_of_group(g as usize).rg_offset != null_part);
        assert_eq!(real.len(), 2, "200 rows in 128-row groups");
        assert_eq!(null.len(), 1);

        // The real part keeps usable boxes...
        for &g in &real {
            let b = boxes[g as usize];
            assert!(b.iter().all(|v| v.is_finite()), "group {g}: {b:?}");
            assert!(b[0] >= 9.0 && b[2] <= 11.0 && b[1] >= 44.0 && b[3] <= 46.0, "{b:?}");
        }
        // ...and the part with no extent gets an empty box, which
        // intersects nothing. Its rows have no geometry, so never
        // selecting them loses nothing drawable.
        for &g in &null {
            assert_eq!(
                boxes[g as usize],
                [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
                "group {g}"
            );
        }

        // The whole point: pruning still works. A viewport over the real
        // part selects its groups and only its groups.
        assert_eq!(intersecting_rgs(&boxes, [9.9, 44.9, 10.2, 45.2]), real);
        // And a viewport somewhere else selects nothing at all, rather
        // than every group for want of an index.
        assert!(intersecting_rgs(&boxes, [-50.0, -20.0, -49.0, -19.0]).is_empty());
    }
}

#[cfg(test)]
mod native_2_0_tests {
    use super::*;
    use arrow::array::BinaryArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;
    use std::sync::Arc;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend(1u32.to_le_bytes());
        b.extend(x.to_le_bytes());
        b.extend(y.to_le_bytes());
        b
    }

    /// 256 points on a grid, in 128-row groups. `geo_meta` None writes no
    /// geo key; the geometry field optionally carries the native
    /// GEOMETRY logical type with a CRS string.
    fn write_points(
        path: &std::path::Path,
        geo_meta: Option<serde_json::Value>,
        native_crs: Option<&str>,
    ) {
        let mut geom = Field::new("geometry", DataType::Binary, false);
        if native_crs.is_some() {
            let md = parquet_geospatial::WkbMetadata::new(native_crs, None);
            geom.try_with_extension_type(parquet_geospatial::WkbType::new(Some(md)))
                .unwrap();
        }
        let schema = Arc::new(Schema::new(vec![geom]));
        let wkbs: Vec<Vec<u8>> = (0..256)
            .map(|i| wkb_point(2.0 + (i % 16) as f64 * 0.01, 48.0 + (i / 16) as f64 * 0.01))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BinaryArray::from_iter_values(wkbs.iter()))],
        )
        .unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(128)).build();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        if let Some(geo) = geo_meta {
            w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
                "geo".to_string(),
                geo.to_string(),
            ));
        }
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// No geo metadata at all: the CRS comes from the GEOMETRY logical
    /// type, EPSG:nnnn form.
    #[test]
    fn crs_from_logical_type_without_geo_metadata() {
        let path = std::env::temp_dir().join("geopq_native_crs.parquet");
        write_points(&path, None, Some("EPSG:2154"));
        let (store, crs, info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        assert!(store.encoding.is_wkb());
        assert!(!store.spherical_edges);
        assert!(
            info.geo.version_label.contains("2.0") || info.geo.encoding.contains("GEOMETRY"),
            "{} / {}",
            info.geo.version_label,
            info.geo.encoding
        );
    }

    /// Absent / CRS84-spelled crs strings mean CRS84.
    #[test]
    fn crs_string_spellings() {
        for s in [None, Some("OGC:CRS84"), Some("EPSG:4326"), Some("")] {
            let crs = super::crs_from_type_string(s).unwrap();
            assert_eq!(crs.epsg, Some(4326), "{s:?}");
        }
        let crs = super::crs_from_type_string(Some("weird")).unwrap();
        assert_eq!(crs.epsg, None);
        assert!(crs.name.contains("rendered as CRS84"));
    }

    /// A 2.0-style file with no covering column still gets exact
    /// per-feature rect selection via the WKB envelope scan.
    #[test]
    fn wkb_envelope_scan_selects_subset() {
        let path = std::env::temp_dir().join("geopq_wkb_rect.parquet");
        write_points(&path, None, Some("OGC:CRS84"));
        let (store, _crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        // Grid spans x 2.00..2.15, y 48.00..48.15; select the lower-left
        // quadrant (8 columns x 8 rows of the 16x16 grid).
        let rect = [1.99, 47.99, 2.074, 48.074];
        let ranges = covering_select(&store, 0, rect).unwrap().expect("scan supported");
        let selected: u32 = ranges.iter().map(|(a, b)| b - a).sum();
        assert_eq!(selected, 64, "{ranges:?}");
        // And the planner uses Rect for a sub-extent viewport.
        let boxes = vec![[2.0, 48.0, 2.15, 48.15], [2.0, 48.0, 2.15, 48.15]];
        let sel = plan_viewport_selection(&store, "t", Some(&boxes), Some(rect), None, "test");
        assert!(
            sel.iter().all(|s| matches!(s, GroupSel::Rect(_, _))),
            "{sel:?}"
        );
    }

    /// `edges: spherical` is read from geo metadata, and densification
    /// bows a long parallel-following segment poleward.
    #[test]
    fn spherical_edges_flag_and_densify() {
        let path = std::env::temp_dir().join("geopq_spherical.parquet");
        let geo = serde_json::json!({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": ["Point"],
                "edges": "spherical",
            }},
        });
        write_points(&path, Some(geo), None);
        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.spherical_edges);
        assert!(crs.is_latlong);

        let mut g = geo_types::Geometry::LineString(geo_types::LineString(vec![
            geo_types::Coord { x: 0.0, y: 60.0 },
            geo_types::Coord { x: 90.0, y: 60.0 },
        ]));
        densify_spherical(&mut g, SPHERICAL_MAX_SEG_DEG);
        let geo_types::Geometry::LineString(ls) = &g else { unreachable!() };
        assert!(ls.0.len() > 20, "densified: {}", ls.0.len());
        // The great circle between (0°E, 60°N) and (90°E, 60°N) peaks at
        // asin(1.732/1.871) ≈ 67.8°N.
        let max_lat = ls.0.iter().map(|c| c.y).fold(f64::MIN, f64::max);
        assert!(max_lat > 67.0 && max_lat < 68.5, "max lat {max_lat}");
        // Endpoints intact.
        assert_eq!(ls.0.first().unwrap().x, 0.0);
        assert_eq!(ls.0.last().unwrap().x, 90.0);

        // An equator-following segment stays on the equator.
        let mut eq = geo_types::Geometry::LineString(geo_types::LineString(vec![
            geo_types::Coord { x: 0.0, y: 0.0 },
            geo_types::Coord { x: 40.0, y: 0.0 },
        ]));
        densify_spherical(&mut eq, SPHERICAL_MAX_SEG_DEG);
        let geo_types::Geometry::LineString(els) = &eq else { unreachable!() };
        assert!(els.0.iter().all(|c| c.y.abs() < 1e-9));
    }
}

#[cfg(test)]
mod reload_plan_tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// Reload planning: pruning by the supplied boxes, whole-group reads
    /// without a covering column, everything without a rect.
    #[test]
    fn plan_prunes_and_selects_whole_groups() {
        let path = std::env::temp_dir().join(format!(
            "geopq_reload_plan_{}.parquet",
            std::process::id()
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let wkbs: Vec<Vec<u8>> = (0..300).map(|i| wkb_point(i as f64 * 0.1, 0.0)).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from((0..300i64).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(128))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let (store, ..) = open_store(&Source::Local(path)).unwrap();
        assert_eq!(store.rg_starts().len() - 1, 3, "3 row groups expected");

        let boxes = vec![
            [0.0, 0.0, 1.0, 1.0],
            [10.0, 0.0, 11.0, 1.0],
            [20.0, 0.0, 21.0, 1.0],
        ];
        // Rect over the middle box only: one group, per-feature rect
        // selection (WKB files resolve it via the envelope scan even
        // without a covering column).
        let sel = plan_viewport_selection(
            &store,
            "t",
            Some(&boxes),
            Some([9.5, -1.0, 12.0, 2.0]),
            None,
            "test",
        );
        assert!(matches!(sel.as_slice(), [GroupSel::Rect(1, _)]), "{sel:?}");
        // Disjoint rect: nothing to read.
        let sel = plan_viewport_selection(
            &store,
            "t",
            Some(&boxes),
            Some([50.0, 0.0, 60.0, 1.0]),
            None,
            "test",
        );
        assert!(sel.is_empty(), "{sel:?}");
        // No rect: everything.
        let sel = plan_viewport_selection(&store, "t", Some(&boxes), None, None, "test");
        assert_eq!(sel.len(), 3);
        assert!(matches!(sel[0], GroupSel::All(0)));
        assert!(matches!(sel[2], GroupSel::All(2)));
    }
}

#[cfg(test)]
mod stac_tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// One STAC part: `n` points around (cx, cy), file-level geo bbox.
    fn write_part(path: &std::path::Path, n: usize, cx: f64, cy: f64) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let geo = serde_json::json!({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "bbox": [cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0],
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let wkbs: Vec<Vec<u8>> = (0..n)
            .map(|i| wkb_point(cx + (i % 10) as f64 * 0.01, cy + (i / 10) as f64 * 0.01))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(128))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn write_json(root: &std::path::Path, rel: &str, body: String) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    /// End-to-end: a two-part STAC collection over loopback HTTP opens as
    /// one multi-fragment remote store, pruned to the viewport's parts.
    #[test]
    fn stac_collection_opens_viewport_parts() {
        let root = std::env::temp_dir().join(format!("geopq_stac_open_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_part(&root.join("west.parquet"), 200, -10.0, 45.0);
        write_part(&root.join("east.parquet"), 300, 10.0, 45.0);
        write_json(
            &root,
            "collection.json",
            r#"{"links": [
                {"rel": "item", "href": "./items/west.json"},
                {"rel": "item", "href": "./items/east.json"}
            ]}"#
            .into(),
        );
        let base = crate::data::source::testserver::spawn_dir(root.clone());
        for (name, bbox, rows) in [
            ("west", "[-11.0, 44.0, -9.0, 46.0]", 200),
            ("east", "[9.0, 44.0, 11.0, 46.0]", 300),
        ] {
            write_json(
                &root,
                &format!("items/{name}.json"),
                format!(
                    r#"{{"bbox": {bbox}, "properties": {{"num_rows": {rows}}},
                        "assets": {{"aws": {{"href": "{base}/{name}.parquet"}}}}}}"#
                ),
            );
        }
        let source = Source::Stac {
            url: format!("{base}/collection.json"),
            name: "building".into(),
        };

        // No viewport: every part joins the store.
        let (store, crs, info, rg_meta) = open_store_with_view(&source, None).unwrap();
        assert!(crs.is_latlong);
        assert_eq!(store.fragments.len(), 2);
        assert_eq!(store.total_rows(), 500);
        assert!(store.part_cols.is_empty(), "no hive columns on STAC parts");
        assert_eq!(info.files, 2);
        let (_, boxes) = rg_meta.expect("file-level geo bboxes back pruning");
        assert_eq!(boxes.len(), store.rg_starts().len() - 1);

        // A viewport over the west part only opens that file.
        let (store, _, info, _) =
            open_store_with_view(&source, Some(ViewHint { rect: [-12.0, 43.0, -8.0, 47.0], view_px: 1600.0 })).unwrap();
        assert_eq!(store.fragments.len(), 1);
        assert_eq!(store.total_rows(), 200);
        assert_eq!(info.files, 1);

        // A viewport intersecting nothing is a load error, not an empty map.
        let Err(err) = open_store_with_view(&source, Some(ViewHint { rect: [100.0, 0.0, 110.0, 5.0], view_px: 1600.0 })) else {
            panic!("disjoint viewport must not open a store");
        };
        assert!(err.contains("no parts"), "{err}");
    }

    /// End-to-end over https: a hive-partitioned dataset published with
    /// this app's own `collection.json` opens from its prefix URL as one
    /// layer, partition columns and viewport pruning included.
    #[test]
    fn an_https_prefix_opens_through_its_collection() {
        let root = std::env::temp_dir().join(format!("geopq_stac_https_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Toulouse and Paris: far enough apart that a viewport over one
        // cannot be read as covering the other.
        write_part(&root.join("dep=31/part-0.parquet"), 200, 1.4, 43.6);
        write_part(&root.join("dep=75/part-0.parquet"), 300, 2.35, 48.85);
        let written =
            crate::data::stac::write_for_output(&root, "Parcelles", &Crs::wgs84()).unwrap();
        assert_eq!(written, root.join("collection.json"));
        let base = crate::data::source::testserver::spawn_dir(root.clone());

        // A prefix routes to the collection sitting at it; HTTP has no
        // listing, so this document is the only way in.
        let source = crate::data::source::route_uri(&format!("{base}/"), None, None);
        let Source::Stac { url, .. } = &source else {
            panic!("a prefix must open as a collection, not {source:?}");
        };
        assert_eq!(url, &format!("{base}/collection.json"));

        let (store, crs, info, rg_meta) = open_store_with_view(&source, None).unwrap();
        assert!(crs.is_latlong);
        assert_eq!(store.fragments.len(), 2);
        assert_eq!(store.total_rows(), 500);
        assert_eq!(info.files, 2);
        let (_, boxes) = rg_meta.expect("file-level geo bboxes back pruning");
        assert_eq!(boxes.len(), store.rg_starts().len() - 1);
        // The hive segments of the asset hrefs are a queryable column,
        // exactly as the s3:// prefix path produces.
        assert_eq!(store.part_cols, vec!["dep".to_string()]);
        let values: Vec<_> = store
            .fragments
            .iter()
            .map(|f| f.part_values[0].clone())
            .collect();
        assert_eq!(values, vec![Some("31".into()), Some("75".into())]);

        // Per-asset bboxes make pruning real: opening both parts here
        // would also be the symptom of a reader handing every part the
        // collection's own extent.
        let (paris, _, info, _) =
            open_store_with_view(&source, Some(ViewHint { rect: [2.0, 48.5, 2.7, 49.1], view_px: 1600.0 })).unwrap();
        assert_eq!(paris.fragments.len(), 1);
        assert_eq!(paris.total_rows(), 300);
        assert_eq!(info.files, 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A prefix with parquet in it but no manifest: the failure has to
    /// name the convention, since nothing else will.
    #[test]
    fn an_https_prefix_without_a_collection_explains_itself() {
        let root = std::env::temp_dir().join(format!("geopq_stac_bare_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_part(&root.join("part-0.parquet"), 10, 0.0, 0.0);
        let base = crate::data::source::testserver::spawn_dir(root.clone());
        let source = crate::data::source::route_uri(&format!("{base}/"), None, None);
        let Err(err) = open_store_with_view(&source, None) else {
            panic!("a prefix with no manifest must not open");
        };
        assert!(err.contains("cannot list a directory"), "{err}");
        assert!(err.contains("collection.json"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Live probe against the public Overture STAC catalog (network):
    ///   cargo test --release stac_live_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn stac_live_probe() {
        use crate::data::repo;
        let base = "https://stac.overturemaps.org";
        let snaps = repo::fetch_snapshots_stac(base).unwrap();
        println!("releases: {:?}", snaps.iter().map(|s| &s.label).collect::<Vec<_>>());
        let ds = repo::discover_datasets_stac(base, "latest/").unwrap();
        println!("themes: {:?}", ds.iter().map(|d| &d.name).collect::<Vec<_>>());
        let snap = &snaps[0].path;
        let m = repo::fetch_stac_manifest(base, snap, "divisions").unwrap();
        println!("divisions types: {:?}", m.themes);
        // Open division_area around Paris; item pruning + footer reads live.
        let source = Source::Stac {
            url: repo::stac_collection_url(base, snap, "divisions", "division_area"),
            name: "division_area".into(),
        };
        let t = Instant::now();
        let (store, crs, info, rg_meta) =
            open_store_with_view(&source, Some(ViewHint { rect: [2.0, 48.5, 3.0, 49.2], view_px: 1600.0 })).unwrap();
        println!(
            "opened {} parts / {} rows / {} rgs in {:?} (crs {})",
            info.files,
            store.total_rows(),
            store.rg_starts().len() - 1,
            t.elapsed(),
            crs.name,
        );
        assert!(crs.is_latlong);
        assert!(info.files >= 1);
        assert!(rg_meta.is_some(), "covering stats expected on Overture");
    }
}

#[cfg(test)]
mod xy_tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    #[test]
    fn lonlat_columns_synthesize_point_geometry() {
        let dir = std::env::temp_dir().join("geopq_xy_open");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("grid.parquet");

        // Plain table: lat, lon, value — no geo metadata, no geometry.
        let n = 400usize;
        let (mut lats, mut lons, mut ids) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            lons.push(-71.0 + (i % 20) as f64 * 0.01);
            lats.push(42.0 + (i / 20) as f64 * 0.01);
            ids.push(i as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("lat", DataType::Float64, false),
            Field::new("lon", DataType::Float64, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(lats.clone())),
                Arc::new(Float64Array::from(lons.clone())),
                Arc::new(Int64Array::from(ids)),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(128))
            .build();
        let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let (store, crs, info, rg_meta) = open_store(&Source::Local(path)).unwrap();
        assert_eq!(store.xy_geom, Some((1, 0)), "x=lon, y=lat");
        assert!(crs.name.contains("assumed CRS84"), "{}", crs.name);
        let geom_idx = store.schema.index_of("geometry").unwrap();
        assert_eq!(geom_idx, store.geom_col);
        assert_eq!(store.encoding, GeomEncoding::Point);
        assert!(info.geo.version_label.contains("coordinate columns"));

        // Row-group bboxes come from the coordinate column statistics.
        let (label, boxes) = rg_meta.expect("stats-backed bboxes");
        assert!(label.contains("x/y"), "{label}");
        assert_eq!(boxes.len(), store.rg_starts().len() - 1);
        assert!(boxes.iter().all(|b| b[0] >= -71.01 && b[3] <= 42.5));

        // Geometry fetch: synthesized points match the source columns.
        let rows = [0u32, 150, 399];
        let geoms = store.fetch_geoms(&rows).unwrap();
        for (row, g) in geoms {
            match g.expect("point") {
                geo_types::Geometry::Point(p) => {
                    assert_eq!(p.x(), lons[row as usize], "row {row}");
                    assert_eq!(p.y(), lats[row as usize], "row {row}");
                }
                other => panic!("expected point, got {other:?}"),
            }
        }

        // Full-row fetch exposes the virtual struct for the info panel.
        let row = store.fetch_row(5).unwrap();
        assert!(row.schema().index_of("geometry").is_ok());

        // Map build path: every row tessellates.
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (_geom, rows_built, bad) =
            build_geometry_for_test(&store, &crs, &display).unwrap();
        assert_eq!(rows_built, n);
        assert_eq!(bad, 0);

        // Per-feature rect selection through the coordinate columns:
        // only in-rect rows of a group are selected, so lat-ordered
        // row groups don't decode as world-wide strips.
        let rect = [-70.95, 42.05, -70.9, 42.1];
        let ranges = covering_select(&store, 0, rect)
            .unwrap()
            .expect("x/y stores have an exact covering");
        let selected: u32 = ranges.iter().map(|(s, e)| e - s).sum();
        let expect = lons
            .iter()
            .zip(&lats)
            .take(128) // group 0
            .filter(|(x, y)| {
                **x >= rect[0] && **x <= rect[2] && **y >= rect[1] && **y <= rect[3]
            })
            .count();
        assert_eq!(selected as usize, expect);
        assert!(selected > 0 && (selected as usize) < 128, "{selected}");

        // Decimated preview: every 4th row decodes, resolved states say so.
        let n_groups = store.rg_starts().len() - 1;
        let sel: Vec<GroupSel> = (0..n_groups)
            .map(|g| GroupSel::Preview { group: g as u32, rect: None, stride: 4 })
            .collect();
        let (_gp, rows_prev, _, _, resolved) =
            build_geometry(&store, &crs, &display, None, sel, None, None).unwrap();
        let expect_prev: usize = (0..n_groups)
            .map(|g| {
                let rows = (store.rg_starts()[g + 1] - store.rg_starts()[g]) as usize;
                rows.div_ceil(4)
            })
            .sum();
        assert_eq!(rows_prev, expect_prev);
        assert!(resolved
            .iter()
            .all(|(_, st)| matches!(st, GroupLoad::Preview { stride: 4, rect: None })));
        assert!(
            !GroupLoad::Preview { stride: 4, rect: None }.covers([0.0, 0.0, 1.0, 1.0])
        );

        // Data-driven styling: graduated bins on the id column spread
        // chunks across bins, and binning matches the value math.
        let id_col = store.schema.index_of("id").unwrap();
        let (lo, hi) = store.column_range(id_col).expect("stats range");
        assert_eq!((lo, hi), (0.0, (n - 1) as f64));
        let style = StyleSel {
            col: id_col,
            binning: Binning::Breaks(crate::data::layer::equal_interval_breaks(lo, hi, crate::data::layer::STYLE_BINS)),
            per_area: false,
            outlines: true,
        };
        let (g_styled, rows_styled, _, _, _) =
            build_geometry_styled_for_test(&store, &crs, &display, &style).unwrap();
        assert_eq!(rows_styled, n);
        let mut bins: Vec<u8> = g_styled.chunks.iter().map(|c| c.bin).collect();
        bins.sort_unstable();
        bins.dedup();
        assert!(bins.len() > 4, "value spread must produce several bins: {bins:?}");
        let total_pts: usize = g_styled.chunks.iter().map(|c| c.point_instances.len()).sum();
        assert_eq!(total_pts, n, "binning must not drop features");

        // Optimize materializes the x/y layer into real GeoParquet:
        // WKB points, geo metadata, covering — loadable without synthesis.
        let opt_dst = dir.join("grid_optimized.parquet");
        let opts = crate::data::optimize::OptimizeOptions {
            row_group_size: 128,
            xy_geom: store.xy_geom,
            ..Default::default()
        };
        let rep = crate::data::optimize::optimize(
            &store.source,
            &opt_dst,
            &opts,
            crs.epsg,
            None,
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(rep.rows, n as u64);
        let (opt_store, opt_crs, opt_info, _) =
            open_store(&Source::Local(opt_dst)).unwrap();
        assert!(opt_store.xy_geom.is_none(), "output has a real geometry column");
        assert_eq!(opt_store.total_rows(), n as u64);
        assert!(opt_store.covering.is_some(), "covering written");
        assert!(opt_info.geo.version_label.contains("GeoParquet"));
        assert!(opt_crs.is_latlong);
        // The optimizer's own output must pass the quality gate.
        let q = opt_info.quality.as_ref().expect("quality report");
        assert!(q.indexable, "optimized output must be indexable: {:?}", q.checks);
        assert!(q.geom_bytes > 0);
        // Original lon/lat stay as ordinary attribute columns.
        assert!(opt_store.schema.index_of("lon").is_ok());
        let g = opt_store.fetch_geoms(&[0]).unwrap();
        assert!(matches!(
            g[0].1.as_ref().unwrap(),
            geo_types::Geometry::Point(_)
        ));

        // SQL over the synthesized geometry, including spatial pushdown
        // through the stats-backed bboxes.
        use crate::sql::engine::{run_query_for_test, SqlLayer};
        let layers = vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: Some(Arc::new(boxes)),
        }];
        let count = |q: &str| -> i64 {
            let out = run_query_for_test(q, &layers).unwrap();
            out.batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap()
                .value(0)
        };
        assert_eq!(count("select count(*) from t"), n as i64);
        assert_eq!(
            count(
                "select count(*) from t \
                 where st_intersects(geometry, st_makeenvelope(-70.95, 41.9, -70.8, 42.05))"
            ),
            count(
                "select count(*) from t \
                 where lon >= -70.95 and lon <= -70.8 and lat <= 42.05"
            )
        );
        // st_x/st_y read through the WKB normalization.
        let out = run_query_for_test(
            "select st_x(geometry) x from t where id = 21",
            &layers,
        )
        .unwrap();
        let x = out
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(x, lons[21]);
    }
}

#[cfg(test)]
mod preview_rect_tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    /// A rect-filtered preview must resolve to a state that records the
    /// rect, and value sampling must reproduce exactly the loaded
    /// selection (never fetching rows the preview skipped).
    #[test]
    fn rect_preview_state_and_sampling_match_load() {
        let dir = std::env::temp_dir().join("geopq_preview_rect");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("grid.parquet");

        // 20x20 lon/lat grid, groups of 128 rows (x/y point store: the
        // coordinate columns are the covering).
        let n = 400usize;
        let (mut lons, mut lats) = (Vec::new(), Vec::new());
        for i in 0..n {
            lons.push(-71.0 + (i % 20) as f64 * 0.01);
            lats.push(42.0 + (i / 20) as f64 * 0.01);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("lon", DataType::Float64, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(lons.clone())),
                Arc::new(Float64Array::from(lats.clone())),
                Arc::new(Int64Array::from((0..n as i64).collect::<Vec<i64>>())),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(128))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let (store, crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.xy_geom.is_some());
        let n_groups = store.rg_starts().len() - 1;
        assert_eq!(n_groups, 4);

        // Rows of group 0 inside the rect, decimated at 1/3 — the ground
        // truth the loader must reproduce.
        let rect = [-70.95, 42.0, -70.85, 42.06];
        let stride = 3u32;
        let in_rect: Vec<u32> = (0..128u32)
            .filter(|&i| {
                let (x, y) = (lons[i as usize], lats[i as usize]);
                x >= rect[0] && x <= rect[2] && y >= rect[1] && y <= rect[3]
            })
            .collect();
        let expected: Vec<u32> = in_rect
            .iter()
            .copied()
            .step_by(stride as usize)
            .collect();
        assert!(expected.len() > 3, "fixture must select several rows");

        // Refinement budgets use the exact coordinate selection, not a
        // row-group bbox area estimate. The resolved ranges are then reused
        // by geometry decoding rather than scanning the coordinates twice.
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let plan = prepare_refinement_jobs(
            &store,
            vec![GroupSel::Rect(0, rect)],
            in_rect.len() as u64,
            MAX_BUILD_GEOM_BYTES,
            &cancel,
        )
        .unwrap();
        match plan {
            RefinePlan::Ready(jobs) => match jobs.as_slice() {
                [GroupSel::ResolvedRect { ranges, .. }] => {
                    let selected: usize =
                        ranges.iter().map(|&(s, e)| (e - s) as usize).sum();
                    assert_eq!(selected, in_rect.len());
                }
                _ => panic!("expected an exact resolved rect"),
            },
            RefinePlan::Deferred { .. } => panic!("selection unexpectedly deferred"),
        }
        assert!(matches!(
            prepare_refinement_jobs(
                &store,
                vec![GroupSel::Rect(0, rect)],
                in_rect.len() as u64 - 1,
                MAX_BUILD_GEOM_BYTES,
                &cancel,
            )
            .unwrap(),
            RefinePlan::Deferred { .. }
        ));

        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let sel = vec![GroupSel::Preview { group: 0, rect: Some(rect), stride }];
        let (_g, rows, _bad, _boxes, resolved) =
            build_geometry(&store, &crs, &display, None, sel, None, None).unwrap();
        assert_eq!(rows, expected.len(), "only in-rect decimated rows decode");
        assert_eq!(resolved.len(), 1);
        match &resolved[0] {
            (0, GroupLoad::Preview { stride: s, rect: r }) => {
                assert_eq!(*s, stride);
                assert_eq!(*r, Some(rect), "resolved state must keep the rect");
            }
            other => panic!("unexpected resolved state: {other:?}"),
        }

        // Sampling classifies only what was loaded: with an uncapped
        // sample, the returned ids are exactly the loaded rows.
        let mut loaded = vec![GroupLoad::None; n_groups];
        loaded[0] = resolved[0].1.clone();
        let id_col = store.schema.index_of("id").unwrap();
        let mut vals =
            sample_loaded_values(&store, &loaded, id_col, 10_000, false, false, None)
                .unwrap();
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let got: Vec<u32> = vals.iter().map(|v| *v as u32).collect();
        assert_eq!(got, expected, "sampling must reproduce the preview selection");
    }
}

#[cfg(test)]
mod budget_tests {
    use super::{preview_stride, MAX_BUILD_GEOM_BYTES, MAX_BUILD_ROWS};

    #[test]
    fn the_budget_counts_bytes_as_well_as_rows() {
        // Points: millions of rows, nothing to decode. Fits.
        assert_eq!(preview_stride(2_000_000, 40 << 20), None);
        // MassGIS parcels: 2.56M rows, 970 MB. Over on both counts now
        // that the byte target reflects the mesh those bytes become —
        // and bytes, not rows, set the stride (1/8, not 1/3).
        assert_eq!(preview_stride(2_557_399, 969 << 20), Some(8));
        // CORINE land cover: 2.38M rows — *under* the row budget, which
        // is why this file used to take the full-decode path and eat
        // tens of gigabytes — but 7.18 GB of geometry.
        let stride = preview_stride(2_375_406, 7_180 << 20).unwrap();
        assert!(stride >= 28, "expected a deep stride, got 1/{stride}");
        // Exactly at both limits is still fine; a byte over is not.
        assert_eq!(preview_stride(MAX_BUILD_ROWS, MAX_BUILD_GEOM_BYTES), None);
        assert!(preview_stride(MAX_BUILD_ROWS, MAX_BUILD_GEOM_BYTES + 1).is_some());
        // A preview is never a no-op stride.
        assert!(preview_stride(MAX_BUILD_ROWS + 1, 0).unwrap() >= 2);
    }
}

#[cfg(test)]
mod norm_bin_tests {
    use super::{ground_area, norm_bin};

    fn square(cx: f64, cy: f64, half: f64) -> geo_types::Geometry<f64> {
        use geo_types::{Coord, LineString, Polygon};
        geo_types::Geometry::Polygon(Polygon::new(
            LineString(vec![
                Coord { x: cx - half, y: cy - half },
                Coord { x: cx + half, y: cy - half },
                Coord { x: cx + half, y: cy + half },
                Coord { x: cx - half, y: cy + half },
                Coord { x: cx - half, y: cy - half },
            ]),
            vec![],
        ))
    }

    #[test]
    fn geographic_areas_are_measured_on_an_equal_area_projection() {
        // Two one-degree squares, at the equator and at 60°N. In degrees²
        // they are identical; on the ground the northern one covers about
        // half as much, so normalizing by the raw shoelace would rank it
        // twice as dense for no reason but its latitude.
        let equator = ground_area(&square(0.0, 0.5, 0.5), true);
        let north = ground_area(&square(0.0, 60.5, 0.5), true);
        assert!(
            (north / equator - 0.5).abs() < 0.02,
            "{north} vs {equator} m²"
        );
        // A degree square at the equator is ~12,300 km².
        assert!(
            (equator - 1.23e10).abs() / 1.23e10 < 0.02,
            "{equator} m² for a degree square"
        );
        // Longitude must not matter: the same square moved east measures
        // the same, which a cylindrical equal-area projection guarantees
        // and a fitted per-layer projection would only approximate.
        let east = ground_area(&square(150.0, 60.5, 0.5), true);
        assert!((east / north - 1.0).abs() < 1e-9, "{east} vs {north}");

        // Projected data is already in ground units and must not move.
        let projected = square(500_000.0, 6_000_000.0, 100.0);
        assert!((ground_area(&projected, false) - 40_000.0).abs() < 1e-6);
    }

    #[test]
    fn both_decode_paths_measure_the_same_area() {
        // The WKB path measures through `ground_area`, the GeoArrow bulk
        // path through `equal_area_projector` and its own shoelace. They
        // must agree, or the same data would classify differently
        // depending on how the file happens to encode its geometry.
        let poly = square(7.0, 62.0, 0.25);
        let wkb = ground_area(&poly, true);

        let proj = super::equal_area_projector(true).expect("geographic");
        let geo_types::Geometry::Polygon(p) = &poly else {
            unreachable!()
        };
        let pts: Vec<(f64, f64)> = p
            .exterior()
            .0
            .iter()
            .map(|c| proj(c.x, c.y).expect("inside the projection domain"))
            .collect();
        let mut shoelace = 0.0;
        for k in 0..pts.len() - 1 {
            shoelace += pts[k].0 * pts[k + 1].1 - pts[k + 1].0 * pts[k].1;
        }
        let ga = (shoelace * 0.5).abs();
        assert!(
            (ga - wkb).abs() / wkb < 1e-9,
            "geoarrow {ga} vs wkb {wkb}"
        );
        assert!(super::equal_area_projector(false).is_none());
    }

    #[test]
    fn area_normalization_orders_by_density() {
        // Breaks on value/area (density): 10 and 100.
        let breaks = vec![10.0, 100.0];
        // Same value, different areas: the small parcel is denser.
        assert_eq!(norm_bin(1000.0, 5.0, &breaks), 2, "200/unit: top class");
        assert_eq!(norm_bin(1000.0, 20.0, &breaks), 1, "50/unit: middle");
        assert_eq!(norm_bin(1000.0, 1000.0, &breaks), 0, "1/unit: bottom");
        // Nulls and degenerate areas fall to bin 0 like the plain path.
        assert_eq!(norm_bin(f64::NAN, 5.0, &breaks), 0);
        assert_eq!(norm_bin(1000.0, 0.0, &breaks), 0);
        assert_eq!(norm_bin(1000.0, -1.0, &breaks), 0);
    }
}

/// Geometry-column detection on files with no `geo` metadata at all: the
/// column has to prove itself, by name and then by its bytes.
#[cfg(test)]
mod wkb_detection_tests {
    use super::*;
    use arrow::array::{ArrayRef, BinaryArray, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend(1u32.to_le_bytes());
        b.extend(x.to_le_bytes());
        b.extend(y.to_le_bytes());
        b
    }

    /// Write a file with no `geo` key from (name, array) pairs.
    fn write(path: &std::path::Path, cols: Vec<(&str, ArrayRef)>) {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let batch =
            RecordBatch::try_new(schema.clone(), cols.into_iter().map(|(_, a)| a).collect())
                .unwrap();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// A blob column is binary, and before the probe existed it was
    /// adopted as geometry on sight — a map of decoded PNG headers. The
    /// x/y columns beside it are the real answer.
    #[test]
    fn a_thumbnail_column_is_not_geometry() {
        let path = std::env::temp_dir().join("geopq_thumbnail.parquet");
        let blobs: Vec<Vec<u8>> = (0..64)
            .map(|i| {
                let mut v = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
                v.push(i as u8);
                v
            })
            .collect();
        let lon: Vec<f64> = (0..64).map(|i| 2.0 + i as f64 * 0.01).collect();
        let lat: Vec<f64> = (0..64).map(|i| 48.0 + i as f64 * 0.01).collect();
        write(
            &path,
            vec![
                ("thumbnail", Arc::new(BinaryArray::from_iter_values(blobs.iter()))),
                ("lon", Arc::new(Float64Array::from(lon))),
                ("lat", Arc::new(Float64Array::from(lat))),
            ],
        );
        let (store, crs, info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.xy_geom.is_some(), "points must come from lon/lat");
        assert_eq!(store.encoding, GeomEncoding::Point);
        assert!(crs.name.contains("points from"), "{}", crs.name);
        assert!(info.geo.encoding.contains("lon"), "{}", info.geo.encoding);
    }

    /// …and with no coordinate columns to fall back on, a blob column is
    /// an honest failure rather than a garbage layer.
    #[test]
    fn a_blob_only_file_refuses_to_open() {
        let path = std::env::temp_dir().join("geopq_blob_only.parquet");
        let blobs: Vec<Vec<u8>> = (0..64).map(|i| vec![9u8, 9, 9, 9, 9, i as u8]).collect();
        write(
            &path,
            vec![("thumbnail", Arc::new(BinaryArray::from_iter_values(blobs.iter())))],
        );
        let err = open_store(&Source::Local(path))
            .err()
            .expect("a blob column is not geometry");
        assert!(err.contains("read as WKB"), "{err}");
    }

    /// A binary column named outside the list still opens when its bytes
    /// are WKB: the name list ranks candidates, it does not gate them.
    #[test]
    fn an_oddly_named_binary_column_opens_when_it_is_wkb() {
        let path = std::env::temp_dir().join("geopq_odd_name.parquet");
        let wkbs: Vec<Vec<u8>> = (0..64).map(|i| wkb_point(i as f64, 1.0)).collect();
        write(
            &path,
            vec![("shape", Arc::new(BinaryArray::from_iter_values(wkbs.iter())))],
        );
        let (store, _crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.xy_geom.is_none());
        assert_eq!(store.schema.field(store.geom_col).name(), "shape");
    }

    /// `geometry_wkb` is one of the conventional names, and a named
    /// candidate wins over an unnamed binary column that also passes.
    #[test]
    fn geometry_wkb_is_a_known_name() {
        let path = std::env::temp_dir().join("geopq_geometry_wkb.parquet");
        let wkbs: Vec<Vec<u8>> = (0..64).map(|i| wkb_point(i as f64, 1.0)).collect();
        write(
            &path,
            vec![
                ("payload", Arc::new(BinaryArray::from_iter_values(wkbs.iter()))),
                ("geometry_wkb", Arc::new(BinaryArray::from_iter_values(wkbs.iter()))),
            ],
        );
        let (store, _crs, _info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert_eq!(store.schema.field(store.geom_col).name(), "geometry_wkb");
    }

    /// Spark / Sedona exports hand out WKB as base64 text. The store
    /// decodes it on read, so the layer is an ordinary WKB layer.
    #[test]
    fn base64_wkb_text_column_opens_and_decodes() {
        use base64::Engine as _;
        let path = std::env::temp_dir().join("geopq_base64_wkb.parquet");
        let b64: Vec<String> = (0..64)
            .map(|i| {
                base64::engine::general_purpose::STANDARD
                    .encode(wkb_point(2.0 + i as f64 * 0.01, 48.0))
            })
            .collect();
        write(
            &path,
            vec![("geometry", Arc::new(StringArray::from(b64)))],
        );
        let (store, _crs, info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.base64_wkb);
        assert!(store.encoding.is_wkb());
        // The schema the app plans against says bytes, not text.
        assert_eq!(store.schema.field(store.geom_col).data_type(), &DataType::Binary);
        assert!(info.geo.encoding.contains("base64"), "{}", info.geo.encoding);
        // And a read really does yield decodable WKB.
        let geoms = store.fetch_geoms(&[0, 5, 63]).unwrap();
        assert_eq!(geoms.len(), 3);
        for (row, g) in geoms {
            let g = g.unwrap_or_else(|| panic!("row {row} did not decode"));
            assert!(matches!(g, geo_types::Geometry::Point(_)));
        }
    }

    /// Text that is not base64, in a column named `geometry`: rejected,
    /// like any other column that fails the probe.
    #[test]
    fn plain_text_named_geometry_is_not_adopted() {
        let path = std::env::temp_dir().join("geopq_text_geometry.parquet");
        let names: Vec<String> = (0..64).map(|i| format!("feature {i}")).collect();
        write(&path, vec![("geometry", Arc::new(StringArray::from(names)))]);
        let err = open_store(&Source::Local(path))
            .err()
            .expect("a blob column is not geometry");
        assert!(err.contains("read as WKB"), "{err}");
    }

    #[test]
    fn wkb_header_check_accepts_iso_and_ewkb() {
        // Little-endian point, big-endian polygon.
        assert!(super::is_wkb_header(&wkb_point(1.0, 2.0)));
        assert!(super::is_wkb_header(&[0, 0, 0, 0, 3]));
        // ISO Z/M/ZM offsets, and the EWKB flag bits.
        assert!(super::is_wkb_header(&[1, 0xE9, 0x03, 0, 0])); // 1001, PointZ
        assert!(super::is_wkb_header(&[1, 0x01, 0, 0, 0x20])); // SRID flag
        // Not WKB: bad byte order, unknown type, truncated.
        assert!(!super::is_wkb_header(&[2, 1, 0, 0, 0]));
        assert!(!super::is_wkb_header(&[1, 0x63, 0, 0, 0])); // type 99
        assert!(!super::is_wkb_header(&[1, 0, 0, 0, 0])); // type 0
        assert!(!super::is_wkb_header(&[1, 1, 0]));
        assert!(!super::is_wkb_header(&[0x89, b'P', b'N', b'G', 0x0d]));
    }
}

/// Cloud Optimized GeoParquet Profile: reading the levels, and planning
/// against the prefix they name.
#[cfg(test)]
mod cogp_tests {
    use super::*;
    use crate::data::cogp::Pruning;

    fn fixture() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_cogp.parquet"
        ));
        p.exists().then_some(p)
    }

    /// The levels come off the footer, and level selection matches the
    /// metadata it was read from: for each declared level, asking for
    /// exactly its gsd must select exactly its prefix.
    #[test]
    fn cogp_levels_open_and_select_their_own_prefix() {
        let Some(path) = fixture() else {
            eprintln!("fixture missing, skipping");
            return;
        };
        let (store, _crs, info, rg) = open_store(&Source::Local(path)).unwrap();
        let rg_rows: Vec<u64> =
            store.rg_starts().windows(2).map(|w| w[1] - w[0]).collect();
        let levels = store.cogp.as_ref().expect("cogp metadata");
        assert_eq!(levels.version, "0.1.0");
        // The spec's own structural invariants, re-checked against what
        // the loader kept: coarse to fine, ending at the last row group.
        assert!(levels.levels.windows(2).all(|w| w[0].row_group_end < w[1].row_group_end));
        assert!(levels.levels.windows(2).all(|w| w[0].gsd > w[1].gsd));
        let n_rg = store.rg_starts().len() - 1;
        assert_eq!(levels.levels.last().unwrap().row_group_end, n_rg - 1);
        // cogp convert reuses the input's covering column, so this is the
        // profile as published rather than the 2.0 extension.
        assert_eq!(levels.pruning, Pruning::Covering);

        for (i, l) in levels.levels.iter().enumerate() {
            assert_eq!(levels.level_for_gsd(l.gsd), i, "level {i} at its own gsd");
            assert_eq!(levels.row_group_end_for_gsd(l.gsd), l.row_group_end);
            // Just finer than this level still selects it, until the next
            // level's gsd is reached.
            if let Some(next) = levels.levels.get(i + 1) {
                assert_eq!(levels.level_for_gsd(next.gsd * 1.001), i, "between {i} and {}", i + 1);
            }
        }
        // Coarser than the file: level 0, per SPEC §7.1.
        assert_eq!(levels.level_for_gsd(levels.levels[0].gsd * 10.0), 0);

        // And the file info panel says what was found.
        let line = info.geo.cogp.as_deref().expect("summary line");
        assert!(line.starts_with("COGP 0.1.0: "), "{line}");
        assert!(!line.contains("2.0 extension"), "{line}");
        assert!(line.contains("prefix row groups 1/"), "{line}");

        // The gate must not condemn a correct COGP layout. Levels
        // overlap each other by construction — measured as one file this
        // fixture reads far past the C2 threshold — so C2 measures
        // inside them (see `quality::Clustering::worst`).
        let q = info.quality.as_ref().unwrap();
        let c2 = q.checks.iter().find(|c| c.code == "C2").unwrap();
        assert_eq!(c2.status, crate::data::quality::Status::Pass, "{}", c2.detail);
        assert!(c2.detail.starts_with("within COGP levels:"), "{}", c2.detail);
        assert!(q.indexable, "a file the reference converter wrote must open ungated");
        // …and the layer panel reads the same run, so it cannot call the
        // same file poorly clustered.
        let rg = rg.as_ref().expect("row-group boxes");
        let boxes = crate::data::layer::RgBboxes::new(
            rg.0.clone(),
            rg.1.clone(),
            store.cogp.as_ref().map(|c| c.runs(&rg_rows)).as_deref(),
        );
        assert!(!boxes.poorly_clustered(), "avg ×{:.1}", boxes.avg_overlap);
        let c8 = q.checks.iter().find(|c| c.code == "C8").unwrap();
        assert_eq!(c8.status, crate::data::quality::Status::Pass);
        assert!(!c8.gating);
        assert!(c8.detail.contains("COGP 0.1.0:"), "{}", c8.detail);
    }

    /// A world-wide view of a COGP layer plans the coarse prefix and
    /// nothing else — exact features at that scale, no boxes, no stride.
    #[test]
    fn a_wide_view_plans_the_coarse_prefix_exactly() {
        let Some(path) = fixture() else {
            eprintln!("fixture missing, skipping");
            return;
        };
        let (store, crs, _info, rg) = open_store(&Source::Local(path)).unwrap();
        let (_, boxes) = rg.expect("row-group boxes");
        let n_rg = store.rg_starts().len() - 1;
        let extent = union_of(&boxes).unwrap();

        // The whole dataset on a 1600 px map: ~100 m of ground per pixel
        // on a state-sized EPSG:26986 extent.
        let view_px = 1600.0;
        let gsd = view_gsd(extent, view_px, &crs).unwrap();
        let end = cogp_prefix_end(&store, Some(extent), view_px, &crs).unwrap();
        assert_eq!(end as usize, store.cogp.as_ref().unwrap().row_group_end_for_gsd(gsd));
        assert!(end as usize + 1 < n_rg, "a wide view must not need every group");

        let sel = plan_viewport_selection(
            &store,
            "t",
            Some(&boxes),
            Some(extent),
            Some(end),
            "test",
        );
        assert!(!sel.is_empty());
        assert!(sel.iter().all(|s| s.group() <= end), "planned past the prefix: {sel:?}");
        // Exact rows, not an approximation: this is the whole point of
        // putting the level check before the budget fallbacks.
        assert!(
            sel.iter()
                .all(|s| matches!(s, GroupSel::All(_) | GroupSel::Rect(_, _)
                    | GroupSel::ResolvedRect { .. })),
            "{sel:?}"
        );

        // Zooming in to a tenth of the extent moves the level finer, so
        // the prefix can only grow — previously read groups stay valid.
        let (w, h) = (extent[2] - extent[0], extent[3] - extent[1]);
        let zoomed = [
            extent[0] + w * 0.45,
            extent[1] + h * 0.45,
            extent[0] + w * 0.55,
            extent[1] + h * 0.55,
        ];
        let zoomed_end = cogp_prefix_end(&store, Some(zoomed), view_px, &crs).unwrap();
        assert!(zoomed_end >= end, "{zoomed_end} < {end}");
    }

    /// Without levels nothing changes: the planner sees the whole file.
    #[test]
    fn a_plain_file_has_no_prefix() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, info, _rg) = open_store(&Source::Local(path)).unwrap();
        assert!(store.cogp.is_none());
        assert!(info.geo.cogp.is_none());
        assert!(cogp_prefix_end(&store, Some([0.0, 0.0, 1.0, 1.0]), 800.0, &crs).is_none());
        // C8 does not penalise it.
        let c8 = info
            .quality
            .as_ref()
            .unwrap()
            .checks
            .iter()
            .find(|c| c.code == "C8")
            .unwrap();
        assert_eq!(c8.status, crate::data::quality::Status::Pass);
        assert!(c8.detail.contains("optional"), "{}", c8.detail);
    }

    /// Metres per pixel: degrees convert at the viewport's centre
    /// latitude, projected units pass through.
    #[test]
    fn view_gsd_converts_degrees_at_the_centre_latitude() {
        let mut geo = Crs::wgs84();
        geo.is_latlong = true;
        // One degree of longitude over 1000 px, at the equator.
        let g = view_gsd([0.0, -0.5, 1.0, 0.5], 1000.0, &geo).unwrap();
        assert!((g - 111.32).abs() < 0.1, "{g}");
        // The same span at 60°N covers half the ground.
        let g60 = view_gsd([0.0, 59.5, 1.0, 60.5], 1000.0, &geo).unwrap();
        assert!((g60 / g - 0.5).abs() < 0.01, "{g60} vs {g}");
        // A projected CRS is already metres.
        let l93 = Crs::from_epsg(2154).unwrap();
        assert_eq!(view_gsd([0.0, 0.0, 10_000.0, 10_000.0], 1000.0, &l93), Some(10.0));
        // Degenerate viewports have no scale.
        assert_eq!(view_gsd([1.0, 0.0, 1.0, 1.0], 1000.0, &l93), None);
        assert_eq!(view_gsd([0.0, 0.0, 1.0, 1.0], 0.0, &l93), None);
    }
}

/// A synthetic H3 pyramid on disk, and the reader's trip back through
/// it: detection, level choice, the exact parts a viewport opens, the
/// adaptive band, and C9.
#[cfg(test)]
pub(crate) mod pyramid_tests {
    use super::*;
    use crate::data::pyramid::{
        part_path, Descriptor, FileMeta, Leaf, Level, Method, PyramidState, DESCRIPTOR, NULL_PART,
        VERSION,
    };
    use arrow::array::{ArrayRef, BinaryArray, Float64Array, Int64Array, StructArray};
    use arrow::datatypes::{Field, Fields, Schema};
    use h3o::{CellIndex, LatLng, Resolution};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::collections::HashSet;
    use std::fs::File;

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// One part file: four points around the cell's centre (well inside
    /// it at every resolution used here), a covering column, and — for
    /// an overview — the `geopq:pyramid` entry the layout asks for.
    fn write_cell(path: &std::path::Path, cell: Option<CellIndex>, meta: Option<FileMeta>) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let pts: Vec<(f64, f64)> = match cell {
            Some(c) => {
                let ll = LatLng::from(c);
                [(0.0, 0.0), (5e-5, 0.0), (0.0, 5e-5), (-5e-5, -5e-5)]
                    .iter()
                    .map(|(dx, dy)| (ll.lng() + dx, ll.lat() + dy))
                    .collect()
            }
            // The null part: one row, no shape, no extent.
            None => Vec::new(),
        };
        let mut col = serde_json::json!({
            "encoding": "WKB",
            "geometry_types": ["Point"],
            "covering": {"bbox": {
                "xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
                "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"],
            }},
        });
        if !pts.is_empty() {
            let (xs, ys): (Vec<f64>, Vec<f64>) = pts.iter().copied().unzip();
            col["bbox"] = serde_json::json!([
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                ys.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ]);
        }
        let geo = serde_json::json!({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": col},
        });
        let bbox_fields: Fields = vec![
            Field::new("xmin", DataType::Float64, true),
            Field::new("ymin", DataType::Float64, true),
            Field::new("xmax", DataType::Float64, true),
            Field::new("ymax", DataType::Float64, true),
        ]
        .into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, true),
            Field::new("id", DataType::Int64, false),
            Field::new("bbox", DataType::Struct(bbox_fields.clone()), true),
        ]));
        let (geoms, xs, ys): (Vec<Option<Vec<u8>>>, Vec<Option<f64>>, Vec<Option<f64>>) =
            if pts.is_empty() {
                (vec![None], vec![None], vec![None])
            } else {
                pts.iter()
                    .map(|(x, y)| (Some(wkb_point(*x, *y)), Some(*x), Some(*y)))
                    .collect()
            };
        let rows = geoms.len();
        let coord = |v: &[Option<f64>]| Arc::new(Float64Array::from(v.to_vec())) as ArrayRef;
        let bbox = StructArray::try_new(
            bbox_fields,
            vec![coord(&xs), coord(&ys), coord(&xs), coord(&ys)],
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter(geoms.iter().map(Option::as_deref))),
                Arc::new(Int64Array::from((0..rows as i64).collect::<Vec<_>>())),
                Arc::new(bbox),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(128)).build();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        if let Some(m) = meta {
            w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
                crate::data::pyramid::FILE_KEY.to_string(),
                serde_json::to_string(&m).unwrap(),
            ));
        }
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    pub(crate) struct Fixture {
        pub root: std::path::PathBuf,
        pub r5: Vec<CellIndex>,
        pub r6: Vec<CellIndex>,
        pub r7: Vec<CellIndex>,
        /// The one adaptive child, and the r7 cell it replaced (which
        /// therefore has no file of its own).
        pub r8: CellIndex,
        pub split: CellIndex,
    }

    /// r5: 2 dissolve cells; r6: 6 dissolve cells; r7: 20 leaf cells;
    /// r8: one adaptive child of a 21st r7 cell; plus the null part.
    pub(crate) fn fixture(name: &str) -> Fixture {
        let root = std::env::temp_dir().join(format!("geopq_pyr_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let a = LatLng::new(45.5, -73.6).unwrap().to_cell(Resolution::Five);
        let b = a.grid_disk::<Vec<_>>(1)[1];
        let r5 = vec![a, b];
        let r6: Vec<CellIndex> = a.children(Resolution::Six).take(6).collect();
        let mut r7all: Vec<CellIndex> =
            r6.iter().flat_map(|c| c.children(Resolution::Seven)).collect();
        r7all.sort_unstable();
        r7all.dedup();
        let r7: Vec<CellIndex> = r7all[..20].to_vec();
        let split = r7all[20];
        let r8 = split.children(Resolution::Eight).next().unwrap();

        let meta = |res: u8| {
            Some(FileMeta { res, method: Method::Dissolve, source_res: res + 1, derived: true })
        };
        for c in &r5 {
            write_cell(&root.join(part_path(5, &c.to_string())), Some(*c), meta(5));
        }
        for c in &r6 {
            write_cell(&root.join(part_path(6, &c.to_string())), Some(*c), meta(6));
        }
        for c in &r7 {
            write_cell(&root.join(part_path(7, &c.to_string())), Some(*c), None);
        }
        write_cell(&root.join(part_path(8, &r8.to_string())), Some(r8), None);
        write_cell(&root.join(part_path(7, NULL_PART)), None, None);
        let f = Fixture { root, r5, r6, r7, r8, split };
        std::fs::write(f.root.join(DESCRIPTOR), f.descriptor().to_json()).unwrap();
        f
    }

    impl Fixture {
        pub fn descriptor(&self) -> Descriptor {
            let ids = |cells: &[CellIndex]| cells.iter().map(CellIndex::to_string).collect();
            Descriptor {
                version: VERSION.into(),
                leaf: Leaf { res: 7, adaptive_max_res: 8, target_rows: 100, null_part: true },
                levels: vec![
                    Level { res: 5, method: Some(Method::Dissolve), cells: ids(&self.r5), rows: None },
                    Level { res: 6, method: Some(Method::Dissolve), cells: ids(&self.r6), rows: None },
                    Level { res: 7, method: None, cells: ids(&self.r7), rows: None },
                    Level { res: 8, method: None, cells: ids(&[self.r8]), rows: None },
                ],
                pixels_per_cell: crate::data::pyramid::DEFAULT_PIXELS_PER_CELL,
                crs: serde_json::Value::Null,
                bbox: None,
                rows: None,
                methods: serde_json::Value::Null,
            }
        }

        pub fn source(&self) -> Source {
            Source::Dir(self.root.clone())
        }
    }

    /// A square viewport centred on a cell, `gsd` metres per pixel wide.
    fn view_at(cell: CellIndex, gsd: f64) -> ViewHint {
        let ll = LatLng::from(cell);
        let view_px = 1600.0;
        let half = gsd * view_px / (111_320.0 * ll.lat().to_radians().cos()) / 2.0;
        ViewHint {
            rect: [ll.lng() - half, ll.lat() - half, ll.lng() + half, ll.lat() + half],
            view_px,
        }
    }

    /// Cell ids of the parts a store opened, from the virtual `h3`
    /// column rather than from any path parsing.
    fn opened_cells(store: &FeatureStore) -> HashSet<String> {
        assert_eq!(store.part_cols, vec![crate::data::pyramid::CELL_COLUMN.to_string()]);
        (0..store.rg_starts().len() - 1)
            .filter_map(|g| store.part_value(g, 0).map(str::to_string))
            .collect()
    }

    fn check<'a>(info: &'a FileInfo, code: &str) -> &'a crate::data::quality::Check {
        info.quality
            .as_ref()
            .expect("a pyramid store went through parquet footers")
            .checks
            .iter()
            .find(|c| c.code == code)
            .expect("check present")
    }

    /// The descriptor is found, the ground scale picks the level, and
    /// the info panel gets its line.
    #[test]
    fn the_ground_scale_picks_the_pyramid_level() {
        let f = fixture("level");
        let src = f.source();
        // r7 edge ~1.2 km at 64 px per cell is ~19 m/px; r6 ~50; r5 ~133.
        for (gsd, res, badge) in [
            (0.02, 7u8, None),
            (40.0, 6, Some("overview r6 (dissolve)")),
            (100.0, 5, Some("overview r5 (dissolve)")),
        ] {
            let (store, _crs, info, _) =
                open_store_with_view(&src, Some(view_at(f.r5[0], gsd))).unwrap();
            let p = store.pyramid.as_ref().expect("pyramid detected");
            assert_eq!(p.active_res, res, "at {gsd} m/px");
            assert_eq!(p.badge().as_deref(), badge, "at {gsd} m/px");
            assert_eq!(
                info.pyramid.as_deref(),
                Some("leaf r7 (adaptive to r8), overviews r5..r6 (dissolve), 64 px/cell")
            );
            assert_eq!(check(&info, "C9").status, crate::data::quality::Status::Pass);
        }
    }

    /// A viewport inside one cell opens that cell and the ring around
    /// it — the cells the descriptor lists there, and no others.
    #[test]
    fn a_viewport_opens_exactly_the_cells_it_covers() {
        let f = fixture("exact");
        let listed: HashSet<CellIndex> = f.r7.iter().copied().collect();
        let centre = f.r7[0];
        let (store, ..) = open_store_with_view(&f.source(), Some(view_at(centre, 0.02))).unwrap();
        let want: HashSet<String> = centre
            .grid_disk::<Vec<_>>(1)
            .into_iter()
            .filter(|c| listed.contains(c))
            .map(|c| c.to_string())
            .collect();
        assert!(want.len() > 1, "the fixture must have neighbours to leave out");
        assert_eq!(opened_cells(&store), want);
        assert!(
            store.fragments.len() < f.r7.len(),
            "the whole level must not be opened for one cell"
        );
    }

    /// The writer replaced a dense cell by children at a finer
    /// resolution: a viewport over that cell must find them.
    #[test]
    fn an_adaptive_child_arrives_with_its_parent_cell() {
        let f = fixture("adaptive");
        let (store, ..) = open_store_with_view(&f.source(), Some(view_at(f.split, 0.02))).unwrap();
        let cells = opened_cells(&store);
        assert!(cells.contains(&f.r8.to_string()), "the r8 child of the split cell: {cells:?}");
        assert!(!cells.contains(&f.split.to_string()), "the split cell has no file");
    }

    #[test]
    fn a_viewport_outside_every_cell_opens_nothing() {
        let f = fixture("empty");
        let far = ViewHint { rect: [10.0, 10.0, 10.01, 10.01], view_px: 1600.0 };
        let Err(err) = open_store_with_view(&f.source(), Some(far)) else {
            panic!("a view outside every cell has nothing to open");
        };
        assert!(err.contains("no cells of this pyramid"), "{err}");
    }

    /// C9 warns when the descriptor promises a file the root does not
    /// hold — and the layer still opens, from the cells that are there.
    #[test]
    fn c9_warns_about_a_missing_cell_file() {
        let f = fixture("missing");
        let gone = f.root.join(part_path(7, &f.r7[3].to_string()));
        std::fs::remove_file(&gone).unwrap();
        let (store, _crs, info, _) =
            open_store_with_view(&f.source(), Some(view_at(f.r5[0], 0.02))).unwrap();
        assert!(store.pyramid.is_some(), "a gap does not stop the pyramid opening");
        let c9 = check(&info, "C9");
        assert_eq!(c9.status, crate::data::quality::Status::Warn);
        assert!(c9.detail.contains("1 of 30 listed files are missing"), "{}", c9.detail);
        assert!(!c9.gating);
    }

    /// A descriptor that does not validate is not fatal: the tree under
    /// it is still parquet, so it opens as a plain partitioned dataset
    /// with every file in it, and C9 says what went wrong.
    #[test]
    fn a_bad_descriptor_falls_back_to_the_plain_open() {
        let f = fixture("bad");
        // An r6 cell listed at r5: structurally impossible.
        let mut d = f.descriptor();
        d.levels[0].cells = vec![f.r6[0].to_string()];
        std::fs::write(f.root.join(DESCRIPTOR), d.to_json()).unwrap();
        let (store, _crs, info, _) =
            open_store_with_view(&f.source(), Some(view_at(f.r7[0], 0.02))).unwrap();
        assert!(store.pyramid.is_none(), "no pyramid state from a descriptor that does not hold");
        // Every part file the plain dataset walk sees, which is every
        // one but the null part: `__HIVE_DEFAULT_PARTITION__.parquet`
        // reads as a sidecar name there, exactly as it did before this
        // work. The pyramid path knows better, and lists it.
        assert_eq!(store.fragments.len(), 2 + 6 + 20 + 1);
        let c9 = check(&info, "C9");
        assert_eq!(c9.status, crate::data::quality::Status::Warn);
        assert!(c9.detail.contains("plain partitioned dataset"), "{}", c9.detail);
    }

    /// No descriptor at all: C9 passes, saying a pyramid is optional.
    #[test]
    fn no_descriptor_is_not_a_fault() {
        let f = fixture("absent");
        std::fs::remove_file(f.root.join(DESCRIPTOR)).unwrap();
        let (store, _crs, info, _) = open_store_with_view(&f.source(), None).unwrap();
        assert!(store.pyramid.is_none());
        let c9 = check(&info, "C9");
        assert_eq!(c9.status, crate::data::quality::Status::Pass);
        assert_eq!(c9.detail, "no pyramid (optional)");
    }

    /// One overview file opened on its own still says it is derived:
    /// the `geopq:pyramid` entry is the only thing that can say so when
    /// the descriptor is not in the picture.
    #[test]
    fn a_lone_overview_file_says_it_is_derived() {
        let f = fixture("lone");
        let path = f.root.join(part_path(6, &f.r6[0].to_string()));
        let (_store, _crs, info, _) = open_store(&Source::Local(path)).unwrap();
        let meta = info.pyramid_file.expect("the file's own pyramid entry");
        assert_eq!(meta, FileMeta { res: 6, method: Method::Dissolve, source_res: 7, derived: true });
        assert_eq!(
            crate::data::pyramid::layer_badge(None, Some(&meta)).as_deref(),
            Some("overview r6 (dissolve)")
        );
        // A leaf file carries none, and badges nothing.
        let leaf = f.root.join(part_path(7, &f.r7[0].to_string()));
        let (_s, _c, leaf_info, _) = open_store(&Source::Local(leaf)).unwrap();
        assert!(leaf_info.pyramid_file.is_none());
        assert_eq!(crate::data::pyramid::layer_badge(None, None), None);
    }

    /// Opened with no viewport at all (the CLI path), a pyramid reads
    /// its leaf band whole.
    #[test]
    fn no_viewport_reads_the_leaf_band() {
        let f = fixture("noview");
        let (store, ..) = open_store(&f.source()).unwrap();
        let p = store.pyramid.as_ref().unwrap();
        assert_eq!(p.active_res, 7);
        assert_eq!(store.fragments.len(), f.r7.len() + 1, "20 leaf cells and the adaptive child");
        assert!(PyramidState::new(f.descriptor(), "x").is_ok());
    }
}
