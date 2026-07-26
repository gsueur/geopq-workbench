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
use parquet::file::metadata::PageIndexPolicy;
use rayon::prelude::*;
use rstar::RTree;
use serde_json::Value;

use super::crs::{BulkTransformer, Crs, DisplayCrs};
use super::geoarrow::{GeomCol, GeomEncoding};
use super::geometry::{FeatureRef, MeshBuilder};
use super::info::{summarize_geo_meta, ColumnInfo, FileInfo};
use super::layer::{GroupLoad, LayerGeometry, LoadStats, PickItem, RgBboxes, VectorLayer};
use super::source::Source;
use super::store::{CoveringCol, FeatureStore};

const BATCH_SIZE: usize = 64 * 1024;

/// Error string of a user-cancelled load (the app treats it quietly).
pub const CANCELLED: &str = "load cancelled";

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
    /// Exact viewport selection is still too large for a safe refinement.
    /// This is not an error: retry after the camera moves to a tighter view.
    RefineDeferred {
        layer_id: u64,
        at_least_rows: u64,
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
}

impl GroupSel {
    fn group(&self) -> u32 {
        match self {
            GroupSel::All(g) | GroupSel::Rect(g, _) | GroupSel::Ranges(g, _) => *g,
            GroupSel::Preview { group, .. } | GroupSel::ResolvedRect { group, .. } => *group,
        }
    }
}

/// Row budget for one build: selections above it decode a decimated
/// preview instead (every Nth row), refined with real rows on zoom-in.
pub const MAX_BUILD_ROWS: u64 = 2_500_000;
/// Preview decimation targets roughly this many features.
const PREVIEW_TARGET_ROWS: u64 = 1_200_000;

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
        // STAC part pruning happens at open, before any CRS is known; item
        // bboxes are WGS84 lon/lat by spec.
        let stac_rect = matches!(source, Source::Stac { .. })
            .then(|| viewport_to_data_bbox(view_world, &display, &Crs::wgs84()))
            .flatten();
        match open_store_with_view(&source, stac_rect) {
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
        )
    };
    let build_t0 = Instant::now();
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
                    let avg_overlap = bbox_overlap_metric(&boxes);
                    RgBboxes {
                        source,
                        boxes,
                        avg_overlap,
                    }
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
            let layer = VectorLayer {
                id: layer_id,
                generation: 0,
                name,
                store,
                crs,
                sections: vec![geometry],
                style: super::layer::LayerStyle::new(color),
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
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let sel: Vec<GroupSel> = loaded
            .iter()
            .enumerate()
            .filter_map(|(g, st)| match st {
                GroupLoad::None => None,
                GroupLoad::Full => Some(GroupSel::All(g as u32)),
                // Empty ranges: a layer-filter group with no matching rows.
                GroupLoad::Rows { ranges, .. } if ranges.is_empty() => None,
                GroupLoad::Rows { ranges, .. } => {
                    Some(GroupSel::Ranges(g as u32, ranges.clone()))
                }
                GroupLoad::Preview { stride, rect } => Some(GroupSel::Preview {
                    group: g as u32,
                    rect: *rect,
                    stride: *stride,
                }),
            })
            .collect();
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
    refinement_budget: Option<u64>,
) {
    std::thread::spawn(move || {
        let jobs = if let Some(budget) = refinement_budget {
            match prepare_refinement_jobs(&store, jobs, budget, &cancel) {
                Ok(RefinePlan::Ready(jobs)) => jobs,
                Ok(RefinePlan::Deferred(at_least_rows)) => {
                    log::debug!(
                        "{}: exact refinement exceeds {budget} rows (at least {at_least_rows}) — zoom in further",
                        store.source.label()
                    );
                    handle.send(LoadMsg::RefineDeferred {
                        layer_id,
                        at_least_rows,
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
        let style_sel = style.as_ref().and_then(|sb| resolve_style(&store, sb));
        // Stream the append: content appears within ~a second instead of
        // after the whole (up to budget-sized) build.
        let batches = append_batches(&store, jobs);
        let last = batches.len().saturating_sub(1);
        for (bi, batch) in batches.into_iter().enumerate() {
            match build_geometry(
                &store,
                &crs,
                &display,
                None,
                batch,
                Some(&cancel),
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
    Deferred(u64),
}

/// Resolve viewport rects and enforce the refinement budget using the exact
/// number of selected features. The previous area-ratio estimate could stay
/// above the budget indefinitely for clustered or overlapping row groups,
/// leaving the every-Nth-row preview visible even at street-level zooms.
fn prepare_refinement_jobs(
    store: &FeatureStore,
    jobs: Vec<GroupSel>,
    budget: u64,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<RefinePlan, String> {
    let starts = store.rg_starts();
    let mut rows = 0u64;
    let mut resolved = Vec::with_capacity(jobs.len());

    for job in jobs {
        if cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        let group = job.group();
        let group_rows = starts[group as usize + 1] - starts[group as usize];
        let (count, job) = match job {
            GroupSel::Rect(group, rect) => match covering_select(store, group, rect)? {
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
        };
        rows = rows.saturating_add(count);
        if rows > budget {
            return Ok(RefinePlan::Deferred(rows));
        }
        resolved.push(job);
    }
    Ok(RefinePlan::Ready(resolved))
}

/// Parse file metadata: geometry column, CRS, row-group layout. Reads no data.
type StoreOpen = (
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
}

/// Quality analysis over an opened file / merged dataset (footer facts
/// only). `boxes` are the merged per-row-group bboxes.
fn quality_report(
    info: &FileInfo,
    boxes: Option<&(String, Vec<[f64; 4]>)>,
    encoding: GeomEncoding,
    xy_synthesized: bool,
    page_index: bool,
    geom_bytes: u64,
) -> super::quality::QualityReport {
    super::quality::analyze(&super::quality::QualityInput {
        rows: info.rows,
        row_groups: info.row_groups,
        rg_rows_max: info.rg_rows_max,
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
) -> Vec<GroupSel> {
    let n_rg = store.rg_starts().len().saturating_sub(1);
    let groups: Vec<u32> = match (boxes, rect) {
        (Some(b), Some(r)) => intersecting_rgs(b, r),
        _ => (0..n_rg as u32).collect(),
    };
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
    if est > MAX_BUILD_ROWS {
        let stride = est.div_ceil(PREVIEW_TARGET_ROWS).max(2) as u32;
        log::info!("{label}: {est} candidate rows exceed the budget — preview at 1/{stride}");
        sel = sel
            .into_iter()
            .map(|s| match s {
                GroupSel::All(g) => GroupSel::Preview { group: g, rect: None, stride },
                GroupSel::Rect(g, r) => GroupSel::Preview { group: g, rect: Some(r), stride },
                other => other,
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
    cancel: Arc<std::sync::atomic::AtomicBool>,
    style: Option<crate::data::layer::StyleBy>,
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let n_rg = store.rg_starts().len().saturating_sub(1);
        let rect = viewport_to_data_bbox(view_world, &display, &crs);
        let sel =
            plan_viewport_selection(&store, &store.source.label(), boxes.as_deref(), rect);
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

/// Un-gated open with no viewport, used by the test helpers below.
#[cfg(test)]
fn open_store(source: &Source) -> Result<StoreOpen, String> {
    open_store_with_view(source, None)
}

/// `stac_rect`: current viewport in WGS84 lon/lat, for part-level pruning
/// of STAC collections (their item bboxes are WGS84 by spec).
fn open_store_with_view(
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
    f.info.quality = Some(quality_report(
        &f.info,
        f.rg_boxes.as_ref(),
        f.encoding,
        f.xy.is_some(),
        f.page_index,
        f.geom_bytes,
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

/// Cap on part files a STAC load opens: each part costs a content-length
/// probe plus a footer fetch, so a world view of a 512-part collection
/// must zoom in rather than stream half a terabyte of metadata.
pub const STAC_PART_CAP: usize = 16;

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

/// Open a STAC type collection as one multi-fragment remote store: fetch
/// the item list, keep the parts whose bbox intersects the viewport, cap.
fn open_stac_store(
    source: &Source,
    collection_url: &str,
    rect: Option<[f64; 4]>,
) -> Result<StoreOpen, String> {
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
    if keep.len() > STAC_PART_CAP {
        return Err(format!(
            "{} of {total} parts intersect the current view (cap {STAC_PART_CAP} \
             files per load) — zoom in and retry",
            keep.len()
        ));
    }
    if keep.len() < total {
        log::info!(
            "{}: part pruning {total} -> {} files",
            source.label(),
            keep.len()
        );
    }
    // Resolve (length probe) every part in parallel.
    let files: Vec<(Source, String)> = {
        use rayon::prelude::*;
        keep.into_par_iter()
            .map(|p| {
                let short = p
                    .url
                    .rsplit('/')
                    .next()
                    .unwrap_or(p.url.as_str())
                    .to_string();
                Source::Remote { url: p.url, len: 0 }
                    .resolve()
                    .map(|src| (src, short.clone()))
                    .map_err(|e| format!("{short}: {e}"))
            })
            .collect::<Result<_, _>>()?
    };
    let hive = vec![Vec::new(); files.len()];
    open_multi_store(source, files, hive)
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
                (None, None) => boxes = None,
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
    info.quality = Some(quality_report(
        &info,
        rg_boxes.as_ref(),
        first.encoding,
        first.xy.is_some(),
        page_index,
        geom_bytes,
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
            // Not GeoParquet-tagged: guess a WKB column, assume CRS84.
            let guess = ["geometry", "geom", "wkb_geometry", "wkb"]
                .iter()
                .find(|n| schema.index_of(n).is_ok())
                .map(|n| n.to_string())
                .or_else(|| {
                    schema
                        .fields()
                        .iter()
                        .find(|f| {
                            matches!(
                                f.data_type(),
                                DataType::Binary | DataType::LargeBinary | DataType::BinaryView
                            )
                        })
                        .map(|f| f.name().clone())
                });
            match guess {
                Some(name) => (name, Crs::wgs84()),
                None => {
                    // Last resort: coordinate columns → synthesized points.
                    let (xi, yi) = xy_columns(&schema).ok_or(
                        "no 'geo' metadata, no binary geometry column, and no \
                         lon/lat or x/y coordinate columns found",
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
        compressed_bytes,
        uncompressed_bytes,
        columns,
        geo: summarize_geo_meta(geo_meta.as_ref(), &primary_name, &crs.name, has_native_geometry),
        files: 1,
        quality: None,
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

    let mut info = info;
    if hidden_wkb.is_some() {
        info.geo.encoding = format!("{} + GeoArrow column (used for display)", info.geo.encoding);
    }
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
fn rg_bboxes_from_metadata(
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
        StyleMode::Categorical { values } => Binning::Categorical {
            map: values
                .iter()
                .take(STYLE_BINS - 1)
                .enumerate()
                .map(|(i, v)| (v.clone(), i as u8))
                .collect(),
        },
    };
    Some(StyleSel { col, binning })
}

/// Sample up to `cap` values of a column from the already-loaded rows of
/// a layer (classification must never fetch the whole dataset). Blocking —
/// run off the UI thread.
pub fn sample_loaded_values(
    store: &FeatureStore,
    loaded: &[GroupLoad],
    col: usize,
    cap: usize,
) -> Result<Vec<f64>, String> {
    let starts = store.rg_starts();
    // Rect-filtered previews: reproduce the load's exact selection (same
    // covering scan, same decimation) so sampling never fetches rows that
    // were never loaded.
    let mut preview_rows: std::collections::HashMap<usize, Vec<u32>> = Default::default();
    for (g, st) in loaded.iter().enumerate() {
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
        .map(|(g, st)| match st {
            GroupLoad::Full => starts[g + 1] - starts[g],
            GroupLoad::Rows { ranges, .. } => {
                ranges.iter().map(|&(s, e)| (e - s) as u64).sum()
            }
            GroupLoad::Preview { rect: Some(_), .. } => preview_rows[&g].len() as u64,
            GroupLoad::Preview { stride, rect: None } => {
                (starts[g + 1] - starts[g]).div_ceil(*stride as u64)
            }
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
            GroupLoad::None => {}
        }
    }
    rows.sort_unstable();
    rows.dedup();
    if rows.is_empty() {
        return Err("no rows loaded yet".into());
    }
    let batches = store.fetch(&rows, Some(&[col]))?;
    let mut out = Vec::with_capacity(rows.len());
    for b in &batches {
        let vals = arrow::compute::cast(b.column(0), &DataType::Float64)
            .map_err(|e| format!("value cast: {e}"))?;
        let vals = vals
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        for i in 0..vals.len() {
            if !arrow::array::Array::is_null(vals, i) && vals.value(i).is_finite() {
                out.push(vals.value(i));
            }
        }
    }
    Ok(out)
}

/// Per-row style bins for one batch's value column.
fn batch_bins(arr: &arrow::array::ArrayRef, binning: &Binning) -> Vec<u8> {
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
                    Some(v) if !v.is_null(i) => {
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
    let rg_starts = store.rg_starts();
    let proj_ref: &[usize] = &proj;
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
    let per_group = move |job: GroupSel| -> Vec<(RowMap, RecordBatch)> {
        if cancelled() {
            return Vec::new();
        }
        let g = job.group();
        let start = starts[g as usize];
        let group_rows = (starts[g as usize + 1] - start) as u32;
        // Resolve the job to optional group-relative ranges + final state.
        let (ranges, state): (Option<Vec<(u32, u32)>>, GroupLoad) = match &job {
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
        };
        resolved_ref.lock().unwrap().push((g, state));
        // Global rows of the selection, for sparse batches.
        let sparse: Option<Arc<Vec<u32>>> = ranges.as_ref().map(|rs| {
            Arc::new(
                rs.iter()
                    .flat_map(|&(s, e)| (start as u32 + s)..(start as u32 + e))
                    .collect::<Vec<u32>>(),
            )
        });
        let empty = sparse.as_ref().is_some_and(|s| s.is_empty());
        let reader = if empty {
            None
        } else {
            match store.reader_for_group(
                g as usize,
                BATCH_SIZE,
                ranges.as_deref(),
                Some(proj_ref),
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    *err_ref.lock().unwrap() = Some(e);
                    None
                }
            }
        };
        let mut out = Vec::new();
        let mut consumed = 0usize;
        for res in reader.into_iter().flatten() {
            if cancelled() {
                return out;
            }
            match res {
                Ok(batch) => {
                    let map = match &sparse {
                        None => RowMap::Contiguous(start + consumed as u64),
                        Some(rows) => RowMap::Sparse(rows.clone(), consumed),
                    };
                    consumed += batch.num_rows();
                    out.push((map, batch));
                }
                Err(e) => {
                    *err_ref.lock().unwrap() = Some(e.to_string());
                    break;
                }
            }
        }
        out
    };

    let (builder, items, rows, bad, rg_boxes) = sel
        .into_par_iter()
        .flat_map(per_group)
        .map(|(map, batch)| {
            let mut mb = MeshBuilder::default();
            let mut items: Vec<PickItem> = Vec::new();
            let mut bad = 0usize;
            let mut rg_boxes: std::collections::HashMap<u32, [f64; 4]> = Default::default();
            let tr = BulkTransformer::new(crs, display);
            let rows = process_batch(
                &batch, &map, encoding, &tr, display, &mut mb, &mut items, &mut bad,
                rg_starts, &mut rg_boxes, geom_pos,
                style.map(|st| (style_pos.unwrap(), &st.binning)),
                spherical,
            );
            if let Some((handle, job)) = progress {
                let d = done.fetch_add(rows, Ordering::Relaxed) + rows;
                // The parallel decode+tessellate pass is ~70% of a load;
                // chunking and the pick index follow single-threaded.
                handle.send(LoadMsg::Progress {
                    job,
                    frac: 0.70 * (d as f32 / total as f32).min(1.0),
                    stage: "decoding & tessellating".into(),
                });
            }
            (mb, items, rows, bad, rg_boxes)
        })
        .reduce(
            || {
                (
                    MeshBuilder::default(),
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
    style: Option<(usize, &Binning)>,
    spherical: bool,
) -> usize {
    let col = batch.column(geom_pos);
    let Some(get) = GeomCol::new(col.as_ref(), encoding) else {
        *bad += batch.num_rows();
        return batch.num_rows();
    };
    // Data-driven styling: per-row bin from the value column; the mesh
    // builder keys chunks by (cell, bin).
    let bins: Option<Vec<u8>> =
        style.map(|(pos, binning)| batch_bins(batch.column(pos), binning));

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
            bins.as_deref(),
        );
    }

    for row in 0..batch.num_rows() {
        if get.is_null(row) {
            continue;
        }
        if let Some(b) = &bins {
            mb.bin = b[row];
        }
        let global = map.global(row);
        let fref = FeatureRef {
            index: global as u32,
        };

        // Fast path: 2D point (WKB parse or GeoArrow coordinate read), no
        // per-feature geo allocation.
        if let Some((x, y)) = get.point2(row) {
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

    /// End-to-end pruning against the Hilbert-sorted covering fixture:
    /// covering stats must be detected, and a small-viewport load must
    /// decode a small fraction of the row groups and rows.
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
        let sel = plan_viewport_selection(&store, "t", Some(&boxes), Some(rect));
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
        );
        assert!(matches!(sel.as_slice(), [GroupSel::Rect(1, _)]), "{sel:?}");
        // Disjoint rect: nothing to read.
        let sel =
            plan_viewport_selection(&store, "t", Some(&boxes), Some([50.0, 0.0, 60.0, 1.0]));
        assert!(sel.is_empty(), "{sel:?}");
        // No rect: everything.
        let sel = plan_viewport_selection(&store, "t", Some(&boxes), None);
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
            open_store_with_view(&source, Some([-12.0, 43.0, -8.0, 47.0])).unwrap();
        assert_eq!(store.fragments.len(), 1);
        assert_eq!(store.total_rows(), 200);
        assert_eq!(info.files, 1);

        // A viewport intersecting nothing is a load error, not an empty map.
        let Err(err) = open_store_with_view(&source, Some([100.0, 0.0, 110.0, 5.0])) else {
            panic!("disjoint viewport must not open a store");
        };
        assert!(err.contains("no parts"), "{err}");
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
            open_store_with_view(&source, Some([2.0, 48.5, 3.0, 49.2])).unwrap();
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
            RefinePlan::Deferred(_) => panic!("selection unexpectedly deferred"),
        }
        assert!(matches!(
            prepare_refinement_jobs(
                &store,
                vec![GroupSel::Rect(0, rect)],
                in_rect.len() as u64 - 1,
                &cancel,
            )
            .unwrap(),
            RefinePlan::Deferred(_)
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
            sample_loaded_values(&store, &loaded, id_col, 10_000).unwrap();
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let got: Vec<u32> = vals.iter().map(|v| *v as u32).collect();
        assert_eq!(got, expected, "sampling must reproduce the preview selection");
    }
}
