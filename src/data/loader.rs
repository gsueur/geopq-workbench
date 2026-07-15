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
    /// Additional rows loaded for an existing layer (new section).
    Appended {
        layer_id: u64,
        generation: u64,
        geometry: LayerGeometry,
        rows: usize,
        /// New decode state per touched row group.
        loaded: Vec<(u32, GroupLoad)>,
    },
    Failed {
        job: u64,
        source: String,
        error: String,
    },
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

/// Union of a set of bboxes.
fn union_of(boxes: &[[f64; 4]]) -> Option<[f64; 4]> {
    boxes.iter().copied().reduce(|a, b| {
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
    /// Explicit group-relative [start, end) row ranges (rebuilds of
    /// partially loaded groups, complement appends).
    Ranges(u32, Vec<(u32, u32)>),
}

impl GroupSel {
    fn group(&self) -> u32 {
        match self {
            GroupSel::All(g) | GroupSel::Rect(g, _) | GroupSel::Ranges(g, _) => *g,
        }
    }
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
    source: &Source,
    meta: &ArrowReaderMetadata,
    covering: Option<&CoveringCol>,
    group: u32,
    rect: [f64; 4],
) -> Result<Option<Vec<(u32, u32)>>, String> {
    use arrow::array::{Array, Float64Array, StructArray};
    let Some(cov) = covering else {
        return Ok(None);
    };
    let reader = FeatureStore::open_reader_for_group(
        source,
        meta,
        group as usize,
        BATCH_SIZE,
        None,
        Some(&[cov.root]),
    )?;
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

/// Load a GeoParquet file in a background thread. Only geometry-derived data
/// stays in memory; attributes are re-read lazily via `FeatureStore`.
///
/// `view_world`: current viewport in world space. When the file carries
/// per-row-group bboxes in its metadata, row groups outside the viewport
/// are pruned and stream in later on demand.
pub fn spawn_load(
    handle: LoaderHandle,
    job: u64,
    layer_id: u64,
    source: Source,
    display: DisplayCrs,
    color: eframe::egui::Color32,
    view_world: [f64; 4],
    auto_project: bool,
) {
    std::thread::spawn(move || {
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
        match open_store(&source) {
            Ok((store, crs, info, rg_meta)) => {
                let store = Arc::new(store);
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
                    let bbox = rg_meta
                        .as_ref()
                        .filter(|_| crs.is_latlong)
                        .and_then(|(_, boxes)| union_of(boxes));
                    if let Some(d) = DisplayCrs::auto_for(&crs, bbox) {
                        log::info!("auto projection: {}", d.name);
                        display = d.clone();
                        adopt_display = Some((d, true));
                    }
                }
                // Prune only with metadata-sourced boxes: a pruned first load
                // never sees the skipped row groups, so computed boxes can't
                // drive it.
                let groups: Vec<u32> = match (&rg_meta, rect) {
                    (Some((_, boxes)), Some(r)) => intersecting_rgs(boxes, r),
                    _ => (0..n_rg as u32).collect(),
                };
                if groups.len() < n_rg {
                    log::info!(
                        "{}: row-group pruning {} -> {} groups",
                        source.label(),
                        n_rg,
                        groups.len()
                    );
                }
                // Per-feature covering selection: only when the viewport
                // doesn't already cover the whole data extent.
                let use_rect = store.covering.is_some()
                    && match (&rg_meta, rect) {
                        (Some((_, boxes)), Some(r)) => {
                            let mut u = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                            for b in boxes {
                                u = [u[0].min(b[0]), u[1].min(b[1]), u[2].max(b[2]), u[3].max(b[3])];
                            }
                            !(r[0] <= u[0] && r[1] <= u[1] && r[2] >= u[2] && r[3] >= u[3])
                        }
                        _ => false,
                    };
                let sel: Vec<GroupSel> = groups
                    .iter()
                    .map(|&g| {
                        if use_rect {
                            GroupSel::Rect(g, rect.unwrap())
                        } else {
                            GroupSel::All(g)
                        }
                    })
                    .collect();
                let build_t0 = Instant::now();
                match build_geometry(&store, &crs, &display, Some((&handle, job)), sel) {
                    Ok((geometry, rows, bad, rg_computed, resolved)) => {
                        // Extent only known post-build (no metadata bboxes):
                        // suggest the auto projection; the app rebuilds.
                        if auto_project && adopt_display.is_none() && crs.is_latlong {
                            if let Some(d) =
                                DisplayCrs::auto_for(&crs, union_of(&rg_computed))
                            {
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
                                (!rg_computed.is_empty())
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
                        let name = source.name();
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
                        };
                        handle.send(LoadMsg::Loaded {
                            job,
                            layer: Box::new(layer),
                            adopt_display,
                        });
                    }
                    Err(e) => handle.send(LoadMsg::Failed {
                        job,
                        source: source.label(),
                        error: e,
                    }),
                }
            }
            Err(e) => handle.send(LoadMsg::Failed {
                job,
                source: source.label(),
                error: e,
            }),
        }
    });
}

/// Rebuild geometry for an existing layer under a new display projection,
/// re-streaming exactly the loaded rows (consolidates any appended
/// sections into one).
pub fn spawn_rebuild(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    loaded: Vec<GroupLoad>,
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let sel: Vec<GroupSel> = loaded
            .iter()
            .enumerate()
            .filter_map(|(g, st)| match st {
                GroupLoad::None => None,
                GroupLoad::Full => Some(GroupSel::All(g as u32)),
                GroupLoad::Rows { ranges, .. } => {
                    Some(GroupSel::Ranges(g as u32, ranges.clone()))
                }
            })
            .collect();
        match build_geometry(&store, &crs, &display, None, sel) {
            Ok((geometry, _rows, bad, _rg, _resolved)) => handle.send(LoadMsg::Rebuilt {
                layer_id,
                generation,
                geometry,
                stats_build_ms: t0.elapsed().as_millis() as u64,
                bad_geoms: bad,
            }),
            Err(e) => handle.send(LoadMsg::Failed {
                job: u64::MAX,
                source: store.source.label(),
                error: format!("projection rebuild failed: {e}"),
            }),
        }
    });
}

/// Load additional rows for an existing layer (viewport refinement or
/// "Load all"). `GroupSel::Ranges` here must be the complement of what is
/// already loaded — the group is marked `Full` afterwards.
pub fn spawn_append(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    jobs: Vec<GroupSel>,
) {
    std::thread::spawn(move || {
        match build_geometry(&store, &crs, &display, None, jobs) {
            Ok((geometry, rows, _bad, _rg, resolved)) => handle.send(LoadMsg::Appended {
                layer_id,
                generation,
                geometry,
                rows,
                loaded: resolved,
            }),
            Err(e) => handle.send(LoadMsg::Failed {
                job: u64::MAX,
                source: store.source.label(),
                error: format!("row append failed: {e}"),
            }),
        }
    });
}

/// Parse file metadata: geometry column, CRS, row-group layout. Reads no data.
type StoreOpen = (
    FeatureStore,
    Crs,
    FileInfo,
    Option<(String, Vec<[f64; 4]>)>,
);

fn open_store(source: &Source) -> Result<StoreOpen, String> {
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
            }
            let crs = Crs::from_geoparquet_crs(col_meta.and_then(|c| c.get("crs")))?;
            (primary, crs)
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
                })
                .ok_or("no 'geo' metadata and no binary geometry column found")?;
            (guess, Crs::wgs84())
        }
    };

    let geom_col = schema
        .index_of(&geom_name)
        .map_err(|_| format!("geometry column '{geom_name}' not found in schema"))?;

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
            if i == geom_col {
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
        geo: summarize_geo_meta(geo_meta.as_ref(), &geom_name, &crs.name, has_native_geometry),
    };

    let rg_meta_boxes =
        rg_bboxes_from_metadata(&builder, geo_meta.as_ref(), geom_leaf, &geom_name, encoding);
    let covering = covering_column(geo_meta.as_ref(), &geom_name, &schema);

    Ok((
        FeatureStore::new(
            source.clone(),
            arrow_meta,
            geom_col,
            schema,
            covering,
            encoding,
            rg_rows,
        ),
        crs,
        info,
        rg_meta_boxes,
    ))
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
                Some([b.get_xmin(), b.get_ymin(), b.get_xmax(), b.get_ymax()])
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
) -> Result<BuildOutput, String> {
    let geom_col = store.geom_col;
    let rg_starts = store.rg_starts();
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

    let source = store.source.clone();
    let arrow_meta = store.meta.clone();
    let covering = store.covering.clone();
    let encoding = store.encoding;
    let err_ref = &stream_error;
    let resolved_ref = &resolved;
    let starts = rg_starts;
    // One task per group: group reads run concurrently (essential for
    // remote sources, where each group is a series of range requests) and
    // the flattened batches then tessellate in parallel.
    let per_group = move |job: GroupSel| -> Vec<(RowMap, RecordBatch)> {
        let g = job.group();
        let start = starts[g as usize];
        let group_rows = (starts[g as usize + 1] - start) as u32;
        // Resolve the job to optional group-relative ranges + final state.
        let (ranges, state): (Option<Vec<(u32, u32)>>, GroupLoad) = match &job {
            GroupSel::All(_) => (None, GroupLoad::Full),
            GroupSel::Ranges(_, r) => (Some(r.clone()), GroupLoad::Full),
            GroupSel::Rect(_, rect) => {
                match covering_select(&source, &arrow_meta, covering.as_ref(), g, *rect) {
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
            match FeatureStore::open_reader_for_group(
                &source,
                &arrow_meta,
                g as usize,
                BATCH_SIZE,
                ranges.as_deref(),
                Some(&[geom_col]),
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
                rg_starts, &mut rg_boxes,
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

    let mut rg_vec: Vec<[f64; 4]> = Vec::new();
    for i in 0..rg_starts.len().saturating_sub(1) {
        if let Some(b) = rg_boxes.get(&(i as u32)) {
            rg_vec.push(*b);
        }
    }

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
) -> usize {
    // The loader projects the read down to the geometry column.
    let col = batch.column(0);
    let Some(get) = GeomCol::new(col.as_ref(), encoding) else {
        *bad += batch.num_rows();
        return batch.num_rows();
    };

    // GeoArrow: bulk path — one linear reprojection pass over the whole
    // coordinate buffer, features emitted straight from the arrow offsets.
    if let GeomCol::Ga(ga) = &get {
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
        );
    }

    for row in 0..batch.num_rows() {
        if get.is_null(row) {
            continue;
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

/// Decode WKB into geo-types (drops Z/M).
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
    build_geometry(store, crs, display, None, all_groups(store))
}

#[cfg(test)]
pub fn build_geometry_for_test(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
) -> Result<(super::layer::LayerGeometry, usize, usize), String> {
    build_geometry(store, crs, display, None, all_groups(store)).map(|(g, r, b, _, _)| (g, r, b))
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
            build_geometry(&store, &crs, &display, None, all_groups(&store)).unwrap();
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
            build_geometry(&store, &crs, &display, None, jobs).unwrap();
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
            build_geometry(&store, &crs, &display, None, jobs).unwrap();
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
            build_geometry(&store_r, &crs_r, &display, None, jobs(())).unwrap();
        let (_g, rows_l, bad_l, _rg, _res) =
            build_geometry(&store_l, &crs_l, &display, None, jobs(())).unwrap();
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
                build_geometry(&store, &crs, &display, None, all_groups(&store)).unwrap();
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
            build_geometry(&store, &crs, &display, None, jobs).unwrap();
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
