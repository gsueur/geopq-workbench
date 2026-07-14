use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use geo::MapCoordsInPlace;
use geo_traits::to_geo::ToGeoGeometry;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rstar::RTree;
use serde_json::Value;

use super::crs::{BulkTransformer, Crs, DisplayCrs};
use super::geometry::{FeatureRef, MeshBuilder};
use super::info::{summarize_geo_meta, ColumnInfo, FileInfo};
use super::layer::{GroupLoad, LayerGeometry, LoadStats, PickItem, RgBboxes, VectorLayer};
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
        path: PathBuf,
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
fn covering_select(
    path: &PathBuf,
    covering: Option<&CoveringCol>,
    group: u32,
    rect: [f64; 4],
) -> Result<Option<Vec<(u32, u32)>>, String> {
    use arrow::array::{Array, Float64Array, StructArray};
    let Some(cov) = covering else {
        return Ok(None);
    };
    let reader = FeatureStore::open_reader_for_group(
        path,
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
    path: PathBuf,
    display: DisplayCrs,
    color: eframe::egui::Color32,
    view_world: [f64; 4],
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        match open_store(&path) {
            Ok((store, crs, info, rg_meta)) => {
                let store = Arc::new(store);
                let n_rg = store.rg_starts().len().saturating_sub(1);
                let rect = viewport_to_data_bbox(view_world, &display, &crs);
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
                        path.display(),
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
                        let name = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "layer".into());
                        log::info!(
                            "loaded {name}: {rows} features in {} ms",
                            t0.elapsed().as_millis()
                        );
                        let layer = VectorLayer {
                            id: layer_id,
                            generation: 0,
                            name,
                            path,
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
                        });
                    }
                    Err(e) => handle.send(LoadMsg::Failed {
                        job,
                        path,
                        error: e,
                    }),
                }
            }
            Err(e) => handle.send(LoadMsg::Failed {
                job,
                path,
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
                path: store.path.clone(),
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
                path: store.path.clone(),
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

fn open_store(path: &PathBuf) -> Result<StoreOpen, String> {
    let file = File::open(path).map_err(|e| format!("cannot open file: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("not a parquet file: {e}"))?;

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

    let (geom_name, crs) = match &geo_meta {
        Some(meta) => {
            let primary = meta
                .get("primary_column")
                .and_then(Value::as_str)
                .unwrap_or("geometry")
                .to_string();
            let col_meta = meta.get("columns").and_then(|c| c.get(&primary));
            if let Some(cm) = col_meta {
                let encoding = cm.get("encoding").and_then(Value::as_str).unwrap_or("WKB");
                if !encoding.eq_ignore_ascii_case("wkb") {
                    return Err(format!(
                        "geometry encoding '{encoding}' not supported yet (only WKB)"
                    ));
                }
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
    let mut has_native_geometry = false;
    let columns: Vec<ColumnInfo> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let logical = pq_columns.get(i).and_then(|c| {
                c.logical_type_ref().map(|lt| format!("{lt:?}"))
            });
            if let Some(l) = &logical {
                if l.starts_with("Geometry") || l.starts_with("Geography") {
                    has_native_geometry = true;
                }
            }
            let compression = meta
                .row_groups()
                .first()
                .and_then(|rg| rg.columns().get(i))
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
        file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
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

    let rg_meta_boxes = rg_bboxes_from_metadata(&builder, geo_meta.as_ref(), geom_col, &geom_name);
    let covering = covering_column(geo_meta.as_ref(), &geom_name, &schema);

    Ok((
        FeatureStore::new(path.clone(), geom_col, schema, covering, rg_rows),
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
    builder: &ParquetRecordBatchReaderBuilder<File>,
    geo_meta: Option<&Value>,
    geom_col: usize,
    primary: &str,
) -> Option<(String, Vec<[f64; 4]>)> {
    let meta = builder.metadata();

    // 1. Native geospatial statistics on the geometry column chunks.
    let native: Option<Vec<[f64; 4]>> = meta
        .row_groups()
        .iter()
        .map(|rg| {
            let stats = rg.columns().get(geom_col)?.geo_statistics()?;
            let b = stats.bounding_box()?;
            Some([b.get_xmin(), b.get_ymin(), b.get_xmax(), b.get_ymax()])
        })
        .collect();
    if let Some(boxes) = native {
        if !boxes.is_empty() {
            return Some(("parquet geospatial statistics".into(), boxes));
        }
    }

    // 2. GeoParquet 1.1 covering bbox columns: column-chunk min/max stats.
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
    let boxes: Option<Vec<[f64; 4]>> = meta
        .row_groups()
        .iter()
        .map(|rg| {
            Some([
                stat_f64(rg, xmin_i, false)?,
                stat_f64(rg, ymin_i, false)?,
                stat_f64(rg, xmax_i, true)?,
                stat_f64(rg, ymax_i, true)?,
            ])
        })
        .collect();
    boxes
        .filter(|b| !b.is_empty())
        .map(|b| ("covering column statistics (GeoParquet 1.1)".into(), b))
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

    let path = store.path.clone();
    let covering = store.covering.clone();
    let err_ref = &stream_error;
    let resolved_ref = &resolved;
    let starts = rg_starts;
    // Batches from a multi-group reader can span groups, so read one group
    // at a time; batches still fan out to all cores via par_bridge.
    let stream = sel.into_iter().flat_map(move |job| {
        let g = job.group();
        let start = starts[g as usize];
        let group_rows = (starts[g as usize + 1] - start) as u32;
        // Resolve the job to optional group-relative ranges + final state.
        let (ranges, state): (Option<Vec<(u32, u32)>>, GroupLoad) = match &job {
            GroupSel::All(_) => (None, GroupLoad::Full),
            GroupSel::Ranges(_, r) => (Some(r.clone()), GroupLoad::Full),
            GroupSel::Rect(_, rect) => match covering_select(&path, covering.as_ref(), g, *rect) {
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
            },
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
                &path,
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
        reader
            .into_iter()
            .flatten()
            .scan(0usize, move |consumed, res| match res {
                Ok(batch) => {
                    let map = match &sparse {
                        None => RowMap::Contiguous(start + *consumed as u64),
                        Some(rows) => RowMap::Sparse(rows.clone(), *consumed),
                    };
                    *consumed += batch.num_rows();
                    Some((map, batch))
                }
                Err(e) => {
                    *err_ref.lock().unwrap() = Some(e.to_string());
                    None
                }
            })
    });

    let (builder, items, rows, bad, rg_boxes) = stream
        .par_bridge()
        .map(|(map, batch)| {
            let mut mb = MeshBuilder::default();
            let mut items: Vec<PickItem> = Vec::new();
            let mut bad = 0usize;
            let mut rg_boxes: std::collections::HashMap<u32, [f64; 4]> = Default::default();
            let tr = BulkTransformer::new(crs, display);
            let rows = process_batch(
                &batch, &map, &tr, display, &mut mb, &mut items, &mut bad, rg_starts,
                &mut rg_boxes,
            );
            if let Some((handle, job)) = progress {
                let d = done.fetch_add(rows, Ordering::Relaxed) + rows;
                handle.send(LoadMsg::Progress {
                    job,
                    frac: (d as f32 / total as f32).min(1.0),
                    stage: "building geometry".into(),
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
            store.path.display(),
            builder.fill_errors
        );
    }
    let bad = bad + builder.fill_errors;
    let chunks = builder.finish();
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
    let Some(get) = BinCol::new(col.as_ref()) else {
        *bad += batch.num_rows();
        return batch.num_rows();
    };

    for row in 0..batch.num_rows() {
        let Some(buf) = get.value(row) else {
            continue;
        };
        let global = map.global(row);
        let fref = FeatureRef {
            index: global as u32,
        };

        // Fast path: 2D WKB point, no per-feature geo allocation.
        if let Some((x, y)) = parse_wkb_point_2d(buf) {
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

        match decode_wkb(buf) {
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
pub fn open_store_for_test(
    path: &PathBuf,
) -> Result<StoreOpen, String> {
    open_store(path)
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
        let (store, crs, _info, rg_meta) = open_store(&path).unwrap();
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
        let (store, _crs, _info, _rg) = open_store(&path.clone().into()).unwrap();
        let total = store.total_rows() as u32;
        let step = (total / 20_000).max(1);
        let rows: Vec<u32> = (0..total).step_by(step as usize).collect();
        let wkbs = store.fetch_wkb(&rows).unwrap();

        let mut checked = 0usize;
        let mut mismatches: Vec<(u32, f64, f64)> = Vec::new();
        for (row, wkb) in wkbs {
            let Some(wkb) = wkb else { continue };
            let Some(geom) = decode_wkb(&wkb) else { continue };
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
        let (store, crs, _info, rg_meta) = open_store(&path).unwrap();
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
        let wkbs = store.fetch_wkb(&picked).unwrap();
        for (row, wkb) in wkbs {
            use geo::BoundingRect;
            let geom = decode_wkb(&wkb.expect("non-null")).unwrap();
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
