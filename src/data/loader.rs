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
use super::layer::{LayerGeometry, LoadStats, PickItem, RgBboxes, VectorLayer};
use super::store::FeatureStore;

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
    /// Additional row groups loaded for an existing layer (new section).
    Appended {
        layer_id: u64,
        generation: u64,
        geometry: LayerGeometry,
        rows: usize,
        loaded_rgs: Vec<u32>,
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
                // Prune only with metadata-sourced boxes: a pruned first load
                // never sees the skipped row groups, so computed boxes can't
                // drive it.
                let (rg_filter, prunable_boxes): (Option<Vec<u32>>, Option<&Vec<[f64; 4]>>) =
                    match &rg_meta {
                        Some((_, boxes)) => {
                            let sel = viewport_to_data_bbox(view_world, &display, &crs)
                                .map(|rect| intersecting_rgs(boxes, rect));
                            (sel, Some(boxes))
                        }
                        None => (None, None),
                    };
                let rg_filter = rg_filter.filter(|sel| sel.len() < n_rg);
                if let Some(sel) = &rg_filter {
                    log::info!(
                        "{}: row-group pruning {} -> {} groups",
                        path.display(),
                        n_rg,
                        sel.len()
                    );
                }
                let _ = prunable_boxes;
                let build_t0 = Instant::now();
                match build_geometry(
                    &store,
                    &crs,
                    &display,
                    Some((&handle, job)),
                    rg_filter.as_deref(),
                ) {
                    Ok((geometry, rows, bad, rg_computed)) => {
                        let loaded_rgs: Vec<u32> = match &rg_filter {
                            Some(sel) => sel.clone(),
                            None => (0..n_rg as u32).collect(),
                        };
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
                            loaded_rgs,
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
/// re-streaming only the loaded row groups (consolidates any appended
/// sections into one).
pub fn spawn_rebuild(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    loaded_rgs: Vec<u32>,
) {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let n_rg = store.rg_starts().len().saturating_sub(1);
        let filter = (loaded_rgs.len() < n_rg).then_some(loaded_rgs);
        match build_geometry(&store, &crs, &display, None, filter.as_deref()) {
            Ok((geometry, _rows, bad, _rg)) => handle.send(LoadMsg::Rebuilt {
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

/// Load additional row groups for an existing layer (viewport refinement).
pub fn spawn_append(
    handle: LoaderHandle,
    layer_id: u64,
    generation: u64,
    store: Arc<FeatureStore>,
    crs: Crs,
    display: DisplayCrs,
    rgs: Vec<u32>,
) {
    std::thread::spawn(move || {
        match build_geometry(&store, &crs, &display, None, Some(&rgs)) {
            Ok((geometry, rows, _bad, _rg)) => handle.send(LoadMsg::Appended {
                layer_id,
                generation,
                geometry,
                rows,
                loaded_rgs: rgs,
            }),
            Err(e) => handle.send(LoadMsg::Failed {
                job: u64::MAX,
                path: store.path.clone(),
                error: format!("row-group append failed: {e}"),
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

    Ok((
        FeatureStore::new(path.clone(), geom_col, schema, rg_rows),
        crs,
        info,
        rg_meta_boxes,
    ))
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
fn bbox_overlap_metric(boxes: &[[f64; 4]]) -> f64 {
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

type BuildOutput = (LayerGeometry, usize, usize, Vec<[f64; 4]>);

/// Stream the file batch by batch, tessellating in parallel (rayon
/// par_bridge). Peak memory is a handful of in-flight batches, not the file.
/// `rg_filter`: row-group indices to read (None = whole file); global row
/// indices stay correct either way, so picking and attributes line up.
fn build_geometry(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
    progress: Option<(&LoaderHandle, u64)>,
    rg_filter: Option<&[u32]>,
) -> Result<BuildOutput, String> {
    let geom_col = store.geom_col;
    let rg_starts = store.rg_starts();
    let stream_error: Mutex<Option<String>> = Mutex::new(None);

    // (global_start_row, batch) stream over the selected row groups.
    // Batches from a multi-group reader can span groups, so read one group
    // at a time; batches still fan out to all cores via par_bridge.
    let groups: Vec<u32> = match rg_filter {
        Some(f) => f.to_vec(),
        None => (0..rg_starts.len().saturating_sub(1) as u32).collect(),
    };
    let total: u64 = groups
        .iter()
        .map(|&g| rg_starts[g as usize + 1] - rg_starts[g as usize])
        .sum::<u64>()
        .max(1);
    let done = AtomicUsize::new(0);

    let path = store.path.clone();
    let err_ref = &stream_error;
    let starts = rg_starts;
    let stream = groups.into_iter().flat_map(move |g| {
        let start = starts[g as usize];
        let reader = match FeatureStore::open_reader_for_group(&path, g as usize, BATCH_SIZE) {
            Ok(r) => Some(r),
            Err(e) => {
                *err_ref.lock().unwrap() = Some(e);
                None
            }
        };
        reader
            .into_iter()
            .flatten()
            .scan(start, move |pos, res| match res {
                Ok(batch) => {
                    let s = *pos;
                    *pos += batch.num_rows() as u64;
                    Some((s, batch))
                }
                Err(e) => {
                    *err_ref.lock().unwrap() = Some(e.to_string());
                    None
                }
            })
    });

    let (builder, items, rows, bad, rg_boxes) = stream
        .par_bridge()
        .map(|(start_row, batch)| {
            let mut mb = MeshBuilder::default();
            let mut items: Vec<PickItem> = Vec::new();
            let mut bad = 0usize;
            let mut rg_boxes: std::collections::HashMap<u32, [f64; 4]> = Default::default();
            let tr = BulkTransformer::new(crs, display);
            let rows = process_batch(
                &batch, start_row, geom_col, &tr, display, &mut mb, &mut items, &mut bad,
                rg_starts, &mut rg_boxes,
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
    start_row: u64,
    geom_col: usize,
    tr: &BulkTransformer,
    display: &DisplayCrs,
    mb: &mut MeshBuilder,
    items: &mut Vec<PickItem>,
    bad: &mut usize,
    rg_starts: &[u64],
    rg_boxes: &mut std::collections::HashMap<u32, [f64; 4]>,
) -> usize {
    let col = batch.column(geom_col);
    let Some(get) = BinCol::new(col.as_ref()) else {
        *bad += batch.num_rows();
        return batch.num_rows();
    };

    for row in 0..batch.num_rows() {
        let Some(buf) = get.value(row) else {
            continue;
        };
        let fref = FeatureRef {
            index: (start_row + row as u64) as u32,
        };

        // Fast path: 2D WKB point, no per-feature geo allocation.
        if let Some((x, y)) = parse_wkb_point_2d(buf) {
            grow_rg_box(rg_boxes, rg_of(start_row + row as u64, rg_starts), [x, y, x, y]);
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
                                rg_of(start_row + row as u64, rg_starts),
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
fn parse_wkb_point_2d(buf: &[u8]) -> Option<(f64, f64)> {
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

#[cfg(test)]
pub fn build_geometry_for_test_full(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
) -> Result<BuildOutput, String> {
    build_geometry(store, crs, display, None, None)
}

#[cfg(test)]
pub fn build_geometry_for_test(
    store: &FeatureStore,
    crs: &Crs,
    display: &DisplayCrs,
) -> Result<(super::layer::LayerGeometry, usize, usize), String> {
    build_geometry(store, crs, display, None, None).map(|(g, r, b, _)| (g, r, b))
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
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (_geom, rows, _bad, computed) =
            build_geometry(&store, &crs, &display, None, None).unwrap();
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
        let (geometry, rows, _bad, _rg) =
            build_geometry(&store, &crs, &display, None, Some(&sel)).unwrap();
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
    }
}
