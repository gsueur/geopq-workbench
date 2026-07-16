//! GeoParquet optimization export.
//!
//! Rewrites a GeoParquet file spatially sorted (Hilbert curve over feature
//! bbox centers) with tuned row groups, so that per-row-group bboxes become
//! compact and metadata-based pruning works. Output is one of:
//!
//! - **GeoParquet 1.1 (WKB)**: WKB column + `geo` metadata + `bbox` covering
//!   struct column (per-row-group min/max column statistics drive pruning).
//! - **GeoParquet 1.1 (GeoArrow)**: geometry as nested coordinate arrays
//!   (single geometry family; singles promote to their multi variant) — the
//!   x/y leaves carry ordinary parquet statistics, so any reader prunes
//!   without covering support.
//! - **GeoParquet 2.0**: parquet-native `GEOMETRY` logical type; the writer
//!   computes native geospatial statistics per column chunk (bbox + types).
//!
//! Geometry transcodes in any direction (WKB ↔ GeoArrow).
//!
//! Bloom filters found on source columns are reproduced on the output, or
//! can be added to all attribute columns.
//!
//! The rewrite is in-memory (decoded batches are held while re-ordering);
//! files whose uncompressed size exceeds `MAX_IN_MEMORY_BYTES` are refused.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StructArray};
use arrow::buffer::NullBuffer;
use arrow::compute::interleave_record_batch;
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use serde_json::{json, Value};

use super::geoarrow::{self, GaBuilder, GeomCol, GeomEncoding};
use super::source::Source;

/// Refuse in-memory rewrites beyond this uncompressed size.
const MAX_IN_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const READ_BATCH: usize = 64 * 1024;
/// Hilbert curve order: 2^16 cells per axis.
const HILBERT_ORDER: u32 = 16;
/// Uncompressed page-size cap for covering bbox leaves: ~4096 f64 rows per
/// page, so the page index can prune well below row-group granularity.
const BBOX_LEAF_PAGE_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpVersion {
    /// WKB + `geo` metadata + covering bbox column.
    V1_1,
    /// GeoArrow native coordinate arrays (single geometry family; singles
    /// promote to their multi variant).
    V1_1GeoArrow,
    /// Parquet-native GEOMETRY logical type + geospatial statistics.
    V2_0,
}

impl GpVersion {
    pub fn label(&self) -> &'static str {
        match self {
            GpVersion::V1_1 => "GeoParquet 1.1 (WKB + covering bbox)",
            GpVersion::V1_1GeoArrow => "GeoParquet 1.1 (GeoArrow coordinate arrays)",
            GpVersion::V2_0 => "GeoParquet 2.0 (native GEOMETRY + geo stats)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    Zstd,
    Snappy,
    Uncompressed,
}

impl Codec {
    pub fn label(&self) -> &'static str {
        match self {
            Codec::Zstd => "zstd",
            Codec::Snappy => "snappy",
            Codec::Uncompressed => "none",
        }
    }
    fn compression(&self) -> Compression {
        match self {
            Codec::Zstd => Compression::ZSTD(ZstdLevel::try_new(3).unwrap()),
            Codec::Snappy => Compression::SNAPPY,
            Codec::Uncompressed => Compression::UNCOMPRESSED,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BloomMode {
    /// Enable bloom filters on the columns that have them in the source.
    Preserve,
    /// Enable on every string/binary/integer attribute column.
    AllAttributes,
    /// Write no bloom filters.
    None,
}

impl BloomMode {
    pub fn label(&self) -> &'static str {
        match self {
            BloomMode::Preserve => "preserve source",
            BloomMode::AllAttributes => "all attribute columns",
            BloomMode::None => "none",
        }
    }
}

#[derive(Clone)]
pub struct OptimizeOptions {
    pub version: GpVersion,
    pub row_group_size: usize,
    pub codec: Codec,
    /// Sort features along a Hilbert curve over bbox centers.
    pub hilbert_sort: bool,
    /// Write a `bbox` covering struct column (always useful for 1.1
    /// readers; redundant but allowed alongside 2.0 native stats).
    pub covering: bool,
    pub bloom: BloomMode,
    /// Export only features whose bbox intersects this rect (data CRS).
    pub filter_rect: Option<[f64; 4]>,
    /// Add an `h3_r{n}` UInt64 cell column (centroid-based).
    pub h3_resolution: Option<u8>,
    /// Split the output into hive directories / adaptive H3 cells.
    pub partition: super::partition::PartitionBy,
    /// Source has no geometry column: synthesize WKB points from these
    /// coordinate columns (x/lon, y/lat) — materializes x/y layers into
    /// real GeoParquet.
    pub xy_geom: Option<(usize, usize)>,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            version: GpVersion::V1_1,
            row_group_size: 65_536,
            codec: Codec::Zstd,
            hilbert_sort: true,
            covering: true,
            bloom: BloomMode::Preserve,
            filter_rect: None,
            h3_resolution: None,
            partition: super::partition::PartitionBy::None,
            xy_geom: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptimizeReport {
    pub rows: u64,
    pub size_before: u64,
    pub size_after: u64,
    pub rg_before: usize,
    pub rg_after: usize,
    /// Avg row-group bbox overlap (see `bbox_overlap_metric`).
    pub overlap_before: f64,
    pub overlap_after: f64,
    pub bloom_columns: Vec<String>,
    pub version_label: String,
    pub elapsed_ms: u64,
    /// Output files written (1 unless partitioned).
    pub files: usize,
}

impl OptimizeReport {
    /// Overlap as a fraction of the possible overlaps (comparable across
    /// different row-group counts, unlike the raw metric).
    pub fn overlap_frac_before(&self) -> f64 {
        self.overlap_before / (self.rg_before.max(2) - 1) as f64
    }
    pub fn overlap_frac_after(&self) -> f64 {
        self.overlap_after / (self.rg_after.max(2) - 1) as f64
    }
}

/// Rewrite `src` into `dst` per `opts`. `epsg_hint`: CRS to record when the
/// source has no usable CRS metadata (e.g. the already-loaded layer's CRS).
/// `progress(frac, stage)` is called from the worker thread.
/// Append a WKB point column synthesized from two coordinate columns
/// (null when either coordinate is null).
fn append_xy_wkb(
    batch: &RecordBatch,
    xi: usize,
    yi: usize,
    schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch, String> {
    use arrow::array::{Array, BinaryBuilder, Float64Array};
    let as_f64 = |i: usize| -> Result<Float64Array, String> {
        arrow::compute::cast(batch.column(i), &DataType::Float64)
            .map_err(|e| format!("coordinate cast: {e}"))
            .map(|a| a.as_any().downcast_ref::<Float64Array>().unwrap().clone())
    };
    let (xs, ys) = (as_f64(xi)?, as_f64(yi)?);
    let mut b = BinaryBuilder::with_capacity(batch.num_rows(), batch.num_rows() * 21);
    for i in 0..batch.num_rows() {
        if xs.is_null(i) || ys.is_null(i) {
            b.append_null();
            continue;
        }
        let mut wkb = [0u8; 21];
        wkb[0] = 1; // little endian
        wkb[1..5].copy_from_slice(&1u32.to_le_bytes());
        wkb[5..13].copy_from_slice(&xs.value(i).to_le_bytes());
        wkb[13..21].copy_from_slice(&ys.value(i).to_le_bytes());
        b.append_value(wkb);
    }
    let mut cols = batch.columns().to_vec();
    cols.push(Arc::new(b.finish()));
    RecordBatch::try_new(Arc::clone(schema), cols)
        .map_err(|e| format!("point synthesis: {e}"))
}

pub fn optimize(
    src: &Source,
    dst: &Path,
    opts: &OptimizeOptions,
    epsg_hint: Option<u32>,
    admin: Option<&super::partition::AdminJoinSpec>,
    progress: &dyn Fn(f32, &str),
) -> Result<OptimizeReport, String> {
    let t0 = std::time::Instant::now();
    progress(0.0, "reading metadata");

    let builder = ParquetRecordBatchReaderBuilder::try_new(src.open()?)
        .map_err(|e| format!("not a parquet file: {e}"))?;
    let src_schema = builder.schema().clone();
    let meta = builder.metadata().clone();
    let fmd = meta.file_metadata();

    let uncompressed: u64 = meta
        .row_groups()
        .iter()
        .map(|rg| rg.total_byte_size().max(0) as u64)
        .sum();
    if uncompressed > MAX_IN_MEMORY_BYTES {
        return Err(format!(
            "file is {} uncompressed; in-memory optimize is capped at {} (streaming rewrite not implemented yet)",
            super::info::fmt_bytes(uncompressed),
            super::info::fmt_bytes(MAX_IN_MEMORY_BYTES)
        ));
    }

    // --- source geo metadata ---
    let kv = fmd.key_value_metadata().cloned().unwrap_or_default();
    let geo_meta: Option<Value> = kv
        .iter()
        .find(|kv| kv.key == "geo")
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| serde_json::from_str(v).ok());
    let (primary, geom_idx, src_encoding) = match opts.xy_geom {
        // x/y source: a WKB point column is synthesized during the read
        // and appended after the source fields.
        Some(_) => {
            let name = if src_schema.index_of("geometry").is_ok() {
                "xy_geometry"
            } else {
                "geometry"
            };
            (name.to_string(), src_schema.fields().len(), GeomEncoding::Wkb)
        }
        None => {
            let primary = geo_meta
                .as_ref()
                .and_then(|m| m.get("primary_column"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    guess_geom_column(&src_schema).unwrap_or_else(|| "geometry".into())
                });
            let geom_idx = src_schema
                .index_of(&primary)
                .map_err(|_| format!("geometry column '{primary}' not found"))?;
            let src_encoding = geo_meta
                .as_ref()
                .and_then(|m| m.get("columns")?.get(&primary)?.get("encoding")?.as_str())
                .map(|e| {
                    GeomEncoding::parse(e).ok_or_else(|| format!("encoding '{e}' not supported"))
                })
                .transpose()?
                .unwrap_or_default();
            (primary, geom_idx, src_encoding)
        }
    };

    // CRS, best source first: geo metadata PROJJSON, then the GEOMETRY
    // logical type's crs string (2.0 sources), then the caller's hint.
    let crs_value: Option<Value> = geo_meta
        .as_ref()
        .and_then(|m| m.get("columns")?.get(&primary)?.get("crs").cloned())
        .filter(|v| !v.is_null())
        .or_else(|| logical_type_crs(&meta, &primary))
        .or_else(|| {
            epsg_hint.filter(|&e| e != 4326).map(|e| {
                json!({"id": {"authority": "EPSG", "code": e}})
            })
        });

    // Source covering bbox column (dropped and rebuilt if we write our own).
    let src_covering_root: Option<String> = geo_meta
        .as_ref()
        .and_then(|m| {
            m.get("columns")?
                .get(&primary)?
                .get("covering")?
                .get("bbox")?
                .get("xmin")?
                .as_array()?
                .first()?
                .as_str()
                .map(str::to_string)
        });

    // Columns carrying bloom filters in the source (leaf paths).
    let src_bloom: Vec<Vec<String>> = meta
        .row_groups()
        .first()
        .map(|rg| {
            rg.columns()
                .iter()
                .filter(|c| c.bloom_filter_offset().is_some())
                .map(|c| c.column_descr().path().parts().to_vec())
                .collect()
        })
        .unwrap_or_default();

    let rg_before = meta.num_row_groups();
    let src_rg_rows: Vec<usize> = meta
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as usize)
        .collect();

    // --- read everything, scanning geometry bboxes as batches arrive ---
    progress(0.02, "reading rows");
    let total_rows = fmd.num_rows().max(0) as usize;
    let reader = ParquetRecordBatchReaderBuilder::try_new(src.open()?)
        .map_err(|e| format!("not a parquet file: {e}"))?
        .with_batch_size(READ_BATCH)
        .build()
        .map_err(|e| format!("parquet read error: {e}"))?;

    // x/y source: extend the schema with the synthesized point column.
    let src_schema = match opts.xy_geom {
        Some(_) => {
            let mut fields: Vec<arrow::datatypes::Field> = src_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            fields.push(arrow::datatypes::Field::new(&primary, DataType::Binary, true));
            Arc::new(arrow::datatypes::Schema::new_with_metadata(
                fields,
                src_schema.metadata().clone(),
            ))
        }
        None => src_schema,
    };

    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut row_bboxes: Vec<Option<[f64; 4]>> = Vec::with_capacity(total_rows);
    let mut geom_types: HashSet<&'static str> = HashSet::new();
    for res in reader {
        let mut batch = res.map_err(|e| format!("parquet decode error: {e}"))?;
        if let Some((xi, yi)) = opts.xy_geom {
            batch = append_xy_wkb(&batch, xi, yi, &src_schema)?;
        }
        scan_bboxes(&batch, geom_idx, src_encoding, &mut row_bboxes, &mut geom_types)?;
        batches.push(batch);
        progress(
            0.02 + 0.48 * (row_bboxes.len() as f32 / total_rows.max(1) as f32),
            "reading rows",
        );
    }
    let rows = row_bboxes.len();
    if rows == 0 {
        return Err("file has no rows".into());
    }

    let file_bbox = union_bboxes(row_bboxes.iter().flatten());
    let overlap_before = {
        let mut boxes = Vec::with_capacity(rg_before);
        let mut off = 0usize;
        for n in &src_rg_rows {
            boxes.extend(union_bboxes(row_bboxes[off..off + n].iter().flatten()));
            off += n;
        }
        super::loader::bbox_overlap_metric(&boxes)
    };

    // --- selection + sort order ---
    progress(0.52, "sorting (Hilbert)");
    let mut order: Vec<u32> = (0..rows as u32).collect();
    if let Some(rect) = opts.filter_rect {
        order.retain(|&i| {
            row_bboxes[i as usize].is_some_and(|b| {
                b[0] <= rect[2] && b[2] >= rect[0] && b[1] <= rect[3] && b[3] >= rect[1]
            })
        });
        if order.is_empty() {
            return Err("no features intersect the current viewport".into());
        }
    }
    let written_rows = order.len();
    // Metadata bbox reflects what is actually exported.
    let file_bbox = if opts.filter_rect.is_some() {
        union_bboxes(order.iter().filter_map(|&i| row_bboxes[i as usize].as_ref()))
    } else {
        file_bbox
    };
    if opts.hilbert_sort {
        let fb = file_bbox.unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let sx = (fb[2] - fb[0]).max(f64::MIN_POSITIVE);
        let sy = (fb[3] - fb[1]).max(f64::MIN_POSITIVE);
        let n = (1u64 << HILBERT_ORDER) as f64;
        let codes: Vec<u64> = row_bboxes
            .iter()
            .map(|b| match b {
                Some(b) => {
                    let cx = ((b[0] + b[2]) * 0.5 - fb[0]) / sx;
                    let cy = ((b[1] + b[3]) * 0.5 - fb[1]) / sy;
                    let gx = ((cx * n) as u64).min((1 << HILBERT_ORDER) - 1) as u32;
                    let gy = ((cy * n) as u64).min((1 << HILBERT_ORDER) - 1) as u32;
                    hilbert_xy2d(HILBERT_ORDER, gx, gy)
                }
                None => u64::MAX, // null geometries sort last
            })
            .collect();
        order.sort_by_key(|&i| codes[i as usize]);
    }

    // --- derived columns + partition plan ---
    use super::partition::{self, PartitionBy};
    let part_fields: Vec<String> = match &opts.partition {
        PartitionBy::Fields(f) if f.is_empty() => {
            return Err("partitioning by fields: none selected".into())
        }
        PartitionBy::Fields(f) => f.clone(),
        _ => Vec::new(),
    };
    let h3_name = opts.h3_resolution.map(|r| format!("h3_r{r}"));
    let need_lonlat = opts.h3_resolution.is_some()
        || matches!(opts.partition, PartitionBy::AdaptiveH3 { .. });
    let data_crs = if need_lonlat || admin.is_some() {
        Some(
            super::crs::Crs::from_geoparquet_crs(crs_value.as_ref())
                .map_err(|e| format!("H3/admin needs a resolvable CRS: {e}"))?,
        )
    } else {
        None
    };
    progress(0.53, "computing derived columns");
    let lonlat: Option<Vec<Option<(f64, f64)>>> = need_lonlat.then(|| {
        partition::centroids_in(&row_bboxes, data_crs.as_ref().unwrap(), &super::crs::Crs::wgs84())
    });
    let h3_vals: Option<Vec<Option<u64>>> = match (opts.h3_resolution, &lonlat) {
        (Some(res), Some(ll)) => Some(partition::h3_cells(ll, res)?),
        _ => None,
    };
    let admin_vals: Option<Vec<Option<String>>> = match admin {
        Some(spec) => {
            let cb = partition::centroids_in(&row_bboxes, data_crs.as_ref().unwrap(), &spec.crs);
            Some(partition::admin_join(spec, &cb)?)
        }
        None => None,
    };
    // Per-row string values for each hive partition field.
    let field_values: Vec<(String, Vec<Option<String>>)> = part_fields
        .iter()
        .map(|name| -> Result<(String, Vec<Option<String>>), String> {
            if Some(name.as_str()) == h3_name.as_deref() {
                let vals = h3_vals
                    .as_ref()
                    .ok_or("internal: h3 partition field without h3 column")?
                    .iter()
                    .map(|v| {
                        v.and_then(|v| h3o::CellIndex::try_from(v).ok())
                            .map(|c| c.to_string())
                    })
                    .collect();
                return Ok((name.clone(), vals));
            }
            if Some(name.as_str()) == admin.map(|a| a.out_name.as_str()) {
                return Ok((name.clone(), admin_vals.clone().unwrap_or_default()));
            }
            let idx = src_schema
                .index_of(name)
                .map_err(|_| format!("partition field '{name}' not found"))?;
            if idx == geom_idx {
                return Err("cannot partition by the geometry column".into());
            }
            use arrow::util::display::{ArrayFormatter, FormatOptions};
            let fopts = FormatOptions::default().with_display_error(true);
            let mut vals: Vec<Option<String>> = Vec::with_capacity(rows);
            for b in &batches {
                let col = b.column(idx);
                let f = ArrayFormatter::try_new(col.as_ref(), &fopts)
                    .map_err(|e| format!("partition field '{name}': {e}"))?;
                for i in 0..b.num_rows() {
                    vals.push((!col.is_null(i)).then(|| f.value(i).to_string()));
                }
            }
            Ok((name.clone(), vals))
        })
        .collect::<Result<_, _>>()?;

    // --- geometry output form ---
    let geom_out: GeomOut = match opts.version {
        GpVersion::V1_1GeoArrow => {
            let target = geoarrow::target_encoding(geom_types.iter().copied())?;
            if target == src_encoding {
                GeomOut::PassThrough
            } else {
                GeomOut::ToGa(target)
            }
        }
        GpVersion::V1_1 | GpVersion::V2_0 if !src_encoding.is_wkb() => GeomOut::ToWkb,
        _ => GeomOut::PassThrough,
    };
    let out_encoding = match (&geom_out, opts.version) {
        (GeomOut::ToGa(t), _) => *t,
        (GeomOut::PassThrough, GpVersion::V1_1GeoArrow) => src_encoding,
        _ => GeomEncoding::Wkb,
    };

    // --- output schema ---
    let write_covering = opts.covering;
    let drop_covering = write_covering.then_some(src_covering_root.as_deref()).flatten();
    let mut fields: Vec<Field> = Vec::new();
    let mut kept_src_indices: Vec<usize> = Vec::new();
    for (i, f) in src_schema.fields().iter().enumerate() {
        if Some(f.name().as_str()) == drop_covering
            || (write_covering && f.name() == "bbox" && i != geom_idx)
            // Hive convention: partition columns live in the path only.
            || (part_fields.iter().any(|p| p == f.name()) && i != geom_idx)
        {
            continue;
        }
        let mut field = f.as_ref().clone();
        if i == geom_idx {
            // Rebuild the geometry field per the output form, then apply
            // version-specific typing.
            match &geom_out {
                GeomOut::ToGa(t) => {
                    field = Field::new(f.name(), geoarrow::data_type(*t), true);
                }
                GeomOut::ToWkb => {
                    field = Field::new(f.name(), DataType::Binary, true);
                }
                GeomOut::PassThrough => {}
            }
            match opts.version {
                GpVersion::V2_0 => {
                    let crs_str = crs_value.as_ref().map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    });
                    let md = parquet_geospatial::WkbMetadata::new(crs_str.as_deref(), None);
                    field
                        .try_with_extension_type(parquet_geospatial::WkbType::new(Some(md)))
                        .map_err(|e| format!("cannot tag geometry column: {e}"))?;
                }
                GpVersion::V1_1 | GpVersion::V1_1GeoArrow => {
                    // A native-GEOMETRY source propagates its extension type
                    // through the arrow schema; strip it so an explicit 1.1
                    // export carries no logical type (max reader
                    // compatibility — the point of choosing 1.1).
                    let mut md = field.metadata().clone();
                    md.remove("ARROW:extension:name");
                    md.remove("ARROW:extension:metadata");
                    field.set_metadata(md);
                }
            }
        }
        kept_src_indices.push(i);
        fields.push(field);
    }
    let bbox_fields: Fields = ["xmin", "ymin", "xmax", "ymax"]
        .iter()
        .map(|n| Field::new(*n, DataType::Float64, true))
        .collect();
    if write_covering {
        fields.push(Field::new("bbox", DataType::Struct(bbox_fields.clone()), true));
    }
    // Derived columns (skipped when hive uses them as path-only keys).
    let write_h3 = h3_name
        .as_deref()
        .filter(|n| !part_fields.iter().any(|p| p == n));
    if let Some(n) = write_h3 {
        fields.push(Field::new(n, DataType::UInt64, true));
    }
    let write_admin = admin
        .map(|a| a.out_name.as_str())
        .filter(|n| !part_fields.iter().any(|p| p == n));
    if let Some(n) = write_admin {
        fields.push(Field::new(n, DataType::Utf8, true));
    }

    let out_schema = Arc::new(Schema::new(fields));

    // --- writer properties (rebuilt per output file) ---
    let mut bloom_columns: Vec<String> = Vec::new();
    match opts.bloom {
        BloomMode::Preserve => {
            for parts in &src_bloom {
                if parts.first().map(String::as_str) == drop_covering {
                    continue; // rebuilt bbox column gets no bloom filter
                }
                if parts
                    .first()
                    .is_some_and(|r| part_fields.iter().any(|p| p == r))
                {
                    continue; // partition columns are path-only
                }
                bloom_columns.push(parts.join("."));
            }
        }
        BloomMode::AllAttributes => {
            for f in out_schema.fields() {
                if f.name() == &primary || f.name() == "bbox" {
                    continue;
                }
                let eligible = matches!(
                    f.data_type(),
                    DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                        | DataType::Binary
                        | DataType::LargeBinary
                        | DataType::BinaryView
                        | DataType::Int8
                        | DataType::Int16
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64
                        | DataType::Date32
                        | DataType::Date64
                );
                if eligible {
                    bloom_columns.push(f.name().clone());
                }
            }
        }
        BloomMode::None => {}
    }
    let src_bloom_paths: Vec<Vec<String>> = match opts.bloom {
        BloomMode::Preserve => bloom_columns
            .iter()
            .map(|c| c.split('.').map(str::to_string).collect())
            .collect(),
        _ => bloom_columns.iter().map(|c| vec![c.clone()]).collect(),
    };
    let make_props = || {
        let mut props = WriterProperties::builder()
            .set_compression(opts.codec.compression())
            .set_max_row_group_row_count(Some(opts.row_group_size))
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_created_by(format!("geopq-viewer {}", env!("CARGO_PKG_VERSION")));
        if write_covering {
            // Small pages on the bbox leaves (~4k rows at 8 B/value) give
            // the page index sub-row-group granularity, so readers can
            // prune at page level instead of whole row groups. Dictionary
            // encoding is disabled there: coordinates are mostly unique
            // (no dict win) and the page-size cap applies to the encoded
            // size, which tiny dict indices would defeat.
            for leaf in ["xmin", "ymin", "xmax", "ymax"] {
                let path = ColumnPath::new(vec!["bbox".into(), leaf.into()]);
                props = props
                    .set_column_data_page_size_limit(path.clone(), BBOX_LEAF_PAGE_BYTES)
                    .set_column_dictionary_enabled(path, false);
            }
        }
        for parts in &src_bloom_paths {
            props = props
                .set_column_bloom_filter_enabled(ColumnPath::new(parts.clone()), true)
                .set_column_bloom_filter_fpp(ColumnPath::new(parts.clone()), 0.01);
        }
        props.build()
    };

    // --- partition plan ---
    let parts: Vec<(String, Vec<u32>)> = match &opts.partition {
        PartitionBy::None => vec![(String::new(), order.clone())],
        PartitionBy::Fields(_) => partition::split_by_fields(&order, &field_values)?,
        PartitionBy::AdaptiveH3 { target_rows, max_res } => partition::split_adaptive_h3(
            &order,
            lonlat.as_ref().ok_or("internal: adaptive H3 without centroids")?,
            (*target_rows).max(1),
            *max_res,
        )?,
    };
    let partitioned = !matches!(opts.partition, PartitionBy::None);

    // --- write, gathering rows in sorted order ---
    progress(0.55, "writing");
    // Global row index -> (batch, offset in batch).
    let mut batch_starts: Vec<usize> = Vec::with_capacity(batches.len() + 1);
    let mut acc = 0usize;
    for b in &batches {
        batch_starts.push(acc);
        acc += b.num_rows();
    }
    batch_starts.push(acc);
    let locate = |row: u32| -> (usize, usize) {
        let row = row as usize;
        let bi = match batch_starts.binary_search(&row) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (bi, row - batch_starts[bi])
    };
    let batch_refs: Vec<&RecordBatch> = batches.iter().collect();
    let chunk_rows = opts.row_group_size.min(READ_BATCH);
    let out_geom_pos = kept_src_indices
        .iter()
        .position(|&i| i == geom_idx)
        .ok_or("geometry column dropped from output")?;

    // One file per partition; everything (covering, bloom, ordering, geo
    // metadata with per-file bbox) applies inside each file.
    let mut size_after = 0u64;
    let mut rg_after_boxes: Vec<[f64; 4]> = Vec::new();
    let mut rg_after = 0usize;
    let mut written = 0usize;
    for (rel, part_order) in &parts {
        let path = if partitioned {
            let dir = dst.join(rel);
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
            dir.join("part-0.parquet")
        } else {
            dst.to_path_buf()
        };
        let part_bbox =
            union_bboxes(part_order.iter().filter_map(|&r| row_bboxes[r as usize].as_ref()));
        let out_file =
            File::create(&path).map_err(|e| format!("cannot create output: {e}"))?;
        let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(make_props()))
            .map_err(|e| format!("writer init: {e}"))?;
        // `geo` is rebuilt; other source key-value metadata passes through.
        writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            build_geo_meta(opts, &primary, crs_value.as_ref(), &geom_types, out_encoding, part_bbox)
                .to_string(),
        ));
        for entry in kv.iter().filter(|kv| kv.key != "geo" && kv.key != "ARROW:schema") {
            writer.append_key_value_metadata(entry.clone());
        }

        for chunk in part_order.chunks(chunk_rows) {
            let indices: Vec<(usize, usize)> = chunk.iter().map(|&r| locate(r)).collect();
            let gathered = interleave_record_batch(&batch_refs, &indices)
                .map_err(|e| format!("gather failed: {e}"))?;
            let mut cols: Vec<ArrayRef> = kept_src_indices
                .iter()
                .map(|&i| gathered.column(i).clone())
                .collect();
            if !matches!(geom_out, GeomOut::PassThrough) {
                cols[out_geom_pos] =
                    transcode_geometry(cols[out_geom_pos].as_ref(), src_encoding, &geom_out)?;
            }
            if write_covering {
                cols.push(build_bbox_column(chunk, &row_bboxes, &bbox_fields));
            }
            if write_h3.is_some() {
                let vals = h3_vals.as_ref().unwrap();
                cols.push(Arc::new(arrow::array::UInt64Array::from_iter(
                    chunk.iter().map(|&r| vals[r as usize]),
                )));
            }
            if write_admin.is_some() {
                let vals = admin_vals.as_ref().unwrap();
                cols.push(Arc::new(arrow::array::StringArray::from_iter(
                    chunk.iter().map(|&r| vals[r as usize].clone()),
                )));
            }
            let out = RecordBatch::try_new(out_schema.clone(), cols)
                .map_err(|e| format!("batch assembly failed: {e}"))?;
            writer.write(&out).map_err(|e| format!("write failed: {e}"))?;
            written += chunk.len();
            progress(0.55 + 0.45 * (written as f32 / written_rows.max(1) as f32), "writing");
        }
        let closed = writer.close().map_err(|e| format!("finalize failed: {e}"))?;

        let mut off = 0usize;
        for rg in closed.row_groups() {
            let n = rg.num_rows().max(0) as usize;
            rg_after_boxes.extend(union_bboxes(
                part_order[off..off + n].iter().flat_map(|&r| &row_bboxes[r as usize]),
            ));
            off += n;
            rg_after += 1;
        }
        size_after += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    }
    let overlap_after = super::loader::bbox_overlap_metric(&rg_after_boxes);

    Ok(OptimizeReport {
        rows: written_rows as u64,
        size_before: src.size(),
        size_after,
        rg_before,
        rg_after,
        overlap_before,
        overlap_after,
        bloom_columns,
        version_label: opts.version.label().into(),
        elapsed_ms: t0.elapsed().as_millis() as u64,
        files: parts.len(),
    })
}

/// How the geometry column is rewritten.
enum GeomOut {
    /// Values move with the row gather unchanged.
    PassThrough,
    /// Decode (WKB or GeoArrow) and re-encode as WKB bytes.
    ToWkb,
    /// Decode and rebuild as GeoArrow coordinate arrays.
    ToGa(GeomEncoding),
}

/// Re-encode one gathered geometry column.
fn transcode_geometry(
    col: &dyn arrow::array::Array,
    src_encoding: GeomEncoding,
    out: &GeomOut,
) -> Result<ArrayRef, String> {
    let geoms =
        GeomCol::new(col, src_encoding).ok_or("geometry column does not match its encoding")?;
    match out {
        GeomOut::PassThrough => unreachable!(),
        GeomOut::ToGa(target) => {
            let mut b = GaBuilder::new(*target);
            for i in 0..col.len() {
                b.push(geoms.geometry(i).as_ref())?;
            }
            Ok(b.finish())
        }
        GeomOut::ToWkb => {
            let opts = wkb::writer::WriteOptions::default();
            let mut buf: Vec<u8> = Vec::new();
            let values: Vec<Option<Vec<u8>>> = (0..col.len())
                .map(|i| -> Result<Option<Vec<u8>>, String> {
                    let Some(g) = geoms.geometry(i) else {
                        return Ok(None);
                    };
                    buf.clear();
                    wkb::writer::write_geometry(&mut buf, &g, &opts)
                        .map_err(|e| format!("WKB encode: {e}"))?;
                    Ok(Some(buf.clone()))
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(arrow::array::BinaryArray::from_iter(values)))
        }
    }
}

/// GeoParquet `geo` file metadata for the output.
fn build_geo_meta(
    opts: &OptimizeOptions,
    primary: &str,
    crs: Option<&Value>,
    geom_types: &HashSet<&'static str>,
    out_encoding: GeomEncoding,
    file_bbox: Option<[f64; 4]>,
) -> Value {
    // GeoArrow columns store exactly one type (singles promoted).
    let types: Vec<&str> = if out_encoding.is_wkb() {
        let mut t: Vec<&str> = geom_types.iter().copied().collect();
        t.sort_unstable();
        t
    } else {
        vec![out_encoding.geometry_type_name()]
    };
    let mut col = json!({
        "encoding": out_encoding.geo_name(),
        "geometry_types": types,
    });
    if let Some(c) = crs {
        col["crs"] = c.clone();
    }
    if let Some(b) = file_bbox {
        col["bbox"] = json!([b[0], b[1], b[2], b[3]]);
    }
    if opts.covering {
        col["covering"] = json!({"bbox": {
            "xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
            "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"],
        }});
    }
    json!({
        "version": match opts.version {
            GpVersion::V1_1 | GpVersion::V1_1GeoArrow => "1.1.0",
            GpVersion::V2_0 => "2.0.0",
        },
        "primary_column": primary,
        "columns": { primary: col },
    })
}

/// CRS string recorded in a GEOMETRY/GEOGRAPHY logical type (2.0 sources).
/// Looks the geometry column up by name: parquet leaf indices don't line up
/// with arrow root indices once nested columns exist.
fn logical_type_crs(
    meta: &parquet::file::metadata::ParquetMetaData,
    geom_name: &str,
) -> Option<Value> {
    use parquet::basic::LogicalType;
    let col = meta
        .row_groups()
        .first()?
        .columns()
        .iter()
        .find(|c| c.column_descr().path().parts().first().map(String::as_str) == Some(geom_name))?;
    let crs = match col.column_descr().logical_type_ref() {
        Some(LogicalType::Geometry { crs }) => crs.clone(),
        Some(LogicalType::Geography { crs, .. }) => crs.clone(),
        _ => None,
    }?;
    serde_json::from_str(&crs)
        .ok()
        .or(Some(Value::String(crs)))
}

fn guess_geom_column(schema: &Schema) -> Option<String> {
    ["geometry", "geom", "wkb_geometry", "wkb"]
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
}

/// Append per-row bboxes (None for null/undecodable geometries).
fn scan_bboxes(
    batch: &RecordBatch,
    geom_idx: usize,
    encoding: GeomEncoding,
    out: &mut Vec<Option<[f64; 4]>>,
    geom_types: &mut HashSet<&'static str>,
) -> Result<(), String> {
    use geo::BoundingRect;
    let col = batch.column(geom_idx);
    let geoms = GeomCol::new(col.as_ref(), encoding)
        .ok_or("geometry column does not match its declared encoding")?;
    for i in 0..batch.num_rows() {
        if geoms.is_null(i) {
            out.push(None);
            continue;
        }
        if let Some((x, y)) = geoms.point2(i) {
            geom_types.insert("Point");
            out.push((x.is_finite() && y.is_finite()).then_some([x, y, x, y]));
            continue;
        }
        match geoms.geometry(i) {
            Some(geom) => {
                geom_types.insert(geom_type_name(&geom));
                out.push(geom.bounding_rect().map(|r| {
                    let (min, max) = (r.min(), r.max());
                    [min.x, min.y, max.x, max.y]
                }));
            }
            None => out.push(None),
        }
    }
    Ok(())
}

fn geom_type_name(g: &geo_types::Geometry<f64>) -> &'static str {
    use geo_types::Geometry::*;
    match g {
        Point(_) => "Point",
        Line(_) | LineString(_) => "LineString",
        Polygon(_) | Rect(_) | Triangle(_) => "Polygon",
        MultiPoint(_) => "MultiPoint",
        MultiLineString(_) => "MultiLineString",
        MultiPolygon(_) => "MultiPolygon",
        GeometryCollection(_) => "GeometryCollection",
    }
}

fn union_bboxes<'a>(boxes: impl Iterator<Item = &'a [f64; 4]>) -> Option<[f64; 4]> {
    let mut out: Option<[f64; 4]> = None;
    for b in boxes {
        out = Some(match out {
            None => *b,
            Some(a) => [
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ],
        });
    }
    out
}

/// Covering `bbox` struct column for the given (sorted) global row indices.
fn build_bbox_column(
    rows: &[u32],
    row_bboxes: &[Option<[f64; 4]>],
    bbox_fields: &Fields,
) -> ArrayRef {
    let get = |k: usize| -> ArrayRef {
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|&r| row_bboxes[r as usize].map(|b| b[k]))
                .collect::<Vec<_>>(),
        ))
    };
    let validity = NullBuffer::from(
        rows.iter()
            .map(|&r| row_bboxes[r as usize].is_some())
            .collect::<Vec<_>>(),
    );
    Arc::new(StructArray::new(
        bbox_fields.clone(),
        vec![get(0), get(1), get(2), get(3)],
        Some(validity),
    ))
}

/// Hilbert curve distance of cell (x, y) on a 2^order × 2^order grid.
fn hilbert_xy2d(order: u32, mut x: u32, mut y: u32) -> u64 {
    let mut d: u64 = 0;
    let mut s: u32 = 1 << (order - 1);
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        // Rotate the quadrant so the curve connects.
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, StringArray};
    use parquet::file::properties::WriterProperties;

    /// Partitioned exports: hive fields (incl. an admin join column),
    /// adaptive H3, and the H3 cell column — every partition file must be
    /// a loadable GeoParquet with the partition columns path-only.
    #[test]
    fn partitioned_export_roundtrips() {
        use crate::data::partition::{AdminJoinSpec, PartitionBy};
        let dir = std::env::temp_dir().join("geopq_optimize_partition");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Four spatial clusters, one per "quad" value (EPSG:2154 meters,
        // clusters ~500 km apart so H3 splits them cleanly).
        let quads = ["A", "B", "C", "D"];
        let centers = [
            (200_000.0, 6_200_000.0),
            (700_000.0, 6_200_000.0),
            (200_000.0, 6_700_000.0),
            (700_000.0, 6_700_000.0),
        ];
        let n_per = 1000usize;
        let (mut wkbs, mut ids, mut quad_vals) = (Vec::new(), Vec::new(), Vec::new());
        for (q, (cx, cy)) in quads.iter().zip(centers) {
            for i in 0..n_per {
                let (dx, dy) = ((i % 32) as f64 * 50.0, (i / 32) as f64 * 50.0);
                wkbs.push(wkb_point(cx + dx, cy + dy));
                ids.push((quad_vals.len()) as i64);
                quad_vals.push(q.to_string());
            }
        }
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": ["Point"],
                "crs": {"id": {"authority": "EPSG", "code": 2154}},
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("quad", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(quad_vals)),
            ],
        )
        .unwrap();
        let src = dir.join("clusters.parquet");
        let mut w =
            ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        // --- hive partition by the quad column ---
        let dst = dir.join("by_quad");
        let opts = OptimizeOptions {
            row_group_size: 2048,
            partition: PartitionBy::Fields(vec!["quad".into()]),
            ..Default::default()
        };
        let rep =
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(rep.files, 4);
        assert_eq!(rep.rows, 4 * n_per as u64);
        for q in quads {
            let f = dst.join(format!("quad={q}")).join("part-0.parquet");
            let (store, crs, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            assert_eq!(store.total_rows(), n_per as u64, "quad={q}");
            assert_eq!(crs.epsg, Some(2154));
            assert!(
                store.schema.index_of("quad").is_err(),
                "partition column must be path-only"
            );
            assert!(store.covering.is_some(), "covering survives partitioning");
        }

        // --- adaptive H3 + H3 column ---
        let dst2 = dir.join("adaptive");
        let opts2 = OptimizeOptions {
            row_group_size: 2048,
            h3_resolution: Some(7),
            partition: PartitionBy::AdaptiveH3 { target_rows: 1500, max_res: 10 },
            ..Default::default()
        };
        let rep2 =
            optimize(&Source::Local(src.clone()), &dst2, &opts2, None, None, &|_, _| {}).unwrap();
        assert!(rep2.files >= 4, "clusters must split: {} files", rep2.files);
        let mut total = 0u64;
        for entry in std::fs::read_dir(&dst2).unwrap() {
            let d = entry.unwrap().path();
            assert!(d.file_name().unwrap().to_str().unwrap().starts_with("h3="));
            let f = d.join("part-0.parquet");
            let (store, _, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            total += store.total_rows();
            let idx = store.schema.index_of("h3_r7").expect("h3 column present");
            assert_eq!(
                store.schema.field(idx).data_type(),
                &DataType::UInt64
            );
            // Values are valid H3 cells at res 7.
            let rows: Vec<u32> = (0..store.total_rows().min(5) as u32).collect();
            let b = store.fetch(&rows, Some(&[idx])).unwrap();
            let col = b[0]
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            for i in 0..col.len() {
                let cell = h3o::CellIndex::try_from(col.value(i)).expect("valid cell");
                assert_eq!(u8::from(cell.resolution()), 7);
            }
        }
        assert_eq!(total, 4 * n_per as u64);

        // --- admin join, partitioned by the joined column ---
        // Two boundary polygons splitting the clusters west/east.
        let mut poly_wkbs: Vec<Vec<u8>> = Vec::new();
        for (x0, x1) in [(0.0, 450_000.0), (450_000.0, 1_000_000.0)] {
            let poly = geo_types::Polygon::new(
                geo_types::LineString::from(vec![
                    (x0, 6_000_000.0),
                    (x1, 6_000_000.0),
                    (x1, 7_000_000.0),
                    (x0, 7_000_000.0),
                    (x0, 6_000_000.0),
                ]),
                vec![],
            );
            let mut buf = Vec::new();
            wkb::writer::write_geometry(
                &mut buf,
                &geo_types::Geometry::Polygon(poly),
                &wkb::writer::WriteOptions::default(),
            )
            .unwrap();
            poly_wkbs.push(buf);
        }
        let bschema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("zone", DataType::Utf8, false),
        ]));
        let bbatch = RecordBatch::try_new(
            bschema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(poly_wkbs.iter())),
                Arc::new(StringArray::from(vec!["west", "east"])),
            ],
        )
        .unwrap();
        let bounds = dir.join("zones.parquet");
        let mut w =
            ArrowWriter::try_new(File::create(&bounds).unwrap(), bschema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&bbatch).unwrap();
        w.close().unwrap();
        let (bstore, bcrs, _, _) = crate::data::loader::open_store_for_test(&bounds).unwrap();
        let admin = AdminJoinSpec {
            out_name: "zone".into(),
            store: Arc::new(bstore),
            value_column: "zone".into(),
            crs: bcrs,
        };

        let dst3 = dir.join("by_zone");
        let opts3 = OptimizeOptions {
            row_group_size: 2048,
            partition: PartitionBy::Fields(vec!["zone".into()]),
            ..Default::default()
        };
        let rep3 = optimize(
            &Source::Local(src.clone()),
            &dst3,
            &opts3,
            None,
            Some(&admin),
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(rep3.files, 2, "west/east");
        for (zone, expect) in [("west", 2 * n_per as u64), ("east", 2 * n_per as u64)] {
            let f = dst3.join(format!("zone={zone}")).join("part-0.parquet");
            let (store, _, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            assert_eq!(store.total_rows(), expect, "zone={zone}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8, 1, 0, 0, 0];
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// Write a spatially-scrambled GeoParquet 1.0 file: `rows` points on a
    /// grid, visited in an order that destroys spatial locality, with a
    /// bloom filter on `name`. Returns (path, expected names by grid pos).
    fn write_scrambled(rows: usize, dir: &Path) -> std::path::PathBuf {
        let path = dir.join(format!("scrambled_{rows}.parquet"));
        let side = (rows as f64).sqrt().ceil() as usize;
        // A large stride coprime with `rows` scrambles spatial order.
        let stride = (rows / 2 + 1) | 1;
        let idx: Vec<usize> = (0..rows).map(|i| (i * stride) % rows).collect();
        let (mut wkbs, mut ids, mut names) = (Vec::new(), Vec::new(), Vec::new());
        for &i in &idx {
            let (gx, gy) = (i % side, i / side);
            let (x, y) = (gx as f64 / side as f64 * 10.0, gy as f64 / side as f64 * 10.0);
            wkbs.push(wkb_point(x, y));
            ids.push(i as i64);
            names.push(format!("cell_{gx}_{gy}"));
        }
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": ["Point"],
                "crs": {"id": {"authority": "EPSG", "code": 2154}},
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2048))
            .set_column_bloom_filter_enabled(ColumnPath::new(vec!["name".into()]), true)
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
        path
    }

    fn bloom_paths(path: &Path) -> Vec<String> {
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        b.metadata().row_groups()[0]
            .columns()
            .iter()
            .filter(|c| c.bloom_filter_offset().is_some())
            .map(|c| c.column_descr().path().string())
            .collect()
    }

    /// Names must stay attached to their geometry after reordering.
    /// Names must stay attached to their geometry after reordering,
    /// whatever the geometry encoding.
    fn assert_rows_consistent(path: &std::path::PathBuf) {
        use geo::BoundingRect;
        let (store, _crs, _info, _rg) =
            crate::data::loader::open_store_for_test(path).unwrap();
        let rows: Vec<u32> = (0..store.total_rows() as u32).step_by(997).collect();
        let geoms = store.fetch_geoms(&rows).unwrap();
        let batches = store.fetch(&rows, None).unwrap();
        let mut names: Vec<String> = Vec::new();
        for batch in batches {
            let col = batch.column(batch.schema().index_of("name").unwrap()).clone();
            let arr = StringArray::from(col.to_data());
            names.extend((0..batch.num_rows()).map(|i| arr.value(i).to_string()));
        }
        assert_eq!(names.len(), geoms.len());
        for ((_, g), name) in geoms.iter().zip(&names) {
            let c = g.as_ref().unwrap().bounding_rect().unwrap().min();
            let side = 200usize; // matches write_scrambled(40_000)
            let gx = (c.x / 10.0 * side as f64).round() as usize;
            let gy = (c.y / 10.0 * side as f64).round() as usize;
            assert_eq!(name, &format!("cell_{gx}_{gy}"));
        }
    }

    #[test]
    fn v1_1_roundtrip_sorts_and_preserves() {
        let dir = std::env::temp_dir().join("geopq_optimize_v11");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_v11.parquet");

        let opts = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        assert_eq!(report.rg_after, 40_000_usize.div_ceil(2048));
        // Scrambled input: every row group spans the full extent. Sorted
        // output: near-disjoint boxes.
        assert!(
            report.overlap_after < report.overlap_before * 0.25,
            "overlap {} -> {}",
            report.overlap_before,
            report.overlap_after
        );
        assert_eq!(report.bloom_columns, vec!["name".to_string()]);
        assert_eq!(bloom_paths(&dst), vec!["name".to_string()]);

        // Our own loader must see a 1.1 covering file with EPSG:2154.
        let (store, crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        assert_eq!(store.total_rows(), 40_000);
        let (source, boxes) = rg_meta.expect("covering stats");
        assert!(source.contains("covering"), "{source}");
        assert_eq!(boxes.len(), report.rg_after);
        assert!(info.geo.version_label.contains("1.1"), "{}", info.geo.version_label);
        assert_rows_consistent(&dst);
    }

    #[test]
    fn v2_0_writes_native_geometry_and_stats() {
        let dir = std::env::temp_dir().join("geopq_optimize_v20");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_v20.parquet");

        let opts = OptimizeOptions {
            version: GpVersion::V2_0,
            row_group_size: 2048,
            covering: false,
            bloom: BloomMode::AllAttributes,
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        let mut bloom = report.bloom_columns.clone();
        bloom.sort();
        assert_eq!(bloom, vec!["id".to_string(), "name".to_string()]);

        // Native GEOMETRY logical type with the CRS, plus geo statistics.
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&dst).unwrap()).unwrap();
        let meta = b.metadata();
        let geom = meta.row_groups()[0]
            .columns()
            .iter()
            .find(|c| c.column_descr().name() == "geometry")
            .expect("geometry column");
        match geom.column_descr().logical_type_ref() {
            Some(parquet::basic::LogicalType::Geometry { crs }) => {
                let crs = crs.as_deref().expect("crs recorded");
                assert!(crs.contains("2154"), "{crs}");
            }
            other => panic!("expected GEOMETRY logical type, got {other:?}"),
        }
        for rg in meta.row_groups() {
            let col = rg
                .columns()
                .iter()
                .find(|c| c.column_descr().name() == "geometry")
                .unwrap();
            let stats = col.geo_statistics().expect("native geo statistics");
            let bb = stats.bounding_box().expect("bbox");
            assert!(bb.get_xmax() >= bb.get_xmin());
        }

        // Loader path: native stats detected as the rg-bbox source, strong
        // clustering, and the file info panel flags 2.0.
        let (_store, crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        let (source, boxes) = rg_meta.expect("native stats");
        assert!(source.contains("geospatial"), "{source}");
        let overlap = crate::data::loader::bbox_overlap_metric(&boxes);
        assert!(overlap < boxes.len() as f64 * 0.25, "clustered: {overlap}");
        assert!(info.geo.version_label.contains("2.0"), "{}", info.geo.version_label);
        assert_rows_consistent(&dst);

        // Downgrade path: re-optimizing a native-GEOMETRY file as 1.1 must
        // strip the logical type (plain WKB) while keeping the CRS.
        let dst11 = dir.join("out_v20_to_v11.parquet");
        let opts11 = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        optimize(&Source::Local(dst.clone()), &dst11, &opts11, None, None, &|_, _| {}).unwrap();
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&dst11).unwrap()).unwrap();
        let geom = b.metadata().row_groups()[0]
            .columns()
            .iter()
            .find(|c| c.column_descr().name() == "geometry")
            .unwrap();
        assert!(
            geom.column_descr().logical_type_ref().is_none(),
            "1.1 export must be plain WKB, got {:?}",
            geom.column_descr().logical_type_ref()
        );
        let (_s, crs11, info11, _rg) =
            crate::data::loader::open_store_for_test(&dst11).unwrap();
        assert_eq!(crs11.epsg, Some(2154));
        assert!(info11.geo.version_label.contains("1.1"), "{}", info11.geo.version_label);
    }

    /// Viewport-only export: only features intersecting the rect survive,
    /// rows stay consistent, and the metadata bbox shrinks to the export.
    #[test]
    fn viewport_only_export() {
        let dir = std::env::temp_dir().join("geopq_optimize_vp");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_vp.parquet");
        // Grid spans 0..10 in both axes; keep the lower-left quadrant.
        let rect = [0.0, 0.0, 4.999, 4.999];
        let opts = OptimizeOptions {
            row_group_size: 2048,
            filter_rect: Some(rect),
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        let quarter = 40_000 / 4;
        assert!(
            (report.rows as i64 - quarter as i64).unsigned_abs() < 500,
            "≈ one quadrant: {}",
            report.rows
        );
        let (store, _crs, info, _rg) = crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.total_rows(), report.rows);
        let b = info.geo.bbox.expect("metadata bbox");
        assert!(b[2] <= 5.1 && b[3] <= 5.1, "bbox shrank to the export: {b:?}");
        assert_rows_consistent(&dst);

        // Empty viewport errors instead of writing a rowless file.
        let err = optimize(
            &Source::Local(src),
            &dir.join("out_empty.parquet"),
            &OptimizeOptions {
                filter_rect: Some([1000.0, 1000.0, 1001.0, 1001.0]),
                ..Default::default()
            },
            None,
            None,
            &|_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("viewport"), "{err}");
    }

    /// GeoArrow points: WKB → coordinate arrays, loader reads them, and the
    /// x/y leaf statistics replace covering/geo stats as the pruning source.
    #[test]
    fn geoarrow_points_roundtrip() {
        let dir = std::env::temp_dir().join("geopq_optimize_ga_pts");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_ga.parquet");
        let opts = OptimizeOptions {
            version: GpVersion::V1_1GeoArrow,
            row_group_size: 2048,
            covering: false, // force the coordinate-stats pruning source
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        assert!(
            report.overlap_frac_after() < 0.25,
            "sorted: {}",
            report.overlap_frac_after()
        );

        let (store, crs, info, rg_meta) = crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        assert_eq!(store.encoding, GeomEncoding::Point);
        assert_eq!(info.geo.encoding, "point");
        let (source, boxes) = rg_meta.expect("rg bboxes from coordinate stats");
        assert!(source.contains("coordinate"), "{source}");
        assert_eq!(boxes.len(), report.rg_after);
        assert_rows_consistent(&dst);

        // Loader builds it: same rows, Point kind, and the mesh path used
        // the GeoArrow fast path (no WKB anywhere).
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (geom, rows, bad) =
            crate::data::loader::build_geometry_for_test(&store, &crs, &display).unwrap();
        assert_eq!((rows, bad), (40_000, 0));
        assert_eq!(geom.kind, crate::data::geometry::GeomKind::Point);

        // Attribute-panel path: a full-row fetch must expose the geometry
        // through the encoding-aware accessor (regression: the panel
        // assumed WKB and showed "<invalid>" on GeoArrow layers).
        let row = store.fetch_row(7).unwrap();
        let g = crate::data::geoarrow::GeomCol::new(
            row.column(store.geom_col).as_ref(),
            store.encoding,
        )
        .and_then(|g| g.geometry(0));
        assert!(
            matches!(g, Some(geo_types::Geometry::Point(_))),
            "geometry accessible from a full-row fetch"
        );

        // And back: GeoArrow → plain 1.1 WKB.
        let dst_back = dir.join("back_wkb.parquet");
        let opts_back = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        optimize(&Source::Local(dst.clone()), &dst_back, &opts_back, None, None, &|_, _| {}).unwrap();
        let (store_b, _crs, info_b, _rg) =
            crate::data::loader::open_store_for_test(&dst_back).unwrap();
        assert_eq!(store_b.encoding, GeomEncoding::Wkb);
        assert!(info_b.geo.encoding.starts_with("WKB"), "{}", info_b.geo.encoding);
        assert_rows_consistent(&dst_back);
    }

    /// GeoArrow polygons with single→multi promotion: a WKB source mixing
    /// Polygon and MultiPolygon becomes one multipolygon-encoded column.
    #[test]
    fn geoarrow_polygon_promotion() {
        use geo_types::{polygon, Geometry, MultiPolygon};
        let dir = std::env::temp_dir().join("geopq_optimize_ga_poly");
        std::fs::create_dir_all(&dir).unwrap();

        // WKB source: squares on a grid, every 5th row a MultiPolygon.
        let n = 5000usize;
        let square = |x0: f64, y0: f64| {
            polygon![(x: x0, y: y0), (x: x0 + 1.0, y: y0), (x: x0 + 1.0, y: y0 + 1.0), (x: x0, y: y0 + 1.0), (x: x0, y: y0)]
        };
        let stride = n / 2 + 1;
        let (mut wkbs, mut ids) = (Vec::new(), Vec::new());
        let wopts = wkb::writer::WriteOptions::default();
        for k in 0..n {
            let i = (k * stride) % n;
            let (gx, gy) = ((i % 71) as f64 * 2.0 - 71.0, (i / 71) as f64 * 1.2);
            let g: Geometry<f64> = if i % 5 == 0 {
                MultiPolygon(vec![square(gx, gy), square(gx + 1.5, gy + 1.5)]).into()
            } else {
                square(gx, gy).into()
            };
            let mut buf = Vec::new();
            wkb::writer::write_geometry(&mut buf, &g, &wopts).unwrap();
            wkbs.push(buf);
            ids.push(i as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
            ],
        )
        .unwrap();
        let src = dir.join("poly_src.parquet");
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon", "MultiPolygon"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let dst = dir.join("poly_ga.parquet");
        let opts = OptimizeOptions {
            version: GpVersion::V1_1GeoArrow,
            row_group_size: 1024,
            ..Default::default()
        };
        optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();

        let (store, _crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.encoding, GeomEncoding::MultiPolygon);
        assert_eq!(info.geo.encoding, "multipolygon");
        assert_eq!(info.geo.geometry_types, vec!["MultiPolygon".to_string()]);
        // Covering written by default → per-feature selection still works.
        assert!(store.covering.is_some());
        let (source, _boxes) = rg_meta.unwrap();
        assert!(source.contains("covering"), "{source}");

        // Every geometry survives promotion: id encodes the grid slot, so
        // the bbox min corner must match the id-derived square origin.
        use geo::BoundingRect;
        let rows: Vec<u32> = (0..5000u32).step_by(97).collect();
        let geoms = store.fetch_geoms(&rows).unwrap();
        let batches = store.fetch(&rows, Some(&[1])).unwrap();
        let mut ids: Vec<i64> = Vec::new();
        for b in &batches {
            let a = Int64Array::from(b.column(0).to_data());
            ids.extend((0..b.num_rows()).map(|i| a.value(i)));
        }
        for ((_, g), id) in geoms.iter().zip(&ids) {
            let g = g.as_ref().unwrap();
            assert!(matches!(g, Geometry::MultiPolygon(_)), "promoted to multi");
            let min = g.bounding_rect().unwrap().min();
            let (gx, gy) = ((id % 71) as f64 * 2.0 - 71.0, (id / 71) as f64 * 1.2);
            assert_eq!((min.x, min.y), (gx, gy), "id {id}");
        }

        // Loader tessellates it end to end (bulk GeoArrow path)...
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (geom, rows_n, bad) =
            crate::data::loader::build_geometry_for_test(&store, &_crs, &display).unwrap();
        assert_eq!((rows_n, bad), (5000, 0));
        assert_eq!(geom.kind, crate::data::geometry::GeomKind::Polygon);

        // ...and produces the same mesh as the per-feature WKB path on the
        // same shapes: identical tessellated area, segments and pick items.
        let mesh_stats = |g: &crate::data::layer::LayerGeometry| {
            let mut area = 0.0f64;
            let mut segs = 0usize;
            for c in g.chunks.iter() {
                for t in c.fill_indices.chunks_exact(3) {
                    let a = c.fill_vertices[t[0] as usize];
                    let b = c.fill_vertices[t[1] as usize];
                    let d = c.fill_vertices[t[2] as usize];
                    area += 0.5
                        * ((b[0] - a[0]) as f64 * (d[1] - a[1]) as f64
                            - (d[0] - a[0]) as f64 * (b[1] - a[1]) as f64)
                            .abs();
                }
                segs += c.lines[0].segments.len();
            }
            (area, segs, g.rtree.size())
        };
        let (src_store, src_crs, _i, _r) =
            crate::data::loader::open_store_for_test(&src).unwrap();
        assert_eq!(src_store.encoding, GeomEncoding::Wkb);
        let (geom_wkb, _rows, _bad) =
            crate::data::loader::build_geometry_for_test(&src_store, &src_crs, &display)
                .unwrap();
        let (a_ga, s_ga, r_ga) = mesh_stats(&geom);
        let (a_wkb, s_wkb, r_wkb) = mesh_stats(&geom_wkb);
        assert_eq!((s_ga, r_ga), (s_wkb, r_wkb), "segment / pick-item counts");
        assert!(
            (a_ga - a_wkb).abs() <= a_wkb * 1e-9,
            "tessellated area differs: {a_ga} vs {a_wkb}"
        );
    }

    /// The covering bbox leaves must be written in small pages so the page
    /// index (ColumnIndex/OffsetIndex) prunes below row-group granularity.
    #[test]
    fn bbox_leaves_get_fine_grained_pages() {
        let dir = std::env::temp_dir().join("geopq_optimize_pages");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(150_000, &dir);
        let dst = dir.join("out_pages.parquet");
        let report =
            optimize(&Source::Local(src.clone()), &dst, &OptimizeOptions::default(), None, None, &|_, _| {})
                .unwrap();
        assert_eq!(report.rg_after, 150_000_usize.div_ceil(65_536));

        let options = parquet::arrow::arrow_reader::ArrowReaderOptions::new()
            .with_page_index_policy(parquet::file::metadata::PageIndexPolicy::Optional);
        let b = ParquetRecordBatchReaderBuilder::try_new_with_options(
            File::open(&dst).unwrap(),
            options,
        )
        .unwrap();
        let leaf_idx = |name: &str| {
            b.parquet_schema()
                .columns()
                .iter()
                .position(|c| c.path().string() == name)
                .unwrap_or_else(|| panic!("leaf {name} not found"))
        };
        let offset_index = b.metadata().offset_index().expect("offset index written");
        // First (full 65k-row) group: expect ~16 pages per bbox leaf
        // (65536 rows / ~4096 rows per 32 KB page), and far fewer for a
        // default-page-size column like the geometry.
        for leaf in ["bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax"] {
            let pages = offset_index[0][leaf_idx(leaf)].page_locations().len();
            assert!(pages >= 8, "{leaf}: {pages} pages, expected fine-grained");
        }
        // Sanity: page row ranges are recoverable via first_row_index.
        let locs = offset_index[0][leaf_idx("bbox.xmin")].page_locations();
        assert_eq!(locs[0].first_row_index, 0);
        assert!(locs[1].first_row_index > 0);
    }

    /// WKB vs GeoArrow decode speed on the real fixtures, opt-in:
    /// cargo test --release geoarrow_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn geoarrow_speed() {
        let dir = std::env::temp_dir().join("geopq_ga_speed");
        std::fs::create_dir_all(&dir).unwrap();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let bench = |label: &str, fixture: &str| {
            let src = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/"))
                .join(fixture);
            if !src.exists() {
                eprintln!("{label}: fixture missing, skipping");
                return;
            }
            // Matched pair: same codec, order, row groups, no covering —
            // the only difference is the geometry encoding.
            let base = OptimizeOptions {
                hilbert_sort: false,
                covering: false,
                ..Default::default()
            };
            let wkb = dir.join(format!("{label}_wkb.parquet"));
            let ga = dir.join(format!("{label}_ga.parquet"));
            optimize(&Source::Local(src.clone()), &wkb, &base, None, None, &|_, _| {}).unwrap();
            let opts = OptimizeOptions {
                version: GpVersion::V1_1GeoArrow,
                ..base
            };
            optimize(&Source::Local(src.clone()), &ga, &opts, None, None, &|_, _| {}).unwrap();
            let time = |path: &std::path::PathBuf| {
                let (store, crs, _i, _r) =
                    crate::data::loader::open_store_for_test(path).unwrap();
                let t = std::time::Instant::now();
                let (_g, rows, _b) =
                    crate::data::loader::build_geometry_for_test(&store, &crs, &display)
                        .unwrap();
                (t.elapsed().as_millis(), rows)
            };
            let size = |p: &std::path::PathBuf| std::fs::metadata(p).unwrap().len() / (1 << 20);
            let (wkb_ms, rows) = time(&wkb);
            let (ga_ms, rows2) = time(&ga);
            assert_eq!(rows, rows2);
            eprintln!(
                "{label}: {rows} rows — load: WKB {wkb_ms} ms vs GeoArrow {ga_ms} ms ({:.2}x) — size: {} MB vs {} MB",
                wkb_ms as f64 / ga_ms as f64,
                size(&wkb),
                size(&ga),
            );
        };
        bench("points_3m75", "points_3m75.parquet");
        bench("parcels", "parcels_hilbert.parquet");
    }

    /// Real-file benchmark, opt-in:
    /// GEOPQ_OPT_FILE=... cargo test --release optimize_real_file -- --ignored --nocapture
    #[test]
    #[ignore]
    fn optimize_real_file() {
        let Ok(src) = std::env::var("GEOPQ_OPT_FILE") else {
            return;
        };
        let src = std::path::PathBuf::from(src);
        let dst = std::env::temp_dir().join("geopq_opt_real.parquet");
        for version in [GpVersion::V1_1, GpVersion::V2_0] {
            let opts = OptimizeOptions {
                version,
                covering: version == GpVersion::V1_1,
                ..Default::default()
            };
            let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
            eprintln!("{report:#?}");
            eprintln!(
                "overlap fraction: {:.0}% -> {:.0}%",
                report.overlap_frac_before() * 100.0,
                report.overlap_frac_after() * 100.0
            );
            // Normalized: raw overlap counts aren't comparable when the
            // row-group count changes.
            assert!(report.overlap_frac_after() <= report.overlap_frac_before());
            let (_store, _crs, info, rg_meta) =
                crate::data::loader::open_store_for_test(&dst).unwrap();
            let (source, boxes) = rg_meta.expect("metadata bboxes on output");
            eprintln!(
                "loader sees: {} ({} boxes), version: {}",
                source,
                boxes.len(),
                info.geo.version_label
            );
        }
        let _ = std::fs::remove_file(&dst);
    }

    /// Exhaustive check on a small grid: xy2d must be a bijection and
    /// consecutive curve positions must be grid-adjacent (unit steps).
    #[test]
    fn hilbert_is_a_space_filling_curve() {
        const ORDER: u32 = 4;
        let n = 1u32 << ORDER;
        let mut cell_of = vec![None; (n * n) as usize];
        for x in 0..n {
            for y in 0..n {
                let d = hilbert_xy2d(ORDER, x, y);
                assert!(d < (n * n) as u64, "code {d} out of range");
                assert!(cell_of[d as usize].is_none(), "duplicate code {d}");
                cell_of[d as usize] = Some((x, y));
            }
        }
        for w in cell_of.windows(2) {
            let (x0, y0) = w[0].unwrap();
            let (x1, y1) = w[1].unwrap();
            let step = x0.abs_diff(x1) + y0.abs_diff(y1);
            assert_eq!(step, 1, "curve jumps from ({x0},{y0}) to ({x1},{y1})");
        }
    }
}
