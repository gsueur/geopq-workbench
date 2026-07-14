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
pub fn optimize(
    src: &Source,
    dst: &Path,
    opts: &OptimizeOptions,
    epsg_hint: Option<u32>,
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
    let primary = geo_meta
        .as_ref()
        .and_then(|m| m.get("primary_column"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| guess_geom_column(&src_schema).unwrap_or_else(|| "geometry".into()));
    let geom_idx = src_schema
        .index_of(&primary)
        .map_err(|_| format!("geometry column '{primary}' not found"))?;
    let src_encoding = geo_meta
        .as_ref()
        .and_then(|m| m.get("columns")?.get(&primary)?.get("encoding")?.as_str())
        .map(|e| GeomEncoding::parse(e).ok_or_else(|| format!("encoding '{e}' not supported")))
        .transpose()?
        .unwrap_or_default();

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

    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut row_bboxes: Vec<Option<[f64; 4]>> = Vec::with_capacity(total_rows);
    let mut geom_types: HashSet<&'static str> = HashSet::new();
    for res in reader {
        let batch = res.map_err(|e| format!("parquet decode error: {e}"))?;
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

    // --- sort order ---
    progress(0.52, "sorting (Hilbert)");
    let mut order: Vec<u32> = (0..rows as u32).collect();
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

    let out_schema = Arc::new(Schema::new(fields));

    // --- writer properties ---
    let mut props = WriterProperties::builder()
        .set_compression(opts.codec.compression())
        .set_max_row_group_row_count(Some(opts.row_group_size))
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_created_by(format!("geopq-viewer {}", env!("CARGO_PKG_VERSION")));
    if write_covering {
        // Small pages on the bbox leaves (~4k rows at 8 B/value) give the
        // page index sub-row-group granularity, so readers can prune at
        // page level instead of whole row groups. Dictionary encoding is
        // disabled there: coordinates are mostly unique (no dict win) and
        // the page-size cap applies to the encoded size, which tiny dict
        // indices would defeat. Other columns keep the defaults.
        for leaf in ["xmin", "ymin", "xmax", "ymax"] {
            let path = ColumnPath::new(vec!["bbox".into(), leaf.into()]);
            props = props
                .set_column_data_page_size_limit(path.clone(), BBOX_LEAF_PAGE_BYTES)
                .set_column_dictionary_enabled(path, false);
        }
    }
    let mut bloom_columns: Vec<String> = Vec::new();
    match opts.bloom {
        BloomMode::Preserve => {
            for parts in &src_bloom {
                if parts.first().map(String::as_str) == drop_covering {
                    continue; // rebuilt bbox column gets no bloom filter
                }
                bloom_columns.push(parts.join("."));
                props = props
                    .set_column_bloom_filter_enabled(ColumnPath::new(parts.clone()), true)
                    .set_column_bloom_filter_fpp(ColumnPath::new(parts.clone()), 0.01);
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
                    let path = ColumnPath::new(vec![f.name().clone()]);
                    bloom_columns.push(f.name().clone());
                    props = props
                        .set_column_bloom_filter_enabled(path.clone(), true)
                        .set_column_bloom_filter_fpp(path, 0.01);
                }
            }
        }
        BloomMode::None => {}
    }

    // --- write, gathering rows in sorted order ---
    progress(0.55, "writing");
    let out_file = File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props.build()))
        .map_err(|e| format!("writer init: {e}"))?;
    // `geo` is rebuilt; other source key-value metadata passes through.
    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        build_geo_meta(opts, &primary, crs_value.as_ref(), &geom_types, out_encoding, file_bbox)
            .to_string(),
    ));
    for entry in kv.iter().filter(|kv| kv.key != "geo" && kv.key != "ARROW:schema") {
        writer.append_key_value_metadata(entry.clone());
    }

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
    let mut written = 0usize;
    for chunk in order.chunks(chunk_rows) {
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
        let out = RecordBatch::try_new(out_schema.clone(), cols)
            .map_err(|e| format!("batch assembly failed: {e}"))?;
        writer.write(&out).map_err(|e| format!("write failed: {e}"))?;
        written += chunk.len();
        progress(0.55 + 0.45 * (written as f32 / rows as f32), "writing");
    }
    let closed = writer.close().map_err(|e| format!("finalize failed: {e}"))?;

    // --- report ---
    let rg_after_rows: Vec<usize> = closed
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows().max(0) as usize)
        .collect();
    let overlap_after = {
        let mut boxes = Vec::with_capacity(rg_after_rows.len());
        let mut off = 0usize;
        for n in &rg_after_rows {
            boxes.extend(union_bboxes(
                order[off..off + n].iter().flat_map(|&r| &row_bboxes[r as usize]),
            ));
            off += n;
        }
        super::loader::bbox_overlap_metric(&boxes)
    };

    Ok(OptimizeReport {
        rows: rows as u64,
        size_before: src.size(),
        size_after: std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0),
        rg_before,
        rg_after: rg_after_rows.len(),
        overlap_before,
        overlap_after,
        bloom_columns,
        version_label: opts.version.label().into(),
        elapsed_ms: t0.elapsed().as_millis() as u64,
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
        Some(LogicalType::Geometry(g)) => g.crs.clone(),
        Some(LogicalType::Geography(g)) => g.crs.clone(),
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
    use crate::data::loader::parse_wkb_point_2d;
    use arrow::array::{BinaryArray, Int64Array, StringArray};
    use parquet::file::properties::WriterProperties;

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
            .set_max_row_group_size(2048)
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
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, &|_, _| {}).unwrap();
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
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, &|_, _| {}).unwrap();
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
            Some(parquet::basic::LogicalType::Geometry(g)) => {
                let crs = g.crs.as_deref().expect("crs recorded");
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
        optimize(&Source::Local(dst.clone()), &dst11, &opts11, None, &|_, _| {}).unwrap();
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
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, &|_, _| {}).unwrap();
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

        // And back: GeoArrow → plain 1.1 WKB.
        let dst_back = dir.join("back_wkb.parquet");
        let opts_back = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        optimize(&Source::Local(dst.clone()), &dst_back, &opts_back, None, &|_, _| {}).unwrap();
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
        optimize(&Source::Local(src.clone()), &dst, &opts, None, &|_, _| {}).unwrap();

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

        // Loader tessellates it end to end.
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (geom, rows_n, bad) =
            crate::data::loader::build_geometry_for_test(&store, &_crs, &display).unwrap();
        assert_eq!((rows_n, bad), (5000, 0));
        assert_eq!(geom.kind, crate::data::geometry::GeomKind::Polygon);
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
            optimize(&Source::Local(src.clone()), &dst, &OptimizeOptions::default(), None, &|_, _| {})
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
        let mut bench = |label: &str, fixture: &str| {
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
            optimize(&Source::Local(src.clone()), &wkb, &base, None, &|_, _| {}).unwrap();
            let opts = OptimizeOptions {
                version: GpVersion::V1_1GeoArrow,
                ..base
            };
            optimize(&Source::Local(src.clone()), &ga, &opts, None, &|_, _| {}).unwrap();
            let mut time = |path: &std::path::PathBuf| {
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
            let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, &|_, _| {}).unwrap();
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
