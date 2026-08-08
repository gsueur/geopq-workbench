//! File Geodatabase import, behind `--features gdal-import`.
//!
//! Unlike every other importer in this module, this one needs a real
//! GDAL install: FileGDB is Esri's proprietary format, undocumented
//! outside GDAL's reverse-engineered `OpenFileGDB` driver, and no
//! pure-Rust reader exists. Off by default — the app's whole premise is
//! opening GeoParquet with no GDAL — this is the one opt-in exception.
//!
//! Same output contract as the other importers: plain WKB GeoParquet
//! 1.1 in source order, with a covering bbox column, so the quality
//! scorecard and Optimize take it from there.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use gdal::vector::{Feature, FieldValue, LayerAccess, OGRFieldType};
use parquet::arrow::ArrowWriter;
use serde_json::json;

use super::gpkg::wkb_type_name;
use super::import::{
    bbox_field, AttrBuilder, BboxBuilder, Cell, GdbLayer, IMPORT_WRITE_ROWS as BATCH_ROWS,
};

fn open_ro(path: &Path) -> Result<gdal::Dataset, String> {
    gdal::Dataset::open(path).map_err(|e| format!("cannot open File Geodatabase: {e}"))
}

/// EPSG code + display name of a layer's spatial reference, when GDAL
/// can resolve one.
fn epsg_of(layer: &gdal::vector::Layer<'_>) -> (Option<u32>, String) {
    match layer.spatial_ref() {
        Some(sr) => {
            let epsg = match (sr.auth_name(), sr.auth_code()) {
                (Some(a), Ok(c)) if a.eq_ignore_ascii_case("EPSG") => u32::try_from(c).ok(),
                _ => None,
            };
            let name = sr.name().unwrap_or_else(|| "unknown SRS".to_string());
            (epsg, name)
        }
        None => (None, "no SRS".to_string()),
    }
}

/// Feature layers of a File Geodatabase, with row counts and CRS.
/// Non-spatial tables (no geometry field) are skipped: this importer
/// only produces GeoParquet layers.
pub fn list_layers(path: &Path) -> Result<Vec<GdbLayer>, String> {
    let ds = open_ro(path)?;
    let n = ds.layer_count();
    let mut out = Vec::new();
    for i in 0..n {
        let layer = ds.layer(i).map_err(|e| format!("layer {i}: {e}"))?;
        if layer.defn().geom_fields().next().is_none() {
            continue; // non-spatial table: not something this dialog produces
        }
        let name = layer.name();
        let rows = layer.feature_count();
        let (epsg, srs_name) = epsg_of(&layer);
        out.push(GdbLayer { name, rows, epsg, srs_name });
    }
    if out.is_empty() {
        return Err("no feature layers in this File Geodatabase".into());
    }
    Ok(out)
}

/// GDAL field type -> arrow type. Lists and dates stringify; the shared
/// `AttrBuilder`/`Cell` machinery coerces the rest.
fn arrow_type(t: OGRFieldType::Type) -> DataType {
    match t {
        OGRFieldType::OFTInteger | OGRFieldType::OFTInteger64 => DataType::Int64,
        OGRFieldType::OFTReal => DataType::Float64,
        OGRFieldType::OFTBinary => DataType::Binary,
        _ => DataType::Utf8, // String, Date, DateTime, Time, *List, ...
    }
}

/// GDAL field value -> shared import cell. Fields GDAL fails to decode
/// (bad binary, unhandled type) come back `None` from `Feature::field`
/// and become nulls rather than aborting the whole import.
fn cell(v: Option<FieldValue>) -> Cell<'static> {
    use std::borrow::Cow;
    match v {
        None => Cell::Null,
        Some(FieldValue::IntegerValue(i)) => Cell::Int(i as i64),
        Some(FieldValue::Integer64Value(i)) => Cell::Int(i),
        Some(FieldValue::RealValue(f)) => Cell::Float(f),
        Some(FieldValue::StringValue(s)) => Cell::Str(Cow::Owned(s)),
        Some(FieldValue::DateValue(d)) => Cell::Str(Cow::Owned(d.format("%Y-%m-%d").to_string())),
        Some(FieldValue::DateTimeValue(dt)) => {
            Cell::Str(Cow::Owned(dt.format("%Y-%m-%dT%H:%M:%S").to_string()))
        }
        Some(FieldValue::IntegerListValue(v)) => Cell::Str(Cow::Owned(format!("{v:?}"))),
        Some(FieldValue::Integer64ListValue(v)) => Cell::Str(Cow::Owned(format!("{v:?}"))),
        Some(FieldValue::RealListValue(v)) => Cell::Str(Cow::Owned(format!("{v:?}"))),
        Some(FieldValue::StringListValue(v)) => Cell::Str(Cow::Owned(v.join(", "))),
    }
}

/// Convert one layer to `dst` as GeoParquet 1.1 (plain WKB). Returns the
/// number of rows written.
pub fn convert(
    src: &Path,
    layer_info: &GdbLayer,
    dst: &Path,
    progress: &dyn Fn(f32),
) -> Result<u64, String> {
    let ds = open_ro(src)?;
    let mut layer =
        ds.layer_by_name(&layer_info.name).map_err(|e| format!("open layer: {e}"))?;

    let defn = layer.defn();
    let geom_name = defn
        .geom_fields()
        .next()
        .map(|g| g.name())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "geometry".to_string());
    let cols: Vec<(String, DataType)> =
        defn.fields().map(|f| (f.name(), arrow_type(f.field_type()))).collect();

    let mut fields: Vec<Field> = vec![Field::new(&geom_name, DataType::Binary, true)];
    fields.extend(cols.iter().map(|(n, dt)| Field::new(n, dt.clone(), true)));
    fields.push(bbox_field());
    let schema = Arc::new(Schema::new(fields));

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer =
        ArrowWriter::try_new(out, schema.clone(), Some(super::import::writer_props()))
            .map_err(|e| e.to_string())?;

    let mut geom_types: std::collections::BTreeSet<String> = Default::default();
    let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut written = 0u64;
    let total = layer_info.rows.max(1);

    let mut geom_b = BinaryBuilder::new();
    let mut bbox_b = BboxBuilder::default();
    let mut attr_b: Vec<AttrBuilder> = cols.iter().map(|(_, dt)| AttrBuilder::new(dt)).collect();
    let mut in_batch = 0usize;

    let flush = |geom_b: &mut BinaryBuilder,
                      bbox_b: &mut BboxBuilder,
                      attr_b: &mut Vec<AttrBuilder>,
                      in_batch: &mut usize,
                      writer: &mut ArrowWriter<std::fs::File>|
     -> Result<(), String> {
        if *in_batch == 0 {
            return Ok(());
        }
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(geom_b.finish())];
        arrays.extend(attr_b.iter_mut().map(AttrBuilder::finish));
        arrays.push(bbox_b.finish());
        let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;
        *in_batch = 0;
        Ok(())
    };

    for feature in layer.features() {
        push_feature(&feature, &mut geom_b, &mut bbox_b, &mut geom_types, &mut bbox);
        for (i, b) in attr_b.iter_mut().enumerate() {
            b.push(cell(feature.field(i).unwrap_or(None)));
        }
        in_batch += 1;
        written += 1;
        if in_batch >= BATCH_ROWS {
            flush(&mut geom_b, &mut bbox_b, &mut attr_b, &mut in_batch, &mut writer)?;
            progress((written as f32 / total as f32).min(1.0));
        }
    }
    flush(&mut geom_b, &mut bbox_b, &mut attr_b, &mut in_batch, &mut writer)?;
    progress(1.0);

    let mut col = json!({
        "encoding": "WKB",
        "geometry_types": geom_types.iter().collect::<Vec<_>>(),
    });
    match layer_info.epsg {
        Some(4326) | None => {}
        Some(code) => {
            col["crs"] = json!({
                "name": layer_info.srs_name,
                "id": {"authority": "EPSG", "code": code},
            });
        }
    }
    if bbox[0].is_finite() && bbox[2].is_finite() {
        col["bbox"] = json!([bbox[0], bbox[1], bbox[2], bbox[3]]);
    }
    col["covering"] = json!({"bbox": {
        "xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
        "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"],
    }});
    let geo = json!({
        "version": "1.1.0",
        "primary_column": geom_name,
        "columns": { &geom_name: col },
    });
    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        geo.to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

/// Encode one feature's geometry as ISO WKB, tracking its type and
/// envelope; null geometries survive as null WKB + null bbox.
fn push_feature(
    feature: &Feature<'_>,
    geom_b: &mut BinaryBuilder,
    bbox_b: &mut BboxBuilder,
    geom_types: &mut std::collections::BTreeSet<String>,
    bbox: &mut [f64; 4],
) {
    let Some(geom) = feature.geometry() else {
        geom_b.append_null();
        bbox_b.push(None);
        return;
    };
    let Ok(wkb) = geom.iso_wkb() else {
        geom_b.append_null();
        bbox_b.push(None);
        return;
    };
    if let Some(n) = wkb_type_name(&wkb) {
        geom_types.insert(n);
    }
    geom_b.append_value(&wkb);
    let env = geom.envelope();
    let e = [env.MinX, env.MinY, env.MaxX, env.MaxY];
    if e.iter().all(|v: &f64| v.is_finite()) {
        bbox[0] = bbox[0].min(e[0]);
        bbox[1] = bbox[1].min(e[1]);
        bbox[2] = bbox[2].max(e[2]);
        bbox[3] = bbox[3].max(e[3]);
        bbox_b.push(Some(e));
    } else {
        bbox_b.push(None);
    }
}
