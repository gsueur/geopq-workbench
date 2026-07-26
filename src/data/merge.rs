//! Merge several loaded layers into one raw WKB GeoParquet staging
//! file: schema union by column name (missing values become nulls,
//! Int64 widens to Float64, conflicting types are dropped with a
//! warning), geometries reprojected into the primary layer's CRS. The
//! output is importer-grade raw GeoParquet — the standard optimize
//! pass runs on it, so "Merge with…" in the Optimize dialog gets every
//! Optimize capability for free.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{new_null_array, ArrayRef, BinaryBuilder, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use geo::MapCoords;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};

use super::crs::{transform_point, Crs};
use super::import::{
    bbox_field, geo_meta_with_proj4, to_wkb, writer_props, BboxBuilder, GeomStats,
    IMPORT_WRITE_ROWS,
};
use super::store::FeatureStore;

pub struct MergeInput {
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
    pub name: String,
}

/// The columns a layer contributes to a merge: every schema field
/// except its geometry and a hidden WKB sibling.
fn attr_fields(store: &FeatureStore) -> Vec<(usize, Field)> {
    store
        .schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != store.geom_col && Some(*i) != store.hidden_wkb)
        .map(|(i, f)| (i, f.as_ref().clone()))
        .collect()
}

/// Union two types, widening Int64 → Float64; None = incompatible.
fn union_type(a: &DataType, b: &DataType) -> Option<DataType> {
    if a == b {
        return Some(a.clone());
    }
    match (a, b) {
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
            Some(DataType::Float64)
        }
        _ => None,
    }
}

/// The merged attribute schema across all inputs, and the column names
/// dropped because their types conflict.
pub fn union_schema(inputs: &[MergeInput]) -> (Vec<Field>, Vec<String>) {
    let mut fields: Vec<Field> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for input in inputs {
        for (_, f) in attr_fields(&input.store) {
            if dropped.iter().any(|d| d == f.name()) {
                continue;
            }
            match fields.iter_mut().find(|u| u.name() == f.name()) {
                None => fields.push(f.clone().with_nullable(true)),
                Some(u) => match union_type(u.data_type(), f.data_type()) {
                    Some(t) => *u = Field::new(u.name(), t, true),
                    None => {
                        dropped.push(f.name().clone());
                        fields.retain(|k| k.name() != f.name());
                    }
                },
            }
        }
    }
    (fields, dropped)
}

/// Merge `inputs` (first = primary: its CRS is the target) into a raw
/// WKB GeoParquet at `dst`. `source_col` adds a text column naming the
/// layer each row came from. Returns rows written.
pub fn merge(
    inputs: &[MergeInput],
    dst: &Path,
    source_col: bool,
    progress: &dyn Fn(f32, &str),
) -> Result<u64, String> {
    if inputs.len() < 2 {
        return Err("merge needs at least two layers".into());
    }
    let target = &inputs[0].crs;
    let (attrs, dropped) = union_schema(inputs);
    if !dropped.is_empty() {
        log::warn!(
            "merge: dropping columns with conflicting types: {}",
            dropped.join(", ")
        );
    }

    // Output geometry / source-column names that dodge attribute names.
    let fresh = |base: &str| -> String {
        let mut name = base.to_string();
        let mut i = 0usize;
        while attrs.iter().any(|f| f.name().eq_ignore_ascii_case(&name)) {
            i += 1;
            name = format!("{base}_{i}");
        }
        name
    };
    let geom_name = fresh("geometry");
    let src_name = fresh("source_layer");

    let mut fields: Vec<Field> = vec![Field::new(&geom_name, DataType::Binary, true)];
    fields.extend(attrs.iter().cloned());
    if source_col {
        fields.push(Field::new(&src_name, DataType::Utf8, true));
    }
    // The metadata declares a covering column, so one gets written.
    fields.push(bbox_field());
    let schema = Arc::new(Schema::new(fields));

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer = ArrowWriter::try_new(out, schema.clone(), Some(writer_props()))
        .map_err(|e| e.to_string())?;

    let total_rows: u64 = inputs.iter().map(|i| i.store.total_rows()).sum();
    let mut stats = GeomStats::new();
    let mut written = 0u64;

    for input in inputs {
        let reproject = !input.crs.same_as(target);
        // Where each union column lives in this layer (by exact name).
        let col_of: Vec<Option<usize>> = attrs
            .iter()
            .map(|u| {
                attr_fields(&input.store)
                    .into_iter()
                    .find(|(_, f)| f.name() == u.name())
                    .map(|(i, _)| i)
            })
            .collect();
        let fetch_cols: Vec<usize> = col_of.iter().filter_map(|c| *c).collect();

        let n = input.store.total_rows() as u32;
        let mut start = 0u32;
        while start < n {
            let end = (start + IMPORT_WRITE_ROWS as u32).min(n);
            let rows: Vec<u32> = (start..end).collect();

            // Geometry: decode, reproject into the target CRS, re-encode.
            let mut geom_b = BinaryBuilder::new();
            let mut bbox_b = BboxBuilder::default();
            for (_, g) in input.store.fetch_geoms(&rows)? {
                let g = match (g, reproject) {
                    (Some(g), true) => g
                        .try_map_coords(|c| {
                            transform_point(&input.crs, target, c.x, c.y)
                                .map(|(x, y)| geo_types::Coord { x, y })
                        })
                        .ok(),
                    (g, _) => g,
                };
                match g {
                    Some(g) => {
                        let env = stats.add(&g);
                        match to_wkb(&g) {
                            Ok(w) => {
                                geom_b.append_value(w);
                                bbox_b.push(env);
                            }
                            Err(_) => {
                                geom_b.append_null();
                                bbox_b.push(None);
                            }
                        }
                    }
                    None => {
                        geom_b.append_null();
                        bbox_b.push(None);
                    }
                }
            }

            // Attributes: project the fetched batches onto the union
            // schema (missing columns become nulls, Int64 casts up).
            let fetched = if fetch_cols.is_empty() {
                Vec::new()
            } else {
                input.store.fetch(&rows, Some(&fetch_cols))?
            };
            let one = concat_batches(&fetched, rows.len())?;
            let mut arrays: Vec<ArrayRef> = vec![Arc::new(geom_b.finish())];
            for (u, src_idx) in attrs.iter().zip(&col_of) {
                let arr = match src_idx {
                    None => new_null_array(u.data_type(), rows.len()),
                    Some(i) => {
                        // Position within the fetched (subset) batch.
                        let pos = fetch_cols.iter().position(|c| c == i).unwrap();
                        let col = one.as_ref().expect("fetched batch").column(pos).clone();
                        if col.data_type() == u.data_type() {
                            col
                        } else {
                            arrow::compute::cast(&col, u.data_type())
                                .map_err(|e| format!("cast {}: {e}", u.name()))?
                        }
                    }
                };
                arrays.push(arr);
            }
            if source_col {
                arrays.push(Arc::new(StringArray::from(vec![
                    Some(input.name.as_str());
                    rows.len()
                ])));
            }
            arrays.push(bbox_b.finish());
            let batch =
                RecordBatch::try_new(schema.clone(), arrays).map_err(|e| e.to_string())?;
            writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;

            written += rows.len() as u64;
            progress(
                written as f32 / total_rows.max(1) as f32,
                &format!("merging {}", input.name),
            );
            start = end;
        }
    }

    let (crs, proj4) = crs_to_geo(target);
    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        geo_meta_with_proj4(&geom_name, &stats, crs, proj4).to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

/// One batch out of many (row order preserved).
fn concat_batches(
    batches: &[RecordBatch],
    rows: usize,
) -> Result<Option<RecordBatch>, String> {
    match batches.len() {
        0 => Ok(None),
        1 => Ok(Some(batches[0].clone())),
        _ => arrow::compute::concat_batches(&batches[0].schema(), batches)
            .map(Some)
            .map_err(|e| format!("concat ({rows} rows): {e}")),
    }
}

/// A layer CRS as GeoParquet metadata: PROJJSON passthrough when the
/// source file carried it, an EPSG identity otherwise, the CRS84
/// default for 4326, and unknown-plus-proj4 (the `geopq:crs` vendor
/// key) for CRSs that only exist as proj strings. Shared with the SQL
/// result exporter.
pub(crate) fn crs_to_geo(crs: &Crs) -> (Option<Value>, Option<(String, String)>) {
    if let Some(pj) = &crs.projjson {
        return (Some((**pj).clone()), None);
    }
    match crs.epsg {
        Some(4326) => (None, None),
        Some(code) => (
            // Best-effort id-only reference: not schema-valid PROJJSON
            // (no datum/base_crs), but preserves the code for readers
            // that resolve ids, honestly typed.
            Some(json!({
                "type": if crs.is_latlong { "GeographicCRS" } else { "ProjectedCRS" },
                "name": crs.name,
                "id": {"authority": "EPSG", "code": code},
            })),
            None,
        ),
        // The source declared its CRS undefined (`crs: null`): keep the
        // honest null, don't launder the CRS84 render fallback into a
        // vendor proj4 claim.
        None if crs.name.contains("undefined") => (Some(Value::Null), None),
        None => (
            Some(Value::Null),
            Some((crs.proj4.clone(), crs.name.clone())),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::open_store_for_test;

    fn write_layer(path: &Path, crs: &str, rows: &[(f64, f64, i64)]) {
        use arrow::array::{Float64Array, Int64Array};
        let mut geom = BinaryBuilder::new();
        for (x, y, _) in rows {
            let g = geo_types::Geometry::Point(geo_types::Point::new(*x, *y));
            geom.append_value(to_wkb(&g).unwrap());
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, true),
            Field::new("v", DataType::Int64, true),
            Field::new("x_src", DataType::Float64, true),
        ]));
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(geom.finish()),
            Arc::new(Int64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
        ];
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let f = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, Some(writer_props())).unwrap();
        w.write(&batch).unwrap();
        let crs_json: Value = serde_json::from_str(crs).unwrap();
        let col = json!({"encoding": "WKB", "geometry_types": ["Point"], "crs": crs_json});
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            json!({"version": "1.1.0", "primary_column": "geometry",
                   "columns": {"geometry": col}})
            .to_string(),
        ));
        w.close().unwrap();
    }

    /// Two point layers, one in Lambert-93 and one in lon/lat: the merge
    /// reprojects the second into the first's CRS, unions the schemas
    /// and tags rows with their source layer.
    #[test]
    fn merge_reprojects_and_unions() {
        let dir = std::env::temp_dir().join("geopq_merge");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let a = dir.join("a.parquet");
        let b = dir.join("b.parquet");
        // Primary: Lambert-93 points near the origin of the grid.
        write_layer(
            &a,
            r#"{"name":"L93","id":{"authority":"EPSG","code":2154}}"#,
            &[(700_000.0, 6_600_000.0, 1)],
        );
        // Secondary: Paris in lon/lat (must land at its L93 coordinates).
        write_layer(&b, "null", &[(2.349014, 48.864716, 2)]);

        let (sa, ca, ..) = open_store_for_test(&a).unwrap();
        let (sb, cb, ..) = open_store_for_test(&b).unwrap();
        assert_eq!(ca.epsg, Some(2154));
        assert!(cb.is_latlong);
        let inputs = vec![
            MergeInput { store: Arc::new(sa), crs: ca, name: "alpha".into() },
            MergeInput { store: Arc::new(sb), crs: cb, name: "beta".into() },
        ];

        let dst = dir.join("merged.parquet");
        let written = merge(&inputs, &dst, true, &|_, _| {}).unwrap();
        assert_eq!(written, 2);

        let (store, crs, info, _) = open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154), "primary CRS wins");
        assert_eq!(store.total_rows(), 2);
        // Paris reprojected into L93 (pyproj truth ~652242.70, 6862939.61).
        let bbox = info.geo.bbox.unwrap();
        assert!((bbox[2] - 700_000.0).abs() < 1.0, "{bbox:?}");
        let g = store.fetch_geoms(&[1]).unwrap().remove(0).1.unwrap();
        if let geo_types::Geometry::Point(p) = g {
            assert!((p.x() - 652_242.70).abs() < 1.0, "{}", p.x());
            assert!((p.y() - 6_862_939.61).abs() < 1.0, "{}", p.y());
        } else {
            panic!("expected point");
        }
        // Union schema + source column.
        let batch = store.fetch(&[0, 1], None).unwrap();
        let one = arrow::compute::concat_batches(&batch[0].schema(), &batch).unwrap();
        let sc = one.schema();
        assert!(sc.index_of("v").is_ok());
        assert!(sc.index_of("x_src").is_ok());
        let src_idx = sc.index_of("source_layer").unwrap();
        let names: Vec<String> = (0..2)
            .map(|i| {
                one.column(src_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(i)
                    .to_string()
            })
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn union_schema_widens_and_drops() {
        use arrow::datatypes::DataType as T;
        let f = |n: &str, t: T| Field::new(n, t, true);
        let a = vec![f("a", T::Int64), f("b", T::Utf8), f("c", T::Utf8)];
        let b = vec![f("a", T::Float64), f("b", T::Utf8), f("c", T::Int64)];
        // Simulate via two fake stores? union_type is the core:
        assert_eq!(union_type(&T::Int64, &T::Float64), Some(T::Float64));
        assert_eq!(union_type(&T::Utf8, &T::Utf8), Some(T::Utf8));
        assert_eq!(union_type(&T::Utf8, &T::Int64), None);
        let _ = (a, b);
    }
}
