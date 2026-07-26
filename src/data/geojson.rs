//! GeoJSON import: FeatureCollection (or a single Feature / bare
//! geometry) to plain WKB GeoParquet. serde_json only — no GDAL.
//!
//! Property columns are inferred across all features: integers widen to
//! floats when mixed, anything else mixed falls back to strings, and
//! nested objects/arrays are kept as raw JSON strings. GeoJSON is WGS84
//! by spec, so no CRS is written (the GeoParquet default, CRS84).

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};

use super::import::{
    bbox_field, geo_meta, to_wkb, AttrBuilder, BboxBuilder, Cell, GeomStats,
    IMPORT_WRITE_ROWS,
};

/// Convert a GeoJSON file to GeoParquet. Returns the rows written.
pub fn convert(src: &Path, dst: &Path, progress: &dyn Fn(f32)) -> Result<u64, String> {
    let bytes = std::fs::read(src).map_err(|e| format!("cannot read file: {e}"))?;
    let root: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON: {e}"))?;
    drop(bytes);
    progress(0.15);

    let wrapped;
    let feats: Vec<&Value> = match root.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => root
            .get("features")
            .and_then(Value::as_array)
            .ok_or("FeatureCollection without a features array")?
            .iter()
            .collect(),
        Some("Feature") => vec![&root],
        Some(_) => {
            // Bare geometry: wrap it as a single feature.
            wrapped = json!({"type": "Feature", "geometry": root, "properties": {}});
            vec![&wrapped]
        }
        None => return Err("not GeoJSON: no \"type\" member".into()),
    };
    if feats.is_empty() {
        return Err("empty FeatureCollection".into());
    }

    // --- property schema, inferred across all features ---
    #[derive(Clone, Copy, PartialEq)]
    enum K {
        Int,
        Float,
        Bool,
        Str,
    }
    let mut order: Vec<String> = Vec::new();
    let mut kinds: std::collections::HashMap<String, K> = Default::default();
    for f in &feats {
        let Some(props) = f.get("properties").and_then(Value::as_object) else {
            continue;
        };
        for (k, v) in props {
            let kv = match v {
                Value::Null => continue,
                Value::Bool(_) => K::Bool,
                Value::Number(n) => {
                    if n.is_i64() || n.is_u64() {
                        K::Int
                    } else {
                        K::Float
                    }
                }
                _ => K::Str, // strings, plus nested values as raw JSON
            };
            match kinds.entry(k.clone()) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    order.push(k.clone());
                    e.insert(kv);
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let merged = match (*e.get(), kv) {
                        (a, b) if a == b => a,
                        (K::Int, K::Float) | (K::Float, K::Int) => K::Float,
                        _ => K::Str,
                    };
                    e.insert(merged);
                }
            }
        }
    }
    let cols: Vec<(String, DataType)> = order
        .into_iter()
        .map(|name| {
            let dt = match kinds[&name] {
                K::Int => DataType::Int64,
                K::Float => DataType::Float64,
                K::Bool => DataType::Boolean,
                K::Str => DataType::Utf8,
            };
            (name, dt)
        })
        .collect();

    // The conventional name, bumped if a property claims it.
    let geom_name = {
        let mut name = "geometry".to_string();
        let mut i = 0usize;
        while cols.iter().any(|(n, _)| n == &name) {
            i += 1;
            name = format!("geometry_{i}");
        }
        name
    };
    let mut fields: Vec<Field> = vec![Field::new(&geom_name, DataType::Binary, true)];
    fields.extend(cols.iter().map(|(n, dt)| Field::new(n, dt.clone(), true)));
    fields.push(bbox_field());
    let schema = Arc::new(Schema::new(fields));

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer = ArrowWriter::try_new(out, schema.clone(), Some(super::import::writer_props()))
        .map_err(|e| e.to_string())?;

    let mut stats = GeomStats::new();
    let mut written = 0u64;
    for chunk in feats.chunks(IMPORT_WRITE_ROWS) {
        let mut geom_b = BinaryBuilder::new();
        let mut bbox_b = BboxBuilder::default();
        let mut attr_b: Vec<AttrBuilder> =
            cols.iter().map(|(_, dt)| AttrBuilder::new(dt)).collect();
        for f in chunk {
            match f.get("geometry") {
                Some(Value::Null) | None => {
                    geom_b.append_null();
                    bbox_b.push(None);
                }
                Some(g) => {
                    let g = parse_geometry(g, 0)?;
                    bbox_b.push(stats.add(&g));
                    geom_b.append_value(to_wkb(&g)?);
                }
            }
            let props = f.get("properties").and_then(Value::as_object);
            for ((name, _), b) in cols.iter().zip(&mut attr_b) {
                let cell = match props.and_then(|p| p.get(name)) {
                    None | Some(Value::Null) => Cell::Null,
                    Some(Value::Bool(x)) => Cell::Bool(*x),
                    Some(Value::Number(n)) => match n.as_i64() {
                        Some(i) => Cell::Int(i),
                        None => Cell::Float(n.as_f64().unwrap_or(f64::NAN)),
                    },
                    Some(Value::String(s)) => Cell::Str(Cow::Borrowed(s)),
                    Some(other) => Cell::Str(Cow::Owned(other.to_string())),
                };
                b.push(cell);
            }
        }
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(geom_b.finish())];
        arrays.extend(attr_b.iter_mut().map(AttrBuilder::finish));
        arrays.push(bbox_b.finish());
        let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;
        written += chunk.len() as u64;
        progress(0.15 + 0.85 * (written as f32 / feats.len() as f32));
    }

    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        geo_meta(&geom_name, &stats, None).to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

fn coord(v: &Value) -> Result<geo_types::Coord<f64>, String> {
    let a = v.as_array().ok_or("coordinate is not an array")?;
    // Extra dimensions (z, m) are dropped.
    match (a.first().and_then(Value::as_f64), a.get(1).and_then(Value::as_f64)) {
        (Some(x), Some(y)) => Ok(geo_types::Coord { x, y }),
        _ => Err("coordinate is not [x, y, ...] numbers".into()),
    }
}

fn coords(v: &Value) -> Result<Vec<geo_types::Coord<f64>>, String> {
    v.as_array()
        .ok_or("coordinates are not an array")?
        .iter()
        .map(coord)
        .collect()
}

fn polygon(v: &Value) -> Result<geo_types::Polygon<f64>, String> {
    let rings = v.as_array().ok_or("polygon coordinates are not an array")?;
    let mut it = rings.iter().map(coords);
    let exterior = it.next().ok_or("polygon without rings")??;
    let interiors: Vec<_> = it
        .map(|r| r.map(geo_types::LineString))
        .collect::<Result<_, _>>()?;
    Ok(geo_types::Polygon::new(geo_types::LineString(exterior), interiors))
}

fn parse_geometry(v: &Value, depth: usize) -> Result<geo_types::Geometry<f64>, String> {
    use geo_types::*;
    if depth > 8 {
        return Err("geometry nesting too deep".into());
    }
    let ty = v
        .get("type")
        .and_then(Value::as_str)
        .ok_or("geometry without a type")?;
    let c = || v.get("coordinates").ok_or("geometry without coordinates");
    Ok(match ty {
        "Point" => Geometry::Point(Point(coord(c()?)?)),
        "MultiPoint" => {
            Geometry::MultiPoint(MultiPoint(coords(c()?)?.into_iter().map(Point).collect()))
        }
        "LineString" => Geometry::LineString(LineString(coords(c()?)?)),
        "MultiLineString" => Geometry::MultiLineString(MultiLineString(
            c()?.as_array()
                .ok_or("coordinates are not an array")?
                .iter()
                .map(|l| coords(l).map(LineString))
                .collect::<Result<_, _>>()?,
        )),
        "Polygon" => Geometry::Polygon(polygon(c()?)?),
        "MultiPolygon" => Geometry::MultiPolygon(MultiPolygon(
            c()?.as_array()
                .ok_or("coordinates are not an array")?
                .iter()
                .map(polygon)
                .collect::<Result<_, _>>()?,
        )),
        "GeometryCollection" => Geometry::GeometryCollection(GeometryCollection(
            v.get("geometries")
                .and_then(Value::as_array)
                .ok_or("GeometryCollection without geometries")?
                .iter()
                .map(|g| parse_geometry(g, depth + 1))
                .collect::<Result<_, _>>()?,
        )),
        other => return Err(format!("unsupported geometry type '{other}'")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    #[test]
    fn geojson_convert_round_trip() {
        let dir = std::env::temp_dir().join("geopq_geojson_import");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mini.geojson");
        let dst = dir.join("mini.parquet");

        // Mixed features: point + polygon, int-then-float widening, a
        // nested property, and a null geometry.
        let gj = json!({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [2.35, 48.85, 35.0]},
                 "properties": {"name": "paris", "pop": 2_100_000, "tags": {"a": 1}}},
                {"type": "Feature",
                 "geometry": {"type": "Polygon", "coordinates":
                     [[[2.0, 48.0], [3.0, 48.0], [3.0, 49.0], [2.0, 48.0]]]},
                 "properties": {"name": "box", "pop": 1.5}},
                {"type": "Feature", "geometry": null,
                 "properties": {"name": "nowhere"}}
            ]
        });
        std::fs::write(&src, gj.to_string()).unwrap();

        let written = convert(&src, &dst, &|_| {}).unwrap();
        assert_eq!(written, 3);

        let (store, crs, info, _rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.total_rows(), 3);
        assert_eq!(crs.epsg, Some(4326)); // spec default CRS84
        assert!(info.geo.geometry_types.contains(&"Point".to_string()));
        assert!(info.geo.geometry_types.contains(&"Polygon".to_string()));
        let b = info.geo.bbox.expect("bbox");
        assert!(b[0] >= 1.9 && b[3] <= 49.1);

        let geoms = store.fetch_geoms(&[0, 1, 2]).unwrap();
        assert!(matches!(geoms[0].1, Some(geo_types::Geometry::Point(_))));
        assert!(matches!(geoms[1].1, Some(geo_types::Geometry::Polygon(_))));
        assert!(geoms[2].1.is_none());

        // pop widened to Float64 (int then float); nested tags stay JSON text.
        let batch = store.fetch(&[0], None).unwrap().remove(0);
        let sc = batch.schema();
        assert_eq!(sc.field_with_name("pop").unwrap().data_type(), &DataType::Float64);
        assert_eq!(sc.field_with_name("tags").unwrap().data_type(), &DataType::Utf8);
    }
}
