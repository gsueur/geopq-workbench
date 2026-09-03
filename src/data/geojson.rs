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
    covering_field, geo_meta, to_wkb, AttrBuilder, BboxBuilder, Cell, GeomStats,
    IMPORT_BATCH_BYTES, IMPORT_WRITE_ROWS,
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
    let (cov_name, cov_field) = covering_field(&fields);
    fields.push(cov_field);
    let schema = Arc::new(Schema::new(fields));

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer = ArrowWriter::try_new(out, schema.clone(), Some(super::import::writer_props()))
        .map_err(|e| e.to_string())?;

    let mut stats = GeomStats::new();
    let mut written = 0u64;
    let mut geom_b = BinaryBuilder::new();
    let mut bbox_b = BboxBuilder::default();
    let mut attr_b: Vec<AttrBuilder> = cols.iter().map(|(_, dt)| AttrBuilder::new(dt)).collect();
    let mut in_batch = 0usize;
    // Rows alone do not bound a batch: a few thousand administrative
    // boundaries are hundreds of megabytes of WKB, the row group cannot
    // close in the middle of one, and the result was import files whose
    // own scorecard flagged their row groups as too heavy to fetch.
    let mut batch_bytes = 0usize;
    for (fi, f) in feats.iter().enumerate() {
        match f.get("geometry") {
            Some(Value::Null) | None => {
                geom_b.append_null();
                bbox_b.push(None);
            }
            // RFC 7946 §3.1: an empty "coordinates" array is an empty
            // geometry, which a processor may treat as null. ArcGIS
            // Hub exports a feature with no location exactly this
            // way, and one unplaced row must not refuse the file.
            Some(g) if is_empty_geometry(g) => {
                geom_b.append_null();
                bbox_b.push(None);
            }
            Some(g) => {
                let g = parse_geometry(g, 0)?;
                bbox_b.push(stats.add(&g));
                let wkb = to_wkb(&g)?;
                batch_bytes += wkb.len();
                geom_b.append_value(wkb);
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
        in_batch += 1;
        if in_batch == IMPORT_WRITE_ROWS
            || batch_bytes >= IMPORT_BATCH_BYTES
            || fi + 1 == feats.len()
        {
            let mut arrays: Vec<ArrayRef> = vec![Arc::new(geom_b.finish())];
            arrays.extend(attr_b.iter_mut().map(AttrBuilder::finish));
            arrays.push(bbox_b.finish());
            let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(|e| e.to_string())?;
            writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;
            written += in_batch as u64;
            in_batch = 0;
            batch_bytes = 0;
            progress(0.15 + 0.85 * (written as f32 / feats.len() as f32));
        }
    }

    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        geo_meta(&geom_name, &cov_name, &stats, None).to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

/// An empty geometry: `"coordinates": []` on any coordinates-bearing
/// type, or a GeometryCollection whose `"geometries"` list is empty. A
/// missing member is still an error, not an empty geometry.
fn is_empty_geometry(v: &Value) -> bool {
    let empty = |key: &str| v.get(key).and_then(Value::as_array).is_some_and(Vec::is_empty);
    empty("coordinates") || empty("geometries")
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

    #[test]
    /// ArcGIS Hub exports a feature with no location as an empty
    /// geometry ({"type": "Point", "coordinates": []}), which RFC 7946
    /// lets a processor treat as null. Boston's parking-meters file
    /// carries exactly one such row at the end, and it must not refuse
    /// the 6,954 placed meters before it. A missing "coordinates"
    /// member stays an error: that is malformed, not empty.
    fn an_empty_geometry_is_a_null_row_not_a_refusal() {
        let dir = std::env::temp_dir().join("geopq_geojson_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("meters.geojson");
        let dst = dir.join("meters.parquet");

        let gj = json!({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [-71.07, 42.35]},
                 "properties": {"METER_ID": 450001}},
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": []},
                 "properties": {"METER_ID": null}},
                {"type": "Feature",
                 "geometry": {"type": "GeometryCollection", "geometries": []},
                 "properties": {"METER_ID": null}}
            ]
        });
        std::fs::write(&src, gj.to_string()).unwrap();
        assert_eq!(convert(&src, &dst, &|_| {}).unwrap(), 3);

        let (store, _crs, _info, _rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        let geoms = store.fetch_geoms(&[0, 1, 2]).unwrap();
        assert!(matches!(geoms[0].1, Some(geo_types::Geometry::Point(_))));
        assert!(geoms[1].1.is_none());
        assert!(geoms[2].1.is_none());

        std::fs::write(
            &src,
            json!({"type": "Feature", "geometry": {"type": "Point"},
                   "properties": {}})
            .to_string(),
        )
        .unwrap();
        let err = convert(&src, &dst, &|_| {}).unwrap_err();
        assert!(err.contains("without coordinates"), "{err}");
    }

    /// A property genuinely called `bbox` is data. The covering column
    /// used to be pushed under that name unconditionally, producing a
    /// file with two `bbox` fields whose `covering` metadata pointed at
    /// whichever one a reader resolved first.
    #[test]
    fn a_property_named_bbox_keeps_its_name() {
        let dir = std::env::temp_dir().join("geopq_geojson_bbox_attr");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("bbox_attr.geojson");
        let dst = dir.join("bbox_attr.parquet");

        let gj = json!({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [2.35, 48.85]},
                 "properties": {"bbox": "the attribute", "id": 1}}
            ]
        });
        std::fs::write(&src, gj.to_string()).unwrap();
        assert_eq!(convert(&src, &dst, &|_| {}).unwrap(), 1);

        let (store, _crs, info, _rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        let sc = store.schema.clone();
        // The attribute kept its name, its type and its value.
        assert_eq!(sc.field_with_name("bbox").unwrap().data_type(), &DataType::Utf8);
        assert!(sc.field_with_name("bbox_1").is_ok(), "covering renamed around it");
        // And the metadata points readers at the renamed one.
        let covering = info.geo.covering.expect("covering declared");
        assert!(covering.contains("\"bbox_1\""), "{covering}");
        // The reader resolves it: a covering column it could not find
        // would leave the store without per-row boxes.
        let g = store.fetch_geoms(&[0]).unwrap();
        assert!(matches!(g[0].1, Some(geo_types::Geometry::Point(_))));
    }

    /// Heavy polygons: a batch has to be bounded by its bytes, not only
    /// by its row count. A few thousand land-cover rings are hundreds of
    /// megabytes, a row group cannot close in the middle of a write, and
    /// the row cap alone therefore produced one enormous group.
    #[test]
    fn heavy_polygons_close_row_groups_by_bytes() {
        let dir = std::env::temp_dir().join("geopq_geojson_heavy");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("heavy.geojson");
        let dst = dir.join("heavy.parquet");

        // 1000 rings of 2000 vertices: ~32 MB of WKB, well inside the
        // 8192-row batch cap and twice the 16 MB row-group cap. The
        // radii are jittered so the coordinates do not compress: a ring
        // every polygon shares would encode small enough for the writer
        // to keep one group whatever the batching does, and the point
        // here is the batching.
        let mut seed = 12345u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let feats: Vec<Value> = (0..1000)
            .map(|k| {
                let (cx, cy) = (k as f64 * 0.01, 45.0);
                let ring: Vec<Value> = (0..2000)
                    .map(|i| {
                        let a = i as f64 / 2000.0 * std::f64::consts::TAU;
                        let r = 0.001 * (0.5 + next());
                        json!([cx + r * a.cos(), cy + r * a.sin()])
                    })
                    .chain(std::iter::once(json!([cx + 0.001, cy])))
                    .collect();
                json!({"type": "Feature", "properties": {"k": k},
                       "geometry": {"type": "Polygon", "coordinates": [ring]}})
            })
            .collect();
        std::fs::write(
            &src,
            json!({"type": "FeatureCollection", "features": feats}).to_string(),
        )
        .unwrap();
        assert_eq!(convert(&src, &dst, &|_| {}).unwrap(), 1000);
        crate::data::import::tests::assert_row_groups_bounded(&dst);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
