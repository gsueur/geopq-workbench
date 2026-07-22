//! Shapefile import (pure Rust, no GDAL): .shp/.dbf/.shx via the
//! `shapefile` crate; a .shp without a .dbf loads as geometry-only.
//! Z/M shapes flatten to 2D. The CRS comes from a sibling .prj when it
//! carries an EPSG authority code; a recognizable WGS84 .prj maps to the
//! CRS84 default, anything else is recorded as unknown (rendered as
//! CRS84, labeled honestly by the viewer).

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};
use shapefile::dbase;

use super::import::{geo_meta, to_wkb, AttrBuilder, Cell, GeomStats, IMPORT_BATCH_ROWS};

/// Convert a shapefile to GeoParquet. Returns the rows written.
pub fn convert(src: &Path, dst: &Path, progress: &dyn Fn(f32)) -> Result<u64, String> {
    let dbf = src.with_extension("dbf");
    let has_dbf = dbf.exists();

    // Attribute columns in file order (separate dbase open: the combined
    // reader only yields field info after consuming itself).
    let cols: Vec<(String, DataType)> = if has_dbf {
        dbase::Reader::from_path(&dbf)
            .map_err(|e| format!("cannot read dbf: {e}"))?
            .fields()
            .iter()
            .map(|f| (f.name().to_string(), arrow_type(f.field_type())))
            .collect()
    } else {
        Vec::new()
    };

    let geom_name = {
        let mut name = "geometry".to_string();
        let mut i = 0usize;
        while cols.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name)) {
            i += 1;
            name = format!("geometry_{i}");
        }
        name
    };
    let mut fields: Vec<Field> = vec![Field::new(&geom_name, DataType::Binary, true)];
    fields.extend(cols.iter().map(|(n, dt)| Field::new(n, dt.clone(), true)));
    let schema = Arc::new(Schema::new(fields));

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer = ArrowWriter::try_new(out, schema.clone(), Some(super::import::writer_props()))
        .map_err(|e| e.to_string())?;

    let mut stats = GeomStats::new();
    let mut out = Out {
        writer: &mut writer,
        schema: schema.clone(),
        geom_b: BinaryBuilder::new(),
        attr_b: cols.iter().map(|(_, dt)| AttrBuilder::new(dt)).collect(),
        in_batch: 0,
        written: 0,
    };

    if has_dbf {
        let mut reader = shapefile::Reader::from_path(src)
            .map_err(|e| format!("cannot open shapefile: {e}"))?;
        let total = reader.shape_count().unwrap_or(0).max(1);
        for pair in reader.iter_shapes_and_records() {
            let (shape, record) = pair.map_err(|e| format!("read failed: {e}"))?;
            push_shape(shape, &mut stats, &mut out.geom_b);
            for ((name, _), b) in cols.iter().zip(&mut out.attr_b) {
                b.push(match record.get(name) {
                    Some(v) => cell(v),
                    None => Cell::Null,
                });
            }
            if out.bump()? {
                progress((out.written as f32 / total as f32).min(1.0));
            }
        }
    } else {
        let mut reader = shapefile::ShapeReader::from_path(src)
            .map_err(|e| format!("cannot open shapefile: {e}"))?;
        for shape in reader.iter_shapes() {
            push_shape(
                shape.map_err(|e| format!("read failed: {e}"))?,
                &mut stats,
                &mut out.geom_b,
            );
            out.bump()?;
        }
    }
    out.flush()?;
    let written = out.written;
    drop(out);

    let crs = prj_crs(&src.with_extension("prj"));
    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        geo_meta(&geom_name, &stats, crs).to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

/// Chunked batch assembly over the shared builders.
struct Out<'a> {
    writer: &'a mut ArrowWriter<std::fs::File>,
    schema: Arc<Schema>,
    geom_b: BinaryBuilder,
    attr_b: Vec<AttrBuilder>,
    in_batch: usize,
    written: u64,
}

impl Out<'_> {
    /// Count the row just pushed; flush on a full batch. Returns whether
    /// a flush happened (progress checkpoint).
    fn bump(&mut self) -> Result<bool, String> {
        self.in_batch += 1;
        if self.in_batch == IMPORT_BATCH_ROWS {
            self.flush()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn flush(&mut self) -> Result<(), String> {
        if self.in_batch == 0 {
            return Ok(());
        }
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(self.geom_b.finish())];
        arrays.extend(self.attr_b.iter_mut().map(AttrBuilder::finish));
        let batch =
            RecordBatch::try_new(self.schema.clone(), arrays).map_err(|e| e.to_string())?;
        self.writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;
        self.written += self.in_batch as u64;
        self.in_batch = 0;
        Ok(())
    }
}

fn push_shape(shape: shapefile::Shape, stats: &mut GeomStats, geom_b: &mut BinaryBuilder) {
    match geo_types::Geometry::<f64>::try_from(shape) {
        Ok(g) => {
            stats.add(&g);
            match to_wkb(&g) {
                Ok(w) => geom_b.append_value(w),
                Err(_) => geom_b.append_null(),
            }
        }
        Err(_) => geom_b.append_null(), // NullShape
    }
}

fn arrow_type(t: dbase::FieldType) -> DataType {
    use dbase::FieldType as T;
    match t {
        T::Integer => DataType::Int64,
        T::Numeric | T::Float | T::Double | T::Currency => DataType::Float64,
        T::Logical => DataType::Boolean,
        _ => DataType::Utf8, // Character, Memo, Date, DateTime, ...
    }
}

fn cell(v: &dbase::FieldValue) -> Cell<'_> {
    use dbase::FieldValue as V;
    match v {
        V::Character(Some(s)) => Cell::Str(Cow::Borrowed(s)),
        V::Memo(s) => Cell::Str(Cow::Borrowed(s)),
        V::Numeric(Some(f)) => Cell::Float(*f),
        V::Float(Some(f)) => Cell::Float(*f as f64),
        V::Double(f) => Cell::Float(*f),
        V::Currency(f) => Cell::Float(*f),
        V::Integer(i) => Cell::Int(*i as i64),
        V::Logical(Some(b)) => Cell::Bool(*b),
        V::Date(Some(d)) => Cell::Str(Cow::Owned(format!(
            "{:04}-{:02}-{:02}",
            d.year(),
            d.month(),
            d.day()
        ))),
        V::DateTime(dt) => {
            let (d, t) = (dt.date(), dt.time());
            Cell::Str(Cow::Owned(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                d.year(),
                d.month(),
                d.day(),
                t.hours(),
                t.minutes(),
                t.seconds()
            )))
        }
        _ => Cell::Null,
    }
}

/// CRS from an ESRI .prj: the last `AUTHORITY["EPSG","<code>"]` wins
/// (GDAL-written WKT carries it on the outermost node last); a
/// recognizable WGS84 geographic WKT maps to the CRS84 default; anything
/// else (including no .prj) is an explicitly unknown CRS.
fn prj_crs(path: &Path) -> Option<Value> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Some(Value::Null);
    };
    let mut last: Option<u32> = None;
    let mut rest = text.as_str();
    while let Some(i) = rest.find("AUTHORITY[\"EPSG\",\"") {
        let tail = &rest[i + 18..];
        if let Some(end) = tail.find('"') {
            if let Ok(code) = tail[..end].parse::<u32>() {
                last = Some(code);
            }
        }
        rest = &rest[i + 18..];
    }
    match last {
        Some(4326) => None,
        Some(code) => {
            // The WKT name is the first quoted string.
            let name = text
                .split('"')
                .nth(1)
                .unwrap_or("from .prj")
                .replace('_', " ");
            Some(json!({"name": name, "id": {"authority": "EPSG", "code": code}}))
        }
        None => {
            let geographic_wgs84 = text.starts_with("GEOGCS")
                && (text.contains("WGS_1984") || text.contains("WGS 84"));
            if geographic_wgs84 {
                None
            } else {
                Some(Value::Null)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;
    use dbase::{FieldValue, Record, TableWriterBuilder};
    use shapefile::{Point, Polygon, PolygonRing};

    /// A shapefile holds exactly one shape type; a two-polygon file with
    /// a .prj carrying an EPSG authority covers the whole pipeline.
    #[test]
    fn shp_convert_round_trip() {
        let dir = std::env::temp_dir().join("geopq_shp_import");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mini.shp");
        let dst = dir.join("mini.parquet");

        let table = TableWriterBuilder::new()
            .add_character_field("name".try_into().unwrap(), 20)
            .add_numeric_field("area".try_into().unwrap(), 12, 3);
        let mut w = shapefile::Writer::from_path(&src, table).unwrap();
        let square = |x0: f64, y0: f64, d: f64| {
            Polygon::new(PolygonRing::Outer(vec![
                Point::new(x0, y0),
                Point::new(x0, y0 + d),
                Point::new(x0 + d, y0 + d),
                Point::new(x0 + d, y0),
                Point::new(x0, y0),
            ]))
        };
        let mut rec = Record::default();
        rec.insert("name".into(), FieldValue::Character(Some("a".into())));
        rec.insert("area".into(), FieldValue::Numeric(Some(1.5)));
        w.write_shape_and_record(&square(700_000.0, 6_600_000.0, 1_000.0), &rec).unwrap();
        let mut rec2 = Record::default();
        rec2.insert("name".into(), FieldValue::Character(Some("b".into())));
        rec2.insert("area".into(), FieldValue::Numeric(None));
        w.write_shape_and_record(&square(705_000.0, 6_605_000.0, 2_000.0), &rec2).unwrap();
        drop(w);
        std::fs::write(
            dir.join("mini.prj"),
            "PROJCS[\"RGF93_Lambert_93\",GEOGCS[\"GCS_RGF_1993\"],\
             AUTHORITY[\"EPSG\",\"2154\"]]",
        )
        .unwrap();

        let written = convert(&src, &dst, &|_| {}).unwrap();
        assert_eq!(written, 2);

        let (store, crs, info, _rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.total_rows(), 2);
        assert_eq!(crs.epsg, Some(2154));
        assert!(info.geo.geometry_types.contains(&"MultiPolygon".to_string()));
        let b = info.geo.bbox.expect("bbox");
        assert_eq!(b[0], 700_000.0);
        assert_eq!(b[2], 707_000.0);
        let geoms = store.fetch_geoms(&[0, 1]).unwrap();
        assert!(matches!(geoms[0].1, Some(geo_types::Geometry::MultiPolygon(_))));
        assert!(matches!(geoms[1].1, Some(geo_types::Geometry::MultiPolygon(_))));
        let batch = store.fetch(&[0], None).unwrap().remove(0);
        let sc = batch.schema();
        assert_eq!(sc.field_with_name("name").unwrap().data_type(), &DataType::Utf8);
        assert_eq!(sc.field_with_name("area").unwrap().data_type(), &DataType::Float64);
    }
}
