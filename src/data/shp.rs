//! Shapefile import (pure Rust, no GDAL): .shp/.dbf/.shx via the
//! `shapefile` crate; a .shp without a .dbf loads as geometry-only.
//! Z/M shapes flatten to 2D. Sidecars are resolved case-insensitively
//! (ROADS.DBF next to roads.shp is a fact of life) and a .cpg sets the
//! attribute text encoding. The CRS comes from the sibling .prj: an
//! EPSG authority code when present, otherwise the WKT is parsed to a
//! proj4 string (ESRI method names normalized) carried as a `geopq:crs`
//! extension — spec `crs` stays null, this app still positions the data
//! correctly. A recognizable WGS84 .prj maps to the CRS84 default;
//! anything unparseable is recorded as unknown.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};
use shapefile::dbase;

use super::import::{bbox_field, BboxBuilder, to_wkb, AttrBuilder, Cell, GeomStats, IMPORT_WRITE_ROWS};

/// Find a shapefile sidecar (`dbf`, `shx`, `prj`, `cpg`) next to the
/// .shp, matching stem and extension case-insensitively — real-world
/// archives mix `roads.shp` with `ROADS.DBF` freely, and lowercase
/// `with_extension` misses them on case-sensitive filesystems.
fn sidecar(src: &Path, ext: &str) -> Option<std::path::PathBuf> {
    let exact = src.with_extension(ext);
    if exact.exists() {
        return Some(exact);
    }
    let stem = src.file_stem()?.to_string_lossy().to_lowercase();
    let dir = src.parent()?;
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        p.file_stem()
            .is_some_and(|s| s.to_string_lossy().to_lowercase() == stem)
            && p.extension()
                .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case(ext))
    })
}

/// The attribute encoding from a sidecar .cpg (fallback: the .dbf's own
/// header language driver via the crate default).
fn cpg_encoding(src: &Path) -> Option<dbase::encoding::DynEncoding> {
    let text = std::fs::read_to_string(sidecar(src, "cpg")?).ok()?;
    dbase::encoding::DynEncoding::from_name(text.trim().trim_start_matches('\u{feff}'))
}

fn open_dbf(
    path: &Path,
    enc: Option<dbase::encoding::DynEncoding>,
) -> Result<dbase::Reader<std::io::BufReader<std::fs::File>>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("cannot read dbf: {e}"))?;
    let src = std::io::BufReader::new(f);
    match enc {
        Some(e) => dbase::ReaderBuilder::new().with_encoding(e).build(src),
        None => dbase::Reader::new(src),
    }
    .map_err(|e| format!("cannot read dbf: {e}"))
}

/// Convert a shapefile to GeoParquet. Returns the rows written.
pub fn convert(src: &Path, dst: &Path, progress: &dyn Fn(f32)) -> Result<u64, String> {
    let dbf = sidecar(src, "dbf");
    let enc = cpg_encoding(src);

    // Attribute columns in file order (separate dbase open: the combined
    // reader only yields field info after consuming itself).
    let cols: Vec<(String, DataType)> = match &dbf {
        Some(p) => open_dbf(p, enc.clone())?
            .fields()
            .iter()
            .map(|f| (f.name().to_string(), arrow_type(f.field_type())))
            .collect(),
        None => Vec::new(),
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
    fields.push(bbox_field());
    let schema = Arc::new(Schema::new(fields));

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer = ArrowWriter::try_new(out, schema.clone(), Some(super::import::writer_props()))
        .map_err(|e| e.to_string())?;

    let mut stats = GeomStats::new();
    let mut out = Out {
        writer: &mut writer,
        schema: schema.clone(),
        geom_b: BinaryBuilder::new(),
        bbox_b: BboxBuilder::default(),
        attr_b: cols.iter().map(|(_, dt)| AttrBuilder::new(dt)).collect(),
        in_batch: 0,
        written: 0,
    };

    // Explicit sources (never `from_path`): sidecars were resolved
    // case-insensitively above, and the .cpg encoding is honored.
    let shp_file = std::fs::File::open(src)
        .map(std::io::BufReader::new)
        .map_err(|e| format!("cannot open shapefile: {e}"))?;
    let shape_reader = match sidecar(src, "shx") {
        Some(shx) => {
            let shx = std::fs::File::open(shx)
                .map(std::io::BufReader::new)
                .map_err(|e| format!("cannot open shx: {e}"))?;
            shapefile::ShapeReader::with_shx(shp_file, shx)
        }
        None => shapefile::ShapeReader::new(shp_file),
    }
    .map_err(|e| format!("cannot open shapefile: {e}"))?;

    if let Some(dbf_path) = &dbf {
        let mut reader = shapefile::Reader::new(shape_reader, open_dbf(dbf_path, enc)?);
        let total = reader.shape_count().unwrap_or(0).max(1);
        for pair in reader.iter_shapes_and_records() {
            let (shape, record) = pair.map_err(|e| format!("read failed: {e}"))?;
            push_shape(shape, &mut stats, &mut out.geom_b, &mut out.bbox_b);
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
        let mut reader = shape_reader;
        for shape in reader.iter_shapes() {
            push_shape(
                shape.map_err(|e| format!("read failed: {e}"))?,
                &mut stats,
                &mut out.geom_b,
                &mut out.bbox_b,
            );
            out.bump()?;
        }
    }
    out.flush()?;
    let written = out.written;
    drop(out);

    let (crs, proj4) = prj_crs(sidecar(src, "prj").as_deref());
    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        super::import::geo_meta_with_proj4(&geom_name, &stats, crs, proj4).to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

/// Chunked batch assembly over the shared builders.
struct Out<'a> {
    writer: &'a mut ArrowWriter<std::fs::File>,
    schema: Arc<Schema>,
    geom_b: BinaryBuilder,
    bbox_b: BboxBuilder,
    attr_b: Vec<AttrBuilder>,
    in_batch: usize,
    written: u64,
}

impl Out<'_> {
    /// Count the row just pushed; flush on a full batch. Returns whether
    /// a flush happened (progress checkpoint).
    fn bump(&mut self) -> Result<bool, String> {
        self.in_batch += 1;
        if self.in_batch == IMPORT_WRITE_ROWS {
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
        arrays.push(self.bbox_b.finish());
        let batch =
            RecordBatch::try_new(self.schema.clone(), arrays).map_err(|e| e.to_string())?;
        self.writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;
        self.written += self.in_batch as u64;
        self.in_batch = 0;
        Ok(())
    }
}

fn push_shape(
    shape: shapefile::Shape,
    stats: &mut GeomStats,
    geom_b: &mut BinaryBuilder,
    bbox_b: &mut BboxBuilder,
) {
    match geo_types::Geometry::<f64>::try_from(shape) {
        Ok(g) => {
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
        Err(_) => {
            // NullShape
            geom_b.append_null();
            bbox_b.push(None);
        }
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

/// CRS from a .prj, best identity first. Returns `(crs, proj4)`:
/// - `AUTHORITY["EPSG","<code>"]` present (GDAL-written WKT): a
///   PROJJSON identity — the last authority wins (it sits on the
///   outermost node last); `crs: None` for 4326 (the CRS84 default).
/// - No authority (ArcGIS-written ESRI WKT): parse the WKT to a proj4
///   string (normalizing ESRI's ambiguous method names first). A
///   geographic WGS84 result maps to the default; anything else is
///   spec-unknown (`crs: null`) but carries the proj4 string for
///   correct display in this app.
/// - Unparseable / missing .prj: explicitly unknown.
fn prj_crs(path: Option<&Path>) -> (Option<Value>, Option<(String, String)>) {
    let text = match path.and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(t) => t,
        None => return (Some(Value::Null), None),
    };
    // The WKT name is the first quoted string.
    let name = text
        .split('"')
        .nth(1)
        .unwrap_or("from .prj")
        .replace('_', " ");

    let mut last: Option<u32> = None;
    let mut rest = text.as_str();
    while let Some(i) = rest.find("AUTHORITY[\"EPSG\",\"") {
        let tail = &rest[i + 18..];
        if let Some(end) = tail.find('"')
            && let Ok(code) = tail[..end].parse::<u32>()
        {
            last = Some(code);
        }
        rest = &rest[i + 18..];
    }
    match last {
        Some(4326) => (None, None),
        Some(code) => (
            Some(json!({"name": name, "id": {"authority": "EPSG", "code": code}})),
            None,
        ),
        None => match proj4_from_wkt(&text) {
            Some(p4) if is_geographic_wgs84(&p4) => (None, None),
            Some(p4) => (Some(Value::Null), Some((p4, name))),
            None => {
                // Last resort for WGS84 spellings the parser rejects.
                let wgs84 = text.starts_with("GEOGCS")
                    && (text.contains("WGS_1984") || text.contains("WGS 84"));
                if wgs84 { (None, None) } else { (Some(Value::Null), None) }
            }
        },
    }
}

/// WKT → proj4 via proj4wkt, after rewriting ESRI method names the
/// parser does not know: ESRI writes `Lambert_Conformal_Conic` for both
/// the 1SP and 2SP variants (told apart by their parameters), and plain
/// `Albers` / `Gauss_Kruger`.
fn proj4_from_wkt(wkt: &str) -> Option<String> {
    let mut t = Cow::Borrowed(wkt);
    if t.contains("PROJECTION[\"Lambert_Conformal_Conic\"]") {
        let variant = if t.contains("Standard_Parallel_2") {
            "PROJECTION[\"Lambert_Conformal_Conic_2SP\"]"
        } else {
            "PROJECTION[\"Lambert_Conformal_Conic_1SP\"]"
        };
        t = Cow::Owned(t.replace("PROJECTION[\"Lambert_Conformal_Conic\"]", variant));
    }
    for (esri, known) in [
        ("PROJECTION[\"Albers\"]", "PROJECTION[\"Albers_Conic_Equal_Area\"]"),
        ("PROJECTION[\"Gauss_Kruger\"]", "PROJECTION[\"Transverse_Mercator\"]"),
    ] {
        if t.contains(esri) {
            t = Cow::Owned(t.replace(esri, known));
        }
    }
    match proj4wkt::wkt_to_projstring(&t) {
        Ok(p4) => Some(p4),
        Err(e) => {
            log::warn!(".prj not convertible to proj4 ({e}); CRS recorded as unknown");
            None
        }
    }
}

/// A proj4 string that is plain geographic WGS84 (ESRI GCS_WGS_1984):
/// longitude/latitude on the WGS84 ellipsoid, no datum shift.
fn is_geographic_wgs84(p4: &str) -> bool {
    let nonzero_shift = p4.split_whitespace().any(|t| {
        t.strip_prefix("+towgs84=").is_some_and(|v| {
            v.split(',').any(|x| x.trim().parse::<f64>().is_ok_and(|f| f != 0.0))
        })
    });
    p4.starts_with("+proj=longlat")
        && (p4.contains("+datum=WGS84")
            || (p4.contains("+a=6378137") && p4.contains("+rf=298.257223563")))
        && !nonzero_shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;
    use dbase::{FieldValue, Record, TableWriterBuilder};
    use shapefile::{Point, Polygon, PolygonRing};

    /// ArcGIS-style delivery: UPPERCASE sidecars, an ESRI .prj without
    /// any AUTHORITY node, a .cpg, and accented attribute text. The
    /// import must find every sidecar, position via the parsed proj4,
    /// and keep the text intact.
    #[test]
    fn esri_sidecars_uppercase_prj_without_authority() {
        use arrow::array::StringArray;

        let dir = std::env::temp_dir().join("geopq_shp_esri");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mini.shp");
        let dst = dir.join("mini.parquet");

        let table = TableWriterBuilder::new()
            .add_character_field("name".try_into().unwrap(), 20);
        let mut w = shapefile::Writer::from_path(&src, table).unwrap();
        let square = Polygon::new(PolygonRing::Outer(vec![
            Point::new(700_000.0, 6_600_000.0),
            Point::new(700_000.0, 6_601_000.0),
            Point::new(701_000.0, 6_601_000.0),
            Point::new(701_000.0, 6_600_000.0),
            Point::new(700_000.0, 6_600_000.0),
        ]));
        let mut rec = Record::default();
        rec.insert("name".into(), FieldValue::Character(Some("café".into())));
        w.write_shape_and_record(&square, &rec).unwrap();
        drop(w);

        // Uppercase every sidecar (Windows-era archives do this), and an
        // ESRI WKT .prj: Lambert-93 with no AUTHORITY anywhere.
        std::fs::rename(dir.join("mini.dbf"), dir.join("MINI.DBF")).unwrap();
        std::fs::rename(dir.join("mini.shx"), dir.join("MINI.SHX")).unwrap();
        std::fs::write(
            dir.join("MINI.PRJ"),
            r#"PROJCS["RGF_1993_Lambert_93",GEOGCS["GCS_RGF_1993",DATUM["D_RGF_1993",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",700000.0],PARAMETER["False_Northing",6600000.0],PARAMETER["Central_Meridian",3.0],PARAMETER["Standard_Parallel_1",49.0],PARAMETER["Standard_Parallel_2",44.0],PARAMETER["Latitude_Of_Origin",46.5],UNIT["Meter",1.0]]"#,
        )
        .unwrap();
        std::fs::write(dir.join("MINI.CPG"), "UTF-8").unwrap();

        let written = convert(&src, &dst, &|_| {}).unwrap();
        assert_eq!(written, 1);

        let (store, crs, info, _rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        // No EPSG identity, but a usable projected CRS from the WKT.
        assert_eq!(crs.epsg, None);
        assert!(crs.proj4.contains("+proj=lcc"), "{}", crs.proj4);
        assert!(!crs.is_latlong);
        assert!(crs.name.contains("Lambert"), "{}", crs.name);
        // The projection is numerically right: the square's corner maps
        // to the Lambert-93 origin (3°E 46.5°N-ish region).
        let (lon, lat) =
            crate::data::crs::transform_point(
                &crs,
                &crate::data::crs::Crs::wgs84(),
                700_000.0,
                6_600_000.0,
            )
            .unwrap();
        assert!((lon - 3.0).abs() < 0.01, "{lon}");
        assert!((lat - 46.5).abs() < 0.3, "{lat}");
        // Attributes found through the uppercase .DBF, text intact.
        let b = info.geo.bbox.expect("bbox");
        assert_eq!(b[0], 700_000.0);
        let batch = store.fetch(&[0], None).unwrap().remove(0);
        let names = batch
            .column(batch.schema().index_of("name").unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
            .to_string();
        assert_eq!(names, "café");
    }

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
