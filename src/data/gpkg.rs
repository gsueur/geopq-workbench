//! GeoPackage import: pure-Rust conversion (bundled SQLite, no GDAL) of a
//! feature table into plain WKB GeoParquet. The output is a faithful raw
//! export — insertion order, no covering column — so the quality scorecard
//! and the Optimize flow take it from there.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};
use serde_json::json;

use super::import::{
    bbox_field, AttrBuilder, BboxBuilder, Cell, IMPORT_WRITE_ROWS as BATCH_ROWS,
};

/// One feature table of a GeoPackage.
#[derive(Clone)]
pub struct GpkgTable {
    pub name: String,
    pub geom_col: String,
    pub rows: u64,
    /// EPSG code, when `gpkg_spatial_ref_sys` maps the srs to one.
    pub epsg: Option<u32>,
    pub srs_name: String,
}

fn open_ro(path: &Path) -> Result<Connection, String> {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("cannot open GeoPackage: {e}"))
}

/// Feature tables listed in `gpkg_contents`, with row counts and CRS.
pub fn list_tables(path: &Path) -> Result<Vec<GpkgTable>, String> {
    let db = open_ro(path)?;
    let mut stmt = db
        .prepare(
            "SELECT c.table_name, g.column_name, g.srs_id
             FROM gpkg_contents c
             JOIN gpkg_geometry_columns g ON g.table_name = c.table_name
             WHERE c.data_type = 'features'
             ORDER BY c.table_name",
        )
        .map_err(|e| format!("not a GeoPackage (no gpkg_contents): {e}"))?;
    let rows: Vec<(String, String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .map_err(|e| e.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    drop(stmt);
    if rows.is_empty() {
        return Err("no feature tables in this GeoPackage".into());
    }
    rows.into_iter()
        .map(|(name, geom_col, srs_id)| {
            let count: i64 = db
                .query_row(&format!("SELECT COUNT(*) FROM \"{name}\""), [], |r| r.get(0))
                .map_err(|e| format!("count {name}: {e}"))?;
            let (org, code, srs_name): (String, i64, String) = db
                .query_row(
                    "SELECT organization, organization_coordsys_id, srs_name
                     FROM gpkg_spatial_ref_sys WHERE srs_id = ?1",
                    [srs_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap_or(("NONE".into(), srs_id, format!("srs {srs_id}")));
            let epsg = org
                .eq_ignore_ascii_case("epsg")
                .then(|| u32::try_from(code).ok())
                .flatten();
            Ok(GpkgTable { name, geom_col, rows: count.max(0) as u64, epsg, srs_name })
        })
        .collect()
}

/// SQLite declared type -> arrow type (GeoPackage core types; SQLite is
/// dynamically typed, so cell values are coerced per row and mismatches
/// become nulls).
fn arrow_type(decl: &str) -> DataType {
    let d = decl.to_ascii_uppercase();
    if d == "BOOLEAN" {
        DataType::Boolean
    } else if d.contains("INT") {
        DataType::Int64
    } else if d.contains("REAL") || d.contains("FLOA") || d.contains("DOUB") {
        DataType::Float64
    } else if d.contains("BLOB") || d.is_empty() {
        DataType::Binary
    } else {
        DataType::Utf8 // TEXT, DATE, DATETIME, ...
    }
}

/// WKB slice + optional header envelope of a non-empty geometry blob.
type GpkgGeom<'a> = Option<(&'a [u8], Option<[f64; 4]>)>;

/// Strip the GeoPackage binary header: returns the WKB slice and the
/// header envelope ([minx, miny, maxx, maxy]) when one is stored. `None`
/// for geometries flagged empty.
fn gpkg_wkb(blob: &[u8]) -> Result<GpkgGeom<'_>, String> {
    if blob.len() < 8 || blob[0] != b'G' || blob[1] != b'P' {
        return Err("not a GeoPackage geometry blob".into());
    }
    let flags = blob[3];
    let little = flags & 1 == 1;
    let env_len = match (flags >> 1) & 0x07 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return Err("invalid envelope indicator".into()),
    };
    let hdr = 8 + env_len;
    if blob.len() < hdr {
        return Err("truncated geometry blob".into());
    }
    if (flags >> 4) & 1 == 1 {
        return Ok(None); // empty geometry
    }
    let env = (env_len >= 32).then(|| {
        let f = |i: usize| -> f64 {
            let b: [u8; 8] = blob[8 + i * 8..16 + i * 8].try_into().unwrap();
            if little { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) }
        };
        // GeoPackage envelope order: minx, maxx, miny, maxy.
        [f(0), f(2), f(1), f(3)]
    });
    Ok(Some((&blob[hdr..], env)))
}

/// Base geometry type name (with " Z" for 3D) from a WKB header, for the
/// `geometry_types` metadata. ISO and EWKB variants.
fn wkb_type_name(wkb: &[u8]) -> Option<String> {
    let little = *wkb.first()? == 1;
    let b: [u8; 4] = wkb.get(1..5)?.try_into().ok()?;
    let raw = if little { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) };
    let ewkb_z = raw & 0x8000_0000 != 0;
    let (base, iso_dims) = if raw & 0xE000_0000 != 0 {
        (raw & 0xFF, 0)
    } else {
        (raw % 1000, raw / 1000)
    };
    let name = match base {
        1 => "Point",
        2 => "LineString",
        3 => "Polygon",
        4 => "MultiPoint",
        5 => "MultiLineString",
        6 => "MultiPolygon",
        7 => "GeometryCollection",
        _ => return None,
    };
    let z = ewkb_z || iso_dims == 1 || iso_dims == 3;
    Some(if z { format!("{name} Z") } else { name.to_string() })
}

/// Convert one feature table to `dst` as GeoParquet 1.1 (plain WKB).
/// Returns the number of rows written.
pub fn convert(
    src: &Path,
    table: &GpkgTable,
    dst: &Path,
    progress: &dyn Fn(f32),
) -> Result<u64, String> {
    let db = open_ro(src)?;
    let t = &table.name;

    // Attribute columns, in table order, geometry excluded.
    let mut cols: Vec<(String, DataType)> = Vec::new();
    {
        let mut stmt = db
            .prepare(&format!("PRAGMA table_info(\"{t}\")"))
            .map_err(|e| e.to_string())?;
        let infos: Vec<(String, String)> = stmt
            .query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, String>(2)?)))
            .map_err(|e| e.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        for (name, decl) in infos {
            if name != table.geom_col {
                cols.push((name, arrow_type(&decl)));
            }
        }
    }

    let mut fields: Vec<Field> = vec![Field::new(&table.geom_col, DataType::Binary, true)];
    fields.extend(cols.iter().map(|(n, dt)| Field::new(n, dt.clone(), true)));
    // Covering bbox last, so attribute column order still mirrors the
    // source table.
    fields.push(bbox_field());
    let schema = Arc::new(Schema::new(fields));

    let select: String = {
        let mut names = vec![format!("\"{}\"", table.geom_col)];
        names.extend(cols.iter().map(|(n, _)| format!("\"{n}\"")));
        format!("SELECT {} FROM \"{t}\"", names.join(", "))
    };

    let out = std::fs::File::create(dst).map_err(|e| format!("cannot create output: {e}"))?;
    let mut writer =
        ArrowWriter::try_new(out, schema.clone(), Some(super::import::writer_props()))
            .map_err(|e| e.to_string())?;

    let mut geom_types: std::collections::BTreeSet<String> = Default::default();
    let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut written = 0u64;

    let mut stmt = db.prepare(&select).map_err(|e| e.to_string())?;
    let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
    let mut geom_b = BinaryBuilder::new();
    let mut bbox_b = BboxBuilder::default();
    let mut attr_b: Vec<AttrBuilder> = cols.iter().map(|(_, dt)| AttrBuilder::new(dt)).collect();
    let mut in_batch = 0usize;
    let mut batch_bytes = 0usize;
    loop {
        let row = rows.next().map_err(|e| e.to_string())?;
        let done = row.is_none();
        if let Some(row) = row {
            match row.get_ref(0).map_err(|e| e.to_string())? {
                ValueRef::Blob(blob) => match gpkg_wkb(blob)? {
                    Some((wkb, env)) => {
                        geom_b.append_value(wkb);
                        batch_bytes += wkb.len();
                        if let Some(n) = wkb_type_name(wkb) {
                            geom_types.insert(n);
                        }
                        // The GeoPackage header usually carries the
                        // envelope already; parse the WKB only when it
                        // does not.
                        let e = match env {
                            Some(e) => Some(e),
                            None => {
                                let mut one = [
                                    f64::INFINITY,
                                    f64::INFINITY,
                                    f64::NEG_INFINITY,
                                    f64::NEG_INFINITY,
                                ];
                                crate::data::loader::grow_wkb_envelope(wkb, &mut one)
                                    .map(|()| one)
                            }
                        };
                        if let Some(e) = e {
                            bbox[0] = bbox[0].min(e[0]);
                            bbox[1] = bbox[1].min(e[1]);
                            bbox[2] = bbox[2].max(e[2]);
                            bbox[3] = bbox[3].max(e[3]);
                        }
                        bbox_b.push(e);
                    }
                    None => {
                        geom_b.append_null();
                        bbox_b.push(None);
                    }
                },
                _ => {
                    geom_b.append_null();
                    bbox_b.push(None);
                }
            }
            for (i, b) in attr_b.iter_mut().enumerate() {
                b.push(cell(row.get_ref(i + 1).map_err(|e| e.to_string())?));
            }
            in_batch += 1;
        }
        if in_batch > 0
            && (done || in_batch == BATCH_ROWS || batch_bytes >= super::import::IMPORT_BATCH_BYTES)
        {
            let mut arrays: Vec<ArrayRef> = vec![Arc::new(geom_b.finish())];
            arrays.extend(attr_b.iter_mut().map(AttrBuilder::finish));
            arrays.push(bbox_b.finish());
            let batch =
                RecordBatch::try_new(schema.clone(), arrays).map_err(|e| e.to_string())?;
            writer.write(&batch).map_err(|e| format!("write failed: {e}"))?;
            written += in_batch as u64;
            batch_bytes = 0;
            in_batch = 0;
            progress((written as f32 / table.rows.max(1) as f32).min(1.0));
        }
        if done {
            break;
        }
    }

    // GeoParquet 1.1 metadata: WKB encoding, observed geometry types, CRS
    // as a minimal PROJJSON id (omitted for 4326: the spec default CRS84).
    let mut col = json!({
        "encoding": "WKB",
        "geometry_types": geom_types.iter().collect::<Vec<_>>(),
    });
    match table.epsg {
        Some(4326) | None => {}
        Some(code) => {
            col["crs"] = json!({
                "name": table.srs_name,
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
        "primary_column": table.geom_col,
        "columns": { &table.geom_col: col },
    });
    writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
        "geo".to_string(),
        geo.to_string(),
    ));
    writer.close().map_err(|e| format!("finalize failed: {e}"))?;
    Ok(written)
}

/// SQLite value -> shared import cell (SQLite is dynamically typed;
/// the shared builder coerces or nulls per the declared column type).
fn cell(v: ValueRef<'_>) -> Cell<'_> {
    match v {
        ValueRef::Null => Cell::Null,
        ValueRef::Integer(i) => Cell::Int(i),
        ValueRef::Real(f) => Cell::Float(f),
        ValueRef::Text(t) => Cell::Str(String::from_utf8_lossy(t)),
        ValueRef::Blob(b) => Cell::Bytes(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal spec-shaped GeoPackage with one point table in EPSG:2154.
    fn make_gpkg(path: &Path, rows: usize) {
        let db = Connection::open(path).unwrap();
        db.execute_batch(
            "PRAGMA application_id = 0x47504B47;
             CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY,
               organization TEXT, organization_coordsys_id INTEGER, definition TEXT);
             INSERT INTO gpkg_spatial_ref_sys VALUES
               ('RGF93 / Lambert-93', 2154, 'EPSG', 2154, 'undefined');
             CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT,
               identifier TEXT, srs_id INTEGER);
             INSERT INTO gpkg_contents VALUES ('pts', 'features', 'pts', 2154);
             CREATE TABLE gpkg_geometry_columns (table_name TEXT PRIMARY KEY,
               column_name TEXT, geometry_type_name TEXT, srs_id INTEGER, z INTEGER, m INTEGER);
             INSERT INTO gpkg_geometry_columns VALUES ('pts', 'geom', 'POINT', 2154, 0, 0);
             CREATE TABLE pts (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT,
               height REAL, flag BOOLEAN);",
        )
        .unwrap();
        let mut ins = db
            .prepare("INSERT INTO pts (geom, name, height, flag) VALUES (?1, ?2, ?3, ?4)")
            .unwrap();
        for i in 0..rows {
            let (x, y) = (700_000.0 + i as f64 * 10.0, 6_600_000.0 + i as f64 * 5.0);
            // GPKG blob: header (little-endian, xy envelope) + WKB point.
            let mut blob: Vec<u8> = vec![b'G', b'P', 0, 0b0000_0011];
            blob.extend(2154i32.to_le_bytes());
            for v in [x, x, y, y] {
                blob.extend(v.to_le_bytes()); // minx, maxx, miny, maxy
            }
            blob.push(1); // little-endian WKB
            blob.extend(1u32.to_le_bytes()); // Point
            blob.extend(x.to_le_bytes());
            blob.extend(y.to_le_bytes());
            ins.execute(rusqlite::params![blob, format!("p{i}"), i as f64 * 0.5, i % 2 == 0])
                .unwrap();
        }
        // A null geometry row must survive as a null.
        db.execute(
            "INSERT INTO pts (geom, name, height, flag) VALUES (NULL, 'no_geom', NULL, NULL)",
            [],
        )
        .unwrap();
    }

    #[test]
    fn gpkg_convert_round_trip() {
        let dir = std::env::temp_dir().join("geopq_gpkg_import");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mini.gpkg");
        let _ = std::fs::remove_file(&src);
        make_gpkg(&src, 500);

        let tables = list_tables(&src).unwrap();
        assert_eq!(tables.len(), 1);
        let t = &tables[0];
        assert_eq!(t.name, "pts");
        assert_eq!(t.geom_col, "geom");
        assert_eq!(t.rows, 501);
        assert_eq!(t.epsg, Some(2154));

        let dst = dir.join("mini.pts.parquet");
        let written = convert(&src, t, &dst, &|_| {}).unwrap();
        assert_eq!(written, 501);

        let (store, crs, info, rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.total_rows(), 501);
        assert_eq!(crs.epsg, Some(2154));
        // The import must be spatially readable on its own: a covering
        // column, declared, so the loader can select by viewport instead
        // of decoding the file whole.
        assert!(store.covering.is_some(), "covering column detected");
        let (label, boxes) = rg.expect("row-group bboxes");
        assert!(label.contains("covering"), "{label}");
        assert_eq!(boxes.len(), info.row_groups);
        let q = info.quality.clone().expect("scorecard");
        assert_eq!(
            q.checks.iter().find(|c| c.code == "C1").unwrap().status,
            crate::data::quality::Status::Pass,
            "C1 must pass on a fresh import"
        );
        assert!(info.geo.geometry_types.contains(&"Point".to_string()));
        let b = info.geo.bbox.expect("bbox from envelopes");
        assert!(b[0] >= 700_000.0 && b[3] <= 6_610_000.0);
        // Geometry decodes as plain WKB points; the null row survives.
        let geoms = store.fetch_geoms(&[0, 250, 500]).unwrap();
        assert!(matches!(
            geoms[0].1.as_ref(),
            Some(geo_types::Geometry::Point(_))
        ));
        assert!(geoms[2].1.is_none());
        // Attribute typing: text, float and boolean-as-int survive.
        let batch = store.fetch(&[1], None).unwrap().remove(0);
        let sc = batch.schema();
        assert_eq!(sc.field_with_name("name").unwrap().data_type(), &DataType::Utf8);
        assert_eq!(sc.field_with_name("height").unwrap().data_type(), &DataType::Float64);
        assert_eq!(sc.field_with_name("flag").unwrap().data_type(), &DataType::Boolean);
    }
}
