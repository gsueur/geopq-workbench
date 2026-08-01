//! Write a query result to a GeoParquet file so it can be loaded back as a
//! regular layer (picking, attributes, refinement all reuse the loader).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, BinaryArray};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use geo::BoundingRect;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::{json, Value};

use crate::data::crs::Crs;
use crate::data::loader::decode_wkb;

/// Write `batches` as GeoParquet 1.1 (WKB) with `geom_col` as the primary
/// geometry column and `crs` recorded in the metadata.
pub fn write_result(
    path: &Path,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    geom_col: usize,
    crs: &Crs,
) -> Result<(), String> {
    let mut w = StreamWriter::new(path, schema, geom_col, crs)?;
    for batch in batches {
        w.write(batch)?;
    }
    w.finish().map(|_| ())
}

/// Incremental GeoParquet 1.1 (WKB) writer: geometry types and the file
/// bbox accumulate per batch, the `geo` metadata is attached at close.
pub struct StreamWriter {
    writer: ArrowWriter<File>,
    geom_col: usize,
    geom_name: String,
    /// `crs` value (None = omit, spec default CRS84) and the optional
    /// `geopq:crs` vendor proj4 for CRSs without EPSG id or PROJJSON.
    crs_meta: (Option<Value>, Option<(String, String)>),
    types: Vec<String>,
    bbox: Option<[f64; 4]>,
    rows: usize,
}

impl StreamWriter {
    pub fn new(
        path: &Path,
        schema: &SchemaRef,
        geom_col: usize,
        crs: &Crs,
    ) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let writer = ArrowWriter::try_new(file, Arc::clone(schema), Some(props))
            .map_err(|e| format!("parquet writer: {e}"))?;
        Ok(Self {
            writer,
            geom_col,
            geom_name: schema.field(geom_col).name().clone(),
            crs_meta: crate::data::merge::crs_to_geo(crs),
            types: Vec::new(),
            bbox: None,
            rows: 0,
        })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), String> {
        let Some(arr) = batch
            .column(self.geom_col)
            .as_any()
            .downcast_ref::<BinaryArray>()
        else {
            return Err("result geometry column is not WKB binary".into());
        };
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let Some(g) = decode_wkb(arr.value(i)) else {
                continue;
            };
            let t = match &g {
                geo_types::Geometry::Point(_) => "Point",
                geo_types::Geometry::LineString(_) | geo_types::Geometry::Line(_) => "LineString",
                geo_types::Geometry::Polygon(_)
                | geo_types::Geometry::Rect(_)
                | geo_types::Geometry::Triangle(_) => "Polygon",
                geo_types::Geometry::MultiPoint(_) => "MultiPoint",
                geo_types::Geometry::MultiLineString(_) => "MultiLineString",
                geo_types::Geometry::MultiPolygon(_) => "MultiPolygon",
                geo_types::Geometry::GeometryCollection(_) => "GeometryCollection",
            };
            // The WKB bytes are written verbatim: 3D values keep their Z,
            // so the spec requires the " Z" type suffix.
            let t = if crate::data::optimize::wkb_has_z(arr.value(i)) {
                format!("{t} Z")
            } else {
                t.to_string()
            };
            if !self.types.contains(&t) {
                self.types.push(t);
            }
            if let Some(r) = g.bounding_rect() {
                self.bbox = Some(match self.bbox {
                    None => [r.min().x, r.min().y, r.max().x, r.max().y],
                    Some(b) => [
                        b[0].min(r.min().x),
                        b[1].min(r.min().y),
                        b[2].max(r.max().x),
                        b[3].max(r.max().y),
                    ],
                });
            }
        }
        self.rows += batch.num_rows();
        self.writer
            .write(batch)
            .map_err(|e| format!("parquet write: {e}"))
    }

    /// Attach the `geo` metadata and close; returns the rows written.
    pub fn finish(mut self) -> Result<usize, String> {
        let mut types = std::mem::take(&mut self.types);
        types.sort_unstable();
        let mut col_meta = json!({
            "encoding": "WKB",
            "geometry_types": types,
        });
        let (crs_v, vendor) = &self.crs_meta;
        if let Some(v) = crs_v {
            col_meta["crs"] = v.clone();
        }
        if let Some((proj4, name)) = vendor {
            col_meta["geopq:crs"] = json!({ "proj4": proj4, "name": name });
        }
        if let Some(b) = self.bbox {
            col_meta["bbox"] = json!(b);
        }
        let geo = json!({
            "version": "1.1.0",
            "primary_column": self.geom_name,
            "columns": { self.geom_name.clone(): col_meta },
        });
        // KV metadata must go through the writer; arrow schema metadata is
        // not copied into the parquet footer.
        self.writer
            .append_key_value_metadata(parquet::file::metadata::KeyValue::new(
                "geo".to_string(),
                geo.to_string(),
            ));
        self.writer
            .close()
            .map_err(|e| format!("parquet close: {e}"))?;
        Ok(self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::open_store_for_test;
    use crate::sql::engine::{run_query_for_test, SqlLayer};

    /// Query a fixture, export the result, load it back through the real
    /// store opener: schema, rows and CRS must survive the roundtrip.
    #[test]
    fn result_roundtrips_through_loader() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/polygons_5k_l93.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _, _) = open_store_for_test(&path).unwrap();
        let layers = vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: None,
        }];
        let out = run_query_for_test(
            "select st_centroid(geometry) geometry, st_area(geometry) a from t limit 100",
            &layers,
        )
        .unwrap();
        let (geom_col, crs) = out.geom.clone().expect("geometry detected");

        let dir = std::env::temp_dir();
        let dst = dir.join("geopq_sql_export_test.parquet");
        write_result(&dst, &out.schema, std::slice::from_ref(&out.batch), geom_col, &crs).unwrap();

        let (store2, crs2, _, _) = open_store_for_test(&dst).unwrap();
        assert_eq!(store2.total_rows(), 100);
        assert_eq!(crs2.epsg, Some(2154), "CRS carried through");
        let geoms = store2.fetch_geoms(&[0, 50, 99]).unwrap();
        for (row, g) in geoms {
            assert!(
                matches!(g.expect("non-null"), geo_types::Geometry::Point(_)),
                "row {row}: centroid should be a point"
            );
        }
        let _ = std::fs::remove_file(&dst);
    }

    fn read_geo_meta(path: &Path) -> serde_json::Value {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let b =
            ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        let kv = b
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .iter()
            .find(|kv| kv.key == "geo")
            .expect("geo metadata");
        serde_json::from_str(kv.value.as_deref().unwrap()).unwrap()
    }

    /// A source-file PROJJSON must round-trip verbatim through the export;
    /// id-only CRSs fall back to a reference typed by is_latlong.
    #[test]
    fn crs_projjson_passthrough_and_fallback() {
        use crate::data::merge::crs_to_geo;
        let projjson = json!({
            "type": "ProjectedCRS", "name": "RGF93 / Lambert-93",
            "id": {"authority": "EPSG", "code": 2154},
            "base_crs": {"name": "RGF93", "type": "GeographicCRS"},
        });
        let crs = Crs::from_geoparquet_crs(Some(&projjson)).unwrap();
        assert_eq!(
            crs_to_geo(&crs),
            (Some(projjson.clone()), None),
            "verbatim passthrough"
        );
        // Id-only geographic CRS: best-effort reference, honestly typed.
        let (v, vendor) = crs_to_geo(&Crs::from_epsg(4269).unwrap());
        let v = v.expect("non-4326 CRS must be recorded");
        assert_eq!(v["type"], "GeographicCRS", "{v}");
        assert_eq!(v["id"]["code"], 4269);
        assert!(vendor.is_none());
        // Id-only projected CRS keeps the projected type.
        let (v, _) = crs_to_geo(&Crs::from_epsg(2154).unwrap());
        assert_eq!(v.expect("recorded")["type"], "ProjectedCRS");
        // 4326 without source PROJJSON still omits (spec default CRS84).
        assert_eq!(crs_to_geo(&Crs::wgs84()), (None, None));
        // Explicit `crs: null` source stays null; the CRS84 render
        // fallback must not be laundered into a vendor proj4 claim.
        let undef = Crs::from_geoparquet_crs(Some(&Value::Null)).unwrap();
        assert_eq!(crs_to_geo(&undef), (Some(Value::Null), None));
    }

    /// A proj4-only CRS (loaded via the `geopq:crs` vendor key, e.g. an
    /// ESRI shapefile import) must survive "Result as layer": the export
    /// writes `crs: null` + the vendor key, and reloading resolves it
    /// instead of falling back to CRS84 (which rejected every projected
    /// coordinate as an out-of-range lon/lat).
    #[test]
    fn vendor_proj4_crs_round_trips() {
        use arrow::datatypes::{DataType, Field, Schema};
        let mass = "+proj=lcc +lat_1=41.71666666666667 +lat_2=42.68333333333333 \
                    +lat_0=41 +lon_0=-71.5 +x_0=200000 +y_0=750000 +ellps=GRS80 \
                    +units=m +no_defs";
        let crs = Crs::from_proj4(mass, None, "NAD83 / Mass Mainland (ESRI)").unwrap();
        // 2D WKB point in StatePlane meters.
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&1u32.to_le_bytes());
        wkb.extend_from_slice(&231000.0f64.to_le_bytes());
        wkb.extend_from_slice(&900000.0f64.to_le_bytes());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "geometry",
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(BinaryArray::from_iter_values([&wkb]))],
        )
        .unwrap();
        let dst = std::env::temp_dir().join("geopq_sql_export_vendor_crs_test.parquet");
        write_result(&dst, &schema, &[batch], 0, &crs).unwrap();

        let geo = read_geo_meta(&dst);
        let col = &geo["columns"]["geometry"];
        assert!(col["crs"].is_null(), "explicit crs: null, {col}");
        assert_eq!(col["geopq:crs"]["proj4"], mass);

        let (_, crs2, _, _) = open_store_for_test(&dst).unwrap();
        assert!(!crs2.is_latlong, "vendor CRS resolved, not CRS84 fallback");
        assert!(crs2.proj4.contains("+proj=lcc"), "{}", crs2.proj4);
        let _ = std::fs::remove_file(&dst);
    }

    /// Z WKB passes through the export verbatim, so geometry_types needs
    /// the " Z" suffix.
    #[test]
    fn z_wkb_export_types_get_suffix() {
        use arrow::array::BinaryArray;
        use arrow::datatypes::{DataType, Field, Schema};
        let mut wkbs: Vec<Vec<u8>> = Vec::new();
        for i in 0..3 {
            // ISO POINT Z (type code 1001).
            let mut b = vec![1u8];
            b.extend_from_slice(&1001u32.to_le_bytes());
            b.extend_from_slice(&(i as f64).to_le_bytes());
            b.extend_from_slice(&1.0f64.to_le_bytes());
            b.extend_from_slice(&5.0f64.to_le_bytes());
            wkbs.push(b);
        }
        let schema = Arc::new(Schema::new(vec![Field::new(
            "geometry",
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(BinaryArray::from_iter_values(wkbs.iter()))],
        )
        .unwrap();
        let dst = std::env::temp_dir().join("geopq_sql_export_z_test.parquet");
        write_result(&dst, &schema, &[batch], 0, &Crs::wgs84()).unwrap();
        let geo = read_geo_meta(&dst);
        assert_eq!(
            geo["columns"]["geometry"]["geometry_types"],
            json!(["Point Z"])
        );
        let _ = std::fs::remove_file(&dst);
    }
}

#[cfg(test)]
mod join_tests {
    use super::*;
    use crate::data::attrs;
    use crate::data::loader::open_store_for_test;
    use crate::data::source::Source;
    use crate::sql::engine::{run_join_for_test, SqlLayer, SqlTable};

    /// The whole workflow a CSV attribute takes to reach the map: open the
    /// file as a table, join it to a layer, export the result, reopen it.
    ///
    /// Each half is covered elsewhere; what this pins is that the joined
    /// column survives to the far end. Styling reads columns off the
    /// reopened store, so a value that gets as far as the query result and
    /// no further would look like a working join right up to the point of
    /// being useless.
    #[test]
    fn a_csv_attribute_survives_all_the_way_onto_the_map() {
        let fixture = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/polygons_5k_l93.parquet"
        ));
        if !fixture.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let dir = std::env::temp_dir().join(format!("geopq_join_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // The CSV a user would drop on the window.
        let csv = dir.join("bands.csv");
        std::fs::write(&csv, "band,label,score\n0,west,10.5\n1,east,20.5\n").unwrap();
        let src = Source::Local(csv.clone());
        let preview = attrs::inspect(&src).expect("csv inspects");
        let data = attrs::import(&src, &preview.plan).expect("csv imports");
        let t = attrs::AttrTable::new(0, "bands".into(), src, data);
        let tables = vec![SqlTable {
            table: "bands".into(),
            schema: Arc::clone(&t.schema),
            batches: Arc::clone(&t.batches),
        }];

        let (store, crs, _, _) = open_store_for_test(&fixture).unwrap();
        let layers = vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: None,
        }];

        // Join on a key the layer computes, keeping the geometry.
        let out = run_join_for_test(
            "select t.geometry, b.label, b.score from t \
             join bands b on b.band = case when st_xmin(t.geometry) > 0 then 1 else 0 end \
             where t.geometry is not null limit 200",
            &layers,
            &tables,
        )
        .expect("join runs");
        let (geom_col, out_crs) = out.geom.clone().expect("result is geometry");

        let dst = dir.join("joined.parquet");
        write_result(
            &dst,
            &out.schema,
            std::slice::from_ref(&out.batch),
            geom_col,
            &out_crs,
        )
        .expect("exports");

        // Reopened the way "Result as layer" reopens it.
        let (store2, crs2, _, _) = open_store_for_test(&dst).expect("reopens as a layer");
        assert_eq!(store2.total_rows(), 200);
        assert_eq!(crs2.epsg, Some(2154), "CRS survives the join and the export");

        // The CSV's columns are on the layer, with their inferred types
        // intact — `score` must still be a number, or classified styling
        // has nothing to classify.
        let names: Vec<&str> = store2
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(names.contains(&"label"), "csv label column: {names:?}");
        assert!(names.contains(&"score"), "csv score column: {names:?}");
        let score = store2
            .schema
            .field_with_name("score")
            .expect("score field")
            .data_type()
            .clone();
        assert_eq!(score, arrow::datatypes::DataType::Float64, "still numeric");

        // And the values arrived, not just the columns.
        let batches = store2.fetch(&[0, 1, 2], None).expect("read back");
        let col = batches[0]
            .column_by_name("label")
            .expect("label column")
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .expect("utf8");
        assert!(
            (0..col.len()).all(|i| col.value(i) == "west" || col.value(i) == "east"),
            "labels came from the CSV",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
