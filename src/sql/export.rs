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
    crs_meta: CrsOut,
    types: Vec<&'static str>,
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
            crs_meta: crs_value(crs),
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
        match &self.crs_meta {
            CrsOut::Omit => {}
            CrsOut::Value(v) => {
                col_meta["crs"] = v.clone();
            }
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

enum CrsOut {
    /// No `crs` key: spec default OGC:CRS84.
    Omit,
    Value(Value),
}

fn crs_value(crs: &Crs) -> CrsOut {
    if let Some(code) = crs.epsg {
        if code == 4326 {
            return CrsOut::Omit;
        }
        return CrsOut::Value(json!({
            "type": "ProjectedCRS",
            "name": crs.name,
            "id": { "authority": "EPSG", "code": code },
        }));
    }
    if crs.name.contains("undefined") {
        // Source layer had an explicit `crs: null`; stay honest.
        return CrsOut::Value(Value::Null);
    }
    CrsOut::Omit
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
}
