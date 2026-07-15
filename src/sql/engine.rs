//! Query execution: a fresh DataFusion session per query, run on a worker
//! thread, results delivered over a channel like the layer loader.

use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Array;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::StreamExt;

use super::table::LayerTable;
use super::udf;
use crate::data::crs::Crs;
use crate::data::loader::decode_wkb;
use crate::data::store::FeatureStore;

/// Cap on collected result rows; queries keep running to completion for
/// aggregates, this only bounds what is materialized for display/export.
pub const MAX_RESULT_ROWS: usize = 500_000;

/// A layer visible to SQL.
pub struct SqlLayer {
    /// Sanitized identifier the layer is registered under.
    pub table: String,
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
    /// Metadata row-group bboxes (index-aligned), for predicate pushdown.
    pub rg_bboxes: Option<Arc<Vec<[f64; 4]>>>,
}

#[derive(Debug)]
pub struct QueryOutput {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
    pub total_rows: usize,
    pub truncated: bool,
    pub elapsed_ms: u64,
    /// Result geometry column (index, CRS carried from the source layer),
    /// when one was detected — enables "add as layer".
    pub geom: Option<(usize, Crs)>,
}

pub struct SqlMsg {
    pub id: u64,
    pub result: Result<QueryOutput, String>,
}

/// Turn a layer display name into a stable SQL identifier: lowercase
/// alphanumerics with everything else collapsed to `_`.
pub fn table_name(layer_name: &str) -> String {
    let base: String = layer_name
        .trim_end_matches(".parquet")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let base = base.trim_matches('_').to_string();
    let mut name = if base.is_empty() { "layer".into() } else { base };
    if name.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        name.insert(0, 't');
    }
    name
}

/// Run `query` on its own thread; the result arrives on `tx`, then
/// `on_done` runs (e.g. to wake the UI).
pub fn spawn_query(
    id: u64,
    query: String,
    layers: Vec<SqlLayer>,
    tx: Sender<SqlMsg>,
    on_done: impl FnOnce() + Send + 'static,
) {
    std::thread::Builder::new()
        .name("sql-query".into())
        .spawn(move || {
            let result = run_query(&query, &layers);
            let _ = tx.send(SqlMsg { id, result });
            on_done();
        })
        .expect("spawn sql thread");
}

#[cfg(test)]
pub fn run_query_for_test(query: &str, layers: &[SqlLayer]) -> Result<QueryOutput, String> {
    run_query(query, layers)
}

fn run_query(query: &str, layers: &[SqlLayer]) -> Result<QueryOutput, String> {
    let started = Instant::now();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;

    // DataFusion partitions the scan per row group already; keep its own
    // repartitioning from shuffling small interactive results.
    let config = SessionConfig::new().with_target_partitions(num_cpus());
    let ctx = SessionContext::new_with_config(config);
    udf::register_all(&ctx);
    for l in layers {
        ctx.register_table(
            l.table.as_str(),
            Arc::new(LayerTable::new(Arc::clone(&l.store), l.rg_bboxes.clone())),
        )
        .map_err(|e| format!("register {}: {e}", l.table))?;
    }

    let (schema, batches, total_rows, truncated) = rt.block_on(async {
        let df = ctx.sql(query).await.map_err(fmt_df_err)?;
        let mut stream = df.execute_stream().await.map_err(fmt_df_err)?;
        let schema = stream.schema();
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut rows = 0usize;
        let mut truncated = false;
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(fmt_df_err)?;
            if batch.num_rows() == 0 {
                continue;
            }
            if rows + batch.num_rows() > MAX_RESULT_ROWS {
                let keep = MAX_RESULT_ROWS - rows;
                if keep > 0 {
                    batches.push(batch.slice(0, keep));
                    rows += keep;
                }
                truncated = true;
                break;
            }
            rows += batch.num_rows();
            batches.push(batch);
        }
        Ok::<_, String>((schema, batches, rows, truncated))
    })?;

    let geom = detect_geometry(&schema, &batches, layers, query);
    Ok(QueryOutput {
        schema,
        batches,
        total_rows,
        truncated,
        elapsed_ms: started.elapsed().as_millis() as u64,
        geom,
    })
}

fn fmt_df_err(e: datafusion::common::DataFusionError) -> String {
    // Strip DataFusion's error-class prefix chatter for the console.
    let s = e.to_string();
    s.strip_prefix("Error during planning: ")
        .map(str::to_string)
        .unwrap_or(s)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Find the result's geometry column: a Binary column whose first non-null
/// value decodes as WKB, preferring well-known geometry column names. The
/// CRS is taken from the first registered layer the query text references.
fn detect_geometry(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    layers: &[SqlLayer],
    query: &str,
) -> Option<(usize, Crs)> {
    let mut candidates: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.data_type() == &DataType::Binary)
        .map(|(i, _)| i)
        .collect();
    // Prefer conventional names when several binary columns exist.
    candidates.sort_by_key(|&i| {
        let n = schema.field(i).name().to_ascii_lowercase();
        match n.as_str() {
            "geometry" | "geom" | "wkb_geometry" | "wkb" => 0,
            _ => 1,
        }
    });
    let col = candidates.into_iter().find(|&i| {
        batches.iter().any(|b| {
            let arr = b.column(i);
            let arr = arr.as_any().downcast_ref::<arrow::array::BinaryArray>();
            arr.is_some_and(|arr| {
                (0..arr.len())
                    .find(|&r| !arr.is_null(r))
                    .is_some_and(|r| decode_wkb(arr.value(r)).is_some())
            })
        })
    })?;

    let q = query.to_ascii_lowercase();
    let crs = layers
        .iter()
        .filter(|l| q.contains(&l.table))
        .map(|l| l.crs.clone())
        .next()
        .or_else(|| layers.first().map(|l| l.crs.clone()))?;
    Some((col, crs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::open_store_for_test;

    fn fixture(name: &str) -> Option<Vec<SqlLayer>> {
        let path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/"))
            .join(name);
        if !path.exists() {
            eprintln!("fixture {name} missing, skipping");
            return None;
        }
        let (store, crs, _info, rg_meta) = open_store_for_test(&path).unwrap();
        Some(vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: rg_meta.map(|(_, boxes)| Arc::new(boxes)),
        }])
    }

    fn get_f64(out: &QueryOutput, col: usize) -> f64 {
        use arrow::array::Float64Array;
        out.batches[0]
            .column(col)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
    }

    #[test]
    fn count_star_matches_store() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let total = layers[0].store.total_rows();
        let out = run_query("select count(*) c from t", &layers).unwrap();
        use arrow::array::Int64Array;
        let c = out.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(c as u64, total);
        assert!(out.geom.is_none(), "count result has no geometry");
    }

    #[test]
    fn spatial_functions_and_geometry_detection() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        // Envelope covering the whole file selects every row.
        let total = layers[0].store.total_rows();
        let out = run_query(
            "select count(*) c from t where st_intersects(geometry, \
             st_makeenvelope(st_xmin(geometry) - 1e9, -1e12, 1e12, 1e12))",
            &layers,
        )
        .unwrap();
        use arrow::array::Int64Array;
        let c = out.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(c as u64, total);

        // Areas are positive, centroids fall inside the feature bbox.
        let out = run_query(
            "select st_area(geometry) a, st_x(st_centroid(geometry)) cx, \
             st_xmin(geometry) x0, st_xmax(geometry) x1, geometry \
             from t where geometry is not null limit 10",
            &layers,
        )
        .unwrap();
        assert!(out.total_rows > 0 && !out.truncated);
        assert!(get_f64(&out, 0) > 0.0, "positive area");
        let (cx, x0, x1) = (get_f64(&out, 1), get_f64(&out, 2), get_f64(&out, 3));
        assert!(x0 <= cx && cx <= x1, "centroid inside bbox: {x0} {cx} {x1}");
        // The geometry column is detected with the layer's CRS (Lambert-93).
        let (col, crs) = out.geom.as_ref().expect("geometry detected");
        assert_eq!(out.schema.field(*col).name(), "geometry");
        assert_eq!(crs.epsg, Some(2154));
    }

    #[test]
    fn wkt_roundtrip_and_constructors() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let out = run_query(
            "select st_astext(st_geomfromtext('POINT(1 2)')) w, \
             st_distance(st_point(0, 0), st_point(3, 4)) d, \
             st_geometrytype(st_buffer(st_point(0, 0), 1)) bt",
            &layers,
        )
        .unwrap();
        use arrow::array::StringArray;
        let w = out.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(w, "POINT(1 2)");
        assert_eq!(get_f64(&out, 1), 5.0);
        let bt = out.batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(bt, "MultiPolygon");
    }

    /// Pushed-down spatial filters must return row-identical results to the
    /// same predicate evaluated without pruning (defeated via an OR the
    /// extractor doesn't handle), and fewer than all rows.
    #[test]
    fn pushdown_matches_full_scan() {
        let Some(layers) = fixture("parcels_hilbert.parquet") else {
            return;
        };
        if layers[0].rg_bboxes.is_none() {
            eprintln!("fixture has no metadata bboxes, skipping");
            return;
        }
        // A window around the first row group's center (data CRS).
        let b = layers[0].rg_bboxes.as_ref().unwrap()[0];
        let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        let (dx, dy) = ((b[2] - b[0]) * 0.1, (b[3] - b[1]) * 0.1);
        let env = format!(
            "st_makeenvelope({}, {}, {}, {})",
            cx - dx,
            cy - dy,
            cx + dx,
            cy + dy
        );
        let count = |pred: &str| -> i64 {
            let out = run_query(&format!("select count(*) c from t where {pred}"), &layers)
                .unwrap();
            use arrow::array::Int64Array;
            out.batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        };
        let pushed = count(&format!("st_intersects(geometry, {env})"));
        let control = count(&format!(
            "st_intersects(geometry, {env}) or geometry is null"
        ));
        let total = layers[0].store.total_rows() as i64;
        assert_eq!(pushed, control, "pruning must not drop matching rows");
        assert!(pushed > 0 && pushed < total, "window count: {pushed}/{total}");
    }

    #[test]
    fn uppercase_columns_are_queryable_lowercase() {
        use arrow::array::{BinaryBuilder, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};

        // A GeoParquet file with an uppercase attribute column.
        let mut geoms = BinaryBuilder::new();
        let wopts = wkb::writer::WriteOptions::default();
        for i in 0..3 {
            let mut buf = Vec::new();
            wkb::writer::write_geometry(
                &mut buf,
                &geo_types::Geometry::Point(geo_types::Point::new(i as f64, 1.0)),
                &wopts,
            )
            .unwrap();
            geoms.append_value(&buf);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("NAME", DataType::Utf8, false),
            Field::new("geometry", DataType::Binary, true),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(geoms.finish()),
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join("geopq_sql_uppercase_test.parquet");
        crate::sql::export::write_result(&path, &schema, &[batch], 1, &Crs::wgs84()).unwrap();

        let (store, crs, _, _) = crate::data::loader::open_store_for_test(&path).unwrap();
        let layers = vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: None,
        }];
        // Unquoted lowercase identifier reaches the uppercase file column.
        let out = run_query(
            "select name, st_x(geometry) x from t where name <> 'b' order by name",
            &layers,
        )
        .unwrap();
        assert_eq!(out.total_rows, 2);
        assert_eq!(out.schema.field(0).name(), "name");
        use arrow::array::StringArray as SA;
        let names = out.batches[0].column(0).as_any().downcast_ref::<SA>().unwrap();
        assert_eq!((names.value(0), names.value(1)), ("a", "c"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn projection_order_and_errors() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        // Reversed column order must come back in query order.
        let out = run_query(
            "select st_ymax(geometry) my, st_npoints(geometry) n from t limit 1",
            &layers,
        )
        .unwrap();
        assert_eq!(out.schema.field(0).name(), "my");
        assert_eq!(out.schema.field(1).name(), "n");

        let err = run_query("select nope from t", &layers).unwrap_err();
        assert!(err.to_lowercase().contains("nope"), "{err}");
        let err = run_query("select st_geomfromtext('POINT(banana)')", &layers).unwrap_err();
        assert!(err.contains("invalid WKT"), "{err}");
    }
}
