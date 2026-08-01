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
#[derive(Clone)]
pub struct SqlLayer {
    /// Sanitized identifier the layer is registered under.
    pub table: String,
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
    /// Metadata row-group bboxes (index-aligned), for predicate pushdown.
    pub rg_bboxes: Option<Arc<Vec<[f64; 4]>>>,
}

/// An attribute table visible to SQL: columns only, no geometry and no
/// map presence. The small side of a join.
#[derive(Clone)]
pub struct SqlTable {
    /// Sanitized identifier the table is registered under.
    pub table: String,
    pub schema: SchemaRef,
    pub batches: Arc<Vec<RecordBatch>>,
}

#[derive(Debug)]
pub struct QueryOutput {
    pub schema: SchemaRef,
    /// The materialized result as one concatenated batch (display cap
    /// applied); pagination/sorting are view concerns over it.
    pub batch: RecordBatch,
    pub total_rows: usize,
    pub truncated: bool,
    pub elapsed_ms: u64,
    /// Result geometry column (index, CRS carried from the source layer),
    /// when one was detected — enables "add as layer".
    pub geom: Option<(usize, Crs)>,
}

/// A finished background job.
pub enum SqlDone {
    Query(QueryOutput),
    /// Full-result export (re-runs the query, streaming): temp file + rows.
    Export {
        path: std::path::PathBuf,
        #[allow(dead_code)] // reported in logs/tests; UI shows the layer
        rows: usize,
    },
}

pub struct SqlMsg {
    pub id: u64,
    pub result: Result<SqlDone, String>,
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

/// One column brought across by a join.
pub struct JoinField {
    /// Its name in the attribute table.
    pub source: String,
    /// Its name in the result. Differs when the layer already has one.
    pub out: String,
}

/// The SQL for joining an attribute table onto a layer.
///
/// `left` keeps every feature and leaves the added columns NULL where the
/// table has no match, which is the only safe default: an inner join with
/// a mismatched key produces an empty layer, and an empty layer looks
/// like a styling problem rather than a key problem.
///
/// Built as text rather than through a plan builder because it is also
/// the answer to "what did that button do" — it is shown in the dialog,
/// and it can be pasted into the console and edited.
pub fn join_sql(
    layer_table: &str,
    layer_key: &str,
    attr_table: &str,
    attr_key: &str,
    fields: &[JoinField],
    keep_unmatched: bool,
) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('"', "\"\"\""));
    let mut cols = vec!["l.*".to_string()];
    for f in fields {
        if f.source == f.out {
            cols.push(format!("t.{}", q(&f.source)));
        } else {
            cols.push(format!("t.{} as {}", q(&f.source), q(&f.out)));
        }
    }
    format!(
        "select {}\nfrom {layer_table} l\n{} join {attr_table} t on l.{} = t.{}",
        cols.join(", "),
        if keep_unmatched { "left" } else { "inner" },
        q(layer_key),
        q(attr_key),
    )
}

/// How many of the layer's rows the table has a match for.
///
/// Run before the join, because a key that matches nothing is the single
/// most common way a join goes wrong and the only symptom is an empty or
/// blank-styled layer afterwards.
pub fn match_count_sql(layer_table: &str, layer_key: &str, attr_table: &str, attr_key: &str) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('"', "\"\"\""));
    format!(
        "select count(*) total, count(t.{}) matched from {layer_table} l \
         left join {attr_table} t on l.{} = t.{}",
        q(attr_key),
        q(layer_key),
        q(attr_key),
    )
}

/// Run `query` on its own thread; the result arrives on `tx`, then
/// `on_done` runs (e.g. to wake the UI).
pub fn spawn_query(
    id: u64,
    query: String,
    layers: Vec<SqlLayer>,
    tables: Vec<SqlTable>,
    tx: Sender<SqlMsg>,
    on_done: impl FnOnce() + Send + 'static,
) {
    std::thread::Builder::new()
        .name("sql-query".into())
        .spawn(move || {
            let result = run_query_with_tables(&query, &layers, &tables).map(SqlDone::Query);
            let _ = tx.send(SqlMsg { id, result });
            on_done();
        })
        .expect("spawn sql thread");
}

/// Re-run `query` and stream the FULL result (no display cap) to a
/// GeoParquet file on a worker thread — "add as layer" beyond what the
/// grid materialized.
pub fn spawn_export(
    id: u64,
    query: String,
    layers: Vec<SqlLayer>,
    tables: Vec<SqlTable>,
    path: std::path::PathBuf,
    tx: Sender<SqlMsg>,
    on_done: impl FnOnce() + Send + 'static,
) {
    std::thread::Builder::new()
        .name("sql-export".into())
        .spawn(move || {
            let result = run_export(&query, &layers, &tables, &path)
                .map(|rows| SqlDone::Export { path, rows });
            let _ = tx.send(SqlMsg { id, result });
            on_done();
        })
        .expect("spawn sql thread");
}

/// Layer-filter computation result: the file rows matching a predicate,
/// as group-relative ranges per row group.
pub struct FilterRows {
    pub matched: usize,
    /// Index-aligned with the file's row groups.
    pub per_group: Vec<Vec<(u32, u32)>>,
}

pub struct FilterMsg {
    pub layer_id: u64,
    pub predicate: String,
    pub result: Result<FilterRows, String>,
}

/// Evaluate `predicate` against one layer and report the matching global
/// rows (background thread). Spatial predicates prune via pushdown, so a
/// location filter on a huge file only reads the relevant row groups.
pub fn spawn_row_filter(
    layer_id: u64,
    layer: SqlLayer,
    predicate: String,
    tx: Sender<FilterMsg>,
    on_done: impl FnOnce() + Send + 'static,
) {
    std::thread::Builder::new()
        .name("sql-layer-filter".into())
        .spawn(move || {
            let result = run_row_filter(&layer, &predicate);
            let _ = tx.send(FilterMsg {
                layer_id,
                predicate,
                result,
            });
            on_done();
        })
        .expect("spawn sql thread");
}

pub(crate) fn run_row_filter(layer: &SqlLayer, predicate: &str) -> Result<FilterRows, String> {
    use super::table::{LayerTable, ROW_INDEX_COL};
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;
    let config = SessionConfig::new().with_target_partitions(num_cpus());
    let ctx = SessionContext::new_with_config(config);
    udf::register_all(&ctx);
    ctx.register_table(
        layer.table.as_str(),
        Arc::new(LayerTable::with_row_index(
            Arc::clone(&layer.store),
            layer.rg_bboxes.clone(),
        )),
    )
    .map_err(|e| format!("register {}: {e}", layer.table))?;

    let sql = format!(
        "select \"{ROW_INDEX_COL}\" from {} where ({predicate})",
        layer.table
    );
    let mut rows: Vec<u32> = rt.block_on(async {
        let df = ctx.sql(&sql).await.map_err(fmt_df_err)?;
        let mut stream = df.execute_stream().await.map_err(fmt_df_err)?;
        let mut rows: Vec<u32> = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(fmt_df_err)?;
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .ok_or("row index column has an unexpected type")?;
            rows.extend(col.values().iter().copied());
        }
        Ok::<_, String>(rows)
    })?;
    // Partitions stream in arbitrary order.
    rows.sort_unstable();

    // Split into group-relative ranges.
    let starts = layer.store.rg_starts();
    let n_groups = starts.len().saturating_sub(1);
    let mut per_group: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_groups];
    let mut g = 0usize;
    let mut run: Option<(u32, u32)> = None; // group-relative [start, end)
    for &r in &rows {
        let r = r as u64;
        while g + 1 < n_groups && r >= starts[g + 1] {
            if let Some(rn) = run.take() {
                per_group[g].push(rn);
            }
            g += 1;
        }
        let rel = (r - starts[g]) as u32;
        run = match run {
            Some((s, e)) if rel == e => Some((s, e + 1)),
            Some(rn) => {
                per_group[g].push(rn);
                Some((rel, rel + 1))
            }
            None => Some((rel, rel + 1)),
        };
    }
    if let Some(rn) = run {
        per_group[g].push(rn);
    }
    Ok(FilterRows {
        matched: rows.len(),
        per_group,
    })
}

/// Quoted SQL identifier for a store-schema column name: mapped through
/// the same lowercase + collision-dedupe renaming the registered table
/// applies, with embedded double quotes doubled.
fn sql_ident(layer: &SqlLayer, column: &str) -> String {
    let name = layer
        .store
        .schema
        .fields()
        .iter()
        .position(|f| f.name() == column)
        .map(|i| super::table::sql_column_names(&layer.store.schema)[i].clone())
        // Unknown store name: fall back to plain lowercasing and let the
        // query surface the error.
        .unwrap_or_else(|| column.to_lowercase());
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Distinct-value count per column (partition-field candidates in the
/// export dialog). One scan, all aggregates at once. Blocking — call from
/// a worker thread. Columns are store-schema names.
pub fn distinct_counts(
    layer: &SqlLayer,
    columns: &[String],
) -> Result<std::collections::HashMap<String, usize>, String> {
    if columns.is_empty() {
        return Ok(Default::default());
    }
    let aggs = columns
        .iter()
        .map(|c| format!("count(distinct {})", sql_ident(layer, c)))
        .collect::<Vec<_>>()
        .join(", ");
    let out = run_query(
        &format!("select {aggs} from {}", layer.table),
        std::slice::from_ref(layer),
    )?;
    let mut map = std::collections::HashMap::new();
    for (i, c) in columns.iter().enumerate() {
        if let Some(a) = out
            .batch
            .column(i)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
        {
            if a.len() > 0 {
                map.insert(c.clone(), a.value(0).max(0) as usize);
            }
        }
    }
    Ok(map)
}

/// Most frequent non-null values of a column (categorical styling).
/// Blocking — run off the UI thread. `column` is the store-schema name.
pub fn top_values(
    layer: &SqlLayer,
    column: &str,
    limit: usize,
) -> Result<Vec<String>, String> {
    let col = sql_ident(layer, column);
    let out = run_query(
        &format!(
            "select cast({col} as varchar) v, count(*) c from {} \
             where {col} is not null group by v order by c desc, v limit {limit}",
            layer.table
        ),
        std::slice::from_ref(layer),
    )?;
    // DataFusion may produce Utf8View (varchar) or dictionary-encoded
    // group keys; normalize to plain Utf8 before reading.
    let col0 = arrow::compute::cast(out.batch.column(0), &arrow::datatypes::DataType::Utf8)
        .map_err(|e| format!("category values: {e}"))?;
    let vals = col0
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or("unexpected type for category values")?;
    Ok((0..vals.len())
        .filter(|&i| !vals.is_null(i))
        .map(|i| vals.value(i).to_string())
        .collect())
}

#[cfg(test)]
pub fn run_query_for_test(query: &str, layers: &[SqlLayer]) -> Result<QueryOutput, String> {
    run_query(query, layers)
}

#[cfg(test)]
pub fn run_join_for_test(
    query: &str,
    layers: &[SqlLayer],
    tables: &[SqlTable],
) -> Result<QueryOutput, String> {
    run_query_with_tables(query, layers, tables)
}

#[cfg(test)]
pub fn run_export_for_test(
    query: &str,
    layers: &[SqlLayer],
    path: &std::path::Path,
) -> Result<usize, String> {
    run_export(query, layers, &[], path)
}

fn make_ctx(layers: &[SqlLayer], tables: &[SqlTable]) -> Result<SessionContext, String> {
    // DataFusion partitions the scan per row group already; keep its own
    // repartitioning from shuffling small interactive results.
    let config = SessionConfig::new().with_target_partitions(num_cpus());
    let ctx = SessionContext::new_with_config(config);
    udf::register_all(&ctx);
    super::agg::register_all(&ctx);
    for l in layers {
        ctx.register_table(
            l.table.as_str(),
            Arc::new(LayerTable::new(Arc::clone(&l.store), l.rg_bboxes.clone())),
        )
        .map_err(|e| format!("register {}: {e}", l.table))?;
    }
    for t in tables {
        // Already in memory, so a `MemTable` is the whole provider: there
        // is no footer to seek and no viewport to prune by.
        let mem = datafusion::datasource::MemTable::try_new(
            Arc::clone(&t.schema),
            vec![t.batches.as_ref().clone()],
        )
        .map_err(|e| format!("register {}: {e}", t.table))?;
        ctx.register_table(t.table.as_str(), Arc::new(mem))
            .map_err(|e| format!("register {}: {e}", t.table))?;
    }
    Ok(ctx)
}

/// Layers only. The internal callers (distinct counts, top values, the
/// tests) have no attribute tables in play.
fn run_query(query: &str, layers: &[SqlLayer]) -> Result<QueryOutput, String> {
    run_query_with_tables(query, layers, &[])
}

fn run_query_with_tables(
    query: &str,
    layers: &[SqlLayer],
    tables: &[SqlTable],
) -> Result<QueryOutput, String> {
    let started = Instant::now();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;
    let ctx = make_ctx(layers, tables)?;

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

    let batch = arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| format!("result concat: {e}"))?;
    let geom = detect_geometry(&schema, &batch, layers, query);
    Ok(QueryOutput {
        schema,
        batch,
        total_rows,
        truncated,
        elapsed_ms: started.elapsed().as_millis() as u64,
        geom,
    })
}

/// Execute and stream every result row into a GeoParquet file.
fn run_export(
    query: &str,
    layers: &[SqlLayer],
    tables: &[SqlTable],
    path: &std::path::Path,
) -> Result<usize, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;
    let ctx = make_ctx(layers, tables)?;
    rt.block_on(async {
        let df = ctx.sql(query).await.map_err(fmt_df_err)?;
        let mut stream = df.execute_stream().await.map_err(fmt_df_err)?;
        let schema = stream.schema();
        // Geometry detection needs a data sample: peek the first batch.
        let mut first = None;
        while let Some(b) = stream.next().await {
            let b = b.map_err(fmt_df_err)?;
            if b.num_rows() > 0 {
                first = Some(b);
                break;
            }
        }
        let Some(first) = first else {
            return Err("no rows to export".into());
        };
        let (geom_col, crs) = detect_geometry(&schema, &first, layers, query)
            .ok_or("no geometry column in result")?;
        let mut writer = super::export::StreamWriter::new(path, &schema, geom_col, &crs)?;
        writer.write(&first)?;
        while let Some(b) = stream.next().await {
            writer.write(&b.map_err(fmt_df_err)?)?;
        }
        writer.finish()
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
    batch: &RecordBatch,
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
        let arr = batch.column(i);
        let arr = arr.as_any().downcast_ref::<arrow::array::BinaryArray>();
        arr.is_some_and(|arr| {
            (0..arr.len())
                .find(|&r| !arr.is_null(r))
                .is_some_and(|r| decode_wkb(arr.value(r)).is_some())
        })
    })?;

    let q = query.to_ascii_lowercase();
    // Match table names as whole identifiers, not substrings — `roads`
    // must not shadow `roads_2` (the console's dedupe suffixes make
    // prefix collisions routine). First registered match wins.
    let tokens: std::collections::HashSet<&str> = q
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .collect();
    let crs = layers
        .iter()
        .find(|l| tokens.contains(l.table.as_str()))
        .map(|l| l.crs.clone())
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
        out.batch
            .column(col)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
    }

    /// Set operations, checked by area rather than by geometry equality:
    /// the exact vertex list an overlay produces is an implementation
    /// detail, the area it covers is the answer.
    ///
    /// No fixture and no FROM clause — these are pure functions over two
    /// literals, so there is nothing to load.
    #[test]
    fn polygon_set_operations() {
        // Two 2×2 squares overlapping in a 1×2 strip.
        let a = "st_geomfromtext('POLYGON((0 0,2 0,2 2,0 2,0 0))')";
        let b = "st_geomfromtext('POLYGON((1 0,3 0,3 2,1 2,1 0))')";
        let out = run_query(
            &format!(
                "select st_area(st_union({a}, {b})) u, \
                 st_area(st_intersection({a}, {b})) i, \
                 st_area(st_difference({a}, {b})) d, \
                 st_area(st_symdifference({a}, {b})) x"
            ),
            &[],
        )
        .unwrap();
        assert_eq!(get_f64(&out, 0), 6.0, "union");
        assert_eq!(get_f64(&out, 1), 2.0, "intersection");
        assert_eq!(get_f64(&out, 2), 2.0, "a minus b");
        assert_eq!(get_f64(&out, 3), 4.0, "symmetric difference");

        // Disjoint squares union to both, and intersect to nothing rather
        // than to NULL: an empty result is an answer.
        let far = "st_geomfromtext('POLYGON((9 9,10 9,10 10,9 10,9 9))')";
        let out = run_query(
            &format!(
                "select st_area(st_union({a}, {far})) u, \
                 st_area(st_intersection({a}, {far})) i"
            ),
            &[],
        )
        .unwrap();
        assert_eq!(get_f64(&out, 0), 5.0, "disjoint union keeps both");
        assert_eq!(get_f64(&out, 1), 0.0, "disjoint intersection is empty");
    }

    /// A line has no area, so there is no region it shares with a polygon.
    /// NULL says that; zero would claim they were checked and found not to
    /// overlap.
    #[test]
    fn set_operations_on_non_areal_input_are_null() {
        let poly = "st_geomfromtext('POLYGON((0 0,2 0,2 2,0 2,0 0))')";
        let line = "st_geomfromtext('LINESTRING(0 0, 2 2)')";
        let out = run_query(
            &format!("select st_union({poly}, {line}) u, st_intersection({line}, {line}) i"),
            &[],
        )
        .unwrap();
        assert!(out.batch.column(0).is_null(0), "polygon ∪ line");
        assert!(out.batch.column(1).is_null(0), "line ∩ line");
    }

    /// Aggregates over a group, on a real layer so the partial-aggregate
    /// path is exercised: DataFusion partitions the scan per row group,
    /// so each partition accumulates its own share and the states have to
    /// merge correctly. A single-partition test would pass with a broken
    /// `merge_batch`.
    #[test]
    fn spatial_aggregates_dissolve_and_bound_a_layer() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let out = run_query(
            "select st_area(st_union_agg(geometry)) dissolved, \
             sum(st_area(geometry)) summed, \
             st_area(st_extent(geometry)) extent_area, \
             st_xmin(st_extent(geometry)) ex0, min(st_xmin(geometry)) mn0, \
             st_xmax(st_extent(geometry)) ex1, max(st_xmax(geometry)) mx1 \
             from t where geometry is not null",
            &layers,
        )
        .unwrap();
        let (dissolved, summed) = (get_f64(&out, 0), get_f64(&out, 1));
        assert!(dissolved > 0.0, "dissolve produced nothing");
        // Overlaps are counted once by a dissolve and twice by a sum, so
        // the dissolve can only be smaller. Equal would mean the polygons
        // happen not to overlap; larger means the union invented area.
        assert!(
            dissolved <= summed * 1.000_001,
            "dissolved {dissolved} exceeds summed {summed}",
        );
        // The extent must contain every feature, so it agrees with the
        // per-row min/max taken the ordinary way.
        assert_eq!(get_f64(&out, 3), get_f64(&out, 4), "extent xmin");
        assert_eq!(get_f64(&out, 5), get_f64(&out, 6), "extent xmax");
        assert!(get_f64(&out, 2) >= dissolved, "extent area covers the union");

        // Grouped, and the result is still geometry: this is the shape a
        // dissolve-by-attribute takes, and "add as layer" needs the
        // geometry column detected on it.
        let out = run_query(
            "select st_union_agg(geometry) geometry, count(*) n from t \
             where geometry is not null group by st_xmin(geometry) > 0",
            &layers,
        )
        .unwrap();
        assert!(out.total_rows >= 1, "at least one group");
        let (col, crs) = out.geom.as_ref().expect("aggregate result is geometry");
        assert_eq!(out.schema.field(*col).name(), "geometry");
        assert_eq!(crs.epsg, Some(2154), "CRS carried from the source layer");
    }

    /// The merge path, which a single-partition query never reaches.
    ///
    /// Each partition accumulates its own share and hands over a partial
    /// state; only `merge_batch` combines them. `union all` splits the
    /// input into two partitions, and unioning a set with a copy of itself
    /// must change nothing — so a merge that dropped or double-counted a
    /// state would show up as a different area. The 1M-point fixture adds
    /// a genuine nine-row-group scan for the extent.
    #[test]
    fn aggregate_states_merge_across_partitions() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let out = run_query(
            "select st_area(st_union_agg(geometry)) one, \
             st_xmin(st_extent(geometry)) x0, st_ymax(st_extent(geometry)) y1 \
             from t where geometry is not null",
            &layers,
        )
        .unwrap();
        let (one, x0, y1) = (get_f64(&out, 0), get_f64(&out, 1), get_f64(&out, 2));

        let out = run_query(
            "select st_area(st_union_agg(geometry)) two, \
             st_xmin(st_extent(geometry)) x0, st_ymax(st_extent(geometry)) y1 from (\
               select geometry from t where geometry is not null \
               union all \
               select geometry from t where geometry is not null)",
            &layers,
        )
        .unwrap();
        assert_eq!(get_f64(&out, 0), one, "a set unioned with itself is itself");
        assert_eq!(get_f64(&out, 1), x0, "extent xmin across partitions");
        assert_eq!(get_f64(&out, 2), y1, "extent ymax across partitions");

        // Union and extent are idempotent, so neither check above can tell
        // a merge that dropped a state from one that counted it twice —
        // and a ratio cannot either, since a uniform merge bug inflates
        // both sides equally. Collect against a plain sum is absolute:
        // every part is kept, so the collected area is the summed area,
        // and any state lost or repeated moves it.
        let out = run_query(
            "select st_area(st_collect(geometry)) collected, \
             sum(st_area(geometry)) summed from t where geometry is not null",
            &layers,
        )
        .unwrap();
        let (collected, summed) = (get_f64(&out, 0), get_f64(&out, 1));
        assert!(summed > 0.0, "fixture has area");
        // Relative: summing 5k polygon areas in a different order moves
        // the last bits, and that is not what this is testing.
        assert!(
            (collected - summed).abs() <= summed * 1e-12,
            "every partition's state, exactly once: {collected} vs {summed}",
        );

        // Nine row groups, so nine partial extents to merge, checked
        // against the same bounds computed row by row.
        let Some(points) = fixture("points_1m_wgs84.parquet") else {
            return;
        };
        assert!(
            points[0].store.rg_starts().len() > 2,
            "the points fixture should span several row groups",
        );
        let out = run_query(
            "select st_xmin(st_extent(geometry)) ex0, min(st_xmin(geometry)) mn0, \
             st_ymax(st_extent(geometry)) ey1, max(st_ymax(geometry)) mx1 \
             from t where geometry is not null",
            &points,
        )
        .unwrap();
        assert_eq!(get_f64(&out, 0), get_f64(&out, 1), "merged extent xmin");
        assert_eq!(get_f64(&out, 2), get_f64(&out, 3), "merged extent ymax");
    }

    /// An empty group has no geometry to report. NULL says that; a zero
    /// polygon would be a shape that is not there.
    #[test]
    fn aggregates_over_nothing_are_null() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let out = run_query(
            "select st_union_agg(geometry) u, st_extent(geometry) e, \
             st_collect(geometry) c from t where false",
            &layers,
        )
        .unwrap();
        for (i, name) in ["union", "extent", "collect"].iter().enumerate() {
            assert!(out.batch.column(i).is_null(0), "{name} over no rows");
        }
    }

    /// `st_collect` keeps the parts; `st_union_agg` merges them. Two
    /// overlapping squares collect to two polygons' worth of area and
    /// dissolve to less.
    #[test]
    fn collect_keeps_what_union_merges() {
        let out = run_query(
            "select st_area(st_union_agg(g)) u, st_area(st_collect(g)) c from (\
               select st_geomfromtext('POLYGON((0 0,2 0,2 2,0 2,0 0))') g \
               union all \
               select st_geomfromtext('POLYGON((1 0,3 0,3 2,1 2,1 0))') g\
             )",
            &[],
        )
        .unwrap();
        assert_eq!(get_f64(&out, 0), 6.0, "dissolved counts the overlap once");
        assert_eq!(get_f64(&out, 1), 8.0, "collected counts both squares");
    }

    /// The join a table exists for: a layer keyed to a lookup table,
    /// producing a result that is still geometry and so can go back on
    /// the map.
    ///
    /// This is the whole point of attribute tables, so it is checked end
    /// to end rather than by unit: real layer, real table, one query.
    #[test]
    fn a_table_joins_to_a_layer_and_the_result_is_still_a_layer() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};

        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        // A key every row of the layer can compute, and a table that
        // labels each of its values.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("band", DataType::Int64, false),
                Field::new("label", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![0_i64, 1])),
                Arc::new(StringArray::from(vec!["west", "east"])),
            ],
        )
        .unwrap();
        let tables = vec![SqlTable {
            table: "bands".into(),
            schema: batch.schema(),
            batches: Arc::new(vec![batch]),
        }];

        let out = run_query_with_tables(
            "select b.label, count(*) n, st_union_agg(t.geometry) geometry              from t join bands b                on b.band = case when st_xmin(t.geometry) > 0 then 1 else 0 end              where t.geometry is not null              group by b.label order by b.label",
            &layers,
            &tables,
        )
        .unwrap();

        assert!(out.total_rows > 0, "the join matched nothing");
        // The label came from the table, so the join really did reach it.
        let labels = out
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("label column");
        let seen: Vec<&str> = (0..labels.len()).map(|i| labels.value(i)).collect();
        assert!(
            seen.iter().all(|l| *l == "west" || *l == "east"),
            "labels came from the table: {seen:?}",
        );
        // Every layer row is accounted for: a join that dropped rows would
        // still look plausible without this.
        let counts = out
            .batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column");
        let total: i64 = (0..counts.len()).map(|i| counts.value(i)).sum();
        let expect = run_query(
            "select count(*) c from t where geometry is not null",
            &layers,
        )
        .unwrap();
        let expect = expect
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(total, expect, "the join kept every row");

        // And the result is still a layer: geometry detected, CRS carried
        // from the layer side even though the join added a table.
        let (col, crs) = out.geom.as_ref().expect("joined result is geometry");
        assert_eq!(out.schema.field(*col).name(), "geometry");
        assert_eq!(crs.epsg, Some(2154));
    }

    #[test]
    fn join_sql_keeps_every_feature_and_avoids_name_clashes() {
        let fields = vec![
            JoinField { source: "_2025".into(), out: "_2025".into() },
            // The layer already has a `nom`, so the added one is suffixed
            // rather than silently shadowing it.
            JoinField { source: "nom".into(), out: "nom_2".into() },
        ];
        let sql = join_sql("communes", "code_insee", "nais", "code", &fields, true);
        assert!(sql.contains("left join"), "{sql}");
        assert!(sql.contains("select l.*, t.\"_2025\", t.\"nom\" as \"nom_2\""), "{sql}");
        assert!(sql.contains("on l.\"code_insee\" = t.\"code\""), "{sql}");
        // Dropping unmatched is the other option, never the default.
        let sql = join_sql("communes", "code_insee", "nais", "code", &fields, false);
        assert!(sql.contains("inner join"), "{sql}");
    }

    /// The generated SQL has to run, not just look right.
    #[test]
    fn generated_join_sql_executes_and_counts_matches() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{Field, Schema};
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        // A key the layer can be joined on, and a table covering half of it.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("band", DataType::Int64, false),
                Field::new("label", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1_i64])),
                Arc::new(StringArray::from(vec!["east"])),
            ],
        )
        .unwrap();
        let tables = vec![SqlTable {
            table: "bands".into(),
            schema: batch.schema(),
            batches: Arc::new(vec![batch]),
        }];
        // A view of the layer exposing a key as a plain column. Lambert-93
        // puts every xmin well above zero, so the key has to come from
        // something that actually varies.
        let keyed = "(select geometry, cast(st_xmin(geometry) as bigint) % 2 band                      from t where geometry is not null)";

        let count = match_count_sql(keyed, "band", "bands", "band");
        let out = run_query_with_tables(&count, &layers, &tables).unwrap();
        let total = out.batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
        let matched = out.batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
        assert!(total > 0, "layer has rows");
        assert!(matched > 0 && matched < total, "half covered: {matched} of {total}");

        // The left join keeps every feature, matched or not.
        let fields = vec![JoinField { source: "label".into(), out: "label".into() }];
        let sql = join_sql(keyed, "band", "bands", "band", &fields, true);
        let out = run_query_with_tables(&sql, &layers, &tables).unwrap();
        assert_eq!(out.total_rows as i64, total, "nothing was dropped");
        let labels = out
            .batch
            .column_by_name("label")
            .expect("the joined column")
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let nulls = (0..labels.len()).filter(|&i| labels.is_null(i)).count();
        assert_eq!(nulls as i64, total - matched, "unmatched rows are NULL, not gone");
        // Still a layer.
        assert!(out.geom.is_some(), "the result carries geometry");

        // And the inner join drops exactly those.
        let sql = join_sql(keyed, "band", "bands", "band", &fields, false);
        let out = run_query_with_tables(&sql, &layers, &tables).unwrap();
        assert_eq!(out.total_rows as i64, matched);
    }

    #[test]
    fn count_star_matches_store() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let total = layers[0].store.total_rows();
        let out = run_query("select count(*) c from t", &layers).unwrap();
        use arrow::array::Int64Array;
        let c = out.batch
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
        let c = out.batch
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
        let w = out.batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(w, "POINT(1 2)");
        assert_eq!(get_f64(&out, 1), 5.0);
        let bt = out.batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(bt, "MultiPolygon");
    }

    #[test]
    fn st_transform_roundtrip_and_lambert93() {
        // 4326 -> 3857 -> 4326 must return the input (needs no fixture).
        let out = run_query(
            "select st_x(p) x, st_y(p) y from (select st_transform(\
             st_transform(st_point(2.349014, 48.864716), 'EPSG:4326', 'EPSG:3857'),\
             '3857', 'EPSG:4326') p)",
            &[],
        )
        .unwrap();
        assert!((get_f64(&out, 0) - 2.349014).abs() < 1e-6);
        assert!((get_f64(&out, 1) - 48.864716).abs() < 1e-6);

        // Paris in Lambert-93, against pyproj-computed coordinates.
        let out = run_query(
            "select st_x(p) x, st_y(p) y from (select \
             st_transform(st_point(2.349014, 48.864716), 'EPSG:4326', 'EPSG:2154') p)",
            &[],
        )
        .unwrap();
        assert!((get_f64(&out, 0) - 652_242.70).abs() < 1.0, "{}", get_f64(&out, 0));
        assert!((get_f64(&out, 1) - 6_862_939.61).abs() < 1.0, "{}", get_f64(&out, 1));

        // Unknown code fails loudly rather than silently passing through.
        assert!(run_query(
            "select st_transform(st_point(0, 0), 'EPSG:999999', 'EPSG:4326')",
            &[],
        )
        .is_err());
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
            out.batch
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

    /// The streaming full-result export ("Result as layer" beyond the
    /// display cap) must write every matching row with the CRS intact.
    #[test]
    fn full_export_streams_all_rows() {
        let Some(layers) = fixture("polygons_5k_l93.parquet") else {
            return;
        };
        let path = std::env::temp_dir().join("geopq_sql_full_export_test.parquet");
        let rows = run_export_for_test(
            "select * from t where st_area(geometry) > 0",
            &layers,
            &path,
        )
        .unwrap();
        assert_eq!(rows as u64, layers[0].store.total_rows());
        let (store, crs, _, _) = crate::data::loader::open_store_for_test(&path).unwrap();
        assert_eq!(store.total_rows(), rows as u64);
        assert_eq!(crs.epsg, Some(2154), "CRS carried through");
        let _ = std::fs::remove_file(&path);
    }

    /// Layer filter: the computed per-group row ranges must cover exactly
    /// the rows the equivalent COUNT query matches, and decoding one range
    /// must yield rows that satisfy the predicate.
    #[test]
    fn row_filter_matches_query_count() {
        let Some(layers) = fixture("parcels_hilbert.parquet") else {
            return;
        };
        let b = layers[0].rg_bboxes.as_ref().expect("metadata bboxes")[0];
        let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        let (dx, dy) = ((b[2] - b[0]) * 0.1, (b[3] - b[1]) * 0.1);
        let pred = format!(
            "st_intersects(geometry, st_makeenvelope({}, {}, {}, {}))",
            cx - dx,
            cy - dy,
            cx + dx,
            cy + dy
        );

        let rows = run_row_filter(&layers[0], &pred).unwrap();
        let expected = {
            let out =
                run_query(&format!("select count(*) c from t where {pred}"), &layers).unwrap();
            use arrow::array::Int64Array;
            out.batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0) as usize
        };
        assert_eq!(rows.matched, expected);
        assert!(rows.matched > 0);
        let range_sum: usize = rows
            .per_group
            .iter()
            .flatten()
            .map(|(s, e)| (e - s) as usize)
            .sum();
        assert_eq!(range_sum, rows.matched, "ranges cover every matched row");
        // Group-pruning sanity: a small window must not touch every group.
        let touched = rows.per_group.iter().filter(|g| !g.is_empty()).count();
        assert!(
            touched < rows.per_group.len(),
            "small window touched {touched}/{} groups",
            rows.per_group.len()
        );
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
        let names = out.batch.column(0).as_any().downcast_ref::<SA>().unwrap();
        assert_eq!((names.value(0), names.value(1)), ("a", "c"));
        let _ = std::fs::remove_file(&path);
    }

    /// Write a small GeoParquet file with the given string columns plus a
    /// point geometry, and open it as a store.
    fn store_with_columns(
        file: &str,
        cols: &[(&str, Vec<&str>)],
    ) -> (std::path::PathBuf, Arc<crate::data::store::FeatureStore>) {
        use arrow::array::{BinaryBuilder, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        let n = cols[0].1.len();
        let mut geoms = BinaryBuilder::new();
        let wopts = wkb::writer::WriteOptions::default();
        for i in 0..n {
            let mut buf = Vec::new();
            wkb::writer::write_geometry(
                &mut buf,
                &geo_types::Geometry::Point(geo_types::Point::new(i as f64, 1.0)),
                &wopts,
            )
            .unwrap();
            geoms.append_value(&buf);
        }
        let mut fields: Vec<Field> = cols
            .iter()
            .map(|(name, _)| Field::new(*name, DataType::Utf8, false))
            .collect();
        fields.push(Field::new("geometry", DataType::Binary, true));
        let schema = Arc::new(Schema::new(fields));
        let mut arrays: Vec<Arc<dyn Array>> = cols
            .iter()
            .map(|(_, vals)| Arc::new(StringArray::from(vals.clone())) as _)
            .collect();
        arrays.push(Arc::new(geoms.finish()));
        let batch = arrow::record_batch::RecordBatch::try_new(Arc::clone(&schema), arrays)
            .unwrap();
        let path = std::env::temp_dir().join(file);
        crate::sql::export::write_result(&path, &schema, &[batch], cols.len(), &Crs::wgs84())
            .unwrap();
        let (store, _, _, _) = crate::data::loader::open_store_for_test(&path).unwrap();
        (path, Arc::new(store))
    }

    /// The result CRS must come from the table the query actually names —
    /// whole-identifier match, not substring (`roads` vs `roads_2`).
    #[test]
    fn result_crs_matches_whole_table_identifier() {
        let (path, store) =
            store_with_columns("geopq_sql_crs_ident_test.parquet", &[("name", vec!["a", "b"])]);
        let layers = vec![
            SqlLayer {
                table: "roads".into(),
                store: Arc::clone(&store),
                crs: Crs::from_epsg(2154).unwrap(),
                rg_bboxes: None,
            },
            SqlLayer {
                table: "roads_2".into(),
                store,
                crs: Crs::wgs84(),
                rg_bboxes: None,
            },
        ];
        let out = run_query("select * from roads_2", &layers).unwrap();
        let (_, crs) = out.geom.expect("geometry detected");
        assert_eq!(crs.epsg, Some(4326), "roads_2 must not match roads");
        let out = run_query("select * from roads", &layers).unwrap();
        let (_, crs) = out.geom.expect("geometry detected");
        assert_eq!(crs.epsg, Some(2154));
        let _ = std::fs::remove_file(&path);
    }

    /// Store-name → sql-name mapping in top_values/distinct_counts: a
    /// NAME/name case collision resolves to the deduped `name_2`, and a
    /// column name containing a double quote is escaped, not spliced raw.
    #[test]
    fn top_values_maps_and_escapes_identifiers() {
        let (path, store) = store_with_columns(
            "geopq_sql_ident_map_test.parquet",
            &[
                ("NAME", vec!["up", "up", "up"]),
                ("name", vec!["low2", "low1", "low2"]),
                ("va\"l", vec!["q", "q", "q"]),
            ],
        );
        let layer = SqlLayer {
            table: "t".into(),
            store,
            crs: Crs::wgs84(),
            rg_bboxes: None,
        };
        // The second collision column queries name_2, not NAME.
        let vals = top_values(&layer, "name", 5).unwrap();
        assert_eq!(vals, vec!["low2".to_string(), "low1".to_string()]);
        let vals = top_values(&layer, "NAME", 5).unwrap();
        assert_eq!(vals, vec!["up".to_string()]);
        // Embedded double quote survives as a quoted identifier.
        let vals = top_values(&layer, "va\"l", 5).unwrap();
        assert_eq!(vals, vec!["q".to_string()]);
        let counts = distinct_counts(
            &layer,
            &["NAME".to_string(), "name".to_string(), "va\"l".to_string()],
        )
        .unwrap();
        assert_eq!(counts.get("NAME"), Some(&1));
        assert_eq!(counts.get("name"), Some(&2));
        assert_eq!(counts.get("va\"l"), Some(&1));
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

    /// Live remote query probe, opt-in:
    /// GEOPQ_QUERY_URI=https://.../roads.parquet \
    ///   GEOPQ_QUERY_SQL="select count(*) from t where ref='MA 2'" \
    ///   cargo test --release remote_query_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn remote_query_probe() {
        let Ok(url) = std::env::var("GEOPQ_QUERY_URI") else {
            eprintln!("set GEOPQ_QUERY_URI");
            return;
        };
        let sql = std::env::var("GEOPQ_QUERY_SQL")
            .unwrap_or_else(|_| "select count(*) from t".into());
        let src = crate::data::source::Source::Remote { url, len: 0 }
            .resolve()
            .unwrap();
        let t0 = std::time::Instant::now();
        let (store, crs, _info, rg) =
            crate::data::loader::open_source_for_test(&src).unwrap();
        eprintln!(
            "open: {} ms; {} rows, {} groups",
            t0.elapsed().as_millis(),
            store.total_rows(),
            store.rg_starts().len() - 1
        );
        let layers = vec![SqlLayer {
            table: "t".into(),
            store: Arc::new(store),
            crs,
            rg_bboxes: rg.map(|(_, b)| Arc::new(b)),
        }];
        let t0 = std::time::Instant::now();
        let out = run_query(&sql, &layers).unwrap();
        eprintln!(
            "query: {} ms, {} rows",
            t0.elapsed().as_millis(),
            out.total_rows
        );
    }
}
