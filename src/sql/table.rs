//! DataFusion table over a loaded layer's `FeatureStore`.
//!
//! Scans stream row groups through the store's cached parquet footer, so a
//! query touches the same code path (and the same remote range-request
//! machinery) as map loading. One partition per row group lets DataFusion
//! parallelize the scan. The geometry column is always exposed as plain
//! WKB `Binary`, whatever the file's native encoding, so the ST_* UDFs see
//! a single representation.
//!
//! Spatial predicate pushdown: `st_intersects/within/contains/dwithin`
//! filters against a literal geometry (constant-folded from
//! `st_makeenvelope`, `st_geomfromtext`, ...) are turned into a bbox that
//! prunes row groups via their metadata bboxes and, on covering-column
//! files, rows via the bbox-leaf scan — the same machinery as viewport
//! loading. Pushdown is *inexact*: DataFusion re-applies the exact
//! predicate on the surviving rows.

use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DfResult, ScalarValue};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;

use crate::data::geoarrow::{GeomCol, GeomEncoding};
use crate::data::loader::{covering_select, decode_wkb, intersecting_rgs};
use crate::data::store::FeatureStore;

const BATCH_SIZE: usize = 8192;

/// Synthetic global-row-index column (layer filters resolve SQL matches
/// back to file rows through it). Name chosen to never collide.
pub const ROW_INDEX_COL: &str = "___row";

/// A loaded layer registered as a SQL table.
pub struct LayerTable {
    store: Arc<FeatureStore>,
    /// Table schema: the file's arrow schema with the geometry field
    /// normalized to nullable `Binary` (WKB), plus the synthetic
    /// [`ROW_INDEX_COL`] when requested.
    schema: SchemaRef,
    /// Per-row-group bboxes (data CRS, index-aligned with the file's row
    /// groups) when the layer's metadata provides them; enables group
    /// pruning for pushed-down spatial predicates.
    rg_bboxes: Option<Arc<Vec<[f64; 4]>>>,
    /// Index of the synthetic row-index field in `schema`, if enabled.
    row_index_field: Option<usize>,
}

impl LayerTable {
    pub fn new(store: Arc<FeatureStore>, rg_bboxes: Option<Arc<Vec<[f64; 4]>>>) -> Self {
        Self::build(store, rg_bboxes, false)
    }

    /// A table that also exposes [`ROW_INDEX_COL`]: each row's global
    /// index in the file (for layer filters).
    pub fn with_row_index(
        store: Arc<FeatureStore>,
        rg_bboxes: Option<Arc<Vec<[f64; 4]>>>,
    ) -> Self {
        Self::build(store, rg_bboxes, true)
    }

    fn build(
        store: Arc<FeatureStore>,
        rg_bboxes: Option<Arc<Vec<[f64; 4]>>>,
        row_index: bool,
    ) -> Self {
        // Column names are lowercased so unquoted SQL identifiers (which
        // DataFusion normalizes to lowercase) match files with uppercase
        // or mixed-case columns; collisions get _2, _3, ... suffixes.
        let mut seen: std::collections::HashMap<String, usize> = Default::default();
        let mut fields: Vec<Field> = store
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let base = f.name().to_lowercase();
                let n = seen.entry(base.clone()).or_insert(0);
                *n += 1;
                let name = if *n > 1 { format!("{base}_{n}") } else { base };
                if i == store.geom_col {
                    Field::new(name, DataType::Binary, true)
                } else {
                    f.as_ref().clone().with_name(name)
                }
            })
            .collect();
        let row_index_field = row_index.then(|| {
            fields.push(Field::new(ROW_INDEX_COL, DataType::UInt32, false));
            fields.len() - 1
        });
        let schema = Arc::new(Schema::new(fields));
        Self {
            store,
            schema,
            rg_bboxes,
            row_index_field,
        }
    }

    fn geom_name(&self) -> &str {
        self.schema.field(self.store.geom_col).name()
    }

    /// Table-schema names of the virtual hive partition columns (they sit
    /// right after the base fields, before the row-index column).
    fn part_field_names(&self) -> Vec<String> {
        let base = self.store.base_fields();
        (0..self.store.part_cols.len())
            .map(|k| self.schema.field(base + k).name().clone())
            .collect()
    }
}

impl fmt::Debug for LayerTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LayerTable({})", self.store.source.name())
    }
}

#[async_trait]
impl TableProvider for LayerTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        let part_names = self.part_field_names();
        Ok(filters
            .iter()
            .map(|f| {
                let mut eq = Vec::new();
                part_eq_constraints(f, &part_names, &mut eq);
                if filter_bbox(f, self.geom_name()).is_some() || !eq.is_empty() {
                    // We only prune candidates; the exact predicate must
                    // still run on the surviving rows.
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let projection: Vec<usize> = match projection {
            Some(p) => p.clone(),
            None => (0..self.schema.fields().len()).collect(),
        };
        let out_schema = Arc::new(self.schema.project(&projection)?);

        // Columns are decoded in file order whatever the requested order;
        // remember where each output column comes from — the decoded batch,
        // a hive partition value, or the synthetic row-index (the latter
        // two are not file columns).
        let base = self.store.base_fields();
        let mut read_cols: Vec<usize> = projection
            .iter()
            .copied()
            .filter(|&p| p < base)
            .collect();
        read_cols.sort_unstable();
        read_cols.dedup();
        let reorder: Vec<ColSrc> = projection
            .iter()
            .map(|p| {
                if Some(*p) == self.row_index_field {
                    ColSrc::RowIndex
                } else if *p >= base {
                    ColSrc::Part(*p - base)
                } else {
                    ColSrc::Read(read_cols.binary_search(p).expect("projected col present"))
                }
            })
            .collect();
        let geom_read_idx = read_cols
            .iter()
            .position(|&c| c == self.store.geom_col);

        let n_groups = self.store.rg_starts().len().saturating_sub(1);

        // Spatial pushdown: intersect the bboxes of every applicable
        // filter, then prune groups (metadata bboxes) and rows (covering
        // bbox-leaf scan) exactly like a viewport load.
        let bbox = filters
            .iter()
            .filter_map(|f| filter_bbox(f, self.geom_name()))
            .reduce(|a, b| [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])]);

        // Hive partition pruning: equality constraints on partition
        // columns skip whole files' row groups before any IO.
        let part_names = self.part_field_names();
        let mut part_eq: Vec<(usize, String)> = Vec::new();
        for f in filters {
            part_eq_constraints(f, &part_names, &mut part_eq);
        }
        let group_kept = |g: usize| {
            part_eq
                .iter()
                .all(|(k, v)| self.store.part_value(g, *k) == Some(v.as_str()))
        };

        let parts: Vec<GroupPart> = match bbox {
            Some(r) if r[0] <= r[2] && r[1] <= r[3] => {
                let groups: Vec<u32> = match &self.rg_bboxes {
                    Some(b) => intersecting_rgs(b, r),
                    None => (0..n_groups as u32).collect(),
                };
                groups
                    .into_iter()
                    .filter(|&g| group_kept(g as usize))
                    .filter_map(|g| {
                        match covering_select(&self.store, g, r) {
                            Ok(Some(ranges)) if ranges.is_empty() => None, // no rows hit
                            Ok(ranges) => Some(Ok(GroupPart {
                                group: g as usize,
                                ranges,
                            })),
                            Err(e) => Some(Err(DataFusionError::Execution(e))),
                        }
                    })
                    .collect::<DfResult<_>>()?
            }
            // Contradictory bbox (empty intersection): scan nothing.
            Some(_) => Vec::new(),
            None => (0..n_groups)
                .filter(|&g| group_kept(g))
                .map(|g| GroupPart {
                    group: g,
                    ranges: None,
                })
                .collect(),
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&out_schema)),
            Partitioning::UnknownPartitioning(parts.len().max(1)),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Arc::new(LayerScanExec {
            store: Arc::clone(&self.store),
            out_schema,
            read_cols,
            reorder,
            geom_read_idx,
            parts,
            properties: Arc::new(properties),
        }))
    }
}

/// One scan partition: a row group, optionally narrowed to row ranges by
/// the covering bbox-leaf scan.
#[derive(Clone, Debug)]
struct GroupPart {
    group: usize,
    ranges: Option<Vec<(u32, u32)>>,
}

/// The pruning bbox implied by one pushed-down filter, if any: a spatial
/// predicate relating the geometry column to a literal WKB geometry.
/// Conjunctions recurse; any other shape yields None.
fn filter_bbox(expr: &Expr, geom_name: &str) -> Option<[f64; 4]> {
    match expr {
        Expr::BinaryExpr(b) if b.op == datafusion::logical_expr::Operator::And => {
            let l = filter_bbox(&b.left, geom_name);
            let r = filter_bbox(&b.right, geom_name);
            match (l, r) {
                (Some(a), Some(b)) => {
                    Some([a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])])
                }
                (one, None) | (None, one) => one,
            }
        }
        Expr::ScalarFunction(f) => {
            let name = f.func.name();
            let spatial = matches!(
                name,
                "st_intersects" | "st_within" | "st_contains" | "st_dwithin"
            );
            if !spatial {
                return None;
            }
            // One side must be the geometry column, the other a literal.
            let (a, b) = (f.args.first()?, f.args.get(1)?);
            let lit = if is_geom_column(a, geom_name) {
                literal_geom_bbox(b)?
            } else if is_geom_column(b, geom_name) {
                literal_geom_bbox(a)?
            } else {
                return None;
            };
            let pad = if name == "st_dwithin" {
                match f.args.get(2)? {
                    Expr::Literal(ScalarValue::Float64(Some(d)), _) => d.abs(),
                    _ => return None,
                }
            } else {
                0.0
            };
            Some([lit[0] - pad, lit[1] - pad, lit[2] + pad, lit[3] + pad])
        }
        _ => None,
    }
}

fn is_geom_column(expr: &Expr, geom_name: &str) -> bool {
    matches!(expr, Expr::Column(c) if c.name == geom_name)
}

fn literal_geom_bbox(expr: &Expr) -> Option<[f64; 4]> {
    use geo::BoundingRect;
    let Expr::Literal(ScalarValue::Binary(Some(bytes)), _) = expr else {
        return None;
    };
    let r = decode_wkb(bytes)?.bounding_rect()?;
    Some([r.min().x, r.min().y, r.max().x, r.max().y])
}

/// Equality constraints on hive partition columns implied by a filter:
/// `col = 'literal'` conjuncts (both orders), recursing through AND.
/// Anything else (OR, NOT, ...) contributes nothing — pruning must only
/// ever use constraints the whole filter requires.
fn part_eq_constraints(expr: &Expr, part_names: &[String], out: &mut Vec<(usize, String)>) {
    match expr {
        Expr::BinaryExpr(b) if b.op == datafusion::logical_expr::Operator::And => {
            part_eq_constraints(&b.left, part_names, out);
            part_eq_constraints(&b.right, part_names, out);
        }
        Expr::BinaryExpr(b) if b.op == datafusion::logical_expr::Operator::Eq => {
            let col_lit = |a: &Expr, b: &Expr| -> Option<(usize, String)> {
                let (Expr::Column(c), Expr::Literal(ScalarValue::Utf8(Some(v)), _)) = (a, b)
                else {
                    return None;
                };
                part_names.iter().position(|n| *n == c.name).map(|k| (k, v.clone()))
            };
            if let Some(c) = col_lit(&b.left, &b.right).or_else(|| col_lit(&b.right, &b.left)) {
                out.push(c);
            }
        }
        _ => {}
    }
}

/// Physical scan: partition N streams row group N from the store.
/// Where one output column comes from.
#[derive(Clone, Copy, Debug)]
enum ColSrc {
    /// Position in the decoded parquet batch.
    Read(usize),
    /// Hive partition column `k`: constant per row group's fragment.
    Part(usize),
    /// The synthetic global row index.
    RowIndex,
}

struct LayerScanExec {
    store: Arc<FeatureStore>,
    out_schema: SchemaRef,
    /// File-order arrow root indices to decode.
    read_cols: Vec<usize>,
    /// For each output column, where it comes from.
    reorder: Vec<ColSrc>,
    /// Position of the geometry column in the decoded batch, if projected.
    geom_read_idx: Option<usize>,
    /// Row groups (with optional row ranges) surviving predicate pushdown;
    /// one partition each.
    parts: Vec<GroupPart>,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for LayerScanExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LayerScanExec({})", self.store.source.name())
    }
}

impl DisplayAs for LayerScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let ranged = self.parts.iter().filter(|p| p.ranges.is_some()).count();
        write!(
            f,
            "LayerScanExec: {} ({} of {} row groups, {ranged} row-pruned)",
            self.store.source.name(),
            self.parts.len(),
            self.store.rg_starts().len().saturating_sub(1),
        )
    }
}

impl ExecutionPlan for LayerScanExec {
    fn name(&self) -> &str {
        "LayerScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let iter = GroupIter {
            store: Arc::clone(&self.store),
            part: self.parts.get(partition).cloned(),
            out_schema: Arc::clone(&self.out_schema),
            read_cols: self.read_cols.clone(),
            reorder: self.reorder.clone(),
            geom_read_idx: self.geom_read_idx,
            reader: None,
            done: false,
            sel_cursor: (0, 0),
            consumed: 0,
            synth_left: None,
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.out_schema),
            futures::stream::iter(iter),
        )))
    }
}

/// Blocking iterator over one row group's batches; the parquet reader is
/// opened lazily on the first pull so planning never touches the network.
struct GroupIter {
    store: Arc<FeatureStore>,
    /// None when the scan has zero partitions (empty file or contradictory
    /// pushdown): the single mandatory partition yields nothing.
    part: Option<GroupPart>,
    out_schema: SchemaRef,
    read_cols: Vec<usize>,
    reorder: Vec<ColSrc>,
    geom_read_idx: Option<usize>,
    reader: Option<ParquetRecordBatchReader>,
    done: bool,
    /// Row-index tracking through a range selection: (range idx, offset).
    sel_cursor: (usize, u32),
    /// Rows emitted so far (contiguous case).
    consumed: u32,
    /// Remaining rows to synthesize when no file columns are projected.
    synth_left: Option<usize>,
}

impl GroupIter {
    /// Global row indices of the next `n` emitted rows, advancing the
    /// cursor: contiguous from the group start, or mapped through the
    /// row-range selection.
    fn take_positions(&mut self, n: usize) -> Vec<u32> {
        let Some(part) = &self.part else {
            return Vec::new();
        };
        let start = self.store.rg_starts()[part.group] as u32;
        let mut out = Vec::with_capacity(n);
        match &part.ranges {
            None => {
                for i in 0..n as u32 {
                    out.push(start + self.consumed + i);
                }
                self.consumed += n as u32;
            }
            Some(ranges) => {
                let (mut ri, mut off) = self.sel_cursor;
                while out.len() < n && ri < ranges.len() {
                    let (s, e) = ranges[ri];
                    let pos = s + off;
                    if pos < e {
                        out.push(start + pos);
                        off += 1;
                    } else {
                        ri += 1;
                        off = 0;
                    }
                }
                self.sel_cursor = (ri, off);
            }
        }
        out
    }

    /// Keep the cursor in sync when the row index isn't projected.
    fn advance_positions(&mut self, n: usize) {
        let _ = self.take_positions(n);
    }

    fn next_inner(&mut self) -> Result<Option<RecordBatch>, String> {
        let Some(part) = &self.part else {
            return Ok(None);
        };
        if part.group >= self.store.rg_starts().len().saturating_sub(1) {
            return Ok(None);
        }
        let group = part.group;

        // Nothing to read from the file (partition / row-index columns
        // only): emit batches straight from the known row counts.
        let (n_rows, cols): (usize, Vec<ArrayRef>) = if self.read_cols.is_empty() {
            let left = *self.synth_left.get_or_insert_with(|| {
                match &part.ranges {
                    Some(r) => r.iter().map(|&(s, e)| (e - s) as usize).sum(),
                    None => {
                        let starts = self.store.rg_starts();
                        (starts[group + 1] - starts[group]) as usize
                    }
                }
            });
            if left == 0 {
                return Ok(None);
            }
            let n = left.min(BATCH_SIZE);
            self.synth_left = Some(left - n);
            (n, Vec::new())
        } else {
            if self.reader.is_none() {
                self.reader = Some(self.store.reader_for_group(
                    group,
                    BATCH_SIZE,
                    part.ranges.as_deref(),
                    Some(&self.read_cols),
                )?);
            }
            let Some(batch) = self.reader.as_mut().unwrap().next() else {
                return Ok(None);
            };
            let batch = batch.map_err(|e| format!("parquet decode error: {e}"))?;
            let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
            if let Some(gi) = self.geom_read_idx {
                cols[gi] = normalize_geometry(cols[gi].as_ref(), self.store.encoding)?;
            }
            (batch.num_rows(), cols)
        };

        let needs_row_index = self
            .reorder
            .iter()
            .any(|s| matches!(s, ColSrc::RowIndex));
        let row_index: Option<ArrayRef> = if needs_row_index {
            Some(Arc::new(arrow::array::UInt32Array::from(
                self.take_positions(n_rows),
            )))
        } else {
            self.advance_positions(n_rows);
            None
        };
        let out_cols: Vec<ArrayRef> = self
            .reorder
            .iter()
            .map(|src| match src {
                ColSrc::Read(i) => Arc::clone(&cols[*i]),
                ColSrc::Part(k) => {
                    let v = self.store.part_value(group, *k);
                    Arc::new(arrow::array::StringArray::from(vec![v; n_rows])) as ArrayRef
                }
                ColSrc::RowIndex => Arc::clone(row_index.as_ref().expect("built above")),
            })
            .collect();
        let opts = arrow::record_batch::RecordBatchOptions::new()
            .with_row_count(Some(n_rows));
        RecordBatch::try_new_with_options(Arc::clone(&self.out_schema), out_cols, &opts)
            .map(Some)
            .map_err(|e| format!("batch shape error: {e}"))
    }
}

impl Iterator for GroupIter {
    type Item = DfResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.next_inner() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(DataFusionError::Execution(e)))
            }
        }
    }
}

/// Re-encode a geometry column as plain WKB `Binary`, whatever its stored
/// encoding (GeoArrow native arrays, LargeBinary WKB, ...).
fn normalize_geometry(col: &dyn Array, encoding: GeomEncoding) -> Result<ArrayRef, String> {
    if encoding == GeomEncoding::Wkb && col.data_type() == &DataType::Binary {
        // Already the exposed representation: zero-copy.
        return Ok(arrow::array::make_array(col.to_data()));
    }
    let geoms = GeomCol::new(col, encoding)
        .ok_or("geometry column does not match its declared encoding")?;
    let wopts = wkb::writer::WriteOptions::default();
    let mut b = BinaryBuilder::new();
    let mut buf: Vec<u8> = Vec::new();
    for i in 0..col.len() {
        match geoms.geometry(i) {
            Some(g) => {
                buf.clear();
                wkb::writer::write_geometry(&mut buf, &g, &wopts)
                    .map_err(|e| format!("wkb encode error: {e}"))?;
                b.append_value(&buf);
            }
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::open_store_for_test;
    use datafusion::logical_expr::col;
    use datafusion::execution::FunctionRegistry;
    use datafusion::prelude::SessionContext;

    fn envelope_wkb(b: [f64; 4]) -> Vec<u8> {
        let r = geo_types::Rect::new((b[0], b[1]), (b[2], b[3]));
        let mut buf = Vec::new();
        wkb::writer::write_geometry(
            &mut buf,
            &geo_types::Geometry::Polygon(r.to_polygon()),
            &wkb::writer::WriteOptions::default(),
        )
        .unwrap();
        buf
    }

    /// Pushing an st_intersects filter must prune row groups (metadata
    /// bboxes) and rows (covering leaves) at plan time, and a contradictory
    /// conjunction must plan an empty scan.
    #[test]
    fn spatial_filter_prunes_partitions() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parcels_hilbert.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, _crs, _info, rg_meta) = open_store_for_test(&path).unwrap();
        let (_, boxes) = rg_meta.expect("hilbert fixture has metadata bboxes");
        let n_groups = boxes.len();
        let store = Arc::new(store);
        assert!(store.covering.is_some(), "fixture has a covering column");
        let table = LayerTable::new(Arc::clone(&store), Some(Arc::new(boxes.clone())));

        let ctx = SessionContext::new();
        crate::sql::udf::register_all(&ctx);
        let st_intersects = ctx.udf("st_intersects").unwrap();
        let state = ctx.state();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();

        // A small window around the center of the first row group.
        let b0 = boxes[0];
        let (cx, cy) = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0);
        let (dx, dy) = ((b0[2] - b0[0]) * 0.05, (b0[3] - b0[1]) * 0.05);
        let small = [cx - dx, cy - dy, cx + dx, cy + dy];
        let filter = st_intersects.call(vec![
            col("geometry"),
            Expr::Literal(ScalarValue::Binary(Some(envelope_wkb(small))), None),
        ]);

        // Pushdown is advertised as inexact.
        let support = table.supports_filters_pushdown(&[&filter]).unwrap();
        assert_eq!(support, vec![TableProviderFilterPushDown::Inexact]);

        let plan = rt
            .block_on(table.scan(&state, None, std::slice::from_ref(&filter), None))
            .unwrap();
        let scan = plan.downcast_ref::<LayerScanExec>().unwrap();
        assert!(
            !scan.parts.is_empty() && scan.parts.len() < n_groups,
            "small window must prune groups: {} of {n_groups}",
            scan.parts.len()
        );
        assert!(
            scan.parts.iter().any(|p| p.ranges.is_some()),
            "covering file must row-prune surviving groups"
        );

        // No filter: every group scans, no row selection.
        let plan = rt.block_on(table.scan(&state, None, &[], None)).unwrap();
        let scan = plan.downcast_ref::<LayerScanExec>().unwrap();
        assert_eq!(scan.parts.len(), n_groups);
        assert!(scan.parts.iter().all(|p| p.ranges.is_none()));

        // Two disjoint windows AND-ed: contradictory, nothing to scan.
        let far = [b0[0] - 1e6, b0[1] - 1e6, b0[0] - 9e5, b0[1] - 9e5];
        let f2 = st_intersects.call(vec![
            col("geometry"),
            Expr::Literal(ScalarValue::Binary(Some(envelope_wkb(far))), None),
        ]);
        let plan = rt
            .block_on(table.scan(&state, None, &[filter, f2], None))
            .unwrap();
        let scan = plan.downcast_ref::<LayerScanExec>().unwrap();
        assert!(scan.parts.is_empty(), "disjoint bboxes scan nothing");
    }
}
