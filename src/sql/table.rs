//! DataFusion table over a loaded layer's `FeatureStore`.
//!
//! Scans stream row groups through the store's cached parquet footer, so a
//! query touches the same code path (and the same remote range-request
//! machinery) as map loading. One partition per row group lets DataFusion
//! parallelize the scan. The geometry column is always exposed as plain
//! WKB `Binary`, whatever the file's native encoding, so the ST_* UDFs see
//! a single representation.

use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DfResult};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;

use crate::data::geoarrow::{GeomCol, GeomEncoding};
use crate::data::store::FeatureStore;

const BATCH_SIZE: usize = 8192;

/// A loaded layer registered as a SQL table.
pub struct LayerTable {
    store: Arc<FeatureStore>,
    /// Table schema: the file's arrow schema with the geometry field
    /// normalized to nullable `Binary` (WKB).
    schema: SchemaRef,
}

impl LayerTable {
    pub fn new(store: Arc<FeatureStore>) -> Self {
        // Column names are lowercased so unquoted SQL identifiers (which
        // DataFusion normalizes to lowercase) match files with uppercase
        // or mixed-case columns; collisions get _2, _3, ... suffixes.
        let mut seen: std::collections::HashMap<String, usize> = Default::default();
        let fields: Vec<Field> = store
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
        let schema = Arc::new(Schema::new(fields));
        Self { store, schema }
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

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let projection: Vec<usize> = match projection {
            Some(p) => p.clone(),
            None => (0..self.schema.fields().len()).collect(),
        };
        let out_schema = Arc::new(self.schema.project(&projection)?);

        // Columns are decoded in file order whatever the requested order;
        // remember where each output column lands in the decoded batch.
        let mut read_cols: Vec<usize> = projection.clone();
        read_cols.sort_unstable();
        read_cols.dedup();
        let reorder: Vec<usize> = projection
            .iter()
            .map(|p| read_cols.binary_search(p).expect("projected col present"))
            .collect();
        let geom_read_idx = read_cols
            .iter()
            .position(|&c| c == self.store.geom_col);

        let n_groups = self.store.rg_starts().len().saturating_sub(1).max(1);
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&out_schema)),
            Partitioning::UnknownPartitioning(n_groups),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Arc::new(LayerScanExec {
            store: Arc::clone(&self.store),
            out_schema,
            read_cols,
            reorder,
            geom_read_idx,
            properties: Arc::new(properties),
        }))
    }
}

/// Physical scan: partition N streams row group N from the store.
struct LayerScanExec {
    store: Arc<FeatureStore>,
    out_schema: SchemaRef,
    /// File-order arrow root indices to decode.
    read_cols: Vec<usize>,
    /// For each output column, its position in the decoded batch.
    reorder: Vec<usize>,
    /// Position of the geometry column in the decoded batch, if projected.
    geom_read_idx: Option<usize>,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for LayerScanExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LayerScanExec({})", self.store.source.name())
    }
}

impl DisplayAs for LayerScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "LayerScanExec: {} ({} row groups)",
            self.store.source.name(),
            self.properties.partitioning.partition_count()
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
            group: partition,
            out_schema: Arc::clone(&self.out_schema),
            read_cols: self.read_cols.clone(),
            reorder: self.reorder.clone(),
            geom_read_idx: self.geom_read_idx,
            reader: None,
            done: false,
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
    group: usize,
    out_schema: SchemaRef,
    read_cols: Vec<usize>,
    reorder: Vec<usize>,
    geom_read_idx: Option<usize>,
    reader: Option<ParquetRecordBatchReader>,
    done: bool,
}

impl GroupIter {
    fn next_inner(&mut self) -> Result<Option<RecordBatch>, String> {
        // Files can have zero row groups; the single partition is empty.
        if self.group >= self.store.rg_starts().len().saturating_sub(1) {
            return Ok(None);
        }
        if self.reader.is_none() {
            self.reader = Some(FeatureStore::open_reader_for_group(
                &self.store.source,
                &self.store.meta,
                self.group,
                BATCH_SIZE,
                None,
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
        let out_cols: Vec<ArrayRef> = self
            .reorder
            .iter()
            .map(|&i| Arc::clone(&cols[i]))
            .collect();
        let opts = arrow::record_batch::RecordBatchOptions::new()
            .with_row_count(Some(batch.num_rows()));
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
