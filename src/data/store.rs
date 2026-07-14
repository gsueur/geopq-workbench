use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::arrow::ProjectionMask;

use super::geoarrow::{GeomCol, GeomEncoding};
use super::source::Source;

/// GeoParquet 1.1 covering bbox column: a root struct with four float
/// children giving each feature's bbox in the data CRS.
#[derive(Clone, Debug)]
pub struct CoveringCol {
    /// Arrow schema index of the root struct column.
    pub root: usize,
    /// Child field names for xmin, ymin, xmax, ymax (usually those names).
    pub children: [String; 4],
}

/// Lazy access to a GeoParquet file's rows.
///
/// Nothing but the schema and per-row-group row counts is kept in memory;
/// attribute and geometry values are re-read from the file on demand
/// (attribute panel, exact pick tests, projection rebuilds). Fetches select
/// only the row groups and pages containing the requested rows.
pub struct FeatureStore {
    pub source: Source,
    /// Parsed parquet footer (schema + metadata + page index), loaded once
    /// at open: every reader is built from this instead of re-reading the
    /// footer — essential for remote sources, whose footers can be MBs.
    pub meta: ArrowReaderMetadata,
    /// Index of the geometry column in the arrow schema.
    pub geom_col: usize,
    #[allow(dead_code)]
    pub schema: SchemaRef,
    /// Covering bbox column, when the file has one (per-feature pruning).
    pub covering: Option<CoveringCol>,
    /// Geometry column encoding (WKB or a GeoArrow native encoding).
    pub encoding: GeomEncoding,
    /// Rows per row group, in file order.
    rg_rows: Vec<u64>,
    /// Cumulative start row of each row group (len = rg_rows.len() + 1).
    rg_starts: Vec<u64>,
}

impl FeatureStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Source,
        meta: ArrowReaderMetadata,
        geom_col: usize,
        schema: SchemaRef,
        covering: Option<CoveringCol>,
        encoding: GeomEncoding,
        rg_rows: Vec<u64>,
    ) -> Self {
        let mut rg_starts = Vec::with_capacity(rg_rows.len() + 1);
        let mut acc = 0u64;
        rg_starts.push(0);
        for r in &rg_rows {
            acc += r;
            rg_starts.push(acc);
        }
        Self {
            source,
            meta,
            geom_col,
            schema,
            covering,
            encoding,
            rg_rows,
            rg_starts,
        }
    }

    pub fn total_rows(&self) -> u64 {
        *self.rg_starts.last().unwrap_or(&0)
    }

    /// Cumulative start row of each row group (len = row groups + 1).
    pub fn rg_starts(&self) -> &[u64] {
        &self.rg_starts
    }

    /// Open a reader over a single row group (row-group pruned loading).
    /// `ranges`: group-relative [start, end) row spans to decode (sorted,
    /// non-overlapping); None reads the whole group. `columns`: arrow root
    /// field indices to project, or None for all.
    pub fn open_reader_for_group(
        source: &Source,
        meta: &ArrowReaderMetadata,
        group: usize,
        batch_size: usize,
        ranges: Option<&[(u32, u32)]>,
        columns: Option<&[usize]>,
    ) -> Result<ParquetRecordBatchReader, String> {
        let reader = source.open()?;
        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(reader, meta.clone())
            .with_row_groups(vec![group])
            .with_batch_size(batch_size);
        if let Some(cols) = columns {
            let mask = ProjectionMask::roots(builder.parquet_schema(), cols.iter().copied());
            builder = builder.with_projection(mask);
        }
        if let Some(ranges) = ranges {
            let mut selectors: Vec<RowSelector> = Vec::with_capacity(ranges.len() * 2);
            let mut pos = 0u32;
            for &(start, end) in ranges {
                debug_assert!(start >= pos && end > start, "sorted non-overlapping ranges");
                if start > pos {
                    selectors.push(RowSelector::skip((start - pos) as usize));
                }
                selectors.push(RowSelector::select((end - start) as usize));
                pos = end;
            }
            builder = builder.with_row_selection(RowSelection::from(selectors));
        }
        builder
            .build()
            .map_err(|e| format!("parquet read error: {e}"))
    }

    /// Fetch specific rows (global row indices, must be sorted and unique).
    /// `columns`: arrow schema field indices to read, or None for all.
    /// Returns the concatenated record batches; rows appear in ascending
    /// row-index order matching the input.
    pub fn fetch(
        &self,
        rows_sorted: &[u32],
        columns: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>, String> {
        if rows_sorted.is_empty() {
            return Ok(Vec::new());
        }
        let reader = self.source.open()?;
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(reader, self.meta.clone());

        if let Some(cols) = columns {
            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                cols.iter().copied(),
            );
            builder = builder.with_projection(mask);
        }

        // Choose row groups and build a RowSelection relative to their
        // concatenation.
        let mut groups: Vec<usize> = Vec::new();
        let mut group_offsets: Vec<u64> = Vec::new(); // start of group within chosen concat
        let mut chosen_rows = 0u64;
        let mut selectors: Vec<RowSelector> = Vec::new();
        let mut pos = 0u64;

        for &row in rows_sorted {
            let row = row as u64;
            if row >= self.total_rows() {
                return Err(format!(
                    "row {row} out of range ({} total)",
                    self.total_rows()
                ));
            }
            let g = match self.rg_starts.binary_search(&row) {
                Ok(i) => i,
                Err(i) => i - 1,
            };
            if groups.last() != Some(&g) {
                groups.push(g);
                group_offsets.push(chosen_rows);
                chosen_rows += self.rg_rows[g];
            }
            let pos_in_concat = group_offsets[groups.len() - 1] + (row - self.rg_starts[g]);
            debug_assert!(pos_in_concat >= pos, "rows must be sorted unique");
            if pos_in_concat > pos {
                selectors.push(RowSelector::skip((pos_in_concat - pos) as usize));
            }
            selectors.push(RowSelector::select(1));
            pos = pos_in_concat + 1;
        }

        let reader = builder
            .with_row_groups(groups)
            .with_row_selection(RowSelection::from(selectors))
            .with_batch_size(rows_sorted.len().min(8192))
            .build()
            .map_err(|e| format!("parquet read error: {e}"))?;

        let batches: Result<Vec<_>, _> = reader.collect();
        batches.map_err(|e| format!("parquet decode error: {e}"))
    }

    /// Fetch and decode the geometries of the given rows (sorted, unique),
    /// whatever the column encoding. Returns (row, geometry) pairs in
    /// ascending row order; None for null/undecodable geometries.
    pub fn fetch_geoms(
        &self,
        rows_sorted: &[u32],
    ) -> Result<Vec<(u32, Option<geo_types::Geometry<f64>>)>, String> {
        let batches = self.fetch(rows_sorted, Some(&[self.geom_col]))?;
        let mut out = Vec::with_capacity(rows_sorted.len());
        let mut it = rows_sorted.iter();
        for batch in &batches {
            let col = batch.column(0);
            let Some(geoms) = GeomCol::new(col.as_ref(), self.encoding) else {
                return Err("geometry column does not match its declared encoding".into());
            };
            for i in 0..batch.num_rows() {
                let row = *it.next().ok_or("row/batch count mismatch")?;
                out.push((row, geoms.geometry(i)));
            }
        }
        Ok(out)
    }

    /// Fetch one full row (all columns) for the attribute panel.
    pub fn fetch_row(&self, row: u32) -> Result<RecordBatch, String> {
        let batches = self.fetch(&[row], None)?;
        batches
            .into_iter()
            .next()
            .ok_or_else(|| format!("row {row} not found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validates row-group selection math against a real multi-row-group file.
    #[test]
    fn fetch_rows_across_row_groups() {
        let path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/points_1m_wgs84.parquet"
        ));
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let source = Source::Local(path);
        let meta = ArrowReaderMetadata::load(
            &source.open().unwrap(),
            Default::default(),
        )
        .unwrap();
        let b = ParquetRecordBatchReaderBuilder::new_with_metadata(source.open().unwrap(), meta.clone());
        let schema = b.schema().clone();
        let rg_rows: Vec<u64> = b
            .metadata()
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as u64)
            .collect();
        assert!(rg_rows.len() > 1, "fixture should have several row groups");
        let geom_col = schema.index_of("geometry").unwrap();
        let store =
            FeatureStore::new(source, meta, geom_col, schema, None, GeomEncoding::Wkb, rg_rows);
        assert_eq!(store.total_rows(), 1_000_000);

        // Rows spread across row groups, including group boundaries.
        let rows = [0u32, 1, 122_879, 122_880, 500_000, 999_999];
        let batches = store.fetch(&rows, None).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, rows.len());

        // The name column must match "pt_<row+1>" (row_number is 1-based).
        use arrow::array::StringArray;
        let mut it = rows.iter();
        for batch in &batches {
            let names = batch
                .column(batch.schema().index_of("name").unwrap())
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .clone();
            for i in 0..batch.num_rows() {
                let row = *it.next().unwrap();
                assert_eq!(names.value(i), format!("pt_{}", row + 1), "row {row}");
            }
        }

        // Geometry fetch: every requested row decodes to a point.
        let geoms = store.fetch_geoms(&rows).unwrap();
        assert_eq!(geoms.len(), rows.len());
        for (row, g) in geoms {
            assert!(
                matches!(g.expect("non-null"), geo_types::Geometry::Point(_)),
                "row {row}"
            );
        }
    }
}
