use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use bytes::Bytes;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
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

/// One file of a (possibly multi-file) dataset: its own source and parquet
/// footer, plus the hive partition-key values parsed from its path.
#[derive(Clone)]
pub struct Fragment {
    pub source: Source,
    /// Parsed parquet footer of this file (schema + metadata + page index).
    pub meta: ArrowReaderMetadata,
    /// Hive values for this file, aligned with the store's `part_cols`;
    /// None = key absent from the path or the NULL partition.
    pub part_values: Vec<Option<String>>,
    /// First global row-group index of this fragment.
    pub rg_offset: usize,
    /// First global row of this fragment.
    pub row_offset: u64,
}

/// Lazy access to a GeoParquet dataset's rows: a single file, or a
/// directory of hive-partitioned files presented as one table.
///
/// Row groups and rows are addressed in a single global space spanning all
/// fragments in path order — pruning, decode state, picking and SQL scans
/// are unaware of file boundaries. Hive partition keys become virtual
/// nullable Utf8 columns appended to the schema; their values live only in
/// directory names, so fetches materialize them as constants per fragment.
///
/// Nothing but schemas and per-row-group row counts is kept in memory;
/// attribute and geometry values are re-read from the files on demand.
#[derive(Clone)]
pub struct FeatureStore {
    /// Display source: the file, or the dataset root directory.
    pub source: Source,
    /// Files in global order (exactly one for single-file layers).
    pub fragments: Vec<Fragment>,
    /// Index of the geometry column in the arrow schema.
    pub geom_col: usize,
    /// Base file schema plus one nullable Utf8 field per hive partition key.
    pub schema: SchemaRef,
    /// Covering bbox column, when the files have one (per-feature pruning).
    pub covering: Option<CoveringCol>,
    /// Geometry column encoding (WKB or a GeoArrow native encoding).
    pub encoding: GeomEncoding,
    /// Hive partition-key column names, in schema order (appended last).
    pub part_cols: Vec<String>,
    /// Geometry synthesized from two coordinate columns (x/lon, y/lat base
    /// field indices): the schema gains a virtual Struct{x,y} field right
    /// after the base fields and `geom_col` points at it.
    pub xy_geom: Option<(usize, usize)>,
    /// WKB primary column superseded for display by a GeoArrow sibling
    /// (`geom_col` points at the sibling): hidden from attribute UIs.
    pub hidden_wkb: Option<usize>,
    /// GeoParquet `edges: spherical`: segments are great-circle arcs, so
    /// long edges are densified on the sphere before projection.
    pub spherical_edges: bool,
    /// The `geo` metadata declares polygons only. A polygon is the one
    /// geometry its bounding box can stand in for on screen; a line's
    /// box is a rectangle the line does not follow, and a point is its
    /// own box already.
    pub polygons_only: bool,
    /// Rows per row group, global order across fragments.
    rg_rows: Vec<u64>,
    /// Cumulative start row of each row group (len = rg_rows.len() + 1).
    rg_starts: Vec<u64>,
    /// Fragment index of each global row group.
    rg_frag: Vec<usize>,
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
        xy_geom: Option<(usize, usize)>,
    ) -> Self {
        let frag = Fragment {
            source: source.clone(),
            meta,
            part_values: Vec::new(),
            rg_offset: 0,
            row_offset: 0,
        };
        Self::from_fragments(
            source,
            vec![(frag, rg_rows)],
            Vec::new(),
            geom_col,
            schema,
            covering,
            encoding,
            xy_geom,
        )
    }

    /// Multi-file store: fragments in global order, each with its per-row-
    /// group row counts. All fragments must share the base schema, CRS,
    /// geometry column and encoding (validated by the opener).
    #[allow(clippy::too_many_arguments)]
    pub fn from_fragments(
        source: Source,
        frags: Vec<(Fragment, Vec<u64>)>,
        part_cols: Vec<String>,
        geom_col: usize,
        base_schema: SchemaRef,
        covering: Option<CoveringCol>,
        encoding: GeomEncoding,
        xy_geom: Option<(usize, usize)>,
    ) -> Self {
        let mut fragments = Vec::with_capacity(frags.len());
        let mut rg_rows: Vec<u64> = Vec::new();
        let mut rg_frag: Vec<usize> = Vec::new();
        let mut acc = 0u64;
        for (i, (mut f, rows)) in frags.into_iter().enumerate() {
            f.rg_offset = rg_rows.len();
            f.row_offset = acc;
            for r in rows {
                rg_frag.push(i);
                rg_rows.push(r);
                acc += r;
            }
            fragments.push(f);
        }
        let mut rg_starts = Vec::with_capacity(rg_rows.len() + 1);
        let mut s = 0u64;
        rg_starts.push(0);
        for r in &rg_rows {
            s += r;
            rg_starts.push(s);
        }
        let base_len = base_schema.fields().len();
        let mut geom_col = geom_col;
        let schema = if part_cols.is_empty() && xy_geom.is_none() {
            base_schema
        } else {
            let mut fields: Vec<Field> =
                base_schema.fields().iter().map(|f| f.as_ref().clone()).collect();
            if xy_geom.is_some() {
                let name = if base_schema.index_of("geometry").is_ok() {
                    "xy_geometry"
                } else {
                    "geometry"
                };
                fields.push(Field::new(name, DataType::Struct(xy_fields()), true));
                geom_col = base_len;
            }
            for name in &part_cols {
                fields.push(Field::new(name, DataType::Utf8, true));
            }
            Arc::new(Schema::new_with_metadata(fields, base_schema.metadata().clone()))
        };
        Self {
            source,
            fragments,
            geom_col,
            schema,
            covering,
            encoding,
            part_cols,
            xy_geom,
            hidden_wkb: None,
            spherical_edges: false,
            polygons_only: false,
            rg_rows,
            rg_starts,
            rg_frag,
        }
    }

    /// The STAC collection this store's parts were selected from, if it
    /// came from one. Panning re-reads the collection to find parts the
    /// store does not hold yet.
    pub fn stac_collection(&self) -> Option<&str> {
        match &self.source {
            Source::Stac { url, .. } => Some(url),
            _ => None,
        }
    }

    /// URLs of the parts already open, so a part is never opened twice.
    pub fn part_urls(&self) -> std::collections::HashSet<String> {
        self.fragments.iter().filter_map(|f| f.source.url()).collect()
    }

    /// The same store with more fragments on the end.
    ///
    /// Growing a multi-part dataset while panning depends on this being
    /// index-stable: fragments are numbered in global order and the new
    /// ones go last, so every existing row-group index, row offset and
    /// fragment offset still means what it meant before. That is what lets
    /// a layer keep the geometry and decode state it already has instead of
    /// rebuilding when a part is added.
    pub fn with_fragments_appended(&self, more: Vec<(Fragment, Vec<u64>)>) -> Self {
        let mut out = self.clone();
        let mut acc = out.total_rows();
        for (mut f, rows) in more {
            f.rg_offset = out.rg_rows.len();
            f.row_offset = acc;
            let frag_idx = out.fragments.len();
            for r in rows {
                out.rg_frag.push(frag_idx);
                out.rg_rows.push(r);
                acc += r;
                out.rg_starts.push(acc);
            }
            out.fragments.push(f);
        }
        out
    }

    pub fn total_rows(&self) -> u64 {
        *self.rg_starts.last().unwrap_or(&0)
    }

    /// Cumulative start row of each row group (len = row groups + 1).
    pub fn rg_starts(&self) -> &[u64] {
        &self.rg_starts
    }

    /// Fields that exist in the files (schema fields before the virtual
    /// x/y geometry and partition columns).
    pub fn base_fields(&self) -> usize {
        self.schema.fields().len()
            - self.part_cols.len()
            - self.xy_geom.map_or(0, |_| 1)
    }

    /// Columns a style can be built from, each flagged numeric (graduated)
    /// or not (categorical). Geometry and the WKB column a GeoArrow
    /// sibling supersedes are excluded, as are the virtual x/y and
    /// partition fields.
    pub fn style_columns(&self) -> Vec<(String, bool)> {
        use arrow::datatypes::DataType as DT;
        self.schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                *i != self.geom_col && Some(*i) != self.hidden_wkb && *i < self.base_fields()
            })
            .filter_map(|(_, f)| match f.data_type() {
                DT::Int8 | DT::Int16 | DT::Int32 | DT::Int64 | DT::UInt8 | DT::UInt16
                | DT::UInt32 | DT::UInt64 | DT::Float16 | DT::Float32 | DT::Float64 => {
                    Some((f.name().clone(), true))
                }
                DT::Utf8 | DT::LargeUtf8 | DT::Utf8View | DT::Boolean | DT::Dictionary(_, _) => {
                    Some((f.name().clone(), false))
                }
                _ => None,
            })
            .collect()
    }

    /// Schema index of the first virtual partition column.
    pub fn first_part_index(&self) -> usize {
        self.base_fields() + self.xy_geom.map_or(0, |_| 1)
    }

    pub fn is_partitioned(&self) -> bool {
        self.fragments.len() > 1
    }

    /// The fragment holding a global row group.
    pub fn frag_of_group(&self, group: usize) -> &Fragment {
        &self.fragments[self.rg_frag[group]]
    }

    /// Uncompressed bytes of the geometry column in one row group.
    ///
    /// The decode cost of a group, which its row count does not measure:
    /// 27k land-cover polygons weigh what a million points do. Read from
    /// the footer, so it costs nothing.
    pub fn rg_geom_bytes(&self, group: u32) -> u64 {
        let g = group as usize;
        let Some(&fi) = self.rg_frag.get(g) else {
            return 0;
        };
        let frag = &self.fragments[fi];
        let local = g - frag.rg_offset;
        let Some(rg) = frag.meta.metadata().row_groups().get(local) else {
            return 0;
        };
        // Roots whose leaves hold coordinates: the geometry column, or
        // the two coordinate columns of a synthesized point layer.
        let roots: Vec<&str> = match self.xy_geom {
            Some((xi, yi)) => vec![
                self.schema.field(xi).name().as_str(),
                self.schema.field(yi).name().as_str(),
            ],
            None => vec![self.schema.field(self.geom_col).name().as_str()],
        };
        rg.columns()
            .iter()
            .filter(|c| {
                c.column_descr()
                    .path()
                    .parts()
                    .first()
                    .is_some_and(|p| roots.contains(&p.as_str()))
            })
            .map(|c| c.uncompressed_size().max(0) as u64)
            .sum()
    }

    /// Rows per read batch for a group, sized so one batch carries about
    /// `target_bytes` of geometry instead of a fixed row count.
    ///
    /// Row size spans orders of magnitude between datasets: a point file
    /// runs 20 bytes a row, a land-cover polygon averages 3 kB and its
    /// largest features reach 200k vertices, over 3 MB in one row. A
    /// fixed row count is then a memory lottery, and it is the batches
    /// in flight across every worker that decide the peak. Never returns
    /// zero: one row at a time is the floor, however big that row is.
    pub fn batch_rows(&self, group: usize, target_bytes: u64, max_rows: usize) -> usize {
        let rows = self.rg_rows.get(group).copied().unwrap_or(0);
        if rows == 0 {
            return max_rows;
        }
        let per_row = (self.rg_geom_bytes(group as u32) / rows).max(1);
        (target_bytes / per_row).clamp(1, max_rows as u64) as usize
    }

    /// Partition column `k`'s value for rows of a global row group.
    pub fn part_value(&self, group: usize, k: usize) -> Option<&str> {
        self.frag_of_group(group).part_values.get(k)?.as_deref()
    }

    /// Open a reader over one global row group (row-group pruned loading).
    /// `ranges`: group-relative [start, end) row spans to decode (sorted,
    /// non-overlapping); None reads the whole group. `columns`: arrow
    /// schema indices to project — base fields plus, on x/y stores, the
    /// virtual geometry (assembled per batch); None = all of those.
    /// Virtual partition columns are not readable here.
    pub fn reader_for_group(
        &self,
        group: usize,
        batch_size: usize,
        ranges: Option<&[(u32, u32)]>,
        columns: Option<&[usize]>,
    ) -> Result<GroupReader, String> {
        let base = self.base_fields();
        debug_assert!(
            columns.is_none_or(|c| c.iter().all(|&i| i < self.first_part_index())),
            "virtual partition columns cannot be read from files"
        );
        let requested: Vec<usize> = match columns {
            Some(c) => {
                let mut c = c.to_vec();
                c.sort_unstable();
                c.dedup();
                c
            }
            None => (0..self.first_part_index()).collect(),
        };
        // File columns to read: base fields, plus x/y when the virtual
        // geometry is requested.
        let geom_requested = self.xy_geom.is_some() && requested.contains(&base);
        let mut reads: Vec<usize> = requested.iter().copied().filter(|&i| i < base).collect();
        if geom_requested {
            let (x, y) = self.xy_geom.unwrap();
            reads.extend([x, y]);
            reads.sort_unstable();
            reads.dedup();
        }

        let frag = self.frag_of_group(group);
        let local = group - frag.rg_offset;
        // Remote file without a page index: the reader would stream every
        // projected column chunk sequentially (one round trip each — SQL
        // scans over wide remote files never finish that way). Prefetch
        // the group's projected chunks in a few coalesced range requests
        // instead. With a page index, per-page fetches are already exact.
        let inner = if frag.source.is_remote()
            && frag.meta.metadata().offset_index().is_none()
        {
            match prefetch_columns(frag, &[local], &reads, GROUP_PREFETCH_MAX_BYTES)? {
                Some(cache) => {
                    open_file_group_reader(cache, &frag.meta, local, batch_size, ranges, Some(&reads))?
                }
                None => open_file_group_reader(
                    frag.source.open()?,
                    &frag.meta,
                    local,
                    batch_size,
                    ranges,
                    Some(&reads),
                )?,
            }
        } else {
            open_file_group_reader(frag.source.open()?, &frag.meta, local, batch_size, ranges, Some(&reads))?
        };

        if !geom_requested {
            return Ok(GroupReader::Plain(inner));
        }
        // Map each requested output column to the read batch / assembly.
        let (x, y) = self.xy_geom.unwrap();
        let pos = |i: usize| reads.binary_search(&i).expect("read col present");
        let cols: Vec<XyColSrc> = requested
            .iter()
            .map(|&i| {
                if i == base {
                    XyColSrc::Geom { x: pos(x), y: pos(y) }
                } else {
                    XyColSrc::Read(pos(i))
                }
            })
            .collect();
        let schema = Arc::new(
            self.schema
                .project(&requested)
                .map_err(|e| format!("projection: {e}"))?,
        );
        Ok(GroupReader::Xy { inner, schema, cols })
    }

    /// Fetch specific rows (global row indices, must be sorted and unique).
    /// `columns`: arrow schema field indices to read (virtual partition
    /// columns allowed), or None for all. Returns record batches whose
    /// schema is the projected fields in schema order; rows appear in
    /// ascending row-index order matching the input.
    pub fn fetch(
        &self,
        rows_sorted: &[u32],
        columns: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>, String> {
        if rows_sorted.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(&last) = rows_sorted.last() {
            if last as u64 >= self.total_rows() {
                return Err(format!(
                    "row {last} out of range ({} total)",
                    self.total_rows()
                ));
            }
        }
        let base = self.base_fields();
        let first_part = self.first_part_index();
        let cols: Vec<usize> = match columns {
            Some(c) => {
                let mut c = c.to_vec();
                c.sort_unstable();
                c.dedup();
                c
            }
            None => (0..self.schema.fields().len()).collect(),
        };
        // File columns to read: base fields, plus x/y when the virtual
        // geometry is requested.
        let geom_requested =
            self.xy_geom.is_some() && cols.iter().any(|&c| c == base);
        let mut real: Vec<usize> = cols.iter().copied().filter(|&i| i < base).collect();
        if geom_requested {
            let (x, y) = self.xy_geom.unwrap();
            real.extend([x, y]);
            real.sort_unstable();
            real.dedup();
        }
        let has_virtual =
            geom_requested || cols.iter().any(|&c| c >= first_part) || real.len() != cols.len();
        let out_schema = Arc::new(
            self.schema
                .project(&cols)
                .map_err(|e| format!("projection: {e}"))?,
        );

        let mut out: Vec<RecordBatch> = Vec::new();
        for (fi, frag) in self.fragments.iter().enumerate() {
            let end = self
                .fragments
                .get(fi + 1)
                .map(|f| f.row_offset)
                .unwrap_or_else(|| self.total_rows());
            let lo = rows_sorted.partition_point(|&r| (r as u64) < frag.row_offset);
            let hi = rows_sorted.partition_point(|&r| (r as u64) < end);
            if lo == hi {
                continue;
            }
            let local: Vec<u32> = rows_sorted[lo..hi]
                .iter()
                .map(|&r| (r as u64 - frag.row_offset) as u32)
                .collect();
            let batches = self.fetch_from_fragment(fi, &local, &real)?;
            for b in batches {
                if !has_virtual {
                    out.push(b);
                    continue;
                }
                let pos = |i: usize| real.binary_search(&i).expect("read col present");
                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(cols.len());
                for &c in &cols {
                    if c < base {
                        arrays.push(Arc::clone(b.column(pos(c))));
                    } else if geom_requested && c == base {
                        let (x, y) = self.xy_geom.unwrap();
                        arrays.push(xy_struct(b.column(pos(x)), b.column(pos(y)))?);
                    } else {
                        let val =
                            frag.part_values.get(c - first_part).and_then(|o| o.as_deref());
                        arrays.push(Arc::new(StringArray::from(vec![val; b.num_rows()])));
                    }
                }
                let opts = RecordBatchOptions::new().with_row_count(Some(b.num_rows()));
                out.push(
                    RecordBatch::try_new_with_options(Arc::clone(&out_schema), arrays, &opts)
                        .map_err(|e| format!("batch assembly: {e}"))?,
                );
            }
        }
        Ok(out)
    }

    /// Fetch rows of one fragment (fragment-local sorted row indices),
    /// reading only the given base columns. `real` empty = no file read;
    /// batches carry a row count only.
    fn fetch_from_fragment(
        &self,
        fi: usize,
        local_rows: &[u32],
        real: &[usize],
    ) -> Result<Vec<RecordBatch>, String> {
        let frag = &self.fragments[fi];
        if real.is_empty() {
            // Virtual/part-only fetch: no file IO, just the row count.
            let opts = RecordBatchOptions::new().with_row_count(Some(local_rows.len()));
            let empty = Arc::new(Schema::empty());
            return RecordBatch::try_new_with_options(empty, vec![], &opts)
                .map(|b| vec![b])
                .map_err(|e| format!("batch assembly: {e}"));
        }

        // Local row-group boundaries of this fragment.
        let g_start = |g: usize| self.rg_starts[frag.rg_offset + g] - frag.row_offset;
        let g_rows = |g: usize| self.rg_rows[frag.rg_offset + g];

        // Choose row groups and build a RowSelection relative to their
        // concatenation.
        let mut groups: Vec<usize> = Vec::new();
        let mut group_offsets: Vec<u64> = Vec::new();
        let mut chosen_rows = 0u64;
        let mut selectors: Vec<RowSelector> = Vec::new();
        let mut pos = 0u64;
        for &row in local_rows {
            let row = row as u64;
            let global = row + frag.row_offset;
            let g = match self.rg_starts.binary_search(&global) {
                Ok(i) => i,
                Err(i) => i - 1,
            } - frag.rg_offset;
            if groups.last() != Some(&g) {
                groups.push(g);
                group_offsets.push(chosen_rows);
                chosen_rows += g_rows(g);
            }
            let pos_in_concat = group_offsets[groups.len() - 1] + (row - g_start(g));
            debug_assert!(pos_in_concat >= pos, "rows must be sorted unique");
            if pos_in_concat > pos {
                selectors.push(RowSelector::skip((pos_in_concat - pos) as usize));
            }
            selectors.push(RowSelector::select(1));
            pos = pos_in_concat + 1;
        }

        let selection = RowSelection::from(selectors);
        let batch = local_rows.len().min(8192);

        // Remote sources without a page index: the reader would issue one
        // sequential read per projected column chunk — hundreds of round
        // trips on wide schemas. Prefetch the needed chunk byte ranges in
        // a few coalesced range requests and serve the reader from memory
        // instead. (With a page index, per-page fetches are already exact
        // and cheaper for sparse rows.)
        if frag.source.is_remote() && frag.meta.metadata().offset_index().is_none() {
            if let Some(cache) = prefetch_columns(frag, &groups, real, PREFETCH_MAX_BYTES)? {
                return Self::read_selected(cache, frag, &groups, selection, batch, real);
            }
        }
        let reader = frag.source.open()?;
        Self::read_selected(reader, frag, &groups, selection, batch, real)
    }

    /// Decode the selected rows from any byte source.
    fn read_selected<R: parquet::file::reader::ChunkReader + 'static>(
        reader: R,
        frag: &Fragment,
        groups: &[usize],
        selection: RowSelection,
        batch_size: usize,
        real: &[usize],
    ) -> Result<Vec<RecordBatch>, String> {
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(reader, frag.meta.clone());
        let mask = ProjectionMask::roots(builder.parquet_schema(), real.iter().copied());
        builder = builder.with_projection(mask);
        let reader = builder
            .with_row_groups(groups.to_vec())
            .with_row_selection(selection)
            .with_batch_size(batch_size)
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

    /// Min/max of a numeric base column across all fragments' row-group
    /// statistics (for auto styling breaks). None when any group lacks
    /// stats or the column isn't a flat numeric leaf.
    pub fn column_range(&self, col: usize) -> Option<(f64, f64)> {
        use parquet::file::statistics::Statistics;
        if col >= self.base_fields() {
            return None;
        }
        let name = self.schema.field(col).name();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for frag in &self.fragments {
            let pmd = frag.meta.metadata();
            let leaf = pmd
                .file_metadata()
                .schema_descr()
                .columns()
                .iter()
                .position(|c| c.path().parts().first().map(String::as_str) == Some(name))?;
            for rg in pmd.row_groups() {
                let st = rg.columns().get(leaf)?.statistics()?;
                let (mn, mx): (f64, f64) = match st {
                    Statistics::Double(s) => (*s.min_opt()?, *s.max_opt()?),
                    Statistics::Float(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
                    Statistics::Int32(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
                    Statistics::Int64(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
                    _ => return None,
                };
                lo = lo.min(mn);
                hi = hi.max(mx);
            }
        }
        (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
    }

    /// Fetch one full row (all columns). The feature panel path caps very
    /// wide schemas instead (see the app's fetch_pick_attrs).
    #[allow(dead_code)] // exercised by tests; kept as the simple-row API
    pub fn fetch_row(&self, row: u32) -> Result<RecordBatch, String> {
        let batches = self.fetch(&[row], None)?;
        batches
            .into_iter()
            .next()
            .ok_or_else(|| format!("row {row} not found"))
    }
}

/// Prefetch budget for coalesced remote reads; larger projections fall
/// back to the streaming per-chunk path.
const PREFETCH_MAX_BYTES: u64 = 128 * 1024 * 1024;
/// Tighter per-row-group budget for scan prefetches: several scan
/// partitions decode concurrently, so the caps multiply.
const GROUP_PREFETCH_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Ranges closer than this merge into one request (a gap this size costs
/// less to download than a fresh round trip).
const PREFETCH_GAP: u64 = 1024 * 1024;

/// The byte ranges of the projected columns' chunks in the chosen row
/// groups, fetched with coalesced range requests. None when the plan
/// exceeds `max_bytes`.
fn prefetch_columns(
    frag: &Fragment,
    groups: &[usize],
    roots: &[usize],
    max_bytes: u64,
) -> Result<Option<Prefetched>, String> {
    let pmd = frag.meta.metadata();
    // Leaf column -> arrow root index (leaves of one root are adjacent).
    let descr = pmd.file_metadata().schema_descr();
    let mut root_of_leaf: Vec<usize> = Vec::with_capacity(descr.num_columns());
    let mut last_root: Option<&str> = None;
    let mut root_idx = 0usize;
    for col in descr.columns() {
        let root = col.path().parts()[0].as_str();
        if last_root != Some(root) {
            if last_root.is_some() {
                root_idx += 1;
            }
            last_root = Some(root);
        }
        root_of_leaf.push(root_idx);
    }
    let want: std::collections::HashSet<usize> = roots.iter().copied().collect();

    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for &g in groups {
        for (leaf, col) in pmd.row_group(g).columns().iter().enumerate() {
            if want.contains(&root_of_leaf[leaf]) {
                let (offset, len) = col.byte_range();
                ranges.push((offset, len));
            }
        }
    }
    ranges.sort_unstable();
    let mut segments: Vec<(u64, u64)> = Vec::new(); // (offset, len)
    for (o, l) in ranges {
        match segments.last_mut() {
            Some((so, sl)) if o <= *so + *sl + PREFETCH_GAP => {
                *sl = (*sl).max(o + l - *so);
            }
            _ => segments.push((o, l)),
        }
    }
    let total: u64 = segments.iter().map(|(_, l)| l).sum();
    if total > max_bytes {
        log::debug!("prefetch plan {total} B exceeds budget; streaming instead");
        return Ok(None);
    }

    let reader = frag.source.open()?;
    let len = parquet::file::reader::Length::len(&reader);
    let mut fetched: Vec<(u64, Bytes)> = Vec::with_capacity(segments.len());
    // Cap individual requests so the agent's body timeout can be tight
    // without failing legitimate large transfers on slow links.
    const REQUEST_CAP: u64 = 32 * 1024 * 1024;
    for (o, l) in segments {
        if l <= REQUEST_CAP {
            let bytes = parquet::file::reader::ChunkReader::get_bytes(&reader, o, l as usize)
                .map_err(|e| format!("prefetch failed: {e}"))?;
            fetched.push((o, bytes));
            continue;
        }
        let mut buf = Vec::with_capacity(l as usize);
        let mut pos = 0u64;
        while pos < l {
            let take = REQUEST_CAP.min(l - pos);
            let part =
                parquet::file::reader::ChunkReader::get_bytes(&reader, o + pos, take as usize)
                    .map_err(|e| format!("prefetch failed: {e}"))?;
            buf.extend_from_slice(&part);
            pos += take;
        }
        fetched.push((o, Bytes::from(buf)));
    }
    log::debug!(
        "prefetched {} segment(s), {} B for {} column roots",
        fetched.len(),
        total,
        roots.len()
    );
    Ok(Some(Prefetched {
        len,
        segments: fetched,
    }))
}

/// In-memory byte ranges of a remote file, served as a `ChunkReader`.
/// Reads outside the prefetched segments error (the fetch plan covers
/// every chunk the projected read can touch).
struct Prefetched {
    len: u64,
    /// (file offset, bytes), sorted by offset, non-overlapping.
    segments: Vec<(u64, Bytes)>,
}

impl Prefetched {
    fn slice_from(&self, start: u64) -> Option<(usize, Bytes)> {
        let i = self.segments.partition_point(|(o, _)| *o <= start);
        let (o, b) = self.segments.get(i.checked_sub(1)?)?;
        let off = (start - o) as usize;
        (off <= b.len()).then(|| (i - 1, b.slice(off..)))
    }
}

impl parquet::file::reader::Length for Prefetched {
    fn len(&self) -> u64 {
        self.len
    }
}

impl parquet::file::reader::ChunkReader for Prefetched {
    type T = std::io::Cursor<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        let (_, b) = self.slice_from(start).ok_or_else(|| {
            parquet::errors::ParquetError::General(format!(
                "read at {start} outside prefetched ranges"
            ))
        })?;
        Ok(std::io::Cursor::new(b))
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let (_, b) = self.slice_from(start).ok_or_else(|| {
            parquet::errors::ParquetError::General(format!(
                "read at {start} outside prefetched ranges"
            ))
        })?;
        if b.len() < length {
            return Err(parquet::errors::ParquetError::EOF(format!(
                "prefetched segment too short at {start}: {} < {length}",
                b.len()
            )));
        }
        Ok(b.slice(..length))
    }
}

/// Child fields of the virtual x/y geometry struct (GeoArrow "point"
/// separated layout — the existing point accessor consumes it directly).
fn xy_fields() -> arrow::datatypes::Fields {
    vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]
    .into()
}

/// Assemble the virtual geometry struct from two coordinate arrays
/// (cast to f64; a row is null when either coordinate is null).
fn xy_struct(x: &ArrayRef, y: &ArrayRef) -> Result<ArrayRef, String> {
    use arrow::array::StructArray;
    use arrow::buffer::NullBuffer;
    let cx = arrow::compute::cast(x, &DataType::Float64)
        .map_err(|e| format!("x column cast: {e}"))?;
    let cy = arrow::compute::cast(y, &DataType::Float64)
        .map_err(|e| format!("y column cast: {e}"))?;
    let nulls = NullBuffer::union(cx.nulls(), cy.nulls());
    Ok(Arc::new(StructArray::new(xy_fields(), vec![cx, cy], nulls)))
}

/// Where one output column of a [`GroupReader::Xy`] batch comes from.
pub enum XyColSrc {
    /// Position in the decoded file batch.
    Read(usize),
    /// The virtual geometry, assembled from these batch positions.
    Geom { x: usize, y: usize },
}

/// Batches of one row group; on x/y stores the virtual geometry struct is
/// assembled into each batch.
pub enum GroupReader {
    Plain(ParquetRecordBatchReader),
    Xy {
        inner: ParquetRecordBatchReader,
        schema: SchemaRef,
        cols: Vec<XyColSrc>,
    },
}

impl Iterator for GroupReader {
    type Item = Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            GroupReader::Plain(r) => r.next(),
            GroupReader::Xy { inner, schema, cols } => {
                let batch = match inner.next()? {
                    Ok(b) => b,
                    Err(e) => return Some(Err(e)),
                };
                let arrays: Result<Vec<ArrayRef>, String> = cols
                    .iter()
                    .map(|c| match c {
                        XyColSrc::Read(i) => Ok(Arc::clone(batch.column(*i))),
                        XyColSrc::Geom { x, y } => {
                            xy_struct(batch.column(*x), batch.column(*y))
                        }
                    })
                    .collect();
                let arrays = match arrays {
                    Ok(a) => a,
                    Err(e) => {
                        return Some(Err(arrow::error::ArrowError::ComputeError(e)))
                    }
                };
                let opts = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                Some(
                    RecordBatch::try_new_with_options(Arc::clone(schema), arrays, &opts)
                )
            }
        }
    }
}

/// Open a reader over one row group of one file (the single-file primitive
/// behind [`FeatureStore::reader_for_group`]).
fn open_file_group_reader<R: parquet::file::reader::ChunkReader + 'static>(
    reader: R,
    meta: &ArrowReaderMetadata,
    group: usize,
    batch_size: usize,
    ranges: Option<&[(u32, u32)]>,
    columns: Option<&[usize]>,
) -> Result<ParquetRecordBatchReader, String> {
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

/// Hive path parsing: `key=value` directory segments of a relative path.
/// Keys and values are percent-decoded; reserved markers preserve the
/// distinction between Hive NULL and a present-but-empty string.
pub fn hive_segments(rel: &std::path::Path) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let n = rel.components().count();
    for (i, c) in rel.components().enumerate() {
        if i + 1 == n {
            break; // the file name itself
        }
        let seg = c.as_os_str().to_string_lossy();
        let Some((k, v)) = seg.split_once('=') else {
            continue;
        };
        if k.is_empty() {
            continue;
        }
        let key = percent_decode(k);
        let value = if v == super::partition::NULL_PARTITION {
            None
        } else {
            Some(percent_decode(v))
        };
        out.push((key, value));
    }
    out
}

pub(crate) fn percent_decode(v: &str) -> String {
    fn hex(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }
    let b = v.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        // Byte-wise only: slicing the &str here would panic when '%' is
        // followed by a multibyte character (external directory names).
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(hi), Some(lo)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hive_segment_parsing() {
        let segs = hive_segments(std::path::Path::new(
            "region%2Fname=New%20York/county=__HIVE_DEFAULT_PARTITION__/empty=/part-0.parquet",
        ));
        assert_eq!(
            segs,
            vec![
                ("region/name".to_string(), Some("New York".to_string())),
                ("county".to_string(), None),
                ("empty".to_string(), Some(String::new())),
            ]
        );
        // A bare file has no segments.
        assert!(hive_segments(std::path::Path::new("part-0.parquet")).is_empty());
    }

    #[test]
    fn percent_decode_multibyte_after_percent() {
        // '%' followed by a multibyte char must not panic (byte-offset
        // slicing off a char boundary); the invalid escape passes through.
        assert_eq!(percent_decode("%a€"), "%a€");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%€ab"), "%€ab");
    }

    #[test]
    fn percent_decode_utf8_roundtrip() {
        assert_eq!(percent_decode("Z%C3%BCrich"), "Zürich");
        assert_eq!(percent_decode("%C5%81"), "Ł");
        assert_eq!(percent_decode("New%20York"), "New York");
        assert_eq!(percent_decode("100%25"), "100%");
        // A lone invalid UTF-8 byte degrades to U+FFFD, not a panic.
        assert_eq!(percent_decode("%FC"), "\u{FFFD}");
    }

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
            FeatureStore::new(source, meta, geom_col, schema, None, GeomEncoding::Wkb, rg_rows, None);
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
