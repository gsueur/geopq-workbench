//! Export partitioning: derived columns (H3 cell, admin attribution via
//! point-in-polygon join) and partition-key assignment (hive fields or
//! adaptive H3 cells), following the OGC GeoParquet distribution guidance:
//! attribute/hive partitioning when a natural key exists, adaptive cell
//! splitting when there is none, features assigned by centroid (no
//! duplication), everything inside each file still spatially ordered.

use std::collections::HashMap;
use std::sync::Arc;

use geo::Contains;
use h3o::{CellIndex, LatLng, Resolution};
use rstar::{RTree, RTreeObject, AABB};

use super::crs::{transform_point, Crs};
use super::store::FeatureStore;

/// Hive directory name for NULL partition values (Hive convention).
pub const NULL_PARTITION: &str = "__HIVE_DEFAULT_PARTITION__";

/// Refuse to explode into pathological file counts.
pub const MAX_PARTITIONS: usize = 4096;

/// How the export splits into files.
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionBy {
    None,
    /// Hive-style directories from these output column names, in order
    /// (`state=MA/county=X/part-0.parquet`). Partition columns are
    /// path-only, not written into the files (Hive convention).
    Fields(Vec<String>),
    /// Adaptive H3 on the feature centroid: cells over `target_rows`
    /// split into their children until balanced (or `max_res` reached).
    /// Non-overlapping, density-responsive (Dunnington 2024).
    AdaptiveH3 { target_rows: usize, max_res: u8 },
}

impl Default for PartitionBy {
    fn default() -> Self {
        Self::None
    }
}

/// Admin attribution: a derived column from a point-in-polygon join
/// against another loaded layer (e.g. states, counties).
pub struct AdminJoinSpec {
    /// Output column name (e.g. "state").
    pub out_name: String,
    /// Boundary layer store + the attribute column to take values from.
    pub store: Arc<FeatureStore>,
    pub value_column: String,
    pub crs: Crs,
}

/// Feature centroids (bbox centers) in a target CRS; None for null geoms.
pub fn centroids_in(
    row_bboxes: &[Option<[f64; 4]>],
    src_crs: &Crs,
    dst_crs: &Crs,
) -> Vec<Option<(f64, f64)>> {
    let same = src_crs.same_as(dst_crs);
    row_bboxes
        .iter()
        .map(|b| {
            let b = b.as_ref()?;
            let (cx, cy) = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5);
            if same {
                return Some((cx, cy));
            }
            transform_point(src_crs, dst_crs, cx, cy)
                .ok()
                .filter(|(x, y)| x.is_finite() && y.is_finite())
        })
        .collect()
}

/// H3 cell per feature at `res`, from lon/lat centroids.
pub fn h3_cells(lonlat: &[Option<(f64, f64)>], res: u8) -> Result<Vec<Option<u64>>, String> {
    let res = Resolution::try_from(res).map_err(|e| format!("H3 resolution: {e}"))?;
    Ok(lonlat
        .iter()
        .map(|c| {
            let (lon, lat) = (*c)?;
            let ll = LatLng::new(lat, lon).ok()?;
            Some(u64::from(ll.to_cell(res)))
        })
        .collect())
}

/// Average H3 hexagon edge scale per resolution, for the UI.
pub fn h3_res_hint(res: u8) -> &'static str {
    match res {
        0 => "~4.4M km² cells",
        1 => "~610k km²",
        2 => "~87k km²",
        3 => "~12.4k km²",
        4 => "~1.8k km²",
        5 => "~253 km²",
        6 => "~36 km²",
        7 => "~5.2 km²",
        8 => "~0.74 km²",
        9 => "~0.11 km²",
        10 => "~0.015 km²",
        11 => "~2100 m²",
        12 => "~310 m²",
        _ => "",
    }
}

struct BoundaryEntry {
    bbox: AABB<[f64; 2]>,
    geom: geo_types::Geometry<f64>,
    value: String,
}

impl RTreeObject for BoundaryEntry {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        self.bbox
    }
}

/// Point-in-polygon join: each feature centroid (in the boundary layer's
/// CRS) gets the boundary polygon's attribute value; None when no polygon
/// contains it. Returned interned, because the values come from a boundary
/// layer of at most 200k polygons however many features are joined — a
/// `String` per exported row would be the largest allocation of the whole
/// export, and every one of them a duplicate.
pub fn admin_join(
    spec: &AdminJoinSpec,
    out_name: &str,
    centroids_boundary_crs: &[Option<(f64, f64)>],
) -> Result<FieldCodes, String> {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    use geo::BoundingRect;

    let total = spec.store.total_rows();
    if total == 0 {
        return Err("boundary layer has no rows".into());
    }
    if total > 200_000 {
        return Err(format!(
            "boundary layer has {total} rows — use an aggregated boundaries layer"
        ));
    }
    let rows: Vec<u32> = (0..total as u32).collect();
    let geoms = spec.store.fetch_geoms(&rows)?;
    let val_idx = spec
        .store
        .schema
        .fields()
        .iter()
        .position(|f| f.name().eq_ignore_ascii_case(&spec.value_column))
        .ok_or_else(|| format!("column '{}' not found in boundary layer", spec.value_column))?;
    let batches = spec.store.fetch(&rows, Some(&[val_idx]))?;
    let opts = FormatOptions::default().with_display_error(true);
    let mut values: Vec<String> = Vec::with_capacity(total as usize);
    for b in &batches {
        let f = ArrayFormatter::try_new(b.column(0).as_ref(), &opts)
            .map_err(|e| format!("boundary values: {e}"))?;
        for i in 0..b.num_rows() {
            values.push(f.value(i).to_string());
        }
    }

    let entries: Vec<BoundaryEntry> = geoms
        .into_iter()
        .filter_map(|(row, g)| {
            let g = g?;
            let r = g.bounding_rect()?;
            Some(BoundaryEntry {
                bbox: AABB::from_corners([r.min().x, r.min().y], [r.max().x, r.max().y]),
                geom: g,
                value: values.get(row as usize)?.clone(),
            })
        })
        .collect();
    let tree = RTree::bulk_load(entries);

    let mut out = Interner::uncapped(centroids_boundary_crs.len());
    for c in centroids_boundary_crs {
        let hit = c.and_then(|(x, y)| {
            let p = geo_types::Point::new(x, y);
            tree.locate_in_envelope_intersecting(AABB::from_point([x, y]))
                .find(|e| e.geom.contains(&p))
        });
        out.push(hit.map(|e| e.value.as_str()))?;
    }
    Ok(out.finish(out_name))
}

/// Per-row partition values held as dictionary codes: `codes[row]` indexes
/// `dict`. A formatted `String` per row per field is what a partitioned
/// export of a large layer spends most of its memory on, and the values are
/// low-cardinality by construction — a key with more than `MAX_PARTITIONS`
/// distinct values is refused either way.
pub struct FieldCodes {
    pub name: String,
    pub dict: Vec<Option<String>>,
    pub codes: Vec<u32>,
}

impl FieldCodes {
    /// The value of one row, for callers writing the column back out.
    pub fn value(&self, row: u32) -> Option<&str> {
        self.codes
            .get(row as usize)
            .and_then(|&c| self.dict.get(c as usize))
            .and_then(Option::as_deref)
    }
}

/// Builds `FieldCodes` one row at a time, so a partition key can be filled
/// during the scan that reads it. Refuses past the partition ceiling: a
/// high-cardinality key would otherwise fill the dictionary long before the
/// partition count itself is checked.
pub struct Interner {
    dict: Vec<Option<String>>,
    index: HashMap<String, u32>,
    null_code: Option<u32>,
    codes: Vec<u32>,
    /// Distinct values allowed before the build is refused.
    cap: usize,
}

impl Default for Interner {
    fn default() -> Self {
        Self {
            dict: Vec::new(),
            index: HashMap::new(),
            null_code: None,
            codes: Vec::new(),
            cap: MAX_PARTITIONS,
        }
    }
}

impl Interner {
    /// For values that are not a partition key. An admin join against a
    /// 200k-polygon boundary layer is a legitimate output column, and only
    /// partitioning has a reason to refuse that many distinct values.
    pub fn uncapped(rows: usize) -> Self {
        Self {
            codes: Vec::with_capacity(rows),
            cap: usize::MAX,
            ..Default::default()
        }
    }

    pub fn push(&mut self, v: Option<&str>) -> Result<(), String> {
        let code = match v {
            None => match self.null_code {
                Some(c) => c,
                None => {
                    let c = self.dict.len() as u32;
                    self.dict.push(None);
                    self.null_code = Some(c);
                    c
                }
            },
            Some(s) => match self.index.get(s) {
                Some(&c) => c,
                None => {
                    let c = self.dict.len() as u32;
                    self.dict.push(Some(s.to_string()));
                    self.index.insert(s.to_string(), c);
                    c
                }
            },
        };
        if self.dict.len() > self.cap {
            return Err(format!(
                "more than {MAX_PARTITIONS} partitions — pick lower-cardinality fields"
            ));
        }
        self.codes.push(code);
        Ok(())
    }

    pub fn finish(self, name: &str) -> FieldCodes {
        FieldCodes { name: name.to_string(), dict: self.dict, codes: self.codes }
    }
}

/// `split_by_field_codes` from raw per-row values. The export interns its
/// partition keys while it scans, so this direct form is only what the
/// tests below name the behaviour with.
#[cfg(test)]
pub fn split_by_fields(
    order: &[u32],
    fields: &[(String, Vec<Option<String>>)],
) -> Result<Vec<(String, Vec<u32>)>, String> {
    let coded: Vec<FieldCodes> = fields
        .iter()
        .map(|(name, values)| {
            let mut it = Interner::default();
            for v in values {
                it.push(v.as_deref())?;
            }
            Ok(it.finish(name))
        })
        .collect::<Result<_, String>>()?;
    split_by_field_codes(order, &coded)
}

/// `split_by_fields` over interned values.
pub fn split_by_field_codes(
    order: &[u32],
    fields: &[FieldCodes],
) -> Result<Vec<(String, Vec<u32>)>, String> {
    // Encode field names and dictionary values once, rather than once per
    // output row. Parquet field names are unrestricted strings; raw '/' or
    // '\\' here would otherwise become path separators when optimize joins
    // this relative directory to the chosen output root.
    let encoded: Vec<(String, Vec<String>)> = fields
        .iter()
        .map(|f| {
            if f.name.is_empty() {
                return Err("cannot partition by an empty field name".to_string());
            }
            let values = f
                .dict
                .iter()
                .map(|v| {
                    v.as_deref()
                        .map(sanitize_hive_value)
                        .unwrap_or_else(|| NULL_PARTITION.to_string())
                })
                .collect();
            Ok((encode_hive_component(&f.name), values))
        })
        .collect::<Result<_, _>>()?;
    let mut parts: HashMap<String, Vec<u32>> = HashMap::new();
    for &r in order {
        let dir = fields
            .iter()
            .zip(&encoded)
            .map(|(f, (name, values))| {
                let v = f
                    .codes
                    .get(r as usize)
                    .and_then(|&c| values.get(c as usize))
                    .map(String::as_str)
                    .unwrap_or(NULL_PARTITION);
                format!("{name}={v}")
            })
            .collect::<Vec<_>>()
            .join("/");
        parts.entry(dir).or_default().push(r);
        if parts.len() > MAX_PARTITIONS {
            return Err(format!(
                "more than {MAX_PARTITIONS} partitions — pick lower-cardinality fields"
            ));
        }
    }
    let mut out: Vec<(String, Vec<u32>)> = parts.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Adaptive H3 split: start at res 0 and recursively split any cell with
/// more than `target_rows` features into its children, down to `max_res`.
/// Null-geometry rows land in a dedicated partition.
pub fn split_adaptive_h3(
    order: &[u32],
    lonlat: &[Option<(f64, f64)>],
    target_rows: usize,
    max_res: u8,
) -> Result<Vec<(String, Vec<u32>)>, String> {
    let (cells, nulls) = adaptive_cells(order, lonlat, 0, target_rows, max_res)?;
    let mut out: Vec<(String, Vec<u32>)> = cells
        .into_iter()
        .map(|(cell, rows)| (format!("h3={cell}"), rows))
        .collect();
    if !nulls.is_empty() {
        out.push((format!("h3={NULL_PARTITION}"), nulls));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// One leaf part of an H3 pyramid: the cell whose file it is (None for the
/// null-geometry part) and the rows it holds, in the caller's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafPart {
    pub cell: Option<CellIndex>,
    pub rows: Vec<u32>,
}

impl LeafPart {
    /// Resolution of the file's directory: the cell's own, or the
    /// reference resolution for the null part (which the layout puts at
    /// the leaf level, `r<R>/__HIVE_DEFAULT_PARTITION__.parquet`).
    pub fn res(&self, reference_res: u8) -> u8 {
        self.cell.map_or(reference_res, |c| u8::from(c.resolution()))
    }

    /// Relative path inside the pyramid root.
    pub fn path(&self, reference_res: u8) -> String {
        match self.cell {
            Some(c) => super::pyramid::part_path(u8::from(c.resolution()), &c.to_string()),
            None => super::pyramid::part_path(reference_res, super::pyramid::NULL_PART),
        }
    }
}

/// Leaf parts of an H3 pyramid: the same adaptive descent, started at the
/// reference resolution instead of res 0 and keyed by cell rather than by
/// a hive `h3=` directory. `max_res == reference_res` means no splitting,
/// i.e. one file per reference cell whatever its row count.
///
/// Parts come back coarse to fine and, within a resolution, in cell order,
/// so the writer's sweeps and the descriptor's cell lists are both stable
/// across runs. The null part sorts last.
pub fn split_pyramid_leaf(
    order: &[u32],
    lonlat: &[Option<(f64, f64)>],
    reference_res: u8,
    target_rows: usize,
    max_res: u8,
) -> Result<Vec<LeafPart>, String> {
    if max_res < reference_res {
        return Err(format!(
            "pyramid: adaptive max resolution r{max_res} is coarser than the reference r{reference_res}"
        ));
    }
    let (cells, nulls) = adaptive_cells(order, lonlat, reference_res, target_rows, max_res)?;
    let mut out: Vec<LeafPart> = cells
        .into_iter()
        .map(|(cell, rows)| LeafPart { cell: Some(cell), rows })
        .collect();
    out.sort_by_key(|p| {
        let c = p.cell.expect("cells only");
        (u8::from(c.resolution()), u64::from(c))
    });
    if !nulls.is_empty() {
        out.push(LeafPart { cell: None, rows: nulls });
    }
    Ok(out)
}

/// The descent both H3 partitionings share: bucket rows by their centroid
/// cell at `start_res`, then split any bucket over `target_rows` into its
/// children until `max_res`. Rows whose centroid is missing come back
/// separately — they have no cell to be placed in at any resolution.
fn adaptive_cells(
    order: &[u32],
    lonlat: &[Option<(f64, f64)>],
    start_res: u8,
    target_rows: usize,
    max_res: u8,
) -> Result<(Vec<(CellIndex, Vec<u32>)>, Vec<u32>), String> {
    let max_res = Resolution::try_from(max_res).map_err(|e| format!("H3 resolution: {e}"))?;
    let start_res = Resolution::try_from(start_res).map_err(|e| format!("H3 resolution: {e}"))?;
    if start_res > max_res {
        return Err("H3 resolution: the start resolution is finer than the maximum".into());
    }
    // Finest cells once, indexed by row; parents are cheap bit ops from there.
    let mut fine_of: Vec<Option<CellIndex>> = vec![None; lonlat.len()];
    let mut nulls: Vec<u32> = Vec::new();
    let mut work: HashMap<CellIndex, Vec<u32>> = HashMap::new(); // keyed at current res
    for &r in order {
        let cell = lonlat[r as usize]
            .and_then(|(lon, lat)| LatLng::new(lat, lon).ok())
            .map(|ll| ll.to_cell(max_res));
        match cell {
            None => nulls.push(r),
            Some(c) => {
                fine_of[r as usize] = Some(c);
                let coarse = c.parent(start_res).unwrap_or(c);
                work.entry(coarse).or_default().push(r);
            }
        }
    }

    let mut done: Vec<(CellIndex, Vec<u32>)> = Vec::new();
    while let Some((&cell, _)) = work.iter().next() {
        let rows = work.remove(&cell).unwrap();
        let res = cell.resolution();
        if rows.len() <= target_rows || res >= max_res {
            done.push((cell, rows));
            continue;
        }
        let child_res = res.succ().unwrap_or(max_res);
        for r in rows {
            let f = fine_of[r as usize].expect("bucketed rows have a fine cell");
            work.entry(f.parent(child_res).unwrap_or(f)).or_default().push(r);
        }
        if work.len() + done.len() > MAX_PARTITIONS {
            return Err(format!(
                "more than {MAX_PARTITIONS} partitions — raise the rows-per-file target"
            ));
        }
    }
    // Buckets are filled by traversing `order` sequentially and re-bucketed
    // in that same relative order on split, so each partition already
    // preserves the global (Hilbert) order — no re-sort needed.
    Ok((done, nulls))
}

/// Resolutions the pyramid density table covers, coarse to fine.
pub const DENSITY_RES: std::ops::RangeInclusive<u8> = 3..=10;

/// Resolution the density table computes every coarser one from. Cells at
/// any res in `DENSITY_RES` are a `parent()` bit op away from one of these.
const DENSITY_BASE_RES: u8 = 10;

/// What one candidate reference resolution would produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DensityRow {
    pub res: u8,
    /// Cells holding at least one feature.
    pub cells: usize,
    /// Rows in the median cell (lower median).
    pub median_rows: usize,
    pub max_rows: usize,
    /// Files after adaptive splitting at `target_rows`, plus the null part
    /// when there is one. Equals `cells` (+1) when splitting is off.
    pub files: usize,
}

/// Density of the layer over H3 cells, res 3..=10, for the pyramid dialog:
/// how many files each candidate reference resolution would produce and
/// how uneven they would be.
///
/// One `LatLng::to_cell` per row at res 10 and nothing but `parent()` bit
/// ops after that — a res-per-row hash join measured several seconds on
/// 2.5M rows, which is too slow for a combo box the user drags through.
/// Sorting the res-10 cells once is what makes the rest linear: an H3
/// index puts the base cell and then one digit per resolution in
/// descending bit order, so cells sharing a parent are contiguous in the
/// sorted array at every resolution at once.
pub fn density_table(
    lonlat: &[Option<(f64, f64)>],
    target_rows: usize,
    max_res: u8,
) -> Result<Vec<DensityRow>, String> {
    let base = Resolution::try_from(DENSITY_BASE_RES).map_err(|e| e.to_string())?;
    let max_res = Resolution::try_from(max_res.min(DENSITY_BASE_RES))
        .map_err(|e| format!("H3 resolution: {e}"))?;
    let target = target_rows.max(1);
    let mut cells: Vec<CellIndex> = Vec::with_capacity(lonlat.len());
    let mut nulls = 0usize;
    for c in lonlat {
        match c.and_then(|(lon, lat)| LatLng::new(lat, lon).ok()) {
            Some(ll) => cells.push(ll.to_cell(base)),
            None => nulls += 1,
        }
    }
    cells.sort_unstable_by_key(|&c| u64::from(c));

    let mut out = Vec::new();
    for r in DENSITY_RES {
        let res = Resolution::try_from(r).map_err(|e| e.to_string())?;
        let mut lens: Vec<usize> = Vec::new();
        let mut files = usize::from(nulls > 0);
        for run in runs_by_parent(&cells, res) {
            lens.push(run.len());
            files += adaptive_files(run, res, target, max_res);
        }
        lens.sort_unstable();
        out.push(DensityRow {
            res: r,
            cells: lens.len(),
            median_rows: lens.get(lens.len() / 2).copied().unwrap_or(0),
            max_rows: lens.last().copied().unwrap_or(0),
            files,
        });
    }
    Ok(out)
}

/// Runs of a resolution-sorted cell array that share a parent at `res`.
fn runs_by_parent(cells: &[CellIndex], res: Resolution) -> impl Iterator<Item = &[CellIndex]> {
    let mut i = 0usize;
    std::iter::from_fn(move || {
        if i >= cells.len() {
            return None;
        }
        let p = cells[i].parent(res);
        let lo = i;
        while i < cells.len() && cells[i].parent(res) == p {
            i += 1;
        }
        Some(&cells[lo..i])
    })
}

/// Files one cell's worth of rows would become under adaptive splitting:
/// itself when it fits or cannot split further, otherwise the sum over its
/// occupied children. Mirrors `adaptive_cells`, on counts alone.
fn adaptive_files(
    cells: &[CellIndex],
    res: Resolution,
    target: usize,
    max_res: Resolution,
) -> usize {
    if cells.len() <= target || res >= max_res {
        return 1;
    }
    let Some(child) = res.succ() else { return 1 };
    runs_by_parent(cells, child)
        .map(|run| adaptive_files(run, child, target, max_res))
        .sum()
}

/// Encode one Hive path component. Keep alphanumerics and a safe subset,
/// percent-encoding everything else per UTF-8 byte (not codepoint).
fn encode_hive_component(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for &b in v.as_bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Hive path value encoding is injective across NULL, empty strings, the
/// literal Hive NULL sentinel and ordinary values. A lossy substitution (or
/// mapping empty to the NULL sentinel) would change equality pushdown and SQL
/// grouping after an export/reload round trip.
fn sanitize_hive_value(v: &str) -> String {
    let mut out = encode_hive_component(v);
    if out == NULL_PARTITION {
        // The loader tests for the raw sentinel before percent-decoding.
        // Escaping one otherwise-safe byte preserves this literal value.
        out.replace_range(..1, "%5F");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_cells_sane() {
        // lon/lat argument order matters: SF must land in a valid res-9
        // cell whose res-2 parent also contains a point 1 km away.
        let sf = Some((-122.388903, 37.769377));
        let nearby = Some((-122.379, 37.771));
        let far = Some((2.35, 48.85)); // Paris
        let cells = h3_cells(&[sf, nearby, far, None], 9).unwrap();
        assert!(cells[3].is_none());
        let a = CellIndex::try_from(cells[0].unwrap()).unwrap();
        let b = CellIndex::try_from(cells[1].unwrap()).unwrap();
        let c = CellIndex::try_from(cells[2].unwrap()).unwrap();
        assert_eq!(a.resolution(), Resolution::Nine);
        assert_ne!(a, c, "SF and Paris in different cells");
        assert_eq!(
            a.parent(Resolution::Two),
            b.parent(Resolution::Two),
            "nearby points share the coarse parent"
        );
        assert_ne!(
            a.parent(Resolution::Two),
            c.parent(Resolution::Two),
            "continents apart"
        );
    }

    #[test]
    fn adaptive_split_balances() {
        // Two dense clusters + a sparse scatter: the dense clusters must
        // split finer than the scatter.
        let mut lonlat: Vec<Option<(f64, f64)>> = Vec::new();
        for i in 0..1000 {
            let d = (i % 100) as f64 * 1e-4;
            lonlat.push(Some((2.35 + d, 48.85 + d))); // Paris cluster
        }
        for i in 0..1000 {
            let d = (i % 100) as f64 * 1e-4;
            lonlat.push(Some((-71.06 + d, 42.36 + d))); // Boston cluster
        }
        for i in 0..50 {
            lonlat.push(Some((-30.0 + i as f64, -20.0 + (i % 40) as f64)));
        }
        lonlat.push(None);
        let order: Vec<u32> = (0..lonlat.len() as u32).collect();
        let parts = split_adaptive_h3(&order, &lonlat, 300, 12).unwrap();
        let total: usize = parts.iter().map(|(_, r)| r.len()).sum();
        assert_eq!(total, lonlat.len());
        assert!(parts.iter().any(|(d, _)| d.ends_with(NULL_PARTITION)));
        // No cell partition above target (splittable clusters are tiny),
        // and each partition preserves the global order without a re-sort.
        for (dir, rows) in &parts {
            if !dir.ends_with(NULL_PARTITION) {
                assert!(rows.len() <= 300, "{dir}: {}", rows.len());
            }
            assert!(rows.windows(2).all(|w| w[0] < w[1]), "{dir} keeps global order");
        }
        assert!(parts.len() > 3, "clusters must split: {}", parts.len());
    }

    /// Two dense clusters and a sparse scatter: the coarse resolutions
    /// pile everything into a handful of cells, the fine ones spread it,
    /// and the file count only ever grows with adaptive splitting on.
    #[test]
    fn density_table_describes_each_resolution() {
        let mut lonlat: Vec<Option<(f64, f64)>> = Vec::new();
        for i in 0..2000 {
            let d = (i % 200) as f64 * 1e-4;
            lonlat.push(Some((2.35 + d, 48.85 + d))); // Paris
        }
        for i in 0..600 {
            let d = (i % 60) as f64 * 1e-3;
            lonlat.push(Some((-71.06 + d, 42.36 + d))); // Boston
        }
        lonlat.push(None);

        let table = density_table(&lonlat, 250, 10).unwrap();
        assert_eq!(table.len(), 8, "res 3..=10");
        assert_eq!(table.first().unwrap().res, 3);
        assert_eq!(table.last().unwrap().res, 10);
        for w in table.windows(2) {
            assert!(w[0].cells <= w[1].cells, "cells grow with resolution: {w:?}");
            assert!(w[0].max_rows >= w[1].max_rows, "cells hold less as they shrink: {w:?}");
            assert!(w[0].median_rows > 0 && w[1].median_rows > 0);
        }
        // Two clusters an ocean apart cannot share a cell at any of these
        // resolutions, and the null row buys exactly one extra file.
        assert!(table[0].cells >= 2);
        assert!(table.iter().all(|r| r.files >= r.cells + 1));

        // Splitting off is one file per occupied cell, plus the null part.
        let flat = density_table(&lonlat, 250, 3).unwrap();
        assert_eq!(flat[0].files, flat[0].cells + 1);
        // ... and the row target is respected once splitting is allowed:
        // every cluster cell over 250 rows must have been broken up.
        assert!(table[0].files > flat[0].files, "{:?} vs {:?}", table[0], flat[0]);
    }

    /// The file count the table promises is the file count the writer
    /// produces — the whole point of showing it before the run.
    #[test]
    fn density_files_match_the_leaf_split() {
        let mut lonlat: Vec<Option<(f64, f64)>> = Vec::new();
        for i in 0..1500 {
            let d = (i % 150) as f64 * 2e-4;
            lonlat.push(Some((-122.39 + d, 37.77 + d)));
        }
        lonlat.push(None);
        let order: Vec<u32> = (0..lonlat.len() as u32).collect();
        let table = density_table(&lonlat, 200, 10).unwrap();
        for row in &table {
            let parts = split_pyramid_leaf(&order, &lonlat, row.res, 200, 10).unwrap();
            assert_eq!(parts.len(), row.files, "r{}: {row:?}", row.res);
            let placed: usize = parts.iter().map(|p| p.rows.len()).sum();
            assert_eq!(placed, lonlat.len(), "every row lands in exactly one part");
        }
    }

    #[test]
    fn pyramid_leaf_parts_are_named_and_ordered() {
        let mut lonlat: Vec<Option<(f64, f64)>> = Vec::new();
        for i in 0..900 {
            let d = (i % 90) as f64 * 3e-4;
            lonlat.push(Some((2.35 + d, 48.85 + d)));
        }
        lonlat.push(None);
        let order: Vec<u32> = (0..lonlat.len() as u32).collect();

        // No splitting: every part sits at the reference resolution.
        let flat = split_pyramid_leaf(&order, &lonlat, 6, 100, 6).unwrap();
        assert!(flat.iter().all(|p| p.res(6) == 6));
        assert!(flat.iter().any(|p| p.rows.len() > 100), "no splitting means big files");

        let parts = split_pyramid_leaf(&order, &lonlat, 6, 100, 9).unwrap();
        assert!(parts.iter().any(|p| p.res(6) > 6), "dense cells split finer");
        // Coarse to fine, cells ordered inside a resolution, null last
        // (the null part carries the reference resolution, not an order).
        let keys: Vec<(u8, u64)> = parts
            .iter()
            .filter_map(|p| p.cell.map(|c| (u8::from(c.resolution()), u64::from(c))))
            .collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "{keys:?}");
        assert_eq!(parts.last().unwrap().cell, None, "the null part sorts last");
        assert_eq!(parts.last().unwrap().path(6), "r6/__HIVE_DEFAULT_PARTITION__.parquet");
        for p in &parts {
            let path = p.path(6);
            assert!(path.starts_with(&format!("r{}/", p.res(6))), "{path}");
            assert!(path.ends_with(".parquet"), "{path}");
            // Rows keep the global (Hilbert) order inside every part.
            assert!(p.rows.windows(2).all(|w| w[0] < w[1]), "{path}");
        }
        // A cell only ever splits when it is over target.
        for p in parts.iter().filter(|p| p.cell.is_some() && p.res(6) < 9) {
            assert!(p.rows.len() <= 100, "{}: {}", p.path(6), p.rows.len());
        }
        assert_eq!(parts.iter().map(|p| p.rows.len()).sum::<usize>(), lonlat.len());
    }

    #[test]
    fn a_reference_finer_than_the_max_is_refused() {
        let ll = vec![Some((2.35, 48.85))];
        assert!(split_pyramid_leaf(&[0], &ll, 9, 100, 7).is_err());
    }

    #[test]
    fn hive_split_and_sanitize() {
        let vals = vec![
            ("state".to_string(),
             vec![Some("MA".into()), Some("MA".into()), None, Some("New York".into())]),
        ];
        let order = [0u32, 1, 2, 3];
        let parts = split_by_fields(&order, &vals).unwrap();
        let dirs: Vec<&str> = parts.iter().map(|(d, _)| d.as_str()).collect();
        assert!(dirs.contains(&"state=MA"));
        assert!(dirs.contains(&"state=New%20York"));
        assert!(dirs.contains(&format!("state={NULL_PARTITION}").as_str()));
        let ma = &parts.iter().find(|(d, _)| d == "state=MA").unwrap().1;
        assert_eq!(ma, &vec![0, 1]);
    }

    #[test]
    fn hive_value_roundtrips_through_decode() {
        use crate::data::store::percent_decode;
        for v in [
            "Zürich",
            "Ł",
            "New York",
            "New_York",
            "a/b=c%d",
            "München 2024",
            "__HIVE_DEFAULT_PARTITION__",
        ] {
            assert_eq!(
                percent_decode(&sanitize_hive_value(v)),
                v,
                "roundtrip of {v:?}"
            );
        }
        // Multibyte codepoints must encode all UTF-8 bytes, not the low
        // byte of the scalar ('Ł' → %C5%81, never %41 which collides with 'A').
        assert_eq!(sanitize_hive_value("Ł"), "%C5%81");
        assert_eq!(sanitize_hive_value("Zürich"), "Z%C3%BCrich");
        // Space encodes reversibly; no collision with a literal underscore.
        assert_eq!(sanitize_hive_value("New York"), "New%20York");
        assert_ne!(sanitize_hive_value("New York"), sanitize_hive_value("New_York"));
    }

    #[test]
    fn hive_value_directory_safety() {
        // Separators and hive metacharacters never survive raw.
        let s = sanitize_hive_value("a/b\\c=d%e?f#g");
        for bad in ['/', '\\', '=', '?', '#', ' '] {
            assert!(!s.contains(bad), "{s} must not contain {bad:?}");
        }
        assert_eq!(sanitize_hive_value(""), "");
        assert_eq!(sanitize_hive_value("~"), "%7E");
    }

    #[test]
    fn hive_reserved_values_stay_distinct() {
        let values = vec![(
            "kind".to_string(),
            vec![
                None,
                Some(String::new()),
                Some(NULL_PARTITION.to_string()),
                Some("ordinary".to_string()),
            ],
        )];
        let parts = split_by_fields(&[0, 1, 2, 3], &values).unwrap();
        assert_eq!(parts.len(), 4, "reserved values must not share a partition");
        assert!(
            parts
                .iter()
                .any(|(d, r)| d == "kind=__HIVE_DEFAULT_PARTITION__" && r == &[0])
        );
        assert!(parts.iter().any(|(d, r)| d == "kind=" && r == &[1]));
        assert!(
            parts
                .iter()
                .any(|(d, r)| d == "kind=%5F_HIVE_DEFAULT_PARTITION__" && r == &[2])
        );
    }

    #[test]
    fn hive_field_names_cannot_create_path_components() {
        let name = "../outside/region\\name=value%";
        let values = vec![(name.to_string(), vec![Some("MA".to_string())])];
        let parts = split_by_fields(&[0], &values).unwrap();
        let rel = &parts[0].0;
        assert_eq!(std::path::Path::new(rel).components().count(), 1, "{rel}");
        assert!(!rel.contains('/'), "{rel}");
        assert!(!rel.contains('\\'), "{rel}");

        let with_file = std::path::PathBuf::from(rel).join("part-0.parquet");
        assert_eq!(
            crate::data::store::hive_segments(&with_file),
            vec![(name.to_string(), Some("MA".to_string()))]
        );
    }
}
