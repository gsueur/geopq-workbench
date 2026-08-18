//! Grid summary: aggregate a column into cells — square grid in
//! data-CRS units, H3, or A5 — with optional kernel smoothing, and
//! materialize the result as a GeoParquet layer (which then inherits
//! styling, classification, export and publish like any other layer).
//! Numeric columns take the numeric statistics; a text column takes
//! majority/minority, and the cell value is the winning category.
//!
//! Cell assignment is area-apportioned: features contained in one cell
//! (the vast majority — a pure columnar scan when the file carries a
//! covering bbox column) contribute wholly to it; features spanning
//! several cells are fetched and their value split by covered area —
//! exact rectangle clipping for square grids, in-polygon sampling for
//! H3/A5 — so a giant parcel never spikes the one cell holding its
//! centroid.

use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryBuilder, Float64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;

use super::crs::{transform_point, Crs};
use super::store::FeatureStore;

/// Cell system for the aggregation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellSystem {
    /// Square-ish cells of `sx` × `sy` data-CRS units. Equal for
    /// projected CRS (meters); for a layer in degrees the dialog sizes
    /// them from meters at the data's mid-latitude, so `sx` runs
    /// 1/cos(lat) wider than `sy` and the cell is locally square on the
    /// ground.
    Square { sx: f64, sy: f64 },
    /// H3 hexagons (geographic; centroids are inverse-projected).
    H3 { res: u8 },
    /// A5 pentagons (geographic, equal-area).
    A5 { res: i32 },
}

impl CellSystem {
    /// Isotropic square cells, for projected (metric) layers.
    pub fn square(size: f64) -> Self {
        CellSystem::Square { sx: size, sy: size }
    }
}

/// A square-on-the-ground cell for a layer in degrees: `meters` at
/// latitude `lat0`, where a degree of longitude is cos(lat0) shorter
/// than a degree of latitude. Fine at city and state scale; over a
/// continental span the width drifts with latitude, which is what H3
/// and A5 are for.
pub fn square_cell_degrees(meters: f64, lat0: f64) -> (f64, f64) {
    /// One degree of arc on the sphere, meters.
    const M_PER_DEG: f64 = 111_195.0;
    let sy = meters / M_PER_DEG;
    // Clamped so polar data gets very wide cells rather than infinite.
    let sx = meters / (M_PER_DEG * lat0.to_radians().cos().max(0.01));
    (sx, sy)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridStat {
    Mean,
    Median,
    Sum,
    Count,
    /// Apportioned sum divided by the cell area (per m² for H3/A5 and
    /// metric CRS; per CRS-unit² otherwise). The right statistic for
    /// extensive values (totals): a giant parcel's mean equals its full
    /// value in every cell it covers (the weights cancel), while its
    /// density spreads honestly.
    Density,
    /// The value with the largest weight in the cell — for a text
    /// column (species, land use, owner): each cell gets the category
    /// that dominates it. Point features count 1 apiece; polygons count
    /// their covered-area share, so a big parcel's category carries the
    /// cells it actually covers. The output column is text.
    Majority,
    /// The rarest value present in the cell — where the odd one out is
    /// the question (the one oak in a pine stand).
    Minority,
}

impl GridStat {
    pub fn label(&self) -> &'static str {
        match self {
            GridStat::Mean => "mean",
            GridStat::Median => "median",
            GridStat::Sum => "sum",
            GridStat::Count => "count",
            GridStat::Density => "density",
            GridStat::Majority => "majority",
            GridStat::Minority => "minority",
        }
    }

    /// Whether this statistic reads the column as text and produces a
    /// text cell value (which no smoothing, contour or focal operation
    /// can apply to).
    pub fn text(&self) -> bool {
        matches!(self, GridStat::Majority | GridStat::Minority)
    }
}

/// 3×3 kernel for square grids; H3/A5 always smooth with the ring mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Box,
    Gaussian,
}

/// How contour levels are placed across the value range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContourSpacing {
    /// Equal steps between the value min and max. Reads naturally, but
    /// on a heavy-tailed surface (parcel values, population) nearly
    /// every cell falls under the first level and the rest of the lines
    /// crowd around a handful of outliers.
    Equal,
    /// Equal counts of cells between levels: level k sits at the
    /// k/(levels+1) quantile of the present cells. Each band holds the
    /// same share of the map, whatever the distribution's shape.
    Quantile,
}

/// Focal operation applied to the cell surface after smoothing. The
/// grid is an image by then, so these are the image operators that earn
/// their place on tabular data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PostOp {
    /// Keep the aggregated values.
    None,
    /// Standard deviation of each cell's neighborhood: local
    /// heterogeneity rather than local level. High where the
    /// neighborhood mixes cheap and expensive, near zero on a uniform
    /// stretch whatever its level.
    FocalStd,
    /// Erosion then dilation: drops isolated highs (single hot cells)
    /// and keeps the rest of the surface where it was.
    Open,
    /// Dilation then erosion: fills isolated lows (single cold cells,
    /// small holes) without inflating the surrounding level.
    Close,
    /// Horn hillshade of the value surface, 0..255. Square grids only —
    /// it needs the regular lattice for the gradient. The vertical
    /// exaggeration is fitted to the data (the median gradient lights
    /// at 45°), because values are dollars or people, not metres, and
    /// no fixed z-factor could suit them.
    Hillshade { azimuth_deg: f64, altitude_deg: f64 },
}

impl PostOp {
    /// Suffix for the output column: operations that change what the
    /// number means say so, the ones that keep the quantity stay silent.
    fn suffix(&self) -> &'static str {
        match self {
            PostOp::None | PostOp::Open | PostOp::Close => "",
            PostOp::FocalStd => "_std",
            PostOp::Hillshade { .. } => "_shade",
        }
    }
}

/// What the grid materializes as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridOutput {
    /// One polygon per cell.
    Cells,
    /// Isolines from the (smoothed) cell values: `levels` levels placed
    /// per `spacing`. Square grids only (marching squares needs the
    /// regular lattice).
    Contours {
        levels: usize,
        spacing: ContourSpacing,
    },
}

pub struct GridSpec {
    /// Store-schema index of the numeric value column.
    pub value_col: usize,
    /// Source column name: the output layer keeps it (its value column
    /// and contour levels are named after the field, not "value").
    pub value_name: String,
    pub system: CellSystem,
    pub stat: GridStat,
    pub kernel: Kernel,
    pub output: GridOutput,
    /// Smoothing passes over the aggregated cells (0 = raw). Each pass
    /// averages a cell with its present neighbors only: empty cells stay
    /// empty and contribute nothing (no bleed past the data's edge).
    pub smooth_passes: u32,
    /// Focal operation applied after smoothing.
    pub post: PostOp,
}

/// Weighted accumulator: features spanning several cells contribute
/// their value apportioned by covered area (weight w in [0, 1] per
/// cell, summing to 1 across the feature's cells).
struct Acc {
    /// Σ w·v
    sum: f64,
    /// Σ w — the apportioned feature count.
    wsum: f64,
    /// (value, weight) pairs, kept only for the median.
    vals: Option<Vec<(f64, f64)>>,
    /// Σ w per interned code, kept only for majority/minority — the
    /// value then travels as a dictionary code cast to f64.
    cats: Option<HashMap<u32, f64>>,
}

impl Acc {
    fn push(&mut self, v: f64, w: f64) {
        self.sum += w * v;
        self.wsum += w;
        if let Some(vals) = &mut self.vals {
            vals.push((v, w));
        }
        if let Some(cats) = &mut self.cats {
            *cats.entry(v as u32).or_insert(0.0) += w;
        }
    }

    fn stat(&mut self, stat: GridStat) -> f64 {
        match stat {
            GridStat::Mean => self.sum / self.wsum.max(f64::EPSILON),
            GridStat::Sum | GridStat::Density => self.sum,
            GridStat::Count => self.wsum,
            // Ties break toward the smaller code (first value seen in
            // the scan), so two runs draw the same map.
            GridStat::Majority => self
                .cats
                .as_ref()
                .expect("majority keeps counts")
                .iter()
                .max_by(|(ka, wa), (kb, wb)| {
                    wa.partial_cmp(wb).unwrap().then(kb.cmp(ka))
                })
                .map(|(k, _)| *k as f64)
                .unwrap_or(f64::NAN),
            GridStat::Minority => self
                .cats
                .as_ref()
                .expect("minority keeps counts")
                .iter()
                .min_by(|(ka, wa), (kb, wb)| {
                    wa.partial_cmp(wb).unwrap().then(ka.cmp(kb))
                })
                .map(|(k, _)| *k as f64)
                .unwrap_or(f64::NAN),
            GridStat::Median => {
                // Weighted median: value where the cumulative weight
                // crosses half the total.
                let vals = self.vals.as_mut().expect("median keeps values");
                vals.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let half = vals.iter().map(|(_, w)| w).sum::<f64>() * 0.5;
                let mut cum = 0.0;
                for &(v, w) in vals.iter() {
                    cum += w;
                    if cum >= half {
                        return v;
                    }
                }
                vals.last().map(|&(v, _)| v).unwrap_or(0.0)
            }
        }
    }
}

/// Aggregate `store`'s `spec.value_col` into cells and write the grid as
/// GeoParquet at `dst`. Returns (cells, rows aggregated).
#[allow(clippy::type_complexity)]
pub fn compute(
    store: &FeatureStore,
    crs: &Crs,
    spec: &GridSpec,
    dst: &Path,
    progress: &(dyn Fn(f32) + Sync),
) -> Result<(usize, u64), String> {
    let wgs84 = Crs::wgs84();
    let geographic = !matches!(spec.system, CellSystem::Square { .. });
    let text = spec.stat.text();
    if text
        && (spec.smooth_passes > 0
            || spec.post != PostOp::None
            || matches!(spec.output, GridOutput::Contours { .. }))
    {
        return Err(format!(
            "{} produces a category per cell, which has no numeric surface \
             to smooth, contour or filter",
            spec.stat.label(),
        ));
    }
    // Dictionary for text statistics: the value travels through the
    // pipeline as its code, and turns back into the string at the end.
    let mut dict: Vec<String> = Vec::new();

    // ---- pass 1: bbox + value per row -------------------------------
    // Features whose bbox fits in one cell assign wholly (weight 1) with
    // no geometry access; the rest are stashed for area apportionment.
    let keep_vals = spec.stat == GridStat::Median;
    let blank = || Acc {
        sum: 0.0,
        wsum: 0.0,
        vals: keep_vals.then(Vec::new),
        cats: text.then(HashMap::new),
    };
    let mut sq: HashMap<(i64, i64), Acc> = HashMap::new();
    let mut geo: HashMap<u64, Acc> = HashMap::new();
    let mut rows_used = 0u64;
    let mut stash: Vec<(u32, f64)> = Vec::new();
    // Geographic cell of a data-CRS point, or None outside the domain.
    let to_geo_cell = |x: f64, y: f64| -> Option<u64> {
        let (lon, lat) = transform_point(crs, &wgs84, x, y).ok()?;
        match spec.system {
            CellSystem::H3 { res } => {
                let res = h3o::Resolution::try_from(res).ok()?;
                let ll = h3o::LatLng::new(lat, lon).ok()?;
                Some(u64::from(ll.to_cell(res)))
            }
            CellSystem::A5 { res } => a5::lonlat_to_cell(a5::LonLat::new(lon, lat), res).ok(),
            CellSystem::Square { .. } => None,
        }
    };
    scan_bbox_values(store, spec.value_col, text.then_some(&mut dict), &mut |row, b, v, frac| {
        progress(frac * 0.55);
        match spec.system {
            CellSystem::Square { sx, sy } => {
                let (ix0, iy0) = ((b[0] / sx).floor() as i64, (b[1] / sy).floor() as i64);
                let (ix1, iy1) = ((b[2] / sx).floor() as i64, (b[3] / sy).floor() as i64);
                if ix0 == ix1 && iy0 == iy1 {
                    sq.entry((ix0, iy0)).or_insert_with(blank).push(v, 1.0);
                    rows_used += 1;
                } else {
                    stash.push((row, v));
                }
            }
            _ => {
                // Same cell for all four bbox corners ⇒ contained.
                let c00 = to_geo_cell(b[0], b[1]);
                let Some(c) = c00 else { return };
                let contained = [(b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
                    .iter()
                    .all(|&(x, y)| to_geo_cell(x, y) == Some(c));
                if contained {
                    geo.entry(c).or_insert_with(blank).push(v, 1.0);
                    rows_used += 1;
                } else {
                    stash.push((row, v));
                }
            }
        }
    })?;

    // ---- pass 2: apportion boundary-crossers by covered area ---------
    // Chunks run in parallel (geometry decode + per-sample cell lookups
    // dominate); each returns (cell, value, weight) triples merged
    // sequentially into the accumulators.
    if !stash.is_empty() {
        use rayon::prelude::*;
        stash.sort_unstable_by_key(|(r, _)| *r);
        const CHUNK: usize = 16_384;
        let chunks: Vec<&[(u32, f64)]> = stash.chunks(CHUNK).collect();
        let n_chunks = chunks.len();
        let done = std::sync::atomic::AtomicUsize::new(0);
        match spec.system {
            CellSystem::Square { sx, sy } => {
                let parts: Result<Vec<Vec<((i64, i64), f64, f64)>>, String> = chunks
                    .par_iter()
                    .map(|chunk| {
                        let rows: Vec<u32> = chunk.iter().map(|(r, _)| *r).collect();
                        let geoms = store.fetch_geoms(&rows)?;
                        let mut out = Vec::new();
                        for ((_, g), &(_, v)) in geoms.iter().zip(chunk.iter()) {
                            let Some(g) = g else { continue };
                            apportion_square(g, sx, sy, &mut |k, w| out.push((k, v, w)));
                        }
                        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        progress(0.55 + 0.15 * d as f32 / n_chunks as f32);
                        Ok(out)
                    })
                    .collect();
                for part in parts? {
                    for (k, v, w) in part {
                        sq.entry(k).or_insert_with(blank).push(v, w);
                    }
                }
                rows_used += stash.len() as u64;
            }
            _ => {
                let parts: Result<Vec<Vec<(u64, f64, f64)>>, String> = chunks
                    .par_iter()
                    .map(|chunk| {
                        let rows: Vec<u32> = chunk.iter().map(|(r, _)| *r).collect();
                        let geoms = store.fetch_geoms(&rows)?;
                        let mut out = Vec::new();
                        for ((_, g), &(_, v)) in geoms.iter().zip(chunk.iter()) {
                            let Some(g) = g else { continue };
                            apportion_sampled(g, &to_geo_cell, &mut |k, w| {
                                out.push((k, v, w))
                            });
                        }
                        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        progress(0.55 + 0.15 * d as f32 / n_chunks as f32);
                        Ok(out)
                    })
                    .collect();
                for part in parts? {
                    for (k, v, w) in part {
                        geo.entry(k).or_insert_with(blank).push(v, w);
                    }
                }
                rows_used += stash.len() as u64;
            }
        }
    }

    // ---- per-cell statistic ------------------------------------------
    let mut sq_vals: HashMap<(i64, i64), (f64, f64)> = sq
        .iter_mut()
        .map(|(k, a)| (*k, (a.stat(spec.stat), a.wsum)))
        .collect();
    let mut geo_vals: HashMap<u64, (f64, f64)> = geo
        .iter_mut()
        .map(|(k, a)| (*k, (a.stat(spec.stat), a.wsum)))
        .collect();
    drop(sq);
    drop(geo);
    if spec.stat == GridStat::Density {
        match spec.system {
            CellSystem::Square { sx, sy } => {
                for v in sq_vals.values_mut() {
                    v.0 /= sx * sy;
                }
            }
            CellSystem::H3 { .. } => {
                for (k, v) in geo_vals.iter_mut() {
                    if let Ok(c) = h3o::CellIndex::try_from(*k) {
                        v.0 /= c.area_m2();
                    }
                }
            }
            CellSystem::A5 { res } => {
                let area = a5::cell_area(res).max(f64::EPSILON);
                for v in geo_vals.values_mut() {
                    v.0 /= area;
                }
            }
        }
    }

    // ---- smoothing passes --------------------------------------------
    for pass in 0..spec.smooth_passes {
        progress(0.7 + 0.12 * (pass as f32 + 1.0) / spec.smooth_passes.max(1) as f32);
        match spec.system {
            CellSystem::Square { .. } => {
                sq_vals = smooth_pass(&sq_vals, square_neighbors, spec.kernel);
            }
            CellSystem::H3 { .. } => {
                geo_vals = smooth_pass(&geo_vals, h3_neighbors, Kernel::Box);
            }
            CellSystem::A5 { .. } => {
                geo_vals = smooth_pass(&geo_vals, a5_neighbors, Kernel::Box);
            }
        }
    }

    // ---- focal operation ----------------------------------------------
    progress(0.85);
    match spec.post {
        PostOp::None => {}
        PostOp::Hillshade {
            azimuth_deg,
            altitude_deg,
        } => {
            let CellSystem::Square { sx, sy } = spec.system else {
                return Err("hillshade needs the square grid (it reads the \
                            gradient off the regular lattice)"
                    .into());
            };
            sq_vals = hillshade(&sq_vals, sx, sy, azimuth_deg, altitude_deg);
        }
        op => {
            // Focal std is one pass; open and close are two passes of the
            // opposite extremes.
            let steps: &[Focal] = match op {
                PostOp::FocalStd => &[Focal::Std],
                PostOp::Open => &[Focal::Min, Focal::Max],
                _ => &[Focal::Max, Focal::Min],
            };
            for &step in steps {
                match spec.system {
                    CellSystem::Square { .. } => {
                        sq_vals = focal_pass(&sq_vals, square_neighbors, step);
                    }
                    CellSystem::H3 { .. } => {
                        geo_vals = focal_pass(&geo_vals, h3_neighbors, step);
                    }
                    CellSystem::A5 { .. } => {
                        geo_vals = focal_pass(&geo_vals, a5_neighbors, step);
                    }
                }
            }
        }
    }
    let value_name = format!("{}{}", spec.value_name, spec.post.suffix());

    // ---- materialize --------------------------------------------------
    progress(0.9);
    let out_crs = if geographic { &wgs84 } else { crs };
    if let GridOutput::Contours { levels, spacing } = spec.output {
        let CellSystem::Square { sx, sy } = spec.system else {
            return Err("contour lines need the square grid (marching squares)".into());
        };
        let (batch, schema, lines) =
            contour_lines(&sq_vals, &value_name, sx, sy, levels, spacing)?;
        crate::sql::export::write_result(dst, &schema, std::slice::from_ref(&batch), 0, out_crs)?;
        progress(1.0);
        return Ok((lines, rows_used));
    }
    let (batch, schema, cells) = match spec.system {
        CellSystem::Square { sx, sy } => {
            materialize(&sq_vals, &value_name, text.then_some(dict.as_slice()), |&(ix, iy)| {
                let (x0, y0) = (ix as f64 * sx, iy as f64 * sy);
                (
                    format!("{ix}:{iy}"),
                    vec![
                        (x0, y0),
                        (x0 + sx, y0),
                        (x0 + sx, y0 + sy),
                        (x0, y0 + sy),
                        (x0, y0),
                    ],
                )
            })?
        }
        CellSystem::H3 { .. } => materialize(&geo_vals, &value_name, text.then_some(dict.as_slice()), |&c| {
            let cell = h3o::CellIndex::try_from(c).expect("aggregated cell is valid");
            let mut ring: Vec<(f64, f64)> = cell
                .boundary()
                .iter()
                .map(|ll| (ll.lng(), ll.lat()))
                .collect();
            unwrap_dateline(&mut ring);
            if let Some(&first) = ring.first() {
                ring.push(first);
            }
            (cell.to_string(), ring)
        })?,
        CellSystem::A5 { .. } => materialize(&geo_vals, &value_name, text.then_some(dict.as_slice()), |&c| {
            let b = a5::cell_to_boundary(c, None).unwrap_or_default();
            let mut ring: Vec<(f64, f64)> =
                b.iter().map(|ll| (ll.longitude(), ll.latitude())).collect();
            unwrap_dateline(&mut ring);
            if let Some(&first) = ring.first() {
                ring.push(first);
            }
            (a5::u64_to_hex(c), ring)
        })?,
    };
    crate::sql::export::write_result(dst, &schema, std::slice::from_ref(&batch), 0, out_crs)?;
    progress(1.0);
    Ok((cells, rows_used))
}

/// Level values for the isolines, strictly inside the value range and
/// strictly increasing. Quantile levels collapse onto each other where
/// many cells share a value (a big flat background), so duplicates are
/// dropped rather than drawn on top of one another.
fn contour_levels(
    vals: &HashMap<(i64, i64), (f64, f64)>,
    levels: usize,
    spacing: ContourSpacing,
) -> Result<Vec<f64>, String> {
    let n = levels.clamp(1, 64);
    let mut sorted: Vec<f64> = vals.values().map(|&(v, _)| v).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (vmin, vmax) = (sorted[0], sorted[sorted.len() - 1]);
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for k in 1..=n {
        let f = k as f64 / (n + 1) as f64;
        let v = match spacing {
            ContourSpacing::Equal => vmin + (vmax - vmin) * f,
            ContourSpacing::Quantile => quantile_sorted(&sorted, f),
        };
        if v <= vmin || v >= vmax {
            continue;
        }
        // Ties in the source distribution map several quantiles to the
        // same value; keep one line, not five coincident ones.
        if out.last().is_some_and(|&p| v <= p) {
            continue;
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err(
            "too many cells share one value for these levels — try equal spacing or fewer levels"
                .into(),
        );
    }
    Ok(out)
}

/// Linear-interpolation quantile of an ascending slice (non-empty).
fn quantile_sorted(sorted: &[f64], f: f64) -> f64 {
    let pos = f.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let t = pos - lo as f64;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// Marching squares over the cell-center lattice: isolines at `levels`
/// values placed per `spacing`, linearly interpolated, stitched into
/// polylines. Cells missing from the grid stop the contours (no data
/// invented past the edge).
#[allow(clippy::type_complexity)]
fn contour_lines(
    vals: &HashMap<(i64, i64), (f64, f64)>,
    value_name: &str,
    sx: f64,
    sy: f64,
    levels: usize,
    spacing: ContourSpacing,
) -> Result<(RecordBatch, SchemaRef, usize), String> {
    if vals.len() < 4 {
        return Err("not enough cells for contours".into());
    }
    let min_ix = vals.keys().map(|k| k.0).min().unwrap();
    let max_ix = vals.keys().map(|k| k.0).max().unwrap();
    let min_iy = vals.keys().map(|k| k.1).min().unwrap();
    let max_iy = vals.keys().map(|k| k.1).max().unwrap();
    let (w, h) = (
        (max_ix - min_ix + 1) as usize,
        (max_iy - min_iy + 1) as usize,
    );
    if w < 2 || h < 2 {
        return Err("grid is a single row/column — no contours".into());
    }
    if w.saturating_mul(h) > 64_000_000 {
        return Err("grid too large for contouring — increase the cell size".into());
    }
    let mut g = vec![f64::NAN; w * h];
    let (mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&(ix, iy), &(v, _)) in vals {
        g[(ix - min_ix) as usize + (iy - min_iy) as usize * w] = v;
        vmin = vmin.min(v);
        vmax = vmax.max(v);
    }
    if vmax <= vmin {
        return Err("flat value surface — no contours".into());
    }
    let level_values = contour_levels(vals, levels, spacing)?;

    // Cell-center position of grid node (i, j), data CRS.
    let pos = |i: usize, j: usize| -> [f64; 2] {
        [
            (min_ix + i as i64) as f64 * sx + sx * 0.5,
            (min_iy + j as i64) as f64 * sy + sy * 0.5,
        ]
    };

    let mut wkb = BinaryBuilder::new();
    let mut level_col: Vec<f64> = Vec::new();
    let mut n_lines = 0usize;
    for &level in &level_values {
        let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for j in 0..h - 1 {
            for i in 0..w - 1 {
                let v = [
                    g[i + j * w],           // c0 bottom-left
                    g[i + 1 + j * w],       // c1 bottom-right
                    g[i + 1 + (j + 1) * w], // c2 top-right
                    g[i + (j + 1) * w],     // c3 top-left
                ];
                if v.iter().any(|x| x.is_nan()) {
                    continue;
                }
                let p = [pos(i, j), pos(i + 1, j), pos(i + 1, j + 1), pos(i, j + 1)];
                let idx = (v[0] >= level) as usize
                    | ((v[1] >= level) as usize) << 1
                    | ((v[2] >= level) as usize) << 2
                    | ((v[3] >= level) as usize) << 3;
                // Interpolated crossing on the edge between corners a, b.
                let cross = |a: usize, b: usize| -> [f64; 2] {
                    let t = ((level - v[a]) / (v[b] - v[a])).clamp(0.0, 1.0);
                    [
                        p[a][0] + t * (p[b][0] - p[a][0]),
                        p[a][1] + t * (p[b][1] - p[a][1]),
                    ]
                };
                // Edges: E0 bottom (0-1), E1 right (1-2), E2 top (3-2), E3 left (0-3).
                let e = |k: usize| -> [f64; 2] {
                    match k {
                        0 => cross(0, 1),
                        1 => cross(1, 2),
                        2 => cross(3, 2),
                        _ => cross(0, 3),
                    }
                };
                match idx {
                    0 | 15 => {}
                    1 | 14 => segments.push((e(3), e(0))),
                    2 | 13 => segments.push((e(0), e(1))),
                    3 | 12 => segments.push((e(3), e(1))),
                    4 | 11 => segments.push((e(1), e(2))),
                    6 | 9 => segments.push((e(0), e(2))),
                    7 | 8 => segments.push((e(2), e(3))),
                    5 | 10 => {
                        // Saddle: disambiguate by the block center.
                        let center = (v[0] + v[1] + v[2] + v[3]) * 0.25;
                        let flip = (center >= level) == (idx == 5);
                        if flip {
                            segments.push((e(3), e(2)));
                            segments.push((e(0), e(1)));
                        } else {
                            segments.push((e(3), e(0)));
                            segments.push((e(1), e(2)));
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        for line in stitch_segments(segments, sx.min(sy) * 1e-6) {
            if line.len() < 2 {
                continue;
            }
            let ls = geo_types::Geometry::LineString(geo_types::LineString(
                line.iter().map(|p| geo_types::Coord { x: p[0], y: p[1] }).collect(),
            ));
            wkb.append_value(crate::data::import::to_wkb(&ls)?);
            level_col.push(level);
            n_lines += 1;
        }
    }
    if n_lines == 0 {
        return Err("no contours crossed the value range".into());
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, false),
        Field::new(value_name, DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(wkb.finish()) as ArrayRef,
            Arc::new(Float64Array::from(level_col)),
        ],
    )
    .map_err(|e| format!("contour batch: {e}"))?;
    Ok((batch, schema, n_lines))
}

/// Join 2-point marching-squares segments into polylines by shared
/// endpoints (quantized to `eps`).
fn stitch_segments(segments: Vec<([f64; 2], [f64; 2])>, eps: f64) -> Vec<Vec<[f64; 2]>> {
    let key = |p: [f64; 2]| -> (i64, i64) {
        ((p[0] / eps).round() as i64, (p[1] / eps).round() as i64)
    };
    // endpoint key -> (segment index, end: 0|1)
    let mut ends: HashMap<(i64, i64), Vec<(usize, u8)>> = HashMap::new();
    for (i, (a, b)) in segments.iter().enumerate() {
        ends.entry(key(*a)).or_default().push((i, 0));
        ends.entry(key(*b)).or_default().push((i, 1));
    }
    let mut used = vec![false; segments.len()];
    let mut out = Vec::new();
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let (a, b) = segments[start];
        let mut chain = vec![a, b];
        // Extend forward from the tail, then backward from the head.
        for _ in 0..2 {
            loop {
                let tail = *chain.last().unwrap();
                let Some(cands) = ends.get(&key(tail)) else { break };
                let Some(&(si, end)) = cands.iter().find(|(si, _)| !used[*si]) else {
                    break;
                };
                used[si] = true;
                let (sa, sb) = segments[si];
                chain.push(if end == 0 { sb } else { sa });
            }
            chain.reverse();
        }
        out.push(chain);
    }
    out
}

/// The eight square-grid neighbors, with their offsets (the kernel
/// weights read them).
fn square_neighbors(&(ix, iy): &(i64, i64)) -> Vec<((i64, i64), (i64, i64))> {
    let mut out = Vec::with_capacity(8);
    for dx in -1..=1i64 {
        for dy in -1..=1i64 {
            if (dx, dy) != (0, 0) {
                out.push(((ix + dx, iy + dy), (dx, dy)));
            }
        }
    }
    out
}

/// H3 ring-1 neighbors. Offsets are meaningless on a hex lattice, so
/// they all read as the box weight.
fn h3_neighbors(c: &u64) -> Vec<(u64, (i64, i64))> {
    let c = *c;
    h3o::CellIndex::try_from(c)
        .map(|cell| {
            cell.grid_disk::<Vec<_>>(1)
                .into_iter()
                .map(u64::from)
                .filter(|n| *n != c)
                .map(|n| (n, (1i64, 0i64)))
                .collect()
        })
        .unwrap_or_default()
}

/// A5 ring-1 neighbors, same convention as H3.
fn a5_neighbors(c: &u64) -> Vec<(u64, (i64, i64))> {
    let c = *c;
    a5::grid_disk(c, 1)
        .unwrap_or_default()
        .into_iter()
        .filter(|n| *n != c)
        .map(|n| (n, (1i64, 0i64)))
        .collect()
}

/// Focal operator over a cell and its present neighbors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focal {
    /// Population standard deviation of the neighborhood.
    Std,
    /// Erosion.
    Min,
    /// Dilation.
    Max,
}

/// One focal pass over present cells. Absent neighbors are absent, not
/// zero: an edge cell is evaluated over the neighbors it has, so the
/// operator never invents data past the edge of the grid.
fn focal_pass<K: Hash + Eq + Copy>(
    vals: &HashMap<K, (f64, f64)>,
    neighbors: impl Fn(&K) -> Vec<(K, (i64, i64))>,
    op: Focal,
) -> HashMap<K, (f64, f64)> {
    vals.iter()
        .map(|(k, &(v, count))| {
            let mut acc = v;
            let (mut sum, mut sq, mut n) = (v, v * v, 1.0f64);
            for (nk, _) in neighbors(k) {
                let Some(&(nv, _)) = vals.get(&nk) else {
                    continue;
                };
                match op {
                    Focal::Std => {
                        sum += nv;
                        sq += nv * nv;
                        n += 1.0;
                    }
                    Focal::Min => acc = acc.min(nv),
                    Focal::Max => acc = acc.max(nv),
                }
            }
            let out = match op {
                Focal::Std => {
                    let mean = sum / n;
                    (sq / n - mean * mean).max(0.0).sqrt()
                }
                _ => acc,
            };
            (*k, (out, count))
        })
        .collect()
}

/// Horn hillshade of the square-cell surface, 0..255.
///
/// Values are dollars, people or parcels rather than metres, so no fixed
/// z-factor makes sense: the exaggeration is fitted so the median
/// gradient lights at 45°, which keeps the relief readable whatever the
/// field's unit and range. Cells missing a neighbor borrow their own
/// value for it (a flat continuation), which is the standard edge rule.
fn hillshade(
    vals: &HashMap<(i64, i64), (f64, f64)>,
    sx: f64,
    sy: f64,
    azimuth_deg: f64,
    altitude_deg: f64,
) -> HashMap<(i64, i64), (f64, f64)> {
    // Horn's 3×3 weighted differences, y increasing north.
    let grad = |&(ix, iy): &(i64, i64), c: f64| -> (f64, f64) {
        let at = |dx: i64, dy: i64| vals.get(&(ix + dx, iy + dy)).map_or(c, |&(v, _)| v);
        let (nw, n, ne) = (at(-1, 1), at(0, 1), at(1, 1));
        let (w, e) = (at(-1, 0), at(1, 0));
        let (sw, s, se) = (at(-1, -1), at(0, -1), at(1, -1));
        let dzdx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) / (8.0 * sx);
        let dzdy = ((nw + 2.0 * n + ne) - (sw + 2.0 * s + se)) / (8.0 * sy);
        (dzdx, dzdy)
    };
    let grads: HashMap<(i64, i64), (f64, f64)> = vals
        .iter()
        .map(|(k, &(v, _))| (*k, grad(k, v)))
        .collect();
    let mut mags: Vec<f64> = grads
        .values()
        .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
        .filter(|m| *m > 0.0)
        .collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let z = match mags.get(mags.len() / 2) {
        Some(&median) if median > 0.0 => 1.0 / median,
        _ => 1.0,
    };

    let zenith = (90.0 - altitude_deg).to_radians();
    let sun = azimuth_deg.to_radians();
    vals.iter()
        .map(|(k, &(_, count))| {
            let (dzdx, dzdy) = grads[k];
            let slope = (z * (dzdx * dzdx + dzdy * dzdy).sqrt()).atan();
            // Compass azimuth (clockwise from north) of the downhill
            // direction: the face the sun sees when sun == aspect.
            let aspect = (-dzdx).atan2(-dzdy);
            let shade = zenith.cos() * slope.cos()
                + zenith.sin() * slope.sin() * (sun - aspect).cos();
            (*k, ((shade.clamp(0.0, 1.0) * 255.0), count))
        })
        .collect()
}

/// One renormalized smoothing pass over present cells.
fn smooth_pass<K: Hash + Eq + Copy>(
    vals: &HashMap<K, (f64, f64)>,
    neighbors: impl Fn(&K) -> Vec<(K, (i64, i64))>,
    kernel: Kernel,
) -> HashMap<K, (f64, f64)> {
    let weight = |off: (i64, i64)| -> f64 {
        match kernel {
            Kernel::Box => 1.0,
            // 3×3 binomial approximation of a Gaussian.
            Kernel::Gaussian => match (off.0.abs(), off.1.abs()) {
                (0, 0) => 4.0,
                (1, 0) | (0, 1) => 2.0,
                _ => 1.0,
            },
        }
    };
    let center_w = match kernel {
        Kernel::Box => 1.0,
        Kernel::Gaussian => 4.0,
    };
    vals.iter()
        .map(|(k, &(v, count))| {
            let (mut num, mut den) = (v * center_w, center_w);
            for (nk, off) in neighbors(k) {
                if let Some(&(nv, _)) = vals.get(&nk) {
                    let w = weight(off);
                    num += nv * w;
                    den += w;
                }
            }
            (*k, (num / den, count))
        })
        .collect()
}

/// Unwrap a geographic ring so consecutive vertices never jump more
/// than 180° of longitude. H3 and A5 return dateline-crossing cells with
/// vertices on both sides of ±180; taken literally that ring wraps the
/// whole globe. Shifting the far side past ±180 keeps the cell the small
/// polygon it is, and the display projection reads it correctly.
fn unwrap_dateline(ring: &mut [(f64, f64)]) {
    let mut shift = 0.0f64;
    for i in 1..ring.len() {
        let d = (ring[i].0 + shift) - ring[i - 1].0;
        if d > 180.0 {
            shift -= 360.0;
        } else if d < -180.0 {
            shift += 360.0;
        }
        ring[i].0 += shift;
    }
}

/// Cells → one arrow batch: WKB polygon, value, count, cell id. With
/// `dict`, the value is a dictionary code and the column comes out as
/// the text it stands for.
#[allow(clippy::type_complexity)]
fn materialize<K: Hash + Eq + Copy>(
    vals: &HashMap<K, (f64, f64)>,
    value_name: &str,
    dict: Option<&[String]>,
    cell_geom: impl Fn(&K) -> (String, Vec<(f64, f64)>),
) -> Result<(RecordBatch, SchemaRef, usize), String> {
    let mut wkb = BinaryBuilder::new();
    let mut value: Vec<f64> = Vec::with_capacity(vals.len());
    let mut text = dict.map(|_| StringBuilder::new());
    let mut count: Vec<f64> = Vec::with_capacity(vals.len());
    let mut ids = StringBuilder::new();
    for (k, &(v, c)) in vals {
        if let (Some(d), Some(_)) = (dict, &text) {
            // A cell with weight has at least one category, so this only
            // guards a NaN that a bug upstream would have produced.
            if !v.is_finite() || v as usize >= d.len() {
                continue;
            }
        }
        let (id, ring) = cell_geom(k);
        if ring.len() < 4 {
            continue;
        }
        let poly = geo_types::Geometry::Polygon(geo_types::Polygon::new(
            geo_types::LineString(
                ring.iter().map(|&(x, y)| geo_types::Coord { x, y }).collect(),
            ),
            vec![],
        ));
        wkb.append_value(crate::data::import::to_wkb(&poly)?);
        match (&mut text, dict) {
            (Some(t), Some(d)) => t.append_value(&d[v as usize]),
            _ => value.push(v),
        }
        count.push(c);
        ids.append_value(id);
    }
    let value_type = if dict.is_some() { DataType::Utf8 } else { DataType::Float64 };
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, false),
        Field::new(value_name, value_type, false),
        Field::new("count", DataType::Float64, false),
        Field::new("cell", DataType::Utf8, false),
    ]));
    let n = count.len();
    let value_col: ArrayRef = match &mut text {
        Some(t) => Arc::new(t.finish()),
        None => Arc::new(Float64Array::from(value)),
    };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(wkb.finish()) as ArrayRef,
            value_col,
            Arc::new(Float64Array::from(count)),
            Arc::new(ids.finish()),
        ],
    )
    .map_err(|e| format!("grid batch: {e}"))?;
    Ok((batch, schema, n))
}

/// Split a feature's value across the square cells it covers: exact
/// axis-aligned rectangle clipping per cell, weight = covered fraction
/// of the feature's total area. Degenerate/non-areal geometries fall
/// back to their bbox-center cell with weight 1.
fn apportion_square(
    g: &geo_types::Geometry<f64>,
    sx: f64,
    sy: f64,
    add: &mut dyn FnMut((i64, i64), f64),
) {
    use geo::{Area, BoundingRect};
    // Only used to detect degenerate (zero-area) input; the weights are
    // normalized against the measured coverage below.
    let total = g.unsigned_area();
    let centroid_fallback = |add: &mut dyn FnMut((i64, i64), f64)| {
        if let Some(r) = g.bounding_rect() {
            let (cx, cy) = ((r.min().x + r.max().x) * 0.5, (r.min().y + r.max().y) * 0.5);
            add(((cx / sx).floor() as i64, (cy / sy).floor() as i64), 1.0);
        }
    };
    if total <= 0.0 {
        centroid_fallback(add);
        return;
    }
    let mut covered: HashMap<(i64, i64), f64> = HashMap::new();
    let mut per_poly = |poly: &geo_types::Polygon<f64>| {
        let Some(r) = poly.bounding_rect() else { return };
        let (ix0, ix1) = ((r.min().x / sx).floor() as i64, (r.max().x / sx).floor() as i64);
        let (iy0, iy1) = ((r.min().y / sy).floor() as i64, (r.max().y / sy).floor() as i64);
        for ix in ix0..=ix1 {
            for iy in iy0..=iy1 {
                let rect = [
                    ix as f64 * sx,
                    iy as f64 * sy,
                    (ix + 1) as f64 * sx,
                    (iy + 1) as f64 * sy,
                ];
                let mut a = ring_area_in_rect(&poly.exterior().0, rect);
                for hole in poly.interiors() {
                    a -= ring_area_in_rect(&hole.0, rect);
                }
                if a > 0.0 {
                    *covered.entry((ix, iy)).or_insert(0.0) += a;
                }
            }
        }
    };
    match g {
        geo_types::Geometry::Polygon(p) => per_poly(p),
        geo_types::Geometry::MultiPolygon(mp) => mp.0.iter().for_each(&mut per_poly),
        _ => {
            centroid_fallback(add);
            return;
        }
    }
    if covered.is_empty() {
        centroid_fallback(add);
        return;
    }
    // Normalize by the area actually accounted for, not by the ring's
    // own area: they are equal for a valid ring (the cells tile its
    // bbox), and for a self-intersecting one — routine in digitized
    // cadastral data — `unsigned_area` returns the winding-cancelled
    // area, which can be far smaller than what the clipper measures and
    // would push the feature's total weight above 1.
    let sum: f64 = covered.values().sum();
    if sum <= 0.0 {
        centroid_fallback(add);
        return;
    }
    for (k, a) in covered {
        add(k, a / sum);
    }
}

/// Unsigned area of `ring ∩ rect` (Sutherland–Hodgman against the four
/// half-planes, then the shoelace).
fn ring_area_in_rect(ring: &[geo_types::Coord<f64>], rect: [f64; 4]) -> f64 {
    let mut pts: Vec<geo_types::Coord<f64>> = ring.to_vec();
    // inside(p) per clip edge; intersection parameterized on the edge.
    for edge in 0..4 {
        if pts.len() < 3 {
            return 0.0;
        }
        let inside = |p: &geo_types::Coord<f64>| -> bool {
            match edge {
                0 => p.x >= rect[0],
                1 => p.x <= rect[2],
                2 => p.y >= rect[1],
                _ => p.y <= rect[3],
            }
        };
        let cross = |a: &geo_types::Coord<f64>, b: &geo_types::Coord<f64>| {
            let t = match edge {
                0 => (rect[0] - a.x) / (b.x - a.x),
                1 => (rect[2] - a.x) / (b.x - a.x),
                2 => (rect[1] - a.y) / (b.y - a.y),
                _ => (rect[3] - a.y) / (b.y - a.y),
            };
            geo_types::Coord {
                x: a.x + t * (b.x - a.x),
                y: a.y + t * (b.y - a.y),
            }
        };
        let mut out = Vec::with_capacity(pts.len() + 4);
        for i in 0..pts.len() {
            let (a, b) = (&pts[i], &pts[(i + 1) % pts.len()]);
            match (inside(a), inside(b)) {
                (true, true) => out.push(*b),
                (true, false) => out.push(cross(a, b)),
                (false, true) => {
                    out.push(cross(a, b));
                    out.push(*b);
                }
                (false, false) => {}
            }
        }
        pts = out;
    }
    if pts.len() < 3 {
        return 0.0;
    }
    let mut a2 = 0.0;
    for i in 0..pts.len() {
        let (p, q) = (&pts[i], &pts[(i + 1) % pts.len()]);
        a2 += p.x * q.y - q.x * p.y;
    }
    (a2 * 0.5).abs()
}

/// Split a feature's value across geographic cells by sampling an 8×8
/// lattice over its bbox: weight = samples in cell / samples inside the
/// polygon. Membership is a scanline even-odd test (one crossing pass
/// over the ring edges per sample row — no per-point ray cast), which
/// handles holes and multipolygons naturally. Exact hexagon/pentagon
/// clipping isn't worth it at carroyage cell sizes. Falls back to the
/// bbox-center cell for degenerate/non-areal geometries.
fn apportion_sampled(
    g: &geo_types::Geometry<f64>,
    to_cell: &dyn Fn(f64, f64) -> Option<u64>,
    add: &mut dyn FnMut(u64, f64),
) {
    use geo::BoundingRect;
    let Some(r) = g.bounding_rect() else { return };
    let centroid_fallback = |add: &mut dyn FnMut(u64, f64)| {
        let (cx, cy) = ((r.min().x + r.max().x) * 0.5, (r.min().y + r.max().y) * 0.5);
        if let Some(c) = to_cell(cx, cy) {
            add(c, 1.0);
        }
    };
    let rings: Vec<&[geo_types::Coord<f64>]> = match g {
        geo_types::Geometry::Polygon(p) => std::iter::once(p.exterior())
            .chain(p.interiors())
            .map(|r| r.0.as_slice())
            .collect(),
        geo_types::Geometry::MultiPolygon(mp) => mp
            .0
            .iter()
            .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
            .map(|r| r.0.as_slice())
            .collect(),
        _ => {
            centroid_fallback(add);
            return;
        }
    };
    // Coarse first; a geometry thinner than the sample spacing (a ring,
    // a river strip, a diagonal sliver) needs a finer lattice before the
    // bbox-center fallback, which for an annulus would land in the hole.
    for n in [8usize, 32] {
        if sample_rings(&rings, &r, n, to_cell, add) {
            return;
        }
    }
    centroid_fallback(add);
}

/// One sampling pass at `n`×`n` over `r`. Returns false when no sample
/// landed inside the rings (or none of those could be classified), so
/// the caller can refine or fall back.
fn sample_rings(
    rings: &[&[geo_types::Coord<f64>]],
    r: &geo_types::Rect<f64>,
    n: usize,
    to_cell: &dyn Fn(f64, f64) -> Option<u64>,
    add: &mut dyn FnMut(u64, f64),
) -> bool {
    let (dx, dy) = (
        (r.max().x - r.min().x) / n as f64,
        (r.max().y - r.min().y) / n as f64,
    );
    if dx <= 0.0 || dy <= 0.0 {
        return false;
    }
    let mut per_cell: HashMap<u64, u32> = HashMap::new();
    let mut xs: Vec<f64> = Vec::new();
    for j in 0..n {
        let y = r.min().y + (j as f64 + 0.5) * dy;
        xs.clear();
        for ring in rings {
            let len = ring.len();
            if len < 3 {
                continue;
            }
            // Rings from WKB carry the duplicate closing point; guard
            // with an explicit wrap segment in case one doesn't.
            let closed = ring[0] == ring[len - 1];
            let segs = if closed { len - 1 } else { len };
            for i in 0..segs {
                let (a, b) = (ring[i], ring[(i + 1) % len]);
                if (a.y > y) != (b.y > y) {
                    let t = (y - a.y) / (b.y - a.y);
                    xs.push(a.x + t * (b.x - a.x));
                }
            }
        }
        if xs.is_empty() {
            continue;
        }
        xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..n {
            let x = r.min().x + (i as f64 + 0.5) * dx;
            // Odd number of crossings to the left ⇒ inside (even-odd).
            if xs.partition_point(|&c| c < x) % 2 == 1
                && let Some(c) = to_cell(x, y)
            {
                *per_cell.entry(c).or_insert(0) += 1;
            }
        }
    }
    // Normalize over the samples that produced a cell, not over every
    // sample inside the rings: a sample whose inverse projection fails
    // (near a pole, or an A5 face singularity) must not shrink the
    // feature's total weight below 1.
    let classified: u32 = per_cell.values().sum();
    if classified == 0 {
        return false;
    }
    for (c, k) in per_cell {
        add(c, k as f64 / classified as f64);
    }
    true
}

/// Stream (row, data-CRS bbox, value, progress) for every row with a
/// usable extent and finite value. Uses the covering bbox column when
/// present (columnar, no geometry decode); falls back to decoding
/// geometries otherwise.
/// The value column of one batch as f64 per row: a numeric cast, or —
/// when interning — the column cast to text, trimmed and dictionary-
/// coded, the code travelling as the value. Null, blank and non-finite
/// slots are None.
fn batch_values(
    col: &ArrayRef,
    interner: &mut Option<(HashMap<String, u32>, &mut Vec<String>)>,
) -> Result<Vec<Option<f64>>, String> {
    match interner {
        None => {
            let a = arrow::compute::cast(col, &DataType::Float64)
                .map_err(|e| format!("value cast: {e}"))?;
            let a = a.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok((0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i)).filter(|v| v.is_finite()))
                .collect())
        }
        Some((map, dict)) => {
            let a = arrow::compute::cast(col, &DataType::Utf8)
                .map_err(|e| format!("text cast: {e}"))?;
            let a = a.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            Ok((0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        return None;
                    }
                    let s = a.value(i).trim();
                    if s.is_empty() {
                        return None;
                    }
                    let code = match map.get(s) {
                        Some(&c) => c,
                        None => {
                            let c = dict.len() as u32;
                            dict.push(s.to_string());
                            map.insert(s.to_string(), c);
                            c
                        }
                    };
                    Some(code as f64)
                })
                .collect())
        }
    }
}

fn scan_bbox_values(
    store: &FeatureStore,
    value_col: usize,
    text_dict: Option<&mut Vec<String>>,
    f: &mut dyn FnMut(u32, [f64; 4], f64, f32),
) -> Result<(), String> {
    const CHUNK: u64 = 131_072;
    let total = store.total_rows();
    if total == 0 {
        return Err("layer has no rows".into());
    }
    let mut interner = text_dict.map(|d| (HashMap::new(), d));
    let covering = store.covering.clone();
    let mut start = 0u64;
    while start < total {
        let end = (start + CHUNK).min(total);
        let rows: Vec<u32> = (start as u32..end as u32).collect();
        let frac = end as f32 / total as f32;
        match &covering {
            Some(cov) => {
                // fetch() returns requested columns sorted by schema index.
                let want = [cov.root, value_col];
                let batches = store.fetch(&rows, Some(&want))?;
                let (bbox_pos, val_pos) = if cov.root < value_col { (0, 1) } else { (1, 0) };
                let mut row_base = 0usize;
                for b in &batches {
                    let s = b
                        .column(bbox_pos)
                        .as_any()
                        .downcast_ref::<arrow::array::StructArray>()
                        .ok_or("covering column is not a struct")?;
                    let child = |name: &str| -> Result<Float64Array, String> {
                        let c = s
                            .column_by_name(name)
                            .ok_or_else(|| format!("covering child {name} missing"))?;
                        let c = arrow::compute::cast(c, &DataType::Float64)
                            .map_err(|e| format!("covering cast: {e}"))?;
                        Ok(c.as_any().downcast_ref::<Float64Array>().unwrap().clone())
                    };
                    let (xmin, ymin) = (child(&cov.children[0])?, child(&cov.children[1])?);
                    let (xmax, ymax) = (child(&cov.children[2])?, child(&cov.children[3])?);
                    let vals = batch_values(b.column(val_pos), &mut interner)?;
                    for i in 0..b.num_rows() {
                        let Some(v) = vals[i] else { continue };
                        if xmin.is_null(i)
                            || ymin.is_null(i)
                            || xmax.is_null(i)
                            || ymax.is_null(i)
                        {
                            continue;
                        }
                        let bb = [xmin.value(i), ymin.value(i), xmax.value(i), ymax.value(i)];
                        // A null Arrow slot reads as 0.0 and NaN floors to
                        // cell 0 rather than erroring, so a corrupt covering
                        // column would quietly pile features onto the origin.
                        if !bb.iter().all(|c| c.is_finite()) {
                            continue;
                        }
                        f(rows[i + row_base], bb, v, frac);
                    }
                    row_base += b.num_rows();
                }
            }
            None => {
                // Geometry decode fallback (raw files without covering).
                let batches = store.fetch(&rows, Some(&[value_col]))?;
                let mut vals_all: Vec<Option<f64>> = Vec::with_capacity(rows.len());
                for b in &batches {
                    vals_all.extend(batch_values(b.column(0), &mut interner)?);
                }
                let geoms = store.fetch_geoms(&rows)?;
                for (k, ((_, g), v)) in geoms.iter().zip(&vals_all).enumerate() {
                    let (Some(g), Some(v)) = (g, v) else { continue };
                    use geo::BoundingRect;
                    if let Some(r) = g.bounding_rect() {
                        f(
                            rows[k],
                            [r.min().x, r.min().y, r.max().x, r.max().y],
                            *v,
                            frac,
                        );
                    }
                }
            }
        }
        start = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(system: CellSystem, stat: GridStat, passes: u32) -> GridSpec {
        GridSpec {
            value_col: 0,
            value_name: "v".into(),
            system,
            stat,
            kernel: Kernel::Box,
            output: GridOutput::Cells,
            smooth_passes: passes,
            post: PostOp::None,
        }
    }

    /// Square grid from a row-major picture, row 0 at the top (so iy
    /// decreases as rows advance, matching a printed map).
    fn surface(rows: &[&[f64]]) -> HashMap<(i64, i64), (f64, f64)> {
        let h = rows.len() as i64;
        let mut out = HashMap::new();
        for (r, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                out.insert((c as i64, h - 1 - r as i64), (v, 1.0));
            }
        }
        out
    }

    #[test]
    fn apportionment_weights_always_sum_to_one() {
        use geo_types::{Coord, LineString, Polygon};
        // A self-intersecting bowtie: unsigned_area cancels the two
        // lobes against each other, so normalizing by it would push the
        // total weight far above 1 and inflate sum/count/density for
        // one bad feature. Digitized parcel data is full of these.
        let bowtie = geo_types::Geometry::Polygon(Polygon::new(
            LineString(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 300.0, y: 300.0 },
                Coord { x: 0.0, y: 300.0 },
                Coord { x: 300.0, y: 0.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));
        let mut w = 0.0;
        apportion_square(&bowtie, 100.0, 100.0, &mut |_, x| w += x);
        assert!((w - 1.0).abs() < 1e-9, "bowtie weights sum to {w}");

        // A thin annulus: no 8×8 sample lands in the 1-unit band and the
        // bbox centre sits in the hole, so the coarse pass must refine
        // rather than assign the whole value to a cell the ring never
        // touches.
        let ring = |r: f64| {
            LineString(
                (0..=64)
                    .map(|i| {
                        let a = i as f64 / 64.0 * std::f64::consts::TAU;
                        Coord {
                            x: 500.0 + r * a.cos(),
                            y: 500.0 + r * a.sin(),
                        }
                    })
                    .collect(),
            )
        };
        let annulus =
            geo_types::Geometry::Polygon(Polygon::new(ring(100.0), vec![ring(99.0)]));
        let mut hits: Vec<((i64, i64), f64)> = Vec::new();
        // Cell 100 units wide: the hole spans cells (4,4)..(5,5).
        let to_cell = |x: f64, y: f64| -> Option<u64> {
            Some((((x / 100.0) as u64) << 32) | ((y / 100.0) as u64))
        };
        let mut total = 0.0;
        apportion_sampled(&annulus, &to_cell, &mut |c, wt| {
            hits.push((((c >> 32) as i64, (c & 0xffff_ffff) as i64), wt));
            total += wt;
        });
        assert!((total - 1.0).abs() < 1e-9, "annulus weights sum to {total}");
        assert!(
            hits.iter().any(|((ix, iy), _)| (*ix, *iy) != (5, 5)),
            "the band must reach cells beyond the one holding the centre"
        );

        // Samples the cell lookup rejects must not shrink the total.
        let square = geo_types::Geometry::Polygon(Polygon::new(
            LineString(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 400.0, y: 0.0 },
                Coord { x: 400.0, y: 400.0 },
                Coord { x: 0.0, y: 400.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));
        let picky = |x: f64, _y: f64| -> Option<u64> {
            // Half the samples fail to project, as they would near a pole.
            (x > 200.0).then_some(7)
        };
        let mut total = 0.0;
        apportion_sampled(&square, &picky, &mut |_, wt| total += wt);
        assert!((total - 1.0).abs() < 1e-9, "projection failures cost {total}");
    }

    #[test]
    fn dateline_cells_stay_small() {
        // An H3-style ring straddling ±180 comes back with vertices on
        // both sides; left as-is it wraps the globe.
        let mut ring = vec![
            (179.6, 51.0),
            (-179.7, 51.2),
            (-179.5, 51.6),
            (179.8, 51.7),
            (179.5, 51.3),
        ];
        unwrap_dateline(&mut ring);
        let (lo, hi) = ring.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &(x, _)| {
            (lo.min(x), hi.max(x))
        });
        assert!(hi - lo < 2.0, "unwrapped span {} deg: {ring:?}", hi - lo);
        // Rings away from the dateline are left exactly as they were.
        let mut plain = vec![(2.0, 48.0), (2.4, 48.1), (2.2, 48.5)];
        let before = plain.clone();
        unwrap_dateline(&mut plain);
        assert_eq!(plain, before);
    }

    #[test]
    fn focal_std_measures_spread_not_level() {
        // A uniform block has no local spread whatever its level; a
        // cell straddling a step does.
        let flat = surface(&[&[7.0, 7.0, 7.0], &[7.0, 7.0, 7.0], &[7.0, 7.0, 7.0]]);
        let out = focal_pass(&flat, square_neighbors, Focal::Std);
        assert!(out.values().all(|&(v, _)| v < 1e-12), "{out:?}");

        let step = surface(&[&[0.0, 0.0, 10.0], &[0.0, 0.0, 10.0], &[0.0, 0.0, 10.0]]);
        let out = focal_pass(&step, square_neighbors, Focal::Std);
        // Interior-left cell sees only zeros; the middle column straddles.
        assert!(out[&(0, 1)].0 < 1e-12);
        assert!(out[&(1, 1)].0 > 4.0, "{:?}", out[&(1, 1)]);
        // Population std of {0,0,0,0,0,0,10,10,10}: mean 10/3.
        let expect = (100.0 * 3.0 / 9.0f64 - (10.0 / 3.0f64).powi(2)).sqrt();
        assert!((out[&(1, 1)].0 - expect).abs() < 1e-9, "{:?}", out[&(1, 1)]);
        // Counts ride along untouched.
        assert_eq!(out[&(1, 1)].1, 1.0);
    }

    #[test]
    fn morphology_drops_spikes_and_fills_holes() {
        // One hot cell in a cold field: open erases it, close leaves it.
        let spike = surface(&[
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 9.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
        ]);
        let opened = focal_pass(
            &focal_pass(&spike, square_neighbors, Focal::Min),
            square_neighbors,
            Focal::Max,
        );
        assert!(opened.values().all(|&(v, _)| v == 0.0), "{opened:?}");

        // One cold cell in a hot field: close fills it, and the plateau
        // around it does not move.
        let hole = surface(&[
            &[5.0, 5.0, 5.0, 5.0],
            &[5.0, 0.0, 5.0, 5.0],
            &[5.0, 5.0, 5.0, 5.0],
        ]);
        let closed = focal_pass(
            &focal_pass(&hole, square_neighbors, Focal::Max),
            square_neighbors,
            Focal::Min,
        );
        assert!(closed.values().all(|&(v, _)| v == 5.0), "{closed:?}");

        // Open must not shave a genuine plateau down. "Genuine" is
        // relative to the 3×3 structuring element: a blob narrower than
        // it is noise by definition and does get removed, so the plateau
        // here is 3×3 and its centre has to survive.
        let plateau = surface(&[
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 9.0, 9.0, 9.0, 0.0],
            &[0.0, 9.0, 9.0, 9.0, 0.0],
            &[0.0, 9.0, 9.0, 9.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        let opened = focal_pass(
            &focal_pass(&plateau, square_neighbors, Focal::Min),
            square_neighbors,
            Focal::Max,
        );
        assert_eq!(opened[&(2, 2)].0, 9.0, "plateau interior survives");
    }

    #[test]
    fn hillshade_lights_the_slope_facing_the_sun() {
        // A ridge running north-south: values rise to the crest in the
        // middle column, so the west flank faces west and the east
        // flank faces east.
        let ridge = surface(&[
            &[0.0, 5.0, 10.0, 5.0, 0.0],
            &[0.0, 5.0, 10.0, 5.0, 0.0],
            &[0.0, 5.0, 10.0, 5.0, 0.0],
        ]);
        let west_sun = hillshade(&ridge, 100.0, 100.0, 270.0, 45.0);
        let (west_flank, east_flank) = (west_sun[&(1, 1)].0, west_sun[&(3, 1)].0);
        assert!(
            west_flank > east_flank + 20.0,
            "west sun must favour the west flank: {west_flank} vs {east_flank}"
        );
        // Move the sun east and the lighting swaps.
        let east_sun = hillshade(&ridge, 100.0, 100.0, 90.0, 45.0);
        assert!(east_sun[&(3, 1)].0 > east_sun[&(1, 1)].0 + 20.0);

        // Flat ground everywhere takes the sun's zenith cosine.
        let flat = surface(&[&[4.0, 4.0, 4.0], &[4.0, 4.0, 4.0], &[4.0, 4.0, 4.0]]);
        let shaded = hillshade(&flat, 100.0, 100.0, 315.0, 45.0);
        let expect = (45.0f64).to_radians().cos() * 255.0;
        assert!(shaded.values().all(|&(v, _)| (v - expect).abs() < 1e-6));
        // Values stay inside the 0..255 shading range.
        assert!(west_sun.values().all(|&(v, _)| (0.0..=255.0).contains(&v)));
    }

    #[test]
    fn square_smoothing_averages_present_neighbors_only() {
        // Two adjacent cells (10, 30) and one far away (100): smoothing
        // must mix the neighbors and leave the loner untouched.
        let mut vals: HashMap<(i64, i64), (f64, f64)> = HashMap::new();
        vals.insert((0, 0), (10.0, 1.0));
        vals.insert((1, 0), (30.0, 1.0));
        vals.insert((50, 50), (100.0, 1.0));
        let out = smooth_pass(&vals, |&(ix, iy)| {
            let mut n = Vec::new();
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    if (dx, dy) != (0, 0) {
                        n.push(((ix + dx, iy + dy), (dx, dy)));
                    }
                }
            }
            n
        }, Kernel::Box);
        assert_eq!(out[&(0, 0)].0, 20.0);
        assert_eq!(out[&(1, 0)].0, 20.0);
        assert_eq!(out[&(50, 50)].0, 100.0, "no bleed into isolated cells");
        // Gaussian weights the center higher: closer to the original.
        let g = smooth_pass(&vals, |&(ix, iy)| {
            let mut n = Vec::new();
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    if (dx, dy) != (0, 0) {
                        n.push(((ix + dx, iy + dy), (dx, dy)));
                    }
                }
            }
            n
        }, Kernel::Gaussian);
        assert!((g[&(0, 0)].0 - (10.0 * 4.0 + 30.0 * 2.0) / 6.0).abs() < 1e-9);
    }

    #[test]
    fn contours_trace_a_value_hill() {
        // A 5×5 hill: value = 10 at the center cell, 0 elsewhere, one
        // smoothing-free surface. Levels must produce closed-ish rings
        // around the peak, all inside the grid, tagged with the level.
        let mut vals: HashMap<(i64, i64), (f64, f64)> = HashMap::new();
        for ix in 0..5i64 {
            for iy in 0..5i64 {
                let d = ((ix - 2).abs()).max((iy - 2).abs()) as f64;
                vals.insert((ix, iy), (10.0 - 4.0 * d, 1.0));
            }
        }
        let (batch, _, n) =
            contour_lines(&vals, "v", 100.0, 100.0, 3, ContourSpacing::Equal).unwrap();
        assert!(n >= 3, "one line per crossed level, got {n}");
        assert_eq!(batch.num_rows(), n);
        // Level column values sit strictly inside the value range.
        let lv = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..lv.len() {
            assert!(lv.value(i) > -2.0 && lv.value(i) < 10.0);
        }
        // Degenerate surfaces refuse politely.
        let flat: HashMap<(i64, i64), (f64, f64)> =
            (0..9).map(|i| ((i % 3, i / 3), (5.0, 1.0))).collect();
        assert!(contour_lines(&flat, "v", 100.0, 100.0, 3, ContourSpacing::Equal).is_err());
    }

    #[test]
    fn quantile_levels_follow_the_distribution() {
        // Heavy tail: 100 cells at 1, one cell at 1000. Equal spacing
        // puts every level in the empty gap above the mass; quantile
        // spacing has to keep them where the cells actually are.
        let mut vals: HashMap<(i64, i64), (f64, f64)> = HashMap::new();
        for i in 0..100i64 {
            vals.insert((i % 10, i / 10), ((i % 10) as f64 + 1.0, 1.0));
        }
        vals.insert((5, 5), (1000.0, 1.0));

        let eq = contour_levels(&vals, 4, ContourSpacing::Equal).unwrap();
        let qt = contour_levels(&vals, 4, ContourSpacing::Quantile).unwrap();
        assert!(
            eq.iter().all(|&v| v > 100.0),
            "equal steps land in the tail gap: {eq:?}"
        );
        assert!(
            qt.iter().all(|&v| v <= 11.0),
            "quantile levels stay in the bulk: {qt:?}"
        );
        // Strictly increasing, strictly inside the range, both ways.
        for lv in [&eq, &qt] {
            assert!(lv.windows(2).all(|w| w[1] > w[0]), "{lv:?}");
            assert!(lv.iter().all(|&v| v > 1.0 && v < 1000.0), "{lv:?}");
        }

        // Ties collapse quantiles onto one value: keep one line each,
        // never a stack of coincident ones.
        let tied: HashMap<(i64, i64), (f64, f64)> = (0..100i64)
            .map(|i| {
                let v = if i < 50 {
                    0.0
                } else if i < 80 {
                    5.0
                } else {
                    9.0
                };
                ((i % 10, i / 10), (v, 1.0))
            })
            .collect();
        let lv = contour_levels(&tied, 10, ContourSpacing::Quantile).unwrap();
        assert!(lv.windows(2).all(|w| w[1] > w[0]), "{lv:?}");
        assert!(lv.len() < 10, "duplicates should be dropped: {lv:?}");
    }

    #[test]
    fn stitch_joins_shared_endpoints() {
        let segs = vec![
            ([0.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.0], [2.0, 0.5]),
            ([5.0, 5.0], [6.0, 5.0]), // disjoint
        ];
        let mut lines = stitch_segments(segs, 1e-6);
        lines.sort_by_key(|l| std::cmp::Reverse(l.len()));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), 3, "{lines:?}");
        assert_eq!(lines[1].len(), 2);
    }

    #[test]
    fn acc_stats() {
        let mut a = Acc { sum: 0.0, wsum: 0.0, vals: Some(Vec::new()), cats: None };
        for v in [1.0, 100.0, 3.0] {
            a.push(v, 1.0);
        }
        assert_eq!(a.stat(GridStat::Median), 3.0);
        assert_eq!(a.stat(GridStat::Sum), 104.0);
        assert_eq!(a.stat(GridStat::Count), 3.0);
        assert!((a.stat(GridStat::Mean) - 104.0 / 3.0).abs() < 1e-12);
        // Weighted: a huge value at tiny weight must not dominate the
        // weighted median.
        let mut a = Acc { sum: 0.0, wsum: 0.0, vals: Some(Vec::new()), cats: None };
        a.push(1_000_000.0, 0.01);
        a.push(10.0, 1.0);
        a.push(20.0, 1.0);
        assert_eq!(a.stat(GridStat::Median), 20.0);
        assert!((a.stat(GridStat::Count) - 2.01).abs() < 1e-12);
    }

    /// The reported artifact: a giant parcel's MEAN equals its full
    /// value in every covered cell (weights cancel), lighting a hot
    /// blob; DENSITY spreads it honestly.
    #[test]
    fn density_defuses_giant_polygon_hotspots() {
        let giant_value = 1_000_000.0;
        let mk = |stat: GridStat| {
            let mut a = Acc { sum: 0.0, wsum: 0.0, vals: None, cats: None };
            a.push(giant_value, 0.5); // half the giant covers this cell
            a.stat(stat)
        };
        // Mean: the blob — the cell reads the giant's entire value.
        assert_eq!(mk(GridStat::Mean), giant_value);
        // Density (100 m cells): value apportioned per m² instead.
        let density = mk(GridStat::Density) / (100.0 * 100.0);
        assert!((density - 50.0).abs() < 1e-9);
        // A neighboring cell with a small local parcel on top barely
        // differs: no spike between "giant only" and "giant + normal".
        let mut b = Acc { sum: 0.0, wsum: 0.0, vals: None, cats: None };
        b.push(giant_value, 0.5);
        b.push(100.0, 1.0);
        let density_b = b.stat(GridStat::Density) / (100.0 * 100.0);
        assert!((density_b - density) < 0.02, "{density_b} vs {density}");
    }

    #[test]
    fn apportion_square_splits_by_covered_area() {
        // A 2×2-cell square polygon centered on the cell corner: each
        // of the four cells covers exactly a quarter.
        let poly = geo_types::Geometry::Polygon(geo_types::Polygon::new(
            geo_types::LineString(vec![
                geo_types::Coord { x: -50.0, y: -50.0 },
                geo_types::Coord { x: 50.0, y: -50.0 },
                geo_types::Coord { x: 50.0, y: 50.0 },
                geo_types::Coord { x: -50.0, y: 50.0 },
                geo_types::Coord { x: -50.0, y: -50.0 },
            ]),
            vec![],
        ));
        let mut got: HashMap<(i64, i64), f64> = HashMap::new();
        apportion_square(&poly, 100.0, 100.0, &mut |k, w| {
            *got.entry(k).or_insert(0.0) += w;
        });
        assert_eq!(got.len(), 4, "{got:?}");
        for (&k, &w) in &got {
            assert!((w - 0.25).abs() < 1e-9, "{k:?} got {w}");
        }
        // Uneven split: 80% in one cell, 20% in the neighbor.
        let poly = geo_types::Geometry::Polygon(geo_types::Polygon::new(
            geo_types::LineString(vec![
                geo_types::Coord { x: 20.0, y: 10.0 },
                geo_types::Coord { x: 120.0, y: 10.0 },
                geo_types::Coord { x: 120.0, y: 90.0 },
                geo_types::Coord { x: 20.0, y: 90.0 },
                geo_types::Coord { x: 20.0, y: 10.0 },
            ]),
            vec![],
        ));
        let mut got: HashMap<(i64, i64), f64> = HashMap::new();
        apportion_square(&poly, 100.0, 100.0, &mut |k, w| {
            *got.entry(k).or_insert(0.0) += w;
        });
        assert!((got[&(0, 0)] - 0.8).abs() < 1e-9, "{got:?}");
        assert!((got[&(1, 0)] - 0.2).abs() < 1e-9, "{got:?}");
        // A hole cut from the covered part shifts the ratio accordingly:
        // 100x80 rect minus a 40x40 hole fully in cell 0 -> areas
        // 8000-1600=6400 total, cell0 6400-1600=4800.
        let poly = geo_types::Geometry::Polygon(geo_types::Polygon::new(
            geo_types::LineString(vec![
                geo_types::Coord { x: 20.0, y: 10.0 },
                geo_types::Coord { x: 120.0, y: 10.0 },
                geo_types::Coord { x: 120.0, y: 90.0 },
                geo_types::Coord { x: 20.0, y: 90.0 },
                geo_types::Coord { x: 20.0, y: 10.0 },
            ]),
            vec![geo_types::LineString(vec![
                geo_types::Coord { x: 30.0, y: 20.0 },
                geo_types::Coord { x: 70.0, y: 20.0 },
                geo_types::Coord { x: 70.0, y: 60.0 },
                geo_types::Coord { x: 30.0, y: 60.0 },
                geo_types::Coord { x: 30.0, y: 20.0 },
            ])],
        ));
        let mut got: HashMap<(i64, i64), f64> = HashMap::new();
        apportion_square(&poly, 100.0, 100.0, &mut |k, w| {
            *got.entry(k).or_insert(0.0) += w;
        });
        assert!((got[&(0, 0)] - 4800.0 / 6400.0).abs() < 1e-9, "{got:?}");
        assert!((got[&(1, 0)] - 1600.0 / 6400.0).abs() < 1e-9, "{got:?}");
    }

    #[test]
    /// A text column grids by majority/minority — the Cambridge trees
    /// question: which species dominates each cell. Points count 1
    /// apiece, blank values drop out, ties break the same way every
    /// run, and the output column is the text itself.
    fn a_text_column_grids_by_majority_and_minority() {
        use arrow::array::StringArray;
        let dir = std::env::temp_dir().join(format!("geopq_grid_text_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("trees.parquet");

        let mut wkb = BinaryBuilder::new();
        let mut species = StringBuilder::new();
        let trees: &[(f64, f64, Option<&str>)] = &[
            // Cell (0,0): two maples, one birch.
            (10.0, 10.0, Some("Acer")),
            (20.0, 20.0, Some("Acer")),
            (30.0, 30.0, Some("Betula")),
            // Cell (1,0): three birches, two dogwoods.
            (150.0, 50.0, Some("Betula")),
            (160.0, 60.0, Some(" Betula ")), // portals pad values
            (170.0, 70.0, Some("Cornus")),
            (180.0, 80.0, Some("Cornus")),
            (190.0, 90.0, Some("Betula")),
            // No species recorded: not a category, not a count.
            (40.0, 40.0, None),
        ];
        for &(x, y, s) in trees {
            let p = geo_types::Geometry::Point(geo_types::Point::new(x, y));
            wkb.append_value(crate::data::import::to_wkb(&p).unwrap());
            match s {
                Some(s) => species.append_value(s),
                None => species.append_null(),
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("SPECIES", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(wkb.finish()) as ArrayRef,
                Arc::new(species.finish()),
            ],
        )
        .unwrap();
        crate::sql::export::write_result(&src, &schema, &[batch], 0, &Crs::wgs84())
            .unwrap();
        let (store, crs, _, _) = crate::data::loader::open_store_for_test(&src).unwrap();
        let col = store.schema.index_of("SPECIES").unwrap();

        let run = |stat: GridStat| -> HashMap<String, String> {
            let spec = GridSpec {
                value_col: col,
                value_name: "SPECIES".into(),
                system: CellSystem::square(100.0),
                stat,
                kernel: Kernel::Box,
                output: GridOutput::Cells,
                smooth_passes: 0,
                post: PostOp::None,
            };
            let dst = dir.join(format!("grid_{}.parquet", stat.label()));
            let (cells, rows) = compute(&store, &crs, &spec, &dst, &|_| {}).unwrap();
            assert_eq!(rows, 8, "the species-less tree is skipped");
            assert_eq!(cells, 2);
            let (gs, _, _, _) = crate::data::loader::open_store_for_test(&dst).unwrap();
            let mut out = HashMap::new();
            for b in &gs.fetch(&[0, 1], None).unwrap() {
                let sc = b.schema();
                let sp = b
                    .column(sc.index_of("SPECIES").unwrap())
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("the value column is text")
                    .clone();
                let cell = b
                    .column(sc.index_of("cell").unwrap())
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .clone();
                for i in 0..b.num_rows() {
                    out.insert(cell.value(i).to_string(), sp.value(i).to_string());
                }
            }
            out
        };
        let maj = run(GridStat::Majority);
        assert_eq!(maj["0:0"], "Acer");
        assert_eq!(maj["1:0"], "Betula", "padded values counted with the clean ones");
        let min = run(GridStat::Minority);
        assert_eq!(min["0:0"], "Betula");
        assert_eq!(min["1:0"], "Cornus");

        // The numeric-surface operations refuse a category rather than
        // averaging dictionary codes into nonsense.
        let smoothed = GridSpec {
            value_col: col,
            value_name: "SPECIES".into(),
            system: CellSystem::square(100.0),
            stat: GridStat::Majority,
            kernel: Kernel::Box,
            output: GridOutput::Cells,
            smooth_passes: 1,
            post: PostOp::None,
        };
        let err = compute(&store, &crs, &smoothed, &dir.join("x.parquet"), &|_| {})
            .unwrap_err();
        assert!(err.contains("smooth"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    /// A layer in degrees takes its cell size in meters: converted at
    /// the data's mid-latitude, wider in longitude by 1/cos(lat) so the
    /// cell is square on the ground, not on the graticule.
    fn a_degree_layer_takes_its_cell_size_in_meters() {
        // Cambridge MA: 1000 m at 42.37°N.
        let (sx, sy) = square_cell_degrees(1000.0, 42.37);
        assert!((sy - 0.0089932).abs() < 1e-5, "{sy}");
        assert!(sx > sy, "longitude degrees are shorter, cells span more of them");
        assert!((sx * (42.37_f64.to_radians().cos()) - sy).abs() < 1e-12);
        // On the equator the two agree.
        let (ex, ey) = square_cell_degrees(1000.0, 0.0);
        assert!((ex - ey).abs() < 1e-15);
    }

    #[test]
    fn cell_systems_produce_valid_cells() {
        // Boston-ish lon/lat through both geographic systems.
        let (lon, lat) = (-71.06, 42.36);
        let h3 = h3o::LatLng::new(lat, lon).unwrap().to_cell(h3o::Resolution::Seven);
        assert!(h3.boundary().len() >= 5);
        let a5c = a5::lonlat_to_cell(a5::LonLat::new(lon, lat), 12).unwrap();
        let b = a5::cell_to_boundary(a5c, None).unwrap();
        assert!(b.len() >= 5, "pentagon boundary, got {}", b.len());
        let _ = spec(CellSystem::A5 { res: 12 }, GridStat::Mean, 0);
    }
}

#[cfg(test)]
mod real_file_tests {
    use super::*;

    /// Opt-in: full pipeline against the local MassGIS parcels file.
    #[test]
    #[ignore = "needs ~/Downloads/Statewide_parcels_SHP/mass_parcels.parquet"]
    fn mass_parcels_grids_end_to_end() {
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Downloads/Statewide_parcels_SHP/mass_parcels.parquet");
        if !path.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let (store, crs, _, _) =
            crate::data::loader::open_store_for_test(&path).unwrap();
        let col = store.schema.index_of("LAND_VAL").unwrap();
        for (system, label) in [
            (CellSystem::square(1000.0), "square 1km"),
            (CellSystem::H3 { res: 7 }, "h3 r7"),
            (CellSystem::A5 { res: 14 }, "a5 r14"),
        ] {
            let spec = GridSpec {
                value_col: col,
                value_name: "LAND_VAL".into(),
                system,
                stat: GridStat::Mean,
                kernel: Kernel::Gaussian,
                output: GridOutput::Cells,
                smooth_passes: 1,
                post: PostOp::None,
            };
            let dst = std::env::temp_dir().join(format!(
                "geopq_grid_e2e_{}.parquet",
                label.replace(' ', "_")
            ));
            let t = std::time::Instant::now();
            let (cells, rows) = compute(&store, &crs, &spec, &dst, &|_| {}).unwrap();
            eprintln!("{label}: {cells} cells from {rows} rows in {:?}", t.elapsed());
            assert!(rows > 2_000_000, "{label}: most parcels aggregated");
            assert!(cells > 500, "{label}: {cells} cells");
            // Reload through the real loader: valid GeoParquet, sane CRS.
            let (gs, gcrs, _, _) = crate::data::loader::open_store_for_test(&dst).unwrap();
            assert_eq!(gs.total_rows() as usize, cells);
            match system {
                CellSystem::Square { .. } => assert!(!gcrs.is_latlong),
                _ => assert!(gcrs.is_latlong),
            }
            let _ = std::fs::remove_file(&dst);
        }
        // Contours over the smoothed 1 km surface.
        let spec = GridSpec {
            value_col: col,
            value_name: "LAND_VAL".into(),
            system: CellSystem::square(1000.0),
            stat: GridStat::Mean,
            kernel: Kernel::Gaussian,
            output: GridOutput::Contours {
                levels: 8,
                spacing: ContourSpacing::Quantile,
            },
            smooth_passes: 2,
            post: PostOp::None,
        };
        let dst = std::env::temp_dir().join("geopq_grid_e2e_contours.parquet");
        let t = std::time::Instant::now();
        let (lines, rows) = compute(&store, &crs, &spec, &dst, &|_| {}).unwrap();
        eprintln!("contours: {lines} lines from {rows} rows in {:?}", t.elapsed());
        assert!(lines > 8, "several stitched isolines, got {lines}");
        let (gs, gcrs, _, _) = crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(gs.total_rows() as usize, lines);
        assert!(!gcrs.is_latlong);
        let g = gs.fetch_geoms(&[0]).unwrap();
        assert!(matches!(
            g[0].1.as_ref().unwrap(),
            geo_types::Geometry::LineString(_)
        ));
        let _ = std::fs::remove_file(&dst);

        // Hillshade of the same surface: shading range, suffixed column.
        let spec = GridSpec {
            value_col: col,
            value_name: "LAND_VAL".into(),
            system: CellSystem::square(1000.0),
            stat: GridStat::Mean,
            kernel: Kernel::Gaussian,
            output: GridOutput::Cells,
            smooth_passes: 2,
            post: PostOp::Hillshade {
                azimuth_deg: 315.0,
                altitude_deg: 45.0,
            },
        };
        let dst = std::env::temp_dir().join("geopq_grid_e2e_shade.parquet");
        let t = std::time::Instant::now();
        let (cells, _) = compute(&store, &crs, &spec, &dst, &|_| {}).unwrap();
        eprintln!("hillshade: {cells} cells in {:?}", t.elapsed());
        let (gs, _, _, _) = crate::data::loader::open_store_for_test(&dst).unwrap();
        assert!(gs.schema.index_of("LAND_VAL_shade").is_ok());
        let _ = std::fs::remove_file(&dst);
    }
}
