//! Grid summary: aggregate a numeric column into cells — square grid in
//! data-CRS units, H3, or A5 — with optional kernel smoothing, and
//! materialize the result as a GeoParquet layer (which then inherits
//! styling, classification, export and publish like any other layer).
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
    /// Square cells of `size` data-CRS units (meters for projected CRS).
    Square { size: f64 },
    /// H3 hexagons (geographic; centroids are inverse-projected).
    H3 { res: u8 },
    /// A5 pentagons (geographic, equal-area).
    A5 { res: i32 },
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
}

impl GridStat {
    pub fn label(&self) -> &'static str {
        match self {
            GridStat::Mean => "mean",
            GridStat::Median => "median",
            GridStat::Sum => "sum",
            GridStat::Count => "count",
            GridStat::Density => "density",
        }
    }
}

/// 3×3 kernel for square grids; H3/A5 always smooth with the ring mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Box,
    Gaussian,
}

/// What the grid materializes as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridOutput {
    /// One polygon per cell.
    Cells,
    /// Isolines from the (smoothed) cell values: `levels` equally
    /// spaced levels between the value min and max. Square grids only
    /// (marching squares needs the regular lattice).
    Contours { levels: usize },
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
}

impl Acc {
    fn push(&mut self, v: f64, w: f64) {
        self.sum += w * v;
        self.wsum += w;
        if let Some(vals) = &mut self.vals {
            vals.push((v, w));
        }
    }

    fn stat(&mut self, stat: GridStat) -> f64 {
        match stat {
            GridStat::Mean => self.sum / self.wsum.max(f64::EPSILON),
            GridStat::Sum | GridStat::Density => self.sum,
            GridStat::Count => self.wsum,
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

    // ---- pass 1: bbox + value per row -------------------------------
    // Features whose bbox fits in one cell assign wholly (weight 1) with
    // no geometry access; the rest are stashed for area apportionment.
    let keep_vals = spec.stat == GridStat::Median;
    let blank = || Acc {
        sum: 0.0,
        wsum: 0.0,
        vals: keep_vals.then(Vec::new),
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
    scan_bbox_values(store, spec.value_col, &mut |row, b, v, frac| {
        progress(frac * 0.55);
        match spec.system {
            CellSystem::Square { size } => {
                let (ix0, iy0) = ((b[0] / size).floor() as i64, (b[1] / size).floor() as i64);
                let (ix1, iy1) = ((b[2] / size).floor() as i64, (b[3] / size).floor() as i64);
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
            CellSystem::Square { size } => {
                let parts: Result<Vec<Vec<((i64, i64), f64, f64)>>, String> = chunks
                    .par_iter()
                    .map(|chunk| {
                        let rows: Vec<u32> = chunk.iter().map(|(r, _)| *r).collect();
                        let geoms = store.fetch_geoms(&rows)?;
                        let mut out = Vec::new();
                        for ((_, g), &(_, v)) in geoms.iter().zip(chunk.iter()) {
                            let Some(g) = g else { continue };
                            apportion_square(g, size, &mut |k, w| out.push((k, v, w)));
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
            CellSystem::Square { size } => {
                for v in sq_vals.values_mut() {
                    v.0 /= size * size;
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
        progress(0.7 + 0.15 * (pass as f32 + 1.0) / spec.smooth_passes.max(1) as f32);
        match spec.system {
            CellSystem::Square { .. } => {
                sq_vals = smooth_pass(&sq_vals, |&(ix, iy)| {
                    let mut out = Vec::with_capacity(8);
                    for dx in -1..=1i64 {
                        for dy in -1..=1i64 {
                            if (dx, dy) != (0, 0) {
                                out.push(((ix + dx, iy + dy), (dx, dy)));
                            }
                        }
                    }
                    out
                }, spec.kernel);
            }
            CellSystem::H3 { .. } => {
                geo_vals = smooth_pass(&geo_vals, |&c| {
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
                }, Kernel::Box);
            }
            CellSystem::A5 { .. } => {
                geo_vals = smooth_pass(&geo_vals, |&c| {
                    a5::grid_disk(c, 1)
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|n| *n != c)
                        .map(|n| (n, (1i64, 0i64)))
                        .collect()
                }, Kernel::Box);
            }
        }
    }

    // ---- materialize --------------------------------------------------
    progress(0.9);
    let out_crs = if geographic { &wgs84 } else { crs };
    if let GridOutput::Contours { levels } = spec.output {
        let CellSystem::Square { size } = spec.system else {
            return Err("contour lines need the square grid (marching squares)".into());
        };
        let (batch, schema, lines) =
            contour_lines(&sq_vals, &spec.value_name, size, levels)?;
        crate::sql::export::write_result(dst, &schema, std::slice::from_ref(&batch), 0, out_crs)?;
        progress(1.0);
        return Ok((lines, rows_used));
    }
    let (batch, schema, cells) = match spec.system {
        CellSystem::Square { size } => {
            materialize(&sq_vals, &spec.value_name, |&(ix, iy)| {
                let (x0, y0) = (ix as f64 * size, iy as f64 * size);
                (
                    format!("{ix}:{iy}"),
                    vec![
                        (x0, y0),
                        (x0 + size, y0),
                        (x0 + size, y0 + size),
                        (x0, y0 + size),
                        (x0, y0),
                    ],
                )
            })?
        }
        CellSystem::H3 { .. } => materialize(&geo_vals, &spec.value_name, |&c| {
            let cell = h3o::CellIndex::try_from(c).expect("aggregated cell is valid");
            let mut ring: Vec<(f64, f64)> = cell
                .boundary()
                .iter()
                .map(|ll| (ll.lng(), ll.lat()))
                .collect();
            if let Some(&first) = ring.first() {
                ring.push(first);
            }
            (cell.to_string(), ring)
        })?,
        CellSystem::A5 { .. } => materialize(&geo_vals, &spec.value_name, |&c| {
            let b = a5::cell_to_boundary(c, None).unwrap_or_default();
            let mut ring: Vec<(f64, f64)> =
                b.iter().map(|ll| (ll.longitude(), ll.latitude())).collect();
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

/// Marching squares over the cell-center lattice: isolines at `levels`
/// equally spaced values, linearly interpolated, stitched into
/// polylines. Cells missing from the grid stop the contours (no data
/// invented past the edge).
#[allow(clippy::type_complexity)]
fn contour_lines(
    vals: &HashMap<(i64, i64), (f64, f64)>,
    value_name: &str,
    size: f64,
    levels: usize,
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
    let level_values: Vec<f64> = (1..=levels.clamp(1, 64))
        .map(|k| vmin + (vmax - vmin) * k as f64 / (levels.clamp(1, 64) + 1) as f64)
        .collect();

    // Cell-center position of grid node (i, j), data CRS.
    let pos = |i: usize, j: usize| -> [f64; 2] {
        [
            (min_ix + i as i64) as f64 * size + size * 0.5,
            (min_iy + j as i64) as f64 * size + size * 0.5,
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
        for line in stitch_segments(segments, size * 1e-6) {
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

/// Cells → one arrow batch: WKB polygon, value, count, cell id.
#[allow(clippy::type_complexity)]
fn materialize<K: Hash + Eq + Copy>(
    vals: &HashMap<K, (f64, f64)>,
    value_name: &str,
    cell_geom: impl Fn(&K) -> (String, Vec<(f64, f64)>),
) -> Result<(RecordBatch, SchemaRef, usize), String> {
    let mut wkb = BinaryBuilder::new();
    let mut value: Vec<f64> = Vec::with_capacity(vals.len());
    let mut count: Vec<f64> = Vec::with_capacity(vals.len());
    let mut ids = StringBuilder::new();
    for (k, &(v, c)) in vals {
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
        value.push(v);
        count.push(c);
        ids.append_value(id);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, false),
        Field::new(value_name, DataType::Float64, false),
        Field::new("count", DataType::Float64, false),
        Field::new("cell", DataType::Utf8, false),
    ]));
    let n = value.len();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(wkb.finish()) as ArrayRef,
            Arc::new(Float64Array::from(value)),
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
    size: f64,
    add: &mut dyn FnMut((i64, i64), f64),
) {
    use geo::{Area, BoundingRect};
    let total = g.unsigned_area();
    let centroid_fallback = |add: &mut dyn FnMut((i64, i64), f64)| {
        if let Some(r) = g.bounding_rect() {
            let (cx, cy) = ((r.min().x + r.max().x) * 0.5, (r.min().y + r.max().y) * 0.5);
            add(((cx / size).floor() as i64, (cy / size).floor() as i64), 1.0);
        }
    };
    if total <= 0.0 {
        centroid_fallback(add);
        return;
    }
    let mut covered: HashMap<(i64, i64), f64> = HashMap::new();
    let mut per_poly = |poly: &geo_types::Polygon<f64>| {
        let Some(r) = poly.bounding_rect() else { return };
        let (ix0, ix1) = ((r.min().x / size).floor() as i64, (r.max().x / size).floor() as i64);
        let (iy0, iy1) = ((r.min().y / size).floor() as i64, (r.max().y / size).floor() as i64);
        for ix in ix0..=ix1 {
            for iy in iy0..=iy1 {
                let rect = [
                    ix as f64 * size,
                    iy as f64 * size,
                    (ix + 1) as f64 * size,
                    (iy + 1) as f64 * size,
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
    for (k, a) in covered {
        add(k, (a / total).min(1.0));
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
    const N: usize = 8;
    let (dx, dy) = (
        (r.max().x - r.min().x) / N as f64,
        (r.max().y - r.min().y) / N as f64,
    );
    if dx <= 0.0 || dy <= 0.0 {
        centroid_fallback(add);
        return;
    }
    let mut per_cell: HashMap<u64, u32> = HashMap::new();
    let mut inside = 0u32;
    let mut xs: Vec<f64> = Vec::new();
    for j in 0..N {
        let y = r.min().y + (j as f64 + 0.5) * dy;
        xs.clear();
        for ring in &rings {
            let n = ring.len();
            if n < 3 {
                continue;
            }
            // Rings from WKB carry the duplicate closing point; guard
            // with an explicit wrap segment in case one doesn't.
            let closed = ring[0] == ring[n - 1];
            let segs = if closed { n - 1 } else { n };
            for i in 0..segs {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
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
        for i in 0..N {
            let x = r.min().x + (i as f64 + 0.5) * dx;
            // Odd number of crossings to the left ⇒ inside (even-odd).
            if xs.partition_point(|&c| c < x) % 2 == 1 {
                inside += 1;
                if let Some(c) = to_cell(x, y) {
                    *per_cell.entry(c).or_insert(0) += 1;
                }
            }
        }
    }
    if inside == 0 || per_cell.is_empty() {
        centroid_fallback(add);
        return;
    }
    for (c, n) in per_cell {
        add(c, n as f64 / inside as f64);
    }
}

/// Stream (row, data-CRS bbox, value, progress) for every row with a
/// usable extent and finite value. Uses the covering bbox column when
/// present (columnar, no geometry decode); falls back to decoding
/// geometries otherwise.
fn scan_bbox_values(
    store: &FeatureStore,
    value_col: usize,
    f: &mut dyn FnMut(u32, [f64; 4], f64, f32),
) -> Result<(), String> {
    const CHUNK: u64 = 131_072;
    let total = store.total_rows();
    if total == 0 {
        return Err("layer has no rows".into());
    }
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
                    let vals = arrow::compute::cast(b.column(val_pos), &DataType::Float64)
                        .map_err(|e| format!("value cast: {e}"))?;
                    let vals = vals.as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..b.num_rows() {
                        if vals.is_null(i) || xmin.is_null(i) {
                            continue;
                        }
                        let v = vals.value(i);
                        if !v.is_finite() {
                            continue;
                        }
                        f(
                            rows[i + row_base],
                            [xmin.value(i), ymin.value(i), xmax.value(i), ymax.value(i)],
                            v,
                            frac,
                        );
                    }
                    row_base += b.num_rows();
                }
            }
            None => {
                // Geometry decode fallback (raw files without covering).
                let batches = store.fetch(&rows, Some(&[value_col]))?;
                let mut vals_all: Vec<Option<f64>> = Vec::with_capacity(rows.len());
                for b in &batches {
                    let vals = arrow::compute::cast(b.column(0), &DataType::Float64)
                        .map_err(|e| format!("value cast: {e}"))?;
                    let vals = vals.as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..b.num_rows() {
                        vals_all.push((!vals.is_null(i)).then(|| vals.value(i)));
                    }
                }
                let geoms = store.fetch_geoms(&rows)?;
                for (k, ((_, g), v)) in geoms.iter().zip(&vals_all).enumerate() {
                    let (Some(g), Some(v)) = (g, v) else { continue };
                    if !v.is_finite() {
                        continue;
                    }
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
        }
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
        let (batch, _, n) = contour_lines(&vals, "v", 100.0, 3).unwrap();
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
        assert!(contour_lines(&flat, "v", 100.0, 3).is_err());
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
        let mut a = Acc { sum: 0.0, wsum: 0.0, vals: Some(Vec::new()) };
        for v in [1.0, 100.0, 3.0] {
            a.push(v, 1.0);
        }
        assert_eq!(a.stat(GridStat::Median), 3.0);
        assert_eq!(a.stat(GridStat::Sum), 104.0);
        assert_eq!(a.stat(GridStat::Count), 3.0);
        assert!((a.stat(GridStat::Mean) - 104.0 / 3.0).abs() < 1e-12);
        // Weighted: a huge value at tiny weight must not dominate the
        // weighted median.
        let mut a = Acc { sum: 0.0, wsum: 0.0, vals: Some(Vec::new()) };
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
            let mut a = Acc { sum: 0.0, wsum: 0.0, vals: None };
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
        let mut b = Acc { sum: 0.0, wsum: 0.0, vals: None };
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
        apportion_square(&poly, 100.0, &mut |k, w| {
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
        apportion_square(&poly, 100.0, &mut |k, w| {
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
        apportion_square(&poly, 100.0, &mut |k, w| {
            *got.entry(k).or_insert(0.0) += w;
        });
        assert!((got[&(0, 0)] - 4800.0 / 6400.0).abs() < 1e-9, "{got:?}");
        assert!((got[&(1, 0)] - 1600.0 / 6400.0).abs() < 1e-9, "{got:?}");
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
            (CellSystem::Square { size: 1000.0 }, "square 1km"),
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
            system: CellSystem::Square { size: 1000.0 },
            stat: GridStat::Mean,
            kernel: Kernel::Gaussian,
            output: GridOutput::Contours { levels: 8 },
            smooth_passes: 2,
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
    }
}
