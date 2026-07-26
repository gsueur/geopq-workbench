//! Grid summary: aggregate a numeric column into cells — square grid in
//! data-CRS units, H3, or A5 — with optional kernel smoothing, and
//! materialize the result as a GeoParquet layer (which then inherits
//! styling, classification, export and publish like any other layer).
//!
//! Cell assignment is by feature centroid (bbox center): standard
//! carroyage practice, and a pure columnar scan when the file carries a
//! covering bbox column (no geometry decode at all).

use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryBuilder, Float64Array, Int64Array, StringBuilder};
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
}

impl GridStat {
    pub const ALL: &[GridStat] = &[GridStat::Mean, GridStat::Median, GridStat::Sum, GridStat::Count];

    pub fn label(&self) -> &'static str {
        match self {
            GridStat::Mean => "mean",
            GridStat::Median => "median",
            GridStat::Sum => "sum",
            GridStat::Count => "count",
        }
    }
}

/// 3×3 kernel for square grids; H3/A5 always smooth with the ring mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Box,
    Gaussian,
}

pub struct GridSpec {
    /// Store-schema index of the numeric value column.
    pub value_col: usize,
    pub system: CellSystem,
    pub stat: GridStat,
    pub kernel: Kernel,
    /// Smoothing passes over the aggregated cells (0 = raw). Each pass
    /// averages a cell with its present neighbors only: empty cells stay
    /// empty and contribute nothing (no bleed past the data's edge).
    pub smooth_passes: u32,
}

struct Acc {
    sum: f64,
    count: u64,
    /// Individual values, kept only for the median.
    vals: Option<Vec<f64>>,
}

impl Acc {
    fn push(&mut self, v: f64) {
        self.sum += v;
        self.count += 1;
        if let Some(vals) = &mut self.vals {
            vals.push(v);
        }
    }

    fn stat(&mut self, stat: GridStat) -> f64 {
        match stat {
            GridStat::Mean => self.sum / self.count.max(1) as f64,
            GridStat::Sum => self.sum,
            GridStat::Count => self.count as f64,
            GridStat::Median => {
                let vals = self.vals.as_mut().expect("median keeps values");
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let n = vals.len();
                if n == 0 {
                    0.0
                } else if n % 2 == 1 {
                    vals[n / 2]
                } else {
                    (vals[n / 2 - 1] + vals[n / 2]) * 0.5
                }
            }
        }
    }
}

/// Aggregate `store`'s `spec.value_col` into cells and write the grid as
/// GeoParquet at `dst`. Returns (cells, rows aggregated).
pub fn compute(
    store: &FeatureStore,
    crs: &Crs,
    spec: &GridSpec,
    dst: &Path,
    progress: &(dyn Fn(f32) + Sync),
) -> Result<(usize, u64), String> {
    let wgs84 = Crs::wgs84();
    let geographic = !matches!(spec.system, CellSystem::Square { .. });

    // ---- scan: centroid + value per row, keyed into cells ------------
    let keep_vals = spec.stat == GridStat::Median;
    let mut sq: HashMap<(i64, i64), Acc> = HashMap::new();
    let mut geo: HashMap<u64, Acc> = HashMap::new();
    let mut rows_used = 0u64;
    scan_centroid_values(store, spec.value_col, &mut |cx, cy, v, frac| {
        progress(frac * 0.7);
        let (cx, cy) = if geographic {
            match transform_point(crs, &wgs84, cx, cy) {
                Ok(p) => p,
                Err(_) => return,
            }
        } else {
            (cx, cy)
        };
        let blank = || Acc {
            sum: 0.0,
            count: 0,
            vals: keep_vals.then(Vec::new),
        };
        match spec.system {
            CellSystem::Square { size } => {
                let k = ((cx / size).floor() as i64, (cy / size).floor() as i64);
                sq.entry(k).or_insert_with(blank).push(v);
            }
            CellSystem::H3 { res } => {
                let Ok(res) = h3o::Resolution::try_from(res) else { return };
                let Ok(ll) = h3o::LatLng::new(cy, cx) else { return };
                geo.entry(u64::from(ll.to_cell(res)))
                    .or_insert_with(blank)
                    .push(v);
            }
            CellSystem::A5 { res } => {
                let Ok(cell) = a5::lonlat_to_cell(a5::LonLat::new(cx, cy), res) else {
                    return;
                };
                geo.entry(cell).or_insert_with(blank).push(v);
            }
        }
        rows_used += 1;
    })?;

    // ---- per-cell statistic ------------------------------------------
    let mut sq_vals: HashMap<(i64, i64), (f64, u64)> = sq
        .iter_mut()
        .map(|(k, a)| (*k, (a.stat(spec.stat), a.count)))
        .collect();
    let mut geo_vals: HashMap<u64, (f64, u64)> = geo
        .iter_mut()
        .map(|(k, a)| (*k, (a.stat(spec.stat), a.count)))
        .collect();
    drop(sq);
    drop(geo);

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
    let (batch, schema, cells) = match spec.system {
        CellSystem::Square { size } => {
            materialize(&sq_vals, |&(ix, iy)| {
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
        CellSystem::H3 { .. } => materialize(&geo_vals, |&c| {
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
        CellSystem::A5 { .. } => materialize(&geo_vals, |&c| {
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

/// One renormalized smoothing pass over present cells.
fn smooth_pass<K: Hash + Eq + Copy>(
    vals: &HashMap<K, (f64, u64)>,
    neighbors: impl Fn(&K) -> Vec<(K, (i64, i64))>,
    kernel: Kernel,
) -> HashMap<K, (f64, u64)> {
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
    vals: &HashMap<K, (f64, u64)>,
    cell_geom: impl Fn(&K) -> (String, Vec<(f64, f64)>),
) -> Result<(RecordBatch, SchemaRef, usize), String> {
    let mut wkb = BinaryBuilder::new();
    let mut value = Vec::with_capacity(vals.len());
    let mut count = Vec::with_capacity(vals.len());
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
        count.push(c as i64);
        ids.append_value(id);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("geometry", DataType::Binary, false),
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int64, false),
        Field::new("cell", DataType::Utf8, false),
    ]));
    let n = value.len();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(wkb.finish()) as ArrayRef,
            Arc::new(Float64Array::from(value)),
            Arc::new(Int64Array::from(count)),
            Arc::new(ids.finish()),
        ],
    )
    .map_err(|e| format!("grid batch: {e}"))?;
    Ok((batch, schema, n))
}

/// Stream (centroid_x, centroid_y, value, progress) for every row with a
/// usable centroid and finite value. Uses the covering bbox column when
/// present (columnar, no geometry decode); falls back to decoding
/// geometries otherwise.
fn scan_centroid_values(
    store: &FeatureStore,
    value_col: usize,
    f: &mut dyn FnMut(f64, f64, f64, f32),
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
                            (xmin.value(i) + xmax.value(i)) * 0.5,
                            (ymin.value(i) + ymax.value(i)) * 0.5,
                            v,
                            frac,
                        );
                    }
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
                for ((_, g), v) in geoms.iter().zip(&vals_all) {
                    let (Some(g), Some(v)) = (g, v) else { continue };
                    if !v.is_finite() {
                        continue;
                    }
                    use geo::BoundingRect;
                    if let Some(r) = g.bounding_rect() {
                        f(
                            (r.min().x + r.max().x) * 0.5,
                            (r.min().y + r.max().y) * 0.5,
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
            system,
            stat,
            kernel: Kernel::Box,
            smooth_passes: passes,
        }
    }

    #[test]
    fn square_smoothing_averages_present_neighbors_only() {
        // Two adjacent cells (10, 30) and one far away (100): smoothing
        // must mix the neighbors and leave the loner untouched.
        let mut vals: HashMap<(i64, i64), (f64, u64)> = HashMap::new();
        vals.insert((0, 0), (10.0, 1));
        vals.insert((1, 0), (30.0, 1));
        vals.insert((50, 50), (100.0, 1));
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
    fn acc_stats() {
        let mut a = Acc { sum: 0.0, count: 0, vals: Some(Vec::new()) };
        for v in [1.0, 100.0, 3.0] {
            a.push(v);
        }
        assert_eq!(a.stat(GridStat::Median), 3.0);
        assert_eq!(a.stat(GridStat::Sum), 104.0);
        assert_eq!(a.stat(GridStat::Count), 3.0);
        assert!((a.stat(GridStat::Mean) - 104.0 / 3.0).abs() < 1e-12);
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
                system,
                stat: GridStat::Mean,
                kernel: Kernel::Gaussian,
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
    }
}
