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
/// contains it.
pub fn admin_join(
    spec: &AdminJoinSpec,
    centroids_boundary_crs: &[Option<(f64, f64)>],
) -> Result<Vec<Option<String>>, String> {
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

    Ok(centroids_boundary_crs
        .iter()
        .map(|c| {
            let (x, y) = (*c)?;
            let p = geo_types::Point::new(x, y);
            tree.locate_in_envelope_intersecting(AABB::from_point([x, y]))
                .find(|e| e.geom.contains(&p))
                .map(|e| e.value.clone())
        })
        .collect())
}

/// Split a (spatially ordered) row order into hive partitions keyed by the
/// per-row values of the partition fields. Returns (relative dir, rows)
/// preserving the input order inside each partition.
pub fn split_by_fields(
    order: &[u32],
    fields: &[(String, Vec<Option<String>>)],
) -> Result<Vec<(String, Vec<u32>)>, String> {
    // Encode field names once, rather than once per output row. Parquet field
    // names are unrestricted strings; raw '/' or '\\' here would otherwise
    // become path separators when optimize joins this relative directory to
    // the chosen output root.
    let fields: Vec<(String, &[Option<String>])> = fields
        .iter()
        .map(|(name, values)| {
            if name.is_empty() {
                return Err("cannot partition by an empty field name".to_string());
            }
            Ok((encode_hive_component(name), values.as_slice()))
        })
        .collect::<Result<_, _>>()?;
    let mut parts: HashMap<String, Vec<u32>> = HashMap::new();
    for &r in order {
        let dir = fields
            .iter()
            .map(|(name, vals)| {
                let v = vals
                    .get(r as usize)
                    .and_then(|v| v.as_deref())
                    .map(sanitize_hive_value)
                    .unwrap_or_else(|| NULL_PARTITION.to_string());
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
    let max_res = Resolution::try_from(max_res).map_err(|e| format!("H3 resolution: {e}"))?;
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
                let coarse = c.parent(Resolution::Zero).unwrap_or(c);
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
    let mut out: Vec<(String, Vec<u32>)> = done
        .into_iter()
        .map(|(cell, rows)| (format!("h3={cell}"), rows))
        .collect();
    if !nulls.is_empty() {
        out.push((format!("h3={NULL_PARTITION}"), nulls));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
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
