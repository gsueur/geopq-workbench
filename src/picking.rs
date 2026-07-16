use std::sync::Arc;

use geo::{Contains, Distance, Euclidean, MapCoordsInPlace};
use geo_types::{Geometry, Point};
use rstar::{RTree, AABB};

use crate::data::crs::{transform_point, BulkTransformer, Crs, DisplayCrs};
use crate::data::geometry::{ChunkMesh, FeatureRef};
use crate::data::layer::{PickItem, VectorLayer};
use crate::data::store::FeatureStore;

/// A picked feature.
#[derive(Clone)]
pub struct Selection {
    pub layer_id: u64,
    pub feature: FeatureRef,
    /// Geometry in current world coordinates (for highlight rendering).
    pub world_geom: Geometry<f64>,
    /// Physical measure of the feature, computed in the data CRS:
    /// geodesic meters for lat/long layers, planar CRS units otherwise.
    pub measure: Option<Measure>,
}

/// Length of a linear feature / area of an areal one.
#[derive(Clone, Copy, Debug)]
pub enum Measure {
    Length(f64),
    Area { area: f64, perimeter: f64 },
}

/// Measure a geometry in its data CRS. Geodesic (WGS84 ellipsoid) when
/// the CRS is geographic, planar in CRS units when projected.
pub fn measure_of(geom: &Geometry<f64>, latlong: bool) -> Option<Measure> {
    use geo::{Area, Geodesic, GeodesicArea, Length};
    let planar_ring_len = |ls: &geo_types::LineString<f64>| Euclidean.length(ls);
    match geom {
        Geometry::Line(_) | Geometry::LineString(_) | Geometry::MultiLineString(_) => {
            let len = if latlong {
                match geom {
                    Geometry::Line(g) => Geodesic.length(g),
                    Geometry::LineString(g) => Geodesic.length(g),
                    Geometry::MultiLineString(g) => Geodesic.length(g),
                    _ => unreachable!(),
                }
            } else {
                match geom {
                    Geometry::Line(g) => Euclidean.length(g),
                    Geometry::LineString(g) => Euclidean.length(g),
                    Geometry::MultiLineString(g) => Euclidean.length(g),
                    _ => unreachable!(),
                }
            };
            Some(Measure::Length(len))
        }
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) | Geometry::Rect(_)
        | Geometry::Triangle(_) => {
            let polys: Vec<geo_types::Polygon<f64>> = match geom {
                Geometry::Polygon(p) => vec![p.clone()],
                Geometry::MultiPolygon(mp) => mp.0.clone(),
                Geometry::Rect(r) => vec![r.to_polygon()],
                Geometry::Triangle(t) => vec![t.to_polygon()],
                _ => unreachable!(),
            };
            let (mut area, mut perim) = (0.0f64, 0.0f64);
            for p in &polys {
                if latlong {
                    area += p.geodesic_area_unsigned();
                    perim += p.geodesic_perimeter();
                } else {
                    area += p.unsigned_area();
                    perim += planar_ring_len(p.exterior());
                    for i in p.interiors() {
                        perim += planar_ring_len(i);
                    }
                }
            }
            Some(Measure::Area { area, perimeter: perim })
        }
        _ => None,
    }
}

/// Cheap snapshot of everything picking needs from a layer, so the whole
/// pick (candidate reads, exact tests, attribute fetch) can run off the
/// UI thread — on remote layers those are network reads.
pub struct PickLayer {
    pub id: u64,
    pub visible: bool,
    pub sections: Vec<(Arc<Vec<ChunkMesh>>, Arc<RTree<PickItem>>)>,
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
}

impl PickLayer {
    pub fn of(l: &VectorLayer) -> Self {
        Self {
            id: l.id,
            visible: l.style.visible,
            sections: l
                .sections
                .iter()
                .map(|s| (Arc::clone(&s.chunks), Arc::clone(&s.rtree)))
                .collect(),
            store: Arc::clone(&l.store),
            crs: l.crs.clone(),
        }
    }
}

/// Transform a geometry from a data CRS into world coordinates.
pub fn to_world_geom(
    mut geom: Geometry<f64>,
    crs: &crate::data::crs::Crs,
    display: &DisplayCrs,
) -> Geometry<f64> {
    let tr = BulkTransformer::new(crs, display);
    geom.map_coords_in_place(|c| {
        let (mut x, mut y) = (c.x, c.y);
        tr.apply(&mut x, &mut y);
        let w = display.world_from_projected(x, y);
        geo_types::Coord { x: w[0], y: w[1] }
    });
    geom
}

/// Scan the point instances of chunks near the cursor. Points carry their
/// FeatureRef inline (no R-tree entries), so this is the point pick path.
/// Returns the hit and its world position (no data read needed).
fn pick_point_in_chunks(
    layer: &PickLayer,
    world: [f64; 2],
    tol_world: f64,
) -> Option<(FeatureRef, [f64; 2])> {
    use crate::data::geometry::CHUNK_WORLD;
    let tol2 = tol_world * tol_world;
    let mut best: Option<(f64, FeatureRef, [f64; 2])> = None;
    for chunk in layer.sections.iter().flat_map(|(c, _)| c.iter()) {
        if chunk.point_instances.is_empty() {
            continue;
        }
        // Points are assigned to the chunk containing them, so only chunks
        // whose cell overlaps the tolerance box can hold a hit.
        let o = chunk.origin;
        if world[0] + tol_world < o[0]
            || world[0] - tol_world > o[0] + CHUNK_WORLD
            || world[1] + tol_world < o[1]
            || world[1] - tol_world > o[1] + CHUNK_WORLD
        {
            continue;
        }
        for (p, fref) in chunk.point_instances.iter().zip(&chunk.point_refs) {
            if !fref.is_valid() {
                continue;
            }
            let (px, py) = (o[0] + p[0] as f64, o[1] + p[1] as f64);
            let (dx, dy) = (px - world[0], py - world[1]);
            let d2 = dx * dx + dy * dy;
            if d2 <= tol2 && best.map(|(bd, _, _)| d2 < bd).unwrap_or(true) {
                best = Some((d2, *fref, [px, py]));
            }
        }
    }
    best.map(|(_, f, p)| (f, p))
}

/// Pick the topmost feature near a world-space point.
///
/// `tol_world` is the pick tolerance in world units (derived from pixels).
/// Remote layers turn the candidate reads into network requests — run this
/// off the UI thread (see [`PickLayer`]).
pub fn pick(
    layers: &[PickLayer],
    display: &DisplayCrs,
    world: [f64; 2],
    tol_world: f64,
) -> Option<Selection> {
    // Iterate top-most layer first.
    for layer in layers.iter().rev() {
        if !layer.visible {
            continue;
        }
        // Points render on top of fills/lines; check them first. The
        // highlight comes straight from the rendered instance position —
        // no geometry read.
        if let Some((fref, pos)) = pick_point_in_chunks(layer, world, tol_world) {
            return Some(Selection {
                layer_id: layer.id,
                feature: fref,
                world_geom: Geometry::Point(Point::new(pos[0], pos[1])),
                measure: None,
            });
        }

        let env = AABB::from_corners(
            [world[0] - tol_world, world[1] - tol_world],
            [world[0] + tol_world, world[1] + tol_world],
        );
        let mut candidates: Vec<FeatureRef> = layer
            .sections
            .iter()
            .flat_map(|(_, rtree)| {
                rtree
                    .locate_in_envelope_intersecting(env)
                    .map(|item| item.feature)
            })
            .collect();
        if candidates.is_empty() {
            continue;
        }
        candidates.sort();
        candidates.dedup();
        // Later rows render on top. If the safety cap is hit, preserve those
        // highest row ids instead of discarding them with `truncate`.
        let drop = candidates.len().saturating_sub(512);
        candidates.drain(..drop);

        // Exact test in the data CRS: transform the click point + tolerance there.
        let (px, py) = display.projected_from_world(world);
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        let Ok((cx, cy)) = transform_point(&display.crs, &layer.crs, px, py) else {
            continue;
        };
        let (tx, ty) = display.projected_from_world([world[0] + tol_world, world[1]]);
        let tol_data = match transform_point(&display.crs, &layer.crs, tx, ty) {
            Ok((ex, ey)) => ((ex - cx).powi(2) + (ey - cy).powi(2)).sqrt().max(1e-12),
            Err(_) => continue,
        };
        let click = Point::new(cx, cy);

        // One batched read for all candidate geometries.
        let rows: Vec<u32> = candidates.iter().map(|f| f.index).collect();
        let Ok(geoms) = layer.store.fetch_geoms(&rows) else {
            continue;
        };

        // Test later rows first: they draw on top.
        let mut hit: Option<(FeatureRef, Geometry<f64>)> = None;
        for (row, geom) in geoms.into_iter().rev() {
            let Some(geom) = geom else { continue };
            let is_hit = match &geom {
                Geometry::Polygon(_)
                | Geometry::MultiPolygon(_)
                | Geometry::Rect(_)
                | Geometry::Triangle(_) => {
                    geom.contains(&click)
                        || Euclidean.distance(&geom, &Geometry::Point(click)) <= tol_data
                }
                _ => Euclidean.distance(&geom, &Geometry::Point(click)) <= tol_data,
            };
            if is_hit {
                hit = Some((FeatureRef { index: row }, geom));
                break;
            }
        }
        if let Some((fref, geom)) = hit {
            let measure = measure_of(&geom, layer.crs.is_latlong);
            return Some(Selection {
                layer_id: layer.id,
                feature: fref,
                world_geom: to_world_geom(geom, &layer.crs, display),
                measure,
            });
        }
    }
    None
}

#[cfg(test)]
mod measure_tests {
    use super::*;
    use geo_types::{polygon, LineString};

    #[test]
    fn measures_are_plausible() {
        // ~111 km of meridian at the equator (1° of latitude, geodesic).
        let ls: Geometry<f64> =
            Geometry::LineString(LineString::from(vec![(0.0, 0.0), (0.0, 1.0)]));
        match measure_of(&ls, true) {
            Some(Measure::Length(l)) => {
                assert!((l - 110_574.0).abs() < 500.0, "meridian degree: {l}")
            }
            other => panic!("{other:?}"),
        }
        // Planar: 3-4-5 triangle path.
        let ls2: Geometry<f64> =
            Geometry::LineString(LineString::from(vec![(0.0, 0.0), (3.0, 4.0)]));
        match measure_of(&ls2, false) {
            Some(Measure::Length(l)) => assert!((l - 5.0).abs() < 1e-9),
            other => panic!("{other:?}"),
        }
        // Planar unit square: area 1, perimeter 4.
        let sq: Geometry<f64> = Geometry::Polygon(polygon![
            (x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 1.0, y: 1.0), (x: 0.0, y: 1.0),
        ]);
        match measure_of(&sq, false) {
            Some(Measure::Area { area, perimeter }) => {
                assert!((area - 1.0).abs() < 1e-9);
                assert!((perimeter - 4.0).abs() < 1e-9);
            }
            other => panic!("{other:?}"),
        }
        // Geodesic ~0.01°x0.01° square near the equator ≈ 1.11 km x 1.11 km.
        let gsq: Geometry<f64> = Geometry::Polygon(polygon![
            (x: 0.0, y: 0.0), (x: 0.01, y: 0.0), (x: 0.01, y: 0.01), (x: 0.0, y: 0.01),
        ]);
        match measure_of(&gsq, true) {
            Some(Measure::Area { area, .. }) => {
                let expect = 1_106.0 * 1_113.0; // rough m²
                assert!((area / expect - 1.0).abs() < 0.05, "geodesic area: {area}");
            }
            other => panic!("{other:?}"),
        }
        // Points measure nothing.
        assert!(measure_of(&Geometry::Point(Point::new(0.0, 0.0)), true).is_none());
    }
}
