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

/// A layer's geometry sections as picking sees them: the chunk meshes
/// (points, and the bin each chunk was drawn with) beside the R-tree over
/// everything else.
pub type PickSections = Vec<(Arc<Vec<ChunkMesh>>, Arc<RTree<PickItem>>)>;

/// Cheap snapshot of everything picking needs from a layer, so the whole
/// pick (candidate reads, exact tests, attribute fetch) can run off the
/// UI thread — on remote layers those are network reads.
pub struct PickLayer {
    pub id: u64,
    pub visible: bool,
    pub sections: PickSections,
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
    /// Legend toggles, mirroring `DrawStyle`: only meaningful while the
    /// layer is data-styled, because an unstyled layer reports bin 0 for
    /// everything and the mask would hide all of it.
    pub styled: bool,
    pub hidden_bins: u64,
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
            styled: l.style.style_by.is_some(),
            hidden_bins: l.style.style_by.as_ref().map_or(0, |sb| sb.hidden_bins),
        }
    }

    /// Whether the map is drawing this bin, on the same terms as
    /// [`crate::map::renderer::DrawStyle::bin_hidden`].
    fn bin_hidden(&self, bin: u8) -> bool {
        bin_hidden(self.styled, self.hidden_bins, bin)
    }
}

/// The legend mask, honoured only under data styling: an unstyled layer
/// reports bin 0 for every feature, so a stale mask would make the whole
/// layer unpickable. Mirrors `DrawStyle::bin_hidden`, which is what
/// decides whether the feature is on screen at all.
fn bin_hidden(styled: bool, hidden_bins: u64, bin: u8) -> bool {
    styled && hidden_bins & (1u64 << bin) != 0
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
fn pick_point_in_chunks<'a>(
    chunks: impl Iterator<Item = &'a ChunkMesh>,
    world: [f64; 2],
    tol_world: f64,
    hidden: impl Fn(u8) -> bool,
) -> Option<(FeatureRef, [f64; 2])> {
    let tol2 = tol_world * tol_world;
    let mut best: Option<(f64, FeatureRef, [f64; 2])> = None;
    for chunk in chunks {
        // A chunk holds one style bin, so a hidden class is a whole chunk
        // that is not on screen. Picking through it would select a feature
        // the user cannot see and highlight nothing.
        if chunk.point_instances.is_empty() || hidden(chunk.bin) {
            continue;
        }
        // Points land in the chunk keyed by their feature's bbox center, so
        // a MultiPoint's members can sit far outside the chunk's grid cell.
        // Cull by the chunk's content bounds (chunk-local offsets), not the
        // cell box, or those members would be unpickable.
        let o = chunk.origin;
        let b = chunk.bounds_local;
        if world[0] + tol_world < o[0] + b[0] as f64
            || world[0] - tol_world > o[0] + b[2] as f64
            || world[1] + tol_world < o[1] + b[1] as f64
            || world[1] - tol_world > o[1] + b[3] as f64
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

/// Features whose bbox meets `env`, in row order, minus the style bins
/// the legend has switched off: a class that is not drawn must not be
/// picked either, or clicking empty map selects a feature nobody can see.
fn candidates_near(
    sections: &PickSections,
    env: AABB<[f64; 2]>,
    hidden: impl Fn(u8) -> bool,
) -> Vec<FeatureRef> {
    let mut candidates: Vec<FeatureRef> = sections
        .iter()
        .flat_map(|(_, rtree)| {
            rtree
                .locate_in_envelope_intersecting(env)
                .filter(|item| !hidden(item.bin))
                .map(|item| item.feature)
        })
        .collect();
    candidates.sort();
    candidates.dedup();
    // Later rows render on top. If the safety cap is hit, preserve those
    // highest row ids instead of discarding them with `truncate`.
    let drop = candidates.len().saturating_sub(512);
    candidates.drain(..drop);
    candidates
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
        let layer_chunks = layer.sections.iter().flat_map(|(c, _)| c.iter());
        if let Some((fref, pos)) =
            pick_point_in_chunks(layer_chunks, world, tol_world, |b| layer.bin_hidden(b))
        {
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
        let candidates = candidates_near(&layer.sections, env, |b| layer.bin_hidden(b));
        if candidates.is_empty() {
            continue;
        }

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
mod point_pick_tests {
    use super::*;
    use crate::data::geometry::MeshBuilder;
    use geo_types::MultiPoint;

    /// A class switched off in the legend is not on the map, and must not
    /// be pickable either: clicking it selected an invisible feature and
    /// highlighted nothing. Chunks carry their bin, and so do the R-tree
    /// entries, so both halves of the pick can skip it.
    #[test]
    fn hidden_bin_is_not_pickable() {
        let mut mb = MeshBuilder::default();
        // Two points, one per style bin (a bin change splits chunks).
        mb.bin = 0;
        mb.add(&Geometry::Point(Point::new(0.50, 0.5)), FeatureRef { index: 1 });
        mb.bin = 1;
        mb.add(&Geometry::Point(Point::new(0.60, 0.5)), FeatureRef { index: 2 });
        let chunks = Arc::new(mb.finish());
        assert!(chunks.iter().any(|c| c.bin == 1), "no bin-1 chunk was built");

        let hide_bin_1 = |b: u8| b == 1;
        let tol = 1e-4;
        assert!(
            pick_point_in_chunks(chunks.iter(), [0.50, 0.5], tol, hide_bin_1).is_some(),
            "a visible class stopped being pickable"
        );
        assert!(
            pick_point_in_chunks(chunks.iter(), [0.60, 0.5], tol, hide_bin_1).is_none(),
            "a hidden class was picked"
        );

        // Areal / linear features go through the R-tree instead.
        let rtree = Arc::new(RTree::bulk_load(vec![
            PickItem { bbox: [0.49, 0.49, 0.51, 0.51], feature: FeatureRef { index: 1 }, bin: 0 },
            PickItem { bbox: [0.59, 0.49, 0.61, 0.51], feature: FeatureRef { index: 2 }, bin: 1 },
        ]));
        let sections = vec![(Arc::clone(&chunks), rtree)];
        let env = |x: f64| AABB::from_corners([x - tol, 0.5 - tol], [x + tol, 0.5 + tol]);
        assert_eq!(
            candidates_near(&sections, env(0.60), |_| false)
                .iter()
                .map(|f| f.index)
                .collect::<Vec<_>>(),
            vec![2],
            "the feature is there when nothing is hidden"
        );
        assert!(
            candidates_near(&sections, env(0.60), hide_bin_1).is_empty(),
            "a hidden class survived the candidate filter"
        );
        assert_eq!(candidates_near(&sections, env(0.50), hide_bin_1).len(), 1);
    }

    /// The mask only means anything under data styling: an unstyled layer
    /// reports bin 0 for every feature, so honouring a stale mask would
    /// make the whole layer unpickable.
    #[test]
    fn an_unstyled_layer_ignores_the_hidden_mask() {
        assert!(bin_hidden(true, 0b101, 0));
        assert!(bin_hidden(true, 0b101, 2));
        assert!(!bin_hidden(true, 0b101, 1));
        assert!(!bin_hidden(false, 0b101, 0), "unstyled layers have no bins");
    }

    #[test]
    fn multipoint_member_outside_center_cell_is_pickable() {
        let mut mb = MeshBuilder::default();
        // Members 0.01 world apart, ~20x the chunk cell (1/2048): both are
        // stored in the single chunk keyed by the feature bbox center.
        let mp = MultiPoint::from(vec![(0.5, 0.5), (0.51, 0.5)]);
        assert!(mb
            .add(&Geometry::MultiPoint(mp), FeatureRef { index: 7 })
            .is_some());
        let chunks = mb.finish();
        let tol = 1e-5;

        let shown = |_: u8| false;
        let (fref, pos) = pick_point_in_chunks(chunks.iter(), [0.51, 0.5], tol, shown)
            .expect("member outside the bbox-center cell must be pickable");
        assert_eq!(fref.index, 7);
        assert!((pos[0] - 0.51).abs() < tol && (pos[1] - 0.5).abs() < tol);

        assert!(pick_point_in_chunks(chunks.iter(), [0.5, 0.5], tol, shown).is_some());
        // Between the members: inside the content bounds, but no hit.
        assert!(pick_point_in_chunks(chunks.iter(), [0.505, 0.5], tol, shown).is_none());
    }
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
