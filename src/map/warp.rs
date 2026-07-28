//! Web Mercator raster tiles in a non-Mercator display projection.
//!
//! A slippy tile is a square in Mercator world space and a curved patch in
//! any other projection, so it is drawn as a subdivided mesh whose vertices
//! are projected exactly and whose interior the GPU interpolates. The error
//! of that approximation falls as O(h²) in the sub-cell size, so a modest
//! grid puts it far under a pixel; adjacent tiles share the vertices along
//! their common edge, so no cracks open between them.
//!
//! What warping cannot fix is text. Labels are baked into the tile pixels,
//! so they shear and rotate with everything else, and no amount of
//! subdivision helps. That is why [`plan`] measures the local distortion
//! and leaves it to the caller to pick a label-free source, or none at all.

use crate::data::crs::{self, BulkTransformer, DisplayCrs};
use crate::map::camera::Camera;
use crate::map::tiles::TileId;

/// Latitude where the Mercator projection is cut off, and with it the tile
/// pyramid. Everything poleward of this simply does not exist in the source.
pub const MERC_MAX_LAT: f64 = 85.051_128_779_806_6;

/// Sub-cells per tile edge. 8 keeps the worst-case interior error well under
/// a pixel for every projection offered, at 81 vertices and 128 triangles per
/// tile, which is nothing next to a vector layer.
pub const TILE_SUBDIV: usize = 8;

/// Beyond this fraction of the world's Mercator width, tiles are refused
/// outright: distortion is extreme, and the missing caps past ±85° would
/// leave a hole exactly where an equal-area world map is supposed to be
/// honest. The projected coastline is the better basemap there.
const WORLD_SPAN_LIMIT: f64 = 0.5;

/// A tile cell whose projected extent exceeds this multiple of its expected
/// size has wrapped around the projection (antimeridian, or a point mapped
/// to the far side of an azimuthal). Dropping it beats drawing a smear
/// across the map.
const CELL_BLOWUP_LIMIT: f64 = 8.0;

/// Longitude/latitude to Mercator world space (slippy [0,1]², +y down).
pub fn merc_world_from_lonlat(lon: f64, lat: f64) -> [f64; 2] {
    let s = lat.clamp(-MERC_MAX_LAT, MERC_MAX_LAT).to_radians().sin();
    [
        (lon + 180.0) / 360.0,
        0.5 - ((1.0 + s) / (1.0 - s)).ln() / (4.0 * std::f64::consts::PI),
    ]
}

/// Inverse of [`merc_world_from_lonlat`].
pub fn lonlat_from_merc_world(m: [f64; 2]) -> (f64, f64) {
    let lon = m[0] * 360.0 - 180.0;
    let t = ((0.5 - m[1]) * 2.0 * std::f64::consts::PI).exp();
    let lat = (2.0 * t.atan() - std::f64::consts::FRAC_PI_2).to_degrees();
    (lon, lat)
}

/// Mercator world space to the display projection's world space.
///
/// Holds the transformer rather than rebuilding it per point: a tile mesh
/// is 81 conversions and a viewport plan another 75.
pub struct Warp {
    to_display: BulkTransformer,
    display: DisplayCrs,
}

impl Warp {
    pub fn new(display: &DisplayCrs) -> Self {
        Self {
            to_display: BulkTransformer::new(crs::wgs84_cached(), display),
            display: display.clone(),
        }
    }

    /// Mercator world → display world. None outside the display projection's
    /// domain (the blank corners around Winkel Tripel, the far hemisphere of
    /// an azimuthal).
    pub fn display_world(&self, m: [f64; 2]) -> Option<[f64; 2]> {
        let (lon, lat) = lonlat_from_merc_world(m);
        let (mut x, mut y) = (lon, lat);
        if !self.to_display.apply(&mut x, &mut y) {
            return None;
        }
        let w = self.display.world_from_projected(x, y);
        (w[0].is_finite() && w[1].is_finite()).then_some(w)
    }

    /// Display world → Mercator world, latitude clamped into the pyramid's
    /// range so a view reaching past ±85° still asks for the tiles that do
    /// exist. None outside the display projection's domain.
    pub fn merc_world(&self, w: [f64; 2]) -> Option<[f64; 2]> {
        let (lon, lat) = crs::world_to_lonlat(&self.display, w)?;
        (lon.is_finite() && lat.is_finite()).then(|| merc_world_from_lonlat(lon, lat))
    }
}

/// What the viewport needs from the tile pyramid, plus how badly the
/// projection deforms the tiles that will fill it.
#[derive(Clone, Copy, Debug)]
pub struct WarpPlan {
    /// Pyramid level whose texels match screen pixels most closely.
    pub zoom: u8,
    /// Mercator-world rect to cover, `[minx, miny, maxx, maxy]`.
    pub merc_bbox: [f64; 4],
    /// Worst local rotation across the viewport, in degrees. This is what
    /// tilts baked-in labels.
    pub rotation_deg: f64,
    /// Worst local anisotropy (ratio of the Jacobian's singular values).
    /// This is what stretches them.
    pub anisotropy: f64,
}

impl WarpPlan {
    /// Whether tiles carrying rendered text still look right. Five degrees
    /// of tilt and a fifth of stretch are both a little under what the eye
    /// picks up on a place name.
    pub fn labels_survive(&self) -> bool {
        self.rotation_deg <= 5.0 && self.anisotropy <= 1.2
    }
}

/// Why the viewport gets no raster basemap at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoTiles {
    /// The view spans too much of the globe. Mercator holds no data past
    /// ±85°, so warped tiles would leave a blank cap over each pole.
    WorldScale,
    /// The viewport does not meet the Mercator domain at all.
    OffMap,
}

impl NoTiles {
    pub fn reason(&self) -> &'static str {
        match self {
            NoTiles::WorldScale => {
                "no tiles at world scale: Mercator stops at ±85°, so the poles \
                 would be blank. The coastline overlay covers this view."
            }
            NoTiles::OffMap => "no tiles: this view falls outside the Mercator domain.",
        }
    }
}

/// Samples per viewport edge when measuring the view. 5×5 catches the
/// curvature four corners would miss without costing a visible amount.
const PLAN_SAMPLES: usize = 5;

/// Work out which tiles the viewport needs and how deformed they will be.
pub fn plan(
    warp: &Warp,
    camera: &Camera,
    viewport_px: [f32; 2],
    max_zoom: u8,
) -> Result<WarpPlan, NoTiles> {
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(PLAN_SAMPLES * PLAN_SAMPLES);
    for iy in 0..PLAN_SAMPLES {
        for ix in 0..PLAN_SAMPLES {
            let sx = viewport_px[0] * ix as f32 / (PLAN_SAMPLES - 1) as f32;
            let sy = viewport_px[1] * iy as f32 / (PLAN_SAMPLES - 1) as f32;
            let w = camera.screen_to_world([sx, sy], viewport_px);
            if let Some(m) = warp.merc_world(w) {
                pts.push(m);
            }
        }
    }
    if pts.len() < 4 {
        return Err(NoTiles::OffMap);
    }
    // A viewport with much of itself off the projection is showing the whole
    // map and then some. The surviving samples would form a narrow column
    // and understate the span, so decide on the sample count instead.
    if pts.len() * 2 < PLAN_SAMPLES * PLAN_SAMPLES {
        return Err(NoTiles::WorldScale);
    }

    let mut bbox = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in &pts {
        bbox[0] = bbox[0].min(p[0]);
        bbox[1] = bbox[1].min(p[1]);
        bbox[2] = bbox[2].max(p[0]);
        bbox[3] = bbox[3].max(p[1]);
    }
    if bbox[2] - bbox[0] > WORLD_SPAN_LIMIT || bbox[3] - bbox[1] > WORLD_SPAN_LIMIT {
        return Err(NoTiles::WorldScale);
    }

    // Tile level: how much Mercator world one screen pixel covers at the
    // view centre. At level z one world unit is 256·2^z texels, so matching
    // texels to pixels means 256·2^z·d = 1.
    let centre = camera.screen_to_world(
        [viewport_px[0] * 0.5, viewport_px[1] * 0.5],
        viewport_px,
    );
    let step = 1.0 / camera.scale();
    let zoom = match (
        warp.merc_world(centre),
        warp.merc_world([centre[0] + step, centre[1]]),
        warp.merc_world([centre[0], centre[1] + step]),
    ) {
        (Some(c), Some(dx), Some(dy)) => {
            let ex = (dx[0] - c[0]).hypot(dx[1] - c[1]);
            let ey = (dy[0] - c[0]).hypot(dy[1] - c[1]);
            // The finer of the two axes: better to over-sample the stretched
            // one than to smear the sharp one.
            let d = ex.min(ey).max(1e-12);
            (-(256.0 * d).log2()).round().clamp(0.0, max_zoom as f64) as u8
        }
        _ => return Err(NoTiles::OffMap),
    };

    // Distortion of Mercator → display, measured on the tiles themselves:
    // one sub-cell is the scale at which the mesh has to be a good fit.
    let h = (bbox[2] - bbox[0]).max(1e-9) / (PLAN_SAMPLES * TILE_SUBDIV) as f64;
    let mut rotation: f64 = 0.0;
    let mut anisotropy: f64 = 1.0;
    for p in &pts {
        let (Some(o), Some(du), Some(dv)) = (
            warp.display_world(*p),
            warp.display_world([p[0] + h, p[1]]),
            warp.display_world([p[0], p[1] + h]),
        ) else {
            continue;
        };
        let j = [
            (du[0] - o[0]) / h,
            (dv[0] - o[0]) / h,
            (du[1] - o[1]) / h,
            (dv[1] - o[1]) / h,
        ];
        rotation = rotation.max(j[2].atan2(j[0]).to_degrees().abs());
        if let Some(a) = anisotropy_of(j) {
            anisotropy = anisotropy.max(a);
        }
    }

    Ok(WarpPlan {
        zoom,
        merc_bbox: bbox,
        rotation_deg: rotation,
        anisotropy,
    })
}

/// Ratio of the singular values of a 2×2 matrix `[a, b, c, d]` (row major):
/// how much a circle is squashed into an ellipse. None if it is degenerate.
fn anisotropy_of(j: [f64; 4]) -> Option<f64> {
    let [a, b, c, d] = j;
    let e = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    let disc = (e * e - 4.0 * det * det).max(0.0).sqrt();
    let hi = ((e + disc) * 0.5).max(0.0).sqrt();
    let lo = ((e - disc) * 0.5).max(0.0).sqrt();
    (lo > 1e-12 && hi.is_finite()).then_some(hi / lo)
}

/// One tile as a curved patch in display world space.
pub struct TileMesh {
    /// World origin the offsets are relative to. Offsets stay f32, which is
    /// why they are relative: at deep zoom absolute world coordinates would
    /// lose the low bits.
    pub origin: [f64; 2],
    /// `[x, y, u, v]`: world offset from `origin`, then texture coordinate.
    pub verts: Vec<[f32; 4]>,
    pub indices: Vec<u16>,
}

/// Project one tile into a subdivided mesh. None when too little of it lands
/// inside the display projection's domain to be worth drawing.
pub fn tile_mesh(warp: &Warp, id: TileId, subdiv: usize) -> Option<TileMesh> {
    let r = id.world_rect();
    let n = subdiv.max(1);
    let (w, h) = (r[2] - r[0], r[3] - r[1]);

    let mut world: Vec<Option<[f64; 2]>> = Vec::with_capacity((n + 1) * (n + 1));
    for iy in 0..=n {
        for ix in 0..=n {
            let m = [
                r[0] + w * ix as f64 / n as f64,
                r[1] + h * iy as f64 / n as f64,
            ];
            world.push(warp.display_world(m));
        }
    }
    let origin = *world.iter().flatten().next()?;

    // Expected size of one sub-cell, from the median-ish diagonal of the
    // valid cells, used to spot cells that wrapped around the projection.
    let cell_limit = {
        let mut d: Vec<f64> = Vec::new();
        for iy in 0..n {
            for ix in 0..n {
                let (a, b) = (
                    world[iy * (n + 1) + ix],
                    world[(iy + 1) * (n + 1) + ix + 1],
                );
                if let (Some(a), Some(b)) = (a, b) {
                    d.push((b[0] - a[0]).hypot(b[1] - a[1]));
                }
            }
        }
        if d.is_empty() {
            return None;
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d[d.len() / 2] * CELL_BLOWUP_LIMIT
    };

    let verts: Vec<[f32; 4]> = world
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let (ix, iy) = (i % (n + 1), i / (n + 1));
            let (u, v) = (ix as f32 / n as f32, iy as f32 / n as f32);
            match w {
                Some(p) => [
                    (p[0] - origin[0]) as f32,
                    (p[1] - origin[1]) as f32,
                    u,
                    v,
                ],
                None => [f32::NAN, f32::NAN, u, v],
            }
        })
        .collect();

    let mut indices: Vec<u16> = Vec::with_capacity(n * n * 6);
    for iy in 0..n {
        for ix in 0..n {
            let c = [
                iy * (n + 1) + ix,
                iy * (n + 1) + ix + 1,
                (iy + 1) * (n + 1) + ix,
                (iy + 1) * (n + 1) + ix + 1,
            ];
            if c.iter().any(|&k| world[k].is_none()) {
                continue;
            }
            let p: Vec<[f64; 2]> = c.iter().map(|&k| world[k].unwrap()).collect();
            let span = p
                .iter()
                .flat_map(|a| p.iter().map(move |b| (b[0] - a[0]).hypot(b[1] - a[1])))
                .fold(0.0f64, f64::max);
            if span > cell_limit {
                continue;
            }
            for &k in &[c[0], c[1], c[2], c[2], c[1], c[3]] {
                indices.push(k as u16);
            }
        }
    }
    (!indices.is_empty()).then_some(TileMesh {
        origin,
        verts,
        indices,
    })
}

/// Tile ids covering a Mercator-world rect at one level, capped so a
/// degenerate view cannot ask for thousands.
pub fn tiles_for(bbox: [f64; 4], zoom: u8, cap: usize) -> Vec<TileId> {
    let n = 1i64 << zoom;
    let x0 = ((bbox[0] * n as f64).floor() as i64).max(0);
    let x1 = ((bbox[2] * n as f64).ceil() as i64).min(n);
    let y0 = ((bbox[1] * n as f64).floor() as i64).max(0);
    let y1 = ((bbox[3] * n as f64).ceil() as i64).min(n);
    let mut out = Vec::new();
    for x in x0..x1 {
        for y in y0..y1 {
            out.push(TileId {
                z: zoom,
                x: x as u32,
                y: y as u32,
            });
            if out.len() >= cap {
                return out;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn merc_world_roundtrip() {
        for &(lon, lat) in &[(0.0, 0.0), (2.35, 48.85), (-71.06, 42.36), (145.0, -37.8)] {
            let m = merc_world_from_lonlat(lon, lat);
            let (l2, t2) = lonlat_from_merc_world(m);
            assert!(approx(lon, l2, 1e-9) && approx(lat, t2, 1e-9), "{lon} {lat}");
        }
    }

    #[test]
    fn merc_world_corners() {
        let tl = merc_world_from_lonlat(-180.0, MERC_MAX_LAT);
        let br = merc_world_from_lonlat(180.0, -MERC_MAX_LAT);
        assert!(approx(tl[0], 0.0, 1e-12) && approx(tl[1], 0.0, 1e-9), "{tl:?}");
        assert!(approx(br[0], 1.0, 1e-12) && approx(br[1], 1.0, 1e-9), "{br:?}");
    }

    #[test]
    fn anisotropy_of_identity_and_stretch() {
        assert!(approx(anisotropy_of([1.0, 0.0, 0.0, 1.0]).unwrap(), 1.0, 1e-12));
        assert!(approx(anisotropy_of([2.0, 0.0, 0.0, 1.0]).unwrap(), 2.0, 1e-12));
        // Pure rotation is not a distortion.
        let (c, s) = (0.6, 0.8);
        assert!(approx(anisotropy_of([c, -s, s, c]).unwrap(), 1.0, 1e-12));
        assert!(anisotropy_of([0.0, 0.0, 0.0, 0.0]).is_none());
    }

    /// In Mercator itself the warp is the identity, so a plan over it must
    /// report no distortion and the camera's own zoom.
    #[test]
    fn mercator_warp_is_the_identity() {
        let d = DisplayCrs::mercator();
        let w = Warp::new(&d);
        for &m in &[[0.5, 0.5], [0.2, 0.7], [0.81, 0.13]] {
            let out = w.display_world(m).unwrap();
            assert!(approx(out[0], m[0], 1e-9) && approx(out[1], m[1], 1e-9), "{out:?}");
        }
        let cam = Camera {
            center: [0.5, 0.5],
            zoom: 6.0,
        };
        let p = plan(&w, &cam, [1024.0, 768.0], 20).expect("mercator plans");
        assert_eq!(p.zoom, 6);
        assert!(p.rotation_deg < 1e-6, "{}", p.rotation_deg);
        assert!(approx(p.anisotropy, 1.0, 1e-6), "{}", p.anisotropy);
        assert!(p.labels_survive());
    }

    fn lambert93() -> DisplayCrs {
        DisplayCrs::from_epsg(2154).expect("EPSG:2154")
    }

    /// A city-scale view in a national grid is where the old all-or-nothing
    /// gate cost the most: the distortion there is invisible.
    #[test]
    fn metro_scale_national_grid_keeps_labels() {
        let d = lambert93();
        let w = Warp::new(&d);
        let toulouse = w
            .display_world(merc_world_from_lonlat(1.44, 43.60))
            .expect("Toulouse projects");
        let cam = Camera {
            center: toulouse,
            zoom: 12.0,
        };
        let p = plan(&w, &cam, [1280.0, 900.0], 20).expect("plans");
        assert!(
            p.labels_survive(),
            "rot {:.3}° aniso {:.4}",
            p.rotation_deg,
            p.anisotropy
        );
        assert_eq!(p.zoom, 12, "tile level should track the camera closely here");
    }

    /// Zoomed out over the same grid the meridians visibly converge, so
    /// label-bearing tiles must be refused even though the mesh is fine.
    #[test]
    fn continental_conic_rejects_labels_but_still_plans() {
        let d = DisplayCrs::from_epsg(3035).expect("EPSG:3035 (Europe LAEA)");
        let w = Warp::new(&d);
        let centre = w
            .display_world(merc_world_from_lonlat(10.0, 52.0))
            .expect("projects");
        let cam = Camera {
            center: centre,
            zoom: 6.0,
        };
        let p = plan(&w, &cam, [1280.0, 900.0], 20).expect("plans");
        assert!(!p.labels_survive(), "rot {:.2}°", p.rotation_deg);
        assert!(p.zoom > 0);
    }

    /// The world view is the one case where no raster basemap is the right
    /// answer, and the caller is told which case it is.
    #[test]
    fn world_view_refuses_tiles() {
        let d = DisplayCrs::hobo_dyer();
        let w = Warp::new(&d);
        let cam = Camera {
            center: [0.5, 0.5],
            zoom: 0.0,
        };
        assert_eq!(
            plan(&w, &cam, [1600.0, 900.0], 20).err(),
            Some(NoTiles::WorldScale)
        );
    }

    /// Adjacent tiles must agree along their shared edge, or seams open up.
    #[test]
    fn neighbouring_meshes_share_their_edge() {
        let d = lambert93();
        let w = Warp::new(&d);
        let (z, x, y) = (7u8, 64u32, 45u32);
        let a = tile_mesh(&w, TileId { z, x, y }, TILE_SUBDIV).expect("left mesh");
        let b = tile_mesh(&w, TileId { z, x: x + 1, y }, TILE_SUBDIV).expect("right mesh");
        let n = TILE_SUBDIV;
        for iy in 0..=n {
            // Right column of a, left column of b, in absolute world space.
            let va = a.verts[iy * (n + 1) + n];
            let vb = b.verts[iy * (n + 1)];
            let ax = a.origin[0] + va[0] as f64;
            let ay = a.origin[1] + va[1] as f64;
            let bx = b.origin[0] + vb[0] as f64;
            let by = b.origin[1] + vb[1] as f64;
            assert!(
                approx(ax, bx, 1e-9) && approx(ay, by, 1e-9),
                "row {iy}: ({ax}, {ay}) vs ({bx}, {by})"
            );
        }
    }

    /// The mesh has to beat a flat quad, which is the whole reason it exists.
    #[test]
    fn subdivision_beats_a_single_quad() {
        let d = DisplayCrs::from_epsg(3035).expect("EPSG:3035");
        let w = Warp::new(&d);
        let id = TileId { z: 4, x: 8, y: 5 };
        let r = id.world_rect();
        let truth = w
            .display_world([(r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5])
            .expect("centre projects");

        // Bilinear centre of the four corners: what one quad would draw.
        let c: Vec<[f64; 2]> = [
            [r[0], r[1]],
            [r[2], r[1]],
            [r[0], r[3]],
            [r[2], r[3]],
        ]
        .iter()
        .map(|p| w.display_world(*p).expect("corner projects"))
        .collect();
        let flat = [
            c.iter().map(|p| p[0]).sum::<f64>() / 4.0,
            c.iter().map(|p| p[1]).sum::<f64>() / 4.0,
        ];
        let flat_err = (flat[0] - truth[0]).hypot(flat[1] - truth[1]);

        let m = tile_mesh(&w, id, TILE_SUBDIV).expect("mesh");
        let mid = m.verts[(TILE_SUBDIV / 2) * (TILE_SUBDIV + 1) + TILE_SUBDIV / 2];
        let mesh_err = (m.origin[0] + mid[0] as f64 - truth[0])
            .hypot(m.origin[1] + mid[1] as f64 - truth[1]);

        assert!(mesh_err <= flat_err / 10.0, "mesh {mesh_err} vs flat {flat_err}");
    }

    #[test]
    fn tiles_for_covers_and_caps() {
        let ids = tiles_for([0.25, 0.25, 0.5, 0.5], 4, 4096);
        assert_eq!(ids.len(), 16);
        assert!(ids.iter().all(|t| t.z == 4 && t.x >= 4 && t.x < 8));
        assert_eq!(tiles_for([0.0, 0.0, 1.0, 1.0], 8, 100).len(), 100);
    }
}
