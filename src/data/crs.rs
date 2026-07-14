use std::sync::Arc;

use proj4rs::transform::transform;
use proj4rs::Proj;
use serde_json::Value;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;
/// Half of the Web Mercator world extent, meters.
pub const MERC_MAX: f64 = 20037508.342789244;
/// Max latitude representable in Web Mercator, radians (85.06°).
const MERC_MAX_LAT_RAD: f64 = 1.484_419_982_2;

/// A coordinate reference system usable for transformation.
#[derive(Clone)]
pub struct Crs {
    pub epsg: Option<u32>,
    pub name: String,
    pub proj4: String,
    pub is_latlong: bool,
    pub proj: Arc<Proj>,
}

impl std::fmt::Debug for Crs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Crs({})", self.name)
    }
}

impl Crs {
    pub fn from_proj4(proj4: &str, epsg: Option<u32>, name: &str) -> Result<Self, String> {
        let proj = Proj::from_proj_string(proj4)
            .map_err(|e| format!("invalid proj string for {name}: {e}"))?;
        let is_latlong = proj.is_latlong();
        Ok(Self {
            epsg,
            name: name.to_string(),
            proj4: proj4.to_string(),
            is_latlong,
            proj: Arc::new(proj),
        })
    }

    pub fn from_epsg(code: u32) -> Result<Self, String> {
        if code == 4326 {
            return Ok(Self::wgs84());
        }
        let code16: u16 = code
            .try_into()
            .map_err(|_| format!("EPSG:{code} out of range"))?;
        let def = crs_definitions::from_code(code16)
            .ok_or_else(|| format!("EPSG:{code} not found in embedded CRS database"))?;
        Self::from_proj4(def.proj4, Some(code), &format!("EPSG:{code}"))
    }

    /// WGS84 lon/lat degrees (also used for OGC:CRS84, which GeoParquet defaults to).
    pub fn wgs84() -> Self {
        Self::from_proj4("+proj=longlat +datum=WGS84 +no_defs", Some(4326), "WGS 84")
            .expect("builtin WGS84 proj string")
    }

    pub fn mercator() -> Self {
        Self::from_proj4(
            "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +no_defs",
            Some(3857),
            "Web Mercator",
        )
        .expect("builtin 3857 proj string")
    }

    pub fn same_as(&self, other: &Crs) -> bool {
        self.proj4 == other.proj4
    }

    /// Parse the `crs` field of GeoParquet column metadata (PROJJSON or missing).
    pub fn from_geoparquet_crs(crs: Option<&Value>) -> Result<Self, String> {
        let Some(v) = crs else {
            // Per spec: missing crs means OGC:CRS84.
            return Ok(Self::wgs84());
        };
        if v.is_null() {
            return Ok(Self::wgs84());
        }
        let name = v
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unnamed CRS")
            .to_string();
        if let Some((auth, code)) = projjson_id(v) {
            match (auth.as_str(), code) {
                ("OGC", _) => return Ok(Self::wgs84()),
                ("EPSG", c) => {
                    let mut crs = Self::from_epsg(c)?;
                    crs.name = format!("{name} (EPSG:{c})");
                    return Ok(crs);
                }
                _ => {}
            }
        }
        // Heuristic fallback: a geographic WGS84-based CRS without an id.
        if v.get("type")
            .and_then(Value::as_str)
            .map(|t| t.contains("Geographic"))
            .unwrap_or(false)
        {
            return Ok(Self::wgs84());
        }
        Err(format!(
            "unsupported CRS '{name}': no EPSG id found in PROJJSON"
        ))
    }
}

fn projjson_id(v: &Value) -> Option<(String, u32)> {
    let id = v.get("id")?;
    let auth = id.get("authority")?.as_str()?.to_uppercase();
    let code = match id.get("code")? {
        Value::Number(n) => n.as_u64()? as u32,
        Value::String(s) => s.parse().ok()?,
        _ => return None,
    };
    Some((auth, code))
}

/// Transform a single point between two CRSs. Input/output in the natural
/// units of each CRS (degrees for geographic ones).
pub fn transform_point(src: &Crs, dst: &Crs, x: f64, y: f64) -> Result<(f64, f64), String> {
    if src.same_as(dst) {
        return Ok((x, y));
    }
    let (mut px, mut py) = (x, y);
    if src.is_latlong {
        px *= DEG2RAD;
        py *= DEG2RAD;
    }
    let mut pt = (px, py, 0.0);
    transform(&src.proj, &dst.proj, &mut pt).map_err(|e| e.to_string())?;
    let (mut ox, mut oy) = (pt.0, pt.1);
    if dst.is_latlong {
        ox *= RAD2DEG;
        oy *= RAD2DEG;
    }
    Ok((ox, oy))
}

/// Reusable bulk transformer between two CRSs, with optional latitude clamping
/// (needed when projecting geographic data to Web Mercator).
pub struct BulkTransformer {
    src: Arc<Proj>,
    dst: Arc<Proj>,
    identity: bool,
    src_deg: bool,
    dst_deg: bool,
    clamp_lat_rad: Option<f64>,
}

impl BulkTransformer {
    pub fn new(src: &Crs, dst: &DisplayCrs) -> Self {
        let clamp = if dst.is_mercator() && src.is_latlong {
            Some(MERC_MAX_LAT_RAD)
        } else {
            None
        };
        Self {
            src: src.proj.clone(),
            dst: dst.crs.proj.clone(),
            identity: src.same_as(&dst.crs),
            src_deg: src.is_latlong,
            dst_deg: dst.crs.is_latlong,
            clamp_lat_rad: clamp,
        }
    }

    /// Transform in place; returns false (and leaves NaNs) on failure.
    pub fn apply(&self, x: &mut f64, y: &mut f64) -> bool {
        if self.identity {
            return true;
        }
        let (mut px, mut py) = (*x, *y);
        if self.src_deg {
            px *= DEG2RAD;
            py *= DEG2RAD;
            if let Some(m) = self.clamp_lat_rad {
                py = py.clamp(-m, m);
            }
        }
        let mut pt = (px, py, 0.0);
        match transform(&self.src, &self.dst, &mut pt) {
            Ok(()) => {
                let (mut ox, mut oy) = (pt.0, pt.1);
                if self.dst_deg {
                    ox *= RAD2DEG;
                    oy *= RAD2DEG;
                }
                *x = ox;
                *y = oy;
                ox.is_finite() && oy.is_finite()
            }
            Err(_) => {
                *x = f64::NAN;
                *y = f64::NAN;
                false
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Winkel Tripel (not in proj4rs; implemented natively).
// Forward is analytic; the inverse solves the 2x2 system with Newton.
// Coordinates are in radians on both sides ("radian units" on output).
// ---------------------------------------------------------------------------

/// cos of Winkel's standard parallel: φ₁ = acos(2/π).
const WINTRI_COS_PHI1: f64 = 2.0 / std::f64::consts::PI;
/// Max |X| of the Winkel Tripel graticule: (2 + π) / 2, at (±180°, 0°).
const WINTRI_X_MAX: f64 = (2.0 + std::f64::consts::PI) / 2.0;
/// Max |Y|: π/2, at the poles.
const WINTRI_Y_MAX: f64 = std::f64::consts::FRAC_PI_2;

/// Winkel Tripel forward, lon/lat in radians.
pub fn wintri_forward(lon: f64, lat: f64) -> (f64, f64) {
    let half_lon = lon * 0.5;
    let cos_lat = lat.cos();
    let alpha = (cos_lat * half_lon.cos()).clamp(-1.0, 1.0).acos();
    let sinc = if alpha.abs() < 1e-12 {
        1.0
    } else {
        alpha.sin() / alpha
    };
    let x = 0.5 * (lon * WINTRI_COS_PHI1 + 2.0 * cos_lat * half_lon.sin() / sinc);
    let y = 0.5 * (lat + lat.sin() / sinc);
    (x, y)
}

/// Winkel Tripel inverse via Newton iteration with a numerical Jacobian.
/// Returns None outside the projection domain / on non-convergence.
pub fn wintri_inverse(x: f64, y: f64) -> Option<(f64, f64)> {
    use std::f64::consts::{FRAC_PI_2, PI};
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    // Initial guess: inverse of the averaged equirectangular component.
    let mut lon = (x / (0.5 * (WINTRI_COS_PHI1 + 1.0))).clamp(-PI, PI);
    let mut lat = y.clamp(-FRAC_PI_2, FRAC_PI_2);
    let h = 1e-7;
    for _ in 0..25 {
        let (fx, fy) = wintri_forward(lon, lat);
        let (rx, ry) = (fx - x, fy - y);
        if rx.abs() < 1e-11 && ry.abs() < 1e-11 {
            return Some((lon, lat));
        }
        let (fxl, fyl) = wintri_forward(lon + h, lat);
        let (fxp, fyp) = wintri_forward(lon, lat + h);
        let j11 = (fxl - fx) / h;
        let j21 = (fyl - fy) / h;
        let j12 = (fxp - fx) / h;
        let j22 = (fyp - fy) / h;
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-14 {
            return None;
        }
        lon = (lon - (rx * j22 - ry * j12) / det).clamp(-PI, PI);
        lat = (lat - (ry * j11 - rx * j21) / det).clamp(-FRAC_PI_2, FRAC_PI_2);
    }
    let (fx, fy) = wintri_forward(lon, lat);
    ((fx - x).abs() < 1e-8 && (fy - y).abs() < 1e-8).then_some((lon, lat))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayKind {
    /// EPSG:3857, world normalized to [0,1]² (slippy tiles line up).
    Mercator,
    /// Any proj4rs-backed CRS, affine world mapping.
    Plain,
    /// Winkel Tripel over WGS84 (custom projection code).
    WinkelTripel,
}

/// The projection used for display, plus the mapping from projected
/// coordinates to normalized "world" space.
///
/// World space conventions:
/// - EPSG:3857 maps to [0,1]² exactly like slippy tiles (so the tile pyramid lines up).
/// - Geographic CRSs map degrees with 1 world unit = 360°.
/// - Other projected CRSs map meters with 1 world unit = one equator length.
/// - Winkel Tripel maps its full graticule to x ∈ [0,1].
///   In all cases +y world points down (screen convention).
///
/// For `WinkelTripel`, `crs` is WGS84: reprojection pipelines bring data to
/// lon/lat degrees first, and `world_from_projected`/`projected_from_world`
/// apply the Winkel Tripel step.
#[derive(Clone, Debug)]
pub struct DisplayCrs {
    pub crs: Crs,
    pub kind: DisplayKind,
    pub name: String,
    world_offset: [f64; 2],
    world_per_unit: f64,
    /// Full world extent in world space (frames the empty view).
    bounds: [f64; 4],
}

impl DisplayCrs {
    pub fn new(crs: Crs) -> Self {
        let (kind, world_offset, world_per_unit) = if crs.epsg == Some(3857) {
            (DisplayKind::Mercator, [-MERC_MAX, MERC_MAX], 1.0 / (2.0 * MERC_MAX))
        } else if crs.is_latlong {
            (DisplayKind::Plain, [-180.0, 90.0], 1.0 / 360.0)
        } else {
            (DisplayKind::Plain, [-MERC_MAX, MERC_MAX], 1.0 / (2.0 * MERC_MAX))
        };
        let name = crs.name.clone();
        let mut d = Self {
            crs,
            kind,
            name,
            world_offset,
            world_per_unit,
            bounds: [0.25, 0.25, 0.75, 0.75],
        };
        d.bounds = d.compute_world_bounds();
        d
    }

    /// Sample the lon/lat globe through this projection to find the world
    /// extent (exact constants for the built-ins).
    fn compute_world_bounds(&self) -> [f64; 4] {
        match self.kind {
            DisplayKind::Mercator => return [0.0, 0.0, 1.0, 1.0],
            DisplayKind::WinkelTripel => {
                return [0.0, 0.0, 1.0, 2.0 * WINTRI_Y_MAX * self.world_per_unit]
            }
            DisplayKind::Plain if self.crs.is_latlong => return [0.0, 0.0, 1.0, 0.5],
            DisplayKind::Plain => {}
        }
        let wgs = Crs::wgs84();
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut any = false;
        for lon_i in (-180..=180).step_by(10) {
            for lat_i in (-89..=89).step_by(178 / 22) {
                if let Ok((x, y)) = transform_point(&wgs, &self.crs, lon_i as f64, lat_i as f64)
                {
                    if x.is_finite() && y.is_finite() {
                        let w = self.world_from_projected(x, y);
                        b[0] = b[0].min(w[0]);
                        b[1] = b[1].min(w[1]);
                        b[2] = b[2].max(w[0]);
                        b[3] = b[3].max(w[1]);
                        any = true;
                    }
                }
            }
        }
        if any && b.iter().all(|v| v.is_finite()) {
            b
        } else {
            [0.25, 0.25, 0.75, 0.75]
        }
    }

    pub fn mercator() -> Self {
        Self::new(Crs::mercator())
    }

    pub fn winkel_tripel() -> Self {
        let world_per_unit = 1.0 / (2.0 * WINTRI_X_MAX);
        Self {
            crs: Crs::wgs84(),
            kind: DisplayKind::WinkelTripel,
            name: "Winkel Tripel".into(),
            world_offset: [-WINTRI_X_MAX, WINTRI_Y_MAX],
            world_per_unit,
            bounds: [0.0, 0.0, 1.0, 2.0 * WINTRI_Y_MAX * world_per_unit],
        }
    }

    /// Hobo–Dyer: Lambert cylindrical equal-area with standard parallels
    /// at 37.5°N/S. Straight meridians, areas preserved everywhere.
    pub fn hobo_dyer() -> Self {
        let crs = Crs::from_proj4(
            "+proj=cea +lat_ts=37.5 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
            None,
            "Hobo–Dyer",
        )
        .expect("builtin Hobo-Dyer proj string");
        let mut d = Self::new(crs);
        d.name = "Hobo–Dyer (equal-area)".into();
        d
    }

    pub fn from_epsg(code: u32) -> Result<Self, String> {
        if code == 3857 {
            return Ok(Self::mercator());
        }
        Ok(Self::new(Crs::from_epsg(code)?))
    }

    pub fn is_mercator(&self) -> bool {
        self.kind == DisplayKind::Mercator
    }

    /// Projected coordinates (in `self.crs` units; degrees for Winkel Tripel)
    /// to normalized world space.
    pub fn world_from_projected(&self, x: f64, y: f64) -> [f64; 2] {
        let (x, y) = match self.kind {
            DisplayKind::WinkelTripel => wintri_forward(x * DEG2RAD, y * DEG2RAD),
            _ => (x, y),
        };
        [
            (x - self.world_offset[0]) * self.world_per_unit,
            (self.world_offset[1] - y) * self.world_per_unit,
        ]
    }

    /// Inverse of `world_from_projected`. Returns NaNs outside the
    /// projection domain (e.g. the blank corners around Winkel Tripel).
    pub fn projected_from_world(&self, w: [f64; 2]) -> (f64, f64) {
        let x = w[0] / self.world_per_unit + self.world_offset[0];
        let y = self.world_offset[1] - w[1] / self.world_per_unit;
        match self.kind {
            DisplayKind::WinkelTripel => match wintri_inverse(x, y) {
                Some((lon, lat)) => (lon * RAD2DEG, lat * RAD2DEG),
                None => (f64::NAN, f64::NAN),
            },
            _ => (x, y),
        }
    }

    /// Full world-space extent of this projection (used to frame the view
    /// when no data is loaded).
    pub fn world_bounds(&self) -> [f64; 4] {
        self.bounds
    }

    /// Approximate projected units per world unit (for pixel-tolerance conversion).
    #[allow(dead_code)]
    pub fn units_per_world(&self) -> f64 {
        1.0 / self.world_per_unit
    }
}

/// Cached WGS84 (the status bar converts the cursor every frame).
pub fn wgs84_cached() -> &'static Crs {
    static WGS84: std::sync::OnceLock<Crs> = std::sync::OnceLock::new();
    WGS84.get_or_init(Crs::wgs84)
}

/// Convert a world-space point to WGS84 lon/lat degrees (for status display).
/// Returns None outside the projection domain.
pub fn world_to_lonlat(display: &DisplayCrs, w: [f64; 2]) -> Option<(f64, f64)> {
    let (x, y) = display.projected_from_world(w);
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    transform_point(&display.crs, wgs84_cached(), x, y)
        .ok()
        .filter(|(lon, lat)| lon.is_finite() && lat.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambert93_origin_roundtrip() {
        let l93 = Crs::from_epsg(2154).unwrap();
        let wgs = Crs::wgs84();
        // Lambert-93 false origin: (700000, 6600000) = (3.0°E, 46.5°N)
        let (lon, lat) = transform_point(&l93, &wgs, 700_000.0, 6_600_000.0).unwrap();
        assert!((lon - 3.0).abs() < 1e-6, "lon {lon}");
        assert!((lat - 46.5).abs() < 1e-6, "lat {lat}");
        let (x, y) = transform_point(&wgs, &l93, 3.0, 46.5).unwrap();
        assert!((x - 700_000.0).abs() < 0.01, "x {x}");
        assert!((y - 6_600_000.0).abs() < 0.01, "y {y}");
    }

    #[test]
    fn mercator_world_mapping() {
        let d = DisplayCrs::mercator();
        let wgs = Crs::wgs84();
        // lon/lat (0,0) -> world (0.5, 0.5)
        let (x, y) = transform_point(&wgs, &d.crs, 0.0, 0.0).unwrap();
        let w = d.world_from_projected(x, y);
        assert!((w[0] - 0.5).abs() < 1e-9 && (w[1] - 0.5).abs() < 1e-9, "{w:?}");
        // lon 180 -> world x = 1
        let (x, _) = transform_point(&wgs, &d.crs, 180.0, 0.0).unwrap();
        let w = d.world_from_projected(x, 0.0);
        assert!((w[0] - 1.0).abs() < 1e-9, "{w:?}");
        // north is up: lat 60 must map to world y < 0.5
        let (x, y) = transform_point(&wgs, &d.crs, 0.0, 60.0).unwrap();
        let w = d.world_from_projected(x, y);
        assert!(w[1] < 0.5, "{w:?}");
        // roundtrip
        let (px, py) = d.projected_from_world(w);
        assert!((px - x).abs() < 1e-6 && (py - y).abs() < 1e-6);
    }

    #[test]
    fn wintri_forward_known_values() {
        // Center of the projection.
        let (x, y) = wintri_forward(0.0, 0.0);
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12);
        // Edge of the graticule at the equator: X = (2 + π) / 2.
        let (x, y) = wintri_forward(std::f64::consts::PI, 0.0);
        assert!((x - WINTRI_X_MAX).abs() < 1e-9, "x {x}");
        assert!(y.abs() < 1e-12);
        // North pole: Y = π/2 for any longitude.
        let (_, y) = wintri_forward(1.0, std::f64::consts::FRAC_PI_2);
        assert!((y - WINTRI_Y_MAX).abs() < 1e-9, "y {y}");
    }

    #[test]
    fn wintri_inverse_roundtrip() {
        for &(lon, lat) in &[
            (0.0, 0.0),
            (2.35, 48.85),
            (-170.0, -55.0),
            (179.0, 80.0),
            (-3.7, 40.4),
            (151.2, -33.9),
        ] {
            let (x, y) = wintri_forward(lon * DEG2RAD, lat * DEG2RAD);
            let (rlon, rlat) = wintri_inverse(x, y).expect("converges");
            assert!(
                (rlon * RAD2DEG - lon).abs() < 1e-6 && (rlat * RAD2DEG - lat).abs() < 1e-6,
                "({lon},{lat}) -> ({},{})",
                rlon * RAD2DEG,
                rlat * RAD2DEG
            );
        }
        // Outside the projection domain: must not "converge".
        assert!(wintri_inverse(WINTRI_X_MAX * 1.2, WINTRI_Y_MAX * 1.1).is_none());
    }

    #[test]
    fn wintri_display_world_mapping() {
        let d = DisplayCrs::winkel_tripel();
        // (0,0) lon/lat -> world center x, and roundtrip through the display API.
        let w = d.world_from_projected(0.0, 0.0);
        assert!((w[0] - 0.5).abs() < 1e-12, "{w:?}");
        let w = d.world_from_projected(2.35, 48.85);
        let (lon, lat) = d.projected_from_world(w);
        assert!((lon - 2.35).abs() < 1e-6 && (lat - 48.85).abs() < 1e-6, "{lon} {lat}");
        // World bounds height matches π / (2 + π).
        let b = d.world_bounds();
        let expect_h = std::f64::consts::PI / (2.0 + std::f64::consts::PI);
        assert!((b[3] - expect_h).abs() < 1e-12, "{b:?}");
    }

    #[test]
    fn hobo_dyer_is_cylindrical_equal_area() {
        let d = DisplayCrs::hobo_dyer();
        let wgs = Crs::wgs84();
        // Cylindrical: x depends only on longitude.
        let (x1, _) = transform_point(&wgs, &d.crs, 100.0, 0.0).unwrap();
        let (x2, _) = transform_point(&wgs, &d.crs, 100.0, 55.0).unwrap();
        assert!((x1 - x2).abs() < 1e-6, "{x1} vs {x2}");
        // Linear in longitude.
        let (x180, _) = transform_point(&wgs, &d.crs, 180.0, 0.0).unwrap();
        let (x90, _) = transform_point(&wgs, &d.crs, 90.0, 0.0).unwrap();
        assert!((x180 - 2.0 * x90).abs() < 1e-3, "{x180} vs {x90}");
        // Roundtrip.
        let (px, py) = transform_point(&wgs, &d.crs, 2.35, 48.85).unwrap();
        let (lon, lat) = transform_point(&d.crs, &wgs, px, py).unwrap();
        assert!((lon - 2.35).abs() < 1e-6 && (lat - 48.85).abs() < 1e-6);
        // World bounds: aspect ratio near the spherical value
        // (2·R/cosφs) / (2·π·R·cosφs) ≈ 0.508 at φs = 37.5°.
        let b = d.world_bounds();
        let aspect = (b[3] - b[1]) / (b[2] - b[0]);
        assert!((0.42..=0.58).contains(&aspect), "aspect {aspect}");
        // Equal-area: equal sin-lat bands map to equal y spans.
        let (_, y0) = transform_point(&wgs, &d.crs, 0.0, 0.0).unwrap();
        let band = |lat: f64| {
            let (_, y) = transform_point(&wgs, &d.crs, 0.0, lat).unwrap();
            y - y0
        };
        // y ∝ q(φ) ≈ sin(φ) (ellipsoidal correction is < 1%)
        let ratio = band(90.0) / band(30.0);
        assert!((ratio - 2.0).abs() < 0.02, "ratio {ratio}");
    }

    #[test]
    fn latlong_display_mapping() {
        let d = DisplayCrs::new(Crs::wgs84());
        let w = d.world_from_projected(-180.0, 90.0);
        assert!((w[0]).abs() < 1e-9 && w[1].abs() < 1e-9, "{w:?}");
        let w = d.world_from_projected(180.0, -90.0);
        assert!((w[0] - 1.0).abs() < 1e-9 && (w[1] - 0.5).abs() < 1e-9, "{w:?}");
    }
}
