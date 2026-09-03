/// Map camera operating in normalized "world" coordinates (see `DisplayCrs`
/// for the mapping from projected coordinates to world space).
///
/// `zoom` follows the slippy-map convention: at zoom z, the full world width
/// spans `256 * 2^z` physical pixels.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World coordinates at the viewport center.
    pub center: [f64; 2],
    pub zoom: f64,
}

pub const MIN_ZOOM: f64 = -2.0;
pub const MAX_ZOOM: f64 = 24.0;

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            zoom: 1.5,
        }
    }
}

impl Camera {
    /// Physical pixels per world unit.
    pub fn scale(&self) -> f64 {
        256.0 * self.zoom.exp2()
    }

    /// Screen position (physical px, origin at viewport top-left) of a world point.
    #[allow(dead_code)]
    pub fn world_to_screen(&self, w: [f64; 2], viewport_px: [f32; 2]) -> [f32; 2] {
        let s = self.scale();
        [
            ((w[0] - self.center[0]) * s + viewport_px[0] as f64 * 0.5) as f32,
            ((w[1] - self.center[1]) * s + viewport_px[1] as f64 * 0.5) as f32,
        ]
    }

    pub fn screen_to_world(&self, p: [f32; 2], viewport_px: [f32; 2]) -> [f64; 2] {
        let s = self.scale();
        [
            self.center[0] + (p[0] as f64 - viewport_px[0] as f64 * 0.5) / s,
            self.center[1] + (p[1] as f64 - viewport_px[1] as f64 * 0.5) / s,
        ]
    }

    pub fn pan_px(&mut self, delta_px: [f32; 2]) {
        let s = self.scale();
        self.center[0] -= delta_px[0] as f64 / s;
        self.center[1] -= delta_px[1] as f64 / s;
    }

    /// Zoom by `dz` keeping the world point under `cursor_px` fixed.
    pub fn zoom_about(&mut self, dz: f64, cursor_px: [f32; 2], viewport_px: [f32; 2]) {
        let anchor = self.screen_to_world(cursor_px, viewport_px);
        self.zoom = (self.zoom + dz).clamp(MIN_ZOOM, MAX_ZOOM);
        let s = self.scale();
        // Solve for center so that anchor stays under the cursor.
        self.center[0] = anchor[0] - (cursor_px[0] as f64 - viewport_px[0] as f64 * 0.5) / s;
        self.center[1] = anchor[1] - (cursor_px[1] as f64 - viewport_px[1] as f64 * 0.5) / s;
    }

    /// Fit world-space bounds `[minx, miny, maxx, maxy]` into the viewport.
    ///
    /// The extent is floored at a world unit that is still meaningful at
    /// MAX_ZOOM (a single point, or a bounds collapsed by a projection
    /// failure, otherwise asks for a scale of 1e12 px per world unit) and
    /// the scale itself is clamped *before* the log, so the zoom the
    /// camera ends at is the one this scale describes. Clamping after it
    /// left `zoom` and `scale` disagreeing, and the fit then framed the
    /// data at the wrong size without saying so.
    pub fn fit(&mut self, bounds: [f64; 4], viewport_px: [f32; 2], padding_px: f64) {
        // 1e-9 world units is ~4 mm on an Earth-sized projection, and
        // below the f32 mesh coordinates the renderer draws with.
        const MIN_EXTENT: f64 = 1e-9;
        let w = (bounds[2] - bounds[0]).max(MIN_EXTENT);
        let h = (bounds[3] - bounds[1]).max(MIN_EXTENT);
        let vw = (viewport_px[0] as f64 - 2.0 * padding_px).max(64.0);
        let vh = (viewport_px[1] as f64 - 2.0 * padding_px).max(64.0);
        let lo = 256.0 * MIN_ZOOM.exp2();
        let hi = 256.0 * MAX_ZOOM.exp2();
        let scale = (vw / w).min(vh / h).clamp(lo, hi);
        self.zoom = (scale / 256.0).log2().clamp(MIN_ZOOM, MAX_ZOOM);
        self.center = [(bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_world_roundtrip() {
        let cam = Camera { center: [0.31, 0.62], zoom: 7.3 };
        let vp = [1600.0, 1000.0];
        let w = cam.screen_to_world([123.0, 456.0], vp);
        let s = cam.world_to_screen(w, vp);
        assert!((s[0] - 123.0).abs() < 1e-3 && (s[1] - 456.0).abs() < 1e-3, "{s:?}");
    }

    #[test]
    fn zoom_about_keeps_anchor() {
        let mut cam = Camera { center: [0.5, 0.5], zoom: 4.0 };
        let vp = [800.0, 600.0];
        let cursor = [200.0, 150.0];
        let before = cam.screen_to_world(cursor, vp);
        cam.zoom_about(1.7, cursor, vp);
        let after = cam.screen_to_world(cursor, vp);
        assert!((before[0] - after[0]).abs() < 1e-12 && (before[1] - after[1]).abs() < 1e-12);
    }

    /// Degenerate bounds are routine: a single-point layer, or one whose
    /// projection collapsed. The fit has to come back with a camera —
    /// finite, inside the zoom limits, centred on the data — instead of
    /// a scale of 1e14 that the zoom clamp then silently disagrees with.
    #[test]
    fn fit_survives_zero_area_and_oversized_bounds() {
        let mut cam = Camera::default();
        let vp = [1000.0, 800.0];

        cam.fit([0.4, 0.7, 0.4, 0.7], vp, 40.0);
        assert!(cam.zoom.is_finite(), "zoom {}", cam.zoom);
        assert_eq!(cam.zoom, MAX_ZOOM, "a point should sit at the zoom cap");
        assert_eq!(cam.center, [0.4, 0.7]);
        assert!(cam.scale().is_finite() && cam.scale() > 0.0);

        // Wider than any world: pinned at the other end, still finite.
        cam.fit([-1e9, -1e9, 1e9, 1e9], vp, 40.0);
        assert_eq!(cam.zoom, MIN_ZOOM);
        assert!(cam.scale().is_finite() && cam.scale() > 0.0);

        // Bounds that arrive non-finite must not poison the zoom.
        cam.fit([f64::NAN, 0.0, f64::NAN, 1.0], vp, 40.0);
        assert!(cam.zoom.is_finite(), "NaN bounds left zoom {}", cam.zoom);
        assert!((MIN_ZOOM..=MAX_ZOOM).contains(&cam.zoom));
    }

    #[test]
    fn fit_contains_bounds() {
        let mut cam = Camera::default();
        let vp = [1000.0, 800.0];
        let b = [0.2, 0.3, 0.25, 0.32];
        cam.fit(b, vp, 40.0);
        let tl = cam.world_to_screen([b[0], b[1]], vp);
        let br = cam.world_to_screen([b[2], b[3]], vp);
        assert!(tl[0] >= 0.0 && tl[1] >= 0.0 && br[0] <= vp[0] && br[1] <= vp[1], "{tl:?} {br:?}");
    }
}
