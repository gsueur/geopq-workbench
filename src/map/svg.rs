//! Scene → SVG for "Export view to SVG".
//!
//! The export is a mirror of the frame, not a re-export of the datasets:
//! it carries exactly the features the viewer decided to draw, at the
//! camera the user is looking through, with the styles the frame
//! resolved. That contract is what makes this module worth isolating —
//! it takes a scene of world-space geometry plus [`DrawStyle`]s and
//! returns a string, so the whole styling and camera argument can be
//! tested without a GPU, a window or a file dialog. Collecting the scene
//! (which decodes geometry, over the network on remote layers) belongs
//! to the app.
//!
//! Colors travel as the shader's floats and are written out as bytes
//! scaled by 255, because the render target is a plain (non-sRGB)
//! framebuffer: what the shader writes is what the display shows, and a
//! published colour map's exact RGB values only survive that way.

use std::fmt::Write as _;
use std::path::Path;

use eframe::egui;
use geo_types::Geometry;

use crate::data::layer::{LineCap, LinePattern, PointShape};
use crate::map::camera::Camera;
use crate::map::renderer::DrawStyle;

/// One collected feature, in display-world coordinates.
pub struct SvgFeature {
    pub geom: Geometry<f64>,
    /// Style bin the map drew it with (0 when the layer is not styled).
    pub bin: u8,
    /// Oversized feature: its fill draws below every normal fill of the
    /// layer, as `geometry::spans_underlay` routes it on the map.
    pub underlay: bool,
}

/// One layer's contribution, in the map's own layer order.
pub struct SvgLayer {
    pub name: String,
    pub style: DrawStyle,
    pub features: Vec<SvgFeature>,
    /// Marker centres in world coordinates, with their bin.
    pub points: Vec<([f64; 2], u8)>,
}

impl SvgLayer {
    /// An empty layer with everything but the style defaulted, for the
    /// built-in overlays that only carry lines.
    pub fn new(name: impl Into<String>, style: DrawStyle) -> Self {
        Self {
            name: name.into(),
            style,
            features: Vec::new(),
            points: Vec::new(),
        }
    }
}

/// Everything one export needs. Sizes are in points (1 SVG unit = 1 CSS
/// pixel = 1 egui point); the camera works in physical pixels, so
/// `pixels_per_point` divides out.
pub struct SvgScene {
    pub camera: Camera,
    pub viewport_px: [f32; 2],
    pub pixels_per_point: f64,
    pub background: egui::Color32,
    /// Credit text and plate colours, from the map's own `credit_colors`.
    pub credit_colors: (egui::Color32, egui::Color32),
    pub layers: Vec<SvgLayer>,
    /// Attribution lines, in the order the map stacks them.
    pub credits: Vec<String>,
    /// Honesty notes for `<desc>`: what the screen showed that the file
    /// cannot reproduce (the basemap, a decimated layer).
    pub notes: Vec<String>,
}

/// World → point mapping. Same arithmetic as [`Camera::world_to_screen`]
/// kept in f64: rounding to 0.01 pt is finer than an f32 screen
/// coordinate resolves once a deep zoom pushes it past ~10⁵ px.
struct Proj {
    center: [f64; 2],
    scale: f64,
    half: [f64; 2],
    ppp: f64,
}

impl Proj {
    fn new(scene: &SvgScene) -> Self {
        Self {
            center: scene.camera.center,
            scale: scene.camera.scale(),
            half: [
                scene.viewport_px[0] as f64 * 0.5,
                scene.viewport_px[1] as f64 * 0.5,
            ],
            ppp: scene.pixels_per_point.max(1e-6),
        }
    }

    fn pt(&self, w: [f64; 2]) -> [f64; 2] {
        [
            ((w[0] - self.center[0]) * self.scale + self.half[0]) / self.ppp,
            ((w[1] - self.center[1]) * self.scale + self.half[1]) / self.ppp,
        ]
    }
}

/// A coordinate, rounded to 0.01 pt and stripped of trailing zeros.
/// Non-finite input would poison a whole path, so it fails loudly to the
/// caller instead (the path is dropped).
fn n(v: f64) -> String {
    let r = (v * 100.0).round() / 100.0;
    if r == 0.0 {
        // `-0` is legal SVG but noise in a diff and in a test.
        return "0".into();
    }
    let s = format!("{r:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Byte triple + unmultiplied alpha of an egui colour.
fn parts(c: egui::Color32) -> ([u8; 3], f32) {
    let a = c.a() as f32 / 255.0;
    if a <= 0.0 {
        return ([0, 0, 0], 0.0);
    }
    let un = |v: u8| ((v as f32 / a).round().clamp(0.0, 255.0)) as u8;
    ([un(c.r()), un(c.g()), un(c.b())], a)
}

fn hex_bytes(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

/// Shader RGB → sRGB hex. See the module note on the framebuffer.
fn hex(rgb: [f32; 3]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    hex_bytes([b(rgb[0]), b(rgb[1]), b(rgb[2])])
}

/// XML attribute / text escaping. Layer names and credit lines come from
/// files, so they carry whatever the publisher wrote.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            // Control characters are not representable in XML 1.0 at all.
            c if (c as u32) < 0x20 && c != '\t' && c != '\n' => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// Append one ring / polyline to a path string. `closed` adds the `Z`.
/// Returns false when too few finite vertices survived to draw.
fn sub_path(pts: impl Iterator<Item = [f64; 2]>, closed: bool, out: &mut String) -> bool {
    // Rounded once, up front: the coordinates that survive rounding are
    // the ones the file will carry, so duplicate detection has to see
    // them, not the pre-rounding values.
    let mut kept: Vec<[f64; 2]> = Vec::new();
    for p in pts {
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        let r = [(p[0] * 100.0).round() / 100.0, (p[1] * 100.0).round() / 100.0];
        if kept.last() != Some(&r) {
            kept.push(r);
        }
    }
    // A ring repeats its first vertex; `Z` says the same thing in one
    // character.
    if closed && kept.len() > 2 && kept.first() == kept.last() {
        kept.pop();
    }
    if kept.len() < 2 {
        return false;
    }
    for (i, p) in kept.iter().enumerate() {
        let cmd = if i == 0 { "M" } else { " L" };
        let _ = write!(out, "{cmd}{} {}", n(p[0]), n(p[1]));
    }
    if closed {
        out.push('Z');
    }
    true
}

/// Split a geometry into its areal and linear path data. A polygon
/// contributes to both: the renderer fills it *and* draws its rings as
/// outlines, so the SVG has to do the same or borders vanish.
fn build_paths(g: &Geometry<f64>, proj: &Proj, areal: &mut String, linear: &mut String) {
    use geo_types::Geometry as G;
    let ring = |ls: &geo_types::LineString<f64>, out: &mut String| {
        let sep = if out.is_empty() { "" } else { " " };
        let mark = out.len();
        out.push_str(sep);
        if !sub_path(ls.0.iter().map(|c| proj.pt([c.x, c.y])), true, out) {
            out.truncate(mark);
        }
    };
    let poly = |p: &geo_types::Polygon<f64>, out: &mut String| {
        ring(p.exterior(), out);
        for i in p.interiors() {
            ring(i, out);
        }
    };
    match g {
        G::Polygon(p) => poly(p, areal),
        G::MultiPolygon(mp) => {
            for p in &mp.0 {
                poly(p, areal);
            }
        }
        G::Rect(r) => poly(&r.to_polygon(), areal),
        G::Triangle(t) => poly(&t.to_polygon(), areal),
        G::LineString(ls) => {
            let sep = if linear.is_empty() { "" } else { " " };
            let mark = linear.len();
            linear.push_str(sep);
            if !sub_path(ls.0.iter().map(|c| proj.pt([c.x, c.y])), false, linear) {
                linear.truncate(mark);
            }
        }
        G::MultiLineString(ml) => {
            for ls in &ml.0 {
                build_paths(&G::LineString(ls.clone()), proj, areal, linear);
            }
        }
        G::Line(l) => {
            let sep = if linear.is_empty() { "" } else { " " };
            let mark = linear.len();
            linear.push_str(sep);
            let pts = [proj.pt([l.start.x, l.start.y]), proj.pt([l.end.x, l.end.y])];
            if !sub_path(pts.into_iter(), false, linear) {
                linear.truncate(mark);
            }
        }
        G::GeometryCollection(gc) => {
            for m in &gc.0 {
                build_paths(m, proj, areal, linear);
            }
        }
        // Points arrive through `SvgLayer::points`, from the same
        // instance buffers the marker pass draws.
        G::Point(_) | G::MultiPoint(_) => {}
    }
}

/// Marker outline for a shape, mirroring `paint_marker` / `sd_marker`:
/// `r` is the circumradius (`radius · reach()`), so every symbol keeps
/// the ink weight of the circle it replaces.
fn marker_element(
    shape: PointShape,
    c: [f64; 2],
    radius: f64,
    attrs: &str,
    out: &mut String,
) {
    let r = radius * shape.reach() as f64;
    let ring = |count: usize, rad: f64, start: f64| -> Vec<[f64; 2]> {
        (0..count)
            .map(|i| {
                let a = start + std::f64::consts::TAU * i as f64 / count as f64;
                [c[0] + rad * a.cos(), c[1] + rad * a.sin()]
            })
            .collect()
    };
    let up = -std::f64::consts::FRAC_PI_2;
    let poly = |pts: Vec<[f64; 2]>, out: &mut String| {
        let mut d = String::new();
        if sub_path(pts.into_iter(), true, &mut d) {
            let _ = writeln!(out, r#"<path d="{d}"{attrs}/>"#);
        }
    };
    match shape {
        PointShape::Circle => {
            let _ = writeln!(
                out,
                r#"<circle cx="{}" cy="{}" r="{}"{attrs}/>"#,
                n(c[0]),
                n(c[1]),
                n(r)
            );
        }
        PointShape::Square => {
            let h = r * std::f64::consts::SQRT_2 * 0.5;
            poly(
                vec![
                    [c[0] - h, c[1] - h],
                    [c[0] + h, c[1] - h],
                    [c[0] + h, c[1] + h],
                    [c[0] - h, c[1] + h],
                ],
                out,
            );
        }
        PointShape::Triangle => poly(ring(3, r, up), out),
        PointShape::Diamond => poly(ring(4, r, up), out),
        PointShape::Hexagon => poly(ring(6, r, up), out),
        PointShape::Star => {
            // One 10-vertex polygon: SVG fills a concave path directly,
            // so the core-plus-tips decomposition egui needs is not.
            let outer = ring(5, r, up);
            let inner = ring(5, r * 0.5, up + std::f64::consts::TAU / 10.0);
            let mut pts = Vec::with_capacity(10);
            for i in 0..5 {
                pts.push(outer[i]);
                pts.push(inner[i]);
            }
            poly(pts, out);
        }
    }
}

/// The document as the bytes the chosen path asks for: gzip (SVGZ) when
/// the name ends in `.svgz`, the plain text otherwise. SVGZ is what
/// Inkscape and Illustrator write for exactly this situation — an SVG is
/// verbose coordinate text and compresses several-fold — and the format
/// is nothing but the gzip stream, so the choice can live in the file
/// name alone.
pub fn encode_for(path: &Path, doc: &str) -> Result<Vec<u8>, String> {
    let svgz = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("svgz"));
    if !svgz {
        return Ok(doc.as_bytes().to_vec());
    }
    use std::io::Write as _;
    let mut enc = flate2::write::GzEncoder::new(
        Vec::with_capacity(doc.len() / 4),
        flate2::Compression::best(),
    );
    enc.write_all(doc.as_bytes())
        .and_then(|()| enc.finish())
        .map_err(|e| format!("svgz compression failed: {e}"))
}

/// Render the scene as a self-contained SVG 1.1 document.
pub fn render(scene: &SvgScene) -> String {
    let proj = Proj::new(scene);
    let w = (scene.viewport_px[0] as f64 / proj.ppp).max(1.0);
    let h = (scene.viewport_px[1] as f64 / proj.ppp).max(1.0);
    let (wn, hn) = (n(w), n(h));

    let mut s = String::with_capacity(64 * 1024);
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{wn}px" height="{hn}px" viewBox="0 0 {wn} {hn}">"#
    );
    s.push_str("<desc>Map view exported by GeoPQ Workbench.\n");
    for note in &scene.notes {
        let _ = writeln!(s, "{}", esc(note));
    }
    s.push_str("</desc>\n");
    let _ = writeln!(
        s,
        r#"<defs><clipPath id="gp-viewport"><rect x="0" y="0" width="{wn}" height="{hn}"/></clipPath></defs>"#
    );
    let (bg, bg_a) = parts(scene.background);
    let _ = writeln!(
        s,
        r#"<rect x="0" y="0" width="{wn}" height="{hn}" fill="{}"{}/>"#,
        hex_bytes(bg),
        opacity_attr("fill-opacity", bg_a)
    );
    // Whole features are emitted, so paths run far past the viewport at
    // any real zoom. The clip is the viewBox made explicit, which is what
    // an editor needs to crop them rather than show them.
    let _ = writeln!(s, r#"<g clip-path="url(#gp-viewport)">"#);

    for (li, layer) in scene.layers.iter().enumerate() {
        render_layer(&mut s, li, layer, &proj);
    }

    s.push_str("</g>\n");
    render_credits(&mut s, scene, w, h);
    s.push_str("</svg>\n");
    s
}

fn render_layer(s: &mut String, li: usize, layer: &SvgLayer, proj: &Proj) {
    let st = &layer.style;
    let _ = writeln!(s, r#"<g id="layer-{li}">"#);
    let _ = writeln!(s, "<title>{}</title>", esc(&layer.name));

    // Project each feature once: a polygon's fill and its outline are the
    // same rings, and the transform is the expensive part of both.
    let draws_shapes = st.fill_color[3] > 0.0 || st.line_color[3] > 0.0;
    let paths: Vec<(String, String)> = layer
        .features
        .iter()
        .map(|f| {
            let (mut areal, mut linear) = (String::new(), String::new());
            if draws_shapes && !st.bin_hidden(f.bin) {
                build_paths(&f.geom, proj, &mut areal, &mut linear);
            }
            (areal, linear)
        })
        .collect();

    // Fills. One group at the layer's fill alpha holding OPAQUE paths:
    // the renderer draws fills opaque into an offscreen texture and
    // composites the whole layer once, so overlapping features do not
    // darken each other. An SVG group opacity is the same composite.
    if st.fill_color[3] > 0.0 {
        let mut body = String::new();
        // Underlay chunks draw before all normal fills on the map: an
        // aggregate polygon overlapping thousands of parcels must not
        // paint over them.
        for pass in [true, false] {
            for (f, (areal, _)) in layer.features.iter().zip(&paths) {
                if f.underlay != pass || areal.is_empty() || st.bin_hidden(f.bin) {
                    continue;
                }
                let rgb = st.rgb_for(f.bin, st.fill_color);
                let _ = writeln!(
                    body,
                    r#"<path d="{areal}" fill="{}" fill-rule="evenodd"/>"#,
                    hex(rgb)
                );
            }
        }
        if !body.is_empty() {
            let _ = writeln!(s, r#"<g opacity="{}">"#, round4(st.fill_color[3]));
            s.push_str(&body);
            s.push_str("</g>\n");
        }
    }

    // Strokes: polygon rings and linear features alike, per-bin colour
    // and per-bin width when a width ramp is set. Joints are round
    // whatever the cap is: the segment-instanced line pass has no other
    // join, so any other value would be a lie about the map.
    if st.line_color[3] > 0.0 {
        let cap = match st.line_cap {
            LineCap::Round => "round",
            LineCap::Square => "square",
            // A flat cap adds nothing past the segment's end — SVG's butt.
            LineCap::Flat => "butt",
        };
        for (f, (areal, linear)) in layer.features.iter().zip(&paths) {
            if st.bin_hidden(f.bin) {
                continue;
            }
            let d = match (areal.is_empty(), linear.is_empty()) {
                (true, true) => continue,
                (false, true) => areal.as_str(),
                (true, false) => linear.as_str(),
                (false, false) => &format!("{areal} {linear}"),
            };
            let e = st.edge_for(f.bin);
            let width_px = st.half_width_for(f.bin) * 2.0;
            let dashes = dash_array(st.line_pattern, st.line_cap, width_px, proj.ppp);
            let _ = writeln!(
                s,
                r#"<path d="{d}" fill="none" stroke="{}"{} stroke-width="{}" stroke-linecap="{cap}" stroke-linejoin="round"{dashes}/>"#,
                hex([e[0], e[1], e[2]]),
                opacity_attr("stroke-opacity", e[3]),
                n(width_px as f64 / proj.ppp),
            );
        }
    }

    // Markers, on top of the layer's own fills and lines.
    if st.point_color[3] > 0.0 && !layer.points.is_empty() {
        let radius = st.point_radius_px as f64 / proj.ppp;
        // The renderer insets the border inside the marker and caps it to
        // half the radius; SVG has no inset stroke, so this is a centred
        // stroke of the same width — half of it sits outside the symbol.
        let border = (st.line_half_width_px * 2.0).min(st.point_radius_px * 0.5) as f64 / proj.ppp;
        let bordered = st.line_color[3] > 0.0 && border > 0.0;
        for (w, bin) in &layer.points {
            if st.bin_hidden(*bin) {
                continue;
            }
            let rgb = st.rgb_for(*bin, st.point_color);
            let e = st.edge_for(*bin);
            let mut attrs = format!(
                r#" fill="{}"{}"#,
                hex(rgb),
                opacity_attr("fill-opacity", st.point_color[3])
            );
            if bordered {
                let _ = write!(
                    attrs,
                    r#" stroke="{}"{} stroke-width="{}""#,
                    hex([e[0], e[1], e[2]]),
                    opacity_attr("stroke-opacity", e[3]),
                    n(border)
                );
            }
            marker_element(st.point_shape, proj.pt(*w), radius, &attrs, s);
        }
    }

    s.push_str("</g>\n");
}

/// Credits bottom-right, stacked upward on translucent plates, as the
/// map paints them. SVG cannot measure text, so the plate width is
/// estimated from the character count at the 10 pt proportional face —
/// generous enough that a plate never ends mid-word.
fn render_credits(s: &mut String, scene: &SvgScene, w: f64, h: f64) {
    if scene.credits.is_empty() {
        return;
    }
    const FONT: f64 = 10.0;
    const PAD: [f64; 2] = [4.0, 1.0];
    let (fg, fg_a) = parts(scene.credit_colors.0);
    let (bg, bg_a) = parts(scene.credit_colors.1);
    let _ = writeln!(s, r#"<g id="credits">"#);
    let mut y = h - 4.0;
    for c in scene.credits.iter().rev() {
        let tw = c.chars().count() as f64 * FONT * 0.52;
        let th = FONT * 1.25;
        let x = w - 6.0 - tw;
        let _ = writeln!(
            s,
            r#"<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="{}"{}/>"#,
            n(x - PAD[0]),
            n(y - th - PAD[1]),
            n(tw + 2.0 * PAD[0]),
            n(th + 2.0 * PAD[1]),
            hex_bytes(bg),
            opacity_attr("fill-opacity", bg_a)
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{}" font-family="sans-serif" font-size="{FONT}" fill="{}"{}>{}</text>"#,
            n(x),
            n(y - th * 0.25),
            hex_bytes(fg),
            opacity_attr("fill-opacity", fg_a),
            esc(c)
        );
        y -= th + 2.0 * PAD[1] + 1.0;
    }
    s.push_str("</g>\n");
}

/// The ` stroke-dasharray="…"` attribute for a pattern, or an empty
/// string for a solid stroke. `dashes_px` is the renderer's own pattern
/// in physical px, including the flat-cap correction and the 3 px floor
/// on the unit; the trailing pair is dropped when the pattern only has
/// one dash-gap cycle, which is what it means there.
fn dash_array(pattern: LinePattern, cap: LineCap, width_px: f32, ppp: f64) -> String {
    let d = pattern.dashes_px(cap, width_px);
    if d[0] < 0.0 {
        return String::new();
    }
    let v = |i: usize| n(d[i] as f64 / ppp);
    if d[2] == 0.0 && d[3] == 0.0 {
        format!(r#" stroke-dasharray="{},{}""#, v(0), v(1))
    } else {
        format!(
            r#" stroke-dasharray="{},{},{},{}""#,
            v(0),
            v(1),
            v(2),
            v(3)
        )
    }
}

/// An opacity attribute, omitted when the value is fully opaque: SVG
/// already defaults to 1 and every byte here is repeated per element.
fn opacity_attr(name: &str, a: f32) -> String {
    if a >= 1.0 {
        return String::new();
    }
    format!(r#" {name}="{}""#, round4(a))
}

/// Opacities and widths: four decimals, trailing zeros gone.
fn round4(v: f32) -> String {
    let r = ((v as f64) * 10_000.0).round() / 10_000.0;
    if r == 0.0 {
        return "0".into();
    }
    let s = format!("{r:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::layer::STYLE_BINS;
    use geo_types::{Coord, LineString, MultiPolygon, Polygon};
    use std::sync::Arc;

    fn style(fill_a: f32, line_a: f32, point_a: f32) -> DrawStyle {
        DrawStyle {
            fill_color: [0.2, 0.4, 0.8, fill_a],
            line_color: [0.1, 0.1, 0.1, line_a],
            point_color: [0.9, 0.3, 0.1, point_a],
            line_half_width_px: 0.6,
            point_radius_px: 4.0,
            point_shape: PointShape::Circle,
            ..Default::default()
        }
    }

    fn bins() -> Arc<[[f32; 3]; STYLE_BINS]> {
        let mut lut = [[0.0f32; 3]; STYLE_BINS];
        for (i, c) in lut.iter_mut().enumerate() {
            let t = i as f32 / (STYLE_BINS - 1) as f32;
            *c = [t, 1.0 - t, 0.5];
        }
        Arc::new(lut)
    }

    /// Half-widths per bin: 0.5, 1.5, 2.5 px for the three classes used.
    fn widths() -> Arc<[f32; STYLE_BINS]> {
        let mut lut = [2.5f32; STYLE_BINS];
        lut[0] = 0.5;
        lut[1] = 1.5;
        Arc::new(lut)
    }

    fn square(cx: f64, cy: f64, r: f64) -> Geometry<f64> {
        Geometry::Polygon(Polygon::new(
            LineString(vec![
                Coord { x: cx - r, y: cy - r },
                Coord { x: cx + r, y: cy - r },
                Coord { x: cx + r, y: cy + r },
                Coord { x: cx - r, y: cy + r },
                Coord { x: cx - r, y: cy - r },
            ]),
            vec![],
        ))
    }

    /// One polygon layer (binned fills + outlines), one line layer, one
    /// star-marker point layer with borders, at a fixed camera.
    fn scene() -> SvgScene {
        let mut polys = style(0.35, 1.0, 0.0);
        polys.bin_colors = Some(bins());
        // A width ramp over the classes: bin 0 thin, bin 2 thick.
        polys.bin_half_widths = Some(widths());
        // Dashed with flat caps, on the line layer only.
        let mut lines = style(0.0, 0.8, 0.0);
        lines.line_half_width_px = 1.5;
        lines.line_pattern = LinePattern::Dash;
        lines.line_cap = LineCap::Flat;
        let mut points = style(0.0, 1.0, 1.0);
        points.point_shape = PointShape::Star;
        points.point_radius_px = 5.0;

        let mut poly_layer = SvgLayer::new("parcels", polys);
        for (i, x) in [0.50, 0.51, 0.52].iter().enumerate() {
            poly_layer.features.push(SvgFeature {
                geom: square(*x, 0.5, 0.001),
                bin: i as u8,
                underlay: i == 2,
            });
        }
        let mut line_layer = SvgLayer::new("roads", lines);
        for y in [0.499, 0.5] {
            line_layer.features.push(SvgFeature {
                geom: Geometry::LineString(LineString(vec![
                    Coord { x: 0.49, y },
                    Coord { x: 0.53, y },
                ])),
                bin: 0,
                underlay: false,
            });
        }
        let mut point_layer = SvgLayer::new("wells", points);
        point_layer.points = vec![([0.5, 0.5], 0), ([0.505, 0.5], 1)];

        SvgScene {
            camera: Camera { center: [0.5, 0.5], zoom: 10.0 },
            viewport_px: [800.0, 600.0],
            pixels_per_point: 2.0,
            background: egui::Color32::from_rgb(24, 24, 28),
            credit_colors: (
                egui::Color32::from_white_alpha(200),
                egui::Color32::from_black_alpha(180),
            ),
            layers: vec![poly_layer, line_layer, point_layer],
            credits: vec!["© OpenStreetMap contributors".into()],
            notes: vec!["The raster basemap is not included.".into()],
        }
    }

    /// Count occurrences of a tag opening, e.g. `<path `.
    fn count(s: &str, needle: &str) -> usize {
        s.matches(needle).count()
    }

    #[test]
    fn document_frame_is_valid_and_sized_in_points() {
        let svg = render(&scene());
        assert!(svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
        assert!(svg.trim_end().ends_with("</svg>"));
        // 800x600 physical px at 2 px/pt.
        assert!(svg.contains(r#"viewBox="0 0 400 300""#), "{}", &svg[..400]);
        assert!(svg.contains(r#"width="400px" height="300px""#));
        assert!(svg.contains(r##"<rect x="0" y="0" width="400" height="300" fill="#18181c""##));
        assert!(svg.contains(r#"clip-path="url(#gp-viewport)""#));
        // Self-contained: nothing to fetch, nothing to execute.
        assert!(!svg.contains("<script"), "no scripts");
        assert!(!svg.contains("http://") || svg.matches("http://").count() == 1);
        assert!(!svg.contains("xlink:href"), "no external references");
        assert_eq!(count(&svg, "<svg"), 1);
        assert_eq!(count(&svg, "</g>"), count(&svg, "<g "));
    }

    #[test]
    fn element_counts_match_the_features() {
        let svg = render(&scene());
        let l0 = layer_block(&svg, 0);
        // Three polygons: three opaque fills inside the group, three
        // stroked outlines outside it.
        assert_eq!(count(&l0, r#"fill-rule="evenodd""#), 3);
        assert_eq!(count(&l0, r#"fill="none""#), 3);
        let l1 = layer_block(&svg, 1);
        // Lines have no fill alpha: strokes only, no fill group.
        assert_eq!(count(&l1, r#"fill="none""#), 2);
        assert_eq!(count(&l1, "<g opacity="), 0);
        let l2 = layer_block(&svg, 2);
        assert_eq!(count(&l2, "<path"), 2, "one star per instance");
        assert_eq!(count(&l2, "<circle"), 0, "star layer draws no circles");
    }

    /// Everything between `<g id="layer-{i}">` and its closing tag.
    fn layer_block(svg: &str, i: usize) -> String {
        let start = svg
            .find(&format!(r#"<g id="layer-{i}">"#))
            .expect("layer group");
        let rest = &svg[start..];
        let end = rest
            .find(&format!(r#"<g id="layer-{}">"#, i + 1))
            .unwrap_or_else(|| rest.find("</g>\n<g id=\"credits\"").unwrap_or(rest.len()));
        rest[..end].to_string()
    }

    #[test]
    fn a_known_world_point_lands_where_the_camera_puts_it() {
        let sc = scene();
        // Centre of the viewport, in points.
        let mid = Proj::new(&sc).pt([0.5, 0.5]);
        assert!((mid[0] - 200.0).abs() < 1e-9 && (mid[1] - 150.0).abs() < 1e-9, "{mid:?}");
        // A world offset of 0.005 at zoom 10 is 0.005·256·2¹⁰ = 1310.72
        // physical px = 655.36 pt.
        let off = Proj::new(&sc).pt([0.505, 0.5]);
        assert!((off[0] - 855.36).abs() < 1e-6, "{off:?}");
        // The document projection must agree with the camera the frame
        // used, or the SVG would be a different view of the same data.
        let cam = sc.camera.world_to_screen([0.505, 0.5], sc.viewport_px);
        assert!((cam[0] as f64 / sc.pixels_per_point - off[0]).abs() < 0.01);

        let svg = render(&sc);
        // The second marker sits at that x, rounded to 0.01 pt.
        assert!(
            svg.contains(r#"M855.36 146.34"#),
            "star at the projected point missing"
        );
        // First marker at the centre, first vertex straight up: the star's
        // outer radius is 5 px · 1.4620 reach / 2 ppp = 3.655 pt.
        assert!(svg.contains(r#"M200 146.34"#), "star tip up at the centre");
    }

    #[test]
    fn fill_group_composites_once_over_opaque_paths() {
        let svg = render(&scene());
        let l0 = layer_block(&svg, 0);
        assert!(l0.contains(r#"<g opacity="0.35">"#), "{l0}");
        // Every member of the group is opaque: the group alpha is the
        // whole composite, exactly as the offscreen pass does it.
        for path in l0.split("<path").skip(1) {
            if path.contains("fill-rule") {
                assert!(
                    !path.contains("fill-opacity"),
                    "fill paths must be opaque: {path}"
                );
            }
        }
        // Underlay feature (bin 2) is emitted before the normal fills, or
        // it would paint over them the way the offscreen pass forbids.
        let group = &l0[l0.find("<g opacity=").unwrap()..l0.find("</g>").unwrap()];
        let first = group.split("<path").nth(1).expect("a fill path");
        let bin2 = hex(bins()[2]);
        assert!(first.contains(&bin2), "underlay must come first: {first}");
    }

    #[test]
    fn per_bin_colors_and_hidden_bins_follow_the_legend() {
        let mut sc = scene();
        let svg = render(&sc);
        for b in 0..3u8 {
            assert!(
                svg.contains(&hex(bins()[b as usize])),
                "bin {b} colour missing"
            );
        }
        // Outlines darken the bin colour by 0.65, as `edge_for` does.
        let e = bins()[1];
        assert!(svg.contains(&hex([e[0] * 0.65, e[1] * 0.65, e[2] * 0.65])));

        // Hiding bin 1 must remove that feature entirely — fill and
        // outline. Equal path counts would also be the symptom of an
        // exporter that ignored the legend and only recoloured.
        sc.layers[0].style.hidden_bins = 1 << 1;
        let hidden = render(&sc);
        let (a, b) = (layer_block(&svg, 0), layer_block(&hidden, 0));
        assert_eq!(count(&a, "<path") - 2, count(&b, "<path"));
        assert!(!b.contains(&hex(bins()[1])), "hidden bin still drawn");
        // Adversarial: unhide and it comes back.
        sc.layers[0].style.hidden_bins = 0;
        assert_eq!(count(&layer_block(&render(&sc), 0), "<path"), count(&a, "<path"));
    }

    #[test]
    fn dashes_are_the_renderers_pattern_and_only_on_the_dashed_layer() {
        let svg = render(&scene());
        let (l0, l1) = (layer_block(&svg, 0), layer_block(&svg, 1));
        // Solid layers carry no dasharray at all; an exporter that always
        // emitted one would still "have dashes" on the dashed layer.
        assert_eq!(count(&l0, "stroke-dasharray"), 0, "{l0}");
        assert_eq!(count(&l1, "stroke-dasharray"), 2, "one per line feature");
        // Dash at 3 px full width, flat cap: dashes_px gives [12, 6, 0, 0]
        // physical px, so 6,3 pt at 2 px/pt, and the empty trailing pair
        // is dropped rather than written as zeros.
        let expect = LinePattern::Dash.dashes_px(LineCap::Flat, 3.0);
        assert_eq!(expect, [12.0, 6.0, 0.0, 0.0], "renderer pattern moved");
        assert!(l1.contains(r#"stroke-dasharray="6,3""#), "{l1}");
        assert!(l1.contains(r#"stroke-linecap="butt""#), "flat cap is butt");
        // Round joints everywhere: the line pass has no other join.
        assert_eq!(count(&svg, r#"stroke-linejoin="round""#), count(&svg, "fill=\"none\""));
    }

    #[test]
    fn a_width_ramp_gives_each_class_its_own_stroke_width() {
        let svg = render(&scene());
        let l0 = layer_block(&svg, 0);
        // Half-widths 0.5/1.5/2.5 px → full 1/3/5 px → 0.5/1.5/2.5 pt.
        for w in ["0.5", "1.5", "2.5"] {
            assert!(
                l0.contains(&format!(r#"stroke-width="{w}""#)),
                "missing width {w} in {l0}"
            );
        }
        // Without the ramp every stroke falls back to the layer width, so
        // three distinct values would collapse to one.
        let mut flat = scene();
        flat.layers[0].style.bin_half_widths = None;
        let f0 = layer_block(&render(&flat), 0);
        assert_eq!(count(&f0, r#"stroke-width="0.6""#), 3, "{f0}");
    }

    #[test]
    fn star_markers_have_ten_vertices_and_a_border() {
        let svg = render(&scene());
        let l2 = layer_block(&svg, 2);
        let d = l2
            .split(r#" d=""#)
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("a marker path");
        // One move plus nine lines: a single closed 10-gon, not the
        // core-plus-five-triangles fan the egui painter needs.
        assert_eq!(d.matches('M').count(), 1, "{d}");
        assert_eq!(d.matches('L').count(), 9, "{d}");
        assert!(d.ends_with('Z'));
        // Border width: min(line_half_width·2, radius·0.5) = 1.2 px at
        // 2 px/pt.
        assert!(l2.contains(r#"stroke-width="0.6""#), "{l2}");
    }

    #[test]
    fn polygon_holes_survive_as_subpaths() {
        let mut sc = scene();
        let outer = LineString(vec![
            Coord { x: 0.49, y: 0.49 },
            Coord { x: 0.51, y: 0.49 },
            Coord { x: 0.51, y: 0.51 },
            Coord { x: 0.49, y: 0.51 },
            Coord { x: 0.49, y: 0.49 },
        ]);
        let hole = LineString(vec![
            Coord { x: 0.497, y: 0.497 },
            Coord { x: 0.503, y: 0.497 },
            Coord { x: 0.503, y: 0.503 },
            Coord { x: 0.497, y: 0.503 },
            Coord { x: 0.497, y: 0.497 },
        ]);
        sc.layers[0].features = vec![SvgFeature {
            geom: Geometry::MultiPolygon(MultiPolygon(vec![Polygon::new(outer, vec![hole])])),
            bin: 0,
            underlay: false,
        }];
        let svg = render(&sc);
        let l0 = layer_block(&svg, 0);
        let fill = l0
            .split(r#" d=""#)
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("fill path");
        assert_eq!(fill.matches('M').count(), 2, "ring + hole: {fill}");
        assert_eq!(fill.matches('Z').count(), 2);
        assert!(l0.contains(r#"fill-rule="evenodd""#), "holes need a fill rule");
    }

    #[test]
    fn an_empty_view_still_yields_a_document() {
        let sc = SvgScene {
            camera: Camera::default(),
            viewport_px: [640.0, 480.0],
            pixels_per_point: 1.0,
            background: egui::Color32::from_rgb(244, 243, 240),
            credit_colors: (
                egui::Color32::from_black_alpha(200),
                egui::Color32::from_white_alpha(200),
            ),
            layers: vec![SvgLayer::new("empty", style(0.4, 1.0, 1.0))],
            credits: Vec::new(),
            notes: Vec::new(),
        };
        let svg = render(&sc);
        assert!(svg.contains(r#"viewBox="0 0 640 480""#));
        assert!(svg.contains(r##"fill="#f4f3f0""##), "background rect");
        assert_eq!(count(&svg, "<path"), 0);
        assert_eq!(count(&svg, "<g opacity="), 0, "no empty fill group");
        assert_eq!(count(&svg, "</g>"), count(&svg, "<g "));
    }

    #[test]
    fn text_from_files_is_escaped() {
        let mut sc = scene();
        sc.layers[0].name = "A & B <raw>".into();
        sc.credits = vec![r#"© "Someone" & co"#.into()];
        let svg = render(&sc);
        assert!(svg.contains("<title>A &amp; B &lt;raw&gt;</title>"));
        assert!(svg.contains("&quot;Someone&quot; &amp; co"));
        // Exactly the tags we wrote: no stray element from the data.
        assert_eq!(count(&svg, "<raw>"), 0);
    }

    #[test]
    fn degenerate_geometry_is_dropped_not_emitted_broken() {
        let mut sc = scene();
        sc.layers[0].features = vec![
            // A ring that collapses to one point after rounding.
            SvgFeature { geom: square(0.5, 0.5, 1e-12), bin: 0, underlay: false },
            // Non-finite coordinates from a failed projection.
            SvgFeature {
                geom: Geometry::LineString(LineString(vec![
                    Coord { x: f64::NAN, y: 0.5 },
                    Coord { x: f64::INFINITY, y: 0.5 },
                ])),
                bin: 0,
                underlay: false,
            },
        ];
        let svg = render(&sc);
        assert_eq!(count(&layer_block(&svg, 0), "<path"), 0);
        assert!(!svg.contains("NaN") && !svg.contains("inf"), "{svg}");
    }

    /// An .svgz path gets a gzip stream that decodes back to the exact
    /// document; an .svg path gets the text untouched. Extension case
    /// must not matter — a file dialog on macOS happily returns .SVGZ.
    #[test]
    fn svgz_is_the_same_document_compressed() {
        use std::io::Read as _;
        let doc = render(&scene());
        let plain = encode_for(Path::new("/tmp/map.svg"), &doc).unwrap();
        assert_eq!(plain, doc.as_bytes());
        for name in ["/tmp/map.svgz", "/tmp/MAP.SVGZ"] {
            let gz = encode_for(Path::new(name), &doc).unwrap();
            assert_eq!(&gz[..2], &[0x1f, 0x8b], "gzip magic");
            assert!(
                gz.len() * 2 < doc.len(),
                "coordinate text must compress well: {} of {}",
                gz.len(),
                doc.len()
            );
            let mut back = String::new();
            flate2::read::GzDecoder::new(&gz[..])
                .read_to_string(&mut back)
                .unwrap();
            assert_eq!(back, doc, "decompression is the identity");
        }
    }
}
