// Shared per-draw uniform (one 256-aligned slot per chunk draw).
struct ChunkU {
    // local -> NDC affine: ndc = local * ab + t
    ab: vec2<f32>,
    t: vec2<f32>,
    color: vec4<f32>,
    // viewport size in physical px
    wh: vec2<f32>,
    // half line width or point radius, in px
    size: f32,
    // AA feather in px
    feather: f32,
    // point marker shape (PointShape::code); ignored by the other passes
    shape: u32,
    // marker border width in px; 0 or a transparent edge = no border
    edge_px: f32,
    // line cap style (LineCap::code): 0 round, 1 square, 2 flat
    cap: u32,
    edge: vec4<f32>,
    // dash pattern in px: (dash1, gap1, dash2, gap2); dash1 < 0 = solid
    dash: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ChunkU;

fn ndc_to_px(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((ndc.x + 1.0) * 0.5 * u.wh.x, (1.0 - ndc.y) * 0.5 * u.wh.y);
}

fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(px.x / u.wh.x * 2.0 - 1.0, 1.0 - px.y / u.wh.y * 2.0);
}

// ---------------- fill ----------------

@vertex
fn vs_fill(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    let ndc = pos * u.ab + u.t;
    return vec4<f32>(ndc, 0.0, 1.0);
}

@fragment
fn fs_fill() -> @location(0) vec4<f32> {
    return u.color;
}

// ---------------- lines (instanced segments, SDF caps and dashes) ----------------

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) p: vec2<f32>,
    @location(1) a: vec2<f32>,
    @location(2) b: vec2<f32>,
    // arc length along the run at `a`, in px (dash phase)
    @location(3) cum_a: f32,
    // bit 0: `a` is a run end; bit 1: `b` is. Interior joints stay round.
    @location(4) @interpolate(flat) ends: u32,
};

@vertex
fn vs_line(
    @builtin(vertex_index) vi: u32,
    @location(0) seg_a: vec2<f32>,
    @location(1) seg_b3: vec3<f32>,
    @location(2) cum_a: f32,
    @location(3) prev: vec3<f32>,
    @location(4) next: vec3<f32>,
) -> LineOut {
    var out: LineOut;
    // An instance that reaches a run sentinel is not a segment. Sentinels
    // are recognised by their negative arc length (`cum_a` is `a`'s,
    // `seg_b3.z` is `b`'s) and the quad is collapsed to a single point,
    // which rasterizes nothing. This used to rely on the sentinel's NaN
    // position clipping the primitive, which fast-math is free to fold
    // away — the sentinel now carries a finite out-of-range position and
    // no NaN reaches the shader at all.
    if (cum_a < 0.0 || seg_b3.z < 0.0) {
        out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.p = vec2<f32>(0.0, 0.0);
        out.a = vec2<f32>(0.0, 0.0);
        out.b = vec2<f32>(0.0, 0.0);
        out.cum_a = 0.0;
        out.ends = 0u;
        return out;
    }
    let seg_b = seg_b3.xy;
    let a_px = ndc_to_px(seg_a * u.ab + u.t);
    let b_px = ndc_to_px(seg_b * u.ab + u.t);
    var dir = b_px - a_px;
    let len = length(dir);
    if (len < 1e-6) {
        dir = vec2<f32>(1.0, 0.0);
    } else {
        dir = dir / len;
    }
    let n = vec2<f32>(-dir.y, dir.x);
    let w = u.size + u.feather;

    // 6 vertices, two triangles covering the extended segment quad.
    var corner: vec2<f32>;
    switch (vi) {
        case 0u: { corner = a_px - dir * w + n * w; }
        case 1u: { corner = a_px - dir * w - n * w; }
        case 2u: { corner = b_px + dir * w + n * w; }
        case 3u: { corner = b_px + dir * w + n * w; }
        case 4u: { corner = a_px - dir * w - n * w; }
        default: { corner = b_px + dir * w - n * w; }
    }

    out.pos = vec4<f32>(px_to_ndc(corner), 0.0, 1.0);
    out.p = corner;
    out.a = a_px;
    out.b = b_px;
    // The stream stores arc length in chunk-local units; the segment's
    // own px/local ratio converts it (the camera transform is isotropic).
    let local_len = length(seg_b - seg_a);
    out.cum_a = cum_a * len / max(local_len, 1e-12);
    // A sentinel neighbour (run break, or the stream's padded ends)
    // marks a run end. Sentinels are recognised by their negative arc
    // length, never by their position.
    out.ends = select(0u, 1u, prev.z < 0.0) | select(0u, 2u, next.z < 0.0);
    return out;
}

// Signed distance past one end of a stroke, in the (along, across)
// frame: `beyond` is px past the end, `s_d` px off the centerline.
fn cap_dist(beyond: f32, s_d: f32, cap: u32, r: f32) -> f32 {
    switch (cap) {
        case 1u: { return max(beyond - r, s_d - r); }  // square
        case 2u: { return max(beyond, s_d - r); }      // flat
        default: { return length(vec2<f32>(max(beyond, 0.0), s_d)) - r; }
    }
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    let ab = in.b - in.a;
    let raw_len = length(ab);
    let len = max(raw_len, 1e-6);
    let dir = ab / len;
    let rel = in.p - in.a;
    let t = dot(rel, dir);
    let s_d = abs(dot(rel, vec2<f32>(-dir.y, dir.x)));
    // Run ends take the user cap; interior joints are always round, so
    // the neighbouring segment's round end fuses with ours at any angle.
    let cap_a = select(0u, u.cap, (in.ends & 1u) != 0u);
    let cap_b = select(0u, u.cap, (in.ends & 2u) != 0u);
    var d = max(cap_dist(-t, s_d, cap_a, u.size),
                cap_dist(t - len, s_d, cap_b, u.size));
    // A segment whose endpoints coincide (repeated coordinates are common
    // in real data) has no direction: with a square or flat cap the two
    // cap distances above then describe the whole extended quad and the
    // segment paints an opaque box. It is a dot, so draw one.
    if (raw_len <= 1e-5) {
        d = length(rel) - u.size;
    }
    if (u.dash.x >= 0.0) {
        // Dash pattern (dash1, gap1, dash2, gap2) tiled along the run's
        // arc length. Each on-interval is a stroke with the user cap at
        // both ends; the nearest of this period's copies and both
        // neighbours wins, so dashes crossing the wrap stay whole.
        let period = max(u.dash.x + u.dash.y + u.dash.z + u.dash.w, 1e-3);
        let l = in.cum_a + t;
        let lf = l - floor(l / period) * period;
        let e1 = u.dash.x;
        let s2 = u.dash.x + u.dash.y;
        let e2 = s2 + u.dash.z;
        var pd = 3.4e38;
        for (var k = -1; k <= 1; k++) {
            let l0 = lf - f32(k) * period;
            pd = min(pd, max(cap_dist(-l0, s_d, u.cap, u.size),
                             cap_dist(l0 - e1, s_d, u.cap, u.size)));
            if (u.dash.z > 0.0 || u.dash.w > 0.0) {
                pd = min(pd, max(cap_dist(s2 - l0, s_d, u.cap, u.size),
                                 cap_dist(l0 - e2, s_d, u.cap, u.size)));
            }
        }
        d = max(d, pd);
    }
    let alpha = clamp(-d / u.feather + 1.0, 0.0, 1.0);
    // No discard: zero-alpha fragments blend to nothing, and avoiding the
    // punch-through path keeps the GPU on its fast blend pipeline.
    return vec4<f32>(u.color.rgb, u.color.a * alpha);
}

// ---------------- points (instanced markers) ----------------

// Marker shapes, sized to the area of the circle of radius u.size they
// replace so the layer keeps its ink weight when the symbol changes.
// Codes and reaches mirror PointShape in src/data/layer.rs.

// How far past u.size the shape extends (square corner, star tip, ...).
fn marker_reach(shape: u32) -> f32 {
    switch (shape) {
        case 1u, 3u: { return 1.2534; }  // square, diamond
        case 2u: { return 1.5535; }      // triangle
        case 4u: { return 1.1000; }      // hexagon
        case 5u: { return 1.4620; }      // star
        default: { return 1.0; }         // circle
    }
}

// Signed distance to a box of half-extents b.
fn sd_box(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

// Equilateral triangle pointing up, r = half of the base width.
fn sd_triangle(pin: vec2<f32>, r: f32) -> f32 {
    let k = sqrt(3.0);
    var p = vec2<f32>(abs(pin.x) - r, -pin.y + r / k);
    if (p.x + k * p.y > 0.0) {
        p = vec2<f32>(p.x - k * p.y, -k * p.x - p.y) * 0.5;
    }
    p.x = p.x - clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

// Regular hexagon with vertices up and down, r = apothem.
fn sd_hexagon(pin: vec2<f32>, r: f32) -> f32 {
    let k = vec3<f32>(-0.8660254, 0.5, 0.5773503);
    var p = abs(vec2<f32>(pin.y, pin.x));
    p = p - 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p = p - vec2<f32>(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

// Five-pointed star, r = outer radius, rf = inner/outer ratio.
fn sd_star5(pin: vec2<f32>, r: f32, rf: f32) -> f32 {
    let k1 = vec2<f32>(0.80901699, -0.58778525);
    let k2 = vec2<f32>(-k1.x, k1.y);
    var p = vec2<f32>(abs(pin.x), -pin.y);
    p = p - 2.0 * max(dot(k1, p), 0.0) * k1;
    p = p - 2.0 * max(dot(k2, p), 0.0) * k2;
    p = vec2<f32>(abs(p.x), p.y - r);
    let ba = rf * vec2<f32>(-k1.y, k1.x) - vec2<f32>(0.0, 1.0);
    let h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

// Distance to the marker's edge: negative inside, in px.
fn sd_marker(p: vec2<f32>, r: f32, shape: u32) -> f32 {
    switch (shape) {
        case 1u: { return sd_box(p, vec2<f32>(r * 0.8862)); }
        case 2u: { return sd_triangle(p, r * 1.3453); }
        case 3u: { return sd_box(vec2<f32>(p.x + p.y, p.x - p.y) * 0.7071,
                                 vec2<f32>(r * 0.8862)); }
        case 4u: { return sd_hexagon(p, r * 0.9525); }
        case 5u: { return sd_star5(p, r * 1.4620, 0.5); }
        default: { return length(p) - r; }
    }
}

struct PointOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) offset_px: vec2<f32>,
};

@vertex
fn vs_point(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2<f32>,
) -> PointOut {
    let c_px = ndc_to_px(center * u.ab + u.t);
    let r = u.size * marker_reach(u.shape) + u.feather;
    var corner: vec2<f32>;
    switch (vi) {
        case 0u: { corner = vec2<f32>(-r, -r); }
        case 1u: { corner = vec2<f32>( r, -r); }
        case 2u: { corner = vec2<f32>(-r,  r); }
        case 3u: { corner = vec2<f32>(-r,  r); }
        case 4u: { corner = vec2<f32>( r, -r); }
        default: { corner = vec2<f32>( r,  r); }
    }
    var out: PointOut;
    out.pos = vec4<f32>(px_to_ndc(c_px + corner), 0.0, 1.0);
    out.offset_px = corner;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4<f32> {
    let d = sd_marker(in.offset_px, u.size, u.shape);
    let alpha = clamp(-d / u.feather + 1.0, 0.0, 1.0);
    if (u.edge.a <= 0.0 || u.edge_px <= 0.0) {
        return vec4<f32>(u.color.rgb, u.color.a * alpha);
    }
    // Border drawn inside the marker, capped to half the radius so a
    // small symbol keeps some fill instead of going solid border color.
    let w = min(u.edge_px, u.size * 0.5);
    let inner = clamp((-d - w) / u.feather + 1.0, 0.0, 1.0);
    let rgb = mix(u.edge.rgb, u.color.rgb, inner);
    let a = mix(u.edge.a, u.color.a, inner);
    return vec4<f32>(rgb, a * alpha);
}

// ---------------- raster tiles ----------------

struct TileOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_tile(@builtin(vertex_index) vi: u32) -> TileOut {
    var corner: vec2<f32>;
    switch (vi) {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(0.0, 1.0); }
        case 4u: { corner = vec2<f32>(1.0, 0.0); }
        default: { corner = vec2<f32>(1.0, 1.0); }
    }
    var out: TileOut;
    out.pos = vec4<f32>(corner * u.ab + u.t, 0.0, 1.0);
    out.uv = corner;
    return out;
}

// Warped tiles: outside Mercator a tile is a curved patch, so it arrives as
// a subdivided mesh whose vertices were projected on the CPU. Positions are
// world offsets from the tile's own origin (same scheme as vector chunks, so
// f32 keeps its precision at deep zoom).
@vertex
fn vs_tile_mesh(
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
) -> TileOut {
    var out: TileOut;
    out.pos = vec4<f32>(pos * u.ab + u.t, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@group(1) @binding(0) var tile_tex: texture_2d<f32>;
@group(1) @binding(1) var tile_samp: sampler;

@fragment
fn fs_tile(in: TileOut) -> @location(0) vec4<f32> {
    let c = textureSample(tile_tex, tile_samp, in.uv);
    return vec4<f32>(c.rgb, c.a * u.color.a);
}

// ---------------- layer fill composite ----------------

// A layer's fills are drawn opaque into an offscreen target that was
// cleared transparent, so the resolve leaves *premultiplied* texels: an
// antialiased edge covering 40% of a pixel comes back as 0.4·C with
// alpha 0.4. Scaling only the alpha (what `fs_tile` does for a raster
// tile, whose texels are unmultiplied) would then blend 0.4·C as if it
// were the full colour and draw a dark fringe along every edge. Both
// channels are scaled here instead, and the pipeline blends
// premultiplied.
//
// The target is rounded up to a 128 px grid so a window resize does not
// reallocate it every frame; `u.edge.xy` carries viewport/texture, the
// fraction of it the fills were rendered into.
@vertex
fn vs_composite(@builtin(vertex_index) vi: u32) -> TileOut {
    var corner: vec2<f32>;
    switch (vi) {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(0.0, 1.0); }
        case 4u: { corner = vec2<f32>(1.0, 0.0); }
        default: { corner = vec2<f32>(1.0, 1.0); }
    }
    var out: TileOut;
    out.pos = vec4<f32>(corner * u.ab + u.t, 0.0, 1.0);
    out.uv = corner * u.edge.xy;
    return out;
}

@fragment
fn fs_composite(in: TileOut) -> @location(0) vec4<f32> {
    return textureSample(tile_tex, tile_samp, in.uv) * u.color.a;
}
