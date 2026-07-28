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

// ---------------- lines (instanced segments, SDF round caps) ----------------

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) p: vec2<f32>,
    @location(1) a: vec2<f32>,
    @location(2) b: vec2<f32>,
};

@vertex
fn vs_line(
    @builtin(vertex_index) vi: u32,
    @location(0) seg_a: vec2<f32>,
    @location(1) seg_b: vec2<f32>,
) -> LineOut {
    // NaN in either endpoint (run sentinel in the shared point stream)
    // yields NaN corners: the primitive is clipped, drawing nothing.
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

    var out: LineOut;
    out.pos = vec4<f32>(px_to_ndc(corner), 0.0, 1.0);
    out.p = corner;
    out.a = a_px;
    out.b = b_px;
    return out;
}

fn dist_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    return length(p - (a + ab * t));
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    let d = dist_to_segment(in.p, in.a, in.b);
    let alpha = clamp((u.size - d) / u.feather + 1.0, 0.0, 1.0);
    // No discard: zero-alpha fragments blend to nothing, and avoiding the
    // punch-through path keeps the GPU on its fast blend pipeline.
    return vec4<f32>(u.color.rgb, u.color.a * alpha);
}

// ---------------- points (instanced circles) ----------------

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
    let r = u.size + u.feather;
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
    let d = length(in.offset_px);
    let alpha = clamp((u.size - d) / u.feather + 1.0, 0.0, 1.0);
    // darker rim for cartographic legibility
    let rim = clamp((u.size - 1.5 - d) / u.feather + 1.0, 0.0, 1.0);
    let rgb = mix(u.color.rgb * 0.55, u.color.rgb, rim);
    return vec4<f32>(rgb, u.color.a * alpha);
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
