use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};
use wgpu::util::DeviceExt;

use crate::data::geometry::{ChunkMesh, LINE_LODS_TOTAL};
use crate::map::camera::Camera;
use crate::map::tiles::{TileDrawCmd, TileKey, TileUpload};

pub const MSAA_SAMPLES: u32 = 4;
const UNIFORM_STRIDE: u64 = 256;

/// Style values resolved to shader units for one layer draw.
#[derive(Clone)]
pub struct DrawStyle {
    pub fill_color: [f32; 4],
    pub line_color: [f32; 4],
    pub point_color: [f32; 4],
    pub line_half_width_px: f32,
    pub point_radius_px: f32,
    /// Data-driven styling: per-bin RGB (chunks carry their bin); alpha
    /// channels come from the colors above. None = uniform colors.
    pub bin_colors: Option<Arc<[[f32; 3]; crate::data::layer::STYLE_BINS]>>,
    /// Bitmask of style bins hidden from the map (legend toggles).
    /// Only honored while `bin_colors` is set: without data styling
    /// every chunk reports bin 0 and the mask would hide the layer.
    pub hidden_bins: u64,
}

impl DrawStyle {
    fn bin_hidden(&self, bin: u8) -> bool {
        self.bin_colors.is_some() && self.hidden_bins & (1u64 << bin) != 0
    }
}

impl DrawStyle {
    /// Fill/point RGB for a chunk (uniform color when unstyled).
    fn rgb_for(&self, bin: u8, base: [f32; 4]) -> [f32; 3] {
        match &self.bin_colors {
            Some(lut) => lut[bin as usize % lut.len()],
            None => [base[0], base[1], base[2]],
        }
    }
}

/// One vector layer section's draw request for this frame.
pub struct LayerDraw {
    pub key: (u64, u64), // (section key, generation)
    pub chunks: Arc<Vec<ChunkMesh>>,
    pub style: DrawStyle,
    /// Sections of one logical layer share a group: their fills render into
    /// one offscreen texture and composite once (uniform opacity).
    /// Consecutive draws with the same group are batched together.
    pub composite_group: u64,
}

/// Everything the paint callback needs for one frame.
pub struct MapCallback {
    pub camera: Camera,
    pub viewport_px: [f32; 2],
    /// Basemap opacity, 0..=1. Applied to every tile, warped or not.
    pub tile_opacity: f32,
    pub tile_draws: Vec<TileDrawCmd>,
    pub tile_uploads: Vec<TileUpload>,
    pub alive_tiles: HashSet<TileKey>,
    /// All (layer id, generation) keys whose GPU buffers must be kept,
    /// including layers not drawn this frame (hidden ones). CPU mesh copies
    /// are freed after upload, so eviction would be irreversible.
    pub alive_layers: HashSet<(u64, u64)>,
    pub layers: Vec<LayerDraw>,
    #[allow(dead_code)]
    pub background: [f32; 4],
}

struct ChunkGpu {
    origin: [f64; 2],
    /// Style bin (0 when the layer is not data-styled).
    bin: u8,
    /// Oversized-feature chunk: its fills draw before all normal fills.
    underlay: bool,
    /// World-space bounds of this chunk's vertices, for viewport culling.
    bounds_world: [f64; 4],
    fill_vbuf: Option<wgpu::Buffer>,
    fill_ibuf: Option<wgpu::Buffer>,
    /// Uint16 when the chunk has ≤ 65536 fill vertices (half the index
    /// bytes — most chunks), Uint32 otherwise.
    fill_index_format: wgpu::IndexFormat,
    fill_index_count: u32,
    /// Index 0 = full detail, 1.. = simplified LODs (see LINE_LOD_TOLERANCE).
    /// Each entry: buffer + feature-size prefix index (see LineLod).
    line_bufs: [Option<(wgpu::Buffer, Vec<(f32, u32)>)>; LINE_LODS_TOTAL],
    point_buf: Option<wgpu::Buffer>,
    point_count: u32,
}

/// Which line LOD to draw at a given camera zoom: full detail above z15.5,
/// otherwise the level whose tolerance is ~1 px at this zoom.
fn line_lod_for_zoom(zoom: f64) -> usize {
    if zoom > 15.0 {
        0
    } else {
        // Half a zoom coarser than the 1 px-error level: segments stay
        // longer on screen (sub-pixel triangles are the GPU bottleneck)
        // for at most ~2 px simplification error.
        ((16.5 - zoom).round() as usize).clamp(1, LINE_LODS_TOTAL - 1)
    }
}

/// Outlines only draw for features at least this many pixels across;
/// smaller features are represented by their fill alone (stroking a
/// feature a few pixels wide just darkens it into outline color).
const MIN_OUTLINE_FEATURE_PX: f64 = 4.0;

/// Segments to draw from a size-sorted LOD buffer given the current scale.
fn line_count_for_scale(index: &[(f32, u32)], scale: f64) -> u32 {
    let cutoff = MIN_OUTLINE_FEATURE_PX / scale;
    let mut count = 0u32;
    for &(threshold, prefix) in index {
        if (threshold as f64) >= cutoff {
            count = prefix;
        } else {
            break;
        }
    }
    count
}

struct LayerGpu {
    chunks: Vec<ChunkGpu>,
}

/// Convert a LOD's 16 B/segment `[x0,y0,x1,y1]` list into a shared point
/// stream (8 B/point): consecutive segments that chain (`prev.b == cur.a`,
/// the common case — ring order survives the stable extent sort) share
/// their joint point, and a NaN sentinel separates runs (its quads collapse
/// to nothing in the vertex shader). Instance `i` reads `points[i]` from
/// slot 0 and `points[i+1]` from slot 1 (same buffer bound at +8 bytes),
/// so segment counts in the size index translate to instance counts here.
/// Net: ~9 B/segment instead of 16.
fn line_point_stream(
    lod: &crate::data::geometry::LineLod,
) -> (Vec<[f32; 2]>, Vec<(f32, u32)>) {
    let segs = &lod.segments;
    if segs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut points: Vec<[f32; 2]> = Vec::with_capacity(segs.len() + segs.len() / 8);
    // Instance index of each segment (transient; only the size-index
    // prefixes need translating).
    let mut seg_instance: Vec<u32> = Vec::with_capacity(segs.len());
    for s in segs {
        let (a, b) = ([s[0], s[1]], [s[2], s[3]]);
        // A NaN sentinel never equals `a`, so it breaks runs naturally.
        if points.last() != Some(&a) {
            if !points.is_empty() {
                points.push([f32::NAN, f32::NAN]);
            }
            points.push(a);
        }
        seg_instance.push((points.len() - 1) as u32);
        points.push(b);
    }
    let total_instances = (points.len() - 1) as u32;
    let index = lod
        .size_index
        .iter()
        .map(|&(threshold, prefix)| {
            let instances = seg_instance
                .get(prefix as usize)
                .copied()
                .unwrap_or(total_instances);
            (threshold, instances)
        })
        .collect();
    (points, index)
}

struct TileGpu {
    bind_group: wgpu::BindGroup,
    _texture: wgpu::Texture,
}

/// Reuse a buffer when it is big enough, otherwise allocate the next power
/// of two. Warped tile meshes are rebuilt every frame but barely change
/// size, so this settles after the first few.
fn grow_and_write(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    existing: Option<wgpu::Buffer>,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    let need = bytes.len() as u64;
    let buf = match existing {
        Some(b) if b.size() >= need => b,
        _ => device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: need.next_power_of_two().max(4096),
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
    };
    queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Offscreen fill target for one layer (resolved, viewport-sized).
struct CompositeTarget {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: [u32; 2],
    _texture: wgpu::Texture,
}

/// Persistent GPU state stored in egui's CallbackResources.
pub struct MapResources {
    fill_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    tile_pipeline: wgpu::RenderPipeline,
    /// Warped tiles: same texture and fragment shader, but vertices come
    /// from a buffer instead of being generated from the vertex index.
    tile_mesh_pipeline: wgpu::RenderPipeline,
    /// Per-frame scratch holding every warped tile's mesh, grown as needed.
    tile_mesh_vbuf: Option<wgpu::Buffer>,
    tile_mesh_ibuf: Option<wgpu::Buffer>,
    chunk_bgl: wgpu::BindGroupLayout,
    tile_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    uniform_capacity: u64,
    layers: HashMap<(u64, u64), LayerGpu>,
    tiles: HashMap<TileKey, TileGpu>,
    /// Per-group offscreen fill textures (fills render opaque here, then
    /// composite once at layer opacity so overlaps don't darken).
    composites: HashMap<u64, CompositeTarget>,
    /// Shared MSAA scratch for offscreen fill passes.
    scratch_msaa: Option<(wgpu::TextureView, [u32; 2])>,
    target_format: wgpu::TextureFormat,
    /// Draw list built in prepare, executed in paint.
    draw_list: Vec<DrawCmd>,
}

impl MapResources {
    /// Whether GPU buffers exist for this (layer id, generation).
    pub fn has_layer_uploaded(&self, key: (u64, u64)) -> bool {
        self.layers.contains_key(&key)
    }
}

enum DrawCmd {
    Tile {
        key: TileKey,
        uoffset: u32,
    },
    /// A tile warped into the display projection, indexed out of the shared
    /// per-frame mesh buffers.
    TileMesh {
        key: TileKey,
        uoffset: u32,
        index_start: u32,
        index_count: u32,
        base_vertex: i32,
    },
    /// Full-viewport quad compositing a group's offscreen fill texture at
    /// the layer's fill opacity (uniform tone regardless of feature overlap).
    Composite {
        group: u64,
        uoffset: u32,
    },
    Line {
        layer: (u64, u64),
        chunk: usize,
        uoffset: u32,
        /// Which LOD buffer to draw (0 = full detail).
        lod: usize,
        /// Size-cutoff prefix count within that buffer.
        count: u32,
    },
    Point {
        layer: (u64, u64),
        chunk: usize,
        uoffset: u32,
        /// Density-capped number of instances to draw this frame.
        count: u32,
    },
}

/// Target point coverage: expected number of point disks overlapping each
/// on-screen pixel before the renderer stops adding more. Chunks whose
/// points exceed this draw a uniform random prefix (instances are shuffled
/// at build time); zooming in raises the cap until every point draws.
const POINT_COVERAGE: f32 = 10.0;

fn point_draw_count(chunk_px: [f32; 2], radius_px: f32, total: u32) -> u32 {
    let r = radius_px.max(0.5) + 1.0;
    // No edge padding: with fine chunks a padding term dominates the true
    // area and inflates the world-view draw count by an order of magnitude.
    let area = chunk_px[0] * chunk_px[1];
    let cap = (POINT_COVERAGE * area / (std::f32::consts::PI * r * r)).ceil();
    total.min((cap as u32).max(4))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkUniform {
    ab: [f32; 2],
    t: [f32; 2],
    color: [f32; 4],
    wh: [f32; 2],
    size: f32,
    feather: f32,
}

impl MapResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // wgpu treats validation errors as fatal by default; a bad buffer or
        // texture from one layer must not take down the whole app.
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            log::error!("wgpu error (rendering may be incomplete): {e}");
        }));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("map shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let chunk_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk uniform bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ChunkUniform>() as u64),
                },
                count: None,
            }],
        });

        let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile texture bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let vector_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector pl"),
            bind_group_layouts: &[Some(&chunk_bgl)],
            immediate_size: 0,
        });
        let tile_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile pl"),
            bind_group_layouts: &[Some(&chunk_bgl), Some(&tile_bgl)],
            immediate_size: 0,
        });

        let target = [Some(wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let make_pipeline = |label: &str,
                             layout: &wgpu::PipelineLayout,
                             vs: &str,
                             fs: &str,
                             buffers: &[wgpu::VertexBufferLayout<'_>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &target,
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let fill_pipeline = make_pipeline(
            "fill",
            &vector_layout,
            "vs_fill",
            "fs_fill",
            &[wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2],
            }],
        );
        let line_pipeline = make_pipeline(
            "line",
            &vector_layout,
            "vs_line",
            "fs_line",
            // One shared point stream bound twice, slot 1 offset by one
            // point: instance i reads its segment as points[i], points[i+1].
            &[
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![1 => Float32x2],
                },
            ],
        );
        let point_pipeline = make_pipeline(
            "point",
            &vector_layout,
            "vs_point",
            "fs_point",
            &[wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2],
            }],
        );
        let tile_pipeline = make_pipeline("tile", &tile_layout, "vs_tile", "fs_tile", &[]);
        let tile_mesh_pipeline = make_pipeline(
            "tile mesh",
            &tile_layout,
            "vs_tile_mesh",
            "fs_tile",
            &[wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
        );

        // Reprojection minifies a tile hard along one axis (roughly 1/cos²φ
        // against a cylindrical equal-area, so 4:1 at 60°N), which is what
        // anisotropic filtering exists for. It needs a mip chain and linear
        // filtering on every axis to have anything to work with.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tile sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let uniform_capacity = 1024 * UNIFORM_STRIDE;
        let (uniform_buf, uniform_bg) =
            Self::make_uniform_buffer(device, &chunk_bgl, uniform_capacity);

        Self {
            fill_pipeline,
            line_pipeline,
            point_pipeline,
            tile_pipeline,
            tile_mesh_pipeline,
            tile_mesh_vbuf: None,
            tile_mesh_ibuf: None,
            chunk_bgl,
            tile_bgl,
            sampler,
            uniform_buf,
            uniform_bg,
            uniform_capacity,
            layers: HashMap::new(),
            tiles: HashMap::new(),
            composites: HashMap::new(),
            scratch_msaa: None,
            target_format,
            draw_list: Vec::new(),
        }
    }

    fn ensure_composite_target(
        &mut self,
        device: &wgpu::Device,
        key: u64,
        size: [u32; 2],
    ) {
        if self
            .composites
            .get(&key)
            .map(|t| t.size == size)
            .unwrap_or(false)
        {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer composite"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer composite bg"),
            layout: &self.tile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.composites.insert(
            key,
            CompositeTarget {
                view,
                bind_group,
                size,
                _texture: texture,
            },
        );
    }

    fn ensure_scratch_msaa(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self
            .scratch_msaa
            .as_ref()
            .map(|(_, s)| *s == size)
            .unwrap_or(false)
        {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("composite msaa scratch"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.scratch_msaa = Some((texture.create_view(&Default::default()), size));
    }

    fn make_uniform_buffer(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        capacity: u64,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk uniforms"),
            size: capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk uniforms bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ChunkUniform>() as u64),
                }),
            }],
        });
        (buf, bg)
    }

    fn upload_tile(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, up: &TileUpload) {
        let Some(base) = up.mips.first() else { return };
        let (w, h) = (base.w, base.h);
        let mips = &up.mips;
        // Tile PNGs are sRGB-encoded. Whether to declare that depends on the
        // target: an sRGB target re-encodes on write, so the sampler should
        // decode; a plain one does not, and egui writes gamma-encoded values
        // into it directly, so the tile must pass through untouched. Getting
        // this wrong decodes without ever re-encoding, which barely shows on
        // a light basemap and crushes a dark one to black.
        let format = if self.target_format.is_srgb() {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, m) in mips.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &m.px,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * m.w),
                    rows_per_image: Some(m.h),
                },
                wgpu::Extent3d {
                    width: m.w,
                    height: m.h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile bg"),
            layout: &self.tile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.tiles.insert(
            up.key,
            TileGpu {
                bind_group,
                _texture: texture,
            },
        );
    }

    fn ensure_layer(&mut self, device: &wgpu::Device, draw: &LayerDraw) {
        if self.layers.contains_key(&draw.key) {
            return;
        }
        let chunks = draw
            .chunks
            .iter()
            .map(|c| {
                let mk_buf = |data: &[u8], usage, label: &str| {
                    if data.is_empty() {
                        None
                    } else {
                        Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(label),
                            contents: data,
                            usage,
                        }))
                    }
                };
                ChunkGpu {
                    origin: c.origin,
                    bin: c.bin,
                    underlay: c.underlay,
                    bounds_world: [
                        c.origin[0] + c.bounds_local[0] as f64,
                        c.origin[1] + c.bounds_local[1] as f64,
                        c.origin[0] + c.bounds_local[2] as f64,
                        c.origin[1] + c.bounds_local[3] as f64,
                    ],
                    fill_vbuf: mk_buf(
                        bytemuck::cast_slice(&c.fill_vertices),
                        wgpu::BufferUsages::VERTEX,
                        "fill verts",
                    ),
                    fill_ibuf: if c.fill_vertices.len() <= (u16::MAX as usize) + 1 {
                        // Half-width indices; pad to a 4-byte buffer size.
                        let mut idx: Vec<u16> =
                            c.fill_indices.iter().map(|&i| i as u16).collect();
                        if idx.len() % 2 == 1 {
                            idx.push(0);
                        }
                        mk_buf(
                            bytemuck::cast_slice(&idx),
                            wgpu::BufferUsages::INDEX,
                            "fill idx u16",
                        )
                    } else {
                        mk_buf(
                            bytemuck::cast_slice(&c.fill_indices),
                            wgpu::BufferUsages::INDEX,
                            "fill idx",
                        )
                    },
                    fill_index_format: if c.fill_vertices.len() <= (u16::MAX as usize) + 1 {
                        wgpu::IndexFormat::Uint16
                    } else {
                        wgpu::IndexFormat::Uint32
                    },
                    fill_index_count: c.fill_indices.len() as u32,
                    line_bufs: {
                        let mk_lines = |lod: &crate::data::geometry::LineLod| {
                            let (points, index) = line_point_stream(lod);
                            mk_buf(
                                bytemuck::cast_slice(&points),
                                wgpu::BufferUsages::VERTEX,
                                "line points",
                            )
                            .map(|b| (b, index))
                        };
                        let mut bufs: [Option<(wgpu::Buffer, Vec<(f32, u32)>)>;
                            LINE_LODS_TOTAL] = Default::default();
                        for k in 0..LINE_LODS_TOTAL {
                            let target = c.line_alias[k] as usize;
                            bufs[k] = if target == k {
                                mk_lines(&c.lines[k])
                            } else {
                                // Aliased: share the finer level's buffer.
                                bufs[target].clone()
                            };
                        }
                        bufs
                    },
                    point_buf: mk_buf(
                        bytemuck::cast_slice(&c.point_instances),
                        wgpu::BufferUsages::VERTEX,
                        "points",
                    ),
                    point_count: c.point_instances.len() as u32,
                }
            })
            .collect();
        self.layers.insert(draw.key, LayerGpu { chunks });
    }
}

impl egui_wgpu::CallbackTrait for MapCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let res: &mut MapResources = resources.get_mut().expect("MapResources installed");

        // Upload new tiles, evict dead ones and dead layer generations.
        for up in &self.tile_uploads {
            res.upload_tile(device, queue, up);
        }
        res.tiles.retain(|k, _| self.alive_tiles.contains(k));
        res.layers
            .retain(|k, _| self.alive_layers.contains(k) || self.layers.iter().any(|l| l.key == *k));
        res.composites
            .retain(|g, _| self.layers.iter().any(|l| l.composite_group == *g));
        for draw in &self.layers {
            res.ensure_layer(device, draw);
        }
        let vp_size = [
            (self.viewport_px[0].max(1.0)) as u32,
            (self.viewport_px[1].max(1.0)) as u32,
        ];

        // Build uniforms + draw list.
        let cam = &self.camera;
        let scale = cam.scale();
        let [vw, vh] = self.viewport_px;
        let ndc_per_world_x = scale * 2.0 / vw as f64;
        let ndc_per_world_y = scale * 2.0 / vh as f64;
        // Viewport rect in world units, padded for line widths / point radii.
        let margin = 32.0 / scale;
        let view = [
            cam.center[0] - vw as f64 * 0.5 / scale - margin,
            cam.center[1] - vh as f64 * 0.5 / scale - margin,
            cam.center[0] + vw as f64 * 0.5 / scale + margin,
            cam.center[1] + vh as f64 * 0.5 / scale + margin,
        ];
        let visible = |b: &[f64; 4]| -> bool {
            b[0] <= view[2] && b[2] >= view[0] && b[1] <= view[3] && b[3] >= view[1]
        };

        let mut uniforms: Vec<u8> = Vec::new();
        let mut draw_list: Vec<DrawCmd> = Vec::new();
        // group -> fill draws collected across that group's sections.
        let mut offscreen_jobs: Vec<(u64, Vec<((u64, u64), usize, u32)>)> = Vec::new();
        let push_uniform = |uniforms: &mut Vec<u8>,
                                origin: [f64; 2],
                                world_scale: [f64; 2],
                                color: [f32; 4],
                                size: f32|
         -> u32 {
            let offset = uniforms.len() as u32;
            let u = ChunkUniform {
                ab: [
                    (world_scale[0] * ndc_per_world_x) as f32,
                    (-world_scale[1] * ndc_per_world_y) as f32,
                ],
                t: [
                    ((origin[0] - cam.center[0]) * ndc_per_world_x) as f32,
                    (-(origin[1] - cam.center[1]) * ndc_per_world_y) as f32,
                ],
                color,
                wh: [vw, vh],
                size,
                feather: 1.0,
            };
            uniforms.extend_from_slice(bytemuck::bytes_of(&u));
            uniforms.resize(uniforms.len().next_multiple_of(UNIFORM_STRIDE as usize), 0);
            offset
        };

        // Tiles first (under vector data). Warped ones carry a mesh and are
        // packed into one shared vertex/index buffer for the frame.
        let mut mesh_verts: Vec<[f32; 4]> = Vec::new();
        let mut mesh_indices: Vec<u16> = Vec::new();
        for tile in &self.tile_draws {
            if !res.tiles.contains_key(&tile.key) {
                continue;
            }
            match &tile.mesh {
                Some(m) => {
                    let base_vertex = mesh_verts.len() as i32;
                    let index_start = mesh_indices.len() as u32;
                    mesh_verts.extend_from_slice(&m.verts);
                    mesh_indices.extend_from_slice(&m.indices);
                    let uoffset = push_uniform(
                        &mut uniforms,
                        m.origin,
                        [1.0, 1.0],
                        [1.0, 1.0, 1.0, self.tile_opacity],
                        0.0,
                    );
                    draw_list.push(DrawCmd::TileMesh {
                        key: tile.key,
                        uoffset,
                        index_start,
                        index_count: m.indices.len() as u32,
                        base_vertex,
                    });
                }
                None => {
                    let r = tile.world_rect;
                    let uoffset = push_uniform(
                        &mut uniforms,
                        [r[0], r[1]],
                        [r[2] - r[0], r[3] - r[1]],
                        [1.0, 1.0, 1.0, self.tile_opacity],
                        0.0,
                    );
                    draw_list.push(DrawCmd::Tile {
                        key: tile.key,
                        uoffset,
                    });
                }
            }
        }
        if !mesh_indices.is_empty() {
            // write_buffer needs a multiple of 4 bytes; cells contribute six
            // indices each, so this only ever pads a malformed mesh.
            if !mesh_indices.len().is_multiple_of(2) {
                mesh_indices.push(0);
            }
            res.tile_mesh_vbuf = Some(grow_and_write(
                device,
                queue,
                res.tile_mesh_vbuf.take(),
                bytemuck::cast_slice(&mesh_verts),
                wgpu::BufferUsages::VERTEX,
                "tile mesh verts",
            ));
            res.tile_mesh_ibuf = Some(grow_and_write(
                device,
                queue,
                res.tile_mesh_ibuf.take(),
                bytemuck::cast_slice(&mesh_indices),
                wgpu::BufferUsages::INDEX,
                "tile mesh indices",
            ));
        }

        // Vector layers: fills (grouped per composite_group), then lines,
        // then points per section. Sections of a group are consecutive.
        let no_fills = cfg!(test) && std::env::var("GEOPQ_NO_FILLS").is_ok();
        let mut li = 0usize;
        while li < self.layers.len() {
            let group = self.layers[li].composite_group;
            let group_end = self.layers[li..]
                .iter()
                .position(|l| l.composite_group != group)
                .map(|p| li + p)
                .unwrap_or(self.layers.len());
            let group_layers = &self.layers[li..group_end];
            let s = &group_layers[0].style;

            // Fills render opaque into the group's offscreen texture and
            // composite once at the layer's fill opacity: overlapping
            // features (stacked condo parcels etc.) don't darken.
            let mut fill_cmds: Vec<((u64, u64), usize, u32)> = Vec::new();
            // Two passes: underlay chunks (oversized features) first, so
            // giant polygons draw below normal fills across ALL sections
            // of the group, not only their own.
            for underlay_pass in [true, false] {
                for layer in group_layers {
                    let Some(gpu) = res.layers.get(&layer.key) else {
                        continue;
                    };
                    for (ci, chunk) in gpu.chunks.iter().enumerate() {
                        if chunk.underlay != underlay_pass || layer.style.bin_hidden(chunk.bin) {
                            continue;
                        }
                        if !no_fills && chunk.fill_index_count > 0 && visible(&chunk.bounds_world) {
                            let rgb = layer.style.rgb_for(chunk.bin, s.fill_color);
                            let opaque = [rgb[0], rgb[1], rgb[2], 1.0];
                            let uoffset =
                                push_uniform(&mut uniforms, chunk.origin, [1.0, 1.0], opaque, 0.0);
                            fill_cmds.push((layer.key, ci, uoffset));
                        }
                    }
                }
            }
            if !fill_cmds.is_empty() && s.fill_color[3] > 0.0 {
                // Full-viewport quad: corner (0,0)..(1,1) -> NDC.
                let quad_offset = uniforms.len() as u32;
                let u = ChunkUniform {
                    ab: [2.0, -2.0],
                    t: [-1.0, 1.0],
                    color: [1.0, 1.0, 1.0, s.fill_color[3]],
                    wh: [vw, vh],
                    size: 0.0,
                    feather: 1.0,
                };
                uniforms.extend_from_slice(bytemuck::bytes_of(&u));
                uniforms.resize(uniforms.len().next_multiple_of(UNIFORM_STRIDE as usize), 0);
                offscreen_jobs.push((group, fill_cmds));
                draw_list.push(DrawCmd::Composite {
                    group,
                    uoffset: quad_offset,
                });
            }

            for layer in group_layers {
            let Some(gpu) = res.layers.get(&layer.key) else {
                continue;
            };
            let s = &layer.style;
            if s.line_color[3] <= 0.0 {
                continue;
            }
            let lod = line_lod_for_zoom(cam.zoom);
            for (ci, chunk) in gpu.chunks.iter().enumerate() {
                if s.bin_hidden(chunk.bin) {
                    continue;
                }
                let count = chunk.line_bufs[lod]
                    .as_ref()
                    .map(|(_, index)| line_count_for_scale(index, scale))
                    .unwrap_or(0);
                if count > 0 && visible(&chunk.bounds_world) {
                    #[cfg(test)]
                    if std::env::var("GEOPQ_DEBUG_DRAWS").is_ok() {
                        eprintln!("chunk {ci}: lod {lod} count {count}");
                    }
                    let line_color = match &s.bin_colors {
                        Some(lut) => {
                            let c = lut[chunk.bin as usize % lut.len()];
                            // Darkened ramp color keeps outlines readable.
                            [c[0] * 0.65, c[1] * 0.65, c[2] * 0.65, s.line_color[3]]
                        }
                        None => s.line_color,
                    };
                    let uoffset = push_uniform(
                        &mut uniforms,
                        chunk.origin,
                        [1.0, 1.0],
                        line_color,
                        s.line_half_width_px,
                    );
                    draw_list.push(DrawCmd::Line {
                        layer: layer.key,
                        chunk: ci,
                        uoffset,
                        lod,
                        count,
                    });
                }
            }
            for (ci, chunk) in gpu.chunks.iter().enumerate() {
                if chunk.point_count > 0
                    && !s.bin_hidden(chunk.bin)
                    && visible(&chunk.bounds_world)
                {
                    let b = &chunk.bounds_world;
                    let chunk_px = [
                        ((b[2] - b[0]) * scale) as f32,
                        ((b[3] - b[1]) * scale) as f32,
                    ];
                    let count =
                        point_draw_count(chunk_px, s.point_radius_px, chunk.point_count);
                    let rgb = s.rgb_for(chunk.bin, s.point_color);
                    let uoffset = push_uniform(
                        &mut uniforms,
                        chunk.origin,
                        [1.0, 1.0],
                        [rgb[0], rgb[1], rgb[2], s.point_color[3]],
                        s.point_radius_px,
                    );
                    draw_list.push(DrawCmd::Point {
                        layer: layer.key,
                        chunk: ci,
                        uoffset,
                        count,
                    });
                }
            }
            }
            li = group_end;
        }

        // Grow uniform buffer if needed, then write.
        let needed = (uniforms.len() as u64).max(UNIFORM_STRIDE);
        if needed > res.uniform_capacity {
            res.uniform_capacity = needed.next_power_of_two();
            let (buf, bg) =
                MapResources::make_uniform_buffer(device, &res.chunk_bgl, res.uniform_capacity);
            res.uniform_buf = buf;
            res.uniform_bg = bg;
        }
        if !uniforms.is_empty() {
            queue.write_buffer(&res.uniform_buf, 0, &uniforms);
        }
        res.draw_list = draw_list;

        // Offscreen fill passes (submitted before the egui render pass).
        if offscreen_jobs.is_empty() {
            return Vec::new();
        }
        res.ensure_scratch_msaa(device, vp_size);
        for (group, _) in &offscreen_jobs {
            res.ensure_composite_target(device, *group, vp_size);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer fill composites"),
        });
        let (msaa_view, _) = res.scratch_msaa.as_ref().expect("scratch msaa");
        for (group, cmds) in &offscreen_jobs {
            let target = res.composites.get(group).expect("composite target");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("layer fills"),
                multiview_mask: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&target.view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&res.fill_pipeline);
            for (section_key, ci, uoffset) in cmds {
                let Some(gpu) = res.layers.get(section_key) else {
                    continue;
                };
                let c = &gpu.chunks[*ci];
                let (Some(vb), Some(ib)) = (&c.fill_vbuf, &c.fill_ibuf) else {
                    continue;
                };
                pass.set_bind_group(0, &res.uniform_bg, &[*uoffset]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), c.fill_index_format);
                pass.draw_indexed(0..c.fill_index_count, 0, 0..1);
            }
        }
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let res: &MapResources = resources.get().expect("MapResources installed");

        for cmd in &res.draw_list {
            match cmd {
                DrawCmd::Tile { key, uoffset } => {
                    let Some(tile) = res.tiles.get(key) else {
                        continue;
                    };
                    pass.set_pipeline(&res.tile_pipeline);
                    pass.set_bind_group(0, &res.uniform_bg, &[*uoffset]);
                    pass.set_bind_group(1, &tile.bind_group, &[]);
                    pass.draw(0..6, 0..1);
                }
                DrawCmd::TileMesh {
                    key,
                    uoffset,
                    index_start,
                    index_count,
                    base_vertex,
                } => {
                    let (Some(tile), Some(vb), Some(ib)) = (
                        res.tiles.get(key),
                        res.tile_mesh_vbuf.as_ref(),
                        res.tile_mesh_ibuf.as_ref(),
                    ) else {
                        continue;
                    };
                    pass.set_pipeline(&res.tile_mesh_pipeline);
                    pass.set_bind_group(0, &res.uniform_bg, &[*uoffset]);
                    pass.set_bind_group(1, &tile.bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint16);
                    pass.draw_indexed(
                        *index_start..*index_start + *index_count,
                        *base_vertex,
                        0..1,
                    );
                }
                DrawCmd::Composite { group, uoffset } => {
                    let Some(target) = res.composites.get(group) else {
                        continue;
                    };
                    pass.set_pipeline(&res.tile_pipeline);
                    pass.set_bind_group(0, &res.uniform_bg, &[*uoffset]);
                    pass.set_bind_group(1, &target.bind_group, &[]);
                    pass.draw(0..6, 0..1);
                }
                DrawCmd::Line {
                    layer,
                    chunk,
                    uoffset,
                    lod,
                    count,
                } => {
                    let Some(l) = res.layers.get(layer) else { continue };
                    let c = &l.chunks[*chunk];
                    let Some((buf, _)) = &c.line_bufs[*lod] else {
                        continue;
                    };
                    pass.set_pipeline(&res.line_pipeline);
                    pass.set_bind_group(0, &res.uniform_bg, &[*uoffset]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.set_vertex_buffer(1, buf.slice(8..));
                    pass.draw(0..6, 0..*count);
                }
                DrawCmd::Point {
                    layer,
                    chunk,
                    uoffset,
                    count,
                } => {
                    let Some(l) = res.layers.get(layer) else { continue };
                    let c = &l.chunks[*chunk];
                    let Some(buf) = &c.point_buf else { continue };
                    pass.set_pipeline(&res.point_pipeline);
                    pass.set_bind_group(0, &res.uniform_bg, &[*uoffset]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..6, 0..*count);
                }
            }
        }
    }
}


/// Device + queue from any available adapter, or None on headless
/// machines without a GPU driver — render tests self-skip then (CI
/// installs a software rasterizer for real coverage).
#[cfg(test)]
pub(crate) fn test_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::geometry::MeshBuilder;
    use egui_wgpu::CallbackTrait;

    /// Render the box path of a real file to a PNG so it can be looked at.
    ///
    /// A build that reports the right feature count can still draw
    /// nothing, and the counts cannot tell you which. Run:
    /// `GEOPQ_FILE=... GEOPQ_GROUPS=40 cargo test --release
    /// boxes_render_to_png -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a local file; writes a PNG to inspect"]
    fn boxes_render_to_png() {
        let Ok(path) = std::env::var("GEOPQ_FILE") else {
            eprintln!("set GEOPQ_FILE");
            return;
        };
        let groups: usize = std::env::var("GEOPQ_GROUPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(40);
        let (store, crs, _info, _rg) =
            crate::data::loader::open_store_for_test(&path.clone().into()).unwrap();
        let total = store.rg_starts().len() - 1;
        // GEOPQ_DISPLAY=auto uses the projection the app adopts on open
        // (the data CRS for a projected file), not the world default.
        let display = if std::env::var("GEOPQ_DISPLAY").is_ok_and(|v| v == "auto") {
            crate::data::crs::DisplayCrs::auto_for(&crs, None)
                .unwrap_or_else(crate::data::crs::DisplayCrs::hobo_dyer)
        } else {
            crate::data::crs::DisplayCrs::hobo_dyer()
        };
        eprintln!("display: {}", display.name);
        // GEOPQ_BOXES=0 builds the same groups from geometry: the control
        // for "is it the box path or the harness".
        let boxes = std::env::var("GEOPQ_BOXES").map_or(true, |v| v == "1");
        let outlines = std::env::var("GEOPQ_OUTLINES").is_ok_and(|v| v == "1");
        // GEOPQ_STYLE=1 applies whatever colour map the schema announces,
        // exactly as a plain open does — the app never draws these
        // unstyled, so neither should the picture we judge it by.
        let style_by = std::env::var("GEOPQ_STYLE")
            .is_ok_and(|v| v == "1")
            .then(|| crate::data::colormap::schema_style(&store.style_columns()))
            .flatten();
        let style_sel = style_by
            .as_ref()
            .and_then(|sb| crate::data::loader::resolve_style(&store, sb));
        eprintln!(
            "style: {}",
            style_by
                .as_ref()
                .map_or("none".into(), |sb| format!("{} ({:?})", sb.column, style_sel.is_some()))
        );
        let (geometry, rows, _bad) = crate::data::loader::build_selection_for_test(
            &store,
            &crs,
            &display,
            groups.min(total),
            boxes,
            style_sel.as_ref(),
        )
        .unwrap();
        eprintln!("source: {}", if boxes { "boxes" } else { "geometry" });
        let mut bin_hist = std::collections::BTreeMap::new();
        for c in geometry.chunks.iter() {
            *bin_hist.entry(c.bin).or_insert(0usize) += c.fill_indices.len() / 3;
        }
        eprintln!("triangles per style bin: {bin_hist:?}");
        let fills: usize = geometry.chunks.iter().map(|c| c.fill_indices.len() / 3).sum();
        eprintln!(
            "{rows} features, {} chunks, {fills} triangles, bounds {:?}",
            geometry.chunks.len(),
            geometry.bounds_world
        );

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        // Width must keep the readback row stride a multiple of 256, or
        // copy_texture_to_buffer fails and every pixel reads back zero.
        let (w, h) = (1280u32, 800u32);
        assert_eq!((w * 4) % 256, 0);
        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, format));

        let mut camera = crate::map::camera::Camera::default();
        camera.fit(geometry.bounds_world, [w as f32, h as f32], 20.0);
        let cb = MapCallback {
            camera,
            viewport_px: [w as f32, h as f32],
            tile_opacity: 1.0,
            tile_draws: vec![],
            tile_uploads: vec![],
            alive_tiles: Default::default(),
            alive_layers: Default::default(),
            layers: vec![LayerDraw {
                key: (1, 0),
                composite_group: 1,
                chunks: geometry.chunks.clone(),
                style: DrawStyle {
                    fill_color: [0.3, 0.7, 0.4, 1.0],
                    // GEOPQ_OUTLINES=1 draws borders, the rendition a
                    // colour map now turns off.
                    line_color: if outlines { [0.15, 0.18, 0.2, 1.0] } else { [0.0; 4] },
                    point_color: [0.0; 4],
                    line_half_width_px: if outlines { 0.6 } else { 0.0 },
                    point_radius_px: 0.0,
                    bin_colors: style_by
                        .as_ref()
                        .map(|sb| std::sync::Arc::new(sb.bin_colors())),
                    hidden_bins: style_by.as_ref().map_or(0, |sb| sb.hidden_bins),
                },
            }],
            background: [0.05, 0.05, 0.08, 1.0],
        };
        let mk_tex = |samples: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa_view = mk_tex(MSAA_SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT)
            .create_view(&Default::default());
        let resolve_tex = mk_tex(
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let resolve_view = resolve_tex.create_view(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: 1.0,
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let bufs = cb.prepare(&device, &queue, &screen, &mut encoder, &mut resources);
        queue.submit(bufs);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    multiview_mask: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &msaa_view,
                        depth_slice: None,
                        resolve_target: Some(&resolve_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05,
                                g: 0.05,
                                b: 0.08,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            let info = eframe::egui::PaintCallbackInfo {
                viewport: eframe::egui::Rect::from_min_size(
                    Default::default(),
                    eframe::egui::vec2(w as f32, h as f32),
                ),
                clip_rect: eframe::egui::Rect::from_min_size(
                    Default::default(),
                    eframe::egui::vec2(w as f32, h as f32),
                ),
                pixels_per_point: 1.0,
                screen_size_px: [w, h],
            };
            cb.paint(info, &mut pass, &resources);
        }
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (w * h * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resolve_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &out,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().to_vec();
        let painted = data
            .chunks_exact(4)
            .filter(|px| px[1] > 60 && px[1] > px[2])
            .count();
        eprintln!(
            "painted {painted} of {} pixels ({:.1}%)",
            w * h,
            100.0 * painted as f64 / (w * h) as f64
        );
        let dst = std::env::var("GEOPQ_PNG").unwrap_or_else(|_| "/tmp/boxes.png".into());
        image::RgbaImage::from_raw(w, h, data).unwrap().save(&dst).ok();
        eprintln!("wrote {dst}");
    }

    /// Frame-time measurement against a real file (set GEOPQ_BENCH_FILE).
    /// Run: GEOPQ_BENCH_FILE=... cargo test --release real_file_frame -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_file_frame_time() {
        let Ok(path) = std::env::var("GEOPQ_BENCH_FILE") else {
            return;
        };
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (store, crs, _info, _rg) =
            crate::data::loader::open_store_for_test(&path.clone().into()).unwrap();
        let store = std::sync::Arc::new(store);
        let (geometry, rows, _bad) =
            crate::data::loader::build_geometry_for_test(&store, &crs, &display).unwrap();
        eprintln!("built {rows} features, {} chunks", geometry.chunks.len());
        let mut lod_counts = [0usize; LINE_LODS_TOTAL];
        let mut total_fill = 0usize;
        let mut total_pts = 0usize;
        for c in geometry.chunks.iter() {
            for (i, l) in c.lines.iter().enumerate() {
                lod_counts[i] += l.segments.len();
            }
            total_fill += c.fill_indices.len() / 3;
            total_pts += c.point_instances.len();
        }
        eprintln!(
            "segments per lod {lod_counts:?}, fill tris {total_fill}, points {total_pts}"
        );

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (1600u32, 1000u32);
        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, format));

        let mut camera = crate::map::camera::Camera::default();
        camera.fit(geometry.bounds_world, [w as f32, h as f32], 20.0);

        // Row-group bbox overlay (computed source for this fixture).
        let (_g2, _r2, _b2, rg_boxes, _resolved) =
            crate::data::loader::build_geometry_for_test_full(&store, &crs, &display).unwrap();
        let rg_overlay = crate::app::build_rg_overlay(&crs, &display, &rg_boxes).0;
        let cb = MapCallback {
            camera,
            viewport_px: [w as f32, h as f32],
            tile_opacity: 1.0,
            tile_draws: vec![],
            tile_uploads: vec![],
            alive_tiles: Default::default(),
            alive_layers: Default::default(),
            layers: vec![
                LayerDraw {
                    key: (1, 0),
                    composite_group: 1,
                    chunks: geometry.chunks.clone(),
                    style: DrawStyle {
                        fill_color: [0.3, 0.5, 0.8, 0.35],
                        line_color: [0.2, 0.3, 0.5, 1.0],
                        point_color: [0.0; 4],
                        line_half_width_px: 0.6,
                        point_radius_px: 3.0,
                        bin_colors: None,
                        hidden_bins: 0,
                    },
                },
                LayerDraw {
                    key: (2, 0),
                    composite_group: 2,
                    chunks: rg_overlay,
                    style: DrawStyle {
                        fill_color: [0.0; 4],
                        line_color: [0.95, 0.55, 0.10, 0.95],
                        point_color: [0.0; 4],
                        line_half_width_px: 1.0,
                        point_radius_px: 0.0,
                        bin_colors: None,
                        hidden_bins: 0,
                    },
                },
            ],
            background: [0.0; 4],
        };

        let mk_tex = |samples: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let msaa_view = mk_tex(MSAA_SAMPLES).create_view(&Default::default());
        let resolve_tex = mk_tex(1);
        let resolve_view = resolve_tex.create_view(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: 1.0,
        };
        let mut frame = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            let bufs = cb.prepare(&device, &queue, &screen, &mut encoder, &mut resources);
            queue.submit(bufs);
            {
                let mut pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        multiview_mask: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &msaa_view,
                            depth_slice: None,
                            resolve_target: Some(&resolve_view),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    })
                    .forget_lifetime();
                let info = eframe::egui::PaintCallbackInfo {
                    viewport: eframe::egui::Rect::from_min_size(
                        Default::default(),
                        eframe::egui::vec2(w as f32, h as f32),
                    ),
                    clip_rect: eframe::egui::Rect::from_min_size(
                        Default::default(),
                        eframe::egui::vec2(w as f32, h as f32),
                    ),
                    pixels_per_point: 1.0,
                    screen_size_px: [w, h],
                };
                cb.paint(info, &mut pass, &resources);
            }
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        use std::time::Instant;
        let t = Instant::now();
        frame();
        eprintln!("first frame (upload): {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        for _ in 0..5 {
            frame();
        }
        eprintln!("steady frame full extent (lod {}): {:.0} ms",
            line_lod_for_zoom(camera.zoom),
            t.elapsed().as_secs_f64() * 1000.0 / 5.0);
        // Mid-zoom views: measure each LOD band.
        // Aim at the densest chunk so the views hit real parcels.
        let dense = geometry
            .chunks
            .iter()
            .max_by_key(|c| c.fill_indices.len())
            .expect("chunks");
        let dense_center = [
            dense.origin[0] + (dense.bounds_local[0] + dense.bounds_local[2]) as f64 * 0.5,
            dense.origin[1] + (dense.bounds_local[1] + dense.bounds_local[3]) as f64 * 0.5,
        ];
        for z in [9.5f64, 12.0, 13.5, 16.5] {
            let mut cam2 = camera;
            cam2.center = dense_center;
            cam2.zoom = z;
            let cb2 = MapCallback {
                camera: cam2,
                viewport_px: [w as f32, h as f32],
                tile_opacity: 1.0,
            tile_draws: vec![],
                tile_uploads: vec![],
                alive_tiles: Default::default(),
                alive_layers: Default::default(),
                layers: vec![
                    LayerDraw {
                        key: (1, 0),
                        composite_group: 1,
                        chunks: geometry.chunks.clone(),
                        style: DrawStyle {
                            fill_color: [0.3, 0.5, 0.8, 0.35],
                            line_color: [0.9, 0.95, 1.0, 1.0],
                            point_color: [0.0; 4],
                            line_half_width_px: 0.6,
                            point_radius_px: 3.0,
                            bin_colors: None,
                            hidden_bins: 0,
                        },
                    },
                    LayerDraw {
                        key: (2, 0),
                        composite_group: 2,
                        chunks: crate::app::build_rg_overlay(&crs, &display, &rg_boxes).0,
                        style: DrawStyle {
                            fill_color: [0.0; 4],
                            line_color: [0.95, 0.55, 0.10, 0.95],
                            point_color: [0.0; 4],
                            line_half_width_px: 1.2,
                            point_radius_px: 0.0,
                            bin_colors: None,
                            hidden_bins: 0,
                        },
                    },
                ],
                background: [0.0; 4],
            };
            let mut frame2 = || {
                let t_cpu = Instant::now();
                let mut encoder = device.create_command_encoder(&Default::default());
                let bufs = cb2.prepare(&device, &queue, &screen, &mut encoder, &mut resources);
                queue.submit(bufs);
                let prep_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
                {
                    let mut pass = encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            multiview_mask: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &msaa_view,
                                depth_slice: None,
                                resolve_target: Some(&resolve_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        })
                        .forget_lifetime();
                    let info = eframe::egui::PaintCallbackInfo {
                        viewport: eframe::egui::Rect::from_min_size(
                            Default::default(),
                            eframe::egui::vec2(w as f32, h as f32),
                        ),
                        clip_rect: eframe::egui::Rect::from_min_size(
                            Default::default(),
                            eframe::egui::vec2(w as f32, h as f32),
                        ),
                        pixels_per_point: 1.0,
                        screen_size_px: [w, h],
                    };
                    cb2.paint(info, &mut pass, &resources);
                }
                let rec_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let total_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
                (prep_ms, rec_ms, total_ms)
            };
            frame2();
            if std::env::var("DUMP_PNG").is_ok() {
                let out = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (w * h * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let resolve2 = mk_tex(1);
                // Re-render to a copyable target.
                {
                    let resolve_view2 = resolve2.create_view(&Default::default());
                    let _ = resolve_view2;
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &resolve_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &out,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * 4),
                            rows_per_image: Some(h),
                        },
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                queue.submit([encoder.finish()]);
                let slice = out.slice(..);
                slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let data = slice.get_mapped_range().to_vec();
                image::RgbaImage::from_raw(w, h, data)
                    .unwrap()
                    .save(format!("/tmp/parcels_z{z}.png"))
                    .ok();
            }
            let mut acc = (0.0f64, 0.0f64, 0.0f64);
            let t = Instant::now();
            for _ in 0..5 {
                let (a, b, c) = frame2();
                acc = (acc.0 + a / 5.0, acc.1 + b / 5.0, acc.2 + c / 5.0);
            }
            let lod = line_lod_for_zoom(z);
            let scale2 = cam2.scale();
            let mut drawn: u64 = 0;
            let mut len_px: f64 = 0.0;
            for c in geometry.chunks.iter() {
                let n = c.lines[lod].count_for_cutoff(MIN_OUTLINE_FEATURE_PX / scale2) as usize;
                drawn += n as u64;
                for seg in &c.lines[lod].segments[..n] {
                    let (dx, dy) = ((seg[2] - seg[0]) as f64, (seg[3] - seg[1]) as f64);
                    len_px += (dx * dx + dy * dy).sqrt() * scale2;
                }
            }
            eprintln!("z{z}: drawn length {:.1} Mpx -> ~{:.0} Mpx shaded (w=3.2)", len_px / 1e6, len_px * 3.2 / 1e6);
            eprintln!(
                "steady z{z} (lod {lod}, {drawn} segs): {:.0} ms (prepare {:.0}, record {:.0}, gpu {:.0})",
                t.elapsed().as_secs_f64() * 1000.0 / 5.0,
                acc.0,
                acc.1 - acc.0,
                acc.2 - acc.1,
            );
        }
    }

    /// 3.75M points at world view: with density-capped draws the frame must
    /// stay in interactive territory. Prints drawn/total and frame time.
    #[test]
    fn dense_points_frame_time() {
        use crate::data::geometry::{FeatureRef, MeshBuilder};
        use std::time::Instant;

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (1600u32, 1000u32);

        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, format));

        // 3.75M random points over a Europe-sized world extent.
        let mut mb = MeshBuilder::default();
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rand01 = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        const N: u32 = 3_750_000;
        for i in 0..N {
            let g = geo_types::Geometry::Point(geo_types::Point::new(
                0.47 + 0.05 * rand01(),
                0.30 + 0.05 * rand01(),
            ));
            mb.add(&g, FeatureRef { index: i });
        }
        let chunks = std::sync::Arc::new(mb.finish());
        let total: u32 = chunks.iter().map(|c| c.point_instances.len() as u32).sum();
        assert_eq!(total, N);

        let mut camera = crate::map::camera::Camera::default();
        camera.fit([0.0, 0.0, 1.0, 0.61], [w as f32, h as f32], 20.0);
        let scale = camera.scale();
        let radius = 3.0f32;
        let drawn: u32 = chunks
            .iter()
            .map(|c| {
                let px = [
                    ((c.bounds_local[2] - c.bounds_local[0]) as f64 * scale) as f32,
                    ((c.bounds_local[3] - c.bounds_local[1]) as f64 * scale) as f32,
                ];
                point_draw_count(px, radius, c.point_instances.len() as u32)
            })
            .sum();

        let cb = MapCallback {
            camera,
            viewport_px: [w as f32, h as f32],
            tile_opacity: 1.0,
            tile_draws: vec![],
            tile_uploads: vec![],
            alive_tiles: Default::default(),
            alive_layers: Default::default(),
            layers: vec![LayerDraw {
                key: (1, 0),
                composite_group: 1,
                chunks,
                style: DrawStyle {
                    fill_color: [0.0; 4],
                    line_color: [0.0; 4],
                    point_color: [0.2, 0.4, 0.9, 1.0],
                    line_half_width_px: 0.6,
                    point_radius_px: radius,
                    bin_colors: None,
                    hidden_bins: 0,
                },
            }],
            background: [0.0; 4],
        };

        let mk_tex = |samples: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let msaa_view = mk_tex(MSAA_SAMPLES).create_view(&Default::default());
        let resolve_view = mk_tex(1).create_view(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: 1.0,
        };

        let mut render_frame = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            cb.prepare(&device, &queue, &screen, &mut encoder, &mut resources);
            {
                let mut pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        multiview_mask: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &msaa_view,
                            depth_slice: None,
                            resolve_target: Some(&resolve_view),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    })
                    .forget_lifetime();
                let info = eframe::egui::PaintCallbackInfo {
                    viewport: eframe::egui::Rect::from_min_size(
                        Default::default(),
                        eframe::egui::vec2(w as f32, h as f32),
                    ),
                    clip_rect: eframe::egui::Rect::from_min_size(
                        Default::default(),
                        eframe::egui::vec2(w as f32, h as f32),
                    ),
                    pixels_per_point: 1.0,
                    screen_size_px: [w, h],
                };
                cb.paint(info, &mut pass, &resources);
            }
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };

        render_frame(); // warmup + buffer upload
        let t0 = Instant::now();
        const FRAMES: u32 = 30;
        for _ in 0..FRAMES {
            render_frame();
        }
        let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;
        eprintln!(
            "world view: drawing {drawn} of {total} points ({:.1}%), avg frame {avg_ms:.2} ms",
            100.0 * drawn as f64 / total as f64
        );
        // GPU cost is the same in both profiles, but the CPU side (prepare,
        // uniform building) is only representative in release builds.
        if !cfg!(debug_assertions) {
            assert!(avg_ms < 25.0, "frame too slow: {avg_ms:.2} ms");
        }
        assert!(drawn < total / 5, "cap not binding: {drawn}/{total}");
    }

    /// Headless render: draw a polygon + line + points through the real
    /// pipelines and assert colored pixels come back.
    #[test]
    fn pipelines_produce_pixels() {
        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = 256u32;

        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, format));

        // Geometry around world (0.5, 0.5).
        let mut mb = MeshBuilder::default();
        let poly = geo_types::Polygon::new(
            geo_types::LineString::from(vec![
                (0.4998, 0.4998),
                (0.5002, 0.4998),
                (0.5002, 0.5002),
                (0.4998, 0.5002),
                (0.4998, 0.4998),
            ]),
            vec![],
        );
        use crate::data::geometry::FeatureRef;
        mb.add(&geo_types::Geometry::Polygon(poly), FeatureRef::INVALID);
        mb.add(
            &geo_types::Geometry::Point(geo_types::Point::new(0.5, 0.5)),
            FeatureRef::INVALID,
        );
        let chunks = std::sync::Arc::new(mb.finish());

        let cb = MapCallback {
            camera: crate::map::camera::Camera {
                center: [0.5, 0.5],
                zoom: 10.0,
            },
            viewport_px: [size as f32, size as f32],
            tile_opacity: 1.0,
            tile_draws: vec![],
            tile_uploads: vec![],
            alive_tiles: Default::default(),
            alive_layers: Default::default(),
            layers: vec![LayerDraw {
                key: (1, 0),
                composite_group: 1,
                chunks,
                style: DrawStyle {
                    fill_color: [1.0, 0.0, 0.0, 1.0],
                    line_color: [0.0, 1.0, 0.0, 1.0],
                    point_color: [0.0, 0.0, 1.0, 1.0],
                    line_half_width_px: 2.0,
                    point_radius_px: 5.0,
                    bin_colors: None,
                    hidden_bins: 0,
                },
            }],
            background: [0.0; 4],
        };

        let mk_tex = |samples: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = mk_tex(MSAA_SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = mk_tex(
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let msaa_view = msaa.create_view(&Default::default());
        let resolve_view = resolve.create_view(&Default::default());

        let mut encoder = device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size, size],
            pixels_per_point: 1.0,
        };
        let bufs = cb.prepare(&device, &queue, &screen, &mut encoder, &mut resources);
        queue.submit(bufs);

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    multiview_mask: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &msaa_view,
                        depth_slice: None,
                        resolve_target: Some(&resolve_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            let info = eframe::egui::PaintCallbackInfo {
                viewport: eframe::egui::Rect::from_min_size(
                    Default::default(),
                    eframe::egui::vec2(size as f32, size as f32),
                ),
                clip_rect: eframe::egui::Rect::from_min_size(
                    Default::default(),
                    eframe::egui::vec2(size as f32, size as f32),
                ),
                pixels_per_point: 1.0,
                screen_size_px: [size, size],
            };
            cb.paint(info, &mut pass, &resources);
        }

        // Read back.
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resolve,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &out,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().to_vec();

        let mut red = 0usize;
        let mut green = 0usize;
        let mut blue = 0usize;
        for px in data.chunks_exact(4) {
            if px[0] > 128 && px[1] < 100 && px[2] < 100 {
                red += 1;
            }
            if px[1] > 128 && px[0] < 128 && px[2] < 100 {
                green += 1;
            }
            if px[2] > 128 && px[0] < 100 {
                blue += 1;
            }
        }
        // ~0.001 world * 262144 px/world = ~262 px square -> tens of thousands of fill px
        assert!(red > 5_000, "fill pixels: {red}");
        assert!(green > 500, "outline pixels: {green}");
        assert!(blue > 20, "point pixels: {blue}");
    }

    /// Two neighbouring tiles warped into a non-Mercator projection and drawn
    /// together. They share one vertex/index buffer, so this is what catches a
    /// wrong `base_vertex` or index range: get either wrong and the second
    /// tile lands on top of the first, or vanishes.
    #[test]
    fn warped_tiles_draw_side_by_side() {
        use crate::map::tiles::{MipLevel, TileDrawCmd, TileId, TileKey, TileUpload};
        use crate::map::warp::{self, Warp};

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = 256u32;
        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, format));

        let display = crate::data::crs::DisplayCrs::from_epsg(3035).expect("EPSG:3035");
        let w = Warp::new(&display);
        let (z, y) = (6u8, 21u32);
        let ids = [TileId { z, x: 32, y }, TileId { z, x: 33, y }];
        let meshes: Vec<_> = ids
            .iter()
            .map(|id| {
                std::sync::Arc::new(
                    warp::tile_mesh(&w, *id, warp::TILE_SUBDIV).expect("mesh"),
                )
            })
            .collect();

        // Frame both tiles, with room to spare so neither touches the edge.
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for m in &meshes {
            for v in &m.verts {
                if !v[0].is_finite() {
                    continue;
                }
                let (x, y) = (m.origin[0] + v[0] as f64, m.origin[1] + v[1] as f64);
                b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
            }
        }
        let mut camera = crate::map::camera::Camera::default();
        camera.fit(b, [size as f32, size as f32], 16.0);

        // Flat colours: one tile red, its neighbour blue.
        let solid = |c: [u8; 4]| -> Vec<MipLevel> {
            crate::map::tiles::mip_chain(&c.repeat(64 * 64), 64, 64)
        };
        let keys: Vec<TileKey> = ids
            .iter()
            .map(|id| TileKey {
                source: 0,
                id: *id,
            })
            .collect();

        let cb = MapCallback {
            camera,
            viewport_px: [size as f32, size as f32],
            tile_opacity: 1.0,
            tile_draws: vec![
                TileDrawCmd {
                    key: keys[0],
                    world_rect: ids[0].world_rect(),
                    mesh: Some(meshes[0].clone()),
                },
                TileDrawCmd {
                    key: keys[1],
                    world_rect: ids[1].world_rect(),
                    mesh: Some(meshes[1].clone()),
                },
            ],
            tile_uploads: vec![
                TileUpload {
                    key: keys[0],
                    mips: solid([255, 0, 0, 255]),
                },
                TileUpload {
                    key: keys[1],
                    mips: solid([0, 0, 255, 255]),
                },
            ],
            alive_tiles: keys.iter().copied().collect(),
            alive_layers: Default::default(),
            layers: vec![],
            background: [0.0; 4],
        };

        let data = render_to_pixels(&device, &queue, &mut resources, &cb, size);

        // Centroid per colour, in pixels.
        let mut acc = [(0f64, 0f64, 0usize); 2];
        for (i, px) in data.chunks_exact(4).enumerate() {
            let (x, y) = ((i as u32 % size) as f64, (i as u32 / size) as f64);
            let slot = if px[0] > 128 && px[2] < 100 {
                0
            } else if px[2] > 128 && px[0] < 100 {
                1
            } else {
                continue;
            };
            acc[slot].0 += x;
            acc[slot].1 += y;
            acc[slot].2 += 1;
        }
        let [red, blue] = acc;
        assert!(red.2 > 2_000, "red tile pixels: {}", red.2);
        assert!(blue.2 > 2_000, "blue tile pixels: {}", blue.2);
        // x=32 is the western tile, so red must sit left of blue.
        assert!(
            red.0 / red.2 as f64 + 20.0 < blue.0 / blue.2 as f64,
            "red centroid {:.1} not left of blue {:.1}",
            red.0 / red.2 as f64,
            blue.0 / blue.2 as f64
        );
    }

    /// A tile's pixels must reach the screen with their values intact.
    ///
    /// The target surface is not sRGB (eframe hands us `Bgra8Unorm` and egui
    /// writes gamma-encoded values into it), so declaring the tile texture
    /// sRGB made the sampler decode without anything re-encoding. Mid-grey
    /// 128 came out at 55. It barely showed on a light basemap and turned the
    /// dark one black.
    #[test]
    fn tile_grey_survives_the_trip_to_the_screen() {
        use crate::map::tiles::{MipLevel, TileDrawCmd, TileId, TileKey, TileUpload};

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let size = 64u32;
        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, wgpu::TextureFormat::Rgba8Unorm));

        let id = TileId { z: 4, x: 8, y: 5 };
        let key = TileKey { source: 0, id };
        let r = id.world_rect();
        let mut camera = crate::map::camera::Camera::default();
        // Inside the tile, so every sampled pixel is tile and none is clear.
        camera.fit(
            [
                r[0] + (r[2] - r[0]) * 0.25,
                r[1] + (r[3] - r[1]) * 0.25,
                r[0] + (r[2] - r[0]) * 0.75,
                r[1] + (r[3] - r[1]) * 0.75,
            ],
            [size as f32, size as f32],
            0.0,
        );

        let cb = MapCallback {
            camera,
            viewport_px: [size as f32, size as f32],
            tile_opacity: 1.0,
            tile_draws: vec![TileDrawCmd {
                key,
                world_rect: r,
                mesh: None,
            }],
            tile_uploads: vec![TileUpload {
                key,
                mips: vec![MipLevel {
                    w: 4,
                    h: 4,
                    px: [128u8, 128, 128, 255].repeat(16),
                }],
            }],
            alive_tiles: [key].into_iter().collect(),
            alive_layers: Default::default(),
            layers: vec![],
            background: [0.0; 4],
        };
        let data = render_to_pixels(&device, &queue, &mut resources, &cb, size);

        let centre = ((size / 2) * size + size / 2) as usize * 4;
        let px = &data[centre..centre + 3];
        assert!(
            px.iter().all(|&v| (v as i32 - 128).abs() <= 2),
            "mid-grey came back as {px:?}, expected ~[128, 128, 128]"
        );
    }

    /// Fading the basemap must actually fade it. The slider is only
    /// useful if it reaches the tiles, and a warped tile takes a different
    /// pipeline from a flat one, so both are checked.
    #[test]
    fn tile_opacity_fades_the_basemap() {
        use crate::map::tiles::{MipLevel, TileDrawCmd, TileId, TileKey, TileUpload};

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let size = 64u32;
        let id = TileId { z: 4, x: 8, y: 5 };
        let key = TileKey { source: 0, id };
        let r = id.world_rect();

        let sample = |opacity: f32| -> u8 {
            let mut resources = egui_wgpu::CallbackResources::default();
            resources.insert(MapResources::new(&device, wgpu::TextureFormat::Rgba8Unorm));
            let mut camera = crate::map::camera::Camera::default();
            camera.fit(
                [
                    r[0] + (r[2] - r[0]) * 0.25,
                    r[1] + (r[3] - r[1]) * 0.25,
                    r[0] + (r[2] - r[0]) * 0.75,
                    r[1] + (r[3] - r[1]) * 0.75,
                ],
                [size as f32, size as f32],
                0.0,
            );
            let cb = MapCallback {
                camera,
                viewport_px: [size as f32, size as f32],
                tile_opacity: opacity,
                tile_draws: vec![TileDrawCmd {
                    key,
                    world_rect: r,
                    mesh: None,
                }],
                tile_uploads: vec![TileUpload {
                    key,
                    mips: vec![MipLevel {
                        w: 4,
                        h: 4,
                        px: [200u8, 200, 200, 255].repeat(16),
                    }],
                }],
                alive_tiles: [key].into_iter().collect(),
                alive_layers: Default::default(),
                layers: vec![],
                background: [0.0; 4],
            };
            let data = render_to_pixels(&device, &queue, &mut resources, &cb, size);
            let centre = ((size / 2) * size + size / 2) as usize * 4;
            data[centre]
        };

        // Over a black clear, the tile arrives scaled by its opacity.
        let full = sample(1.0);
        let half = sample(0.5);
        let none = sample(0.0);
        assert!((full as i32 - 200).abs() <= 2, "opaque: {full}");
        assert!((half as i32 - 100).abs() <= 3, "half: {half}");
        assert_eq!(none, 0, "fully transparent should leave the clear colour");
    }

    /// Render a real reprojected basemap to a PNG, for looking at.
    ///
    /// Ignored by default: it fetches live tiles. The things worth checking
    /// in the output are that coastlines run continuously across tile
    /// boundaries (the meshes agree on their shared edges) and that the
    /// graticule-shaped curvature is present at all (a flat quad per tile
    /// would leave straight tile edges and visible kinks).
    ///
    /// The view is set by environment so both halves of the rule can be
    /// looked at: a continental view (labels shear, so the label-free twin
    /// is drawn) and a metro one (labels survive, so they are kept).
    ///
    ///   cargo test --bin geopq-workbench warped_basemap_snapshot -- --ignored --nocapture
    ///   GEOPQ_SNAP_EPSG=2154 GEOPQ_SNAP_LON=1.44 GEOPQ_SNAP_LAT=43.60 \
    ///   GEOPQ_SNAP_ZOOM=12 cargo test ... (same)
    #[test]
    #[ignore]
    fn warped_basemap_snapshot() {
        use crate::map::tiles::{self, TileDrawCmd, TileKey, TileUpload, TILE_SOURCES};
        use crate::map::warp::{self, Warp};

        let Some((device, queue)) = super::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let size = 640u32;
        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, wgpu::TextureFormat::Rgba8Unorm));

        let env = |k: &str, d: f64| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        // Default: Europe LAEA over the continent, far enough out that the
        // meridians visibly converge and close enough that tiles still exist.
        let epsg = env("GEOPQ_SNAP_EPSG", 3035.0) as u32;
        let display = crate::data::crs::DisplayCrs::from_epsg(epsg).expect("display CRS");
        let w = Warp::new(&display);
        let centre = w
            .display_world(warp::merc_world_from_lonlat(
                env("GEOPQ_SNAP_LON", 10.0),
                env("GEOPQ_SNAP_LAT", 50.0),
            ))
            .expect("centre projects");
        let camera = crate::map::camera::Camera {
            center: centre,
            zoom: env("GEOPQ_SNAP_ZOOM", 5.0),
        };

        // Follow the app's own rule: start from a labelled source and fall
        // back to its label-free twin only when the view would shear text.
        let mut src = env("GEOPQ_SNAP_SRC", 0.0) as usize;
        let plan = warp::plan(&w, &camera, [size as f32, size as f32], TILE_SOURCES[src].max_zoom)
            .expect("plans");
        if TILE_SOURCES[src].labels && !plan.labels_survive() {
            src = tiles::nolabels_twin(src).expect("twin");
        }
        eprintln!(
            "level {}, rotation {:.1}°, anisotropy {:.2}, labels survive: {}",
            plan.zoom,
            plan.rotation_deg,
            plan.anisotropy,
            plan.labels_survive()
        );

        let ids = warp::tiles_for(plan.merc_bbox, plan.zoom, 64);
        eprintln!("fetching {} tiles from {}", ids.len(), TILE_SOURCES[src].name);
        let mut draws = Vec::new();
        let mut uploads = Vec::new();
        let mut alive = std::collections::HashSet::new();
        for id in ids {
            let key = TileKey {
                source: src as u8,
                id,
            };
            let Some(mips) = tiles::fetch_tile_blocking(key) else {
                eprintln!("  tile {id:?} failed to fetch");
                continue;
            };
            // GEOPQ_SNAP_QUAD renders through the plain Mercator quad path
            // instead, which is what most views still use and what the mip
            // chain and anisotropic sampler could regress.
            let mesh = match std::env::var("GEOPQ_SNAP_QUAD").is_ok() {
                true => None,
                false => match warp::tile_mesh(&w, id, warp::TILE_SUBDIV) {
                    Some(m) => Some(std::sync::Arc::new(m)),
                    None => continue,
                },
            };
            uploads.push(TileUpload { key, mips });
            draws.push(TileDrawCmd {
                key,
                world_rect: id.world_rect(),
                mesh,
            });
            alive.insert(key);
        }
        assert!(!draws.is_empty(), "no tiles fetched; is the network up?");

        let cb = MapCallback {
            camera,
            viewport_px: [size as f32, size as f32],
            tile_opacity: 1.0,
            tile_draws: draws,
            tile_uploads: uploads,
            alive_tiles: alive,
            alive_layers: Default::default(),
            layers: vec![],
            background: [0.0; 4],
        };
        let data = render_to_pixels(&device, &queue, &mut resources, &cb, size);

        let out = std::env::var("GEOPQ_SNAPSHOT")
            .unwrap_or_else(|_| "target/warped-basemap.png".into());
        image::RgbaImage::from_raw(size, size, data)
            .expect("frame")
            .save(&out)
            .expect("write png");
        eprintln!("wrote {out}");
    }

    /// Run one callback through prepare + paint and read the frame back.
    fn render_to_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &mut egui_wgpu::CallbackResources,
        cb: &MapCallback,
        size: u32,
    ) -> Vec<u8> {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mk_tex = |samples: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = mk_tex(MSAA_SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = mk_tex(
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let msaa_view = msaa.create_view(&Default::default());
        let resolve_view = resolve.create_view(&Default::default());

        let mut encoder = device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size, size],
            pixels_per_point: 1.0,
        };
        queue.submit(cb.prepare(device, queue, &screen, &mut encoder, resources));
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    multiview_mask: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &msaa_view,
                        depth_slice: None,
                        resolve_target: Some(&resolve_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            let r = eframe::egui::Rect::from_min_size(
                Default::default(),
                eframe::egui::vec2(size as f32, size as f32),
            );
            cb.paint(
                eframe::egui::PaintCallbackInfo {
                    viewport: r,
                    clip_rect: r,
                    pixels_per_point: 1.0,
                    screen_size_px: [size, size],
                },
                &mut pass,
                resources,
            );
        }
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resolve,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &out,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        slice.get_mapped_range().to_vec()
    }
}
