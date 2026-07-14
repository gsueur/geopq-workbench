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
#[derive(Clone, Copy)]
pub struct DrawStyle {
    pub fill_color: [f32; 4],
    pub line_color: [f32; 4],
    pub point_color: [f32; 4],
    pub line_half_width_px: f32,
    pub point_radius_px: f32,
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
    /// World-space bounds of this chunk's vertices, for viewport culling.
    bounds_world: [f64; 4],
    fill_vbuf: Option<wgpu::Buffer>,
    fill_ibuf: Option<wgpu::Buffer>,
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

struct TileGpu {
    bind_group: wgpu::BindGroup,
    _texture: wgpu::Texture,
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
            &[wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4],
            }],
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tile sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
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
        let (w, h) = (up.rgba.width(), up.rgba.height());
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            up.rgba.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
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
                    fill_ibuf: mk_buf(
                        bytemuck::cast_slice(&c.fill_indices),
                        wgpu::BufferUsages::INDEX,
                        "fill idx",
                    ),
                    fill_index_count: c.fill_indices.len() as u32,
                    line_bufs: {
                        let mk_lines = |lod: &crate::data::geometry::LineLod| {
                            mk_buf(
                                bytemuck::cast_slice(&lod.segments),
                                wgpu::BufferUsages::VERTEX,
                                "lines",
                            )
                            .map(|b| (b, lod.size_index.clone()))
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

        // Tiles first (under vector data).
        for tile in &self.tile_draws {
            if !res.tiles.contains_key(&tile.key) {
                continue;
            }
            let r = tile.world_rect;
            let uoffset = push_uniform(
                &mut uniforms,
                [r[0], r[1]],
                [r[2] - r[0], r[3] - r[1]],
                [1.0, 1.0, 1.0, 1.0],
                0.0,
            );
            draw_list.push(DrawCmd::Tile {
                key: tile.key,
                uoffset,
            });
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
            for layer in group_layers {
                let Some(gpu) = res.layers.get(&layer.key) else {
                    continue;
                };
                for (ci, chunk) in gpu.chunks.iter().enumerate() {
                    if !no_fills && chunk.fill_index_count > 0 && visible(&chunk.bounds_world) {
                        let opaque =
                            [s.fill_color[0], s.fill_color[1], s.fill_color[2], 1.0];
                        let uoffset =
                            push_uniform(&mut uniforms, chunk.origin, [1.0, 1.0], opaque, 0.0);
                        fill_cmds.push((layer.key, ci, uoffset));
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
            let lod = line_lod_for_zoom(cam.zoom);
            for (ci, chunk) in gpu.chunks.iter().enumerate() {
                let count = chunk.line_bufs[lod]
                    .as_ref()
                    .map(|(_, index)| line_count_for_scale(index, scale))
                    .unwrap_or(0);
                if count > 0 && visible(&chunk.bounds_world) {
                    #[cfg(test)]
                    if std::env::var("GEOPQ_DEBUG_DRAWS").is_ok() {
                        eprintln!("chunk {ci}: lod {lod} count {count}");
                    }
                    let uoffset = push_uniform(
                        &mut uniforms,
                        chunk.origin,
                        [1.0, 1.0],
                        s.line_color,
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
                if chunk.point_count > 0 && visible(&chunk.bounds_world) {
                    let b = &chunk.bounds_world;
                    let chunk_px = [
                        ((b[2] - b[0]) * scale) as f32,
                        ((b[3] - b[1]) * scale) as f32,
                    ];
                    let count =
                        point_draw_count(chunk_px, s.point_radius_px, chunk.point_count);
                    let uoffset = push_uniform(
                        &mut uniforms,
                        chunk.origin,
                        [1.0, 1.0],
                        s.point_color,
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
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::geometry::MeshBuilder;
    use egui_wgpu::CallbackTrait;

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

        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
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

        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
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
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");

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
}
