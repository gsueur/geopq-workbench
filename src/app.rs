use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use eframe::egui::{self, Color32, RichText};
use eframe::egui_wgpu;

use crate::data::crs::{world_to_lonlat, BulkTransformer, Crs, DisplayCrs, DisplayKind};
use crate::data::geometry::MeshBuilder;
use crate::data::layer::{palette_color, VectorLayer};
use crate::data::loader::{self, LoadMsg, LoaderHandle};
use crate::map::camera::Camera;
use crate::map::renderer::{DrawStyle, LayerDraw, MapCallback, MapResources};
use crate::map::tiles::{TileCache, TILE_SOURCES};
use crate::picking::{self, Selection};

const HIGHLIGHT_KEY: u64 = u64::MAX;
const GRATICULE_KEY: u64 = u64::MAX - 1;
const COASTLINE_KEY: u64 = u64::MAX - 2;
/// Row-group bbox overlays: key = RG_OVERLAY_BASE | layer id.
const RG_OVERLAY_BASE: u64 = 1 << 62;

struct LoadingJob {
    path: PathBuf,
    frac: f32,
    stage: String,
}

pub struct ViewerApp {
    camera: Camera,
    display: DisplayCrs,
    layers: Vec<VectorLayer>,
    rebuilding: HashSet<u64>,
    selection: Option<Selection>,
    /// Single-row record batch (all columns) for the selected feature,
    /// fetched lazily from the layer's FeatureStore.
    selection_attrs: Option<arrow::record_batch::RecordBatch>,
    selection_generation: u64,
    highlight_chunks: Option<Arc<Vec<crate::data::geometry::ChunkMesh>>>,
    graticule_chunks: Arc<Vec<crate::data::geometry::ChunkMesh>>,
    coastline_chunks: Arc<Vec<crate::data::geometry::ChunkMesh>>,
    /// Bumped when overlays (graticule/coastline) are rebuilt for a new projection.
    graticule_generation: u64,
    show_graticule: bool,
    show_coastline: bool,
    /// Cached row-group bbox overlays per layer id: (layer generation,
    /// chunks, world-space label anchors).
    rg_overlays: HashMap<u64, (u64, Arc<Vec<crate::data::geometry::ChunkMesh>>, Vec<[f64; 2]>)>,

    tiles: TileCache,
    basemap: Option<usize>,

    load_tx: Sender<LoadMsg>,
    load_rx: Receiver<LoadMsg>,
    loading: HashMap<u64, LoadingJob>,
    next_job: u64,
    next_layer_id: u64,
    palette_idx: usize,
    pending_fit: bool,
    fit_bounds: Option<[f64; 4]>,
    /// Layers with a row-group append in flight.
    appending: HashSet<u64>,
    /// Last camera pose + when it last changed (for refinement debounce).
    last_cam: Option<([f64; 2], f64)>,
    cam_changed_at: f64,
    /// Current viewport in world coords (for load-time pruning).
    last_view_world: [f64; 4],

    errors: Vec<String>,
    show_errors: bool,
    info_open: Option<u64>,
    /// Layer generations whose CPU-side fill/line arrays were freed after
    /// GPU upload (points are kept for picking).
    stripped: HashSet<(u64, u64)>,
    epsg_input: String,
    cursor_world: Option<[f64; 2]>,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("wgpu render state (run with the wgpu backend)");
        rs.renderer
            .write()
            .callback_resources
            .insert(MapResources::new(&rs.device, rs.target_format));

        let (load_tx, load_rx) = channel();
        let display = DisplayCrs::hobo_dyer();
        let graticule_chunks = build_graticule(&display);
        let coastline_chunks = crate::data::coastline::build_coastline(&display);
        let mut app = Self {
            camera: Camera::default(),
            display,
            layers: Vec::new(),
            rebuilding: HashSet::new(),
            selection: None,
            selection_attrs: None,
            selection_generation: 0,
            highlight_chunks: None,
            graticule_chunks,
            coastline_chunks,
            graticule_generation: 0,
            show_graticule: true,
            show_coastline: true,
            rg_overlays: HashMap::new(),
            tiles: TileCache::new(cc.egui_ctx.clone()),
            basemap: Some(0),
            load_tx,
            load_rx,
            loading: HashMap::new(),
            next_job: 0,
            next_layer_id: 0,
            palette_idx: 0,
            pending_fit: true,
            fit_bounds: None,
            appending: HashSet::new(),
            last_cam: None,
            cam_changed_at: 0.0,
            last_view_world: [-10.0, -10.0, 10.0, 10.0],
            errors: Vec::new(),
            show_errors: false,
            info_open: None,
            stripped: HashSet::new(),
            epsg_input: String::new(),
            cursor_world: None,
        };
        for f in files {
            app.enqueue_load(f, &cc.egui_ctx);
        }
        app
    }

    fn enqueue_load(&mut self, path: PathBuf, ctx: &egui::Context) {
        let job = self.next_job;
        self.next_job += 1;
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;
        let color = palette_color(self.palette_idx);
        self.palette_idx += 1;
        self.loading.insert(
            job,
            LoadingJob {
                path: path.clone(),
                frac: 0.0,
                stage: "queued".into(),
            },
        );
        loader::spawn_load(
            LoaderHandle {
                tx: self.load_tx.clone(),
                egui_ctx: ctx.clone(),
            },
            job,
            layer_id,
            path,
            self.display.clone(),
            color,
            self.last_view_world,
        );
    }

    fn poll_loader(&mut self) {
        while let Ok(msg) = self.load_rx.try_recv() {
            match msg {
                LoadMsg::Progress { job, frac, stage } => {
                    if let Some(j) = self.loading.get_mut(&job) {
                        j.frac = frac;
                        j.stage = stage;
                    }
                }
                LoadMsg::Loaded { job, layer } => {
                    self.loading.remove(&job);
                    let first = self.layers.is_empty();
                    if layer.stats.bad_geoms > 0 {
                        self.push_error(format!(
                            "{}: {} geometries could not be decoded/projected",
                            layer.name, layer.stats.bad_geoms
                        ));
                    }
                    self.layers.push(*layer);
                    if first {
                        self.pending_fit = true;
                    }
                }
                LoadMsg::Rebuilt {
                    layer_id,
                    generation,
                    geometry,
                    stats_build_ms,
                    bad_geoms,
                } => {
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            l.sections = vec![geometry];
                            l.stats.build_ms = stats_build_ms;
                            l.stats.bad_geoms = bad_geoms;
                            self.rebuilding.remove(&layer_id);
                        }
                    }
                    if self.rebuilding.is_empty() {
                        self.pending_fit = true;
                    }
                }
                LoadMsg::Appended {
                    layer_id,
                    generation,
                    geometry,
                    rows,
                    loaded_rgs,
                } => {
                    self.appending.remove(&layer_id);
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            log::info!(
                                "{}: appended {} row groups ({rows} features)",
                                l.name,
                                loaded_rgs.len()
                            );
                            l.sections.push(geometry);
                            l.feature_count += rows;
                            l.loaded_rgs.extend(loaded_rgs);
                        }
                    }
                }
                LoadMsg::Failed { job, path, error } => {
                    self.loading.remove(&job);
                    self.push_error(format!("{}: {error}", path.display()));
                }
            }
        }
    }

    fn push_error(&mut self, e: String) {
        log::warn!("{e}");
        self.errors.push(e);
        self.show_errors = true;
    }

    fn union_bounds(&self) -> Option<[f64; 4]> {
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut any = false;
        for l in &self.layers {
            if self.rebuilding.contains(&l.id) {
                continue;
            }
            let lb = l.bounds_world();
            b[0] = b[0].min(lb[0]);
            b[1] = b[1].min(lb[1]);
            b[2] = b[2].max(lb[2]);
            b[3] = b[3].max(lb[3]);
            any = true;
        }
        (any && b[0].is_finite()).then_some(b)
    }

    fn set_display(&mut self, display: DisplayCrs, ctx: &egui::Context) {
        self.display = display;
        self.selection = None;
        self.highlight_chunks = None;
        self.graticule_chunks = build_graticule(&self.display);
        self.coastline_chunks = crate::data::coastline::build_coastline(&self.display);
        self.graticule_generation += 1;
        for l in &mut self.layers {
            l.generation += 1;
            self.rebuilding.insert(l.id);
            loader::spawn_rebuild(
                LoaderHandle {
                    tx: self.load_tx.clone(),
                    egui_ctx: ctx.clone(),
                },
                l.id,
                l.generation,
                l.store.clone(),
                l.crs.clone(),
                self.display.clone(),
                l.loaded_rgs.clone(),
            );
        }
        if self.layers.is_empty() {
            self.pending_fit = true;
        }
    }

    /// Load row groups that entered the viewport of partially loaded layers.
    fn refine_partial_layers(&mut self, ctx: &egui::Context) {
        use std::collections::HashSet as HS;
        let view = self.last_view_world;
        for l in &self.layers {
            if !l.is_partial()
                || !l.style.visible
                || self.appending.contains(&l.id)
                || self.rebuilding.contains(&l.id)
            {
                continue;
            }
            let Some(rg) = &l.rg_bboxes else { continue };
            let Some(rect) = loader::viewport_to_data_bbox(view, &self.display, &l.crs) else {
                continue;
            };
            let loaded: HS<u32> = l.loaded_rgs.iter().copied().collect();
            let needed: Vec<u32> = loader::intersecting_rgs(&rg.boxes, rect)
                .into_iter()
                .filter(|g| !loaded.contains(g))
                .collect();
            if needed.is_empty() {
                continue;
            }
            log::info!("{}: refining with {} row groups", l.name, needed.len());
            self.appending.insert(l.id);
            loader::spawn_append(
                LoaderHandle {
                    tx: self.load_tx.clone(),
                    egui_ctx: ctx.clone(),
                },
                l.id,
                l.generation,
                l.store.clone(),
                l.crs.clone(),
                self.display.clone(),
                needed,
            );
        }
    }

    fn select(&mut self, sel: Option<Selection>) {
        self.selection_generation += 1;
        self.highlight_chunks = sel.as_ref().map(|s| {
            let mut mb = MeshBuilder::default();
            mb.add(&s.world_geom, crate::data::geometry::FeatureRef::INVALID);
            Arc::new(mb.finish())
        });
        self.selection_attrs = sel.as_ref().and_then(|s| {
            let layer = self.layers.iter().find(|l| l.id == s.layer_id)?;
            match layer.store.fetch_row(s.feature.index) {
                Ok(batch) => Some(batch),
                Err(e) => {
                    log::warn!("attribute fetch failed: {e}");
                    None
                }
            }
        });
        self.selection = sel;
    }

    // ------------------------------------------------------------------
    // UI
    // ------------------------------------------------------------------

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button("📂 Open…").clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("GeoParquet", &["parquet", "geoparquet", "pq"])
                    .pick_files()
                {
                    for p in paths {
                        self.enqueue_load(p, &ctx);
                    }
                }
            }
            if ui
                .add_enabled(!self.layers.is_empty(), egui::Button::new("🌍 Fit all"))
                .clicked()
            {
                self.pending_fit = true;
            }

            ui.separator();

            ui.label("Basemap:");
            let basemap_enabled = self.display.is_mercator();
            ui.add_enabled_ui(basemap_enabled, |ui| {
                let current = self
                    .basemap
                    .map(|i| TILE_SOURCES[i].name)
                    .unwrap_or("None");
                egui::ComboBox::from_id_salt("basemap")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (i, s) in TILE_SOURCES.iter().enumerate() {
                            ui.selectable_value(&mut self.basemap, Some(i), s.name);
                        }
                        ui.selectable_value(&mut self.basemap, None, "None");
                    });
            });
            if !basemap_enabled {
                ui.label(RichText::new("(EPSG:3857 only)").weak().small());
            }

            ui.separator();

            ui.label("Projection:");
            let is_hobo = self.display.name.starts_with("Hobo");
            let is_wintri = self.display.kind == DisplayKind::WinkelTripel;
            let is_4326 = self.display.kind == DisplayKind::Plain && self.display.crs.epsg == Some(4326);
            let mut pick: Option<DisplayCrs> = None;
            egui::ComboBox::from_id_salt("projection")
                .selected_text(self.display.name.clone())
                .width(200.0)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(is_hobo, "Hobo–Dyer (equal-area)")
                        .clicked()
                        && !is_hobo
                    {
                        pick = Some(DisplayCrs::hobo_dyer());
                    }
                    if ui.selectable_label(is_wintri, "Winkel Tripel").clicked() && !is_wintri {
                        pick = Some(DisplayCrs::winkel_tripel());
                    }
                    if ui
                        .selectable_label(self.display.is_mercator(), "Web Mercator (3857)")
                        .clicked()
                        && !self.display.is_mercator()
                    {
                        pick = Some(DisplayCrs::mercator());
                    }
                    if ui.selectable_label(is_4326, "WGS 84 (4326)").clicked() && !is_4326 {
                        pick = Some(DisplayCrs::new(Crs::wgs84()));
                    }
                });
            ui.add(
                egui::TextEdit::singleline(&mut self.epsg_input)
                    .hint_text("EPSG…")
                    .desired_width(56.0),
            );
            if ui.button("Apply").clicked() {
                match self.epsg_input.trim().parse::<u32>() {
                    Ok(code) => match DisplayCrs::from_epsg(code) {
                        Ok(d) => pick = Some(d),
                        Err(e) => self.push_error(e),
                    },
                    Err(_) => self.push_error("invalid EPSG code".into()),
                }
            }
            if let Some(d) = pick {
                self.set_display(d, &ctx);
            }

            ui.separator();
            ui.checkbox(&mut self.show_graticule, "Graticule");
            ui.checkbox(&mut self.show_coastline, "Coastline");

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.errors.is_empty() {
                    let btn = egui::Button::new(
                        RichText::new(format!("⚠ {}", self.errors.len()))
                            .color(Color32::from_rgb(220, 60, 60)),
                    );
                    if ui.add(btn).clicked() {
                        self.show_errors = !self.show_errors;
                    }
                }
                for job in self.loading.values() {
                    ui.add(
                        egui::ProgressBar::new(job.frac)
                            .desired_width(140.0)
                            .text(format!(
                                "{} — {}",
                                job.path
                                    .file_name()
                                    .map(|s| s.to_string_lossy().into_owned())
                                    .unwrap_or_default(),
                                job.stage
                            )),
                    );
                }
            });
        });
    }

    fn layers_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.heading("Layers");
        ui.separator();
        if self.layers.is_empty() && self.loading.is_empty() {
            ui.add_space(12.0);
            ui.label(RichText::new("Drop .parquet files here\nor use Open…").weak());
        }

        let mut remove: Option<u64> = None;
        let mut fit_to: Option<[f64; 4]> = None;
        let mut info_open: Option<u64> = None;
        let mut load_all: Option<u64> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Top-most layer first in the list.
            let rebuilding = &self.rebuilding;
            for l in self.layers.iter_mut().rev() {
                let is_rebuilding = rebuilding.contains(&l.id);
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut l.style.visible, "");
                        let mut c = l.style.color;
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            l.style.color = c;
                        }
                        ui.label(RichText::new(&l.name).strong())
                            .on_hover_text(l.path.display().to_string());
                    });
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "{} {} · {}",
                                fmt_count(l.feature_count),
                                l.kind().label(),
                                l.crs.name
                            ))
                            .weak()
                            .small(),
                        );
                        if is_rebuilding {
                            ui.spinner();
                        }
                    });
                    ui.horizontal(|ui| {
                        match l.kind() {
                            crate::data::geometry::GeomKind::Point => {
                                ui.label("r:");
                                ui.add(
                                    egui::Slider::new(&mut l.style.point_radius_px, 0.5..=12.0)
                                        .show_value(false),
                                );
                            }
                            crate::data::geometry::GeomKind::Polygon => {
                                ui.label("fill:");
                                ui.add(
                                    egui::Slider::new(&mut l.style.fill_opacity, 0.0..=1.0)
                                        .show_value(false),
                                );
                                ui.label("w:");
                                ui.add(
                                    egui::Slider::new(&mut l.style.line_width_px, 0.0..=6.0)
                                        .show_value(false),
                                );
                            }
                            _ => {
                                ui.label("w:");
                                ui.add(
                                    egui::Slider::new(&mut l.style.line_width_px, 0.2..=8.0)
                                        .show_value(false),
                                );
                            }
                        }
                        ui.label("α:");
                        ui.add(
                            egui::Slider::new(&mut l.style.opacity, 0.0..=1.0).show_value(false),
                        );
                    });
                    if l.is_partial() {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "partial: {}/{} row groups loaded",
                                    l.loaded_rgs.len(),
                                    l.total_rgs()
                                ))
                                .color(Color32::from_rgb(242, 140, 26))
                                .small(),
                            );
                            if ui.small_button("Load all").clicked() {
                                load_all = Some(l.id);
                            }
                            if self.appending.contains(&l.id) {
                                ui.spinner();
                            }
                        });
                    }
                    if let Some(rg) = &l.rg_bboxes {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut l.style.show_rg_bboxes, "RG bboxes")
                                .on_hover_text(format!(
                                    "{} row groups — source: {}\navg overlap ×{:.1} {}",
                                    rg.boxes.len(),
                                    rg.source,
                                    rg.avg_overlap,
                                    if rg.avg_overlap > 4.0 {
                                        "(poorly clustered: consider a spatial-order rewrite)"
                                    } else {
                                        "(well clustered)"
                                    }
                                ));
                        });
                    }
                    ui.horizontal(|ui| {
                        if ui.small_button("Zoom to").clicked() {
                            fit_to = Some(l.bounds_world());
                        }
                        if ui.small_button("Info").clicked() {
                            info_open = Some(l.id);
                        }
                        if ui.small_button("Remove").clicked() {
                            remove = Some(l.id);
                        }
                        ui.label(
                            RichText::new(format!(
                                "read {} ms · build {} ms",
                                l.stats.read_ms, l.stats.build_ms
                            ))
                            .weak()
                            .small(),
                        );
                    });
                });
                ui.add_space(4.0);
            }
        });

        if let Some(id) = remove {
            self.layers.retain(|l| l.id != id);
            self.rebuilding.remove(&id);
            if self.selection.as_ref().map(|s| s.layer_id) == Some(id) {
                self.select(None);
            }
        }
        if let Some(b) = fit_to {
            self.fit_bounds = Some(b);
        }
        if info_open.is_some() {
            self.info_open = info_open;
        }
        if let Some(id) = load_all {
            let ctx = ui.ctx().clone();
            if let Some(l) = self.layers.iter().find(|l| l.id == id) {
                if !self.appending.contains(&id) {
                    let loaded: std::collections::HashSet<u32> =
                        l.loaded_rgs.iter().copied().collect();
                    let missing: Vec<u32> = (0..l.total_rgs() as u32)
                        .filter(|g| !loaded.contains(g))
                        .collect();
                    if !missing.is_empty() {
                        self.appending.insert(id);
                        loader::spawn_append(
                            LoaderHandle {
                                tx: self.load_tx.clone(),
                                egui_ctx: ctx,
                            },
                            l.id,
                            l.generation,
                            l.store.clone(),
                            l.crs.clone(),
                            self.display.clone(),
                            missing,
                        );
                    }
                }
            }
        }
    }

    fn info_window(&mut self, ctx: &egui::Context) {
        use crate::data::info::fmt_bytes;
        let Some(id) = self.info_open else { return };
        let Some(layer) = self.layers.iter().find(|l| l.id == id) else {
            self.info_open = None;
            return;
        };
        let info = &layer.info;
        let mut open = true;
        egui::Window::new(format!("File info — {}", layer.name))
            .id(egui::Id::new("file_info").with(id))
            .open(&mut open)
            .default_width(460.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.label(
                        RichText::new(&info.geo.version_label)
                            .strong()
                            .color(ui.visuals().hyperlink_color),
                    );
                    ui.add_space(6.0);

                    egui::Grid::new("gp_info").num_columns(2).striped(true).show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, k: &str, v: String| {
                            ui.label(RichText::new(k).strong());
                            ui.label(v);
                            ui.end_row();
                        };
                        row(ui, "geometry column", info.geo.primary_column.clone());
                        row(ui, "encoding", info.geo.encoding.clone());
                        if !info.geo.geometry_types.is_empty() {
                            row(ui, "geometry types", info.geo.geometry_types.join(", "));
                        }
                        row(ui, "CRS", format!("{} ({})", layer.crs.name, layer.crs.proj4));
                        if let Some(b) = info.geo.bbox {
                            row(
                                ui,
                                "bbox (metadata)",
                                format!("{:.5}, {:.5} — {:.5}, {:.5}", b[0], b[1], b[2], b[3]),
                            );
                        }
                        if let Some(c) = &info.geo.covering {
                            row(ui, "covering", c.clone());
                        }
                        if let Some(e) = &info.geo.edges {
                            row(ui, "edges", e.clone());
                        }
                        if let Some(rg) = &layer.rg_bboxes {
                            row(
                                ui,
                                "row-group bboxes",
                                format!(
                                    "{} boxes — {} — avg overlap ×{:.1} {}",
                                    rg.boxes.len(),
                                    rg.source,
                                    rg.avg_overlap,
                                    if rg.avg_overlap > 4.0 {
                                        "→ consider a spatial-order rewrite"
                                    } else {
                                        "(well clustered)"
                                    }
                                ),
                            );
                        }
                    });

                    ui.add_space(8.0);
                    ui.separator();
                    egui::Grid::new("file_info_grid").num_columns(2).striped(true).show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, k: &str, v: String| {
                            ui.label(RichText::new(k).strong());
                            ui.label(v);
                            ui.end_row();
                        };
                        row(ui, "path", layer.path.display().to_string());
                        row(ui, "file size", fmt_bytes(info.file_size));
                        row(ui, "rows", info.rows.to_string());
                        row(
                            ui,
                            "row groups",
                            format!(
                                "{} ({}–{} rows)",
                                info.row_groups, info.rg_rows_min, info.rg_rows_max
                            ),
                        );
                        row(
                            ui,
                            "data size",
                            format!(
                                "{} compressed / {} raw ({:.1}×)",
                                fmt_bytes(info.compressed_bytes),
                                fmt_bytes(info.uncompressed_bytes),
                                info.uncompressed_bytes.max(1) as f64
                                    / info.compressed_bytes.max(1) as f64
                            ),
                        );
                        row(
                            ui,
                            "parquet format",
                            format!("v{}", info.parquet_format_version),
                        );
                        if let Some(cb) = &info.created_by {
                            row(ui, "created by", cb.clone());
                        }
                    });

                    ui.add_space(8.0);
                    egui::CollapsingHeader::new(format!("Columns ({})", info.columns.len()))
                        .show(ui, |ui| {
                            egui::Grid::new("cols_grid").num_columns(3).striped(true).show(
                                ui,
                                |ui| {
                                    ui.label(RichText::new("name").weak());
                                    ui.label(RichText::new("type").weak());
                                    ui.label(RichText::new("compression").weak());
                                    ui.end_row();
                                    for c in &info.columns {
                                        let name = if c.is_geometry {
                                            RichText::new(&c.name).strong()
                                        } else {
                                            RichText::new(&c.name)
                                        };
                                        ui.label(name);
                                        let ty = match &c.logical {
                                            Some(l) if c.is_geometry => {
                                                format!("{} [{}]", c.arrow_type, l)
                                            }
                                            _ => c.arrow_type.clone(),
                                        };
                                        ui.label(ty);
                                        ui.label(&c.compression);
                                        ui.end_row();
                                    }
                                },
                            );
                        });

                    if let Some(json) = &info.geo.raw_geo_json {
                        egui::CollapsingHeader::new("Raw geo metadata").show(ui, |ui| {
                            if ui.small_button("Copy JSON").clicked() {
                                ui.ctx().copy_text(json.clone());
                            }
                            ui.add(
                                egui::TextEdit::multiline(&mut json.as_str())
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                    }
                });
            });
        if !open {
            self.info_open = None;
        }
    }

    fn attributes_panel(&mut self, ui: &mut egui::Ui) {
        let Some(sel) = self.selection.clone() else {
            return;
        };
        let Some(layer) = self.layers.iter().find(|l| l.id == sel.layer_id) else {
            return;
        };
        let geom_col = layer.store.geom_col;
        let layer_name = layer.name.clone();
        let mut open = true;
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Feature");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("✕").clicked() {
                    open = false;
                }
            });
        });
        ui.label(
            RichText::new(format!("{layer_name} · row {}", sel.feature.index))
                .weak()
                .small(),
        );
        ui.separator();
        match &self.selection_attrs {
            Some(batch) => {
                egui::ScrollArea::both().show(ui, |ui| {
                    egui::Grid::new("attrs")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            use arrow::util::display::{ArrayFormatter, FormatOptions};
                            let opts = FormatOptions::default().with_display_error(true);
                            for (i, field) in batch.schema().fields().iter().enumerate() {
                                ui.label(RichText::new(field.name()).strong());
                                if i == geom_col {
                                    ui.label(
                                        RichText::new(geom_summary(batch, geom_col)).weak(),
                                    );
                                } else {
                                    let col = batch.column(i);
                                    let text = ArrayFormatter::try_new(col.as_ref(), &opts)
                                        .map(|f| f.value(0).to_string())
                                        .unwrap_or_else(|_| "<?>".into());
                                    ui.label(text);
                                }
                                ui.end_row();
                            }
                        });
                });
            }
            None => {
                ui.label(RichText::new("attributes unavailable (see Problems)").weak());
            }
        }
        if !open {
            self.select(None);
        }
    }

    fn errors_window(&mut self, ctx: &egui::Context) {
        if !self.show_errors || self.errors.is_empty() {
            return;
        }
        let mut open = self.show_errors;
        egui::Window::new("Problems")
            .open(&mut open)
            .default_width(420.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(240.0)
                    .show(ui, |ui| {
                        for e in &self.errors {
                            ui.label(e);
                        }
                    });
                if ui.button("Clear").clicked() {
                    self.errors.clear();
                }
            });
        self.show_errors = open && !self.errors.is_empty();
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(w) = self.cursor_world {
                if let Some((lon, lat)) = world_to_lonlat(&self.display, w) {
                    ui.monospace(format!("{lon:.6}, {lat:.6}"));
                }
                if !self.display.crs.is_latlong {
                    let (x, y) = self.display.projected_from_world(w);
                    ui.monospace(format!("| {x:.1}, {y:.1} ({})", self.display.crs.name));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.monospace(format!("z {:.2}", self.camera.zoom));
                let pending = self.tiles.pending_count();
                if pending > 0 {
                    ui.monospace(format!("· {pending} tiles"));
                }
                let total: usize = self.layers.iter().map(|l| l.feature_count).sum();
                if total > 0 {
                    ui.monospace(format!("· {} features", fmt_count(total)));
                }
                let dt = ui.ctx().input(|i| i.unstable_dt);
                ui.monospace(format!("{:.1} ms", dt * 1000.0));
            });
        });
    }

    fn map_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let ppp = ctx.pixels_per_point();
        let vp = [rect.width() * ppp, rect.height() * ppp];

        // --- input ---
        if response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let d = response.drag_delta();
            self.camera.pan_px([d.x * ppp, d.y * ppp]);
        }
        let hover_px = response
            .hover_pos()
            .map(|p| [(p.x - rect.min.x) * ppp, (p.y - rect.min.y) * ppp]);
        self.cursor_world = hover_px.map(|p| self.camera.screen_to_world(p, vp));

        if response.hovered() {
            let (scroll, zoom_mult) = ctx.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let mut dz = scroll as f64 / 240.0;
            if zoom_mult != 1.0 {
                dz += (zoom_mult as f64).log2();
            }
            if dz != 0.0 {
                let cursor = hover_px.unwrap_or([vp[0] * 0.5, vp[1] * 0.5]);
                self.camera.zoom_about(dz, cursor, vp);
            }
        }
        if response.double_clicked() {
            if let Some(cursor) = hover_px {
                self.camera.zoom_about(1.0, cursor, vp);
            }
        }
        if response.clicked() {
            if let Some(w) = self.cursor_world {
                let tol = 6.0 * ppp as f64 / self.camera.scale();
                let sel = picking::pick(&self.layers, &self.display, w, tol);
                self.select(sel);
            }
        }

        if let Some(b) = self.fit_bounds.take() {
            self.camera.fit(b, vp, 40.0);
            self.pending_fit = false;
        }
        if self.pending_fit {
            if let Some(b) = self.union_bounds() {
                self.camera.fit(b, vp, 40.0);
                self.pending_fit = false;
            } else if self.layers.is_empty() {
                self.camera.fit(self.display.world_bounds(), vp, 40.0);
                self.pending_fit = false;
            }
        }

        // --- viewport tracking + row-group refinement (debounced) ---
        {
            let tl = self.camera.screen_to_world([0.0, 0.0], vp);
            let br = self.camera.screen_to_world(vp, vp);
            self.last_view_world = [tl[0], tl[1], br[0], br[1]];
            let now = ctx.input(|i| i.time);
            let pose = (self.camera.center, self.camera.zoom);
            if self.last_cam != Some(pose) {
                self.last_cam = Some(pose);
                self.cam_changed_at = now;
            } else if now - self.cam_changed_at > 0.35 {
                self.refine_partial_layers(&ctx);
            }
        }

        // --- background ---
        let dark = ui.visuals().dark_mode;
        let bg = if dark {
            Color32::from_rgb(24, 24, 28)
        } else {
            Color32::from_rgb(244, 243, 240)
        };
        ui.painter().rect_filled(rect, 0.0, bg);

        // --- build draw call ---
        self.tiles.poll();
        let tile_draws = match (self.basemap, self.display.is_mercator()) {
            (Some(src), true) => self.tiles.draws(src, &self.camera, vp),
            _ => Vec::new(),
        };
        let tile_uploads = self.tiles.take_uploads();
        let alive_tiles = self.tiles.alive_keys();
        let alive_layers: std::collections::HashSet<(u64, u64)> = self
            .layers
            .iter()
            .flat_map(|l| {
                (0..l.sections.len())
                    .map(|si| (section_key(l.id, si), l.generation))
                    .chain(std::iter::once((RG_OVERLAY_BASE | l.id, l.generation)))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut draws: Vec<LayerDraw> = Vec::new();
        if self.show_graticule && !self.graticule_chunks.is_empty() {
            let g = if dark { 0.42 } else { 0.55 };
            draws.push(LayerDraw {
                key: (GRATICULE_KEY, self.graticule_generation),
                composite_group: GRATICULE_KEY,
                chunks: self.graticule_chunks.clone(),
                style: DrawStyle {
                    fill_color: [0.0; 4],
                    line_color: [g, g, g, 0.45],
                    point_color: [0.0; 4],
                    line_half_width_px: 0.4,
                    point_radius_px: 0.0,
                },
            });
        }
        if self.show_coastline && !self.coastline_chunks.is_empty() {
            let c = if dark {
                [0.62, 0.66, 0.70]
            } else {
                [0.35, 0.38, 0.42]
            };
            draws.push(LayerDraw {
                key: (COASTLINE_KEY, self.graticule_generation),
                composite_group: COASTLINE_KEY,
                chunks: self.coastline_chunks.clone(),
                style: DrawStyle {
                    fill_color: [0.0; 4],
                    line_color: [c[0], c[1], c[2], 0.85],
                    point_color: [0.0; 4],
                    line_half_width_px: 0.5,
                    point_radius_px: 0.0,
                },
            });
        }
        for l in &self.layers {
            if !l.style.visible || self.rebuilding.contains(&l.id) {
                continue;
            }
            for (si, section) in l.sections.iter().enumerate() {
                draws.push(LayerDraw {
                    key: (section_key(l.id, si), l.generation),
                    composite_group: l.id,
                    chunks: section.chunks.clone(),
                    style: resolve_style(&l.style),
                });
            }
        }
        // Row-group bbox overlays (drawn above their layer).
        let mut rg_labels: Vec<(egui::Pos2, usize)> = Vec::new();
        for l in &self.layers {
            if !l.style.visible || !l.style.show_rg_bboxes || self.rebuilding.contains(&l.id) {
                continue;
            }
            let Some(rg) = &l.rg_bboxes else { continue };
            let cached = self
                .rg_overlays
                .get(&l.id)
                .filter(|(g, _, _)| *g == l.generation);
            let (chunks, anchors) = match cached {
                Some((_, c, a)) => (c.clone(), a.clone()),
                None => {
                    let (c, a) = build_rg_overlay(&l.crs, &self.display, &rg.boxes);
                    self.rg_overlays
                        .insert(l.id, (l.generation, c.clone(), a.clone()));
                    (c, a)
                }
            };
            draws.push(LayerDraw {
                key: (RG_OVERLAY_BASE | l.id, l.generation),
                composite_group: RG_OVERLAY_BASE | l.id,
                chunks,
                style: DrawStyle {
                    fill_color: [0.0; 4],
                    line_color: [0.95, 0.55, 0.10, 0.9],
                    point_color: [0.0; 4],
                    line_half_width_px: 0.8,
                    point_radius_px: 0.0,
                },
            });
            if anchors.len() <= 64 {
                for (i, w) in anchors.iter().enumerate() {
                    let s = self.camera.world_to_screen(*w, vp);
                    let pos = egui::pos2(rect.min.x + s[0] / ppp, rect.min.y + s[1] / ppp);
                    if rect.contains(pos) {
                        rg_labels.push((pos, i));
                    }
                }
            }
        }
        self.rg_overlays.retain(|id, _| {
            self.layers
                .iter()
                .any(|l| l.id == *id && l.style.show_rg_bboxes)
        });

        if let (Some(chunks), Some(_)) = (&self.highlight_chunks, &self.selection) {
            draws.push(LayerDraw {
                key: (HIGHLIGHT_KEY, self.selection_generation),
                composite_group: HIGHLIGHT_KEY,
                chunks: chunks.clone(),
                style: DrawStyle {
                    fill_color: [1.0, 0.75, 0.05, 0.4],
                    line_color: [1.0, 0.8, 0.1, 1.0],
                    point_color: [1.0, 0.8, 0.1, 1.0],
                    line_half_width_px: 1.8,
                    point_radius_px: 6.0,
                },
            });
        }

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            MapCallback {
                camera: self.camera,
                viewport_px: vp,
                tile_draws,
                tile_uploads,
                alive_tiles,
                alive_layers,
                layers: draws,
                background: [0.0; 4],
            },
        ));

        // --- overlays ---
        for (pos, i) in &rg_labels {
            ui.painter().text(
                *pos,
                egui::Align2::CENTER_CENTER,
                format!("rg {i}"),
                egui::FontId::monospace(10.0),
                Color32::from_rgb(242, 140, 26),
            );
        }
        if let (Some(src), true) = (self.basemap, self.display.is_mercator()) {
            ui.painter().text(
                rect.right_bottom() - egui::vec2(6.0, 4.0),
                egui::Align2::RIGHT_BOTTOM,
                TILE_SOURCES[src].attribution,
                egui::FontId::proportional(10.0),
                if dark {
                    Color32::from_white_alpha(120)
                } else {
                    Color32::from_black_alpha(140)
                },
            );
        }
    }
}

/// Build graticule line meshes (meridians/parallels every 15°) for the given
/// display projection. Cheap: a few thousand densified vertices.
fn build_graticule(display: &DisplayCrs) -> Arc<Vec<crate::data::geometry::ChunkMesh>> {
    let wgs = Crs::wgs84();
    let tr = BulkTransformer::new(&wgs, display);
    let max_lat: f64 = if display.is_mercator() { 85.0 } else { 90.0 };
    let mut mb = MeshBuilder::default();

    let mut add_line = |pts: &[(f64, f64)]| {
        let coords: Vec<geo_types::Coord<f64>> = pts
            .iter()
            .filter_map(|&(lon, lat)| {
                let (mut x, mut y) = (lon, lat);
                if !tr.apply(&mut x, &mut y) {
                    return None;
                }
                let w = display.world_from_projected(x, y);
                (w[0].is_finite() && w[1].is_finite())
                    .then_some(geo_types::Coord { x: w[0], y: w[1] })
            })
            .collect();
        if coords.len() >= 2 {
            mb.add(
                &geo_types::Geometry::LineString(geo_types::LineString(coords)),
                crate::data::geometry::FeatureRef::INVALID,
            );
        }
    };

    let mut pts: Vec<(f64, f64)> = Vec::new();
    for lon_i in (-180..=180).step_by(15) {
        pts.clear();
        let mut lat = -max_lat;
        while lat <= max_lat + 1e-9 {
            pts.push((lon_i as f64, lat));
            lat += 2.0;
        }
        add_line(&pts);
    }
    for lat_i in (-75..=75).step_by(15) {
        pts.clear();
        let mut lon = -180.0;
        while lon <= 180.0 + 1e-9 {
            pts.push((lon, lat_i as f64));
            lon += 2.0;
        }
        add_line(&pts);
    }
    drop(add_line);
    Arc::new(mb.finish())
}

/// Stable renderer key for a layer section.
fn section_key(layer_id: u64, section: usize) -> u64 {
    layer_id | ((section as u64 + 1) << 40)
}

/// Build the row-group bbox overlay: densified rectangles projected from
/// the layer's data CRS into the current display projection, plus world
/// anchors for index labels.
pub(crate) fn build_rg_overlay(
    layer_crs: &Crs,
    display: &DisplayCrs,
    boxes: &[[f64; 4]],
) -> (Arc<Vec<crate::data::geometry::ChunkMesh>>, Vec<[f64; 2]>) {
    let tr = BulkTransformer::new(layer_crs, display);
    let mut mb = MeshBuilder::default();
    let mut anchors: Vec<[f64; 2]> = Vec::new();
    const N: usize = 16;
    for b in boxes {
        let mut ring: Vec<geo_types::Coord<f64>> = Vec::with_capacity(4 * N + 1);
        let mut push = |x: f64, y: f64, ring: &mut Vec<geo_types::Coord<f64>>| {
            let (mut px, mut py) = (x, y);
            if tr.apply(&mut px, &mut py) {
                let w = display.world_from_projected(px, py);
                if w[0].is_finite() && w[1].is_finite() {
                    ring.push(geo_types::Coord { x: w[0], y: w[1] });
                }
            }
        };
        for i in 0..N {
            let t = i as f64 / N as f64;
            push(b[0] + (b[2] - b[0]) * t, b[1], &mut ring);
        }
        for i in 0..N {
            let t = i as f64 / N as f64;
            push(b[2], b[1] + (b[3] - b[1]) * t, &mut ring);
        }
        for i in 0..N {
            let t = i as f64 / N as f64;
            push(b[2] - (b[2] - b[0]) * t, b[3], &mut ring);
        }
        for i in 0..N {
            let t = i as f64 / N as f64;
            push(b[0], b[3] - (b[3] - b[1]) * t, &mut ring);
        }
        if let Some(first) = ring.first().copied() {
            ring.push(first);
        }
        if ring.len() >= 2 {
            // Anchor: projected box center.
            let (mut cx, mut cy) = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5);
            if tr.apply(&mut cx, &mut cy) {
                anchors.push(display.world_from_projected(cx, cy));
            } else {
                anchors.push([f64::NAN, f64::NAN]);
            }
            mb.add(
                &geo_types::Geometry::LineString(geo_types::LineString(ring)),
                crate::data::geometry::FeatureRef::INVALID,
            );
        }
    }
    (Arc::new(mb.finish()), anchors)
}

fn resolve_style(s: &crate::data::layer::LayerStyle) -> DrawStyle {
    let rgba = egui::Rgba::from(s.color);
    let (r, g, b) = (rgba.r(), rgba.g(), rgba.b());
    DrawStyle {
        fill_color: [r, g, b, s.fill_opacity * s.opacity],
        line_color: [r * 0.55, g * 0.55, b * 0.55, s.opacity],
        point_color: [r, g, b, s.opacity],
        line_half_width_px: (s.line_width_px * 0.5).max(0.01),
        point_radius_px: s.point_radius_px.max(0.1),
    }
}

fn geom_summary(batch: &arrow::record_batch::RecordBatch, geom_col: usize) -> String {
    let wkb = crate::data::loader::BinCol::new(batch.column(geom_col).as_ref())
        .and_then(|b| b.value(0).map(|v| v.to_vec()));
    match wkb.and_then(|w| crate::data::loader::decode_wkb(&w)) {
        Some(g) => {
            use geo::CoordsIter;
            let n = g.coords_count();
            let ty = match g {
                geo_types::Geometry::Point(_) => "Point",
                geo_types::Geometry::MultiPoint(_) => "MultiPoint",
                geo_types::Geometry::LineString(_) => "LineString",
                geo_types::Geometry::MultiLineString(_) => "MultiLineString",
                geo_types::Geometry::Polygon(_) => "Polygon",
                geo_types::Geometry::MultiPolygon(_) => "MultiPolygon",
                _ => "Geometry",
            };
            format!("{ty} · {n} vertices")
        }
        None => "<invalid>".into(),
    }
}

fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

impl ViewerApp {
    /// Once a layer's buffers are resident on the GPU, drop the CPU copy of
    /// its fill/line arrays (they are only needed for the initial upload;
    /// projection rebuilds re-stream from disk). Point instances and refs
    /// stay: they are the point pick index. Saves ~2 GB on state-scale
    /// polygon layers.
    fn strip_uploaded_cpu_meshes(&mut self, frame: &eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };
        let renderer = rs.renderer.read();
        let Some(res) = renderer
            .callback_resources
            .get::<crate::map::renderer::MapResources>()
        else {
            return;
        };
        for l in &mut self.layers {
            if self.rebuilding.contains(&l.id) {
                continue;
            }
            for (si, section) in l.sections.iter_mut().enumerate() {
                let key = (section_key(l.id, si), l.generation);
                if self.stripped.contains(&key) || !res.has_layer_uploaded(key) {
                    continue;
                }
                if let Some(chunks) = Arc::get_mut(&mut section.chunks) {
                    let mut freed = 0usize;
                    for c in chunks.iter_mut() {
                        freed += c.fill_vertices.capacity() * 8
                            + c.fill_indices.capacity() * 4
                            + c
                                .lines
                                .iter()
                                .map(|l| l.segments.capacity() * 16)
                                .sum::<usize>();
                        c.fill_vertices = Vec::new();
                        c.fill_indices = Vec::new();
                        c.lines = Default::default();
                    }
                    if freed > 1 << 20 {
                        log::info!(
                            "{}: freed {:.0} MB of CPU mesh after GPU upload",
                            l.name,
                            freed as f64 / (1 << 20) as f64
                        );
                    }
                    self.stripped.insert(key);
                }
            }
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_loader();
        self.strip_uploaded_cpu_meshes(frame);

        // Drag & drop.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        for p in dropped {
            self.enqueue_load(p, &ctx);
        }

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("layers")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| self.layers_panel(ui));
        if self.selection.is_some() {
            egui::Panel::right("attributes")
                .resizable(true)
                .default_size(300.0)
                .show(ui, |ui| self.attributes_panel(ui));
        }
        self.errors_window(&ctx);
        self.info_window(&ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.map_panel(ui));

        if !self.loading.is_empty() || !self.rebuilding.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::renderer::{MapCallback, MapResources, MSAA_SAMPLES};
    use eframe::egui_wgpu::{self, wgpu, CallbackTrait};

    /// Render the world graticule headless in both built-in world
    /// projections: verifies each produces a sensible world frame.
    #[test]
    fn wintri_graticule_renders() {
        graticule_case(DisplayCrs::winkel_tripel(), "wintri");
    }

    #[test]
    fn hobo_dyer_graticule_renders() {
        graticule_case(DisplayCrs::hobo_dyer(), "hobo_dyer");
    }

    fn graticule_case(display: DisplayCrs, tag: &str) {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (512u32, 320u32);

        let mut resources = egui_wgpu::CallbackResources::default();
        resources.insert(MapResources::new(&device, format));

        let chunks = build_graticule(&display);
        assert!(!chunks.is_empty());

        let mut camera = crate::map::camera::Camera::default();
        camera.fit(display.world_bounds(), [w as f32, h as f32], 10.0);

        let cb = MapCallback {
            camera,
            viewport_px: [w as f32, h as f32],
            tile_draws: vec![],
            tile_uploads: vec![],
            alive_tiles: Default::default(),
            alive_layers: Default::default(),
            layers: vec![
                crate::map::renderer::LayerDraw {
                    key: (1, 0),
                    composite_group: 1,
                    chunks,
                    style: crate::map::renderer::DrawStyle {
                        fill_color: [0.0; 4],
                        line_color: [0.35, 0.35, 0.35, 1.0],
                        point_color: [0.0; 4],
                        line_half_width_px: 0.5,
                        point_radius_px: 0.0,
                    },
                },
                crate::map::renderer::LayerDraw {
                    key: (2, 0),
                    composite_group: 2,
                    chunks: crate::data::coastline::build_coastline(&display),
                    style: crate::map::renderer::DrawStyle {
                        fill_color: [0.0; 4],
                        line_color: [1.0, 1.0, 1.0, 1.0],
                        point_color: [0.0; 4],
                        line_half_width_px: 0.6,
                        point_radius_px: 0.0,
                    },
                },
            ],
            background: [0.0; 4],
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
        let msaa = mk_tex(MSAA_SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = mk_tex(1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let msaa_view = msaa.create_view(&Default::default());
        let resolve_view = resolve.create_view(&Default::default());

        let mut encoder = device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: 1.0 };
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
            let info = egui::PaintCallbackInfo {
                viewport: egui::Rect::from_min_size(Default::default(), egui::vec2(w as f32, h as f32)),
                clip_rect: egui::Rect::from_min_size(Default::default(), egui::vec2(w as f32, h as f32)),
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
                texture: &resolve,
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

        if std::env::var("DUMP_PNG").is_ok() {
            image::RgbaImage::from_raw(w, h, data.clone())
                .unwrap()
                .save(format!("/tmp/{tag}_graticule.png"))
                .ok();
        }
        let lit = data.chunks_exact(4).filter(|px| px[0] > 100).count();
        assert!(lit > 3_000, "graticule pixels: {lit}");
    }
}
