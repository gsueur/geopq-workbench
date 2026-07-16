use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use eframe::egui::{self, Color32, RichText};
use eframe::egui_wgpu;

use crate::data::crs::{world_to_lonlat, BulkTransformer, Crs, DisplayCrs, DisplayKind};
use crate::data::source::Source;
use crate::data::geometry::MeshBuilder;
use crate::data::layer::{palette_color, VectorLayer};
use crate::data::loader::{self, LoadMsg, LoaderHandle};
use crate::map::camera::Camera;
use crate::map::renderer::{DrawStyle, LayerDraw, MapCallback, MapResources};
use crate::map::tiles::{TileCache, TILE_SOURCES};
use crate::picking::{self, Selection};

const HIGHLIGHT_KEY: u64 = u64::MAX;
/// Separate render layer for the SQL checked-rows selection, so picking a
/// feature on the map doesn't wipe it.
const SQL_HIGHLIGHT_KEY: u64 = u64::MAX - 1;
const GRATICULE_KEY: u64 = u64::MAX - 2;
const COASTLINE_KEY: u64 = u64::MAX - 3;
/// Row-group bbox overlays: key = RG_OVERLAY_BASE | layer id.
const RG_OVERLAY_BASE: u64 = 1 << 62;

struct LoadingJob {
    label: String,
    frac: f32,
    stage: String,
    /// Display generation this job builds geometry for; a mismatch at
    /// arrival means the projection changed mid-load and the layer must
    /// rebuild (its world coordinates are in the old display's frame).
    display_gen: u64,
    /// Set by the status-bar stop button; the loader checks it between
    /// row groups and batches.
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

enum OptMsg {
    Progress(f32, String),
    Done(Box<crate::data::optimize::OptimizeReport>, PathBuf),
    Failed(String),
    /// Distinct-value counts for partition-field candidates.
    Cardinalities(std::collections::HashMap<String, usize>),
}

/// State of the per-layer "Optimize" export dialog (one at a time).
/// Partition mode chosen in the optimize dialog.
#[derive(PartialEq, Clone, Copy)]
enum PartMode {
    None,
    Fields,
    AdaptiveH3,
}

struct OptimizeState {
    layer_id: u64,
    layer_name: String,
    src: Source,
    epsg: Option<u32>,
    /// Layer CRS, for viewport-rect conversion at export time.
    crs: Crs,
    viewport_only: bool,
    opts: crate::data::optimize::OptimizeOptions,
    running: bool,
    progress: (f32, String),
    report: Option<(crate::data::optimize::OptimizeReport, PathBuf)>,
    error: Option<String>,
    /// Admin attribution: boundary layer + its value column + output name.
    admin_layer: Option<u64>,
    admin_column: String,
    admin_out: String,
    part_mode: PartMode,
    part_fields: Vec<String>,
    adaptive_target: usize,
    /// Distinct counts per candidate partition field (computed on demand).
    cardinalities: Option<std::collections::HashMap<String, usize>>,
    card_pending: bool,
}

pub struct ViewerApp {
    camera: Camera,
    display: DisplayCrs,
    layers: Vec<VectorLayer>,
    rebuilding: HashSet<u64>,
    selection: Option<Selection>,
    /// Single-row record batch for the selected feature, fetched lazily
    /// from the layer's FeatureStore (capped to [`ATTR_COLS_CAP`] columns).
    selection_attrs: Option<arrow::record_batch::RecordBatch>,
    /// (shown, total) when the attribute fetch was column-capped.
    attrs_truncated: Option<(usize, usize)>,
    selection_generation: u64,
    highlight_chunks: Option<Arc<Vec<crate::data::geometry::ChunkMesh>>>,
    /// SQL checked-rows selection, rendered independently of the picked
    /// feature so map clicks don't wipe it.
    sql_highlight_chunks: Option<Arc<Vec<crate::data::geometry::ChunkMesh>>>,
    sql_highlight_generation: u64,
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
    opt_tx: Sender<OptMsg>,
    opt_rx: Receiver<OptMsg>,
    optimize: Option<OptimizeState>,
    loading: HashMap<u64, LoadingJob>,
    /// Styles to apply when a context-restored layer finishes loading
    /// (keyed by job id); also suppresses fit-on-first-load.
    pending_styles: HashMap<u64, crate::data::layer::LayerStyle>,
    next_job: u64,
    next_layer_id: u64,
    palette_idx: usize,
    pending_fit: bool,
    fit_bounds: Option<[f64; 4]>,
    /// Pick the display projection automatically from the first loaded
    /// layer (data CRS if projected, extent-based equal-area otherwise);
    /// turned off by any manual projection choice.
    auto_projection: bool,
    /// Layers with a row-group append in flight.
    appending: HashSet<u64>,
    /// Cancel flags of in-flight appends (layer id -> flag).
    append_cancel: HashMap<u64, Arc<std::sync::atomic::AtomicBool>>,
    /// Layers whose refinement was stopped by the user: paused until the
    /// camera moves again (else the same viewport would respawn it).
    refine_hold: HashSet<u64>,
    /// Last camera pose + when it last changed (for refinement debounce).
    last_cam: Option<([f64; 2], f64)>,
    cam_changed_at: f64,
    /// Current viewport in world coords (for load-time pruning).
    last_view_world: [f64; 4],

    errors: Vec<String>,
    show_errors: bool,
    /// URL entry dialog (Some = dialog open): text, selected AWS
    /// profile, discovered profiles, and custom S3 endpoint.
    url_input: Option<(String, Option<String>, Vec<String>, String)>,
    info_open: Option<u64>,
    /// Layer generations whose CPU-side fill/line arrays were freed after
    /// GPU upload (points are kept for picking).
    stripped: HashSet<(u64, u64)>,
    epsg_input: String,
    cursor_world: Option<[f64; 2]>,
    sql: crate::sql::console::SqlConsole,
    /// Layer-filter dialog (Some = open).
    filter_dialog: Option<FilterDialog>,
    /// Data-driven styling dialog (Some = open).
    style_dialog: Option<StyleDialog>,
    /// Category-value fetches for the styling dialog.
    cat_tx: Sender<(u64, String, Result<Vec<String>, String>)>,
    cat_rx: Receiver<(u64, String, Result<Vec<String>, String>)>,
    /// Classification runs for the styling dialog (breaks from loaded rows).
    class_tx: Sender<(u64, String, Result<Vec<f64>, String>)>,
    class_rx: Receiver<(u64, String, Result<Vec<f64>, String>)>,
    filter_tx: Sender<crate::sql::engine::FilterMsg>,
    filter_rx: Receiver<crate::sql::engine::FilterMsg>,
    /// Layers with a filter computation in flight.
    filter_pending: HashSet<u64>,
    /// Filter-dialog "Test" runs (separate channel: results go to the
    /// dialog, not to the layer).
    test_tx: Sender<crate::sql::engine::FilterMsg>,
    test_rx: Receiver<crate::sql::engine::FilterMsg>,
    /// Filters to apply once a context-restored layer finishes loading.
    pending_filters: HashMap<u64, String>,
    /// Async feature picking (remote layers turn picks into network reads,
    /// so the whole pipeline runs off the UI thread).
    pick_tx: Sender<PickMsg>,
    pick_rx: Receiver<PickMsg>,
    /// Monotonic pick id; results from superseded jobs are dropped.
    pick_job: u64,
    /// A pick is in flight (status-bar spinner).
    pick_pending: bool,
    /// Repository browser dialog (Some = open).
    repo_browser: Option<RepoBrowser>,
    repo_tx: Sender<RepoMsg>,
    repo_rx: Receiver<RepoMsg>,
    /// Names to give layers when their load finishes (keyed by job id) —
    /// repository themes would otherwise all be called "buildings".
    pending_names: HashMap<u64, String>,
    /// Bumped on every display projection change; load jobs record it to
    /// detect a switch that happened while they were building.
    display_gen: u64,
}

/// Browser over external GeoParquet repositories (parquetry layout):
/// snapshot picker, discovered datasets, per-theme loading.
struct RepoBrowser {
    repos: Vec<crate::data::repo::Repository>,
    sel_repo: usize,
    snapshots: Vec<crate::data::repo::Snapshot>,
    sel_snapshot: usize,
    /// None = discovery in flight.
    datasets: Option<Result<Vec<crate::data::repo::Dataset>, String>>,
    filter: String,
    /// Country filter over the dataset list; empty = all.
    country: String,
    /// Selected dataset path + its manifest (None = fetch in flight).
    selected: Option<(usize, Option<Result<crate::data::repo::Manifest, String>>)>,
    /// Themes ticked for loading in the selected dataset.
    checked: std::collections::HashSet<String>,
    /// Unix seconds the dataset list was cached at (None = fetched live).
    cache_age: Option<u64>,
    /// Add-repository row (name, base URL).
    add: (String, String),
    /// Drops stale async results after repo/snapshot switches.
    generation: u64,
}

enum RepoMsg {
    Snapshots(u64, Result<Vec<crate::data::repo::Snapshot>, String>),
    /// Dataset list + the cache timestamp it came from (None = live fetch).
    Datasets(
        u64,
        Result<Vec<crate::data::repo::Dataset>, String>,
        Option<u64>,
    ),
    Manifest(u64, Result<crate::data::repo::Manifest, String>),
}

/// Cap on attribute columns fetched for the feature info panel. Wide
/// files (time series in columns: thousands of fields) would otherwise
/// need one range request per column chunk on remote sources.
const ATTR_COLS_CAP: usize = 256;

/// Result of an async pick job.
struct PickMsg {
    job: u64,
    sel: Option<Selection>,
    attrs: Option<arrow::record_batch::RecordBatch>,
    /// (shown, total) when the attribute fetch was column-capped.
    truncated: Option<(usize, usize)>,
}

/// Draw-order category of a geometry kind: polygons under lines under
/// points (mixed/unknown sink to the bottom with the polygons).
fn kind_rank(k: crate::data::geometry::GeomKind) -> u8 {
    use crate::data::geometry::GeomKind;
    match k {
        GeomKind::Line => 1,
        GeomKind::Point => 2,
        _ => 0,
    }
}

/// Small painter-drawn geometry-type glyph (font-independent) for the
/// layer rows: filled square = polygons, diagonal = lines, dot = points.
fn geom_kind_icon(ui: &mut egui::Ui, kind: crate::data::geometry::GeomKind) {
    use crate::data::geometry::GeomKind;
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
    let color = ui.visuals().weak_text_color();
    let p = ui.painter();
    let c = rect.center();
    match kind {
        GeomKind::Point => {
            p.circle_filled(c, 3.0, color);
        }
        GeomKind::Line => {
            p.line_segment(
                [
                    rect.left_bottom() + egui::vec2(2.5, -2.5),
                    rect.right_top() + egui::vec2(-2.5, 2.5),
                ],
                egui::Stroke::new(2.0, color),
            );
        }
        GeomKind::Polygon => {
            p.rect_filled(
                egui::Rect::from_center_size(c, egui::vec2(9.0, 9.0)),
                1.0,
                color,
            );
        }
        GeomKind::Mixed | GeomKind::Unknown => {
            p.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(9.0, 9.0)),
                1.0,
                egui::Stroke::new(1.5, color),
                egui::StrokeKind::Inside,
            );
            p.circle_filled(c, 1.8, color);
        }
    }
    resp.on_hover_text(kind.label());
}

/// Fetch the picked feature's attribute row, capping very wide schemas to
/// the first [`ATTR_COLS_CAP`] columns (+ geometry). Worker-thread only.
fn fetch_pick_attrs(
    layers: &[picking::PickLayer],
    s: &Selection,
) -> (
    Option<arrow::record_batch::RecordBatch>,
    Option<(usize, usize)>,
) {
    let Some(l) = layers.iter().find(|l| l.id == s.layer_id) else {
        return (None, None);
    };
    let total = l.store.schema.fields().len();
    let (fetched, truncated) = if total > ATTR_COLS_CAP {
        let mut cols: Vec<usize> = (0..ATTR_COLS_CAP).collect();
        if l.store.geom_col >= ATTR_COLS_CAP {
            cols.push(l.store.geom_col);
        }
        let n = cols.len();
        (
            l.store.fetch(&[s.feature.index], Some(&cols)),
            Some((n, total)),
        )
    } else {
        (l.store.fetch(&[s.feature.index], None), None)
    };
    match fetched {
        Ok(batches) => (batches.into_iter().next(), truncated),
        Err(e) => {
            log::warn!("attribute fetch failed: {e}");
            (None, None)
        }
    }
}

/// Data-driven styling editor for one layer.
struct StyleDialog {
    layer_id: u64,
    column: String,
    /// Whether the selected column is numeric (graduated) or text
    /// (categorical).
    numeric: bool,
    ramp: crate::data::layer::Ramp,
    method: crate::data::layer::ClassMethod,
    /// Equal-interval bounds (from column statistics, editable).
    min: f64,
    max: f64,
    /// Computed class breaks (None = classification in flight for
    /// data-dependent methods).
    breaks: Option<Result<Vec<f64>, String>>,
    /// Top values for categorical columns (None = fetch in flight).
    categories: Option<Result<Vec<String>, String>>,
}

/// Editor for a layer's persistent SQL filter.
struct FilterDialog {
    layer_id: u64,
    text: String,
    ac: crate::sql::console::AcState,
    /// Test run state: predicate tested, in-flight flag, and the outcome
    /// (kept so Apply can reuse the computed rows without a second run).
    test_pred: String,
    testing: bool,
    test: Option<Result<crate::sql::engine::FilterRows, String>>,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<Source>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("wgpu render state (run with the wgpu backend)");
        rs.renderer
            .write()
            .callback_resources
            .insert(MapResources::new(&rs.device, rs.target_format));

        let (load_tx, load_rx) = channel();
        let (opt_tx, opt_rx) = channel();
        let (filter_tx, filter_rx) = channel();
        let (test_tx, test_rx) = channel();
        let (pick_tx, pick_rx) = channel();
        let (repo_tx, repo_rx) = channel();
        let (cat_tx, cat_rx) = channel();
        let (class_tx, class_rx) = channel();
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
            attrs_truncated: None,
            selection_generation: 0,
            highlight_chunks: None,
            sql_highlight_chunks: None,
            sql_highlight_generation: 0,
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
            opt_tx,
            opt_rx,
            optimize: None,
            loading: HashMap::new(),
            pending_styles: HashMap::new(),
            next_job: 0,
            next_layer_id: 0,
            palette_idx: 0,
            pending_fit: true,
            fit_bounds: None,
            auto_projection: true,
            appending: HashSet::new(),
            append_cancel: HashMap::new(),
            refine_hold: HashSet::new(),
            last_cam: None,
            cam_changed_at: 0.0,
            last_view_world: [-10.0, -10.0, 10.0, 10.0],
            errors: Vec::new(),
            show_errors: false,
            url_input: None,
            info_open: None,
            stripped: HashSet::new(),
            epsg_input: String::new(),
            cursor_world: None,
            sql: crate::sql::console::SqlConsole::new(),
            filter_dialog: None,
            style_dialog: None,
            cat_tx,
            cat_rx,
            class_tx,
            class_rx,
            filter_tx,
            filter_rx,
            filter_pending: HashSet::new(),
            test_tx,
            test_rx,
            pending_filters: HashMap::new(),
            pick_tx,
            pick_rx,
            pick_job: 0,
            pick_pending: false,
            repo_browser: None,
            repo_tx,
            repo_rx,
            pending_names: HashMap::new(),
            display_gen: 0,
        };
        for f in files {
            app.enqueue_load(f, &cc.egui_ctx);
        }
        app
    }

    fn enqueue_load(&mut self, source: Source, ctx: &egui::Context) -> u64 {
        // Auto-projection only applies to the very first layer of a session
        // (evaluated before this job is registered).
        let auto_project = self.auto_projection
            && self.layers.is_empty()
            && self.loading.is_empty()
            && self.pending_styles.is_empty();
        let job = self.next_job;
        self.next_job += 1;
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;
        let color = palette_color(self.palette_idx);
        self.palette_idx += 1;
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.loading.insert(
            job,
            LoadingJob {
                label: source.label(),
                frac: 0.0,
                stage: "queued".into(),
                display_gen: self.display_gen,
                cancel: Arc::clone(&cancel),
            },
        );
        loader::spawn_load(
            LoaderHandle {
                tx: self.load_tx.clone(),
                egui_ctx: ctx.clone(),
            },
            job,
            layer_id,
            source,
            self.display.clone(),
            color,
            self.last_view_world,
            auto_project,
            cancel,
            // Context-restored layers carry their styling into the first
            // build (pending style registered by the caller right after).
            None,
        );
        job
    }

    /// Highlight the SQL-checked features on the map (own render layer —
    /// independent of the picked-feature highlight). The geometries are
    /// whatever the query returned (possibly computed, e.g. centroids), so
    /// the highlight is built from them directly instead of going through
    /// layer picking.
    fn apply_sql_selection(&mut self, crs: &Crs, geoms: Vec<geo_types::Geometry<f64>>) {
        self.sql_highlight_generation += 1;
        if geoms.is_empty() {
            self.sql_highlight_chunks = None;
            return;
        }
        let mut mb = MeshBuilder::default();
        for g in geoms {
            let world = crate::picking::to_world_geom(g, crs, &self.display);
            mb.add(&world, crate::data::geometry::FeatureRef::INVALID);
        }
        self.sql_highlight_chunks = Some(Arc::new(mb.finish()));
    }

    /// The single-layer SQL registration used by filter computations.
    fn sql_layer_of(&self, layer_id: u64) -> Option<crate::sql::engine::SqlLayer> {
        let l = self.layers.iter().find(|l| l.id == layer_id)?;
        Some(crate::sql::engine::SqlLayer {
            table: crate::sql::engine::table_name(&l.name),
            store: Arc::clone(&l.store),
            crs: l.crs.clone(),
            rg_bboxes: l
                .rg_bboxes
                .as_ref()
                .filter(|r| r.boxes.len() == l.store.rg_starts().len().saturating_sub(1))
                .map(|r| Arc::new(r.boxes.clone())),
        })
    }

    /// Kick off a layer-filter computation (SQL predicate → matching row
    /// ranges) on a background thread.
    fn start_layer_filter(&mut self, layer_id: u64, predicate: String, ctx: &egui::Context) {
        let Some(sql_layer) = self.sql_layer_of(layer_id) else {
            return;
        };
        self.filter_pending.insert(layer_id);
        let egui_ctx = ctx.clone();
        crate::sql::engine::spawn_row_filter(
            layer_id,
            sql_layer,
            predicate,
            self.filter_tx.clone(),
            move || egui_ctx.request_repaint(),
        );
    }

    /// Apply computed filter ranges: the layer's loaded state becomes
    /// exactly the matching rows (with infinite coverage rects so viewport
    /// refinement never adds rows back), then the geometry rebuilds.
    fn poll_filters(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.filter_rx.try_recv() {
            self.filter_pending.remove(&msg.layer_id);
            match msg.result {
                Err(e) => {
                    let name = self
                        .layers
                        .iter()
                        .find(|l| l.id == msg.layer_id)
                        .map(|l| l.name.clone())
                        .unwrap_or_default();
                    self.push_error(format!("filter on {name}: {e}"));
                }
                Ok(rows) => self.apply_filter_rows(msg.layer_id, msg.predicate, rows, ctx),
            }
        }
        // Test runs: results land in the dialog, the layer is untouched.
        while let Ok(msg) = self.test_rx.try_recv() {
            if let Some(d) = &mut self.filter_dialog {
                if d.layer_id == msg.layer_id && d.test_pred == msg.predicate {
                    d.testing = false;
                    d.test = Some(msg.result);
                }
            }
        }
    }

    /// Make computed filter rows the layer's working subset and rebuild.
    fn apply_filter_rows(
        &mut self,
        layer_id: u64,
        predicate: String,
        rows: crate::sql::engine::FilterRows,
        ctx: &egui::Context,
    ) {
        let display = self.display.clone();
        let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) else {
            return;
        };
        const INF: [f64; 4] =
            [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY];
        let starts = l.store.rg_starts();
        l.loaded = rows
            .per_group
            .iter()
            .enumerate()
            .map(|(g, ranges)| {
                let n = (starts[g + 1] - starts[g]) as u32;
                if ranges.len() == 1 && ranges[0] == (0, n) {
                    crate::data::layer::GroupLoad::Full
                } else {
                    crate::data::layer::GroupLoad::Rows {
                        ranges: ranges.clone(),
                        rect: INF,
                    }
                }
            })
            .collect();
        l.filter = Some(crate::data::layer::LayerFilter {
            sql: predicate,
            matched: rows.matched,
        });
        l.feature_count = rows.matched;
        l.generation += 1;
        self.rebuilding.insert(l.id);
        loader::spawn_rebuild(
            LoaderHandle {
                tx: self.load_tx.clone(),
                egui_ctx: ctx.clone(),
            },
            l.id,
            l.generation,
            Arc::clone(&l.store),
            l.crs.clone(),
            display,
            l.loaded.clone(),
            l.style.style_by.clone(),
        );
    }

    /// Remove a layer's filter: everything loads again.
    fn clear_layer_filter(&mut self, layer_id: u64, ctx: &egui::Context) {
        let display = self.display.clone();
        let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) else {
            return;
        };
        let n_groups = l.store.rg_starts().len().saturating_sub(1);
        l.filter = None;
        l.loaded = vec![crate::data::layer::GroupLoad::Full; n_groups];
        l.feature_count = l.store.total_rows() as usize;
        l.generation += 1;
        self.rebuilding.insert(l.id);
        loader::spawn_rebuild(
            LoaderHandle {
                tx: self.load_tx.clone(),
                egui_ctx: ctx.clone(),
            },
            l.id,
            l.generation,
            Arc::clone(&l.store),
            l.crs.clone(),
            display,
            l.loaded.clone(),
            l.style.style_by.clone(),
        );
    }

    /// The layer-filter dialog: predicate editor with autocomplete.
    fn filter_window(&mut self, ctx: &egui::Context) {
        let Some(layer_id) = self.filter_dialog.as_ref().map(|d| d.layer_id) else {
            return;
        };
        let Some(layer) = self.layers.iter().find(|l| l.id == layer_id) else {
            self.filter_dialog = None;
            return;
        };
        let layer_name = layer.name.clone();
        let has_filter = layer.filter.is_some();
        // Columns of this layer + ST_* functions + a few predicate keywords.
        let mut dict: Vec<String> = layer
            .store
            .schema
            .fields()
            .iter()
            .map(|f| f.name().to_lowercase())
            .collect();
        dict.extend(crate::sql::udf::NAMES.iter().map(|s| s.to_string()));
        for k in ["and", "or", "not", "like", "between", "in", "is null", "is not null"] {
            dict.push(k.into());
        }
        let sql_layer = self.sql_layer_of(layer_id);
        let test_tx = self.test_tx.clone();

        let Some(dialog) = &mut self.filter_dialog else {
            return;
        };
        let mut open = true;
        let mut apply = false;
        let mut clear = false;
        egui::Window::new(format!("Filter — {layer_name}"))
            .id(egui::Id::new("layer_filter"))
            .open(&mut open)
            .default_width(460.0)
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(
                        "Persistent SQL predicate: the layer shows only matching \
                         rows until the filter is cleared (spatial predicates are \
                         pruned via row-group/page statistics).",
                    )
                    .weak()
                    .small(),
                );
                let id = egui::Id::new("layer_filter_edit");
                crate::sql::console::autocomplete_edit(
                    ui,
                    id,
                    &mut dialog.text,
                    &mut dialog.ac,
                    &dict,
                    |text| {
                        egui::TextEdit::multiline(text)
                            .id(id)
                            .code_editor()
                            .desired_rows(2)
                            .desired_width(f32::INFINITY)
                            .hint_text(
                                "status = 'active' and st_area(geometry) > 1000",
                            )
                    },
                );
                let mut test = false;
                ui.horizontal(|ui| {
                    let busy = self.filter_pending.contains(&layer_id) || dialog.testing;
                    let has_text = !dialog.text.trim().is_empty();
                    test = ui
                        .add_enabled(!busy && has_text, egui::Button::new("Test"))
                        .on_hover_text(
                            "Validate the expression and count matching rows \
                             without changing the layer",
                        )
                        .clicked();
                    apply = ui
                        .add_enabled(!busy && has_text, egui::Button::new("Apply"))
                        .clicked();
                    if has_filter {
                        clear = ui.button("Clear filter").clicked();
                    }
                    if busy {
                        ui.spinner();
                        ui.label(RichText::new("computing…").weak().small());
                    }
                });
                // Test outcome (only while it still matches the text).
                if let Some(result) = &dialog.test {
                    if dialog.test_pred == dialog.text.trim() {
                        match result {
                            Ok(rows) => {
                                ui.label(
                                    RichText::new(format!(
                                        "✔ {} rows match",
                                        fmt_count(rows.matched)
                                    ))
                                    .color(Color32::from_rgb(80, 200, 120)),
                                );
                            }
                            Err(e) => {
                                ui.label(RichText::new(e).color(ui.visuals().error_fg_color));
                            }
                        }
                    }
                }
                if test {
                    dialog.test_pred = dialog.text.trim().to_string();
                    dialog.testing = true;
                    dialog.test = None;
                    if let Some(sql_layer) = sql_layer.clone() {
                        let egui_ctx = ctx.clone();
                        crate::sql::engine::spawn_row_filter(
                            layer_id,
                            sql_layer,
                            dialog.test_pred.clone(),
                            test_tx.clone(),
                            move || egui_ctx.request_repaint(),
                        );
                    }
                }
            });
        if apply {
            let predicate = self
                .filter_dialog
                .as_ref()
                .map(|d| d.text.trim().to_string())
                .unwrap_or_default();
            // Reuse the tested rows when the text hasn't changed since.
            let cached = self.filter_dialog.as_mut().and_then(|d| {
                if d.test_pred == predicate {
                    d.test.take().and_then(Result::ok)
                } else {
                    None
                }
            });
            match cached {
                Some(rows) => self.apply_filter_rows(layer_id, predicate, rows, ctx),
                None => self.start_layer_filter(layer_id, predicate, ctx),
            }
            self.filter_dialog = None;
        } else if clear {
            self.clear_layer_filter(layer_id, ctx);
            self.filter_dialog = None;
        } else if !open {
            self.filter_dialog = None;
        }
    }

    /// Zoom the map to a data-CRS bbox (e.g. a feature clicked in the SQL
    /// results grid). Pads by 20% of the span, with a floor so point
    /// features land at a usable zoom instead of the camera's max.
    fn zoom_to_data_bbox(&mut self, bbox: [f64; 4], crs: &Crs) {
        let min_pad = if crs.is_latlong { 1e-3 } else { 100.0 };
        let px = ((bbox[2] - bbox[0]) * 0.2).max(min_pad);
        let py = ((bbox[3] - bbox[1]) * 0.2).max(min_pad);
        let padded = [bbox[0] - px, bbox[1] - py, bbox[2] + px, bbox[3] + py];
        if let Some(world) = loader::data_bbox_to_world(padded, crs, &self.display) {
            self.fit_bounds = Some(world);
        }
    }

    fn poll_loader(&mut self, ctx: &egui::Context) {
        let mut rebuild_display: Option<DisplayCrs> = None;
        while let Ok(msg) = self.load_rx.try_recv() {
            match msg {
                LoadMsg::Progress { job, frac, stage } => {
                    if let Some(j) = self.loading.get_mut(&job) {
                        j.frac = frac;
                        j.stage = stage;
                    }
                }
                LoadMsg::Loaded {
                    job,
                    layer,
                    adopt_display,
                } => {
                    // Projection switched while this job was building?
                    // Its world geometry is in the old display's frame.
                    let stale = self
                        .loading
                        .get(&job)
                        .is_some_and(|j| j.display_gen != self.display_gen);
                    self.loading.remove(&job);
                    if layer.stats.bad_geoms > 0 {
                        self.push_error(format!(
                            "{}: {} geometries could not be decoded/projected",
                            layer.name, layer.stats.bad_geoms
                        ));
                    }
                    let mut layer = *layer;
                    if let Some(name) = self.pending_names.remove(&job) {
                        layer.name = name;
                    }
                    if let Some(style) = self.pending_styles.remove(&job) {
                        layer.style = style;
                    }
                    let new_layer_id = layer.id;
                    if stale && adopt_display.is_none() {
                        layer.generation += 1;
                        self.rebuilding.insert(layer.id);
                        loader::spawn_rebuild(
                            LoaderHandle {
                                tx: self.load_tx.clone(),
                                egui_ctx: ctx.clone(),
                            },
                            layer.id,
                            layer.generation,
                            layer.store.clone(),
                            layer.crs.clone(),
                            self.display.clone(),
                            layer.loaded.clone(),
                        layer.style.style_by.clone(),
                        );
                    }
                    // Category z-order: polygons at the bottom, lines above,
                    // points on top; a new layer lands on top of its own
                    // category (vec order is draw order, last = top-most).
                    let r = kind_rank(layer.kind());
                    let pos = self
                        .layers
                        .iter()
                        .rposition(|l| kind_rank(l.kind()) <= r)
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    self.layers.insert(pos, layer);
                    // Context-restored filter: compute once the layer exists.
                    if let Some(f) = self.pending_filters.remove(&job) {
                        self.start_layer_filter(new_layer_id, f, ctx);
                    }
                    match adopt_display {
                        // Geometry already built in the auto-adopted
                        // display — but layers that finished earlier are
                        // still in the previous frame and must rebuild.
                        Some((d, true)) => {
                            self.adopt_display_lite(d);
                            self.rebuild_layers_for_display(Some(new_layer_id), ctx);
                        }
                        // Post-build suggestion: full projection rebuild.
                        Some((d, false)) => rebuild_display = Some(d),
                        None => {}
                    }
                    // Never move the viewport because a layer finished
                    // loading — a dense full-extent layer would yank the
                    // user away (and could trigger a huge refinement).
                    // Fit all layers / per-layer zoom are one click away.
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
                    loaded,
                } => {
                    self.appending.remove(&layer_id);
                    self.append_cancel.remove(&layer_id);
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            log::info!(
                                "{}: appended {} row groups ({rows} features)",
                                l.name,
                                loaded.len()
                            );
                            l.sections.push(geometry);
                            l.feature_count += rows;
                            for (g, st) in loaded {
                                if let Some(slot) = l.loaded.get_mut(g as usize) {
                                    *slot = st;
                                }
                            }
                        }
                    }
                }
                LoadMsg::RebuildFailed { layer_id, error } => {
                    // The layer keeps drawing its previous-generation
                    // sections; without this the spinner and the rebuild
                    // gate stayed on forever.
                    self.rebuilding.remove(&layer_id);
                    self.push_error(error);
                }
                LoadMsg::AppendEnded { layer_id, error } => {
                    self.appending.remove(&layer_id);
                    self.append_cancel.remove(&layer_id);
                    // Hold refinement until the camera moves — for cancels
                    // AND failures: the unchanged viewport would otherwise
                    // respawn the identical (failing) job every frame,
                    // spamming errors and network requests.
                    self.refine_hold.insert(layer_id);
                    if error != loader::CANCELLED {
                        self.push_error(error);
                    }
                }
                LoadMsg::Failed { job, source, error } => {
                    self.loading.remove(&job);
                    // User-initiated stop: not an error.
                    if error != loader::CANCELLED {
                        self.push_error(format!("{source}: {error}"));
                    }
                }
            }
        }
        if let Some(d) = rebuild_display {
            self.set_display(d, ctx);
        }
    }

    /// Switch the display projection without rebuilding layers (their
    /// geometry was already built in it by the loader).
    fn adopt_display_lite(&mut self, d: DisplayCrs) {
        self.display_gen += 1;
        self.display = d;
        self.clear_selection();
        self.graticule_chunks = build_graticule(&self.display);
        self.coastline_chunks = crate::data::coastline::build_coastline(&self.display);
        self.graticule_generation += 1;
        self.rg_overlays.clear();
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
        self.display_gen += 1;
        self.display = display;
        self.selection = None;
        self.highlight_chunks = None;
        // World-space geometry; must rebuild for the new projection.
        self.sql_highlight_chunks = None;
        self.sql_highlight_generation += 1;
        self.graticule_chunks = build_graticule(&self.display);
        self.coastline_chunks = crate::data::coastline::build_coastline(&self.display);
        self.graticule_generation += 1;
        self.rebuild_layers_for_display(None, ctx);
        if self.layers.is_empty() {
            self.pending_fit = true;
        }
    }

    /// Rebuild every layer's world geometry for the current display,
    /// except `skip` (a layer already built in it).
    fn rebuild_layers_for_display(&mut self, skip: Option<u64>, ctx: &egui::Context) {
        for l in &mut self.layers {
            if Some(l.id) == skip {
                continue;
            }
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
                l.loaded.clone(),
            l.style.style_by.clone(),
                        );
        }
    }

    /// Load rows that entered the viewport of partially loaded layers:
    /// unseen row groups get a per-feature viewport selection; groups whose
    /// earlier selection no longer covers the viewport are completed
    /// (complement rows) and become Full.
    fn refine_partial_layers(&mut self, ctx: &egui::Context) {
        use crate::data::layer::GroupLoad;
        use crate::data::loader::{complement_ranges, GroupSel};
        let view = self.last_view_world;
        for l in &self.layers {
            if !l.is_partial()
                || !l.style.visible
                || self.appending.contains(&l.id)
                || self.rebuilding.contains(&l.id)
                || self.refine_hold.contains(&l.id)
            {
                continue;
            }
            let Some(rg) = &l.rg_bboxes else { continue };
            let Some(rect) = loader::viewport_to_data_bbox(view, &self.display, &l.crs) else {
                continue;
            };
            let starts = l.store.rg_starts();
            let mut jobs: Vec<GroupSel> = Vec::new();
            for g in loader::intersecting_rgs(&rg.boxes, rect) {
                let gb = rg.boxes[g as usize];
                // The part of the viewport this group can contribute to.
                let need = [
                    rect[0].max(gb[0]),
                    rect[1].max(gb[1]),
                    rect[2].min(gb[2]),
                    rect[3].min(gb[3]),
                ];
                match &l.loaded[g as usize] {
                    GroupLoad::Full => {}
                    // Preview refines like an unseen group: the in-rect
                    // sampled rows re-decode (a ~1/stride duplicate
                    // fraction — invisible next to real coverage).
                    GroupLoad::None | GroupLoad::Preview { .. } => {
                        jobs.push(GroupSel::Rect(g, rect))
                    }
                    st @ GroupLoad::Rows { ranges, .. } => {
                        if !st.covers(need) {
                            let n = (starts[g as usize + 1] - starts[g as usize]) as u32;
                            jobs.push(GroupSel::Ranges(g, complement_ranges(ranges, n)));
                        }
                    }
                }
            }
            if jobs.is_empty() {
                continue;
            }
            // Row-budget guard: refining a preview at a wide viewport
            // would decode the very load the preview avoided. Estimate
            // rect jobs by viewport/bbox area overlap and wait for a
            // tighter zoom when the estimate exceeds the budget. The
            // area estimate is only meaningful when the store can subset
            // rows per feature (covering column or x/y coordinates) —
            // otherwise a Rect job decodes the WHOLE group.
            let selectable =
                l.store.covering.is_some() || l.store.xy_geom.is_some();
            let est: u64 = jobs
                .iter()
                .map(|j| match j {
                    GroupSel::Ranges(_, rs) => {
                        rs.iter().map(|&(s, e)| (e - s) as u64).sum()
                    }
                    GroupSel::Rect(g, r) | GroupSel::Preview { group: g, rect: Some(r), .. }
                        if selectable =>
                    {
                        let gb = rg.boxes[*g as usize];
                        let n = starts[*g as usize + 1] - starts[*g as usize];
                        let (bw, bh) = (gb[2] - gb[0], gb[3] - gb[1]);
                        let iw = (r[2].min(gb[2]) - r[0].max(gb[0])).max(0.0);
                        let ih = (r[3].min(gb[3]) - r[1].max(gb[1])).max(0.0);
                        if bw <= 0.0 || bh <= 0.0 {
                            n
                        } else {
                            ((iw * ih) / (bw * bh) * n as f64).ceil() as u64
                        }
                    }
                    j => {
                        let g = match j {
                            GroupSel::All(g)
                            | GroupSel::Rect(g, _)
                            | GroupSel::Preview { group: g, .. } => *g,
                            GroupSel::Ranges(g, _) => *g,
                        };
                        starts[g as usize + 1] - starts[g as usize]
                    }
                })
                .sum();
            if est > crate::data::loader::MAX_BUILD_ROWS {
                log::debug!(
                    "{}: refinement estimate {est} rows over budget — zoom in further",
                    l.name
                );
                continue;
            }
            log::info!("{}: refining with {} row groups", l.name, jobs.len());
            self.appending.insert(l.id);
            let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
            self.append_cancel.insert(l.id, Arc::clone(&cancel));
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
                jobs,
                cancel,
            l.style.style_by.clone(),
                        );
        }
    }

    /// Clear the picked feature and cancel any pick in flight.
    fn clear_selection(&mut self) {
        self.pick_job += 1;
        self.pick_pending = false;
        self.apply_pick(None, None, None);
    }

    /// Kick off an async pick: hit test + attribute fetch in a worker
    /// thread (network reads on remote layers must never block the UI).
    fn start_pick(&mut self, world: [f64; 2], tol: f64, ctx: &egui::Context) {
        self.pick_job += 1;
        let job = self.pick_job;
        self.pick_pending = true;
        let snaps: Vec<picking::PickLayer> =
            self.layers.iter().map(picking::PickLayer::of).collect();
        let display = self.display.clone();
        let tx = self.pick_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let sel = picking::pick(&snaps, &display, world, tol);
            let (attrs, truncated) = match &sel {
                Some(s) => fetch_pick_attrs(&snaps, s),
                None => (None, None),
            };
            let _ = tx.send(PickMsg {
                job,
                sel,
                attrs,
                truncated,
            });
            ctx.request_repaint();
        });
    }

    fn poll_picks(&mut self) {
        while let Ok(m) = self.pick_rx.try_recv() {
            if m.job != self.pick_job {
                continue; // superseded by a newer click / clear
            }
            self.pick_pending = false;
            self.apply_pick(m.sel, m.attrs, m.truncated);
        }
    }

    fn apply_pick(
        &mut self,
        sel: Option<Selection>,
        attrs: Option<arrow::record_batch::RecordBatch>,
        truncated: Option<(usize, usize)>,
    ) {
        self.selection_generation += 1;
        self.highlight_chunks = sel.as_ref().map(|s| {
            let mut mb = MeshBuilder::default();
            mb.add(&s.world_geom, crate::data::geometry::FeatureRef::INVALID);
            Arc::new(mb.finish())
        });
        self.selection_attrs = attrs;
        self.attrs_truncated = truncated;
        self.selection = sel;
    }

    // ------------------------------------------------------------------
    // UI
    // ------------------------------------------------------------------

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::containers::menu::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("📂 Open…").clicked() {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("GeoParquet", &["parquet", "geoparquet", "pq"])
                        .pick_files()
                    {
                        for p in paths {
                            self.enqueue_load(Source::Local(p), &ctx);
                        }
                    }
                }
                if ui
                    .button("📂 Open folder…")
                    .on_hover_text(
                        "Load a directory of GeoParquet files (hive-partitioned or not) \
                         as a single layer; key=value path segments become columns",
                    )
                    .clicked()
                {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.enqueue_load(Source::Dir(dir), &ctx);
                    }
                }
                if ui.button("🌐 Open URL…").clicked() && self.url_input.is_none() {
                    self.url_input = Some((
                        String::new(),
                        None,
                        crate::data::source::aws::profiles(),
                        String::new(),
                    ));
                }
                if ui
                    .button("🌐 Repositories…")
                    .on_hover_text(
                        "Browse preconfigured GeoParquet repositories and load \
                         their layers directly",
                    )
                    .clicked()
                    && self.repo_browser.is_none()
                {
                    self.open_repo_browser(&ctx);
                }
                ui.separator();
                if ui
                    .button("💾 Save context…")
                    .on_hover_text("Save layers, styles, camera and projection to a JSON file")
                    .clicked()
                {
                    self.save_context();
                }
                if ui
                    .button("📥 Load context…")
                    .on_hover_text("Restore a saved context (replaces current layers)")
                    .clicked()
                {
                    self.load_context(&ctx);
                }
            });
            ui.menu_button("View", |ui| {
                if ui
                    .add_enabled(!self.layers.is_empty(), egui::Button::new("🌍 Fit all layers"))
                    .clicked()
                {
                    self.pending_fit = true;
                }
                ui.separator();
                ui.checkbox(&mut self.show_graticule, "Graticule");
                ui.checkbox(&mut self.show_coastline, "Coastline");
                ui.menu_button("Basemap", |ui| {
                    if !self.display.is_mercator() {
                        ui.label(
                            RichText::new("tiles render in Web Mercator only").weak().small(),
                        );
                    }
                    for (i, s) in TILE_SOURCES.iter().enumerate() {
                        ui.selectable_value(&mut self.basemap, Some(i), s.name);
                    }
                    ui.selectable_value(&mut self.basemap, None, "None");
                });
                ui.separator();
                ui.checkbox(&mut self.sql.open, "SQL console");
            });
            ui.menu_button("Help", |ui| {
                if ui.button("ST_* function reference").clicked() {
                    self.sql.open_with_help();
                }
                ui.label(
                    RichText::new(concat!("geopq-viewer ", env!("CARGO_PKG_VERSION")))
                        .weak()
                        .small(),
                );
            });
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button("📂 Open…").clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("GeoParquet", &["parquet", "geoparquet", "pq"])
                    .pick_files()
                {
                    for p in paths {
                        self.enqueue_load(Source::Local(p), &ctx);
                    }
                }
            }
            if ui.button("🌐 URL…").clicked() && self.url_input.is_none() {
                self.url_input = Some((
                    String::new(),
                    None,
                    crate::data::source::aws::profiles(),
                    String::new(),
                ));
            }
            if ui
                .add_enabled(!self.layers.is_empty(), egui::Button::new("🌍 Fit all"))
                .clicked()
            {
                self.pending_fit = true;
            }
            ui.toggle_value(&mut self.sql.open, "🖩 SQL")
                .on_hover_text("Query loaded layers with SQL (ST_* spatial functions)");
        });
    }

    /// Projection picker (status bar): built-ins, national grids, EPSG entry.
    fn projection_selector(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        {
            let is_hobo = self.display.name.starts_with("Hobo");
            let is_wintri = self.display.kind == DisplayKind::WinkelTripel;
            let is_4326 = self.display.kind == DisplayKind::Plain && self.display.crs.epsg == Some(4326);
            let mut pick: Option<DisplayCrs> = None;
            let mut picked_auto = false;
            egui::ComboBox::from_id_salt("projection")
                .selected_text(self.display.name.clone())
                .width(200.0)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(self.auto_projection, "Auto (fit first layer)")
                        .on_hover_text(
                            "Projected data CRS if the layer has one; otherwise an\n\
                             equal-area projection fit to the extent (Albers / LAEA /\n\
                             cylindrical), or Hobo–Dyer for world-scale data.",
                        )
                        .clicked()
                    {
                        picked_auto = true;
                        if let Some(l) = self.layers.first() {
                            let bbox = l
                                .rg_bboxes
                                .as_ref()
                                .filter(|_| l.crs.is_latlong)
                                .and_then(|r| {
                                    r.boxes.iter().copied().reduce(|a, b| {
                                        [
                                            a[0].min(b[0]),
                                            a[1].min(b[1]),
                                            a[2].max(b[2]),
                                            a[3].max(b[3]),
                                        ]
                                    })
                                });
                            pick = Some(
                                DisplayCrs::auto_for(&l.crs, bbox)
                                    .unwrap_or_else(DisplayCrs::hobo_dyer),
                            );
                        }
                    }
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
                    ui.separator();
                    ui.label(RichText::new("National grids").weak().small());
                    for n in crate::data::national::NATIONAL_CRS {
                        let selected = self.display.crs.epsg == Some(n.epsg);
                        let label = format!("{} — {}", n.country, n.name);
                        if ui.selectable_label(selected, label).clicked() && !selected {
                            match DisplayCrs::from_epsg(n.epsg) {
                                Ok(mut d) => {
                                    d.name = format!("{} ({})", n.name, n.country);
                                    pick = Some(d);
                                }
                                Err(e) => self.push_error(e),
                            }
                        }
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut self.epsg_input)
                                .hint_text("EPSG code…")
                                .desired_width(80.0),
                        );
                        let apply = ui.button("Apply").clicked()
                            || (resp.lost_focus()
                                && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                        if apply {
                            match self.epsg_input.trim().parse::<u32>() {
                                Ok(code) => match DisplayCrs::from_epsg(code) {
                                    Ok(d) => pick = Some(d),
                                    Err(e) => self.push_error(e),
                                },
                                Err(_) => self.push_error("invalid EPSG code".into()),
                            }
                        }
                    });
                });
            if let Some(d) = pick {
                self.auto_projection = picked_auto;
                self.set_display(d, &ctx);
            } else if picked_auto {
                self.auto_projection = true;
            }
        }
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
        let mut reorder: Option<(u64, i32)> = None;
        let mut fit_to: Option<[f64; 4]> = None;
        let mut info_open: Option<u64> = None;
        let mut load_all: Option<u64> = None;
        let mut optimize_open: Option<u64> = None;
        let mut filter_open: Option<u64> = None;
        let mut style_open: Option<u64> = None;
        let mut filter_clear: Option<u64> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Top-most layer first in the list.
            let rebuilding = &self.rebuilding;
            let filter_pending = &self.filter_pending;
            let n_layers = self.layers.len();
            for (idx, l) in self.layers.iter_mut().enumerate().rev() {
                let is_rebuilding = rebuilding.contains(&l.id);
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut l.style.visible, "");
                        geom_kind_icon(ui, l.kind());
                        let mut c = l.style.color;
                        if swatch_color_button(
                            ui,
                            &format!("fill{}", l.id),
                            &mut c,
                            "fill / point color",
                        ) {
                            l.style.color = c;
                        }
                        if !matches!(l.kind(), crate::data::geometry::GeomKind::Point) {
                            let mut lc = l
                                .style
                                .line_color
                                .unwrap_or_else(|| derived_line_color(l.style.color));
                            if swatch_color_button(
                                ui,
                                &format!("line{}", l.id),
                                &mut lc,
                                "border / line color",
                            ) {
                                l.style.line_color = Some(lc);
                            }
                            if l.style.line_color.is_some()
                                && ui
                                    .small_button("↺")
                                    .on_hover_text("reset border color to auto")
                                    .clicked()
                            {
                                l.style.line_color = None;
                            }
                        }
                        ui.label(RichText::new(&l.name).strong())
                            .on_hover_text(l.store.source.label());
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.menu_button("☰", |ui| {
                                    if ui.button("Info…").clicked() {
                                        info_open = Some(l.id);
                                    }
                                    let single = !l.store.is_partitioned();
                                    if ui
                                        .add_enabled(single, egui::Button::new("Optimize…"))
                                        .on_hover_text(if !single {
                                            "Optimize works on single files; this layer is a \
                                             multi-file dataset"
                                        } else if l.store.xy_geom.is_some() {
                                            "Materialize this x/y layer as real GeoParquet: \
                                             WKB points, Hilbert order, covering bbox / \
                                             native geo stats"
                                        } else {
                                            "Rewrite as a spatially sorted GeoParquet 1.1 or \
                                             2.0 file (Hilbert order, covering bbox / native \
                                             geo stats, bloom filters)"
                                        })
                                        .clicked()
                                    {
                                        optimize_open = Some(l.id);
                                    }
                                    if ui.button("Filter…").clicked() {
                                        filter_open = Some(l.id);
                                    }
                                    if ui.button("Style by value…").clicked() {
                                        style_open = Some(l.id);
                                    }
                                    if l.style.style_by.is_some()
                                        && ui.button("Clear styling").clicked()
                                    {
                                        // Bins stay in the meshes; without
                                        // bin colors everything draws in
                                        // the uniform layer color again.
                                        l.style.style_by = None;
                                    }
                                    if l.is_partial()
                                        && l.filter.is_none()
                                        && ui.button("Load all row groups").clicked()
                                    {
                                        load_all = Some(l.id);
                                    }
                                    if let Some(rg) = &l.rg_bboxes {
                                        ui.separator();
                                        ui.checkbox(&mut l.style.show_rg_bboxes, "RG bboxes")
                                            .on_hover_text(format!(
                                                "{} row groups — source: {}\navg overlap \
                                                 ×{:.1} = {:.0}% of possible {}",
                                                rg.boxes.len(),
                                                rg.source,
                                                rg.avg_overlap,
                                                rg.overlap_frac() * 100.0,
                                                if rg.poorly_clustered() {
                                                    "(poorly clustered: consider Optimize…)"
                                                } else {
                                                    "(well clustered)"
                                                }
                                            ));
                                    }
                                    ui.separator();
                                    if ui.button("Remove layer").clicked() {
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
                                if ui
                                    .small_button("🔍")
                                    .on_hover_text("Zoom to layer")
                                    .clicked()
                                {
                                    fit_to = Some(l.bounds_world());
                                }
                                // Vec order is draw order: last = top-most.
                                if n_layers > 1 {
                                    if ui
                                        .add_enabled(
                                            idx > 0,
                                            egui::Button::new("⏷").small(),
                                        )
                                        .on_hover_text("Move down")
                                        .clicked()
                                    {
                                        reorder = Some((l.id, -1));
                                    }
                                    if ui
                                        .add_enabled(
                                            idx + 1 < n_layers,
                                            egui::Button::new("⏶").small(),
                                        )
                                        .on_hover_text("Move up")
                                        .clicked()
                                    {
                                        reorder = Some((l.id, 1));
                                    }
                                }
                            },
                        );
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
                    if let Some(f) = &l.filter {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "filter: {} of {} rows",
                                    fmt_count(f.matched),
                                    fmt_count(l.store.total_rows() as usize)
                                ))
                                .color(Color32::from_rgb(80, 180, 240))
                                .small(),
                            )
                            .on_hover_text(&f.sql);
                            if ui.small_button("Edit").clicked() {
                                filter_open = Some(l.id);
                            }
                            if ui
                                .small_button("✕")
                                .on_hover_text("Clear the filter (reload all rows)")
                                .clicked()
                            {
                                filter_clear = Some(l.id);
                            }
                        });
                    }
                    if filter_pending.contains(&l.id) {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(RichText::new("computing filter…").weak().small());
                        });
                    }
                    // Data-classified styling goes stale when the loaded
                    // rows drift (classification never reads beyond them).
                    if let Some(sb) = &l.style.style_by {
                        if let Some(n0) = sb.classified_rows {
                            let n1 = l.loaded_rows() as f64;
                            let drift = (n1 - n0 as f64).abs() / (n0.max(1) as f64);
                            if drift > 0.25 {
                                ui.label(
                                    RichText::new(
                                        "style classes computed from a previous extent — \
                                         reopen Style by value… to reclassify",
                                    )
                                    .color(Color32::from_rgb(242, 140, 26))
                                    .small(),
                                );
                            }
                        }
                    }
                    if l.is_partial() && l.filter.is_none() {
                        ui.horizontal(|ui| {
                            let preview = l.preview_rgs();
                            let partial = l.partial_rgs();
                            let text = if preview > 0 {
                                format!(
                                    "preview: {} of {} row groups decimated — zoom in \
                                     to load real rows",
                                    preview,
                                    l.total_rgs()
                                )
                            } else if partial > 0 {
                                format!(
                                    "partial: {}/{} row groups full, {} viewport-filtered",
                                    l.full_rgs(),
                                    l.total_rgs(),
                                    partial
                                )
                            } else {
                                format!(
                                    "partial: {}/{} row groups loaded",
                                    l.full_rgs(),
                                    l.total_rgs()
                                )
                            };
                            ui.label(
                                RichText::new(text)
                                    .color(Color32::from_rgb(242, 140, 26))
                                    .small(),
                            );
                            if ui.small_button("Load all").clicked() {
                                load_all = Some(l.id);
                            }
                            if self.appending.contains(&l.id) {
                                ui.spinner();
                                if ui
                                    .small_button("✖")
                                    .on_hover_text(
                                        "Stop loading rows (resumes when the map moves)",
                                    )
                                    .clicked()
                                {
                                    if let Some(c) = self.append_cancel.get(&l.id) {
                                        c.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }
                            }
                        });
                    }
                });
                ui.add_space(4.0);
            }
        });

        if let Some((id, dir)) = reorder {
            // Vec order is draw order (bottom→top); ▲ raises the layer.
            if let Some(i) = self.layers.iter().position(|l| l.id == id) {
                let j = i as i64 + dir as i64;
                if j >= 0 && (j as usize) < self.layers.len() {
                    self.layers.swap(i, j as usize);
                }
            }
        }
        if let Some(id) = remove {
            self.layers.retain(|l| l.id != id);
            self.rebuilding.remove(&id);
            if self.selection.as_ref().map(|s| s.layer_id) == Some(id) {
                self.clear_selection();
            }
        }
        if let Some(b) = fit_to {
            self.fit_bounds = Some(b);
        }
        if info_open.is_some() {
            self.info_open = info_open;
        }
        if let Some(id) = optimize_open {
            let already_running = self.optimize.as_ref().is_some_and(|o| o.running);
            if !already_running {
                if let Some(l) = self.layers.iter().find(|l| l.id == id) {
                    self.optimize = Some(OptimizeState {
                        layer_id: l.id,
                        layer_name: l.name.clone(),
                        src: l.store.source.clone(),
                        epsg: l.crs.epsg,
                        crs: l.crs.clone(),
                        viewport_only: false,
                        opts: crate::data::optimize::OptimizeOptions {
                            xy_geom: l.store.xy_geom,
                            ..Default::default()
                        },
                        running: false,
                        progress: (0.0, String::new()),
                        report: None,
                        error: None,
                        admin_layer: None,
                        admin_column: String::new(),
                        admin_out: "admin".into(),
                        part_mode: PartMode::None,
                        part_fields: Vec::new(),
                        adaptive_target: 1_000_000,
                        cardinalities: None,
                        card_pending: false,
                    });
                }
            }
        }
        if let Some(id) = filter_open {
            let text = self
                .layers
                .iter()
                .find(|l| l.id == id)
                .and_then(|l| l.filter.as_ref().map(|f| f.sql.clone()))
                .unwrap_or_default();
            self.filter_dialog = Some(FilterDialog {
                layer_id: id,
                text,
                ac: Default::default(),
                test_pred: String::new(),
                testing: false,
                test: None,
            });
        }
        if let Some(id) = filter_clear {
            let ctx = ui.ctx().clone();
            self.clear_layer_filter(id, &ctx);
        }
        if let Some(id) = style_open {
            let ctx = ui.ctx().clone();
            self.open_style_dialog(id, &ctx);
        }
        if let Some(id) = load_all {
            use crate::data::layer::GroupLoad;
            use crate::data::loader::{complement_ranges, GroupSel};
            let ctx = ui.ctx().clone();
            if let Some(l) = self.layers.iter().find(|l| l.id == id) {
                if !self.appending.contains(&id) && !self.rebuilding.contains(&id) {
                    let starts = l.store.rg_starts();
                    let missing: Vec<GroupSel> = l
                        .loaded
                        .iter()
                        .enumerate()
                        .filter_map(|(g, st)| match st {
                            GroupLoad::Full => None,
                            GroupLoad::None | GroupLoad::Preview { .. } => {
                                Some(GroupSel::All(g as u32))
                            }
                            GroupLoad::Rows { ranges, .. } => {
                                let n = (starts[g + 1] - starts[g]) as u32;
                                Some(GroupSel::Ranges(g as u32, complement_ranges(ranges, n)))
                            }
                        })
                        .collect();
                    if !missing.is_empty() {
                        self.appending.insert(id);
                        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                        self.append_cancel.insert(id, Arc::clone(&cancel));
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
                            cancel,
                        l.style.style_by.clone(),
                        );
                    }
                }
            }
        }
    }

    fn save_context(&mut self) {
        use crate::context::{Context, LayerCtx, SourceCtx, StyleCtx, CONTEXT_VERSION};
        let ctx = Context {
            version: CONTEXT_VERSION,
            camera_center: self.camera.center,
            camera_zoom: self.camera.zoom,
            projection: crate::context::projection_token(&self.display),
            projection_name: Some(self.display.name.clone()),
            basemap: self.basemap,
            show_graticule: self.show_graticule,
            show_coastline: self.show_coastline,
            layers: self
                .layers
                .iter()
                .map(|l| LayerCtx {
                    source: SourceCtx::of(&l.store.source),
                    style: StyleCtx::of(&l.style),
                    filter: l.filter.as_ref().map(|f| f.sql.clone()),
                })
                .collect(),
        };
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("session.geopq.json")
            .add_filter("geopq context", &["json"])
            .save_file()
        else {
            return;
        };
        match serde_json::to_string_pretty(&ctx)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
        {
            Ok(()) => log::info!("context saved to {}", path.display()),
            Err(e) => self.push_error(format!("context save failed: {e}")),
        }
    }

    fn load_context(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("geopq context", &["json"])
            .pick_file()
        else {
            return;
        };
        let parsed: Result<crate::context::Context, String> = std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()));
        let saved = match parsed {
            Ok(c) => c,
            Err(e) => {
                self.push_error(format!("context load failed: {e}"));
                return;
            }
        };

        // Replace the current session: projection first (so loads build in
        // the right space), then camera, then layers with their styles.
        match crate::context::projection_from_token(&saved.projection) {
            Ok(mut d) => {
                if let Some(n) = &saved.projection_name {
                    if saved.projection.starts_with("proj4:") {
                        d.name = n.clone();
                    }
                }
                self.layers.clear();
                self.rebuilding.clear();
                self.appending.clear();
                self.clear_selection();
                self.set_display(d, ctx);
            }
            Err(e) => {
                self.push_error(format!("context load failed: {e}"));
                return;
            }
        }
        self.camera.center = saved.camera_center;
        self.camera.zoom = saved.camera_zoom;
        self.pending_fit = false;
        self.auto_projection = false;
        self.basemap = saved.basemap;
        self.show_graticule = saved.show_graticule;
        self.show_coastline = saved.show_coastline;
        // Loads prune against the restored viewport once the map panel has
        // recomputed it; seed with the whole world until then.
        for layer in saved.layers {
            let job = self.enqueue_load(layer.source.into_source(), ctx);
            self.pending_styles.insert(job, layer.style.into_style());
            if let Some(f) = layer.filter {
                self.pending_filters.insert(job, f);
            }
        }
    }

    // ------------------------------------------------------------------
    // Repository browser
    // ------------------------------------------------------------------

    fn open_repo_browser(&mut self, ctx: &egui::Context) {
        self.repo_browser = Some(RepoBrowser {
            repos: crate::data::repo::load_repos(),
            sel_repo: 0,
            snapshots: vec![crate::data::repo::Snapshot::latest()],
            sel_snapshot: 0,
            datasets: None,
            filter: String::new(),
            country: String::new(),
            selected: None,
            checked: Default::default(),
            cache_age: None,
            add: (String::new(), String::new()),
            generation: 0,
        });
        self.repo_refetch(ctx, false);
    }

    /// (Re)fetch snapshots + datasets for the browser's current repository
    /// and snapshot, dropping any stale in-flight results. Discovery uses
    /// the on-disk cache unless `force` clears and re-probes.
    fn repo_refetch(&mut self, ctx: &egui::Context, force: bool) {
        let Some(b) = &mut self.repo_browser else { return };
        b.generation += 1;
        b.datasets = None;
        b.selected = None;
        b.checked.clear();
        b.cache_age = None;
        let generation = b.generation;
        let base = b.repos[b.sel_repo].url.trim_end_matches('/').to_string();
        let snapshot = b.snapshots[b.sel_snapshot].path.clone();
        let tx = self.repo_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(RepoMsg::Snapshots(
                generation,
                crate::data::repo::fetch_snapshots(&base),
            ));
            if force {
                crate::data::repo::clear_cached_datasets(&base, &snapshot);
            } else if let Some((ds, at)) = crate::data::repo::cached_datasets(&base, &snapshot)
            {
                let _ = tx.send(RepoMsg::Datasets(generation, Ok(ds), Some(at)));
                ctx.request_repaint();
                return;
            }
            let res = crate::data::repo::discover_datasets(&base, &snapshot);
            if let Ok(ds) = &res {
                crate::data::repo::store_datasets(&base, &snapshot, ds);
            }
            let _ = tx.send(RepoMsg::Datasets(generation, res, None));
            ctx.request_repaint();
        });
    }

    fn repo_fetch_manifest(&mut self, ds_idx: usize, ctx: &egui::Context) {
        let Some(b) = &mut self.repo_browser else { return };
        b.selected = Some((ds_idx, None));
        b.checked.clear();
        let generation = b.generation;
        let base = b.repos[b.sel_repo].url.trim_end_matches('/').to_string();
        let snapshot = b.snapshots[b.sel_snapshot].path.clone();
        let Some(Ok(ds)) = &b.datasets else { return };
        let path = ds[ds_idx].path.clone();
        let tx = self.repo_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(RepoMsg::Manifest(
                generation,
                crate::data::repo::fetch_manifest(&base, &snapshot, &path),
            ));
            ctx.request_repaint();
        });
    }

    fn poll_repo(&mut self) {
        while let Ok(msg) = self.repo_rx.try_recv() {
            let Some(b) = &mut self.repo_browser else { continue };
            match msg {
                RepoMsg::Snapshots(g, res) if g == b.generation => {
                    if let Ok(snaps) = res {
                        // Keep the current selection by path across refreshes.
                        let cur = b.snapshots[b.sel_snapshot].path.clone();
                        b.sel_snapshot =
                            snaps.iter().position(|s| s.path == cur).unwrap_or(0);
                        b.snapshots = snaps;
                    }
                }
                RepoMsg::Datasets(g, res, cached_at) if g == b.generation => {
                    b.datasets = Some(res);
                    b.cache_age = cached_at;
                }
                RepoMsg::Manifest(g, res) if g == b.generation => {
                    if let Some((_, m)) = &mut b.selected {
                        *m = Some(res);
                    }
                }
                _ => {} // stale generation
            }
        }
    }

    fn repo_window(&mut self, ctx: &egui::Context) {
        if self.repo_browser.is_none() {
            return;
        }
        let mut open = true;
        let mut refetch = false;
        let mut force_refetch = false;
        let mut fetch_manifest: Option<usize> = None;
        let mut load: Vec<(String, String)> = Vec::new(); // (url, layer name)

        {
            let b = self.repo_browser.as_mut().unwrap();
            egui::Window::new("GeoParquet repositories")
                .id(egui::Id::new("repo_browser"))
                .open(&mut open)
                .default_width(560.0)
                .show(ctx, |ui| {
                    // --- repository + snapshot row ---
                    ui.horizontal(|ui| {
                        let before = b.sel_repo;
                        egui::ComboBox::from_id_salt("repo_sel")
                            .width(260.0)
                            .selected_text(&b.repos[b.sel_repo].name)
                            .show_ui(ui, |ui| {
                                for (i, r) in b.repos.iter().enumerate() {
                                    ui.selectable_value(&mut b.sel_repo, i, &r.name)
                                        .on_hover_text(&r.url);
                                }
                            });
                        if b.sel_repo != before {
                            b.snapshots = vec![crate::data::repo::Snapshot::latest()];
                            b.sel_snapshot = 0;
                            refetch = true;
                        }
                        ui.label("snapshot:");
                        let before = b.sel_snapshot;
                        egui::ComboBox::from_id_salt("repo_snap")
                            .selected_text(&b.snapshots[b.sel_snapshot].label)
                            .show_ui(ui, |ui| {
                                for (i, s) in b.snapshots.iter().enumerate() {
                                    ui.selectable_value(&mut b.sel_snapshot, i, &s.label);
                                }
                            });
                        if b.sel_snapshot != before {
                            refetch = true;
                        }
                        if ui
                            .button("⟳")
                            .on_hover_text("Clear the cached dataset list and re-discover")
                            .clicked()
                        {
                            force_refetch = true;
                        }
                        if let Some(at) = b.cache_age {
                            ui.label(
                                RichText::new(format!(
                                    "cached {}",
                                    crate::data::repo::age_label(at)
                                ))
                                .weak()
                                .small(),
                            );
                        }
                    });
                    ui.separator();

                    // --- datasets (left) + themes (right) ---
                    // Owned views of the async state, so the widgets below
                    // can mutate the browser (filters, checkboxes) freely.
                    let ds_view: Option<Result<Vec<(usize, String, String, String)>, String>> =
                        match &b.datasets {
                            None => None,
                            Some(Err(e)) => Some(Err(e.clone())),
                            Some(Ok(ds)) => Some(Ok(ds
                                .iter()
                                .enumerate()
                                .map(|(i, d)| {
                                    (i, d.name.clone(), d.code.clone(), d.path.clone())
                                })
                                .collect())),
                        };
                    let sel_view: Option<(
                        String,
                        String,
                        String,
                        Option<Result<crate::data::repo::Manifest, String>>,
                    )> = match (&b.selected, &b.datasets) {
                        (Some((i, m)), Some(Ok(ds))) => Some((
                            ds[*i].code.clone(),
                            ds[*i].path.clone(),
                            ds[*i].name.clone(),
                            m.clone(),
                        )),
                        _ => None,
                    };
                    let sel_idx = b.selected.as_ref().map(|(i, _)| *i);

                    ui.horizontal_top(|ui| {
                        ui.vertical(|ui| {
                            ui.set_width(230.0);
                            match &ds_view {
                                None => {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label(RichText::new("discovering datasets…").weak());
                                    });
                                }
                                Some(Err(e)) => {
                                    ui.label(
                                        RichText::new(e).color(Color32::from_rgb(220, 60, 60)),
                                    );
                                }
                                Some(Ok(ds)) => {
                                    // Country selector, derived from the
                                    // dataset paths (hidden when the repo
                                    // has no country= level).
                                    let mut countries: Vec<&str> = ds
                                        .iter()
                                        .filter_map(|(_, _, _, path)| {
                                            path.split('/')
                                                .find_map(|s| s.strip_prefix("country="))
                                        })
                                        .collect();
                                    countries.sort_unstable();
                                    countries.dedup();
                                    ui.horizontal(|ui| {
                                        if countries.len() > 1 {
                                            egui::ComboBox::from_id_salt("repo_country")
                                                .width(70.0)
                                                .selected_text(if b.country.is_empty() {
                                                    "All".to_string()
                                                } else {
                                                    b.country.clone()
                                                })
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut b.country,
                                                        String::new(),
                                                        "All",
                                                    );
                                                    for c in &countries {
                                                        ui.selectable_value(
                                                            &mut b.country,
                                                            c.to_string(),
                                                            *c,
                                                        );
                                                    }
                                                });
                                        }
                                        ui.add(
                                            egui::TextEdit::singleline(&mut b.filter)
                                                .hint_text("filter…")
                                                .desired_width(ui.available_width()),
                                        );
                                    });
                                    let needle = b.filter.to_lowercase();
                                    let country = format!("country={}", b.country);
                                    egui::ScrollArea::vertical()
                                        .max_height(320.0)
                                        .show(ui, |ui| {
                                            for (i, name, code, path) in ds {
                                                if !b.country.is_empty()
                                                    && !path.contains(&country)
                                                {
                                                    continue;
                                                }
                                                if !needle.is_empty()
                                                    && !name.to_lowercase().contains(&needle)
                                                    && !code.to_lowercase().contains(&needle)
                                                {
                                                    continue;
                                                }
                                                if ui
                                                    .selectable_label(
                                                        sel_idx == Some(*i),
                                                        format!("{name} ({code})"),
                                                    )
                                                    .clicked()
                                                {
                                                    fetch_manifest = Some(*i);
                                                }
                                            }
                                        });
                                }
                            }
                        });
                        ui.separator();
                        ui.vertical(|ui| match &sel_view {
                            Some((code, path, name, manifest)) => {
                                // The manifest's own name is authoritative
                                // (discovery may only know the ISO code).
                                let title = match manifest {
                                    Some(Ok(m)) => {
                                        m.state_name.as_deref().unwrap_or(name)
                                    }
                                    _ => name,
                                };
                                ui.label(RichText::new(title).strong());
                                match manifest {
                                    None => {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label(RichText::new("reading manifest…").weak());
                                        });
                                    }
                                    Some(Err(e)) => {
                                        ui.label(
                                            RichText::new(e)
                                                .color(Color32::from_rgb(220, 60, 60)),
                                        );
                                    }
                                    Some(Ok(m)) => {
                                        if let Some(t) = m.total_features {
                                            ui.label(
                                                RichText::new(format!(
                                                    "{} features",
                                                    fmt_count(t as usize)
                                                ))
                                                .weak()
                                                .small(),
                                            );
                                        }
                                        let base = b.repos[b.sel_repo]
                                            .url
                                            .trim_end_matches('/')
                                            .to_string();
                                        let snap = b.snapshots[b.sel_snapshot].path.clone();
                                        egui::ScrollArea::vertical()
                                            .id_salt("repo_themes")
                                            .max_height(300.0)
                                            .show(ui, |ui| {
                                                egui::Grid::new("repo_theme_grid")
                                                    .num_columns(2)
                                                    .striped(true)
                                                    .show(ui, |ui| {
                                                        for (theme, count) in &m.themes {
                                                            let mut on =
                                                                b.checked.contains(theme);
                                                            if ui
                                                                .checkbox(&mut on, theme)
                                                                .changed()
                                                            {
                                                                if on {
                                                                    b.checked
                                                                        .insert(theme.clone());
                                                                } else {
                                                                    b.checked.remove(theme);
                                                                }
                                                            }
                                                            ui.label(
                                                                RichText::new(fmt_count(
                                                                    *count as usize,
                                                                ))
                                                                .weak(),
                                                            );
                                                            ui.end_row();
                                                        }
                                                    });
                                            });
                                        ui.horizontal(|ui| {
                                            let n = b.checked.len();
                                            if ui
                                                .add_enabled(
                                                    n > 0,
                                                    egui::Button::new(format!(
                                                        "Load {n} layer{}",
                                                        if n == 1 { "" } else { "s" }
                                                    )),
                                                )
                                                .clicked()
                                            {
                                                for (theme, _) in &m.themes {
                                                    if b.checked.contains(theme) {
                                                        load.push((
                                                            crate::data::repo::theme_url(
                                                                &base, &snap, path, theme,
                                                            ),
                                                            format!("{code} {theme}"),
                                                        ));
                                                    }
                                                }
                                                b.checked.clear();
                                            }
                                            if ui.small_button("all").clicked() {
                                                b.checked = m
                                                    .themes
                                                    .iter()
                                                    .map(|(t, _)| t.clone())
                                                    .collect();
                                            }
                                            if ui.small_button("none").clicked() {
                                                b.checked.clear();
                                            }
                                        });
                                    }
                                }
                            }
                            None => {
                                ui.label(
                                    RichText::new("select a dataset to list its layers").weak(),
                                );
                            }
                        });
                    });

                    // --- add repository ---
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut b.add.0)
                                .hint_text("name")
                                .desired_width(140.0),
                        );
                        ui.add(
                            egui::TextEdit::singleline(&mut b.add.1)
                                .hint_text("https://repo.example.com")
                                .desired_width(240.0),
                        );
                        let valid = !b.add.0.trim().is_empty()
                            && b.add.1.trim().starts_with("https://");
                        if ui
                            .add_enabled(valid, egui::Button::new("Add repository"))
                            .clicked()
                        {
                            b.repos.push(crate::data::repo::Repository {
                                name: b.add.0.trim().to_string(),
                                url: b.add.1.trim().trim_end_matches('/').to_string(),
                            });
                            b.add = (String::new(), String::new());
                            b.sel_repo = b.repos.len() - 1;
                            if let Err(e) = crate::data::repo::save_repos(&b.repos) {
                                log::warn!("saving repositories: {e}");
                            }
                            b.snapshots = vec![crate::data::repo::Snapshot::latest()];
                            b.sel_snapshot = 0;
                            refetch = true;
                        }
                        if b.repos.len() > 1 && ui.button("Remove current").clicked() {
                            b.repos.remove(b.sel_repo);
                            b.sel_repo = 0;
                            if let Err(e) = crate::data::repo::save_repos(&b.repos) {
                                log::warn!("saving repositories: {e}");
                            }
                            b.snapshots = vec![crate::data::repo::Snapshot::latest()];
                            b.sel_snapshot = 0;
                            refetch = true;
                        }
                    });
                });
        }

        if refetch || force_refetch {
            self.repo_refetch(ctx, force_refetch);
        }
        if let Some(i) = fetch_manifest {
            self.repo_fetch_manifest(i, ctx);
        }
        for (url, name) in load {
            let job = self.enqueue_load(Source::Remote { url, len: 0 }, ctx);
            self.pending_names.insert(job, name);
        }
        if !open {
            self.repo_browser = None;
        }
    }

    // ------------------------------------------------------------------
    // Data-driven styling dialog
    // ------------------------------------------------------------------

    /// Columns of a layer eligible for styling: (name, numeric).
    fn style_columns(store: &crate::data::store::FeatureStore) -> Vec<(String, bool)> {
        use arrow::datatypes::DataType as DT;
        store
            .schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != store.geom_col && *i < store.base_fields())
            .filter_map(|(_, f)| match f.data_type() {
                DT::Int8 | DT::Int16 | DT::Int32 | DT::Int64 | DT::UInt8 | DT::UInt16
                | DT::UInt32 | DT::UInt64 | DT::Float16 | DT::Float32 | DT::Float64 => {
                    Some((f.name().clone(), true))
                }
                DT::Utf8 | DT::LargeUtf8 | DT::Utf8View | DT::Boolean | DT::Dictionary(_, _) => {
                    Some((f.name().clone(), false))
                }
                _ => None,
            })
            .collect()
    }

    fn open_style_dialog(&mut self, layer_id: u64, ctx: &egui::Context) {
        use crate::data::layer::{Ramp, StyleMode};
        let Some(l) = self.layers.iter().find(|l| l.id == layer_id) else { return };
        let cols = Self::style_columns(&l.store);
        if cols.is_empty() {
            self.push_error(format!("{}: no styleable columns", l.name));
            return;
        }
        // Start from the active styling, else the first numeric column.
        use crate::data::layer::ClassMethod;
        let (column, ramp, method) = match &l.style.style_by {
            Some(sb) => match &sb.mode {
                StyleMode::Graduated { method, .. } => (sb.column.clone(), sb.ramp, *method),
                StyleMode::Categorical { .. } => {
                    (sb.column.clone(), sb.ramp, ClassMethod::EqualInterval)
                }
            },
            None => {
                let c = cols
                    .iter()
                    .find(|(_, num)| *num)
                    .unwrap_or(&cols[0])
                    .0
                    .clone();
                (c, Ramp::Viridis, ClassMethod::EqualInterval)
            }
        };
        let mut d = StyleDialog {
            layer_id,
            column,
            numeric: true,
            ramp,
            method,
            min: 0.0,
            max: 1.0,
            breaks: None,
            categories: None,
        };
        self.style_dialog_select_column(&mut d, ctx, true);
        self.style_dialog = Some(d);
    }

    /// Resolve column kind + auto bounds / category fetch on (re)selection.
    fn style_dialog_select_column(
        &self,
        d: &mut StyleDialog,
        ctx: &egui::Context,
        auto_bounds: bool,
    ) {
        let Some(l) = self.layers.iter().find(|l| l.id == d.layer_id) else { return };
        let cols = Self::style_columns(&l.store);
        let Some((_, numeric)) = cols.iter().find(|(n, _)| *n == d.column) else { return };
        d.numeric = *numeric;
        d.categories = None;
        d.breaks = None;
        if d.numeric {
            if auto_bounds {
                let idx = l
                    .store
                    .schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == &d.column);
                if let Some((lo, hi)) = idx.and_then(|i| l.store.column_range(i)) {
                    d.min = lo;
                    d.max = hi;
                }
            }
            if d.method.needs_values() {
                // Classify from already-loaded rows in the background —
                // never from the whole dataset (see the dialog note).
                let idx = l
                    .store
                    .schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == &d.column);
                if let Some(idx) = idx {
                    let store = Arc::clone(&l.store);
                    let loaded = l.loaded.clone();
                    let (id, col, method) = (d.layer_id, d.column.clone(), d.method);
                    let tx = self.class_tx.clone();
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        let res = crate::data::loader::sample_loaded_values(
                            &store, &loaded, idx, 50_000,
                        )
                        .map(|mut vals| {
                            crate::data::layer::classify_breaks(method, &mut vals)
                        });
                        let _ = tx.send((id, col, res));
                        ctx.request_repaint();
                    });
                }
            } else {
                d.breaks = Some(Ok(crate::data::layer::equal_interval_breaks(
                    d.min, d.max,
                )));
            }
        } else if let Some(sql) = self.sql_layer_of(d.layer_id) {
            // Fetch the top category values in the background.
            let tx = self.cat_tx.clone();
            let (id, col) = (d.layer_id, d.column.clone());
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let res = crate::sql::engine::top_values(
                    &sql,
                    &col,
                    crate::data::layer::STYLE_BINS - 1,
                );
                let _ = tx.send((id, col, res));
                ctx.request_repaint();
            });
        }
    }

    fn poll_classes(&mut self) {
        while let Ok((layer_id, column, res)) = self.class_rx.try_recv() {
            if let Some(d) = &mut self.style_dialog {
                if d.layer_id == layer_id
                    && d.column == column
                    && d.numeric
                    && d.method.needs_values()
                {
                    d.breaks = Some(res);
                }
            }
        }
    }

    fn poll_categories(&mut self) {
        while let Ok((layer_id, column, res)) = self.cat_rx.try_recv() {
            if let Some(d) = &mut self.style_dialog {
                if d.layer_id == layer_id && d.column == column && !d.numeric {
                    d.categories = Some(res);
                }
            }
        }
    }

    fn style_window(&mut self, ctx: &egui::Context) {
        use crate::data::layer::{Ramp, StyleBy, StyleMode};
        if self.style_dialog.is_none() {
            return;
        }
        let layer_id = self.style_dialog.as_ref().unwrap().layer_id;
        let Some(layer_idx) = self.layers.iter().position(|l| l.id == layer_id) else {
            self.style_dialog = None;
            return;
        };
        let layer_name = self.layers[layer_idx].name.clone();
        let cols = Self::style_columns(&self.layers[layer_idx].store);
        let current = self.layers[layer_idx].style.style_by.clone();

        let mut open = true;
        let mut reselect = false;
        let mut apply: Option<StyleBy> = None;
        {
            let d = self.style_dialog.as_mut().unwrap();
            egui::Window::new(format!("Style — {layer_name}"))
                .id(egui::Id::new("style_dialog"))
                .open(&mut open)
                .default_width(360.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("column:");
                        let before = d.column.clone();
                        egui::ComboBox::from_id_salt("style_col")
                            .width(200.0)
                            .selected_text(&d.column)
                            .show_ui(ui, |ui| {
                                for (name, numeric) in &cols {
                                    ui.selectable_value(
                                        &mut d.column,
                                        name.clone(),
                                        format!(
                                            "{name} {}",
                                            if *numeric { "(numeric)" } else { "(text)" }
                                        ),
                                    );
                                }
                            });
                        if d.column != before {
                            reselect = true;
                        }
                    });
                    if d.numeric {
                        ui.horizontal(|ui| {
                            ui.label("ramp:");
                            egui::ComboBox::from_id_salt("style_ramp")
                                .selected_text(d.ramp.label())
                                .show_ui(ui, |ui| {
                                    for r in Ramp::ALL {
                                        ui.selectable_value(&mut d.ramp, *r, r.label());
                                    }
                                });
                            ui.label("classes:");
                            let before = d.method;
                            egui::ComboBox::from_id_salt("style_method")
                                .selected_text(d.method.label())
                                .show_ui(ui, |ui| {
                                    for m in crate::data::layer::ClassMethod::ALL {
                                        ui.selectable_value(&mut d.method, *m, m.label());
                                    }
                                });
                            if d.method != before {
                                reselect = true;
                            }
                        });
                        if d.method.needs_values() {
                            ui.label(
                                RichText::new(
                                    "classified from the currently loaded rows only \
                                     (never the whole dataset)",
                                )
                                .weak()
                                .small(),
                            );
                            match &d.breaks {
                                None => {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label(
                                            RichText::new("classifying loaded rows…").weak(),
                                        );
                                    });
                                }
                                Some(Err(e)) => {
                                    ui.label(
                                        RichText::new(e)
                                            .color(Color32::from_rgb(220, 60, 60)),
                                    );
                                }
                                Some(Ok(b)) => {
                                    ui.label(
                                        RichText::new(format!(
                                            "{} classes · breaks {:.4} … {:.4}",
                                            b.len() + 1,
                                            b.first().copied().unwrap_or(0.0),
                                            b.last().copied().unwrap_or(0.0),
                                        ))
                                        .weak()
                                        .small(),
                                    );
                                }
                            }
                        } else {
                            let before = (d.min, d.max);
                            ui.horizontal(|ui| {
                                ui.label("min:");
                                ui.add(egui::DragValue::new(&mut d.min).speed(0.1));
                                ui.label("max:");
                                ui.add(egui::DragValue::new(&mut d.max).speed(0.1));
                            });
                            if (d.min, d.max) != before {
                                d.breaks =
                                    Some(Ok(crate::data::layer::equal_interval_breaks(
                                        d.min, d.max,
                                    )));
                            }
                        }
                        // Ramp preview strip.
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width().min(320.0), 14.0),
                            egui::Sense::hover(),
                        );
                        let p = ui.painter();
                        let n = crate::data::layer::STYLE_BINS;
                        for i in 0..n {
                            let c = d.ramp.sample(i as f32 / (n - 1) as f32);
                            let x0 = rect.left() + rect.width() * i as f32 / n as f32;
                            let x1 = rect.left() + rect.width() * (i + 1) as f32 / n as f32;
                            p.rect_filled(
                                egui::Rect::from_min_max(
                                    egui::pos2(x0, rect.top()),
                                    egui::pos2(x1, rect.bottom()),
                                ),
                                0.0,
                                Color32::from_rgb(
                                    (c[0] * 255.0) as u8,
                                    (c[1] * 255.0) as u8,
                                    (c[2] * 255.0) as u8,
                                ),
                            );
                        }
                    } else {
                        match &d.categories {
                            None => {
                                ui.horizontal(|ui| {
                                    ui.spinner();
                                    ui.label(RichText::new("reading category values…").weak());
                                });
                            }
                            Some(Err(e)) => {
                                ui.label(
                                    RichText::new(e).color(Color32::from_rgb(220, 60, 60)),
                                );
                            }
                            Some(Ok(values)) => {
                                egui::ScrollArea::vertical().max_height(220.0).show(
                                    ui,
                                    |ui| {
                                        for (i, v) in values.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                let c = palette_color(i);
                                                let (r, _) = ui.allocate_exact_size(
                                                    egui::vec2(12.0, 12.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().rect_filled(r, 2.0, c);
                                                ui.label(v);
                                            });
                                        }
                                        ui.horizontal(|ui| {
                                            let (r, _) = ui.allocate_exact_size(
                                                egui::vec2(12.0, 12.0),
                                                egui::Sense::hover(),
                                            );
                                            ui.painter().rect_filled(
                                                r,
                                                2.0,
                                                Color32::from_gray(140),
                                            );
                                            ui.label(RichText::new("(other)").weak());
                                        });
                                    },
                                );
                            }
                        }
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        let ready = if d.numeric {
                            matches!(&d.breaks, Some(Ok(_)))
                        } else {
                            matches!(&d.categories, Some(Ok(v)) if !v.is_empty())
                        };
                        if ui.add_enabled(ready, egui::Button::new("Apply")).clicked() {
                            apply = Some(StyleBy {
                                column: d.column.clone(),
                                ramp: d.ramp,
                                mode: if d.numeric {
                                    StyleMode::Graduated {
                                        method: d.method,
                                        breaks: match &d.breaks {
                                            Some(Ok(b)) => b.clone(),
                                            _ => Vec::new(),
                                        },
                                    }
                                } else {
                                    StyleMode::Categorical {
                                        values: match &d.categories {
                                            Some(Ok(v)) => v.clone(),
                                            _ => Vec::new(),
                                        },
                                    }
                                },
                                classified_rows: None, // stamped on apply below
                            });
                        }
                        if current.is_some() && ui.button("Remove styling").clicked() {
                            apply = None;
                            // Explicit clear: applied below via marker.
                            d.min = f64::NAN; // marker consumed below
                        }
                    });
                });
        }

        if reselect {
            let mut d = self.style_dialog.take().unwrap();
            self.style_dialog_select_column(&mut d, ctx, true);
            self.style_dialog = Some(d);
        }
        let clear = self
            .style_dialog
            .as_ref()
            .is_some_and(|d| d.min.is_nan());
        if clear {
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                l.style.style_by = None;
            }
            self.style_dialog = None;
        } else if let Some(mut sb) = apply {
            // Same column + breaks: bins are unchanged, so ramp swaps are
            // free. Anything else re-bins the meshes.
            let needs_rebuild = match &current {
                Some(cur) => cur.column != sb.column || cur.mode != sb.mode,
                None => true,
            };
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                // Data-dependent classes remember the loaded extent they
                // were computed from (drives the staleness hint).
                if matches!(
                    &sb.mode,
                    crate::data::layer::StyleMode::Graduated { method, .. }
                        if method.needs_values()
                ) {
                    sb.classified_rows = Some(l.loaded_rows() as usize);
                }
                l.style.style_by = Some(sb);
            }
            if needs_rebuild {
                self.restyle_layer(layer_id, ctx);
            }
            self.style_dialog = None;
        }
        if !open {
            self.style_dialog = None;
        }
    }

    /// Rebuild a layer's meshes so features land in their style bins.
    fn restyle_layer(&mut self, layer_id: u64, ctx: &egui::Context) {
        let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) else { return };
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
            l.loaded.clone(),
            l.style.style_by.clone(),
        );
    }

    fn url_window(&mut self, ctx: &egui::Context) {
        let Some((url, profile, profiles, endpoint)) = &mut self.url_input else { return };
        let mut open = true;
        let mut submit: Option<Source> = None;
        egui::Window::new("Open URL")
            .id(egui::Id::new("open_url"))
            .open(&mut open)
            .default_width(440.0)
            .show(ctx, |ui| {
                ui.label("GeoParquet over HTTP(S) (needs range requests) or s3://bucket/key:");
                let edit = ui.add(
                    egui::TextEdit::singleline(url)
                        .hint_text("https://host/data.parquet · s3://bucket/key.parquet")
                        .desired_width(f32::INFINITY),
                );
                let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let is_s3 = url.starts_with("s3://");
                if is_s3 {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Endpoint:");
                        ui.add(
                            egui::TextEdit::singleline(endpoint)
                                .hint_text("(AWS) · s3.example.com · https://minio:9000")
                                .desired_width(f32::INFINITY),
                        );
                    })
                    .response
                    .on_hover_text(
                        "S3-compatible endpoint (path-style requests).\n\
                         Leave empty for AWS; profile endpoint_url and\n\
                         AWS_ENDPOINT_URL are also honored.",
                    );
                    ui.horizontal(|ui| {
                        ui.label("AWS profile:");
                        let current = profile
                            .clone()
                            .unwrap_or_else(|| "(auto: env / default)".into());
                        egui::ComboBox::from_id_salt("aws_profile")
                            .selected_text(current)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(profile, None, "(auto: env / default)");
                                for p in profiles.iter() {
                                    ui.selectable_value(profile, Some(p.clone()), p);
                                }
                            });
                    })
                    .response
                    .on_hover_text(
                        "Static credentials from ~/.aws/credentials.
                         Public buckets work anonymously with (auto).",
                    );
                }
                ui.add_space(4.0);
                let valid =
                    url.starts_with("http://") || url.starts_with("https://") || is_s3;
                if ui.add_enabled(valid, egui::Button::new("Open")).clicked()
                    || (enter && valid)
                {
                    let text = url.trim().to_string();
                    submit = Some(if is_s3 {
                        let ep = endpoint.trim();
                        Source::S3 {
                            uri: text,
                            profile: profile.clone(),
                            endpoint: (!ep.is_empty()).then(|| ep.to_string()),
                            url: String::new(),
                            len: 0,
                        }
                    } else {
                        Source::Remote { url: text, len: 0 }
                    });
                }
            });
        if let Some(src) = submit {
            // Length probe / presign run in the loader thread.
            self.enqueue_load(src, ctx);
            self.url_input = None;
        } else if !open {
            self.url_input = None;
        }
    }

    fn poll_optimizer(&mut self) {
        while let Ok(msg) = self.opt_rx.try_recv() {
            let Some(o) = &mut self.optimize else { continue };
            match msg {
                OptMsg::Progress(f, s) => o.progress = (f, s),
                OptMsg::Done(report, path) => {
                    o.running = false;
                    o.report = Some((*report, path));
                }
                OptMsg::Failed(e) => {
                    o.running = false;
                    o.error = Some(e);
                }
                OptMsg::Cardinalities(c) => {
                    o.cardinalities = Some(c);
                    o.card_pending = false;
                }
            }
        }
    }

    fn start_optimize(&mut self, dst: PathBuf, ctx: &egui::Context) {
        let (view, display) = (self.last_view_world, self.display.clone());
        let Some(o) = &mut self.optimize else { return };
        o.running = true;
        o.error = None;
        o.report = None;
        o.progress = (0.0, "starting".into());
        o.opts.filter_rect = if o.viewport_only {
            let rect = loader::viewport_to_data_bbox(view, &display, &o.crs);
            if rect.is_none() {
                o.running = false;
                o.error = Some("cannot map the viewport into the layer's CRS".into());
                return;
            }
            rect
        } else {
            None
        };
        // Partition plan from the dialog state.
        use crate::data::partition::{AdminJoinSpec, PartitionBy};
        o.opts.partition = match o.part_mode {
            PartMode::None => PartitionBy::None,
            PartMode::Fields => {
                if o.part_fields.is_empty() {
                    o.running = false;
                    o.error = Some("select at least one partition field".into());
                    return;
                }
                PartitionBy::Fields(o.part_fields.clone())
            }
            PartMode::AdaptiveH3 => PartitionBy::AdaptiveH3 {
                target_rows: o.adaptive_target,
                max_res: 10,
            },
        };
        if o.admin_layer.is_some() && o.admin_column.is_empty() {
            o.running = false;
            o.error = Some("pick the boundary layer's value column".into());
            return;
        }
        let admin_sel = (o.admin_layer, o.admin_column.clone(), o.admin_out.clone());
        let (src, opts, epsg) = (o.src.clone(), o.opts.clone(), o.epsg);
        let admin: Option<AdminJoinSpec> = admin_sel
            .0
            .and_then(|id| self.layers.iter().find(|l| l.id == id))
            .map(|bl| AdminJoinSpec {
                out_name: {
                    let n = admin_sel.2.trim();
                    if n.is_empty() { "admin".into() } else { n.to_string() }
                },
                store: Arc::clone(&bl.store),
                value_column: admin_sel.1.clone(),
                crs: bl.crs.clone(),
            });
        let tx = self.opt_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let progress = |f: f32, s: &str| {
                let _ = tx.send(OptMsg::Progress(f, s.to_string()));
                ctx.request_repaint();
            };
            let msg = match crate::data::optimize::optimize(&src, &dst, &opts, epsg, admin.as_ref(), &progress) {
                Ok(r) => OptMsg::Done(Box::new(r), dst),
                Err(e) => OptMsg::Failed(e),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    fn optimize_window(&mut self, ctx: &egui::Context) {
        use crate::data::optimize::{BloomMode, Codec, GpVersion};
        // Gathered before the dialog borrow: partition-field candidates of
        // the exported layer and polygon layers usable for admin joins.
        let (candidates, other_layers) = match &self.optimize {
            Some(o) => {
                let candidates: Vec<String> = self
                    .layers
                    .iter()
                    .find(|l| l.id == o.layer_id)
                    .map(|l| {
                        l.store
                            .schema
                            .fields()
                            .iter()
                            .enumerate()
                            .filter(|(i, f)| {
                                *i != l.store.geom_col
                                    && f.name() != "bbox"
                                    && matches!(
                                        f.data_type(),
                                        arrow::datatypes::DataType::Utf8
                                            | arrow::datatypes::DataType::LargeUtf8
                                            | arrow::datatypes::DataType::Utf8View
                                            | arrow::datatypes::DataType::Boolean
                                            | arrow::datatypes::DataType::Int8
                                            | arrow::datatypes::DataType::Int16
                                            | arrow::datatypes::DataType::Int32
                                            | arrow::datatypes::DataType::Int64
                                            | arrow::datatypes::DataType::UInt8
                                            | arrow::datatypes::DataType::UInt16
                                            | arrow::datatypes::DataType::UInt32
                                            | arrow::datatypes::DataType::UInt64
                                            | arrow::datatypes::DataType::Date32
                                            | arrow::datatypes::DataType::Date64
                                    )
                            })
                            .map(|(_, f)| f.name().clone())
                            .collect()
                    })
                    .unwrap_or_default();
                let others: Vec<(u64, String, Vec<String>)> = self
                    .layers
                    .iter()
                    .filter(|l| {
                        l.id != o.layer_id
                            && matches!(l.kind(), crate::data::geometry::GeomKind::Polygon)
                    })
                    .map(|l| {
                        let cols = l
                            .store
                            .schema
                            .fields()
                            .iter()
                            .filter(|f| {
                                matches!(
                                    f.data_type(),
                                    arrow::datatypes::DataType::Utf8
                                        | arrow::datatypes::DataType::LargeUtf8
                                        | arrow::datatypes::DataType::Utf8View
                                )
                            })
                            .map(|f| f.name().clone())
                            .collect();
                        (l.id, l.name.clone(), cols)
                    })
                    .collect();
                (candidates, others)
            }
            None => (Vec::new(), Vec::new()),
        };
        let Some(o) = &mut self.optimize else { return };
        let layer_id = o.layer_id;
        let mut open = true;
        let mut start: Option<PathBuf> = None;
        let mut load_result: Option<PathBuf> = None;
        let mut close = false;
        let mut want_cards = false;
        egui::Window::new(format!("Optimize — {}", o.layer_name))
            .id(egui::Id::new("optimize_dialog"))
            .open(&mut open)
            .default_width(400.0)
            .show(ctx, |ui| {
                if let Some((rep, path)) = &o.report {
                    use crate::data::info::fmt_bytes;
                    ui.label(
                        RichText::new(format!("Written: {}", path.display())).strong(),
                    );
                    ui.add_space(4.0);
                    egui::Grid::new("opt_report").num_columns(2).striped(true).show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, k: &str, v: String| {
                            ui.label(RichText::new(k).strong());
                            ui.label(v);
                            ui.end_row();
                        };
                        row(ui, "format", rep.version_label.clone());
                        row(ui, "rows", fmt_count(rep.rows as usize));
                        if rep.files > 1 {
                            row(ui, "files", format!("{} (partitioned)", rep.files));
                        }
                        row(
                            ui,
                            "size",
                            format!(
                                "{} ➡ {}",
                                fmt_bytes(rep.size_before),
                                fmt_bytes(rep.size_after)
                            ),
                        );
                        row(ui, "row groups", format!("{} ➡ {}", rep.rg_before, rep.rg_after));
                        row(
                            ui,
                            "rg bbox overlap",
                            format!(
                                "×{:.1} ➡ ×{:.1} ({:.0}% ➡ {:.0}% of possible)",
                                rep.overlap_before,
                                rep.overlap_after,
                                rep.overlap_frac_before() * 100.0,
                                rep.overlap_frac_after() * 100.0,
                            ),
                        );
                        row(
                            ui,
                            "bloom filters",
                            if rep.bloom_columns.is_empty() {
                                "none".into()
                            } else {
                                rep.bloom_columns.join(", ")
                            },
                        );
                        row(ui, "elapsed", format!("{} ms", rep.elapsed_ms));
                    });
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if rep.files <= 1 && ui.button("Load as layer").clicked() {
                            load_result = Some(path.clone());
                        }
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                    return;
                }

                ui.add_enabled_ui(!o.running, |ui| {
                    if ui
                        .radio(o.opts.version == GpVersion::V1_1, GpVersion::V1_1.label())
                        .clicked()
                    {
                        o.opts.version = GpVersion::V1_1;
                        o.opts.covering = true;
                    }
                    if ui
                        .radio(
                            o.opts.version == GpVersion::V1_1GeoArrow,
                            GpVersion::V1_1GeoArrow.label(),
                        )
                        .on_hover_text(
                            "Geometry as raw coordinate arrays: fastest decode, x/y column\n\
                             statistics prune for free. Needs a single geometry family\n\
                             (singles are promoted to their multi variant).",
                        )
                        .clicked()
                    {
                        o.opts.version = GpVersion::V1_1GeoArrow;
                        o.opts.covering = true;
                    }
                    if ui
                        .radio(o.opts.version == GpVersion::V2_0, GpVersion::V2_0.label())
                        .on_hover_text(
                            "Native geo statistics replace the covering column for pruning;\n\
                             needs GeoParquet 2.0 aware readers",
                        )
                        .clicked()
                    {
                        o.opts.version = GpVersion::V2_0;
                        o.opts.covering = false;
                    }
                    ui.separator();
                    egui::Grid::new("opt_opts").num_columns(2).show(ui, |ui| {
                        ui.label("Row group size");
                        egui::ComboBox::from_id_salt("opt_rg")
                            .selected_text(fmt_count(o.opts.row_group_size))
                            .show_ui(ui, |ui| {
                                for s in [16_384usize, 32_768, 65_536, 131_072] {
                                    ui.selectable_value(
                                        &mut o.opts.row_group_size,
                                        s,
                                        fmt_count(s),
                                    );
                                }
                            });
                        ui.end_row();
                        ui.label("Compression");
                        egui::ComboBox::from_id_salt("opt_codec")
                            .selected_text(o.opts.codec.label())
                            .show_ui(ui, |ui| {
                                for c in [Codec::Zstd, Codec::Snappy, Codec::Uncompressed] {
                                    ui.selectable_value(&mut o.opts.codec, c, c.label());
                                }
                            });
                        ui.end_row();
                        ui.label("Bloom filters");
                        egui::ComboBox::from_id_salt("opt_bloom")
                            .selected_text(o.opts.bloom.label())
                            .show_ui(ui, |ui| {
                                for b in
                                    [BloomMode::Preserve, BloomMode::AllAttributes, BloomMode::None]
                                {
                                    ui.selectable_value(&mut o.opts.bloom, b, b.label());
                                }
                            });
                        ui.end_row();
                    });
                    ui.checkbox(&mut o.opts.hilbert_sort, "Hilbert spatial sort")
                        .on_hover_text("Reorder features along a Hilbert curve over bbox centers");
                    ui.checkbox(&mut o.opts.covering, "bbox covering column")
                        .on_hover_text("Per-feature bbox struct column (GeoParquet 1.1 covering)");
                    ui.checkbox(&mut o.viewport_only, "viewport only")
                        .on_hover_text(
                            "Export only features intersecting the current map viewport",
                        );

                    ui.separator();
                    // --- derived columns ---
                    ui.horizontal(|ui| {
                        let mut on = o.opts.h3_resolution.is_some();
                        if ui
                            .checkbox(&mut on, "H3 cell column")
                            .on_hover_text(
                                "Add an h3_r{n} UInt64 column: the H3 cell of each \
                                 feature's centroid (joins/aggregations, or a \
                                 partition key below)",
                            )
                            .changed()
                        {
                            o.opts.h3_resolution = on.then_some(8);
                        }
                        if let Some(res) = o.opts.h3_resolution {
                            let mut r = res;
                            egui::ComboBox::from_id_salt("opt_h3res")
                                .selected_text(format!(
                                    "r{res} ({})",
                                    crate::data::partition::h3_res_hint(res)
                                ))
                                .width(170.0)
                                .show_ui(ui, |ui| {
                                    for cand in 0..=12u8 {
                                        ui.selectable_value(
                                            &mut r,
                                            cand,
                                            format!(
                                                "r{cand} ({})",
                                                crate::data::partition::h3_res_hint(cand)
                                            ),
                                        );
                                    }
                                });
                            o.opts.h3_resolution = Some(r);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Admin column:")
                            .on_hover_text(
                                "Attribute each feature from a boundary polygon layer \
                                 (state, county, ...) by centroid point-in-polygon — \
                                 load the boundaries as a layer first",
                            );
                        let current = o
                            .admin_layer
                            .and_then(|id| other_layers.iter().find(|(i, _, _)| *i == id))
                            .map(|(_, n, _)| n.clone())
                            .unwrap_or_else(|| "none".into());
                        egui::ComboBox::from_id_salt("opt_admin_layer")
                            .selected_text(current)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(o.admin_layer.is_none(), "none").clicked()
                                {
                                    o.admin_layer = None;
                                }
                                for (id, name, _) in &other_layers {
                                    if ui
                                        .selectable_label(o.admin_layer == Some(*id), name)
                                        .clicked()
                                    {
                                        o.admin_layer = Some(*id);
                                        o.admin_column.clear();
                                    }
                                }
                            });
                        if other_layers.is_empty() {
                            ui.label(
                                RichText::new("(load a boundary polygon layer)")
                                    .weak()
                                    .small(),
                            );
                        }
                    });
                    if let Some(aid) = o.admin_layer {
                        if let Some((_, _, cols)) =
                            other_layers.iter().find(|(i, _, _)| *i == aid)
                        {
                            ui.horizontal(|ui| {
                                ui.label("  value:");
                                egui::ComboBox::from_id_salt("opt_admin_col")
                                    .selected_text(if o.admin_column.is_empty() {
                                        "pick column…".into()
                                    } else {
                                        o.admin_column.clone()
                                    })
                                    .show_ui(ui, |ui| {
                                        for c in cols {
                                            ui.selectable_value(
                                                &mut o.admin_column,
                                                c.clone(),
                                                c,
                                            );
                                        }
                                    });
                                ui.label("as");
                                ui.add(
                                    egui::TextEdit::singleline(&mut o.admin_out)
                                        .desired_width(90.0),
                                );
                            });
                        }
                    }

                    // --- partitioning ---
                    ui.separator();
                    ui.label(RichText::new("Partition output").strong());
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut o.part_mode, PartMode::None, "single file");
                        ui.radio_value(&mut o.part_mode, PartMode::Fields, "by fields")
                            .on_hover_text(
                                "Hive directories (field=value/part-0.parquet); \
                                 partition columns live in the path only",
                            );
                        ui.radio_value(&mut o.part_mode, PartMode::AdaptiveH3, "adaptive H3")
                            .on_hover_text(
                                "Split centroid H3 cells until each file is under the \
                                 row target — balanced, non-overlapping spatial \
                                 partitions when no natural field exists",
                            );
                    });
                    match o.part_mode {
                        PartMode::None => {}
                        PartMode::Fields => {
                            if o.cardinalities.is_none() && !o.card_pending {
                                o.card_pending = true;
                                want_cards = true;
                            }
                            let mut derived: Vec<String> = Vec::new();
                            if let Some(r) = o.opts.h3_resolution {
                                derived.push(format!("h3_r{r}"));
                            }
                            if o.admin_layer.is_some() {
                                let n = o.admin_out.trim();
                                derived.push(if n.is_empty() { "admin".into() } else { n.into() });
                            }
                            for name in derived.iter().chain(candidates.iter()) {
                                let mut on = o.part_fields.contains(name);
                                let card = o
                                    .cardinalities
                                    .as_ref()
                                    .and_then(|m| m.get(name))
                                    .map(|c| format!("{c} distinct"))
                                    .unwrap_or_else(|| {
                                        if derived.contains(name) {
                                            "derived".into()
                                        } else if o.card_pending {
                                            "counting…".into()
                                        } else {
                                            String::new()
                                        }
                                    });
                                let warn = o
                                    .cardinalities
                                    .as_ref()
                                    .and_then(|m| m.get(name))
                                    .is_some_and(|&c| c > 500);
                                let label = format!("{name}  ({card})");
                                let text = if warn {
                                    RichText::new(label).color(Color32::from_rgb(230, 130, 60))
                                } else {
                                    RichText::new(label)
                                };
                                if ui.checkbox(&mut on, text).changed() {
                                    if on {
                                        o.part_fields.push(name.clone());
                                    } else {
                                        o.part_fields.retain(|f| f != name);
                                    }
                                }
                            }
                            if !o.part_fields.is_empty() {
                                ui.label(
                                    RichText::new(format!(
                                        "dirs: {}",
                                        o.part_fields
                                            .iter()
                                            .map(|f| format!("{f}=…"))
                                            .collect::<Vec<_>>()
                                            .join("/")
                                    ))
                                    .weak()
                                    .small(),
                                );
                            }
                        }
                        PartMode::AdaptiveH3 => {
                            ui.horizontal(|ui| {
                                ui.label("target rows per file:");
                                ui.add(
                                    egui::DragValue::new(&mut o.adaptive_target)
                                        .range(10_000..=10_000_000)
                                        .speed(10_000),
                                );
                            });
                        }
                    }
                });

                ui.add_space(6.0);
                if o.running {
                    ui.add(
                        egui::ProgressBar::new(o.progress.0)
                            .text(o.progress.1.clone())
                            .animate(true),
                    );
                } else if ui.button("Export…").clicked() {
                    let stem = o.src.name();
                    let stem = stem.trim_end_matches(".parquet");
                    let mut dialog = rfd::FileDialog::new();
                    if let Source::Local(p) = &o.src {
                        if let Some(dir) = p.parent() {
                            dialog = dialog.set_directory(dir);
                        }
                    }
                    if o.part_mode == PartMode::None {
                        if let Some(dst) = dialog
                            .set_file_name(format!("{stem}_optimized.parquet"))
                            .add_filter("GeoParquet", &["parquet"])
                            .save_file()
                        {
                            start = Some(dst);
                        }
                    } else if let Some(dir) = dialog.pick_folder() {
                        // Dataset root inside the chosen folder.
                        start = Some(dir.join(format!("{stem}_partitioned")));
                    }
                }
                if let Some(e) = &o.error {
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_rgb(230, 80, 80), e);
                }
            });
        // Count distinct values of the partition candidates (one scan, on
        // a worker thread) the first time "by fields" is selected.
        if want_cards {
            match self.sql_layer_of(layer_id) {
                Some(sl) => {
                    let cols = candidates.clone();
                    let tx = self.opt_tx.clone();
                    let egui_ctx = ctx.clone();
                    std::thread::spawn(move || {
                        let counts =
                            crate::sql::engine::distinct_counts(&sl, &cols).unwrap_or_default();
                        let _ = tx.send(OptMsg::Cardinalities(counts));
                        egui_ctx.request_repaint();
                    });
                }
                None => {
                    if let Some(o) = &mut self.optimize {
                        o.card_pending = false;
                    }
                }
            }
        }
        if let Some(dst) = start {
            self.start_optimize(dst, ctx);
        }
        if let Some(p) = load_result {
            self.enqueue_load(Source::Local(p), ctx);
            close = true;
        }
        // Keep the worker's state visible: ignore window close while running.
        if (close || !open) && self.optimize.as_ref().is_some_and(|o| !o.running) {
            self.optimize = None;
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
                                    "{} boxes — {} — avg overlap ×{:.1} = {:.0}% of possible {}",
                                    rg.boxes.len(),
                                    rg.source,
                                    rg.avg_overlap,
                                    rg.overlap_frac() * 100.0,
                                    if rg.poorly_clustered() {
                                        "(poorly clustered: consider Optimize…)"
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
                        row(
                            ui,
                            if layer.store.source.is_remote() { "url" } else { "path" },
                            layer.store.source.label(),
                        );
                        if info.files > 1 {
                            row(
                                ui,
                                "files",
                                format!(
                                    "{} (hive keys: {})",
                                    info.files,
                                    if layer.store.part_cols.is_empty() {
                                        "none".to_string()
                                    } else {
                                        layer.store.part_cols.join(", ")
                                    }
                                ),
                            );
                        }
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
        // The attribute batch may be column-capped, so locate the geometry
        // by name instead of the store's schema index.
        let geom_name = layer
            .store
            .schema
            .field(layer.store.geom_col)
            .name()
            .clone();
        let encoding = layer.store.encoding;
        let layer_name = layer.name.clone();
        ui.label(
            RichText::new(format!("{layer_name} · row {}", sel.feature.index))
                .weak()
                .small(),
        );
        if let Some(m) = sel.measure {
            let latlong = layer.crs.is_latlong;
            let text = match m {
                crate::picking::Measure::Length(l) => {
                    format!("length {}", fmt_length(l))
                }
                crate::picking::Measure::Area { area, perimeter } => {
                    format!("area {} · perimeter {}", fmt_area(area), fmt_length(perimeter))
                }
            };
            ui.label(RichText::new(text).small()).on_hover_text(if latlong {
                "geodesic on the WGS84 ellipsoid"
            } else {
                "planar, in the layer's CRS units (assumed meters)"
            });
        }
        if let Some((shown, total)) = self.attrs_truncated {
            ui.label(
                RichText::new(format!("showing {shown} of {total} columns"))
                    .weak()
                    .small(),
            )
            .on_hover_text(
                "Very wide schema: fetching every column would need one \
                 read per column chunk (slow on remote files)",
            );
        }
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
                                if field.name() == &geom_name {
                                    ui.label(
                                        RichText::new(geom_summary(
                                            batch,
                                            i,
                                            encoding,
                                        ))
                                        .weak(),
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
            if self.pick_pending {
                ui.spinner();
                ui.label(RichText::new("fetching feature…").weak());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.projection_selector(ui);
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
                if !self.errors.is_empty() {
                    let btn = egui::Button::new(
                        RichText::new(format!("⚠ {}", self.errors.len()))
                            .color(Color32::from_rgb(220, 60, 60)),
                    );
                    if ui.add(btn).clicked() {
                        self.show_errors = !self.show_errors;
                    }
                }
                for job in self.loading.values_mut() {
                    if ui
                        .small_button("✖")
                        .on_hover_text("Stop loading this file")
                        .clicked()
                    {
                        job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                        job.stage = "cancelling".into();
                    }
                    ui.add(
                        egui::ProgressBar::new(job.frac)
                            .desired_width(220.0)
                            .text(format!(
                                "{} — {} {:.0}%",
                                job.label.rsplit('/').next().unwrap_or(&job.label),
                                job.stage,
                                job.frac * 100.0
                            )),
                    );
                }
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
                let ctx = ui.ctx().clone();
                self.start_pick(w, tol, &ctx);
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
                self.refine_hold.clear();
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
                    bin_colors: None,
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
                    bin_colors: None,
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
                    bin_colors: None,
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

        // Two independent highlight layers: the SQL checked-rows selection
        // (cyan, below) and the picked feature (amber, on top).
        if let Some(chunks) = &self.sql_highlight_chunks {
            draws.push(LayerDraw {
                key: (SQL_HIGHLIGHT_KEY, self.sql_highlight_generation),
                composite_group: SQL_HIGHLIGHT_KEY,
                chunks: chunks.clone(),
                style: DrawStyle {
                    fill_color: [0.05, 0.75, 0.95, 0.35],
                    line_color: [0.1, 0.85, 1.0, 1.0],
                    point_color: [0.1, 0.85, 1.0, 1.0],
                    line_half_width_px: 1.8,
                    point_radius_px: 6.0,
                    bin_colors: None,
                },
            });
        }
        if let Some(chunks) = &self.highlight_chunks {
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
                    bin_colors: None,
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
        let push = |x: f64, y: f64, ring: &mut Vec<geo_types::Coord<f64>>| {
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

/// Default border/line color: a darkened shade of the layer color.
/// Curated, well-balanced palettes for the layer color pickers — picking
/// from swatches beats the infinite color wheel for a data workbench.
/// Rows: Tableau 10, ColorBrewer Dark2, ColorBrewer Set2 (soft).
const SWATCH_ROWS: &[(&str, &[[u8; 3]])] = &[
    ("Tableau", &[
        [0x4E, 0x79, 0xA7], [0xF2, 0x8E, 0x2B], [0xE1, 0x57, 0x59], [0x76, 0xB7, 0xB2],
        [0x59, 0xA1, 0x4F], [0xED, 0xC9, 0x48], [0xB0, 0x7A, 0xA1], [0xFF, 0x9D, 0xA7],
        [0x9C, 0x75, 0x5F], [0xBA, 0xB0, 0xAC],
    ]),
    ("Dark", &[
        [0x1B, 0x9E, 0x77], [0xD9, 0x5F, 0x02], [0x75, 0x70, 0xB3], [0xE7, 0x29, 0x8A],
        [0x66, 0xA6, 0x1E], [0xE6, 0xAB, 0x02], [0xA6, 0x76, 0x1D], [0x66, 0x66, 0x66],
    ]),
    ("Soft", &[
        [0x66, 0xC2, 0xA5], [0xFC, 0x8D, 0x62], [0x8D, 0xA0, 0xCB], [0xE7, 0x8A, 0xC3],
        [0xA6, 0xD8, 0x54], [0xFF, 0xD9, 0x2F], [0xE5, 0xC4, 0x94], [0xB3, 0xB3, 0xB3],
    ]),
];

/// Color button backed by the curated swatches: click opens a compact
/// palette popup (with the free picker as "custom" fallback).
/// Returns true when the color changed.
fn swatch_color_button(
    ui: &mut egui::Ui,
    id_salt: &str,
    color: &mut egui::Color32,
    hover: &str,
) -> bool {
    let size = egui::vec2(20.0, 14.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let resp = resp.on_hover_text(hover);
    let visuals = ui.style().interact(&resp);
    ui.painter().rect_filled(rect, 2.0, *color);
    ui.painter().rect_stroke(
        rect,
        2.0,
        visuals.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let mut changed = false;
    let popup_id = egui::Id::new(("swatch_popup", id_salt));
    // A plain popup (not a menu): menus close on any inner click, which
    // would kill the inline custom picker; swatches close explicitly.
    egui::Popup::from_toggle_button_response(&resp)
        .id(popup_id)
        .kind(egui::PopupKind::Popup)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
        for (name, row) in SWATCH_ROWS {
            ui.horizontal(|ui| {
                ui.add_sized(
                    egui::vec2(44.0, 14.0),
                    egui::Label::new(RichText::new(*name).weak().small()),
                );
                for c in *row {
                    let col = Color32::from_rgb(c[0], c[1], c[2]);
                    let (r, sw) =
                        ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
                    let stroke = if *color == col {
                        egui::Stroke::new(2.0, ui.visuals().strong_text_color())
                    } else {
                        ui.visuals().widgets.noninteractive.bg_stroke
                    };
                    ui.painter().rect_filled(r, 3.0, col);
                    ui.painter().rect_stroke(r, 3.0, stroke, egui::StrokeKind::Inside);
                    if sw.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        *color = col;
                        changed = true;
                        ui.close();
                    }
                }
            });
        }
        ui.separator();
        // Inline picker (no nested popup — those die with the parent).
        egui::CollapsingHeader::new(RichText::new("custom").weak().small())
            .id_salt("swatch_custom")
            .show(ui, |ui| {
                ui.set_max_width(220.0);
                if egui::color_picker::color_picker_color32(
                    ui,
                    color,
                    egui::color_picker::Alpha::Opaque,
                ) {
                    changed = true;
                }
            });
    });
    changed
}

/// "832 m" / "12.42 km" (meters in; CRS units for projected layers).
fn fmt_length(m: f64) -> String {
    if m < 1000.0 {
        format!("{m:.1} m")
    } else {
        format!("{:.2} km", m / 1000.0)
    }
}

/// "512.3 m²" / "3.42 ha" / "18.75 km²".
fn fmt_area(m2: f64) -> String {
    if m2 < 10_000.0 {
        format!("{m2:.1} m²")
    } else if m2 < 1_000_000.0 {
        format!("{:.2} ha", m2 / 10_000.0)
    } else {
        format!("{:.2} km²", m2 / 1_000_000.0)
    }
}

fn derived_line_color(color: egui::Color32) -> egui::Color32 {
    let c = egui::Rgba::from(color);
    egui::Rgba::from_rgba_premultiplied(c.r() * 0.55, c.g() * 0.55, c.b() * 0.55, 1.0).into()
}

fn resolve_style(s: &crate::data::layer::LayerStyle) -> DrawStyle {
    let rgba = egui::Rgba::from(s.color);
    let (r, g, b) = (rgba.r(), rgba.g(), rgba.b());
    let lc = egui::Rgba::from(s.line_color.unwrap_or_else(|| derived_line_color(s.color)));
    DrawStyle {
        fill_color: [r, g, b, s.fill_opacity * s.opacity],
        line_color: [lc.r(), lc.g(), lc.b(), lc.a() * s.opacity],
        point_color: [r, g, b, s.opacity],
        line_half_width_px: (s.line_width_px * 0.5).max(0.01),
        point_radius_px: s.point_radius_px.max(0.1),
        bin_colors: s.style_by.as_ref().map(|sb| Arc::new(sb.bin_colors())),
    }
}

fn geom_summary(
    batch: &arrow::record_batch::RecordBatch,
    geom_col: usize,
    encoding: crate::data::geoarrow::GeomEncoding,
) -> String {
    // Encoding-aware accessor: WKB and GeoArrow nested arrays alike.
    let geom = crate::data::geoarrow::GeomCol::new(batch.column(geom_col).as_ref(), encoding)
        .and_then(|g| g.geometry(0));
    match geom {
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
        self.poll_loader(&ctx);
        self.poll_optimizer();
        self.poll_picks();
        self.poll_repo();
        self.poll_categories();
        self.poll_classes();
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
            let src = if p.is_dir() { Source::Dir(p) } else { Source::Local(p) };
            self.enqueue_load(src, &ctx);
        }

        egui::Panel::top("menubar").show(ui, |ui| self.menu_bar(ui));
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        let mut sql_action = self.sql.poll();
        if self.sql.open {
            let view_world = self.last_view_world;
            let display = self.display.clone();
            egui::Panel::bottom("sql_console")
                .resizable(true)
                .default_size(240.0)
                .show(ui, |ui| {
                    if let Some(a) = self.sql.panel_ui(ui, &self.layers, view_world, &display)
                    {
                        sql_action = Some(a);
                    }
                });
        }
        match sql_action {
            Some(crate::sql::console::ConsoleAction::LoadLayer(path)) => {
                self.enqueue_load(Source::Local(path), &ctx);
            }
            Some(crate::sql::console::ConsoleAction::Select { crs, geoms }) => {
                self.apply_sql_selection(&crs, geoms);
            }
            Some(crate::sql::console::ConsoleAction::Zoom {
                crs,
                zoom,
                highlight,
            }) => {
                use geo::BoundingRect;
                if let Some(r) = zoom.bounding_rect() {
                    self.zoom_to_data_bbox([r.min().x, r.min().y, r.max().x, r.max().y], &crs);
                }
                self.apply_sql_selection(&crs, highlight);
            }
            Some(crate::sql::console::ConsoleAction::ClearSelection) => {
                self.sql_highlight_chunks = None;
                self.sql_highlight_generation += 1;
            }
            None => {}
        }
        egui::Panel::left("layers")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| self.layers_panel(ui));
        if self.selection.is_some() {
            // Floating over the map (upper right) instead of a side panel:
            // opening it must not resize the viewport.
            let map_corner =
                ui.available_rect_before_wrap().right_top() + egui::vec2(-12.0, 12.0);
            let mut open = true;
            egui::Window::new("Feature")
                .id(egui::Id::new("feature_attrs"))
                .open(&mut open)
                .pivot(egui::Align2::RIGHT_TOP)
                .default_pos(map_corner)
                .default_width(300.0)
                .resizable(true)
                .collapsible(false)
                .show(&ctx, |ui| self.attributes_panel(ui));
            if !open {
                self.clear_selection();
            }
        }
        self.errors_window(&ctx);
        self.info_window(&ctx);
        self.optimize_window(&ctx);
        self.url_window(&ctx);
        self.repo_window(&ctx);
        self.style_window(&ctx);
        self.poll_filters(&ctx);
        self.filter_window(&ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.map_panel(ui));

        if !self.loading.is_empty()
            || !self.rebuilding.is_empty()
            || self.sql.is_running()
            || !self.filter_pending.is_empty()
            || self.filter_dialog.as_ref().is_some_and(|d| d.testing)
        {
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
                        bin_colors: None,
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
                        bin_colors: None,
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
