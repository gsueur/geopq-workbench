use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use eframe::egui::{self, Color32, RichText};
use egui_phosphor::regular as ph;
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

/// The basemap combo's entry for drawing no tiles at all.
const NO_BASEMAP: &str = "No basemap";

// View settings a session starts from, and the values "Reset layout"
// returns to. Named once so the two cannot drift apart.
const DEFAULT_BASEMAP: usize = 0;
const DEFAULT_BOX_THRESHOLD_PX: f64 = 3.0;
const DEFAULT_REFINE_BUDGET_MB: u32 = 512;

/// How the raster basemap meets the current display projection.
#[derive(Clone, Copy)]
enum BasemapPlan {
    /// Display is Web Mercator: tiles are axis-aligned quads, drawn as-is.
    Mercator(usize),
    /// Tiles are reprojected onto a mesh. The index is the source actually
    /// drawn, which may be the label-free twin of the one selected.
    Warped(usize, crate::map::warp::WarpPlan),
    /// No tiles this frame. Carries the reason when there is one worth
    /// showing; None simply means the user chose no basemap.
    Off(Option<&'static str>),
}

impl BasemapPlan {
    /// The tile source on screen, for attribution.
    fn drawn_source(&self) -> Option<usize> {
        match self {
            BasemapPlan::Mercator(s) | BasemapPlan::Warped(s, _) => Some(*s),
            BasemapPlan::Off(_) => None,
        }
    }
}

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

/// A load queued behind the first job's projection decision: batch loads
/// must build in the display the first layer picks, not race it.
struct DeferredLoad {
    job: u64,
    layer_id: u64,
    source: Source,
    color: Color32,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

enum OptMsg {
    Progress(f32, String),
    Done(Box<crate::data::optimize::OptimizeReport>, PathBuf, Option<S3Dest>),
    /// As-is publish (no rewrite): destination, bytes uploaded.
    PublishedAsIs(S3Dest, u64),
    Failed(String),
    /// Distinct-value counts for partition-field candidates, tagged with
    /// the layer they were scanned for (the dialog may have moved on).
    Cardinalities(u64, std::collections::HashMap<String, usize>),
}

/// State of the per-layer "Optimize" export dialog (one at a time).
/// Partition mode chosen in the optimize dialog.
#[derive(PartialEq, Clone, Copy)]
enum PartMode {
    None,
    Fields,
    AdaptiveH3,
}

/// A load the loader paused at the quality gate: the opened store plus
/// everything needed to resume it (docs/OPEN_POLICY.md).
struct QualityGateState {
    job: u64,
    layer_id: u64,
    opened: loader::OpenedStore,
    color: Color32,
    auto_project: bool,
}

impl QualityGateState {
    /// Decline-memory key: enough to survive renames-in-place staying
    /// the "same" file only while the size matches.
    fn key(&self) -> String {
        format!(
            "{}|{}",
            self.opened.store.source.label(),
            self.opened.info.file_size
        )
    }
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
    /// Quality-gate flow: when the export completes, open the optimized
    /// file as a layer automatically.
    open_result: bool,
    /// Format pre-selected from the source's geometry types (GeoArrow
    /// when one family fits); tagged "recommended" in the picker.
    recommended: crate::data::optimize::GpVersion,
    /// Distinct counts per candidate partition field (computed on demand).
    cardinalities: Option<std::collections::HashMap<String, usize>>,
    card_pending: bool,
    /// Publish destination: upload the optimized output to S3/R2
    /// instead of keeping a local file.
    dest_s3: bool,
    s3_uri: String,
    s3_endpoint: String,
    s3_profile: Option<String>,
    s3_profiles: Vec<String>,
    /// Set when the finished report's output lives on S3/R2.
    report_s3: Option<S3Dest>,
    /// Layer ids merged into this export before optimizing.
    merge_with: std::collections::HashSet<u64>,
    /// Tag merged rows with the layer they came from.
    merge_source_col: bool,
    /// Publish the source file unchanged (no rewrite) — local sources
    /// with an S3 destination and no rewrite-requiring options only.
    upload_as_is: bool,
    /// Publish a STAC `collection.json` beside the data, as the
    /// distributing-geoparquet best practices recommend.
    stac: bool,
    /// Overwrite a dataset already published at the destination prefix.
    /// A partitioned publish that finds a foreign `collection.json`
    /// there refuses unless this is ticked: that prefix is somebody's
    /// dataset, and a wrong prefix is likelier than an intended replace.
    /// (Single-file publishes never need it — they merge instead.)
    replace_remote: bool,
    /// Finished as-is publish: destination and bytes uploaded.
    report_as_is: Option<(S3Dest, u64)>,
}

/// Where an optimized output was published, for the report and the
/// "Load as layer" button.
#[derive(Clone)]
struct S3Dest {
    uri: String,
    profile: Option<String>,
    endpoint: Option<String>,
}

/// Upload `local` (our freshly written collection.json) to the sibling
/// key of `data_uri`, merging it into whatever collection is already
/// published there — publishing a file into a prefix that has one means
/// adding a part to that dataset, not replacing it. Fetch trouble other
/// than a clean 404 aborts: not being able to look must never turn into
/// a blind overwrite.
fn upload_collection_merged(
    local: &std::path::Path,
    data_uri: &str,
    profile: Option<&str>,
    endpoint: Option<&str>,
) -> Result<(), String> {
    use crate::data::{source::aws, stac};
    let sibling = stac::sibling_uri(data_uri).ok_or("no prefix for collection.json")?;
    let existing = aws::fetch_small(&sibling, profile, endpoint)
        .map_err(|e| format!("cannot check for an existing collection.json: {e}"))?;
    if let Some(text) = existing {
        let ours: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(local).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let merged = crate::data::stac::merge_into(&text, &ours)?;
        let pretty = serde_json::to_string_pretty(&merged).map_err(|e| e.to_string())?;
        std::fs::write(local, pretty).map_err(|e| e.to_string())?;
    }
    aws::upload_file(local, &sibling, profile, endpoint, &|_, _| {})
        .map_err(|e| format!("collection.json upload failed: {e}"))
}

pub struct ViewerApp {
    camera: Camera,
    display: DisplayCrs,
    layers: Vec<VectorLayer>,
    /// Attribute tables: sources with no geometry, queryable and joinable
    /// but with no presence on the map.
    attr_tables: Vec<crate::data::attrs::AttrTable>,
    next_attr_id: u64,
    /// Serial for the imported parquet copies written to the temp dir.
    next_table_file: u64,
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
    /// A feature narrower than this many screen pixels is drawn from its
    /// bounding box instead of its geometry. Raise it to keep the fast
    /// box view longer, lower it to get real outlines sooner.
    box_threshold_px: f64,
    /// Geometry a single refinement pass may decode, in MB. The safety
    /// cap, not the display rule: it exists so one dense viewport cannot
    /// exhaust memory. Raising it loads more before it gives up.
    refine_budget_mb: u32,
    show_coastline: bool,
    /// Coastline generation the overlay chunks are currently built from
    /// (embedded 1:50m at world zoom, fetched 1:10m when zoomed in).
    coast_level: crate::data::coastline::CoastLevel,
    /// Cached row-group bbox overlays per layer id: (layer generation,
    /// chunks, world-space label anchors).
    rg_overlays: HashMap<u64, (u64, Arc<Vec<crate::data::geometry::ChunkMesh>>, Vec<[f64; 2]>)>,

    tiles: TileCache,
    basemap: Option<usize>,
    /// Basemap opacity. It sits under the data, so fading it back is how
    /// you keep context without the data competing with it.
    basemap_opacity: f32,
    /// The source to come back to when the basemap is switched off and on.
    last_basemap: usize,
    /// Last frame's basemap decision, so the layers panel can explain a
    /// substitution without redoing the projection sampling that produced
    /// it. The panel draws before the map, so it is one frame behind — for
    /// a note about the current view that is not worth a second pass.
    last_basemap_plan: BasemapPlan,

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
    /// Provisional framing from a loading layer's metadata extent, applied
    /// before its geometry exists. Unlike `fit_bounds` this does not count
    /// as the user moving the camera, so the exact fit still happens when
    /// the layer lands.
    frame_bounds: Option<[f64; 4]>,
    /// Pick the display projection automatically from the first loaded
    /// layer (data CRS if projected, extent-based equal-area otherwise);
    /// turned off by any manual projection choice.
    auto_projection: bool,
    /// Layers with a row-group append in flight.
    appending: HashSet<u64>,
    /// Job whose in-flight load decides the session projection
    /// (auto-projection, first layer); later jobs wait for it.
    projection_decider: Option<u64>,
    deferred_loads: Vec<DeferredLoad>,
    /// Has the user explicitly moved the camera (pan/zoom/fit) since
    /// startup? The automatic empty-map world fit does not count: "still
    /// the original full-world viewport" is what first-layer adoption
    /// checks before fitting to the layer.
    camera_moved: bool,
    /// Cancel flags of in-flight appends (layer id -> flag).
    append_cancel: HashMap<u64, Arc<std::sync::atomic::AtomicBool>>,
    /// Cancel flags of in-flight rebuilds (layer id -> flag). Spawning a
    /// new rebuild for a layer cancels the previous one (projection
    /// flip-flops), removal cancels outright.
    rebuild_cancel: HashMap<u64, Arc<std::sync::atomic::AtomicBool>>,
    /// Refit the camera when the current rebuild wave completes — set only
    /// by an explicit projection switch (world coordinates changed under
    /// the camera). Filter/restyle/load-triggered rebuilds must not move
    /// the viewport.
    fit_after_rebuilds: bool,
    /// Layers whose refinement was stopped by the user: paused until the
    /// camera moves again (else the same viewport would respawn it).
    refine_hold: HashSet<u64>,
    /// Layers with a part-file append in flight (distinct from a row
    /// append: it grows the store before it builds anything).
    part_appending: HashSet<u64>,
    /// Layers whose current viewport holds no part files they lack. Held
    /// separately from `refine_hold` so "no new parts here" never blocks
    /// row refinement of the parts already open.
    part_hold: HashSet<u64>,
    /// Layers rebuilding purely to merge their sections (after boxes were
    /// replaced by real rows). They keep drawing throughout: nothing
    /// about their coordinates has changed.
    consolidating: HashSet<u64>,
    /// Last RefineDeferred verdict per layer: the rows the selection had
    /// reached, and the geometry bytes when those are what stopped it.
    /// Drives the badge, and is cleared with refine_hold.
    refine_deferred: HashMap<u64, (u64, Option<u64>)>,
    /// Bumped on every camera move; refinement verdicts carry the epoch
    /// they were spawned under, so a deferral computed for an obsolete
    /// viewport can never arm the hold against the current one.
    cam_epoch: u64,
    /// Camera epoch each layer's in-flight refinement was spawned at.
    refine_epoch: HashMap<u64, u64>,
    /// Frames still owed after new geometry arrived: uploads happen in
    /// paint and CPU-mesh stripping observes them in the NEXT update, so
    /// an idle app would otherwise never strip (meshes stay resident
    /// twice). Armed by loader messages, counts down to zero.
    strip_probe: u8,
    /// Vector import dialog / conversion in flight.
    gpkg_import: Option<ImportState>,
    /// Grid summary dialog / aggregation in flight.
    grid_dialog: Option<GridState>,
    /// Layers side panel visibility (toolbar toggle).
    layers_open: bool,
    grid_n: u64,
    /// Files dropped while an import dialog was already open; imported
    /// one after another as each dialog closes.
    import_queue: Vec<PathBuf>,
    /// Inline layer rename in the layers panel: (layer id, draft label).
    rename_layer: Option<(u64, String)>,
    about_open: bool,
    /// "Reset layout" asked for, waiting on the confirmation it costs
    /// loaded layers.
    confirm_reset: bool,
    /// The one native file dialog allowed on screen at a time.
    pick_dialog: Option<PendingPick>,
    /// Attribute file waiting on the import dialog.
    attr_import: Option<AttrImport>,
    /// Kept so a worker spawned from a path without one can still wake
    /// the UI when it finishes.
    egui_ctx: egui::Context,
    attr_tx: Sender<AttrMsg>,
    attr_rx: Receiver<AttrMsg>,
    /// What the attribute worker is doing, for the status bar. Reading a
    /// wide remote file takes tens of seconds, which is far too long to
    /// hold the frame for.
    attr_busy: Option<String>,
    join_dialog: Option<JoinDialog>,
    join_tx: Sender<crate::sql::engine::SqlMsg>,
    join_rx: Receiver<crate::sql::engine::SqlMsg>,
    next_join_id: u64,
    /// Joined output written to temp, and the layer it replaces if any.
    join_replaces: Option<u64>,
    cookbook_open: bool,
    /// Decoded app icon for the About dialog (lazy).
    about_icon: Option<egui::TextureHandle>,
    /// Map panel rect (points) of the last frame, for cropping the
    /// Export image… screenshot to map content only.
    map_rect: egui::Rect,
    /// Loads paused at the quality gate, waiting for the user's answer
    /// (docs/OPEN_POLICY.md). The dialog shows the front entry.
    quality_gates: Vec<QualityGateState>,
    /// Files the user answered "load all" for (key: label|size), so the
    /// gate never re-asks. Persisted to disk.
    direct_files: HashSet<String>,
    /// SVG export collecting on a worker; the answer is the finished
    /// document, which then travels through the save dialog.
    svg_export: Option<Receiver<SvgExport>>,
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
    /// "Open as attribute table" in the URL dialog. A remote parquet
    /// could be either a layer or a table and nothing in the URL says
    /// which, so this is asked rather than guessed.
    url_as_table: bool,
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
    /// (layer, column) scans already started: the dialog asks every
    /// frame while the values are missing, and one scan is enough.
    cat_pending: std::collections::HashSet<(u64, String)>,
    cat_rx: Receiver<(u64, String, Result<Vec<String>, String>)>,
    /// Classification runs for the styling dialog (breaks from loaded rows).
    class_tx: Sender<(u64, String, Result<Vec<f64>, String>)>,
    class_rx: Receiver<(u64, String, Result<Vec<f64>, String>)>,
    /// "Reclassify from viewport" runs (legend button): new breaks for
    /// an already-styled layer.
    vreclass_tx: Sender<(u64, Result<Vec<f64>, String>)>,
    vreclass_rx: Receiver<(u64, Result<Vec<f64>, String>)>,
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
    /// Open-data catalog browser dialog (Some = open).
    catalog_browser: Option<CatalogBrowser>,
    /// Catalogs added this session and not saved for good. They survive
    /// the dialog closing — "for the session" means the app run — and
    /// die with it, which is the point.
    session_catalogs: Vec<crate::data::repo::Catalog>,
    dcat_tx: Sender<CatMsg>,
    dcat_rx: Receiver<CatMsg>,
    /// Portal downloads in flight. They outlive the browser window: the
    /// import dialog is what the user waits for, not the dialog they
    /// started from.
    downloads: Vec<Download>,
    dl_tx: Sender<DlMsg>,
    dl_rx: Receiver<DlMsg>,
    dl_next: u64,
    /// Names to give layers when their load finishes (keyed by job id) —
    /// repository themes would otherwise all be called "buildings".
    pending_names: HashMap<u64, String>,
    /// Grid outputs written to the temp directory. They back live layers
    /// for the session, so they can only go at exit; Export… is how a
    /// grid becomes a file the user keeps.
    temp_outputs: Vec<PathBuf>,
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
    /// Country-wide theme list (the union over every dataset manifest
    /// of the country), shown when a country is picked and no state is
    /// selected: (country code, None = fetch in flight).
    country_themes: Option<(String, Option<CountryThemesResult>)>,
    /// Unix seconds the dataset list was cached at (None = fetched live).
    cache_age: Option<u64>,
    /// Add-repository row (name, base URL).
    add: (String, String),
    /// Drops stale async results after repo/snapshot switches.
    generation: u64,
}

/// Browser over open-data portals (DCAT `data.json` catalogs): the saved
/// list, the session-only list, and the dataset pane of whichever one is
/// selected. Its own dialog, not a repository kind — a portal lists
/// datasets to fetch, it serves nothing this app reads in place.
struct CatalogBrowser {
    /// Catalogs saved for good (`catalogs.json`), in saved order.
    saved: Vec<crate::data::repo::Catalog>,
    /// Catalogs added this session only; mirrored to the app when the
    /// dialog closes so they outlive it.
    session: Vec<crate::data::repo::Catalog>,
    /// Selected catalog: (in the saved list?, index). None = just the lists.
    sel: Option<(bool, usize)>,
    /// Add-catalog row (base or `/data.json` URL).
    add_url: String,
    /// The selected portal's catalog (None = fetch in flight). Session
    /// state only: a portal rewrites its data.json in place, so there is
    /// nothing here worth caching on disk.
    dcat: Option<Result<crate::data::repo::DcatCatalog, String>>,
    /// Portal datasets ticked for opening, by index into the catalog.
    dcat_checked: std::collections::HashSet<usize>,
    /// Search over the dataset list.
    filter: String,
    /// Hide datasets whose only openable format is CSV: an attribute
    /// table, not a layer. A session-wide preference, not per catalog.
    geo_only: bool,
    /// Drops stale fetch results after a selection switch.
    generation: u64,
}

/// A portal's whole catalog answering, which is one document.
type CatMsg = (u64, Result<crate::data::repo::DcatCatalog, String>);

impl CatalogBrowser {
    fn selected_catalog(&self) -> Option<&crate::data::repo::Catalog> {
        let (saved, i) = self.sel?;
        if saved { self.saved.get(i) } else { self.session.get(i) }
    }

    /// Move session entry `i` to the saved list, stamped with today as
    /// its added-on date, and keep the selection pointing at it. The
    /// saved list stays alphabetical, as the next load would make it.
    fn save_for_good(&mut self, i: usize) {
        let mut c = self.session.remove(i);
        c.added_on = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .ok();
        let url = c.url.clone();
        self.saved.push(c);
        crate::data::repo::sort_catalogs(&mut self.saved);
        if let Err(e) = crate::data::repo::save_catalogs(&self.saved) {
            log::warn!("saving catalogs: {e}");
        }
        match self.sel {
            Some((false, j)) if j == i => {
                self.sel =
                    self.saved.iter().position(|c| c.url == url).map(|k| (true, k));
            }
            Some((false, j)) if j > i => self.sel = Some((false, j - 1)),
            _ => {}
        }
    }
}

/// One theme aggregated across a country: total features and the
/// dataset paths that actually carry it (exact part list, no 404s).
#[derive(Clone)]
struct CountryTheme {
    theme: String,
    features: u64,
    paths: Vec<String>,
}

type CountryThemesResult = Result<Vec<CountryTheme>, String>;

enum RepoMsg {
    Snapshots(u64, Result<Vec<crate::data::repo::Snapshot>, String>),
    /// Dataset list + the cache timestamp it came from (None = live fetch).
    Datasets(
        u64,
        Result<Vec<crate::data::repo::Dataset>, String>,
        Option<u64>,
    ),
    /// Manifest of one dataset: (generation, dataset index, result). The
    /// index pins the result to the dataset it was fetched for — a slow
    /// fetch must not fill the panel of a dataset selected later.
    Manifest(u64, usize, Result<crate::data::repo::Manifest, String>),
    /// Union of every dataset manifest of one country: (generation,
    /// country code, themes). The code pins the result like Manifest's
    /// index does.
    CountryThemes(u64, String, CountryThemesResult),
}

/// A portal file being fetched before it enters the import path.
///
/// GeoPackage and GeoJSON readers take a path, so these formats are the
/// only ones that touch the disk on the way in; parquet and CSV open
/// over HTTPS directly and never appear here.
struct Download {
    id: u64,
    /// Dataset title, for the status bar.
    label: String,
    got: u64,
    /// Only when the server states a Content-Length: portal endpoints
    /// that generate the export on the fly do not.
    total: Option<u64>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

enum DlMsg {
    Progress(u64, u64, Option<u64>),
    Done(u64, PathBuf),
    Failed(u64, String),
}

/// Cap on attribute columns fetched for the feature info panel. Wide
/// files (time series in columns: thousands of fields) would otherwise
/// need one range request per column chunk on remote sources.
const ATTR_COLS_CAP: usize = 256;

/// Vector-import dialog state (GeoPackage / Shapefile / GeoJSON): an
/// optional table picker (GeoPackage only), then a background conversion
/// to GeoParquet whose output opens as a layer.
struct ImportState {
    format: crate::data::import::ImportFormat,
    src: PathBuf,
    /// GeoPackage feature tables; empty for single-table formats.
    tables: Vec<crate::data::gpkg::GpkgTable>,
    selected: usize,
    running: bool,
    progress: f32,
    error: Option<String>,
    rx: Option<std::sync::mpsc::Receiver<ImportMsg>>,
}

impl ImportState {
    /// Where the conversion lands: beside the source, named after it
    /// (and the chosen table, for a GeoPackage).
    fn dst_for(&self, table: Option<&crate::data::gpkg::GpkgTable>) -> PathBuf {
        match table {
            Some(t) => {
                let stem = self
                    .src
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                self.src.with_file_name(format!("{stem}.{}.parquet", t.name))
            }
            None => self.src.with_extension("parquet"),
        }
    }
}

/// `dst` exists, is not older than `src`, and ends with the parquet
/// magic. Equal mtimes count as current: converting right after
/// downloading lands in the same second. The magic check keeps a
/// truncated file — left by a conversion killed before the rename-into-
/// place era — from turning one crash into a persistent open failure.
fn up_to_date(dst: &std::path::Path, src: &std::path::Path) -> bool {
    let modified = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if !matches!((modified(dst), modified(src)), (Some(d), Some(s)) if d >= s) {
        return false;
    }
    std::fs::File::open(dst)
        .and_then(|mut f| {
            use std::io::{Read, Seek, SeekFrom};
            f.seek(SeekFrom::End(-4))?;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            Ok(magic == *b"PAR1")
        })
        .unwrap_or(false)
}

enum ImportMsg {
    Progress(f32),
    Done(PathBuf),
    Failed(String),
}

/// What a native file dialog was opened for, so its answer can be routed
/// when it arrives a frame or more later.
///
/// The payload is data, not a closure, so the whole thing stays `Send` and
/// inspectable. Anything derivable from app state when the answer lands
/// (the optimize target, the style dialog being edited) is left there and
/// re-read rather than captured here.
enum PickFor {
    OpenFiles,
    OpenFolder,
    ImportVector,
    AttributeTable,
    /// Where to keep the imported copy of the table with this id.
    SaveTable(u64),
    LoadContext,
    SaveContext,
    /// A QGIS colour map for the style dialog still on screen.
    ColorMap,
    /// Optimize output: one file, or the folder a partitioned tree goes in.
    OptimizeFile,
    OptimizeFolder,
    /// The frame was captured before the dialog opened, so it travels with
    /// the request.
    Screenshot(Box<egui::ColorImage>),
    /// Same reason as the screenshot: the document is built from the
    /// camera of the frame the export was asked for, and the camera can
    /// move while the panel is up.
    ExportSvg(String),
}

/// A join being set up between an attribute table and a layer.
struct JoinDialog {
    table_id: u64,
    layer_id: u64,
    /// Column names on each side, and which of the table's to bring over.
    layer_key: String,
    table_key: String,
    fields: Vec<(String, bool)>,
    /// Keep features the table has no row for, with the added columns
    /// NULL. The safe default: an inner join on a mismatched key gives an
    /// empty layer, which reads as a styling fault rather than a key one.
    keep_unmatched: bool,
    /// Replace the layer rather than adding one beside it.
    replace: bool,
    /// (total, matched) from the last probe, or why it failed.
    probe: Option<Result<(i64, i64), String>>,
    /// Probe or join in flight, by job id.
    pending: Option<u64>,
    running: bool,
}

/// A finished attribute-file job from the worker thread.
enum AttrMsg {
    Inspected(String, Source, Result<Box<crate::data::attrs::Preview>, String>),
    /// The sample values for a dialog already on screen.
    Sampled(
        Source,
        Result<
            (
                Vec<crate::data::attrs::ColumnPreview>,
                usize,
                Option<crate::data::attrs::GeometryPlan>,
            ),
            String,
        >,
    ),
    Imported(
        String,
        Source,
        crate::data::attrs::GeometryPlan,
        Result<Box<crate::data::attrs::AttrData>, String>,
    ),
}

/// An attribute file being set up for import.
struct AttrImport {
    source: Source,
    name: String,
    preview: crate::data::attrs::Preview,
    /// Set when the delimiter or header setting changed and the columns
    /// have to be worked out again.
    reread: bool,
}

/// A native file dialog in flight on its own thread. See `spawn_pick`.
struct PendingPick {
    what: PickFor,
    /// Empty when the user cancelled.
    rx: Receiver<Vec<PathBuf>>,
}

/// Run a dialog that answers with one path, on the calling worker thread.
fn awaited_path(fut: impl std::future::Future<Output = Option<rfd::FileHandle>>) -> Vec<PathBuf> {
    futures::executor::block_on(fut)
        .map(|h| h.path().to_path_buf())
        .into_iter()
        .collect()
}

/// Run a dialog that answers with several.
fn awaited_paths(
    fut: impl std::future::Future<Output = Option<Vec<rfd::FileHandle>>>,
) -> Vec<PathBuf> {
    futures::executor::block_on(fut)
        .unwrap_or_default()
        .iter()
        .map(|h| h.path().to_path_buf())
        .collect()
}

/// The Open dialog, shared by the menu, the toolbar and Cmd/Ctrl+O.
fn pick_parquet_files(d: rfd::AsyncFileDialog) -> Vec<PathBuf> {
    awaited_paths(
        d.add_filter("GeoParquet", &["parquet", "geoparquet", "pq"])
            .pick_files(),
    )
}

fn pick_context_file(d: rfd::AsyncFileDialog) -> Vec<PathBuf> {
    awaited_path(d.add_filter("geopq context", &["json"]).pick_file())
}

/// Result of an async pick job.
struct PickMsg {
    job: u64,
    sel: Option<Selection>,
    attrs: Option<arrow::record_batch::RecordBatch>,
    /// (shown, total) when the attribute fetch was column-capped.
    truncated: Option<(usize, usize)>,
}

/// Arm a fresh cancel flag for a layer's background worker, cancelling
/// the one already in flight (if any). Free function so call sites can
/// hold disjoint borrows of other `ViewerApp` fields.
fn fresh_cancel(
    map: &mut HashMap<u64, Arc<std::sync::atomic::AtomicBool>>,
    layer_id: u64,
) -> Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};
    if let Some(prev) = map.get(&layer_id) {
        prev.store(true, Ordering::Relaxed);
    }
    let c = Arc::new(AtomicBool::new(false));
    map.insert(layer_id, Arc::clone(&c));
    c
}

/// Ask the allocator to hand freed pages back to the OS. Tessellation
/// churn leaves gigabytes of empty malloc depots parked in free lists on
/// macOS, and Activity Monitor keeps counting them as app memory; the
/// explicit relief makes "Reload to viewport" and layer removal visibly
/// free what they free. Worker thread — walking a large heap takes tens
/// of milliseconds. No-op on other platforms.
fn release_freed_memory() {
    #[cfg(target_os = "macos")]
    std::thread::spawn(|| {
        unsafe extern "C" {
            /// libmalloc; NULL zone = every zone, goal 0 = everything.
            fn malloc_zone_pressure_relief(
                zone: *mut std::ffi::c_void,
                goal: usize,
            ) -> usize;
        }
        let freed = unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
        log::info!("malloc pressure relief: {} MB returned to the OS", freed >> 20);
    });
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

/// Grid summary dialog: aggregate a numeric column into cells.
struct GridState {
    layer_id: u64,
    layer_name: String,
    /// Aggregable columns of the source layer, numeric and text alike.
    columns: Vec<String>,
    /// The text ones among them: they take majority/minority and turn
    /// the numeric-surface controls off.
    text_cols: std::collections::HashSet<String>,
    column: String,
    /// 0 = square grid, 1 = H3, 2 = A5.
    system: usize,
    /// Square cell size, data-CRS units.
    size: f64,
    h3_res: u8,
    a5_res: i32,
    stat: crate::data::grid::GridStat,
    kernel: crate::data::grid::Kernel,
    passes: u32,
    /// Square grids only: output isolines instead of cell polygons.
    /// Focal operation index: 0 none, 1 focal std, 2 open, 3 close,
    /// 4 hillshade.
    post: usize,
    /// Hillshade sun position, degrees.
    azimuth: f64,
    altitude: f64,
    contours: bool,
    levels: u32,
    /// Contour levels at value quantiles instead of equal steps.
    quantile_levels: bool,
    running: bool,
    progress: f32,
    error: Option<String>,
    rx: Option<Receiver<GridMsg>>,
}

enum GridMsg {
    Progress(f32),
    /// (temp file, layer name, cells, rows aggregated)
    Done(PathBuf, String, usize, u64),
    Failed(String),
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
    /// Number of classes (2..=STYLE_BINS); breaks carry classes - 1 values.
    classes: usize,
    /// Normalize by polygon area before classifying/rendering.
    per_area: bool,
    /// Equal-interval bounds (from column statistics, editable).
    min: f64,
    max: f64,
    /// Computed class breaks (None = classification in flight for
    /// data-dependent methods).
    breaks: Option<Result<Vec<f64>, String>>,
    /// Top values for categorical columns (None = fetch in flight).
    categories: Option<Result<Vec<String>, String>>,
    /// Official colour map recognized from those values (CORINE land
    /// cover and friends), and whether to use it. Detected on arrival,
    /// on by default: a dataset with a canonical palette is unreadable
    /// in any other one.
    color_map: Option<crate::data::colormap::ColorMap>,
    use_color_map: bool,
    /// Graduated only: also drive line width by the classes, ramping
    /// linearly from `width_min` to `width_max` px.
    width_by: bool,
    width_min: f32,
    width_max: f32,
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
        crate::theme::apply(&cc.egui_ctx);
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
        let (dcat_tx, dcat_rx) = channel();
        let (dl_tx, dl_rx) = channel();
        let (join_tx, join_rx) = channel();
        let (attr_tx, attr_rx) = channel();
        let (cat_tx, cat_rx) = channel();
        let (class_tx, class_rx) = channel();
        let (vreclass_tx, vreclass_rx) = channel();
        let display = DisplayCrs::hobo_dyer();
        let graticule_chunks = build_graticule(&display);
        let coastline_chunks = crate::data::coastline::build_coastline(&display);
        let mut app = Self {
            camera: Camera::default(),
            display,
            layers: Vec::new(),
            attr_tables: Vec::new(),
            next_attr_id: 0,
            next_table_file: 0,
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
            box_threshold_px: DEFAULT_BOX_THRESHOLD_PX,
            refine_budget_mb: DEFAULT_REFINE_BUDGET_MB,
            show_coastline: true,
            coast_level: Default::default(),
            rg_overlays: HashMap::new(),
            tiles: TileCache::new(cc.egui_ctx.clone()),
            basemap: Some(DEFAULT_BASEMAP),
            basemap_opacity: 1.0,
            last_basemap: DEFAULT_BASEMAP,
            last_basemap_plan: BasemapPlan::Off(None),
            load_tx,
            load_rx,
            opt_tx,
            opt_rx,
            optimize: None,
            gpkg_import: None,
            grid_dialog: None,
            layers_open: true,
            grid_n: 0,
            import_queue: Vec::new(),
            rename_layer: None,
            loading: HashMap::new(),
            pending_styles: HashMap::new(),
            next_job: 0,
            next_layer_id: 0,
            palette_idx: 0,
            pending_fit: true,
            fit_bounds: None,
            frame_bounds: None,
            auto_projection: true,
            appending: HashSet::new(),
            projection_decider: None,
            deferred_loads: Vec::new(),
            camera_moved: false,
            append_cancel: HashMap::new(),
            rebuild_cancel: HashMap::new(),
            fit_after_rebuilds: false,
            refine_hold: HashSet::new(),
            part_appending: HashSet::new(),
            part_hold: HashSet::new(),
            refine_deferred: HashMap::new(),
            consolidating: HashSet::new(),
            cam_epoch: 0,
            refine_epoch: HashMap::new(),
            strip_probe: 0,
            about_open: false,
            confirm_reset: false,
            pick_dialog: None,
            attr_import: None,
            egui_ctx: cc.egui_ctx.clone(),
            attr_tx,
            attr_rx,
            attr_busy: None,
            join_dialog: None,
            join_tx,
            join_rx,
            next_join_id: 0,
            join_replaces: None,
            cookbook_open: false,
            about_icon: None,
            map_rect: egui::Rect::ZERO,
            quality_gates: Vec::new(),
            direct_files: load_direct_files(),
            svg_export: None,
            last_cam: None,
            cam_changed_at: 0.0,
            last_view_world: [-10.0, -10.0, 10.0, 10.0],
            errors: Vec::new(),
            show_errors: false,
            url_input: None,
            url_as_table: false,
            info_open: None,
            stripped: HashSet::new(),
            epsg_input: String::new(),
            cursor_world: None,
            sql: crate::sql::console::SqlConsole::new(),
            filter_dialog: None,
            style_dialog: None,
            cat_tx,
            cat_pending: std::collections::HashSet::new(),
            cat_rx,
            class_tx,
            vreclass_tx,
            vreclass_rx,
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
            catalog_browser: None,
            session_catalogs: Vec::new(),
            dcat_tx,
            dcat_rx,
            downloads: Vec::new(),
            dl_tx,
            dl_rx,
            dl_next: 0,
            pending_names: HashMap::new(),
            temp_outputs: Vec::new(),
            display_gen: 0,
        };
        for f in files {
            // A CSV on the command line means the same thing as one
            // dropped on the window.
            if crate::data::attrs::is_tabular(&f) {
                app.open_attr_table(f);
            } else {
                app.enqueue_load(f, &cc.egui_ctx);
            }
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
        // Thematic default when the name gives the game away ("rivers" →
        // water blue); the rotating palette otherwise (not advanced on a
        // thematic hit, so unrecognized layers keep their variety).
        let color = crate::data::layer::name_color(&source.name()).unwrap_or_else(|| {
            let c = palette_color(self.palette_idx);
            self.palette_idx += 1;
            c
        });
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
        if auto_project {
            self.projection_decider = Some(job);
        }
        // Batch loads: while the first job's projection decision is in
        // flight, later jobs wait for it — spawning them now would build
        // their geometry in a display about to be replaced.
        if self.projection_decider.is_some() && self.projection_decider != Some(job) {
            if let Some(j) = self.loading.get_mut(&job) {
                j.stage = "waiting for projection".into();
            }
            self.deferred_loads.push(DeferredLoad {
                job,
                layer_id,
                source,
                color,
                cancel,
            });
            return job;
        }
        self.spawn_load_job(job, layer_id, source, color, cancel, auto_project, ctx);
        job
    }

    fn spawn_load_job(
        &mut self,
        job: u64,
        layer_id: u64,
        source: Source,
        color: Color32,
        cancel: Arc<std::sync::atomic::AtomicBool>,
        auto_project: bool,
        ctx: &egui::Context,
    ) {
        // Deferred jobs spawn later: stamp the display generation they
        // actually build for.
        if let Some(j) = self.loading.get_mut(&job) {
            j.display_gen = self.display_gen;
            j.stage = "queued".into();
        }
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
    }

    /// Resume a quality-gated load in Direct mode (decode everything).
    fn resume_gated(&mut self, gate: QualityGateState, ctx: &egui::Context) {
        let cancel = self
            .loading
            .get(&gate.job)
            .map(|j| Arc::clone(&j.cancel))
            .unwrap_or_default();
        if let Some(j) = self.loading.get_mut(&gate.job) {
            j.stage = "loading all rows".into();
        }
        loader::spawn_load_gated(
            LoaderHandle {
                tx: self.load_tx.clone(),
                egui_ctx: ctx.clone(),
            },
            gate.job,
            gate.layer_id,
            gate.opened,
            self.display.clone(),
            gate.color,
            self.last_view_world,
            gate.auto_project,
            cancel,
            None,
            true,
        );
    }

    /// Abandon a quality-gated load (dialog Cancel, or Optimize taking
    /// over): same cleanup as a failed load, without an error.
    fn drop_gated(&mut self, gate: &QualityGateState, ctx: &egui::Context) {
        self.loading.remove(&gate.job);
        if self.projection_decider == Some(gate.job) {
            self.projection_decider = None;
            self.flush_deferred_loads(ctx);
        }
    }

    /// Start the loads that waited on the projection decision.
    fn flush_deferred_loads(&mut self, ctx: &egui::Context) {
        for d in std::mem::take(&mut self.deferred_loads) {
            self.spawn_load_job(d.job, d.layer_id, d.source, d.color, d.cancel, false, ctx);
        }
    }

    /// Move the camera to the same geographic place after a projection
    /// switch (center preserved, ground scale approximated by a short
    /// east-west segment through it). Falls back to a layer fit when the
    /// transform has no finite answer there.
    fn transfer_camera(&mut self, old: &DisplayCrs) {
        use crate::data::crs::transform_point;
        let c = self.camera.center;
        let (px, py) = old.projected_from_world(c);
        let d = 1e-4; // measuring segment, world units in the old display
        let (qx, qy) = old.projected_from_world([c[0] + d, c[1]]);
        let moved = (|| {
            let (cx, cy) = transform_point(&old.crs, &self.display.crs, px, py).ok()?;
            let (ex, ey) = transform_point(&old.crs, &self.display.crs, qx, qy).ok()?;
            let cw = self.display.world_from_projected(cx, cy);
            let ew = self.display.world_from_projected(ex, ey);
            let r = ((ew[0] - cw[0]).powi(2) + (ew[1] - cw[1]).powi(2)).sqrt() / d;
            (cw[0].is_finite() && cw[1].is_finite() && r.is_finite() && r > 0.0)
                .then_some((cw, r))
        })();
        match moved {
            Some((cw, r)) => {
                use crate::map::camera::{MAX_ZOOM, MIN_ZOOM};
                self.camera.center = cw;
                self.camera.zoom = (self.camera.zoom - r.log2()).clamp(MIN_ZOOM, MAX_ZOOM);
            }
            None => self.pending_fit = true,
        }
    }

    /// Camera policy when the first layer's auto-projection lands: an
    /// untouched startup camera fits to the layer; a user-framed one keeps
    /// showing the same place, re-projected into the new display.
    fn camera_after_adoption(&mut self, old: &DisplayCrs) {
        if self.camera_moved {
            self.transfer_camera(old);
        } else {
            self.pending_fit = true;
        }
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
            fresh_cancel(&mut self.rebuild_cancel, l.id),
            l.style.style_by.clone(),
            l.box_layer,
        );
    }

    /// Remove a layer's filter: everything loads again.
    /// Drop every layer's loaded geometry and reload only what intersects
    /// the current viewport — memory relief after loading large extents.
    /// Uses the same planning as a first load (row-group pruning,
    /// per-feature selection, preview fallback), so a dense viewport still
    /// lands within the row budget.
    fn reload_layers_to_viewport(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        self.clear_selection();
        for l in &mut self.layers {
            // Direct layers hold everything by design; a viewport reload
            // could not prune them (no usable spatial index) and would
            // only re-decode the whole file.
            if l.mode == crate::data::layer::LayerMode::Direct {
                continue;
            }
            // A reload supersedes any in-flight refinement.
            self.appending.remove(&l.id);
            if let Some(c) = self.append_cancel.remove(&l.id) {
                c.store(true, Ordering::Relaxed);
            }
            self.refine_hold.remove(&l.id);
            self.part_hold.remove(&l.id);
            self.refine_deferred.remove(&l.id);
            // Row filters select rows the reload will not decode; clear
            // them rather than silently showing a different subset.
            l.filter = None;
            l.generation += 1;
            // Free the old meshes immediately — that is the point.
            l.sections = Vec::new();
            l.draw_gen += 1;
            let n = l.store.rg_starts().len().saturating_sub(1);
            l.loaded = vec![crate::data::layer::GroupLoad::None; n];
            self.rebuilding.insert(l.id);
            loader::spawn_reload(
                LoaderHandle {
                    tx: self.load_tx.clone(),
                    egui_ctx: ctx.clone(),
                },
                l.id,
                l.generation,
                l.store.clone(),
                l.crs.clone(),
                l.rg_bboxes.as_ref().map(|r| r.boxes.clone()),
                self.display.clone(),
                self.last_view_world,
                fresh_cancel(&mut self.rebuild_cancel, l.id),
                l.style.style_by.clone(),
            );
        }
        release_freed_memory();
    }

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
            fresh_cancel(&mut self.rebuild_cancel, l.id),
            l.style.style_by.clone(),
            l.box_layer,
        );
    }

    /// The layer-filter dialog: predicate editor with autocomplete.
    fn filter_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
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
        dict.extend(
            crate::sql::udf::NAMES
                .iter()
                .chain(crate::sql::agg::NAMES)
                .map(|s| s.to_string()),
        );
        for k in ["and", "or", "not", "like", "between", "in", "is null", "is not null"] {
            dict.push(k.into());
        }
        // A filter predicate has no FROM clause; no table names to offer.
        let dict = crate::sql::console::CompletionDict {
            tables: Vec::new(),
            all: dict,
            // A filter predicate names one layer's columns bare; there is
            // no alias to qualify them with.
            columns: Default::default(),
        };
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
            .constrain_to(floating_area).show(ctx, |ui| {
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
        let mut decider_done = false;
        let mut first_layer_fit = false;
        while let Ok(msg) = self.load_rx.try_recv() {
            // New geometry may arrive: owe the frames that upload it,
            // strip the CPU copies, evict superseded GPU buffers and let
            // wgpu's deferred destroy actually run — none of which happen
            // while the app idles without repaint requests.
            self.strip_probe = 8;
            match msg {
                LoadMsg::Progress { job, frac, stage } => {
                    if let Some(j) = self.loading.get_mut(&job) {
                        j.frac = frac;
                        j.stage = stage;
                    }
                }
                LoadMsg::Framed {
                    job,
                    display,
                    world,
                } => {
                    // Only worth doing for the first layer onto an empty
                    // map that nobody has framed themselves: a restored
                    // context brings its own camera, and a second layer
                    // must not yank the view off the first.
                    let restored = self.pending_styles.contains_key(&job);
                    if self.layers.is_empty()
                        && !self.camera_moved
                        && !restored
                        && self.loading.contains_key(&job)
                    {
                        if let Some(d) = display {
                            // Switch now so the framing, and the tiles it
                            // triggers, are already in the projection the
                            // layer will arrive in. No layers exist yet, so
                            // nothing has to be rebuilt.
                            self.adopt_display_lite(d);
                            // This job is building geometry for the display
                            // we just moved to — it is the reason we moved.
                            // Without this the generation bump would read as
                            // "projection changed mid-load" and cost a full
                            // second build on arrival.
                            if let Some(j) = self.loading.get_mut(&job) {
                                j.display_gen = self.display_gen;
                            }
                        }
                        self.frame_bounds = Some(world);
                    }
                }
                LoadMsg::Loaded {
                    job,
                    layer,
                    adopt_display,
                } => {
                    // Untracked job: cancelled by a session reset (e.g.
                    // Load context) — drop the result instead of
                    // resurrecting the layer.
                    let Some(j) = self.loading.remove(&job) else {
                        continue;
                    };
                    // Projection switched while this job was building?
                    // Its world geometry is in the old display's frame.
                    let stale = j.display_gen != self.display_gen;
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
                    let context_restored = self.pending_styles.contains_key(&job);
                    if let Some(style) = self.pending_styles.remove(&job) {
                        layer.style = style;
                    }
                    let new_layer_id = layer.id;
                    // The loader's auto-projection only applies if nothing
                    // changed mid-load: a manual pick bumps display_gen
                    // (stale) and turns auto_projection off.
                    let built_in_adopted = matches!(&adopt_display, Some((_, true)));
                    let adopt = if !stale && self.auto_projection {
                        adopt_display
                    } else {
                        None
                    };
                    if adopt.is_none() && (stale || built_in_adopted) {
                        // The arriving geometry is in another display's
                        // frame; rebuild it for the current one.
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
                            fresh_cancel(&mut self.rebuild_cancel, layer.id),
                            layer.style.style_by.clone(),
                            layer.box_layer,
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
                    // The only layer on the map always gets framed once its
                    // geometry is in the current display's coordinates —
                    // even if the camera was touched while it loaded (long
                    // direct loads invite panning around the empty world)
                    // and even without projection adoption. Restored
                    // contexts keep their saved camera instead.
                    if self.layers.len() == 1 && !context_restored {
                        first_layer_fit = true;
                        match &adopt {
                            // World coordinates are about to change: the
                            // fit must wait for the projection rebuild.
                            Some((_, false)) => {}
                            _ if self.rebuilding.contains(&new_layer_id) => {
                                self.fit_after_rebuilds = true;
                            }
                            _ => self.pending_fit = true,
                        }
                    }
                    match adopt {
                        // Geometry already built in the auto-adopted
                        // display — but layers that finished earlier are
                        // still in the previous frame and must rebuild.
                        Some((d, true)) => {
                            let old = self.display.clone();
                            self.adopt_display_lite(d);
                            self.rebuild_layers_for_display(Some(new_layer_id), ctx);
                            self.camera_after_adoption(&old);
                        }
                        // Post-build suggestion: full projection rebuild.
                        Some((d, false)) => rebuild_display = Some(d),
                        None => {}
                    }
                    // Beyond first-layer projection adoption, never move
                    // the viewport because a layer finished loading — a
                    // dense full-extent layer would yank the user away
                    // (and could trigger a huge refinement).
                    if self.projection_decider == Some(job) {
                        self.projection_decider = None;
                        decider_done = true;
                    }
                }
                LoadMsg::Rebuilt {
                    layer_id,
                    generation,
                    geometry,
                    stats_build_ms,
                    bad_geoms,
                } => {
                    let mut applied = false;
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            l.sections = vec![geometry];
                            l.draw_gen += 1;
                            l.stats.build_ms = stats_build_ms;
                            l.stats.bad_geoms = bad_geoms;
                            self.rebuilding.remove(&layer_id);
                            self.consolidating.remove(&layer_id);
                            self.rebuild_cancel.remove(&layer_id);
                            applied = true;
                        }
                    }
                    // Fit only when this message really finished the last
                    // rebuild of an explicit projection switch (world
                    // coordinates changed under the camera). Stale
                    // generations, removed layers and filter/restyle/load
                    // rebuilds must not move the viewport.
                    if applied && self.rebuilding.is_empty() && self.fit_after_rebuilds {
                        self.pending_fit = true;
                    }
                    if self.rebuilding.is_empty() {
                        self.fit_after_rebuilds = false;
                    }
                }
                LoadMsg::Reloaded {
                    layer_id,
                    generation,
                    geometry,
                    loaded,
                    rows,
                    bad_geoms,
                    build_ms,
                } => {
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            l.sections = vec![geometry];
                            l.draw_gen += 1;
                            l.loaded = loaded;
                            l.feature_count = rows;
                            l.stats.build_ms = build_ms;
                            l.stats.bad_geoms = bad_geoms;
                            self.rebuilding.remove(&layer_id);
                            self.consolidating.remove(&layer_id);
                            self.rebuild_cancel.remove(&layer_id);
                            if self.rebuilding.is_empty() {
                                release_freed_memory();
                            }
                        }
                    }
                    // A reload never moves the viewport.
                }
                LoadMsg::Appended {
                    layer_id,
                    generation,
                    geometry,
                    rows,
                    loaded,
                    done,
                } => {
                    let mut boxes_replaced = false;
                    // Appends stream in batches; the layer stays "appending"
                    // (no re-refine, spinner shown) until the last one.
                    if done {
                        self.appending.remove(&layer_id);
                        self.part_appending.remove(&layer_id);
                        self.append_cancel.remove(&layer_id);
                    }
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            log::info!(
                                "{}: appended {} row groups ({rows} features)",
                                l.name,
                                loaded.len()
                            );
                            // No draw_gen bump: a push lands at a fresh
                            // section index, so it takes a key of its own.
                            // Advancing the generation would re-key the
                            // sections already uploaded and force them to
                            // re-upload from CPU meshes that were freed
                            // after their first upload — the layer would
                            // blank out on every append.
                            l.sections.push(geometry);
                            l.feature_count += rows;
                            for (g, st) in loaded {
                                if let Some(slot) = l.loaded.get_mut(g as usize) {
                                    // A group leaving box display leaves its
                                    // boxes behind in the earlier section:
                                    // an append only adds. On a preview that
                                    // overlap was 1/stride of the features
                                    // and invisible under real coverage;
                                    // here it is every feature, so the box
                                    // fills sit under the real polygons
                                    // until the layer is consolidated.
                                    if matches!(slot, crate::data::layer::GroupLoad::Boxes { .. })
                                    {
                                        boxes_replaced = true;
                                    }
                                    *slot = st;
                                }
                            }
                            // Every append on a box layer is consolidated:
                            // its jobs are viewport rects, which re-decode
                            // rows the layer already had, and the rebuild
                            // is what removes those duplicates and
                            // restores boxes over the rest of the group.
                            if l.box_layer {
                                boxes_replaced = true;
                            }
                        }
                    }
                    // Rebuild once the append is complete, not per batch.
                    if boxes_replaced && done {
                        self.consolidate_after_boxes(layer_id, ctx);
                    }
                }
                LoadMsg::RebuildFailed {
                    layer_id,
                    generation,
                    error,
                } => {
                    // The layer keeps drawing its previous-generation
                    // sections. Only the LATEST rebuild's failure clears
                    // the gate — a superseded (cancelled) rebuild must not
                    // unmask the newer one still in flight.
                    let current = self
                        .layers
                        .iter()
                        .find(|l| l.id == layer_id)
                        .is_some_and(|l| l.generation == generation);
                    if current {
                        self.rebuilding.remove(&layer_id);
                        self.consolidating.remove(&layer_id);
                        self.rebuild_cancel.remove(&layer_id);
                    }
                    if error != loader::CANCELLED {
                        self.push_error(error);
                    }
                }
                LoadMsg::PartsOpened {
                    layer_id,
                    generation,
                    store,
                    added_boxes,
                    added_groups,
                    names,
                } => {
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                        if l.generation == generation {
                            log::info!("{}: opened parts {}", l.name, names.join(", "));
                            // Fragments append in global order, so every
                            // row-group index the layer already holds still
                            // means the same group. The new groups start
                            // undecoded; the Appended messages behind this
                            // one fill them in.
                            // Size everything from the store itself, not
                            // from the count in the message: `loaded` and
                            // `rg_bboxes` are indexed by global row group,
                            // and a length that disagreed with the store
                            // would silently point refinement at the wrong
                            // groups rather than fail.
                            let groups = store.rg_starts().len() - 1;
                            l.store = store;
                            l.loaded
                                .resize(groups, crate::data::layer::GroupLoad::None);
                            if let Some(rg) = &mut l.rg_bboxes {
                                if added_boxes.len() == added_groups {
                                    rg.boxes.extend(added_boxes);
                                }
                                if rg.boxes.len() != groups {
                                    // Better no boxes than boxes that
                                    // belong to other groups: pruning would
                                    // read the wrong parts. Dropping them
                                    // only costs refinement precision.
                                    log::warn!(
                                        "{}: row-group boxes out of step after a part append \
                                         ({} boxes, {groups} groups); dropping them",
                                        l.name,
                                        rg.boxes.len()
                                    );
                                    l.rg_bboxes = None;
                                }
                            }
                            l.info.files += names.len();
                        }
                    }
                }
                LoadMsg::AppendEnded { layer_id, error } => {
                    self.appending.remove(&layer_id);
                    self.append_cancel.remove(&layer_id);
                    let was_parts = self.part_appending.remove(&layer_id);
                    // Hold until the camera moves — for cancels AND
                    // failures: the unchanged viewport would otherwise
                    // respawn the identical (failing) job every frame,
                    // spamming errors and network requests. Stale-viewport
                    // endings don't hold: the current viewport still wants
                    // its own check.
                    //
                    // A part append that found nothing holds only itself:
                    // the parts already open may still have rows worth
                    // refining in this very viewport.
                    if was_parts && error == loader::NOTHING_TO_APPEND {
                        self.part_hold.insert(layer_id);
                    } else if self.refine_epoch.get(&layer_id) == Some(&self.cam_epoch) {
                        self.refine_hold.insert(layer_id);
                        if was_parts {
                            self.part_hold.insert(layer_id);
                        }
                    }
                    // NOTHING_TO_APPEND is the normal outcome of a pan
                    // that found no new parts, not a failure to report.
                    if error != loader::CANCELLED && error != loader::NOTHING_TO_APPEND {
                        self.push_error(error);
                    }
                }
                LoadMsg::RefineDeferred {
                    layer_id,
                    at_least_rows,
                    geom_bytes,
                } => {
                    self.appending.remove(&layer_id);
                    self.append_cancel.remove(&layer_id);
                    // The exact covering scan proved that viewport is still
                    // over budget. Hold (retry on camera move, with the
                    // "zoom in" badge) — but only if the camera hasn't
                    // moved since the check was spawned: a verdict about an
                    // old viewport must not block refining the current one,
                    // which the next frame will kick off.
                    if self.refine_epoch.get(&layer_id) == Some(&self.cam_epoch) {
                        self.refine_hold.insert(layer_id);
                        self.refine_deferred
                            .insert(layer_id, (at_least_rows, geom_bytes));
                    }
                }
                LoadMsg::QualityGate {
                    job,
                    layer_id,
                    opened,
                    color,
                    auto_project,
                } => {
                    let gate = QualityGateState {
                        job,
                        layer_id,
                        opened: *opened,
                        color,
                        auto_project,
                    };
                    if self.direct_files.contains(&gate.key()) {
                        // Standing "load all" answer for this file.
                        self.resume_gated(gate, ctx);
                    } else {
                        if let Some(j) = self.loading.get_mut(&job) {
                            j.stage = "file not optimized — waiting for your answer".into();
                        }
                        self.quality_gates.push(gate);
                    }
                }
                LoadMsg::Failed { job, source, error } => {
                    self.loading.remove(&job);
                    if self.projection_decider == Some(job) {
                        self.projection_decider = None;
                        decider_done = true;
                    }
                    // User-initiated stop: not an error.
                    if error != loader::CANCELLED {
                        // Low-level errors (HTTP status lines) already
                        // name the URL; don't print it twice.
                        if error.starts_with(source.as_str()) {
                            self.push_error(error);
                        } else {
                            self.push_error(format!("{source}: {error}"));
                        }
                    }
                }
            }
        }
        if let Some(d) = rebuild_display {
            // First-layer adoption via full rebuild: the fit waits for the
            // projection rebuilds to finish (fit_after_rebuilds, armed by
            // set_display). A touched camera normally transfers its framing
            // instead — but not for the map's only layer, which the user
            // always wants framed once it appears.
            let old = self.display.clone();
            self.set_display(d, ctx);
            if self.camera_moved && !first_layer_fit {
                self.fit_after_rebuilds = false;
                self.transfer_camera(&old);
            }
        }
        // The projection decision landed (or died): start the loads that
        // waited for it, in the display that decision produced.
        if decider_done {
            self.flush_deferred_loads(ctx);
        }
    }

    /// Switch the display projection without rebuilding layers (their
    /// geometry was already built in it by the loader).
    fn adopt_display_lite(&mut self, d: DisplayCrs) {
        self.display_gen += 1;
        self.display = d;
        self.clear_selection();
        self.graticule_chunks = build_graticule(&self.display);
        self.coastline_chunks =
            crate::data::coastline::build_coastline_at(&self.display, self.coast_level);
        self.graticule_generation += 1;
        self.rg_overlays.clear();
    }

    /// Stop every worker still streaming into the current session and drop
    /// what it was going to be applied to. Callers that replace the session
    /// wholesale need this: a late Loaded/Rebuilt/Appended message would
    /// otherwise land in the session that took its place.
    fn cancel_in_flight(&mut self) {
        use std::sync::atomic::Ordering;
        for j in self.loading.values() {
            j.cancel.store(true, Ordering::Relaxed);
        }
        self.loading.clear();
        for c in self.append_cancel.values() {
            c.store(true, Ordering::Relaxed);
        }
        self.append_cancel.clear();
        for c in self.rebuild_cancel.values() {
            c.store(true, Ordering::Relaxed);
        }
        self.rebuild_cancel.clear();
        self.projection_decider = None;
        self.deferred_loads.clear();
        self.pending_styles.clear();
        self.pending_filters.clear();
        self.pending_names.clear();
    }

    /// Back to a blank session: no layers, and the projection, camera and
    /// view settings the app starts with.
    ///
    /// Closing layers one at a time does not get you here. The first
    /// dataset of a session picks the projection and the app then stops
    /// choosing, so a file loaded in EPSG:3035 leaves every later one in
    /// it; the palette, the basemap and the detail thresholds all persist
    /// the same way. This is the one action that puts those back.
    fn reset_layout(&mut self, ctx: &egui::Context) {
        self.cancel_in_flight();
        self.layers.clear();
        self.attr_tables.clear();
        self.rebuilding.clear();
        self.appending.clear();
        self.consolidating.clear();
        self.refine_hold.clear();
        self.part_appending.clear();
        self.part_hold.clear();
        self.refine_deferred.clear();
        self.refine_epoch.clear();
        self.filter_pending.clear();
        self.cat_pending.clear();
        self.stripped.clear();
        self.rg_overlays.clear();
        self.import_queue.clear();
        self.quality_gates.clear();
        self.errors.clear();
        self.show_errors = false;
        self.clear_selection();
        self.sql_highlight_chunks = None;
        self.sql_highlight_generation += 1;
        // Dialogs and panels keyed to a layer that no longer exists.
        self.filter_dialog = None;
        self.style_dialog = None;
        self.grid_dialog = None;
        self.optimize = None;
        self.rename_layer = None;
        self.info_open = None;
        self.url_input = None;
        // View settings, back to the values `new` starts from.
        self.palette_idx = 0;
        self.auto_projection = true;
        self.fit_bounds = None;
        self.fit_after_rebuilds = false;
        self.basemap = Some(DEFAULT_BASEMAP);
        self.basemap_opacity = 1.0;
        self.last_basemap = DEFAULT_BASEMAP;
        self.show_graticule = true;
        self.show_coastline = true;
        self.coast_level = Default::default();
        self.box_threshold_px = DEFAULT_BOX_THRESHOLD_PX;
        self.refine_budget_mb = DEFAULT_REFINE_BUDGET_MB;
        self.layers_open = true;
        self.last_cam = None;
        self.last_view_world = [-10.0, -10.0, 10.0, 10.0];
        // Bumps display_gen, which is what makes any message that slipped
        // past the cancel above land as stale, and arms the empty-map fit
        // that pulls the camera back out to the whole world — the same
        // path a cold start takes.
        self.set_display(DisplayCrs::hobo_dyer(), ctx);
        self.camera = Camera::default();
        self.camera_moved = true;
        release_freed_memory();
    }

    /// Open a native file dialog on its own thread and route the answer on
    /// a later frame.
    ///
    /// The blocking dialogs cannot be used here. macOS drives those with
    /// `runModal`, which spins a nested event loop, so every OS event that
    /// arrives while the panel is up is delivered from inside it. Called
    /// from `update` — itself inside winit's event handler — that re-enters
    /// the handler, and winit aborts the process rather than unwinding:
    /// dropping a file on the window while the Open panel was up killed the
    /// app outright.
    ///
    /// The async dialog attaches a sheet with a completion handler instead,
    /// driven by the run loop that is already turning, so no loop nests at
    /// any point. Awaiting it blocks this worker, never the frame.
    ///
    /// One at a time: a file panel is modal, and the guard also stops the
    /// toolbar opening a second one behind the menu's.
    fn spawn_pick(
        &mut self,
        what: PickFor,
        ctx: &egui::Context,
        open: impl FnOnce(rfd::AsyncFileDialog) -> Vec<PathBuf> + Send + 'static,
    ) {
        if self.pick_dialog.is_some() {
            return;
        }
        let (tx, rx) = channel();
        let egui_ctx = ctx.clone();
        std::thread::spawn(move || {
            let picked = open(rfd::AsyncFileDialog::new());
            let _ = tx.send(picked);
            egui_ctx.request_repaint();
        });
        self.pick_dialog = Some(PendingPick { what, rx });
    }

    /// Apply a finished file dialog, if one has answered since last frame.
    fn poll_pick(&mut self, ctx: &egui::Context) {
        let Some(p) = &self.pick_dialog else { return };
        let paths = match p.rx.try_recv() {
            Ok(v) => v,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            // The worker died without answering: nothing to apply, but the
            // slot has to reopen or no dialog can be opened again.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pick_dialog = None;
                return;
            }
        };
        let what = self.pick_dialog.take().expect("checked above").what;
        // Cancelled.
        let Some(first) = paths.first().cloned() else {
            return;
        };
        match what {
            PickFor::OpenFiles => {
                for p in paths {
                    self.enqueue_load(Source::Local(p), ctx);
                }
            }
            PickFor::OpenFolder => {
                self.enqueue_load(Source::Dir(first), ctx);
            }
            PickFor::ImportVector => self.begin_import(first, ctx),
            PickFor::AttributeTable => {
                for p in paths {
                    self.open_attr_table(Source::Local(p));
                }
            }
            PickFor::SaveTable(id) => self.save_table_copy(id, first),
            PickFor::LoadContext => self.read_context_file(first, ctx),
            PickFor::SaveContext => self.write_context_file(first),
            PickFor::ColorMap => self.apply_color_map_file(first),
            PickFor::OptimizeFile => self.start_optimize(first, ctx),
            PickFor::OptimizeFolder => {
                // The dataset root goes inside the folder chosen.
                let Some(o) = &self.optimize else { return };
                let stem = o.src.name().trim_end_matches(".parquet").to_string();
                self.start_optimize(first.join(format!("{stem}_partitioned")), ctx);
            }
            PickFor::Screenshot(img) => self.write_screenshot(&img, &first),
            PickFor::ExportSvg(doc) => {
                let write = crate::map::svg::encode_for(&first, &doc)
                    .and_then(|bytes| {
                        std::fs::write(&first, bytes).map_err(|e| e.to_string())
                    });
                if let Err(e) = write {
                    self.push_error(format!("could not save {}: {e}", first.display()));
                }
            }
        }
    }

    /// Open a source as an attribute table. Synchronous: these are read
    /// whole and are the small side of a join, so the wait is a blink and
    /// a background job would cost more in machinery than it saves.
    fn open_attr_table(&mut self, source: Source) {
        let name = source.name();
        self.open_attr_table_named(source, name);
    }

    /// Read a sample and open the import dialog. Nothing is loaded until
    /// the plan is approved: inference is a guess, and a column silently
    /// typed wrong is found much later and in the wrong place.
    fn open_attr_table_named(&mut self, source: Source, name: String) {
        if self.attr_busy.is_some() {
            self.push_error("another table is still being read".into());
            return;
        }
        self.attr_busy = Some(format!("reading {name}"));
        let tx = self.attr_tx.clone();
        let egui_ctx = self.egui_ctx.clone();
        std::thread::spawn(move || {
            // Remote sources carry no length until probed, and an unprobed
            // one reads as an empty file rather than failing.
            let result = source
                .clone()
                .resolve()
                .and_then(|s| crate::data::attrs::inspect_fast(&s).map(|p| (s, Box::new(p))));
            let msg = match result {
                Ok((resolved, p)) => AttrMsg::Inspected(name, resolved, Ok(p)),
                Err(e) => AttrMsg::Inspected(name, source, Err(e)),
            };
            let _ = tx.send(msg);
            egui_ctx.request_repaint();
        });
    }

    /// Fetch the sample values for a dialog that is already up.
    fn spawn_sampling(&self, source: &Source, plan: &crate::data::attrs::ImportPlan) {
        let (tx, egui_ctx) = (self.attr_tx.clone(), self.egui_ctx.clone());
        let (source, plan) = (source.clone(), plan.clone());
        std::thread::spawn(move || {
            let out = crate::data::attrs::sample_columns(&source, &plan);
            let _ = tx.send(AttrMsg::Sampled(source, out));
            egui_ctx.request_repaint();
        });
    }

    fn poll_attrs(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.attr_rx.try_recv() {
            self.attr_busy = None;
            match msg {
                AttrMsg::Inspected(name, source, Ok(preview)) => {
                    // Up immediately with names and types; the values
                    // behind them follow.
                    if !preview.sampled {
                        self.spawn_sampling(&source, &preview.plan);
                    }
                    self.attr_import = Some(AttrImport {
                        source,
                        name,
                        preview: *preview,
                        reread: false,
                    });
                }
                AttrMsg::Sampled(source, result) => {
                    let Some(job) = &mut self.attr_import else { continue };
                    // The dialog may have moved on to another file.
                    if job.source.label() != source.label() {
                        continue;
                    }
                    match result {
                        Ok((columns, rows, points)) => {
                            if columns.len() == job.preview.columns.len() {
                                job.preview.columns = columns;
                                job.preview.sampled_rows = rows;
                                job.preview.sampled = true;
                                // A coordinate pair can only be spotted
                                // once the values are in.
                                if let Some(g) = points {
                                    job.preview.plan.geometry = g;
                                }
                            }
                        }
                        Err(e) => {
                            job.preview.sampled = true;
                            self.push_error(format!("could not read sample values: {e}"));
                        }
                    }
                }
                AttrMsg::Inspected(name, _, Err(e)) => {
                    self.push_error(format!("{name}: {e}"))
                }
                AttrMsg::Imported(name, source, geometry, Ok(d)) => {
                    self.finish_attr_import(name, source, geometry, *d, ctx)
                }
                AttrMsg::Imported(name, _, _, Err(e)) => {
                    self.push_error(format!("{name}: {e}"))
                }
            }
        }
    }

    /// Run an approved plan on a worker. A wide remote file takes tens of
    /// seconds to read, which is far too long to hold the frame for.
    fn import_attr_table(&mut self, job: AttrImport, _ctx: &egui::Context) {
        let AttrImport {
            source,
            name,
            preview,
            ..
        } = job;
        self.attr_busy = Some(format!("importing {name}"));
        let tx = self.attr_tx.clone();
        let egui_ctx = self.egui_ctx.clone();
        let plan = preview.plan.clone();
        std::thread::spawn(move || {
            let result = crate::data::attrs::import(&source, &plan).map(Box::new);
            let _ = tx.send(AttrMsg::Imported(name, source, plan.geometry, result));
            egui_ctx.request_repaint();
        });
    }

    /// Take delivery of a finished import.
    fn finish_attr_import(
        &mut self,
        name: String,
        source: Source,
        geometry: crate::data::attrs::GeometryPlan,
        d: crate::data::attrs::AttrData,
        ctx: &egui::Context,
    ) {
        use crate::data::attrs::{AttrTable, GeometryPlan};
        if let GeometryPlan::Points { x, y, epsg } = &geometry {
            self.import_as_points(&name, &d, x, y, *epsg, ctx);
            return;
        }
        let nulled = d.nulled.clone();
        // The typed, renamed, filtered result written once. Live reads
        // come from the batches already in hand; the file is what makes
        // the import survive as something you can keep or hand to another
        // tool.
        self.next_table_file += 1;
        let path = std::env::temp_dir().join(format!(
            "geopq_table_{}_{}.parquet",
            std::process::id(),
            self.next_table_file,
        ));
        let written = match crate::data::attrs::write_parquet(&path, &d) {
            Ok(()) => {
                self.temp_outputs.push(path.clone());
                Some(path)
            }
            Err(e) => {
                // The table is usable without it; say so and move on.
                self.push_error(format!("{name}: could not write the imported copy: {e}"));
                None
            }
        };
        let mut t = AttrTable::new(self.next_attr_id, name.clone(), source, d);
        t.parquet = written;
        log::info!(
            "attribute table {name}: {} rows, {} columns, {}",
            t.rows,
            t.schema.fields().len(),
            crate::data::info::fmt_bytes(t.bytes as u64),
        );
        self.next_attr_id += 1;
        self.attr_tables.push(t);
        // A table has no geometry, so nothing on the map changes; the
        // console is where it becomes visible.
        self.sql.open = true;
        // Said once, plainly: forcing a type is allowed to lose values,
        // but never quietly.
        for (col, n) in nulled {
            self.push_error(format!(
                "{name}: {n} value{} in {col} did not fit its type and became NULL",
                if n == 1 { "" } else { "s" },
            ));
        }
    }

    /// Copy a table's imported parquet somewhere the user keeps it.
    ///
    /// A copy, not a move: the original stays where the live table reads
    /// from, and saving twice to two places both work.
    fn save_table_copy(&mut self, id: u64, dst: PathBuf) {
        let Some(t) = self.attr_tables.iter().find(|t| t.id == id) else {
            return;
        };
        let (name, src) = (t.name.clone(), t.parquet.clone());
        let Some(src) = src else {
            self.push_error(format!("{name}: there is no imported copy to save"));
            return;
        };
        match std::fs::copy(&src, &dst) {
            Ok(_) => log::info!("{name}: saved to {}", dst.display()),
            Err(e) => self.push_error(format!("{name}: cannot write {}: {e}", dst.display())),
        }
    }

    /// Write the imported rows as GeoParquet points and open the result
    /// as a layer. A file with coordinates is a map layer, not a table.
    fn import_as_points(
        &mut self,
        name: &str,
        data: &crate::data::attrs::AttrData,
        x: &str,
        y: &str,
        epsg: u32,
        ctx: &egui::Context,
    ) {
        let crs = match crate::data::crs::Crs::from_epsg(epsg) {
            Ok(c) => c,
            Err(e) => {
                self.push_error(format!("{name}: EPSG:{epsg} — {e}"));
                return;
            }
        };
        self.next_table_file += 1;
        let path = std::env::temp_dir().join(format!(
            "geopq_points_{}_{}.parquet",
            std::process::id(),
            self.next_table_file,
        ));
        match crate::data::attrs::write_points(&path, data, x, y, &crs) {
            Ok(placed) => {
                self.temp_outputs.push(path.clone());
                let missing = data.rows.saturating_sub(placed);
                if missing > 0 {
                    // Rows without coordinates are kept, with no geometry.
                    // Saying so beats a feature count that does not add up.
                    self.push_error(format!(
                        "{name}: {missing} of {} rows had no usable coordinates and                          carry no geometry",
                        fmt_count(data.rows),
                    ));
                }
                self.enqueue_load(Source::Local(path), ctx);
            }
            Err(e) => self.push_error(format!("{name}: {e}")),
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
        self.display_gen += 1;
        self.display = display;
        self.selection = None;
        self.highlight_chunks = None;
        // World-space geometry; must rebuild for the new projection.
        self.sql_highlight_chunks = None;
        self.sql_highlight_generation += 1;
        self.graticule_chunks = build_graticule(&self.display);
        self.coastline_chunks =
            crate::data::coastline::build_coastline_at(&self.display, self.coast_level);
        self.graticule_generation += 1;
        self.rebuild_layers_for_display(None, ctx);
        if self.layers.is_empty() {
            self.pending_fit = true;
        } else {
            self.fit_after_rebuilds = true;
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
                fresh_cancel(&mut self.rebuild_cancel, l.id),
                l.style.style_by.clone(),
                l.box_layer,
            );
        }
    }

    /// Load rows that entered the viewport of partially loaded layers:
    /// unseen row groups get a per-feature viewport selection; groups whose
    /// earlier selection no longer covers the viewport are completed
    /// (complement rows) and become Full.
    /// Pull in part files of a multi-part collection that the current
    /// viewport wants and the layer does not have.
    ///
    /// Runs before row refinement and takes the same in-flight slot: a
    /// pass either grows the store or refines what is in it, never both,
    /// and whichever loses goes on the next camera settle. Growing first
    /// is the right order — refining rows of parts you are not looking at
    /// is work for a viewport the user has left.
    fn append_parts_for_view(&mut self, ctx: &egui::Context) {
        let view = self.last_view_world;
        for l in &self.layers {
            if l.store.stac_collection().is_none()
                || !l.style.visible
                || l.mode == crate::data::layer::LayerMode::Direct
                || self.appending.contains(&l.id)
                || self.rebuilding.contains(&l.id)
                || self.part_hold.contains(&l.id)
            {
                continue;
            }
            // STAC item bboxes are WGS84 lon/lat by spec, whatever the
            // parts themselves are in.
            let Some(rect) =
                loader::viewport_to_data_bbox(view, &self.display, crate::data::crs::wgs84_cached())
            else {
                continue;
            };
            self.appending.insert(l.id);
            self.part_appending.insert(l.id);
            self.refine_epoch.insert(l.id, self.cam_epoch);
            let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
            self.append_cancel.insert(l.id, Arc::clone(&cancel));
            loader::spawn_part_append(
                LoaderHandle {
                    tx: self.load_tx.clone(),
                    egui_ctx: ctx.clone(),
                },
                l.id,
                l.generation,
                l.store.clone(),
                l.crs.clone(),
                self.display.clone(),
                rect,
                l.box_layer,
                cancel,
                l.style.style_by.clone(),
                Some((
                    crate::data::loader::MAX_BUILD_ROWS,
                    (self.refine_budget_mb as u64) << 20,
                )),
            );
        }
    }

    fn refine_partial_layers(&mut self, ctx: &egui::Context) {
        use crate::data::layer::GroupLoad;
        use crate::data::loader::{complement_ranges, GroupSel};
        self.append_parts_for_view(ctx);
        let view = self.last_view_world;
        for l in &self.layers {
            if !l.is_partial()
                || l.mode == crate::data::layer::LayerMode::Direct
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
            // Box layers switch representation by scale, not by how much
            // data happens to sit under the viewport. While a typical
            // feature is a couple of pixels across, its bounding box *is*
            // its outline on screen and there is nothing to refine to;
            // past that, every viewport at that zoom refines, whether it
            // covers a city or a field. A density test instead would
            // refine one and refuse the other at the same zoom.
            if l.box_layer {
                // Data-CRS units per screen pixel: the viewport's data
                // width over its pixel width (camera scale is px per
                // world unit).
                let view_px = ((view[2] - view[0]) * self.camera.scale()).max(1.0);
                let px = (rect[2] - rect[0]) / view_px;
                let span = l.feature_span();
                if span > 0.0 && span < self.box_threshold_px * px {
                    continue;
                }
            }
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
                    GroupLoad::None | GroupLoad::Preview { .. } | GroupLoad::Boxes { .. } => {
                        jobs.push(GroupSel::Rect(g, rect))
                    }
                    st @ GroupLoad::Rows { ranges, .. } => {
                        if !st.covers(need) {
                            if l.box_layer {
                                // Ask for what this viewport needs. The
                                // complement below means "the whole rest of
                                // the group", which on land cover is tens
                                // of megabytes of geometry per group and
                                // busts the budget at every zoom level —
                                // the layer would ask you to zoom in
                                // forever. Rows already loaded get decoded
                                // again and the consolidating rebuild that
                                // follows drops the duplicates.
                                jobs.push(GroupSel::Rect(g, rect));
                            } else {
                                let n = (starts[g as usize + 1] - starts[g as usize]) as u32;
                                jobs.push(GroupSel::Ranges(g, complement_ranges(ranges, n)));
                            }
                        }
                    }
                }
            }
            if jobs.is_empty() {
                continue;
            }
            // The worker resolves covering/x-y rows exactly before it
            // applies the budget; never gate refinement on bbox area.
            log::info!("{}: checking {} row groups for refinement", l.name, jobs.len());
            self.appending.insert(l.id);
            self.refine_epoch.insert(l.id, self.cam_epoch);
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
                Some((
                    crate::data::loader::MAX_BUILD_ROWS,
                    (self.refine_budget_mb as u64) << 20,
                )),
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
                if ui.button(format!("{} Open…", ph::FOLDER_OPEN)).clicked() {
                    self.spawn_pick(PickFor::OpenFiles, &ctx, pick_parquet_files);
                }
                if ui
                    .button(format!("{} Open folder…", ph::FOLDERS))
                    .on_hover_text(
                        "Load a directory of GeoParquet files (hive-partitioned or not) \
                         as a single layer; key=value path segments become columns",
                    )
                    .clicked()
                {
                    self.spawn_pick(PickFor::OpenFolder, &ctx, |d| {
                        awaited_path(d.pick_folder())
                    });
                }
                if ui
                    .button(format!("{} Open attribute table…", ph::TABLE))
                    .on_hover_text(
                        "Load a parquet or CSV file with no geometry: columns to                          query and to join a layer against. It gets a SQL table,                          not a place on the map",
                    )
                    .clicked()
                {
                    self.spawn_pick(PickFor::AttributeTable, &ctx, |d| {
                        awaited_paths(
                            d.add_filter("Tabular data", &["parquet", "csv", "tsv", "txt"])
                                .pick_files(),
                        )
                    });
                }
                if ui.button(format!("{} Open URL…", ph::GLOBE)).clicked() && self.url_input.is_none() {
                    self.url_input = Some((
                        String::new(),
                        None,
                        crate::data::source::aws::profiles(),
                        String::new(),
                    ));
                }
                if ui
                    .button(format!("{} Repositories…", ph::GLOBE_HEMISPHERE_WEST))
                    .on_hover_text(
                        "Browse preconfigured GeoParquet repositories and load \
                         their layers directly",
                    )
                    .clicked()
                    && self.repo_browser.is_none()
                {
                    self.open_repo_browser(&ctx);
                }
                if ui
                    .button(format!("{} Data catalogs…", ph::BOOKS))
                    .on_hover_text(
                        "Browse open-data portals through their DCAT catalog \
                         (ArcGIS Hub, Socrata, CKAN) and open their datasets",
                    )
                    .clicked()
                    && self.catalog_browser.is_none()
                {
                    self.open_catalog_browser();
                }
                if ui
                    .button(format!("{} Import vector file…", ph::TRAY_ARROW_DOWN))
                    .on_hover_text(
                        "Convert a GeoPackage, Shapefile or GeoJSON file to \
                         GeoParquet (pure Rust, no GDAL) and open it",
                    )
                    .clicked()
                    && self.gpkg_import.is_none()
                {
                    self.spawn_pick(PickFor::ImportVector, &ctx, |d| {
                        awaited_path(
                            d.add_filter(
                                "Vector data (GeoPackage, Shapefile, GeoJSON)",
                                &["gpkg", "shp", "geojson", "json"],
                            )
                            .pick_file(),
                        )
                    });
                }
                ui.separator();
                if ui
                    .button(format!("{} Save context…", ph::FLOPPY_DISK))
                    .on_hover_text("Save layers, styles, camera and projection to a JSON file")
                    .clicked()
                {
                    self.spawn_pick(PickFor::SaveContext, &ctx, |d| {
                        awaited_path(
                            d.set_file_name("session.geopq.json")
                                .add_filter("geopq context", &["json"])
                                .save_file(),
                        )
                    });
                }
                if ui
                    .button(format!("{} Load context…", ph::DOWNLOAD_SIMPLE))
                    .on_hover_text("Restore a saved context (replaces current layers)")
                    .clicked()
                {
                    self.spawn_pick(PickFor::LoadContext, &ctx, pick_context_file);
                }
                if ui
                    .add_enabled(
                        !self.layers.is_empty()
                            || !self.loading.is_empty()
                            || !self.auto_projection,
                        egui::Button::new(format!(
                            "{} Reset layout",
                            ph::ARROW_COUNTER_CLOCKWISE
                        )),
                    )
                    .on_hover_text(
                        "Close every layer and go back to the projection, camera \
                         and view settings the app starts with",
                    )
                    .clicked()
                {
                    self.confirm_reset = true;
                }
                ui.separator();
                if ui
                    .button(format!("{} Export map image…", ph::CAMERA))
                    .on_hover_text(
                        "Save the current map view as a PNG (print-friendly; \
                         panels and menus are cropped out)",
                    )
                    .clicked()
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                        egui::UserData::default(),
                    ));
                }
                if ui
                    .add_enabled(
                        self.svg_export.is_none(),
                        egui::Button::new(format!("{} Export view to SVG…", ph::PEN_NIB)),
                    )
                    .on_hover_text(
                        "Save the current map view as vector paths, for a figure \
                         that goes into a document. The basemap is raster and is \
                         left out; everything else is exported as it is drawn",
                    )
                    .clicked()
                {
                    self.begin_svg_export(&ctx);
                }
                ui.separator();
                let quit_shortcut = if cfg!(target_os = "macos") { "⌘Q" } else { "Ctrl+Q" };
                if ui
                    .add(egui::Button::new("Quit").shortcut_text(quit_shortcut))
                    .clicked()
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("View", |ui| {
                if ui
                    .add_enabled(!self.layers.is_empty(), egui::Button::new(format!("{} Fit all layers", ph::CORNERS_OUT)))
                    .clicked()
                {
                    self.pending_fit = true;
                    self.camera_moved = true;
                }
                if ui
                    .add_enabled(
                        !self.layers.is_empty(),
                        egui::Button::new("↺ Reload to viewport"),
                    )
                    .on_hover_text(
                        "Drop all loaded geometry and reload only what intersects the \
                         current viewport — frees memory after loading large extents. \
                         Layer row filters are cleared.",
                    )
                    .clicked()
                {
                    let ctx = ui.ctx().clone();
                    self.reload_layers_to_viewport(&ctx);
                }
                ui.separator();
                ui.menu_button("Detail", |ui| {
                    ui.label(
                        RichText::new("When a dataset is too heavy to draw in full")
                            .small()
                            .weak(),
                    );
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.box_threshold_px)
                                .speed(0.1)
                                .range(0.5..=40.0)
                                .suffix(" px"),
                        )
                        .on_hover_text(
                            "Features narrower than this are drawn from their bounding \
                             boxes; wider ones get their real geometry. The switch \
                             depends on scale alone, so it behaves the same over a \
                             city and over open country. Raise it to keep the fast \
                             box view longer, lower it for real outlines sooner.",
                        );
                        ui.label("box below");
                    });
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.refine_budget_mb)
                                .speed(16.0)
                                .range(64..=4096u32)
                                .suffix(" MB"),
                        )
                        .on_hover_text(
                            "Most geometry one refinement pass may decode. A safety \
                             cap against a viewport dense enough to exhaust memory, \
                             not a display rule — a build costs roughly twice this \
                             in RAM, more when outlines are drawn.",
                        );
                        ui.label("per refine");
                    });
                });
                ui.checkbox(&mut self.show_graticule, "Graticule");
                ui.checkbox(&mut self.show_coastline, "Coastline");
                ui.separator();
                ui.checkbox(&mut self.sql.open, "SQL console");
            });
            ui.menu_button("Help", |ui| {
                if ui
                    .button(format!("{} GeoParquet cookbook", ph::BOOK_OPEN))
                    .on_hover_text(
                        "Versions, geometry encodings, and what makes a file fast",
                    )
                    .clicked()
                {
                    self.cookbook_open = true;
                }
                if ui.button(format!("{} ST_* function reference", ph::TABLE)).clicked() {
                    self.sql.open_with_help();
                }
                ui.separator();
                if ui.button(format!("{} About GeoPQ Workbench…", ph::INFO)).clicked() {
                    self.about_open = true;
                }
            });
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button(format!("{} Open…", ph::FOLDER_OPEN)).clicked() {
                self.spawn_pick(PickFor::OpenFiles, &ctx, pick_parquet_files);
            }
            if ui
                .button(format!("{} Table…", ph::TABLE))
                .on_hover_text(
                    "Open a parquet or CSV file with no geometry: columns to                      query and to join a layer against",
                )
                .clicked()
            {
                self.spawn_pick(PickFor::AttributeTable, &ctx, |d| {
                    awaited_paths(
                        d.add_filter("Tabular data", &["parquet", "csv", "tsv", "txt"])
                            .pick_files(),
                    )
                });
            }
            if ui.button(format!("{} URL…", ph::GLOBE)).clicked() && self.url_input.is_none() {
                self.url_input = Some((
                    String::new(),
                    None,
                    crate::data::source::aws::profiles(),
                    String::new(),
                ));
            }
            if ui
                .add_enabled(!self.layers.is_empty(), egui::Button::new(format!("{} Fit all", ph::CORNERS_OUT)))
                .clicked()
            {
                self.pending_fit = true;
                self.camera_moved = true;
            }
            ui.toggle_value(&mut self.sql.open, format!("{} SQL", ph::TERMINAL_WINDOW))
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
                // The default CloseOnClick would dismiss the dropdown the
                // moment the EPSG textbox is clicked; picks close it
                // explicitly below instead.
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
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
                                .and_then(|r| loader::union_of(&r.boxes));
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
                    if pick.is_some() || picked_auto {
                        ui.close();
                    }
                });
            if let Some(d) = pick {
                self.auto_projection = picked_auto;
                self.set_display(d, &ctx);
            } else if picked_auto {
                self.auto_projection = true;
            }
        }
    }

    /// The basemap row at the foot of the layers panel: visibility, which
    /// source, and how far to fade it back.
    ///
    /// It behaves like a layer because that is what it is on screen, and
    /// the two controls that matter for it — which one, and how strongly —
    /// belong next to the data it sits under, not in a menu.
    fn basemap_card(&mut self, ui: &mut egui::Ui) {
        let plan = self.last_basemap_plan;
        crate::theme::card(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut on = self.basemap.is_some();
                if ui.checkbox(&mut on, "").changed() {
                    // Remember the choice across an off/on toggle.
                    self.basemap = on.then_some(self.last_basemap);
                }
                ui.label(RichText::new(ph::MAP_TRIFOLD).weak())
                    .on_hover_text("Basemap");
                let current = match self.basemap {
                    Some(i) => TILE_SOURCES[i].name,
                    None => NO_BASEMAP,
                };
                egui::ComboBox::from_id_salt("basemap_source")
                    .selected_text(current)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        // First, so turning the basemap off does not mean
                        // scrolling past every source to find it.
                        ui.selectable_value(&mut self.basemap, None, NO_BASEMAP);
                        ui.separator();
                        for (i, src) in TILE_SOURCES.iter().enumerate() {
                            if ui
                                .selectable_value(&mut self.basemap, Some(i), src.name)
                                .clicked()
                            {
                                self.last_basemap = i;
                            }
                        }
                    });
            });
            if self.basemap.is_some() {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut self.basemap_opacity, 0.0..=1.0)
                            .show_value(false)
                            .text(""),
                    )
                    .on_hover_text("Basemap opacity");
                    ui.label(
                        RichText::new(format!("{:.0}%", self.basemap_opacity * 100.0))
                            .weak()
                            .small(),
                    );
                });
                // Why the map may not match the choice above: outside
                // Mercator a labelled style can be substituted or dropped.
                let note = match (plan, self.basemap) {
                    (BasemapPlan::Off(Some(why)), _) => Some(why.to_string()),
                    (BasemapPlan::Warped(drawn, _), Some(picked)) if drawn != picked => {
                        Some(format!("drawn as {}", TILE_SOURCES[drawn].name))
                    }
                    _ => None,
                };
                if let Some(n) = note {
                    ui.label(RichText::new(n).weak().small());
                }
            }
        });
        ui.add_space(4.0);
    }

    /// The attribute tables strip: what is loaded, how big, and the SQL
    /// name to write in a query.
    ///
    /// They have no visibility toggle and no order, because they have
    /// nothing on the map to show or to stack. The name is the point —
    /// it is what a join has to spell.
    fn attr_tables_card(&mut self, ui: &mut egui::Ui) {
        use crate::data::info::fmt_bytes;
        let names = crate::sql::console::attr_sql_names(&self.layers, &self.attr_tables);
        let mut remove: Option<u64> = None;
        let mut save: Option<u64> = None;
        let mut join: Option<u64> = None;
        let can_join = !self.layers.is_empty();
        ui.add_space(2.0);
        ui.label(RichText::new("Tables").weak().small());
        for (t, sql) in self.attr_tables.iter().zip(names) {
            crate::theme::card(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(ph::TABLE).weak())
                        .on_hover_text("Attribute table: no geometry, no map presence");
                    ui.label(&t.name).on_hover_text(t.source.label());
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui
                                .small_button(ph::X)
                                .on_hover_text("Close this table")
                                .clicked()
                            {
                                remove = Some(t.id);
                            }
                            if ui
                                .add_enabled(
                                    can_join,
                                    egui::Button::new(RichText::new("join").small()),
                                )
                                .on_hover_text(
                                    "Join this table onto a layer on a shared \
                                     column, and put the result on the map",
                                )
                                .on_disabled_hover_text("load a layer to join onto")
                                .clicked()
                            {
                                join = Some(t.id);
                            }
                            if t.parquet.is_some()
                                && ui
                                    .small_button(ph::FLOPPY_DISK)
                                    .on_hover_text(
                                        "Save the imported table as a GeoParquet-free \
                                         parquet file: the types, names and columns \
                                         chosen at import, ready to reopen without \
                                         the dialog",
                                    )
                                    .clicked()
                            {
                                save = Some(t.id);
                            }
                        },
                    );
                });
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} rows · {} cols · {}",
                            fmt_count(t.rows),
                            t.schema.fields().len(),
                            fmt_bytes(t.bytes as u64),
                        ))
                        .weak()
                        .small(),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.label(RichText::new(&sql).monospace().small())
                                .on_hover_text("Its name in SQL");
                        },
                    );
                });
            });
            ui.add_space(2.0);
        }
        if let Some(id) = remove {
            self.attr_tables.retain(|t| t.id != id);
            release_freed_memory();
        }
        if let Some(id) = join {
            let fields = self
                .attr_tables
                .iter()
                .find(|t| t.id == id)
                .map(|t| {
                    t.schema
                        .fields()
                        .iter()
                        .map(|f| (f.name().clone(), true))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            self.join_dialog = Some(JoinDialog {
                table_id: id,
                layer_id: self.layers.first().map(|l| l.id).unwrap_or(0),
                layer_key: String::new(),
                table_key: String::new(),
                fields,
                keep_unmatched: true,
                replace: false,
                probe: None,
                pending: None,
                running: false,
            });
        }
        if let Some(id) = save {
            let stem = self
                .attr_tables
                .iter()
                .find(|t| t.id == id)
                .map(|t| crate::data::attrs::sanitize(&t.name))
                .unwrap_or_else(|| "table".into());
            let ctx = ui.ctx().clone();
            self.spawn_pick(PickFor::SaveTable(id), &ctx, move |d| {
                awaited_path(
                    d.set_file_name(format!("{stem}.parquet"))
                        .add_filter("Parquet", &["parquet"])
                        .save_file(),
                )
            });
        }
    }

    fn layers_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Layers");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button(ph::CARET_DOUBLE_LEFT)
                    .on_hover_text("Collapse the layers panel")
                    .clicked()
                {
                    self.layers_open = false;
                }
            });
        });
        ui.separator();
        if self.layers.is_empty() && self.loading.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(28.0);
                ui.label(RichText::new(ph::STACK).size(30.0).weak());
                ui.add_space(4.0);
                ui.label(RichText::new("Drop .parquet files here\nor use Open…").weak());
            });
        }

        let mut remove: Option<u64> = None;
        let mut reorder: Option<(u64, i32)> = None;
        let mut fit_to: Option<[f64; 4]> = None;
        let mut info_open: Option<u64> = None;
        let mut load_all: Option<u64> = None;
        let mut optimize_open: Option<u64> = None;
        let mut filter_open: Option<u64> = None;
        let mut style_open: Option<u64> = None;
        let mut grid_open: Option<u64> = None;
        let mut filter_clear: Option<u64> = None;
        let mut reclass_req: Option<u64> = None;
        // Borders switched back on for a layer built without them.
        let mut rebuild_outlines: Option<u64> = None;
        let mut renaming = self.rename_layer.take();

        // The basemap, under everything on the map, pinned to the foot of
        // the panel. A bottom panel has to be declared before the content
        // that fills the rest, so it is placed here and drawn last.
        egui::Panel::bottom("basemap_row").show(ui, |ui| {
            crate::theme::compact(ui);
            self.basemap_card(ui);
        });
        if !self.attr_tables.is_empty() {
            // Above the basemap, below the layers: they are neither drawn
            // nor stacked, so they get their own strip rather than a place
            // in an order that means nothing for them.
            egui::Panel::bottom("attr_tables").show(ui, |ui| {
                crate::theme::compact(ui);
                self.attr_tables_card(ui);
            });
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            crate::theme::compact(ui);
            // Top-most layer first in the list.
            let rebuilding = &self.rebuilding;
            let filter_pending = &self.filter_pending;
            let n_layers = self.layers.len();
            for (idx, l) in self.layers.iter_mut().enumerate().rev() {
                let is_rebuilding = rebuilding.contains(&l.id);
                crate::theme::card(ui.style()).show(ui, |ui| {
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
                        let te_id = egui::Id::new(("layer_rename", l.id));
                        if renaming.as_ref().is_some_and(|(id, _)| *id == l.id) {
                            let draft = &mut renaming.as_mut().unwrap().1;
                            let resp = ui.add(
                                egui::TextEdit::singleline(draft)
                                    .id(te_id)
                                    .desired_width(150.0),
                            );
                            let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
                            if resp.lost_focus() {
                                if !esc {
                                    let t = draft.trim();
                                    if !t.is_empty() {
                                        l.name = t.to_string();
                                    }
                                }
                                renaming = None;
                            }
                        } else {
                            let resp = ui
                                .label(RichText::new(&l.name).strong())
                                .on_hover_text(format!(
                                    "{}\ndouble-click to rename the label",
                                    l.store.source.label()
                                ))
                                .interact(egui::Sense::click());
                            if resp.double_clicked() {
                                renaming = Some((l.id, l.name.clone()));
                                ui.ctx().memory_mut(|m| m.request_focus(te_id));
                            }
                        }
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.menu_button(ph::LIST, |ui| {
                                    if ui.button("Info…").clicked() {
                                        info_open = Some(l.id);
                                    }
                                    if ui
                                        .button("Rename…")
                                        .on_hover_text(
                                            "Display label only — the file is untouched. \
                                             The SQL table name follows the label.",
                                        )
                                        .clicked()
                                    {
                                        renaming = Some((l.id, l.name.clone()));
                                        ui.ctx().memory_mut(|m| {
                                            m.request_focus(egui::Id::new((
                                                "layer_rename",
                                                l.id,
                                            )))
                                        });
                                    }
                                    let single = !l.store.is_partitioned();
                                    if ui
                                        .add_enabled(single, egui::Button::new("Export…"))
                                        .on_hover_text(if !single {
                                            "Export works on single files; this layer is a \
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
                                    if ui
                                        .button("Grid summary…")
                                        .on_hover_text(
                                            "Aggregate a numeric column into square / H3 / A5 cells, \
                                             apportioned by covered area, with optional \
                                             smoothing and contour lines — the grid \
                                             becomes a new layer",
                                        )
                                        .clicked()
                                    {
                                        grid_open = Some(l.id);
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
                                                    "(poorly clustered: consider Export…)"
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
                                    .small_button(ph::MAGNIFYING_GLASS)
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
                        let toggle = |ui: &mut egui::Ui, on: &mut bool, txt: &str, hover: &str| {
                            let label = if *on {
                                RichText::new(txt)
                            } else {
                                RichText::new(txt).weak().strikethrough()
                            };
                            if ui
                                .label(label)
                                .interact(egui::Sense::click())
                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                .on_hover_text(hover)
                                .clicked()
                            {
                                *on = !*on;
                            }
                        };
                        match l.kind() {
                            crate::data::geometry::GeomKind::Point => {
                                ui.label("r:");
                                ui.add(
                                    egui::Slider::new(&mut l.style.point_radius_px, 0.5..=12.0)
                                        .show_value(false),
                                );
                                let color = l.style.color;
                                marker_shape_button(
                                    ui,
                                    &format!("layer{}", l.id),
                                    &mut l.style.point_shape,
                                    color,
                                );
                                toggle(
                                    ui,
                                    &mut l.style.lines_on,
                                    "border:",
                                    "click to toggle the symbol border",
                                );
                                ui.add_enabled(
                                    l.style.lines_on,
                                    egui::Slider::new(&mut l.style.line_width_px, 0.0..=4.0)
                                        .show_value(false),
                                );
                            }
                            crate::data::geometry::GeomKind::Polygon => {
                                toggle(
                                    ui,
                                    &mut l.style.fill_on,
                                    "fill:",
                                    "click to toggle fills (borders-only display)",
                                );
                                ui.add_enabled(
                                    l.style.fill_on,
                                    egui::Slider::new(&mut l.style.fill_opacity, 0.0..=1.0)
                                        .show_value(false),
                                );
                                let before = l.style.lines_on;
                                toggle(
                                    ui,
                                    &mut l.style.lines_on,
                                    "w:",
                                    "click to toggle borders (fill-only display)",
                                );
                                // A colour-map layer is built without
                                // outlines, so switching them on has
                                // nothing to draw until the meshes are
                                // rebuilt with them.
                                if !before
                                    && l.style.lines_on
                                    && l.style
                                        .style_by
                                        .as_ref()
                                        .is_some_and(|sb| sb.mode.is_color_map())
                                {
                                    rebuild_outlines = Some(l.id);
                                }
                                ui.add_enabled(
                                    l.style.lines_on,
                                    egui::Slider::new(&mut l.style.line_width_px, 0.0..=6.0)
                                        .show_value(false),
                                );
                                if l.style.lines_on {
                                    line_style_button(
                                        ui,
                                        &format!("layer{}", l.id),
                                        &mut l.style.line_pattern,
                                        &mut l.style.line_cap,
                                        l.style.color,
                                    );
                                }
                            }
                            _ => {
                                ui.label("w:");
                                ui.add(
                                    egui::Slider::new(&mut l.style.line_width_px, 0.2..=8.0)
                                        .show_value(false),
                                );
                                line_style_button(
                                    ui,
                                    &format!("layer{}", l.id),
                                    &mut l.style.line_pattern,
                                    &mut l.style.line_cap,
                                    l.style.color,
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
                                .small_button(ph::X)
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
                    let (layer_id, loaded_rows) = (l.id, l.loaded_rows());
                    let (kind, fill_on, fill_opacity, opacity) = (
                        l.kind(),
                        l.style.fill_on,
                        l.style.fill_opacity,
                        l.style.opacity,
                    );
                    let area_unit = l.crs.area_unit();
                    if let Some(sb) = &mut l.style.style_by {
                        if let Some(n0) = sb.classified_rows {
                            let n1 = loaded_rows as f64;
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
                        let fill_alpha = match kind {
                            crate::data::geometry::GeomKind::Polygon => {
                                if fill_on {
                                    fill_opacity * opacity
                                } else {
                                    1.0
                                }
                            }
                            _ => opacity,
                        };
                        if style_legend(ui, layer_id, sb, fill_alpha, area_unit) {
                            reclass_req = Some(layer_id);
                        }
                    }
                    if l.mode == crate::data::layer::LayerMode::Direct {
                        ui.label(
                            RichText::new(format!(
                                "loaded fully ({} rows) — unoptimized file, \
                                 viewport loading unavailable",
                                fmt_count(l.feature_count)
                            ))
                            .weak()
                            .small(),
                        );
                    }
                    if let Some((rows, geom_bytes)) = self.refine_deferred.get(&l.id) {
                        // Name what actually stopped it. On land cover the
                        // row count is trivial and the geometry is
                        // gigabytes, so "too dense" reads as nonsense
                        // against a number the user knows is small.
                        let text = match geom_bytes {
                            Some(b) => format!(
                                "real geometry here is {} ({} features)\nzoom in further",
                                crate::data::info::fmt_bytes(*b),
                                fmt_count(*rows as usize)
                            ),
                            None => format!(
                                "viewport too dense to refine (≥{} rows)\nzoom in",
                                fmt_count(*rows as usize)
                            ),
                        };
                        ui.label(
                            RichText::new(text)
                            .color(Color32::from_rgb(242, 140, 26))
                            .small(),
                        );
                    }
                    if l.is_partial() && l.filter.is_none() {
                        ui.horizontal(|ui| {
                            let preview = l.preview_rgs();
                            let partial = l.partial_rgs();
                            // Does the CURRENT viewport still show preview /
                            // missing rows? Off-screen groups staying
                            // decimated is normal and must not keep nagging
                            // "zoom in" after the view has refined.
                            let view_pending = {
                                use crate::data::layer::GroupLoad;
                                let rect = loader::viewport_to_data_bbox(
                                    self.last_view_world,
                                    &self.display,
                                    &l.crs,
                                );
                                match (l.rg_bboxes.as_ref(), rect) {
                                    (Some(rg), Some(rect)) => {
                                        loader::intersecting_rgs(&rg.boxes, rect)
                                            .into_iter()
                                            .any(|g| {
                                                let gb = rg.boxes[g as usize];
                                                let need = [
                                                    rect[0].max(gb[0]),
                                                    rect[1].max(gb[1]),
                                                    rect[2].min(gb[2]),
                                                    rect[3].min(gb[3]),
                                                ];
                                                match &l.loaded[g as usize] {
                                                    GroupLoad::Full => false,
                                                    GroupLoad::None
                                                    | GroupLoad::Preview { .. }
                                                    | GroupLoad::Boxes { .. } => true,
                                                    st @ GroupLoad::Rows { .. } => {
                                                        !st.covers(need)
                                                    }
                                                }
                                            })
                                    }
                                    _ => true,
                                }
                            };
                            let boxes = l.boxes_rgs();
                            // At a scale where a feature is a couple of
                            // pixels, its box is its outline: say so
                            // plainly rather than asking for a zoom that
                            // would change nothing visible.
                            let box_scale = {
                                let view_px = ((self.last_view_world[2]
                                    - self.last_view_world[0])
                                    * self.camera.scale())
                                .max(1.0);
                                loader::viewport_to_data_bbox(
                                    self.last_view_world,
                                    &self.display,
                                    &l.crs,
                                )
                                .map(|r| (r[2] - r[0]) / view_px)
                                .zip(Some(l.feature_span()))
                                .is_some_and(|(px, span)| {
                                    span > 0.0 && span < self.box_threshold_px * px
                                })
                            };
                            let (text, attention) = if boxes > 0 && box_scale {
                                (
                                    format!(
                                        "all {} features drawn from their\nbounding boxes: at this scale a feature\nis under a few pixels wide",
                                        fmt_count(l.feature_count)
                                    ),
                                    false,
                                )
                            } else if boxes > 0 && view_pending {
                                (
                                    format!(
                                        "all features drawn from their bounding\nboxes ({} of {} row groups)\nzoom in for real geometry",
                                        boxes,
                                        l.total_rgs()
                                    ),
                                    true,
                                )
                            } else if boxes > 0 {
                                (
                                    format!(
                                        "viewport loaded\n{} of {} row groups still drawn as\nbounding boxes off-screen",
                                        boxes,
                                        l.total_rgs()
                                    ),
                                    false,
                                )
                            } else if preview > 0 && view_pending {
                                (
                                    format!(
                                        "preview: {} of {} row groups decimated\nzoom in to load real rows",
                                        preview,
                                        l.total_rgs()
                                    ),
                                    true,
                                )
                            } else if preview > 0 {
                                (
                                    format!(
                                        "viewport loaded\n{} of {} row groups still decimated\noff-screen",
                                        preview,
                                        l.total_rgs()
                                    ),
                                    false,
                                )
                            } else if partial > 0 {
                                (
                                    format!(
                                        "partial: {}/{} row groups full, {} \
                                         viewport-filtered",
                                        l.full_rgs(),
                                        l.total_rgs(),
                                        partial
                                    ),
                                    true,
                                )
                            } else {
                                (
                                    format!(
                                        "partial: {}/{} row groups loaded",
                                        l.full_rgs(),
                                        l.total_rgs()
                                    ),
                                    true,
                                )
                            };
                            let label = if attention {
                                RichText::new(text).color(Color32::from_rgb(242, 140, 26))
                            } else {
                                RichText::new(text).weak()
                            };
                            ui.label(label.small());
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
        self.rename_layer = renaming;
        if let Some(id) = reclass_req {
            let ctx = ui.ctx().clone();
            self.start_viewport_reclassify(id, &ctx);
        }

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
            use std::sync::atomic::Ordering;
            self.layers.retain(|l| l.id != id);
            self.rebuilding.remove(&id);
            self.consolidating.remove(&id);
            self.appending.remove(&id);
            self.refine_hold.remove(&id);
            self.part_hold.remove(&id);
            self.refine_deferred.remove(&id);
            // Stop the removed layer's in-flight workers instead of letting
            // them stream/rebuild into the void.
            if let Some(c) = self.append_cancel.remove(&id) {
                c.store(true, Ordering::Relaxed);
            }
            if let Some(c) = self.rebuild_cancel.remove(&id) {
                c.store(true, Ordering::Relaxed);
            }
            if self.selection.as_ref().map(|s| s.layer_id) == Some(id) {
                self.clear_selection();
            }
            release_freed_memory();
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
                    let recommended = crate::data::optimize::GpVersion::preferred(
                        &l.info.geo.geometry_types,
                    );
                    self.optimize = Some(OptimizeState {
                        layer_id: l.id,
                        layer_name: l.name.clone(),
                        src: l.store.source.clone(),
                        epsg: l.crs.epsg,
                        crs: l.crs.clone(),
                        viewport_only: false,
                        opts: crate::data::optimize::OptimizeOptions {
                            xy_geom: l.store.xy_geom,
                            version: recommended,
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
                        open_result: false,
                        recommended,
                        dest_s3: false,
                        stac: true,
                        replace_remote: false,
                        s3_uri: String::new(),
                        s3_endpoint: String::new(),
                        s3_profile: None,
                        s3_profiles: crate::data::source::aws::profiles(),
                        report_s3: None,
                        merge_with: Default::default(),
                        merge_source_col: true,
                        upload_as_is: false,
                        report_as_is: None,
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
        if let Some(id) = rebuild_outlines {
            let ctx = ui.ctx().clone();
            self.restyle_layer(id, &ctx);
        }
        if let Some(id) = style_open {
            let ctx = ui.ctx().clone();
            self.open_style_dialog(id, &ctx);
        }
        if let Some(id) = grid_open {
            self.open_grid_dialog(id);
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
                            GroupLoad::None
                            | GroupLoad::Preview { .. }
                            | GroupLoad::Boxes { .. } => Some(GroupSel::All(g as u32)),
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
                            None,
                        );
                    }
                }
            }
        }
    }

    /// Serialize the session to `path`. The dialog that chose it ran on a
    /// worker thread, so the state written here is the state as of now,
    /// not as of the click.
    fn write_context_file(&mut self, path: PathBuf) {
        use crate::context::{Context, LayerCtx, SourceCtx, StyleCtx, TableCtx, CONTEXT_VERSION};
        let ctx = Context {
            version: CONTEXT_VERSION,
            camera_center: self.camera.center,
            camera_zoom: self.camera.zoom,
            projection: crate::context::projection_token(&self.display),
            projection_name: Some(self.display.name.clone()),
            basemap: self.basemap,
            basemap_opacity: Some(self.basemap_opacity),
            show_graticule: self.show_graticule,
            show_coastline: self.show_coastline,
            box_threshold_px: Some(self.box_threshold_px),
            refine_budget_mb: Some(self.refine_budget_mb),
            layers: self
                .layers
                .iter()
                .map(|l| LayerCtx {
                    source: SourceCtx::of(&l.store.source),
                    style: StyleCtx::of(&l.style),
                    filter: l.filter.as_ref().map(|f| f.sql.clone()),
                    name: Some(l.name.clone()),
                })
                .collect(),
            tables: self
                .attr_tables
                .iter()
                .map(|t| TableCtx {
                    source: SourceCtx::of(&t.source),
                    name: t.name.clone(),
                })
                .collect(),
        };
        match serde_json::to_string_pretty(&ctx)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
        {
            Ok(()) => log::info!("context saved to {}", path.display()),
            Err(e) => self.push_error(format!("context save failed: {e}")),
        }
    }

    fn read_context_file(&mut self, path: PathBuf, ctx: &egui::Context) {
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
                // Cancel everything still in flight from the previous
                // session: a late Loaded/Rebuilt/Appended message must not
                // land in the restored one (or clobber its camera).
                self.cancel_in_flight();
                self.layers.clear();
                self.attr_tables.clear();
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
        self.camera_moved = true;
        self.pending_fit = false;
        self.auto_projection = false;
        self.basemap = saved.basemap;
        self.basemap_opacity = saved.basemap_opacity.unwrap_or(1.0).clamp(0.0, 1.0);
        self.last_basemap = saved.basemap.unwrap_or(self.last_basemap);
        self.show_graticule = saved.show_graticule;
        if let Some(px) = saved.box_threshold_px {
            self.box_threshold_px = px;
        }
        if let Some(mb) = saved.refine_budget_mb {
            self.refine_budget_mb = mb;
        }
        self.show_coastline = saved.show_coastline;
        // Loads prune against the restored viewport once the map panel has
        // recomputed it; seed with the whole world until then.
        // Tables first: they are read synchronously, so a query typed
        // straight after a restore finds them already there.
        for t in saved.tables {
            self.open_attr_table_named(t.source.into_source(), t.name);
        }
        for layer in saved.layers {
            let job = self.enqueue_load(layer.source.into_source(), ctx);
            self.pending_styles.insert(job, layer.style.into_style());
            if let Some(f) = layer.filter {
                self.pending_filters.insert(job, f);
            }
            if let Some(n) = layer.name {
                self.pending_names.insert(job, n);
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
            country_themes: None,
            cache_age: None,
            add: (String::new(), String::new()),
            generation: 0,
        });
        self.repo_refetch(ctx, false);
    }

    fn open_catalog_browser(&mut self) {
        self.catalog_browser = Some(CatalogBrowser {
            saved: crate::data::repo::load_catalogs(),
            session: self.session_catalogs.clone(),
            sel: None,
            add_url: String::new(),
            dcat: None,
            dcat_checked: Default::default(),
            filter: String::new(),
            geo_only: false,
            generation: 0,
        });
    }

    /// Show the catalog browser on a portal, adding it for the session if
    /// it is new. Saving it for good stays the user's call, made in the
    /// dialog once the catalog has answered.
    ///
    /// A `/data.json` is a catalog, not a file, so it never reaches
    /// `route_uri`: routing it would hand a JSON document to the parquet
    /// reader.
    fn open_portal(&mut self, url: &str, ctx: &egui::Context) {
        let trimmed = url.trim().trim_end_matches('/');
        let base = trimmed.strip_suffix("/data.json").unwrap_or(trimmed).trim_end_matches('/');
        if self.catalog_browser.is_none() {
            self.open_catalog_browser();
        }
        let b = self.catalog_browser.as_mut().unwrap();
        b.sel = if let Some(i) = b.saved.iter().position(|c| c.url == base) {
            Some((true, i))
        } else if let Some(i) = b.session.iter().position(|c| c.url == base) {
            Some((false, i))
        } else {
            b.session.push(crate::data::repo::Catalog {
                // Provisional: the catalog's own title replaces it when
                // the fetch lands.
                name: base
                    .trim_start_matches("https://")
                    .trim_start_matches("http://")
                    .to_string(),
                url: base.to_string(),
                added_on: None,
            });
            Some((false, b.session.len() - 1))
        };
        self.catalog_fetch(ctx);
    }

    /// Fetch the selected portal's catalog, dropping any stale in-flight
    /// result. One live document per portal: nothing here is cached.
    fn catalog_fetch(&mut self, ctx: &egui::Context) {
        let Some(b) = &mut self.catalog_browser else { return };
        b.generation += 1;
        b.dcat = None;
        b.dcat_checked.clear();
        let Some(url) = b.selected_catalog().map(|c| c.url.clone()) else { return };
        let generation = b.generation;
        let tx = self.dcat_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send((generation, crate::data::repo::fetch_dcat(&url)));
            ctx.request_repaint();
        });
    }

    fn poll_catalog(&mut self) {
        while let Ok((g, res)) = self.dcat_rx.try_recv() {
            let Some(b) = &mut self.catalog_browser else { continue };
            if g != b.generation {
                continue; // stale
            }
            // The catalog knows its own name; a session entry adopts it.
            // A saved entry keeps the name it was saved under.
            if let (Ok(cat), Some((false, i))) = (&res, b.sel) {
                if let (Some(t), Some(c)) = (&cat.title, b.session.get_mut(i)) {
                    c.name = t.clone();
                }
            }
            b.dcat = Some(res);
            b.dcat_checked.clear();
        }
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
        let kind = b.repos[b.sel_repo].kind;
        let snapshot = b.snapshots[b.sel_snapshot].path.clone();
        let tx = self.repo_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            use crate::data::repo::{self, RepoKind};
            let _ = tx.send(RepoMsg::Snapshots(
                generation,
                match kind {
                    RepoKind::Parquetry => repo::fetch_snapshots(&base),
                    RepoKind::Stac => repo::fetch_snapshots_stac(&base),
                },
            ));
            if force {
                repo::clear_cached_datasets(&base, &snapshot);
                if kind == RepoKind::Stac {
                    repo::clear_cached_stac_parts(&base);
                }
            } else if let Some((ds, at)) = repo::cached_datasets(&base, &snapshot) {
                let _ = tx.send(RepoMsg::Datasets(generation, Ok(ds), Some(at)));
                ctx.request_repaint();
                return;
            }
            let res = match kind {
                RepoKind::Parquetry => repo::discover_datasets(&base, &snapshot),
                RepoKind::Stac => repo::discover_datasets_stac(&base, &snapshot),
            };
            if let Ok(ds) = &res {
                repo::store_datasets(&base, &snapshot, ds);
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
        let kind = b.repos[b.sel_repo].kind;
        let snapshot = b.snapshots[b.sel_snapshot].path.clone();
        let Some(Ok(ds)) = &b.datasets else { return };
        let path = ds[ds_idx].path.clone();
        let tx = self.repo_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            use crate::data::repo::{self, RepoKind};
            let _ = tx.send(RepoMsg::Manifest(
                generation,
                ds_idx,
                match kind {
                    RepoKind::Parquetry => repo::fetch_manifest(&base, &snapshot, &path),
                    RepoKind::Stac => repo::fetch_stac_manifest(&base, &snapshot, &path),
                },
            ));
            ctx.request_repaint();
        });
    }

    /// Fetch every dataset manifest of `country` in parallel and union
    /// the themes, for the country-wide panel (parquetry repos only).
    fn repo_fetch_country(&mut self, country: String, ctx: &egui::Context) {
        let Some(b) = &mut self.repo_browser else { return };
        b.country_themes = Some((country.clone(), None));
        b.checked.clear();
        let generation = b.generation;
        let base = b.repos[b.sel_repo].url.trim_end_matches('/').to_string();
        let snapshot = b.snapshots[b.sel_snapshot].path.clone();
        let Some(Ok(ds)) = &b.datasets else { return };
        let seg = format!("country={country}");
        let paths: Vec<String> = ds
            .iter()
            .filter(|d| d.path.split('/').any(|s| s == seg))
            .map(|d| d.path.clone())
            .collect();
        let tx = self.repo_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            use rayon::prelude::*;
            let manifests: Vec<(String, crate::data::repo::Manifest)> = paths
                .par_iter()
                .filter_map(|p| {
                    match crate::data::repo::fetch_manifest(&base, &snapshot, p) {
                        Ok(m) => Some((p.clone(), m)),
                        Err(e) => {
                            log::warn!("{p}: manifest fetch failed: {e}");
                            None
                        }
                    }
                })
                .collect();
            let res = if manifests.is_empty() {
                Err(format!("no manifests could be read for country={country}"))
            } else {
                let mut by_theme: std::collections::BTreeMap<String, CountryTheme> =
                    Default::default();
                for (path, m) in &manifests {
                    for (theme, count) in &m.themes {
                        let e = by_theme.entry(theme.clone()).or_insert_with(|| {
                            CountryTheme {
                                theme: theme.clone(),
                                features: 0,
                                paths: Vec::new(),
                            }
                        });
                        e.features += *count;
                        e.paths.push(path.clone());
                    }
                }
                Ok(by_theme.into_values().collect())
            };
            let _ = tx.send(RepoMsg::CountryThemes(generation, country, res));
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
                RepoMsg::Manifest(g, ds_idx, res) if g == b.generation => {
                    if let Some((sel, m)) = &mut b.selected {
                        if *sel == ds_idx {
                            *m = Some(res);
                        }
                    }
                }
                RepoMsg::CountryThemes(g, country, res) if g == b.generation => {
                    if let Some((c, slot)) = &mut b.country_themes
                        && *c == country
                    {
                        *slot = Some(res);
                    }
                }
                _ => {} // stale generation
            }
        }
    }

    fn poll_downloads(&mut self, ctx: &egui::Context) {
        let mut ready: Vec<PathBuf> = Vec::new();
        while let Ok(msg) = self.dl_rx.try_recv() {
            match msg {
                DlMsg::Progress(id, got, total) => {
                    if let Some(d) = self.downloads.iter_mut().find(|d| d.id == id) {
                        d.got = got;
                        d.total = total;
                    }
                }
                DlMsg::Done(id, path) => {
                    self.downloads.retain(|d| d.id != id);
                    ready.push(path);
                }
                DlMsg::Failed(id, e) => {
                    // A download the user stopped is not a failure to
                    // report back to them.
                    let cancelled = self.downloads.iter().any(|d| {
                        d.id == id && d.cancel.load(std::sync::atomic::Ordering::Relaxed)
                    });
                    self.downloads.retain(|d| d.id != id);
                    if !cancelled {
                        self.push_error(e);
                    }
                }
            }
        }
        for p in ready {
            self.begin_import(p, ctx);
        }
    }

    /// Open one portal dataset through the path its format already has:
    /// parquet range-reads in place, a CSV goes to the import dialog over
    /// HTTPS, and the two filesystem-only formats download first and then
    /// enter `begin_import` exactly as a dropped file would.
    fn open_portal_dataset(&mut self, idx: usize, ctx: &egui::Context) {
        use crate::data::repo::{self, DcatFormat};
        use std::sync::atomic::{AtomicBool, Ordering};
        let Some(b) = &self.catalog_browser else { return };
        let Some(portal) = b.selected_catalog().map(|c| c.url.clone()) else { return };
        let Some(Ok(cat)) = &b.dcat else { return };
        let Some(ds) = cat.datasets.get(idx).cloned() else { return };
        let Some(dist) = ds.distributions.first().cloned() else { return };
        if !dist.format.needs_download() {
            match dist.format {
                DcatFormat::Csv => {
                    self.open_attr_table_named(
                        Source::Remote { url: dist.url, len: 0 },
                        ds.title.clone(),
                    );
                }
                _ => {
                    let job = self.enqueue_load(Source::Remote { url: dist.url, len: 0 }, ctx);
                    self.pending_names.insert(job, ds.title.clone());
                }
            }
            return;
        }
        let Some(dir) = repo::portal_dir(&portal, &ds.slug()) else {
            self.push_error("no config directory to download portal datasets into".into());
            return;
        };
        // Before the file opens, not after: the sidecar lookup memoizes
        // its misses per directory, so a notice written later is unread.
        if let Err(e) = repo::write_dcat_attribution(&dir, &ds, &portal) {
            log::warn!("{}: {e}", ds.title);
        }
        let dst = dir.join(format!("{}.{}", ds.slug(), dist.format.ext()));
        if dst.exists() {
            // Downloaded in an earlier session: the local copy is the
            // point of keeping it outside the temp directory.
            self.begin_import(dst, ctx);
            return;
        }
        let id = self.dl_next;
        self.dl_next += 1;
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        self.downloads.push(Download {
            id,
            label: ds.title.clone(),
            got: 0,
            total: None,
            cancel: cancel.clone(),
        });
        let tx = self.dl_tx.clone();
        let (ctx_prog, ctx_done) = (ctx.clone(), ctx.clone());
        let url = dist.url.clone();
        std::thread::spawn(move || {
            let tx_prog = tx.clone();
            let progress = move |got: u64, total: Option<u64>| {
                let _ = tx_prog.send(DlMsg::Progress(id, got, total));
                ctx_prog.request_repaint();
                !cancel.load(Ordering::Relaxed)
            };
            let res = repo::download_to(&url, &dst, &progress);
            let _ = tx.send(match res {
                Ok(()) => DlMsg::Done(id, dst),
                Err(e) => DlMsg::Failed(id, e),
            });
            ctx_done.request_repaint();
        });
    }

    fn repo_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
        if self.repo_browser.is_none() {
            return;
        }
        let mut open = true;
        let mut refetch = false;
        let mut force_refetch = false;
        let mut fetch_manifest: Option<usize> = None;
        let mut fetch_country: Option<String> = None;
        let mut load: Vec<(Source, String)> = Vec::new(); // (source, layer name)

        {
            let b = self.repo_browser.as_mut().unwrap();
            let kind = b.repos[b.sel_repo].kind;
            egui::Window::new("GeoParquet repositories")
                .id(egui::Id::new("repo_browser"))
                .open(&mut open)
                .default_width(560.0)
                .constrain_to(floating_area).show(ctx, |ui| {
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
                            .button(ph::ARROWS_CLOCKWISE)
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
                                        let country_before = b.country.clone();
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
                                        if b.country != country_before {
                                            // Country-wide themes replace any
                                            // state selection until a state
                                            // is clicked.
                                            b.selected = None;
                                            b.checked.clear();
                                            if b.country.is_empty() {
                                                b.country_themes = None;
                                            } else {
                                                fetch_country =
                                                    Some(b.country.clone());
                                            }
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
                                                let row = if code == name {
                                                    name.clone()
                                                } else {
                                                    format!("{name} ({code})")
                                                };
                                                if ui
                                                    .selectable_label(
                                                        sel_idx == Some(*i),
                                                        row,
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
                                                            let n = *count as usize;
                                                            ui.label(
                                                                RichText::new(match kind {
                                                                    crate::data::repo::RepoKind::Stac => {
                                                                        format!(
                                                                            "{n} part{}",
                                                                            if n == 1 { "" } else { "s" }
                                                                        )
                                                                    }
                                                                    _ => fmt_count(n),
                                                                })
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
                                                    if !b.checked.contains(theme) {
                                                        continue;
                                                    }
                                                    use crate::data::repo::{self, RepoKind};
                                                    load.push(match kind {
                                                        RepoKind::Parquetry => (
                                                            Source::Remote {
                                                                url: repo::theme_url(
                                                                    &base, &snap, path, theme,
                                                                ),
                                                                len: 0,
                                                            },
                                                            if code.is_empty() {
                                                                theme.clone()
                                                            } else {
                                                                format!("{code} {theme}")
                                                            },
                                                        ),
                                                        RepoKind::Stac => (
                                                            Source::Stac {
                                                                url: repo::stac_collection_url(
                                                                    &base, &snap, path, theme,
                                                                ),
                                                                name: theme.clone(),
                                                            },
                                                            theme.clone(),
                                                        ),
                                                    });
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
                                        if kind == crate::data::repo::RepoKind::Stac {
                                            ui.label(
                                                RichText::new(format!(
                                                    "opens the {} parts covering most of the \
                                                     current view; the rest load as you pan",
                                                    crate::data::loader::STAC_PART_CAP
                                                ))
                                                .weak()
                                                .small(),
                                            );
                                        }
                                    }
                                }
                            }
                            None => {
                                let ct_view = b.country_themes.clone();
                                match &ct_view {
                                    Some((c, slot))
                                        if kind == crate::data::repo::RepoKind::Parquetry
                                            && !b.country.is_empty()
                                            && *c == b.country =>
                                    {
                                        ui.label(
                                            RichText::new(format!("All of country={c}"))
                                                .strong(),
                                        );
                                        match slot {
                                            None => {
                                                ui.horizontal(|ui| {
                                                    ui.spinner();
                                                    ui.label(
                                                        RichText::new(
                                                            "reading state manifests…",
                                                        )
                                                        .weak(),
                                                    );
                                                });
                                            }
                                            Some(Err(e)) => {
                                                ui.label(
                                                    RichText::new(e).color(
                                                        Color32::from_rgb(220, 60, 60),
                                                    ),
                                                );
                                            }
                                            Some(Ok(themes)) => {
                                                let base = b.repos[b.sel_repo]
                                                    .url
                                                    .trim_end_matches('/')
                                                    .to_string();
                                                let snap = b.snapshots[b.sel_snapshot]
                                                    .path
                                                    .clone();
                                                egui::ScrollArea::vertical()
                                                    .id_salt("repo_country_themes")
                                                    .max_height(300.0)
                                                    .show(ui, |ui| {
                                                        egui::Grid::new("repo_country_grid")
                                                            .num_columns(2)
                                                            .striped(true)
                                                            .show(ui, |ui| {
                                                                for t in themes {
                                                                    // Country-wide totals
                                                                    // beyond the display
                                                                    // budget would load as
                                                                    // a huge, disappointing
                                                                    // preview: gate them to
                                                                    // state-level loads.
                                                                    let too_big = t.features
                                                                        > crate::data::loader::MAX_BUILD_ROWS;
                                                                    let mut on = b
                                                                        .checked
                                                                        .contains(&t.theme);
                                                                    if ui
                                                                        .add_enabled(
                                                                            !too_big,
                                                                            egui::Checkbox::new(
                                                                                &mut on,
                                                                                &t.theme,
                                                                            ),
                                                                        )
                                                                        .on_disabled_hover_text(
                                                                            format!(
                                                                            "{} features — too \
                                                                             large to display \
                                                                             country-wide; pick \
                                                                             a state on the left \
                                                                             instead",
                                                                            fmt_count(
                                                                                t.features
                                                                                    as usize
                                                                            )
                                                                        ),
                                                                        )
                                                                        .changed()
                                                                    {
                                                                        if on {
                                                                            b.checked.insert(
                                                                                t.theme
                                                                                    .clone(),
                                                                            );
                                                                        } else {
                                                                            b.checked
                                                                                .remove(
                                                                                    &t.theme,
                                                                                );
                                                                        }
                                                                    }
                                                                    ui.label(
                                                                        RichText::new(
                                                                            format!(
                                                                            "{} features · {} states",
                                                                            fmt_count(
                                                                                t.features
                                                                                    as usize
                                                                            ),
                                                                            t.paths.len()
                                                                        ),
                                                                        )
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
                                                        for t in themes {
                                                            if !b.checked.contains(&t.theme)
                                                                || t.features
                                                                    > crate::data::loader::MAX_BUILD_ROWS
                                                            {
                                                                continue;
                                                            }
                                                            let name = format!(
                                                                "{c} {} (all)",
                                                                t.theme
                                                            );
                                                            load.push((
                                                                Source::Multi {
                                                                    name: name.clone(),
                                                                    urls: t
                                                                        .paths
                                                                        .iter()
                                                                        .map(|p| {
                                                                            crate::data::repo::theme_url(
                                                                                &base, &snap,
                                                                                p, &t.theme,
                                                                            )
                                                                        })
                                                                        .collect(),
                                                                },
                                                                name,
                                                            ));
                                                        }
                                                        b.checked.clear();
                                                    }
                                                    if ui.small_button("all").clicked() {
                                                        b.checked = themes
                                                            .iter()
                                                            .filter(|t| {
                                                                t.features
                                                                    <= crate::data::loader::MAX_BUILD_ROWS
                                                            })
                                                            .map(|t| t.theme.clone())
                                                            .collect();
                                                    }
                                                    if ui.small_button("none").clicked() {
                                                        b.checked.clear();
                                                    }
                                                });
                                                ui.label(
                                                    RichText::new(format!(
                                                        "each theme loads as one layer \
                                                         across all its states, with \
                                                         state as a column; themes over \
                                                         {} features are state-level \
                                                         only",
                                                        fmt_count(
                                                            crate::data::loader::MAX_BUILD_ROWS
                                                                as usize
                                                        )
                                                    ))
                                                    .weak()
                                                    .small(),
                                                );
                                            }
                                        }
                                    }
                                    _ => {
                                        ui.label(
                                            RichText::new(
                                                "select a dataset to list its layers — \
                                                 or pick a country to load themes \
                                                 across all its states",
                                            )
                                            .weak(),
                                        );
                                    }
                                }
                            }
                        });
                    });

                    ui.separator();
                    add_repo_row(ui, b, &mut refetch);
                });
        }

        if refetch || force_refetch {
            self.repo_refetch(ctx, force_refetch);
        }
        if let Some(c) = fetch_country {
            self.repo_fetch_country(c, ctx);
        }
        if let Some(i) = fetch_manifest {
            self.repo_fetch_manifest(i, ctx);
        }
        for (source, name) in load {
            let job = self.enqueue_load(source, ctx);
            self.pending_names.insert(job, name);
        }
        if !open {
            self.repo_browser = None;
        }
    }

    /// The open-data catalog dialog: saved catalogs with their added-on
    /// date, this session's catalogs with the offer to save them for
    /// good, an add row, and the dataset pane of the selected portal.
    fn catalog_window(&mut self, ctx: &egui::Context) {
        use crate::data::repo;
        let floating_area = self.floating_area(ctx);
        if self.catalog_browser.is_none() {
            return;
        }
        let mut open = true;
        let mut fetch = false;
        let mut add_url: Option<String> = None;
        let mut open_datasets: Vec<usize> = Vec::new(); // catalog indices

        {
            let b = self.catalog_browser.as_mut().unwrap();
            egui::Window::new("Open-data catalogs")
                .id(egui::Id::new("catalog_browser"))
                .open(&mut open)
                .default_width(560.0)
                // Bounded, not auto-sized: with fifteen catalogs above a
                // dataset list, auto-height runs past the screen and the
                // Open button ends up under the window border. Bounding
                // the window is what makes the panes' available_height
                // real, so they can share what actually fits.
                .max_height((floating_area.height() - 24.0).max(240.0))
                .constrain_to(floating_area)
                .show(ctx, |ui| {
                    if b.saved.is_empty() && b.session.is_empty() {
                        ui.label(
                            RichText::new(
                                "No catalogs yet. Paste an open-data portal below — \
                                 ArcGIS Hub, Socrata and CKAN sites all publish a \
                                 DCAT catalog at /data.json.",
                            )
                            .weak(),
                        );
                    }
                    // Row actions, applied after the loops: the lists
                    // cannot be edited while they are being drawn.
                    let mut select: Option<(bool, usize)> = None;
                    let mut forget: Option<(bool, usize)> = None; // (saved list?, index)
                    let mut keep: Option<usize> = None; // session index
                    // The catalog rows scroll in their own strip so a
                    // long list leaves the dataset pane its share.
                    egui::ScrollArea::vertical()
                        .id_salt("catalog_rows")
                        .max_height(if b.sel.is_some() { 160.0 } else { 400.0 })
                        .show(ui, |ui| {
                    // Cities and user-added portals first, then the
                    // state section under its own heading; each keeps
                    // the list's alphabetical order.
                    let is_state: Vec<bool> = b
                        .saved
                        .iter()
                        .map(|c| repo::is_default_state(&c.url))
                        .collect();
                    let order: Vec<usize> = (0..b.saved.len())
                        .filter(|&i| !is_state[i])
                        .chain((0..b.saved.len()).filter(|&i| is_state[i]))
                        .collect();
                    let mut states_started = false;
                    for i in order {
                        if is_state[i] && !states_started {
                            states_started = true;
                            ui.add_space(4.0);
                            ui.label(RichText::new("US states").weak().small());
                        }
                        let c = &b.saved[i];
                        ui.horizontal(|ui| {
                            let on = b.sel == Some((true, i));
                            if ui.selectable_label(on, &c.name).on_hover_text(&c.url).clicked()
                                && !on
                            {
                                select = Some((true, i));
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button(ph::TRASH)
                                        .on_hover_text("Forget this catalog")
                                        .clicked()
                                    {
                                        forget = Some((true, i));
                                    }
                                    let when = match c.added_on {
                                        Some(at) => format!("added {}", repo::date_label(at)),
                                        None => "built-in".into(),
                                    };
                                    ui.label(RichText::new(when).weak().small());
                                },
                            );
                        });
                    }
                    for (i, c) in b.session.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let on = b.sel == Some((false, i));
                            if ui.selectable_label(on, &c.name).on_hover_text(&c.url).clicked()
                                && !on
                            {
                                select = Some((false, i));
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button(ph::TRASH)
                                        .on_hover_text("Drop this session entry")
                                        .clicked()
                                    {
                                        forget = Some((false, i));
                                    }
                                    if ui
                                        .small_button("Save")
                                        .on_hover_text(
                                            "Keep this catalog for future sessions",
                                        )
                                        .clicked()
                                    {
                                        keep = Some(i);
                                    }
                                    ui.label(
                                        RichText::new("this session only").weak().small(),
                                    );
                                },
                            );
                        });
                    }
                    });
                    if let Some((in_saved, i)) = forget {
                        if in_saved {
                            b.saved.remove(i);
                            if let Err(e) = repo::save_catalogs(&b.saved) {
                                log::warn!("saving catalogs: {e}");
                            }
                        } else {
                            b.session.remove(i);
                        }
                        match b.sel {
                            Some((s, j)) if s == in_saved && j == i => {
                                b.sel = None;
                                b.dcat = None;
                            }
                            Some((s, j)) if s == in_saved && j > i => {
                                b.sel = Some((s, j - 1));
                            }
                            _ => {}
                        }
                    }
                    if let Some(i) = keep {
                        b.save_for_good(i);
                    }
                    if let Some(s) = select {
                        b.sel = Some(s);
                        b.filter.clear();
                        fetch = true;
                    }
                    // --- add row ---
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut b.add_url)
                                .hint_text("https://data.city.gov · https://data.city.gov/data.json")
                                .desired_width(300.0),
                        );
                        let url = b.add_url.trim().to_string();
                        let http = url.starts_with("https://") || url.starts_with("http://");
                        if ui
                            .add_enabled(http, egui::Button::new("Add catalog"))
                            .on_hover_text(
                                "Added for this session; whether it is kept for good \
                                 is asked once its catalog answers. The portal names \
                                 itself from the catalog.",
                            )
                            .clicked()
                        {
                            add_url = Some(url);
                            b.add_url.clear();
                        }
                    });
                    // --- dataset pane of the selected catalog ---
                    if b.sel.is_some() {
                        ui.separator();
                        if let Some((false, i)) = b.sel
                            && matches!(b.dcat, Some(Ok(_)))
                        {
                            // The ask. Answering "no" costs nothing: a
                            // session entry simply dies with the app.
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(
                                        "This catalog is kept for this session only.",
                                    )
                                    .weak(),
                                );
                                if ui
                                    .button("Save it for good")
                                    .on_hover_text(
                                        "Remember this catalog across sessions, with \
                                         today as its added-on date",
                                    )
                                    .clicked()
                                {
                                    b.save_for_good(i);
                                }
                            });
                        }
                        dcat_pane(ui, b, &mut open_datasets);
                    }
                });
        }

        if let Some(url) = add_url {
            self.open_portal(&url, ctx);
        } else if fetch {
            self.catalog_fetch(ctx);
        }
        for i in open_datasets {
            self.open_portal_dataset(i, ctx);
        }
        if !open {
            // "For the session" means the app run, not the dialog's
            // lifetime: the session list survives the window closing.
            let b = self.catalog_browser.take().unwrap();
            self.session_catalogs = b.session;
        }
    }

    // ------------------------------------------------------------------
    // Data-driven styling dialog
    // ------------------------------------------------------------------

    /// Columns of a layer eligible for styling: (name, numeric).
    fn style_columns(store: &crate::data::store::FeatureStore) -> Vec<(String, bool)> {
        store.style_columns()
    }

    fn open_grid_dialog(&mut self, layer_id: u64) {
        let Some(l) = self.layers.iter().find(|l| l.id == layer_id) else { return };
        let all = Self::style_columns(&l.store);
        let text_cols: std::collections::HashSet<String> = all
            .iter()
            .filter(|(_, numeric)| !numeric)
            .map(|(n, _)| n.clone())
            .collect();
        let columns: Vec<String> = all.into_iter().map(|(n, _)| n).collect();
        if columns.is_empty() {
            self.push_error(format!("{}: no columns to aggregate", l.name));
            return;
        }
        // A numeric column first when there is one: the numeric
        // statistics are the common case.
        let column = columns
            .iter()
            .find(|c| !text_cols.contains(*c))
            .unwrap_or(&columns[0])
            .clone();
        self.grid_dialog = Some(GridState {
            layer_id,
            layer_name: l.name.clone(),
            columns,
            text_cols,
            column,
            system: 0,
            size: 1000.0,
            h3_res: 7,
            a5_res: 14,
            stat: crate::data::grid::GridStat::Mean,
            kernel: crate::data::grid::Kernel::Box,
            passes: 0,
            post: 0,
            azimuth: 315.0,
            altitude: 45.0,
            contours: false,
            levels: 10,
            quantile_levels: true,
            running: false,
            progress: 0.0,
            error: None,
            rx: None,
        });
    }

    fn grid_window(&mut self, ctx: &egui::Context) {
        use crate::data::grid::{CellSystem, GridOutput, GridStat, Kernel};
        let floating_area = self.floating_area(ctx);
        let Some(st) = self.grid_dialog.as_ref() else {
            return;
        };
        // The dialog outliving its layer would leave "Compute grid"
        // silently inert, as the style and filter dialogs already guard
        // against. A computation in flight owns clones and finishes.
        if !st.running && !self.layers.iter().any(|l| l.id == st.layer_id) {
            self.grid_dialog = None;
            return;
        }
        // Drain progress / completion messages.
        let mut done: Option<(PathBuf, String, usize, u64)> = None;
        {
            let st = self.grid_dialog.as_mut().unwrap();
            let mut failed = None;
            if let Some(rx) = &st.rx {
                loop {
                    match rx.try_recv() {
                        Ok(GridMsg::Progress(f)) => st.progress = f,
                        Ok(GridMsg::Done(p, name, cells, rows)) => {
                            done = Some((p, name, cells, rows))
                        }
                        Ok(GridMsg::Failed(e)) => failed = Some(e),
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            // The worker died without reporting (a panic
                            // in compute): say so rather than leaving the
                            // form disabled behind a frozen progress bar.
                            if done.is_none() && failed.is_none() && st.running {
                                failed = Some("grid computation stopped unexpectedly".into());
                            }
                            break;
                        }
                    }
                }
            }
            if let Some(e) = failed {
                st.error = Some(e);
                st.running = false;
                st.rx = None;
            }
        }
        if let Some((path, name, cells, rows)) = done {
            self.grid_dialog = None;
            self.temp_outputs.push(path.clone());
            let job = self.enqueue_load(Source::Local(path), ctx);
            self.pending_names.insert(job, name);
            log::info!("grid: {cells} cells from {rows} features");
            return;
        }

        let mut open = true;
        let mut start = false;
        let st = self.grid_dialog.as_mut().unwrap();
        egui::Window::new(format!("Grid summary — {}", st.layer_name))
            .id(egui::Id::new("grid_dialog"))
            .open(&mut open)
            .default_width(380.0)
            .collapsible(false)
            .constrain_to(floating_area)
            .show(ctx, |ui| {
                ui.add_enabled_ui(!st.running, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("column:");
                        egui::ComboBox::from_id_salt("grid_col")
                            .width(180.0)
                            .selected_text(&st.column)
                            .show_ui(ui, |ui| {
                                for c in &st.columns {
                                    let label = if st.text_cols.contains(c) {
                                        format!("{c} (text)")
                                    } else {
                                        c.clone()
                                    };
                                    ui.selectable_value(&mut st.column, c.clone(), label);
                                }
                            });
                        // A text column takes the categorical statistics
                        // and nothing numeric; switching back restores a
                        // numeric default. The surface controls below
                        // (smoothing, focal, contours) have no meaning
                        // over categories and are cleared with it.
                        let is_text = st.text_cols.contains(&st.column);
                        if is_text && !st.stat.text() {
                            st.stat = GridStat::Majority;
                            st.passes = 0;
                            st.post = 0;
                            st.contours = false;
                        } else if !is_text && st.stat.text() {
                            st.stat = GridStat::Mean;
                        }
                        ui.label("statistic:");
                        egui::ComboBox::from_id_salt("grid_stat")
                            .width(90.0)
                            .selected_text(st.stat.label())
                            .show_ui(ui, |ui| {
                                if is_text {
                                    ui.selectable_value(
                                        &mut st.stat,
                                        GridStat::Majority,
                                        "majority",
                                    )
                                    .on_hover_text(
                                        "The value that dominates the cell — points \
                                         count 1 apiece, polygons their covered-area \
                                         share",
                                    );
                                    ui.selectable_value(
                                        &mut st.stat,
                                        GridStat::Minority,
                                        "minority",
                                    )
                                    .on_hover_text(
                                        "The rarest value present in the cell — the \
                                         one oak in the pine stand",
                                    );
                                    return;
                                }
                                ui.label(
                                    RichText::new("rates (per-m² prices, years, ratios)")
                                        .weak()
                                        .small(),
                                );
                                for s in [GridStat::Mean, GridStat::Median] {
                                    ui.selectable_value(&mut st.stat, s, s.label());
                                }
                                ui.separator();
                                ui.label(
                                    RichText::new("totals (values, populations, counts)")
                                        .weak()
                                        .small(),
                                );
                                for s in [GridStat::Sum, GridStat::Count, GridStat::Density] {
                                    ui.selectable_value(&mut st.stat, s, s.label());
                                }
                            })
                            .response
                            .on_hover_text(if is_text {
                                "A text column: each cell gets its majority (or \
                                 minority) value, and the layer styles by category"
                            } else {
                                "Pick by column type: totals grow with polygon size \
                                 (use sum / count / density — a giant polygon's mean \
                                 equals its full value in every covered cell); rates \
                                 don't (use mean / median)"
                            });
                    });
                    let is_text = st.text_cols.contains(&st.column);
                    ui.horizontal(|ui| {
                        ui.label("cells:");
                        ui.selectable_value(&mut st.system, 0, "square");
                        ui.selectable_value(&mut st.system, 1, "H3");
                        ui.selectable_value(&mut st.system, 2, "A5");
                        if st.system != 0 && st.post == 4 {
                            // Hillshade and contours are square-only; drop
                            // them rather than leave a dead selection on
                            // screen that would fail at compute time.
                            st.post = 0;
                        }
                        match st.system {
                            0 => {
                                ui.label("size:");
                                ui.add(
                                    egui::DragValue::new(&mut st.size)
                                        .range(1.0..=1e7)
                                        .speed(50.0),
                                )
                                .on_hover_text(
                                    "Cell size in the layer's CRS units (meters for \
                                     projected data — avoid degrees, \
                                     st_transform first)",
                                );
                            }
                            1 => {
                                ui.label("res:");
                                ui.add(egui::DragValue::new(&mut st.h3_res).range(0..=13))
                                    .on_hover_text(
                                        "H3 resolution (7 ≈ 5 km², 8 ≈ 0.7 km², 9 ≈ 0.1 km²)",
                                    );
                            }
                            _ => {
                                ui.label("res:");
                                ui.add(egui::DragValue::new(&mut st.a5_res).range(0..=20))
                                    .on_hover_text(
                                        "A5 resolution (equal-area pentagons; \
                                         ~14 is comparable to H3 7)",
                                    );
                            }
                        }
                    });
                    ui.add_enabled_ui(!is_text, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("smoothing:");
                        ui.add(egui::DragValue::new(&mut st.passes).range(0..=5))
                            .on_hover_text(
                                "Passes of neighbor averaging over present \
                                 cells (0 = raw aggregation)",
                            );
                        if st.system == 0 {
                            egui::ComboBox::from_id_salt("grid_kernel")
                                .width(90.0)
                                .selected_text(match st.kernel {
                                    Kernel::Box => "box 3×3",
                                    Kernel::Gaussian => "gaussian",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut st.kernel, Kernel::Box, "box 3×3");
                                    ui.selectable_value(
                                        &mut st.kernel,
                                        Kernel::Gaussian,
                                        "gaussian",
                                    );
                                });
                        } else {
                            ui.label(RichText::new("(ring mean)").weak().small());
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("focal:");
                        const POST: [&str; 5] =
                            ["none", "focal std", "open", "close", "hillshade"];
                        egui::ComboBox::from_id_salt("grid_post")
                            .width(110.0)
                            .selected_text(POST[st.post.min(4)])
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut st.post, 0, POST[0]);
                                ui.selectable_value(&mut st.post, 1, POST[1]).on_hover_text(
                                    "Standard deviation of each cell's neighborhood: \
                                     where values are mixed rather than where they \
                                     are high",
                                );
                                ui.selectable_value(&mut st.post, 2, POST[2]).on_hover_text(
                                    "Erode then dilate: drops isolated hot cells, \
                                     leaves the rest of the surface in place",
                                );
                                ui.selectable_value(&mut st.post, 3, POST[3]).on_hover_text(
                                    "Dilate then erode: fills isolated holes without \
                                     inflating the level around them",
                                );
                                ui.add_enabled_ui(st.system == 0, |ui| {
                                    ui.selectable_value(&mut st.post, 4, POST[4]).on_hover_text(
                                        if st.system == 0 {
                                            "Shaded relief of the value surface, 0-255. \
                                             Exaggeration is fitted to the data, so any \
                                             unit reads; style it with a grey ramp"
                                        } else {
                                            "Square grids only — the gradient needs the \
                                             regular lattice"
                                        },
                                    );
                                });
                            });
                        if st.post == 4 && st.system == 0 {
                            ui.label("sun:");
                            ui.add(
                                egui::DragValue::new(&mut st.azimuth)
                                    .range(0.0..=360.0)
                                    .suffix("°az"),
                            )
                            .on_hover_text("Azimuth the light comes from, clockwise from north");
                            ui.add(
                                egui::DragValue::new(&mut st.altitude)
                                    .range(1.0..=89.0)
                                    .suffix("°alt"),
                            )
                            .on_hover_text("Sun height above the horizon");
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_enabled_ui(st.system == 0, |ui| {
                            ui.checkbox(&mut st.contours, "contour lines")
                                .on_hover_text(if st.system == 0 {
                                    "Marching squares over the (smoothed) cell values: \
                                     the output layer is isolines with the field's value \
                                     per line instead of cell polygons"
                                } else {
                                    "Square grids only — marching squares needs the \
                                     regular lattice"
                                });
                        });
                        if st.contours && st.system == 0 {
                            ui.label("levels:");
                            ui.add(egui::DragValue::new(&mut st.levels).range(1..=64));
                            egui::ComboBox::from_id_salt("grid_spacing")
                                .width(96.0)
                                .selected_text(if st.quantile_levels {
                                    "quantile"
                                } else {
                                    "equal"
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut st.quantile_levels,
                                        false,
                                        "equal",
                                    )
                                    .on_hover_text(
                                        "Equal steps between min and max",
                                    );
                                    ui.selectable_value(
                                        &mut st.quantile_levels,
                                        true,
                                        "quantile",
                                    )
                                    .on_hover_text(
                                        "Equal counts of cells per band — the \
                                         readable choice on skewed surfaces, \
                                         where equal steps push every line into \
                                         the outlier tail",
                                    );
                                });
                        }
                    });
                    });
                    ui.label(
                        RichText::new(
                            "Whole-file scan; features spanning several cells are apportioned by covered area. Columnar fast path when a covering bbox column exists",
                        )
                        .weak()
                        .small(),
                    );
                });
                if let Some(e) = &st.error {
                    ui.colored_label(Color32::from_rgb(220, 60, 60), e);
                }
                ui.add_space(4.0);
                if st.running {
                    ui.add(egui::ProgressBar::new(st.progress).show_percentage());
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                } else if ui.button("Compute grid").clicked() {
                    start = true;
                }
            });
        if start {
            let grid_n = self.next_grid_n();
            let st = self.grid_dialog.as_mut().unwrap();
            let Some(l) = self.layers.iter().find(|l| l.id == st.layer_id) else { return };
            let Some(col) = l
                .store
                .schema
                .fields()
                .iter()
                .position(|f| f.name() == &st.column)
            else {
                st.error = Some(format!("column {} not found", st.column));
                return;
            };
            let system = match st.system {
                0 => CellSystem::Square { size: st.size },
                1 => CellSystem::H3 { res: st.h3_res },
                _ => CellSystem::A5 { res: st.a5_res },
            };
            let output = if st.contours && st.system == 0 {
                GridOutput::Contours {
                    levels: st.levels as usize,
                    spacing: if st.quantile_levels {
                        crate::data::grid::ContourSpacing::Quantile
                    } else {
                        crate::data::grid::ContourSpacing::Equal
                    },
                }
            } else {
                GridOutput::Cells
            };
            let spec = crate::data::grid::GridSpec {
                value_col: col,
                value_name: st.column.clone(),
                system,
                stat: st.stat,
                kernel: st.kernel,
                output,
                smooth_passes: st.passes,
                post: match st.post {
                    1 => crate::data::grid::PostOp::FocalStd,
                    2 => crate::data::grid::PostOp::Open,
                    3 => crate::data::grid::PostOp::Close,
                    4 => crate::data::grid::PostOp::Hillshade {
                        azimuth_deg: st.azimuth,
                        altitude_deg: st.altitude,
                    },
                    _ => crate::data::grid::PostOp::None,
                },
            };
            let cell_label = match st.system {
                0 if st.contours => format!("{}u contours", st.size),
                0 => format!("{}u", st.size),
                1 => format!("h3r{}", st.h3_res),
                _ => format!("a5r{}", st.a5_res),
            };
            let post_label = match st.post {
                1 => " std",
                2 => " open",
                3 => " close",
                4 if st.system == 0 => " hillshade",
                _ => "",
            };
            let name = format!(
                "{} · {}({}) {}{}",
                st.layer_name,
                st.stat.label(),
                st.column,
                cell_label,
                post_label
            );
            let dst = std::env::temp_dir().join(format!(
                "geopq_grid_{}_{}_{grid_n}.parquet",
                std::process::id(),
                st.layer_id,
            ));
            let (tx, rx) = std::sync::mpsc::channel();
            st.rx = Some(rx);
            st.running = true;
            st.progress = 0.0;
            st.error = None;
            let store = Arc::clone(&l.store);
            let crs = l.crs.clone();
            let ctx2 = ctx.clone();
            std::thread::spawn(move || {
                let tx_prog = tx.clone();
                let ctx3 = ctx2.clone();
                let prog = move |f: f32| {
                    let _ = tx_prog.send(GridMsg::Progress(f));
                    ctx3.request_repaint();
                };
                let res = crate::data::grid::compute(&store, &crs, &spec, &dst, &prog);
                let _ = tx.send(match res {
                    Ok((cells, rows)) => GridMsg::Done(dst, name, cells, rows),
                    Err(e) => GridMsg::Failed(e),
                });
                ctx2.request_repaint();
            });
        }
        if !open && !self.grid_dialog.as_ref().is_some_and(|s| s.running) {
            self.grid_dialog = None;
        }
    }

    /// Monotonic counter for grid temp files.
    fn next_grid_n(&mut self) -> u64 {
        self.grid_n += 1;
        self.grid_n
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
        let per_area = l.style.style_by.as_ref().is_some_and(|sb| sb.per_area);
        let (column, ramp, method, classes) = match &l.style.style_by {
            Some(sb) => match &sb.mode {
                StyleMode::Graduated { method, breaks } => {
                    (sb.column.clone(), sb.ramp, *method, breaks.len() + 1)
                }
                StyleMode::Categorical { .. } => {
                    (sb.column.clone(), sb.ramp, ClassMethod::EqualInterval, 8)
                }
            },
            None => {
                let c = cols
                    .iter()
                    .find(|(_, num)| *num)
                    .unwrap_or(&cols[0])
                    .0
                    .clone();
                (c, Ramp::Viridis, ClassMethod::EqualInterval, 8)
            }
        };
        let width_px = l.style.style_by.as_ref().and_then(|sb| sb.width_px);
        let mut d = StyleDialog {
            layer_id,
            column,
            numeric: true,
            ramp,
            method,
            classes,
            per_area,
            min: 0.0,
            max: 1.0,
            breaks: None,
            categories: None,
            color_map: None,
            use_color_map: true,
            width_by: width_px.is_some(),
            width_min: width_px.map_or(0.6, |(a, _)| a),
            width_max: width_px.map_or(4.0, |(_, b)| b),
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
        d.color_map = None;
        d.use_color_map = true;
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
            if d.method.needs_values() || d.per_area {
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
                    let (id, col, method, classes, per_area) =
                        (d.layer_id, d.column.clone(), d.method, d.classes, d.per_area);
                    let latlong = l.crs.is_latlong;
                    let tx = self.class_tx.clone();
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        // Normalized values have no usable column stats:
                        // equal interval classifies from the sample too.
                        let cap = if per_area { 20_000 } else { 50_000 };
                        let res = crate::data::loader::sample_loaded_values(
                            &store, &loaded, idx, cap, per_area, latlong, None,
                        )
                        .map(|mut vals| {
                            crate::data::layer::classify_breaks(method, &mut vals, classes)
                        });
                        let _ = tx.send((id, col, res));
                        ctx.request_repaint();
                    });
                }
            } else {
                d.breaks = Some(Ok(crate::data::layer::bounds_breaks(
                    d.method, d.min, d.max, d.classes,
                )));
            }
        } else if let Some(map) = crate::data::colormap::match_column(&d.column) {
            // The map names the classes, so there is nothing to look up:
            // scanning millions of rows to rediscover a published
            // nomenclature would be work for its own sake.
            d.color_map = Some(map);
            d.use_color_map = true;
        }
        // No eager scan for text columns: the dialog asks for one only
        // when the generic palette is actually on screen.
    }

    fn poll_classes(&mut self) {
        while let Ok((layer_id, column, res)) = self.class_rx.try_recv() {
            if let Some(d) = &mut self.style_dialog {
                if d.layer_id == layer_id
                    && d.column == column
                    && d.numeric
                    && (d.method.needs_values() || d.per_area)
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
                    d.color_map = match &res {
                        Ok(v) => crate::data::colormap::match_builtin(v),
                        Err(_) => None,
                    };
                    d.categories = Some(res);
                }
            }
        }
    }

    /// Apply a QGIS colour-map file to the style dialog that asked for it,
    /// if it is still on screen.
    fn apply_color_map_file(&mut self, path: PathBuf) {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "colour map".into());
        let parsed = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| crate::data::colormap::parse_qgis(&t, &name));
        match parsed {
            Some(m) => {
                if let Some(d) = &mut self.style_dialog {
                    d.color_map = Some(m);
                    d.use_color_map = true;
                }
            }
            None => self.push_error(format!("{}: no colour classes found", path.display())),
        }
    }

    fn style_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
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
        // Set when the generic palette is on screen with no values yet:
        // the scan is started only for the path that actually needs it.
        let mut want_categories = false;
        // The colour-map picker is a native dialog, so it cannot open from
        // inside this frame (see `spawn_pick`); the click is recorded and
        // acted on once the window closure has released `self`.
        let mut want_color_map = false;
        let current = self.layers[layer_idx].style.style_by.clone();
        let is_poly = matches!(
            self.layers[layer_idx].kind(),
            crate::data::geometry::GeomKind::Polygon
        );

        let mut open = true;
        let mut reselect = false;
        let mut apply: Option<StyleBy> = None;
        {
            let d = self.style_dialog.as_mut().unwrap();
            egui::Window::new(format!("Style — {layer_name}"))
                .id(egui::Id::new("style_dialog"))
                .open(&mut open)
                .default_width(360.0)
                .constrain_to(floating_area).show(ctx, |ui| {
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
                            ui.label("method:");
                            let before = (d.method, d.classes);
                            egui::ComboBox::from_id_salt("style_method")
                                .selected_text(d.method.label())
                                .show_ui(ui, |ui| {
                                    for m in crate::data::layer::ClassMethod::ALL {
                                        ui.selectable_value(&mut d.method, *m, m.label());
                                    }
                                });
                            ui.label("classes:");
                            ui.add(
                                egui::DragValue::new(&mut d.classes)
                                    .range(2..=crate::data::layer::STYLE_BINS)
                                    .speed(0.1),
                            );
                            if (d.method, d.classes) != before {
                                reselect = true;
                            }
                        });
                        if is_poly {
                            let before = d.per_area;
                            ui.checkbox(&mut d.per_area, "normalize by area")
                                .on_hover_text(
                                    "Classify and color value / polygon area \
                                     (data-CRS units, e.g. $/m²) so large polygons \
                                     don't dominate the choropleth",
                                );
                            if d.per_area != before {
                                reselect = true;
                            }
                        }
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut d.width_by, "line width by class")
                                .on_hover_text(
                                    "Ramp the stroke width across the classes, \
                                     the way the colors already do",
                                );
                            if d.width_by {
                                ui.add(
                                    egui::DragValue::new(&mut d.width_min)
                                        .range(0.1..=20.0)
                                        .speed(0.05)
                                        .suffix(" px"),
                                );
                                ui.label("to");
                                ui.add(
                                    egui::DragValue::new(&mut d.width_max)
                                        .range(0.1..=20.0)
                                        .speed(0.05)
                                        .suffix(" px"),
                                );
                            }
                        });
                        if d.method.needs_values() || d.per_area {
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
                                            "{} classes · breaks {} … {}",
                                            b.len() + 1,
                                            fmt_sig(b.first().copied().unwrap_or(0.0), 3),
                                            fmt_sig(b.last().copied().unwrap_or(0.0), 3),
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
                                    Some(Ok(crate::data::layer::bounds_breaks(
                                        d.method, d.min, d.max, d.classes,
                                    )));
                            }
                        }
                        // Ramp preview strip.
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width().min(320.0), 14.0),
                            egui::Sense::hover(),
                        );
                        let p = ui.painter();
                        // Head/tail breaks can return fewer classes than
                        // asked for; preview what will actually be drawn,
                        // not what was requested.
                        let n = match &d.breaks {
                            Some(Ok(b)) => (b.len() + 1).max(2),
                            _ => d.classes.max(2),
                        };
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
                        ui.horizontal(|ui| {
                            if let Some(map) = &d.color_map {
                                ui.checkbox(
                                    &mut d.use_color_map,
                                    format!("{} ({} classes)", map.name, map.classes.len()),
                                )
                                .on_hover_text(
                                    "This column carries a published nomenclature. \
                                     Its colours, class names and order are used — \
                                     and the classes are known in advance, so nothing \
                                     has to be read from the data",
                                );
                            }
                            if ui
                                .small_button("Load colour map…")
                                .on_hover_text(
                                    "A QGIS colour-map export (the .txt beside most \
                                     published datasets)",
                                )
                                .clicked()
                            {
                                want_color_map = true;
                            }
                        });
                        let map = d.color_map.as_ref().filter(|_| d.use_color_map);
                        // A map answers on its own; only the generic
                        // palette needs to know which values occur.
                        if let Some(map) = map {
                            let mode = crate::data::colormap::categorical_mode(
                                Vec::new(),
                                Some(map),
                            );
                            category_preview(ui, &mode);
                        } else {
                            if d.categories.is_none() {
                                want_categories = true;
                            }
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
                                category_preview(
                                    ui,
                                    &crate::data::colormap::categorical_mode(values.clone(), None),
                                );
                            }
                        }
                        }
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        let ready = if d.numeric {
                            matches!(&d.breaks, Some(Ok(_)))
                        } else {
                            // A kept colour map is a complete class list on
                            // its own; only the generic palette has to wait
                            // for the scan.
                            (d.color_map.is_some() && d.use_color_map)
                                || matches!(&d.categories, Some(Ok(v)) if !v.is_empty())
                        };
                        if ui.add_enabled(ready, egui::Button::new("Apply")).clicked() {
                            apply = Some(StyleBy {
                                column: d.column.clone(),
                                ramp: d.ramp,
                                hidden_bins: 0,
                                per_area: d.per_area && d.numeric,
                                mode: if d.numeric {
                                    StyleMode::Graduated {
                                        method: d.method,
                                        breaks: match &d.breaks {
                                            Some(Ok(b)) => b.clone(),
                                            _ => Vec::new(),
                                        },
                                    }
                                } else {
                                    let values = match &d.categories {
                                        Some(Ok(v)) => v.clone(),
                                        _ => Vec::new(),
                                    };
                                    crate::data::colormap::categorical_mode(
                                        values,
                                        d.color_map.as_ref().filter(|_| d.use_color_map),
                                    )
                                },
                                classified_rows: None, // stamped on apply below
                                width_px: (d.numeric && d.width_by)
                                    .then_some((d.width_min, d.width_max)),
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
        if want_color_map {
            self.spawn_pick(PickFor::ColorMap, ctx, |d| {
                awaited_path(d.add_filter("colour map", &["txt", "clr"]).pick_file())
            });
        }
        // The generic palette is showing and has nothing to show yet:
        // start the scan now rather than on column selection, so a
        // column a colour map already answers never triggers one.
        if want_categories {
            let (id, col) = match &self.style_dialog {
                Some(d) => (d.layer_id, d.column.clone()),
                None => return,
            };
            if let Some(sql) = self.sql_layer_of(id) {
                if let Some(d) = &mut self.style_dialog {
                    // Marks the fetch as in flight; the poll fills it in.
                    d.categories = None;
                }
                let tx = self.cat_tx.clone();
                let ctx = ctx.clone();
                let started = self.cat_pending.insert((id, col.clone()));
                if started {
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
                        if method.needs_values() || sb.per_area
                ) {
                    sb.classified_rows = Some(l.loaded_rows() as usize);
                }
                // A colour map brings its own rendition with it.
                if sb.mode.is_color_map() {
                    l.style.adopt_palette();
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

    /// Rebuild a layer whose refinement just replaced box-drawn groups
    /// with real geometry, so the boxes stop showing under it.
    ///
    /// Sections only ever accumulate, and each one draws: the rebuild is
    /// what consolidates them back into a single mesh built from the
    /// per-group states as they now stand. Groups still on boxes rebuild
    /// as boxes (cheap — four doubles a feature), refined ones as their
    /// rows.
    fn consolidate_after_boxes(&mut self, layer_id: u64, ctx: &egui::Context) {
        let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) else {
            return;
        };
        l.generation += 1;
        let (generation, store, crs, loaded, style, box_layer) = (
            l.generation,
            l.store.clone(),
            l.crs.clone(),
            l.loaded.clone(),
            l.style.style_by.clone(),
            l.box_layer,
        );
        self.rebuilding.insert(layer_id);
        self.consolidating.insert(layer_id);
        loader::spawn_rebuild(
            loader::LoaderHandle {
                tx: self.load_tx.clone(),
                egui_ctx: ctx.clone(),
            },
            layer_id,
            generation,
            store,
            crs,
            self.display.clone(),
            loaded,
            fresh_cancel(&mut self.rebuild_cancel, layer_id),
            style,
            box_layer,
        );
    }

    /// Rebuild a layer's meshes so features land in their style bins.
    /// Legend ⟳: recompute the graduated breaks from the loaded rows of
    /// the row groups under the current viewport, then rebuild.
    fn start_viewport_reclassify(&mut self, layer_id: u64, ctx: &egui::Context) {
        use crate::data::layer::StyleMode;
        let Some(l) = self.layers.iter().find(|l| l.id == layer_id) else { return };
        let Some(sb) = &l.style.style_by else { return };
        let StyleMode::Graduated { method, breaks } = &sb.mode else { return };
        let (method, classes, per_area) = (*method, breaks.len() + 1, sb.per_area);
        let Some(idx) =
            l.store.schema.fields().iter().position(|f| f.name() == &sb.column)
        else {
            return;
        };
        let Some(rect) =
            loader::viewport_to_data_bbox(self.last_view_world, &self.display, &l.crs)
        else {
            self.push_error(format!(
                "{}: viewport does not map into the layer's CRS",
                l.name
            ));
            return;
        };
        let Some(rg) = &l.rg_bboxes else {
            self.push_error(format!(
                "{}: no row-group spatial extents — viewport reclassification \
                 needs a spatially indexed file",
                l.name
            ));
            return;
        };
        let groups = loader::intersecting_rgs(&rg.boxes, rect);
        if groups.is_empty() {
            self.push_error(format!("{}: no data under the current viewport", l.name));
            return;
        }
        let store = Arc::clone(&l.store);
        let loaded = l.loaded.clone();
        let latlong = l.crs.is_latlong;
        let tx = self.vreclass_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let cap = if per_area { 20_000 } else { 50_000 };
            let res = crate::data::loader::sample_loaded_values(
                &store,
                &loaded,
                idx,
                cap,
                per_area,
                latlong,
                Some(&groups),
            )
            .map(|mut vals| crate::data::layer::classify_breaks(method, &mut vals, classes));
            let _ = tx.send((layer_id, res));
            ctx.request_repaint();
        });
    }

    fn poll_viewport_reclass(&mut self, ctx: &egui::Context) {
        use crate::data::layer::StyleMode;
        let mut restyle: Vec<u64> = Vec::new();
        while let Ok((layer_id, res)) = self.vreclass_rx.try_recv() {
            match res {
                Ok(new_breaks) => {
                    let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) else {
                        continue;
                    };
                    let rows = l.loaded_rows() as usize;
                    if let Some(sb) = &mut l.style.style_by {
                        if let StyleMode::Graduated { breaks, .. } = &mut sb.mode {
                            *breaks = new_breaks;
                            sb.classified_rows = Some(rows);
                            restyle.push(layer_id);
                        }
                    }
                }
                Err(e) => self.push_error(format!("viewport reclassify: {e}")),
            }
        }
        for id in restyle {
            self.restyle_layer(id, ctx);
        }
    }

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
            fresh_cancel(&mut self.rebuild_cancel, l.id),
            l.style.style_by.clone(),
            l.box_layer,
        );
    }

    fn url_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
        let Some((url, profile, profiles, endpoint)) = &mut self.url_input else { return };
        let mut open = true;
        let mut submit: Option<Source> = None;
        let mut portal: Option<String> = None;
        let mut as_table = self.url_as_table;
        egui::Window::new("Open URL")
            .id(egui::Id::new("open_url"))
            .open(&mut open)
            .default_width(440.0)
            .constrain_to(floating_area).show(ctx, |ui| {
                ui.label(
                    "GeoParquet over HTTP(S) (needs range requests) or s3://bucket/key. \
                     An s3:// prefix ending in /, or a * glob, opens every matching \
                     parquet part as one layer (hive key=value segments become columns). \
                     An https:// prefix does the same through the STAC collection.json \
                     published at it — HTTP cannot list a directory, so that document is \
                     the listing:",
                );
                let edit = ui.add(
                    egui::TextEdit::singleline(url)
                        .hint_text(
                            "https://host/data.parquet · https://host/dataset/ · \
                             s3://bucket/dataset/state=MA/ · \
                             s3://bucket/dataset/state=*/roads.parquet",
                        )
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
                ui.checkbox(&mut as_table, "open as an attribute table")
                    .on_hover_text(
                        "For a file with no geometry: its columns become a SQL                          table to query and to join a layer against. A .csv is                          always one; a parquet could be either, so it is asked",
                    );
                if ui.add_enabled(valid, egui::Button::new("Open")).clicked()
                    || (enter && valid)
                {
                    // A DCAT catalog is a list of datasets, not a file:
                    // it belongs in the browser, and handing it to the
                    // parquet reader would only produce a puzzling error.
                    if url.trim().trim_end_matches('/').ends_with("/data.json") {
                        portal = Some(url.trim().to_string());
                    } else {
                        let ep = endpoint.trim();
                        submit = Some(crate::data::source::route_uri(
                            url,
                            profile.clone(),
                            (!ep.is_empty()).then(|| ep.to_string()),
                        ));
                    }
                }
            });
        self.url_as_table = as_table;
        if let Some(p) = portal {
            self.url_input = None;
            self.open_portal(&p, ctx);
            return;
        }
        if let Some(src) = submit {
            // A collection is a set of part files, which the attribute
            // reader has no way to open; the checkbox cannot make it one.
            let multi = matches!(src, Source::Stac { .. });
            if !multi && (as_table || crate::data::attrs::is_tabular(&src)) {
                // A CSV has no geometry to draw wherever it lives; a
                // parquet goes here when asked to.
                self.open_attr_table(src);
            } else {
                // Length probe / presign run in the loader thread.
                self.enqueue_load(src, ctx);
            }
            self.url_input = None;
        } else if !open {
            self.url_input = None;
        }
    }

    /// Zoom-dependent coastline detail: switch the overlay between the
    /// embedded 1:50m and the fetched 1:10m generation, with hysteresis
    /// (in at zoom ≥ 6, out at zoom ≤ 5) so the boundary never thrashes.
    /// The first time the detailed level is wanted it is fetched in the
    /// background (disk-cached); the embedded coastline stays up until
    /// the switch, and a failed fetch quietly stays there for good.
    fn update_coastline_detail(&mut self, ctx: &egui::Context) {
        use crate::data::coastline::{self, CoastLevel};
        if !self.show_coastline {
            return;
        }
        let want = match self.coast_level {
            CoastLevel::Embedded if self.camera.zoom >= 6.0 => CoastLevel::Detailed,
            CoastLevel::Detailed if self.camera.zoom <= 5.0 => CoastLevel::Embedded,
            cur => cur,
        };
        if want == self.coast_level {
            return;
        }
        if want == CoastLevel::Detailed && coastline::detailed_lines().is_none() {
            // Not fetched yet: kick it off and stay on the embedded
            // lines. The repaint on completion re-enters here.
            let ctx2 = ctx.clone();
            coastline::request_detailed(move || ctx2.request_repaint());
            return;
        }
        self.coast_level = want;
        self.coastline_chunks =
            coastline::build_coastline_at(&self.display, self.coast_level);
        self.graticule_generation += 1; // new draw key -> fresh GPU upload
        let segs: usize =
            self.coastline_chunks.iter().map(|c| c.lines[0].segments.len()).sum();
        log::info!(
            "coastline: overlay switched to {} ({segs} segments)",
            match self.coast_level {
                coastline::CoastLevel::Embedded => "1:50m (embedded)",
                coastline::CoastLevel::Detailed => "1:10m",
            }
        );
    }

    /// Where floating windows may live: full width, but between the top
    /// bars and the status bar (from the map rect measured last frame),
    /// so no window ever slides under the chrome.
    fn floating_area(&self, ctx: &egui::Context) -> egui::Rect {
        let screen = ctx.content_rect();
        if self.map_rect.width() >= 1.0 {
            egui::Rect::from_min_max(
                egui::pos2(screen.min.x, self.map_rect.min.y),
                egui::pos2(screen.max.x, self.map_rect.max.y),
            )
        } else {
            screen
        }
    }

    /// Open the vector-import dialog for a GPKG / SHP / GeoJSON path
    /// (from the File menu or a drag & drop).
    /// Start importing a vector file. There is no confirmation step: the
    /// destination is derived from the source, converting is what opening
    /// the file means, and the dialog exists to pick a table when a
    /// GeoPackage has several, to show progress, and to surface errors.
    /// A conversion that already exists and is newer than its source
    /// opens as is — re-opening a portal dataset costs nothing.
    fn begin_import(&mut self, path: PathBuf, ctx: &egui::Context) {
        use crate::data::import::ImportFormat;
        if self.gpkg_import.is_some() {
            // One dialog at a time: queue the rest, imported as each closes.
            self.import_queue.push(path);
            return;
        }
        let format = ImportFormat::from_path(&path);
        let (tables, error) = match format {
            Some(ImportFormat::Gpkg) => match crate::data::gpkg::list_tables(&path) {
                Ok(t) if t.is_empty() => {
                    (Vec::new(), Some("no feature tables in this GeoPackage".into()))
                }
                Ok(t) => (t, None),
                Err(e) => (Vec::new(), Some(e)),
            },
            Some(_) => (Vec::new(), None),
            None => (Vec::new(), Some("unsupported file type".into())),
        };
        let st = ImportState {
            format: format.unwrap_or(ImportFormat::Gpkg),
            src: path,
            tables,
            selected: 0,
            running: false,
            progress: 0.0,
            error,
            rx: None,
        };
        // The dialog pauses for input only when there is input to give:
        // an error to read, or several tables to choose between.
        let ambiguous = st.error.is_some() || st.tables.len() > 1;
        let table = st.tables.first().cloned();
        let dst = st.dst_for(table.as_ref());
        let src = st.src.clone();
        self.gpkg_import = Some(st);
        if ambiguous {
            return;
        }
        if up_to_date(&dst, &src) {
            // Converted earlier and the source unchanged: the output is
            // derived, so there is nothing to redo.
            self.gpkg_import = None;
            self.enqueue_load(Source::Local(dst), ctx);
            if !self.import_queue.is_empty() {
                let next = self.import_queue.remove(0);
                self.begin_import(next, ctx);
            }
            return;
        }
        self.start_convert(table, dst, ctx);
    }

    /// Kick off the conversion on a worker thread. It writes a `.part`
    /// renamed into place on success, so an interrupted run never leaves
    /// something that looks like a finished conversion — which is what
    /// lets `begin_import` trust an output it finds already there.
    fn start_convert(
        &mut self,
        table: Option<crate::data::gpkg::GpkgTable>,
        dst: PathBuf,
        ctx: &egui::Context,
    ) {
        use crate::data::import::ImportFormat;
        let Some(st) = self.gpkg_import.as_mut() else { return };
        let (tx, rx) = std::sync::mpsc::channel();
        let format = st.format;
        let src = st.src.clone();
        st.rx = Some(rx);
        st.running = true;
        st.progress = 0.0;
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let tx_prog = tx.clone();
            let prog = move |f: f32| {
                let _ = tx_prog.send(ImportMsg::Progress(f));
            };
            let part = dst.with_file_name(format!(
                "{}.part",
                dst.file_name().unwrap_or_default().to_string_lossy()
            ));
            let res = match (format, &table) {
                (ImportFormat::Gpkg, Some(t)) => {
                    crate::data::gpkg::convert(&src, t, &part, &prog)
                }
                (ImportFormat::Shapefile, _) => crate::data::shp::convert(&src, &part, &prog),
                (ImportFormat::GeoJson, _) => {
                    crate::data::geojson::convert(&src, &part, &prog)
                }
                (ImportFormat::Gpkg, None) => Err("no table selected".into()),
            };
            let res = res.and_then(|n| {
                std::fs::rename(&part, &dst)
                    .map(|_| n)
                    .map_err(|e| format!("cannot finish {}: {e}", dst.display()))
            });
            if res.is_err() {
                let _ = std::fs::remove_file(&part);
            }
            let _ = tx.send(match res {
                Ok(_) => ImportMsg::Done(dst),
                Err(e) => ImportMsg::Failed(e),
            });
            ctx2.request_repaint();
        });
    }

    fn gpkg_import_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
        if self.gpkg_import.is_none() {
            return;
        }
        // Drain conversion messages.
        let mut done: Option<PathBuf> = None;
        {
            let st = self.gpkg_import.as_mut().unwrap();
            let mut failed = None;
            if let Some(rx) = &st.rx {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        ImportMsg::Progress(f) => st.progress = f,
                        ImportMsg::Done(p) => done = Some(p),
                        ImportMsg::Failed(e) => failed = Some(e),
                    }
                }
            }
            if let Some(e) = failed {
                st.error = Some(e);
                st.running = false;
                st.rx = None;
            }
        }
        if let Some(p) = done {
            self.gpkg_import = None;
            self.enqueue_load(Source::Local(p), ctx);
            if !self.import_queue.is_empty() {
                let next = self.import_queue.remove(0);
                self.begin_import(next, ctx);
            }
            return;
        }

        use crate::data::import::ImportFormat;
        let mut open = true;
        let mut start: Option<(Option<crate::data::gpkg::GpkgTable>, PathBuf)> = None;
        let queued = self.import_queue.len();
        let st = self.gpkg_import.as_mut().unwrap();
        let title = if queued > 0 {
            format!("Import {} ({queued} more queued)", st.format.label())
        } else {
            format!("Import {}", st.format.label())
        };
        egui::Window::new(title)
            .id(egui::Id::new("gpkg_import"))
            .open(&mut open)
            .default_width(430.0)
            .collapsible(false)
            .constrain_to(floating_area).show(ctx, |ui| {
                ui.label(RichText::new(st.src.display().to_string()).weak().small());
                if let Some(e) = &st.error {
                    ui.colored_label(egui::Color32::from_rgb(220, 60, 60), e);
                    return;
                }
                let table = if st.format == ImportFormat::Gpkg {
                    ui.add_enabled_ui(!st.running, |ui| {
                        let label = |t: &crate::data::gpkg::GpkgTable| {
                            format!(
                                "{} ({} rows, {})",
                                t.name,
                                fmt_count(t.rows as usize),
                                t.srs_name
                            )
                        };
                        egui::ComboBox::from_label("feature table")
                            .selected_text(label(&st.tables[st.selected]))
                            .show_ui(ui, |ui| {
                                for (i, t) in st.tables.iter().enumerate() {
                                    ui.selectable_value(&mut st.selected, i, label(t));
                                }
                            });
                    });
                    Some(st.tables[st.selected].clone())
                } else {
                    None
                };
                let dst = st.dst_for(table.as_ref());
                ui.label(RichText::new(format!("→ {}", dst.display())).weak().small())
                    .on_hover_text(
                        "Converted to plain WKB GeoParquet (raw import: the quality \
                         scorecard and Optimize take it from there)",
                    );
                ui.add_space(4.0);
                if st.running {
                    ui.add(egui::ProgressBar::new(st.progress).show_percentage());
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                } else if ui.button("Import").clicked() {
                    // Reached only when there was something to decide (a
                    // table among several) or to retry after a failure;
                    // the unambiguous case never waits for this click.
                    start = Some((table, dst));
                }
            });
        if let Some((table, dst)) = start {
            self.start_convert(table, dst, ctx);
        }
        if !open && !self.gpkg_import.as_ref().is_some_and(|s| s.running) {
            self.gpkg_import = None;
            if !self.import_queue.is_empty() {
                let next = self.import_queue.remove(0);
                self.begin_import(next, ctx);
            }
        }
    }

    fn poll_optimizer(&mut self, ctx: &egui::Context) {
        let mut open_result: Option<PathBuf> = None;
        let mut open_remote: Option<S3Dest> = None;
        while let Ok(msg) = self.opt_rx.try_recv() {
            let Some(o) = &mut self.optimize else { continue };
            match msg {
                OptMsg::Progress(f, s) => o.progress = (f, s),
                OptMsg::Done(report, path, dest) => {
                    o.running = false;
                    // Quality-gate flow: the optimized copy opens as the
                    // layer the original never became.
                    if o.open_result {
                        match &dest {
                            Some(d) => open_remote = Some(d.clone()),
                            None => open_result = Some(path.clone()),
                        }
                    }
                    o.report_s3 = dest;
                    o.report = Some((*report, path));
                }
                OptMsg::PublishedAsIs(dest, size) => {
                    o.running = false;
                    o.report_as_is = Some((dest, size));
                }
                OptMsg::Failed(e) => {
                    o.running = false;
                    o.error = Some(e);
                }
                OptMsg::Cardinalities(id, c) => {
                    if o.layer_id == id {
                        o.cardinalities = Some(c);
                        o.card_pending = false;
                    }
                }
            }
        }
        if let Some(path) = open_result {
            self.enqueue_load(Source::Local(path), ctx);
        }
        if let Some(d) = open_remote {
            self.enqueue_load(
                Source::S3 {
                    uri: d.uri,
                    profile: d.profile,
                    endpoint: d.endpoint,
                    url: String::new(),
                    len: 0,
                },
                ctx,
            );
        }
    }

    /// Publish the source file to S3/R2 unchanged — the "upload as-is"
    /// path that skips the optimize rewrite entirely.
    fn start_publish_as_is(&mut self, src_path: PathBuf, ctx: &egui::Context) {
        let Some(o) = &mut self.optimize else { return };
        o.running = true;
        o.error = None;
        o.report = None;
        o.report_as_is = None;
        o.progress = (0.0, "starting upload".into());
        let dest = S3Dest {
            uri: o.s3_uri.trim().to_string(),
            profile: o.s3_profile.clone(),
            endpoint: {
                let e = o.s3_endpoint.trim();
                (!e.is_empty()).then(|| e.to_string())
            },
        };
        let stac = o.stac.then(|| (o.crs.clone(), o.layer_name.clone()));
        let tx = self.opt_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            use crate::data::info::fmt_bytes;
            let size = std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);
            let name = src_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            // The as-is STAC sidecar: staged in the temp dir (never
            // beside the user's source file), uploaded after the data.
            let publish_stac = || -> Result<(), String> {
                let Some((crs, title)) = &stac else { return Ok(()) };
                let json = std::env::temp_dir()
                    .join(format!("geopq_stac_{}.json", std::process::id()));
                crate::data::stac::write_for_file_at(&src_path, &json, title, crs)?;
                let r = upload_collection_merged(
                    &json,
                    &dest.uri,
                    dest.profile.as_deref(),
                    dest.endpoint.as_deref(),
                );
                let _ = std::fs::remove_file(&json);
                r.map_err(|e| format!("data uploaded, but {e}"))
            };
            let msg = match crate::data::source::aws::upload_file(
                &src_path,
                &dest.uri,
                dest.profile.as_deref(),
                dest.endpoint.as_deref(),
                &|sent, total| {
                    let frac = if total > 0 { sent as f32 / total as f32 } else { 0.0 };
                    let _ = tx.send(OptMsg::Progress(
                        frac,
                        format!("uploading {name}: {} / {}", fmt_bytes(sent), fmt_bytes(total)),
                    ));
                    ctx.request_repaint();
                },
            )
            .and_then(|()| publish_stac())
            {
                Ok(()) => OptMsg::PublishedAsIs(dest, size),
                Err(e) => OptMsg::Failed(e),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
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
        // STAC sidecar for publishes: needs the CRS (extent goes to
        // WGS84) and a human title.
        let stac = (o.dest_s3 && o.stac).then(|| (o.crs.clone(), o.layer_name.clone()));
        let replace_remote = o.replace_remote;
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
        // Publish destination, when the output goes to S3/R2 instead of
        // staying local (dst is then a temp path).
        let dest: Option<S3Dest> = o.dest_s3.then(|| S3Dest {
            uri: o.s3_uri.trim().to_string(),
            profile: o.s3_profile.clone(),
            endpoint: {
                let e = o.s3_endpoint.trim();
                (!e.is_empty()).then(|| e.to_string())
            },
        });
        // "Merge with…": the checked layers plus the primary, staged
        // into one raw file that the optimizer then treats as the source.
        let merge_inputs: Vec<crate::data::merge::MergeInput> = if o.merge_with.is_empty() {
            Vec::new()
        } else {
            std::iter::once(o.layer_id)
                .chain(o.merge_with.iter().copied())
                .filter_map(|id| self.layers.iter().find(|l| l.id == id))
                .map(|l| crate::data::merge::MergeInput {
                    store: Arc::clone(&l.store),
                    crs: l.crs.clone(),
                    name: l.name.clone(),
                })
                .collect()
        };
        let merge_source_col = self.optimize.as_ref().is_some_and(|o| o.merge_source_col);
        let tx = self.opt_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let progress = |f: f32, s: &str| {
                let _ = tx.send(OptMsg::Progress(f, s.to_string()));
                ctx.request_repaint();
            };
            // Merge phase (when requested): 0–35% of the bar, then the
            // optimize pass on the staged file takes the rest.
            let mut src = src;
            let mut opts = opts;
            let mut staging: Option<PathBuf> = None;
            if merge_inputs.len() > 1 {
                let stage = std::env::temp_dir()
                    .join(format!("geopq_merge_{}.parquet", std::process::id()));
                match crate::data::merge::merge(
                    &merge_inputs,
                    &stage,
                    merge_source_col,
                    &|f, s| progress(f * 0.35, s),
                ) {
                    Ok(_) => {
                        src = Source::Local(stage.clone());
                        // The staged file is plain WKB: coordinate-column
                        // synthesis from the primary no longer applies.
                        opts.xy_geom = None;
                        staging = Some(stage);
                    }
                    Err(e) => {
                        let _ = tx.send(OptMsg::Failed(format!("merge failed: {e}")));
                        ctx.request_repaint();
                        return;
                    }
                }
            }
            let scale = if staging.is_some() { 0.35 } else { 0.0 };
            let progress =
                |f: f32, s: &str| progress(scale + f * (1.0 - scale), s);
            let msg = match crate::data::optimize::optimize(&src, &dst, &opts, epsg, admin.as_ref(), &progress) {
                Ok(r) => match &dest {
                    None => OptMsg::Done(Box::new(r), dst, None),
                    Some(d) => {
                        // Upload phase: file to key, dataset dir to prefix.
                        use crate::data::info::fmt_bytes;
                        use crate::data::source::aws;
                        // The STAC sidecar rides inside a dataset dir
                        // (upload_tree carries it); a single file gets
                        // it uploaded beside itself afterwards.
                        let stac_json: Result<Option<PathBuf>, String> = match &stac {
                            Some((crs, title)) => {
                                crate::data::stac::write_for_output(&dst, title, crs)
                                    .map(Some)
                                    .map_err(|e| format!("STAC collection.json failed: {e}"))
                            }
                            None => Ok(None),
                        };
                        let up = |sent: u64, total: u64, name: &str| {
                            let frac = if total > 0 {
                                sent as f32 / total as f32
                            } else {
                                0.0
                            };
                            progress(
                                frac,
                                &format!(
                                    "uploading {name}: {} / {}",
                                    fmt_bytes(sent),
                                    fmt_bytes(total)
                                ),
                            );
                        };
                        let uploaded = stac_json.and_then(|stac_json| {
                            if dst.is_dir() {
                                // A dataset tree replaces the prefix
                                // wholesale, so a collection already
                                // published there is a decision, not a
                                // merge: likelier a wrong prefix than an
                                // intended replace.
                                let sibling = format!(
                                    "{}/collection.json",
                                    d.uri.trim_end_matches('/')
                                );
                                if !replace_remote
                                    && aws::fetch_small(
                                        &sibling,
                                        d.profile.as_deref(),
                                        d.endpoint.as_deref(),
                                    )
                                    .map_err(|e| {
                                        format!("cannot check the destination: {e}")
                                    })?
                                    .is_some()
                                {
                                    return Err(format!(
                                        "{sibling} already exists — the destination \
                                         publishes a dataset. Tick \"Replace the \
                                         dataset at the destination\" to overwrite \
                                         it, or publish to another prefix."
                                    ));
                                }
                                // collection.json sits inside: one tree upload.
                                aws::upload_tree(
                                    &dst,
                                    &d.uri,
                                    d.profile.as_deref(),
                                    d.endpoint.as_deref(),
                                    &up,
                                )
                                .map(|_| ())
                            } else {
                                let name = dst
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                aws::upload_file(
                                    &dst,
                                    &d.uri,
                                    d.profile.as_deref(),
                                    d.endpoint.as_deref(),
                                    &|s, t| up(s, t, &name),
                                )?;
                                if let Some(json) = &stac_json {
                                    progress(1.0, "uploading collection.json");
                                    upload_collection_merged(
                                        json,
                                        &d.uri,
                                        d.profile.as_deref(),
                                        d.endpoint.as_deref(),
                                    )
                                    .map_err(|e| format!("data uploaded, but {e}"))?;
                                    let _ = std::fs::remove_file(json);
                                }
                                Ok(())
                            }
                        });
                        match uploaded {
                            Ok(()) => {
                                // The local copy was only a staging file.
                                let _ = if dst.is_dir() {
                                    std::fs::remove_dir_all(&dst)
                                } else {
                                    std::fs::remove_file(&dst)
                                };
                                OptMsg::Done(Box::new(r), dst, Some(d.clone()))
                            }
                            Err(e) => OptMsg::Failed(format!(
                                "optimized, but the upload failed: {e}\n\
                                 (the local copy is at {})",
                                dst.display()
                            )),
                        }
                    }
                },
                Err(e) => OptMsg::Failed(e),
            };
            if let Some(s) = &staging {
                let _ = std::fs::remove_file(s);
            }
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    fn optimize_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
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
        // Layers offered for "Merge with…": other vector layers of the
        // SAME geometry family as the primary (merging points into a
        // polygon export is never what anyone means, and the filter
        // keeps the list short), with shared/conflicting column counts.
        let merge_candidates: Vec<(u64, String, u64, usize, usize)> = match &self.optimize {
            Some(o) => {
                let primary = self.layers.iter().find(|l| l.id == o.layer_id);
                let primary_kind = primary.map(|l| l.kind());
                self.layers
                    .iter()
                    .filter(|l| l.id != o.layer_id && Some(l.kind()) == primary_kind)
                    .map(|l| {
                        let (mut shared, mut conflicts) = (0usize, 0usize);
                        if let Some(p) = primary {
                            for f in p.store.schema.fields() {
                                match l.store.schema.field_with_name(f.name()) {
                                    Ok(g) if g.data_type() == f.data_type() => shared += 1,
                                    Ok(_) => conflicts += 1,
                                    Err(_) => {}
                                }
                            }
                        }
                        (l.id, l.name.clone(), l.store.total_rows(), shared, conflicts)
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        // Scorecard verdict of the primary layer, for the as-is nudge.
        let primary_indexable: Option<bool> = self.optimize.as_ref().and_then(|o| {
            self.layers
                .iter()
                .find(|l| l.id == o.layer_id)
                .and_then(|l| l.info.quality.as_ref().map(|q| q.indexable))
        });
        let Some(o) = &mut self.optimize else { return };
        let layer_id = o.layer_id;
        let mut open = true;
        let mut start: Option<PathBuf> = None;
        // The output picker is a native dialog and cannot open from inside
        // this frame (see `spawn_pick`): what it needs is recorded here and
        // the dialog is opened once the window closure has released `self`.
        // (start directory, file stem, partitioned).
        let mut want_output: Option<(Option<PathBuf>, String, bool)> = None;
        let mut start_upload: Option<PathBuf> = None;
        let mut load_result: Option<PathBuf> = None;
        let mut load_remote: Option<S3Dest> = None;
        let mut close = false;
        let mut want_cards = false;
        egui::Window::new(format!("Export — {}", o.layer_name))
            .id(egui::Id::new("optimize_dialog"))
            .open(&mut open)
            .default_width(400.0)
            .constrain_to(floating_area).show(ctx, |ui| {
                if let Some((d, size)) = &o.report_as_is {
                    use crate::data::info::fmt_bytes;
                    ui.label(RichText::new(format!("Published: {}", d.uri)).strong());
                    ui.label(
                        RichText::new(format!(
                            "{} uploaded as-is (no rewrite)",
                            fmt_bytes(*size)
                        ))
                        .weak(),
                    );
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Load as layer").clicked() {
                            load_remote = Some(d.clone());
                        }
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                    return;
                }
                if let Some((rep, path)) = &o.report {
                    use crate::data::info::fmt_bytes;
                    ui.label(
                        RichText::new(match &o.report_s3 {
                            Some(d) => format!("Published: {}", d.uri),
                            None => format!("Written: {}", path.display()),
                        })
                        .strong(),
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
                        // Published outputs always reload in place (a
                        // partitioned prefix opens via the listing);
                        // local partitioned datasets load via Open folder.
                        let loadable = o.report_s3.is_some() || rep.files <= 1;
                        if loadable && ui.button("Load as layer").clicked() {
                            match &o.report_s3 {
                                Some(d) => load_remote = Some(d.clone()),
                                None => load_result = Some(path.clone()),
                            }
                        }
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                    return;
                }

                ui.add_enabled_ui(!o.running, |ui| {
                    let label = |v: GpVersion, rec: GpVersion| {
                        if v == rec {
                            format!("{} — recommended", v.label())
                        } else {
                            v.label().to_string()
                        }
                    };
                    if ui
                        .radio(
                            o.opts.version == GpVersion::V1_1,
                            label(GpVersion::V1_1, o.recommended),
                        )
                        .on_hover_text(
                            "WKB opens everywhere; the covering bbox column drives\n\
                             row/page pruning and per-feature viewport selection.\n\
                             The safe choice for files that travel.",
                        )
                        .clicked()
                    {
                        o.opts.version = GpVersion::V1_1;
                        o.opts.covering = true;
                        o.opts.geoarrow_aux = false;
                    }
                    if ui
                        .radio(
                            o.opts.version == GpVersion::V1_1GeoArrow,
                            label(GpVersion::V1_1GeoArrow, o.recommended),
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
                        o.opts.geoarrow_aux = false;
                    }
                    if ui
                        .radio(
                            o.opts.version == GpVersion::V2_0,
                            label(GpVersion::V2_0, o.recommended),
                        )
                        .on_hover_text(
                            "Native geo statistics replace the covering column for pruning;\n\
                             needs GeoParquet 2.0 aware readers.\n\
                             Selecting it applies the official recommended settings;\n\
                             the flavor options below are workbench extras.",
                        )
                        .clicked()
                    {
                        // Official recommended settings: native GEOMETRY
                        // (WKB) + native stats only. The flavor checkboxes
                        // below opt back into the extras.
                        o.opts.version = GpVersion::V2_0;
                        o.opts.covering = false;
                        o.opts.geoarrow_aux = false;
                    }
                    if o.opts.version == GpVersion::V2_0 {
                        let fits = o.recommended == GpVersion::V1_1GeoArrow;
                        ui.indent("v2_flavor", |ui| {
                            let resp = ui
                                .add_enabled(
                                    fits,
                                    egui::Checkbox::new(
                                        &mut o.opts.geoarrow_aux,
                                        "GeoArrow geometry column (extra)",
                                    ),
                                )
                                .on_hover_text(
                                    "A decode format, not an index: also writes the \
                                     geometry as GeoArrow coordinate arrays in a \
                                     second column next to the official native \
                                     GEOMETRY (WKB) one. The file stays valid 2.0; \
                                     this workbench and other GeoArrow-aware readers \
                                     decode the fast column with no per-feature WKB \
                                     parsing. Roughly doubles geometry storage.",
                                );
                            if !fits {
                                resp.on_disabled_hover_text(
                                    "Needs a single geometry family (points, lines \
                                     or polygons; singles promote to multi)",
                                );
                            }
                        });
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
                        .on_hover_text(
                            "A spatial index: per-feature bboxes drive exact viewport \
                             selection and page-level pruning (native 2.0 stats only \
                             prune whole row groups). Cheap — about 32 bytes per \
                             feature before compression. Independent of the GeoArrow \
                             column, which changes the decode format, not the index.",
                        );
                    ui.checkbox(&mut o.viewport_only, "viewport only")
                        .on_hover_text(
                            "Export only features intersecting the current map viewport",
                        );

                    // --- merge with other layers ---
                    if !merge_candidates.is_empty() {
                        ui.separator();
                        ui.label(RichText::new("Merge with:").strong()).on_hover_text(
                            "Concatenate other loaded layers into this export before \
                             optimizing. Schemas union by column name (missing values \
                             become NULL, conflicting types are dropped); geometries \
                             reproject into this layer's CRS.",
                        );
                        egui::ScrollArea::vertical()
                            .id_salt("export_merge_list")
                            .max_height(132.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                for (id, name, rows, shared, conflicts) in &merge_candidates {
                                    let mut on = o.merge_with.contains(id);
                                    let label =
                                        format!("{name} — {}", fmt_count(*rows as usize));
                                    let hover = if *conflicts > 0 {
                                        format!(
                                            "{shared} shared columns; {conflicts} dropped \
                                             (type conflicts)"
                                        )
                                    } else {
                                        format!("{shared} shared columns")
                                    };
                                    if ui
                                        .checkbox(&mut on, label)
                                        .on_hover_text(hover)
                                        .changed()
                                    {
                                        if on {
                                            o.merge_with.insert(*id);
                                        } else {
                                            o.merge_with.remove(id);
                                        }
                                    }
                                }
                            });
                        if !o.merge_with.is_empty() {
                            ui.checkbox(
                                &mut o.merge_source_col,
                                "add a source_layer column",
                            )
                            .on_hover_text(
                                "Tag every row with the layer it came from — handy \
                                 for styling by value and SQL filters",
                            );
                        }
                    }

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
                ui.add_enabled_ui(!o.running, |ui| {
                    ui.checkbox(&mut o.dest_s3, "Publish to S3 / R2")
                        .on_hover_text(
                            "Upload the optimized output to a bucket instead of \
                             saving locally. Needs credentials with write access \
                             (~/.aws; for R2, an API token and the account \
                             endpoint). The result opens in place from its \
                             s3:// URI.",
                        );
                    if o.dest_s3 {
                        ui.add(
                            egui::TextEdit::singleline(&mut o.s3_uri)
                                .hint_text(if o.part_mode == PartMode::None {
                                    "s3://bucket/ (file name appended) · s3://bucket/path/file.parquet"
                                } else {
                                    "s3://bucket/ (dataset prefix appended) · s3://bucket/dataset/"
                                })
                                .desired_width(f32::INFINITY),
                        );
                        ui.horizontal(|ui| {
                            ui.label("Endpoint:");
                            ui.add(
                                egui::TextEdit::singleline(&mut o.s3_endpoint)
                                    .hint_text("(AWS) · <account>.r2.cloudflarestorage.com")
                                    .desired_width(180.0),
                            );
                            ui.label("profile:");
                            let current = o
                                .s3_profile
                                .clone()
                                .unwrap_or_else(|| "(auto)".into());
                            egui::ComboBox::from_id_salt("opt_s3_profile")
                                .width(110.0)
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut o.s3_profile,
                                        None,
                                        "(auto: env / default)",
                                    );
                                    for p in o.s3_profiles.clone() {
                                        ui.selectable_value(
                                            &mut o.s3_profile,
                                            Some(p.clone()),
                                            p,
                                        );
                                    }
                                });
                        });
                        ui.checkbox(&mut o.stac, "write a STAC collection.json")
                            .on_hover_text(
                                "Describe the published data the standard way: a \
                                 STAC Collection uploaded beside it, with the \
                                 extent read from the parquet footers. The \
                                 distributing-geoparquet best practices \
                                 recommend one for any published dataset. \
                                 Publishing a file where a collection already \
                                 exists adds it to that collection.",
                            );
                        ui.checkbox(
                            &mut o.replace_remote,
                            "Replace the dataset at the destination",
                        )
                        .on_hover_text(
                            "A partitioned publish that finds a collection.json \
                             already at the destination prefix refuses unless \
                             this is ticked: that prefix is somebody's dataset, \
                             and a wrong prefix is likelier than an intended \
                             replace. Single-file publishes never need it — \
                             they merge into the existing collection instead.",
                        );
                        if let Source::Local(p) = &o.src {
                            let mut reasons: Vec<&str> = Vec::new();
                            if !o.merge_with.is_empty() {
                                reasons.push("merge");
                            }
                            if o.viewport_only {
                                reasons.push("viewport only");
                            }
                            if o.opts.h3_resolution.is_some() {
                                reasons.push("H3 column");
                            }
                            if o.admin_layer.is_some() {
                                reasons.push("admin column");
                            }
                            if o.part_mode != PartMode::None {
                                reasons.push("partitioning");
                            }
                            let enabled = reasons.is_empty();
                            if !enabled {
                                o.upload_as_is = false;
                            }
                            let hover = if !enabled {
                                format!("needs a rewrite: {}", reasons.join(", "))
                            } else {
                                match primary_indexable {
                                    Some(true) => {
                                        "This file already passes the gating checks — \
                                         uploading it unchanged is fine."
                                            .into()
                                    }
                                    Some(false) => {
                                        "The scorecard says this file would benefit \
                                         from a rewrite — consider exporting \
                                         instead."
                                            .into()
                                    }
                                    None => format!(
                                        "Upload {} byte-for-byte, skipping the rewrite",
                                        p.display()
                                    ),
                                }
                            };
                            ui.add_enabled(
                                enabled,
                                egui::Checkbox::new(
                                    &mut o.upload_as_is,
                                    "upload the file as-is (skip the rewrite)",
                                ),
                            )
                            .on_hover_text(hover)
                            .on_disabled_hover_text(format!(
                                "needs a rewrite: {}",
                                reasons.join(", ")
                            ));
                        }
                    }
                });
                ui.add_space(4.0);
                if o.running {
                    ui.add(
                        egui::ProgressBar::new(o.progress.0)
                            .text(o.progress.1.clone())
                            .animate(true),
                    );
                } else if ui
                    .button(if o.dest_s3 { "Optimize & upload…" } else { "Export…" })
                    .clicked()
                {
                    let stem = o.src.name();
                    let stem = stem.trim_end_matches(".parquet").to_string();
                    if o.dest_s3 {
                        // Normalize the destination, then stage locally in
                        // a temp path; the worker uploads and removes it.
                        // A bare bucket or a prefix ending in `/` gets the
                        // layer's file name appended, mirroring the local
                        // save dialog's pre-filled name.
                        let trimmed = o.s3_uri.trim().to_string();
                        let uri = trimmed.trim_end_matches('/').to_string();
                        let rest = uri.strip_prefix("s3://").unwrap_or("");
                        if rest.is_empty() {
                            o.error = Some(
                                "destination must start with s3://bucket".into(),
                            );
                        } else {
                            let needs_name =
                                trimmed.ends_with('/') || !rest.contains('/');
                            if o.part_mode == PartMode::None {
                                o.s3_uri = if needs_name && o.upload_as_is {
                                    // As-is keeps the source's own name.
                                    format!("{uri}/{stem}.parquet")
                                } else if needs_name {
                                    format!("{uri}/{stem}_optimized.parquet")
                                } else if uri.ends_with(".parquet") {
                                    uri
                                } else {
                                    format!("{uri}.parquet")
                                };
                                if o.upload_as_is
                                    && let Source::Local(p) = &o.src
                                {
                                    start_upload = Some(p.clone());
                                } else {
                                    start = Some(std::env::temp_dir().join(format!(
                                        "geopq_publish_{stem}_{}.parquet",
                                        std::process::id()
                                    )));
                                }
                            } else {
                                o.s3_uri = if needs_name {
                                    format!("{uri}/{stem}_partitioned/")
                                } else {
                                    format!("{uri}/")
                                };
                                start = Some(std::env::temp_dir().join(format!(
                                    "geopq_publish_{stem}_{}_parts",
                                    std::process::id()
                                )));
                            }
                        }
                    } else {
                        let dir = match &o.src {
                            Source::Local(p) => p.parent().map(PathBuf::from),
                            _ => None,
                        };
                        want_output =
                            Some((dir, stem.clone(), o.part_mode != PartMode::None));
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
                        let _ = tx.send(OptMsg::Cardinalities(layer_id, counts));
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
        if let Some((dir, stem, partitioned)) = want_output {
            let what = if partitioned {
                PickFor::OptimizeFolder
            } else {
                PickFor::OptimizeFile
            };
            self.spawn_pick(what, ctx, move |mut d| {
                if let Some(dir) = dir {
                    d = d.set_directory(dir);
                }
                if partitioned {
                    // The dataset root is named under the folder chosen.
                    awaited_path(d.pick_folder())
                } else {
                    awaited_path(
                        d.set_file_name(format!("{stem}_optimized.parquet"))
                            .add_filter("GeoParquet", &["parquet"])
                            .save_file(),
                    )
                }
            });
        }
        if let Some(dst) = start {
            self.start_optimize(dst, ctx);
        }
        if let Some(p) = start_upload {
            self.start_publish_as_is(p, ctx);
        }
        if let Some(p) = load_result {
            self.enqueue_load(Source::Local(p), ctx);
            close = true;
        }
        if let Some(d) = load_remote {
            self.enqueue_load(
                Source::S3 {
                    uri: d.uri,
                    profile: d.profile,
                    endpoint: d.endpoint,
                    url: String::new(),
                    len: 0,
                },
                ctx,
            );
            close = true;
        }
        // Keep the worker's state visible: ignore window close while running.
        if (close || !open) && self.optimize.as_ref().is_some_and(|o| !o.running) {
            self.optimize = None;
        }
    }

    /// Confirmation for "Reset layout". Worth asking: closing the layers is
    /// cheap to undo, the styles, filters and renames on them are not.
    fn reset_confirm_window(&mut self, ctx: &egui::Context) {
        if !self.confirm_reset {
            return;
        }
        let floating_area = self.floating_area(ctx);
        let n = self.layers.len() + self.loading.len();
        let mut reset = false;
        egui::Window::new("Reset layout")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(380.0)
            .constrain_to(floating_area)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Close {} and return to the default projection, camera \
                     and view settings.",
                    if n == 1 { "1 layer".to_string() } else { format!("{n} layers") },
                ));
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Styles, filters and layer names are not kept. Save the \
                         context first if you want them back.",
                    )
                    .weak()
                    .small(),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.confirm_reset = false;
                    }
                    if ui.button(format!("{} Reset", ph::ARROW_COUNTER_CLOCKWISE)).clicked() {
                        reset = true;
                    }
                });
            });
        if reset {
            self.confirm_reset = false;
            self.reset_layout(ctx);
        }
    }

    /// The attribute import dialog: every column, what it will be called,
    /// what type it will get, and what that costs.
    ///
    /// The values matter as much as the types. A column of counts that
    /// inference called text is a mystery until you can see the `NA` that
    /// did it, and then it is a one-click decision.
    fn attr_import_window(&mut self, ctx: &egui::Context) {
        use crate::data::attrs::{self, ColType};
        if self.attr_import.is_none() {
            return;
        }
        let floating_area = self.floating_area(ctx);
        let mut open = true;
        let mut go = false;
        let mut retype = false;
        let job = self.attr_import.as_mut().expect("checked above");
        let title = format!("Import table — {}", job.name);
        egui::Window::new(title)
            .id(egui::Id::new("attr_import"))
            .open(&mut open)
            .collapsible(false)
            .default_width(720.0)
            .constrain_to(floating_area)
            .show(ctx, |ui| {
                crate::theme::compact(ui);
                ui.horizontal(|ui| {
                    if job.preview.sampled {
                        ui.label(
                            RichText::new(format!(
                                "{} columns, {} rows sampled",
                                job.preview.plan.columns.len(),
                                fmt_count(job.preview.sampled_rows),
                            ))
                            .weak()
                            .small(),
                        );
                    } else {
                        // Names and types are already right; only the
                        // values and the coordinate guess are pending.
                        ui.spinner();
                        ui.label(
                            RichText::new(format!(
                                "{} columns from the file's footer — reading values…",
                                job.preview.plan.columns.len(),
                            ))
                            .weak()
                            .small(),
                        );
                    }
                });
                if !job.preview.typed_source {
                    ui.horizontal(|ui| {
                        ui.label("separator:");
                        for (d, label) in
                            [(b',', "comma"), (b';', "semicolon"), (b'\t', "tab"), (b'|', "pipe")]
                        {
                            if ui
                                .selectable_label(job.preview.plan.delimiter == d, label)
                                .clicked()
                                && job.preview.plan.delimiter != d
                            {
                                job.preview.plan.delimiter = d;
                                job.reread = true;
                            }
                        }
                        let mut header = job.preview.plan.has_header;
                        if ui.checkbox(&mut header, "first row is names").changed() {
                            job.preview.plan.has_header = header;
                            job.reread = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("numbers:").on_hover_text(
                            "How the file groups digits. A comma means \
                             opposite things in 1,234.56 and 1 234,56, so \
                             this cannot be guessed from a value alone",
                        );
                        for f in attrs::NumberFormat::ALL {
                            if ui
                                .selectable_label(job.preview.plan.numbers == f, f.label())
                                .clicked()
                                && job.preview.plan.numbers != f
                            {
                                job.preview.plan.numbers = f;
                                retype = true;
                            }
                        }
                    });
                } else {
                    ui.label(
                        RichText::new("Types come from the file. Names and selection still apply.")
                            .weak()
                            .small(),
                    );
                }
                ui.separator();

                egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                    egui::Grid::new("attr_cols")
                        .num_columns(6)
                        .striped(true)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            // Header checkbox: on when everything is in,
                            // and clicking it takes everything the other
                            // way. The usual table gesture, and the only
                            // sane one for a file with sixty columns of
                            // which you want three.
                            let all = job.preview.plan.columns.iter().all(|c| c.include);
                            let mut toggle = all;
                            if ui
                                .checkbox(&mut toggle, "")
                                .on_hover_text(if all { "Select none" } else { "Select all" })
                                .clicked()
                            {
                                for c in job.preview.plan.columns.iter_mut() {
                                    c.include = !all;
                                }
                            }
                            ui.label(RichText::new("column").weak().small());
                            ui.label(RichText::new("name in SQL").weak().small());
                            ui.label(RichText::new("type").weak().small());
                            ui.label("");
                            ui.label(RichText::new("values").weak().small());
                            ui.end_row();

                            for (i, c) in job.preview.plan.columns.iter_mut().enumerate() {
                                let pv = &job.preview.columns[i];
                                ui.checkbox(&mut c.include, "")
                                    .on_hover_text("Import this column");
                                ui.label(RichText::new(&c.source_name).small());
                                ui.add_enabled(
                                    c.include,
                                    egui::TextEdit::singleline(&mut c.name)
                                        .desired_width(150.0)
                                        .font(egui::TextStyle::Monospace),
                                );
                                let before = c.ty;
                                egui::ComboBox::from_id_salt(("attr_ty", i))
                                    .selected_text(c.ty.label())
                                    .width(90.0)
                                    .show_ui(ui, |ui| {
                                        for t in ColType::ALL {
                                            ui.selectable_value(&mut c.ty, t, t.label());
                                        }
                                    });
                                if c.ty != before {
                                    retype = true;
                                }
                                if c.ty != pv.inferred {
                                    ui.label(
                                        RichText::new("*").small().color(
                                            Color32::from_rgb(242, 140, 26),
                                        ),
                                    )
                                    .on_hover_text(format!(
                                        "changed from {}",
                                        pv.inferred.label()
                                    ));
                                } else {
                                    ui.label("");
                                }
                                ui.vertical(|ui| {
                                    if !job.preview.sampled {
                                        ui.label(RichText::new("…").weak().small());
                                    }
                                    if !pv.samples.is_empty() {
                                        ui.label(
                                            RichText::new(pv.samples.join("  ·  "))
                                                .monospace()
                                                .small()
                                                .weak(),
                                        );
                                    }
                                    if pv.bad > 0 {
                                        ui.label(
                                            RichText::new(format!(
                                                "{} won't parse: {}",
                                                fmt_count(pv.bad),
                                                pv.bad_examples.join(", "),
                                            ))
                                            .small()
                                            .color(Color32::from_rgb(242, 140, 26)),
                                        );
                                    }
                                });
                                ui.end_row();
                            }
                        });
                });

                ui.separator();
                {
                    use crate::data::attrs::GeometryPlan;
                    let numeric: Vec<String> = job
                        .preview
                        .plan
                        .columns
                        .iter()
                        .filter(|c| matches!(c.ty, ColType::Integer | ColType::Float))
                        .map(|c| c.name.clone())
                        .collect();
                    let is_points = matches!(job.preview.plan.geometry, GeometryPlan::Points { .. });
                    ui.horizontal(|ui| {
                        ui.label("geometry:");
                        if ui.selectable_label(!is_points, "none (a table)").clicked() {
                            job.preview.plan.geometry = GeometryPlan::None;
                        }
                        let can = numeric.len() >= 2;
                        let mut pick_points = is_points;
                        if ui
                            .add_enabled_ui(can, |ui| {
                                ui.selectable_value(&mut pick_points, true, "points from X, Y")
                            })
                            .inner
                            .on_disabled_hover_text(
                                "needs two numeric columns to take coordinates from",
                            )
                            .clicked()
                            && !is_points
                        {
                            job.preview.plan.geometry = GeometryPlan::Points {
                                x: numeric[0].clone(),
                                y: numeric[1].clone(),
                                epsg: 4326,
                            };
                        }
                    });
                    if let GeometryPlan::Points { x, y, epsg } = &mut job.preview.plan.geometry {
                        ui.horizontal(|ui| {
                            let pick = |ui: &mut egui::Ui, salt: &str, cur: &mut String| {
                                egui::ComboBox::from_id_salt(salt)
                                    .selected_text(cur.as_str())
                                    .width(130.0)
                                    .show_ui(ui, |ui| {
                                        for n in &numeric {
                                            if ui.selectable_label(cur == n, n).clicked() {
                                                *cur = n.clone();
                                            }
                                        }
                                    });
                            };
                            ui.label("X");
                            pick(ui, "geom_x", x);
                            ui.label("Y");
                            pick(ui, "geom_y", y);
                            ui.label("EPSG");
                            ui.add(
                                egui::DragValue::new(epsg)
                                    .speed(1.0)
                                    .range(0..=99_999u32),
                            )
                            .on_hover_text(
                                "The CRS the coordinates are in. 4326 is \
                                 longitude/latitude in degrees",
                            );
                        });
                        if *epsg == 0 {
                            ui.label(
                                RichText::new(
                                    "These coordinates are outside the range of \
                                     degrees, so they are a projected grid. Give \
                                     its EPSG code — guessing 4326 would put the \
                                     data a continent away.",
                                )
                                .small()
                                .color(Color32::from_rgb(242, 140, 26)),
                            );
                        }
                        ui.label(
                            RichText::new(
                                "It becomes a layer, written as GeoParquet, not a table.",
                            )
                            .weak()
                            .small(),
                        );
                    }
                }

                ui.separator();
                let any = job.preview.plan.columns.iter().any(|c| c.include);
                let lost: usize = job
                    .preview
                    .plan
                    .columns
                    .iter()
                    .zip(&job.preview.columns)
                    .filter(|(c, _)| c.include)
                    .map(|(_, p)| p.bad)
                    .sum();
                ui.horizontal(|ui| {
                    let points = matches!(
                        job.preview.plan.geometry,
                        crate::data::attrs::GeometryPlan::Points { epsg, .. } if epsg != 0
                    );
                    let no_crs = matches!(
                        job.preview.plan.geometry,
                        crate::data::attrs::GeometryPlan::Points { epsg: 0, .. }
                    );
                    // The coordinates are read from the imported columns,
                    // so leaving one out leaves the geometry with nothing
                    // to build from.
                    let dropped_coords = match &job.preview.plan.geometry {
                        crate::data::attrs::GeometryPlan::Points { x, y, .. } => job
                            .preview
                            .plan
                            .columns
                            .iter()
                            .any(|c| (c.name == *x || c.name == *y) && !c.include),
                        crate::data::attrs::GeometryPlan::None => false,
                    };
                    let blocked = no_crs || dropped_coords;
                    let (icon, what) = if points {
                        (ph::STACK, "Import as layer")
                    } else {
                        (ph::TABLE, "Import")
                    };
                    if ui
                        .add_enabled(any && !blocked, egui::Button::new(format!("{icon} {what}")))
                        .clicked()
                    {
                        go = true;
                    }
                    if !any {
                        ui.label(RichText::new("nothing selected").weak().small());
                    } else if dropped_coords {
                        ui.label(
                            RichText::new(
                                "the X and Y columns have to be imported for the \
                                 geometry to be built from them",
                            )
                            .small()
                            .color(Color32::from_rgb(242, 140, 26)),
                        );
                    } else if lost > 0 {
                        ui.label(
                            RichText::new(format!(
                                "{} sampled value{} will become NULL",
                                fmt_count(lost),
                                if lost == 1 { "" } else { "s" },
                            ))
                            .weak()
                            .small(),
                        );
                    }
                });
            });

        // Re-reading rebuilds the column list, so it cannot happen while
        // the rows above are borrowed.
        if job.reread {
            job.reread = false;
            let (src, d, h) = (
                job.source.clone(),
                job.preview.plan.delimiter,
                job.preview.plan.has_header,
            );
            match attrs::reinspect(&src, d, h) {
                Ok(p) => job.preview = p,
                Err(e) => self.push_error(e),
            }
        } else if retype {
            let src = job.source.clone();
            if let Err(e) = attrs::recheck(&src, &mut job.preview) {
                self.push_error(e);
            }
        }

        if go {
            if let Some(job) = self.attr_import.take() {
                self.import_attr_table(job, ctx);
            }
        } else if !open {
            self.attr_import = None;
        }
    }

    /// Column names a join can key on, as SQL sees them.
    fn join_layer_columns(&self, layer_id: u64) -> Vec<String> {
        self.layers
            .iter()
            .find(|l| l.id == layer_id)
            .map(|l| {
                let store = &l.store;
                crate::sql::table::sql_column_names(&store.schema)
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| *i != store.geom_col)
                    .map(|(_, n)| n)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The SQL names of the two sides, as the console would register them.
    fn join_table_names(&self, layer_id: u64, table_id: u64) -> Option<(String, String)> {
        let (ln, tn) = crate::sql::console::sql_table_names(&self.layers, &self.attr_tables);
        let li = self.layers.iter().position(|l| l.id == layer_id)?;
        let ti = self.attr_tables.iter().position(|t| t.id == table_id)?;
        Some((ln.get(li)?.clone(), tn.get(ti)?.clone()))
    }

    /// Count how many of the layer's rows the table matches, on a worker.
    fn probe_join(&mut self, ctx: &egui::Context) {
        let Some(d) = &self.join_dialog else { return };
        let Some((lt, tt)) = self.join_table_names(d.layer_id, d.table_id) else {
            return;
        };
        let sql = crate::sql::engine::match_count_sql(&lt, &d.layer_key, &tt, &d.table_key);
        let id = self.next_join_id;
        self.next_join_id += 1;
        if let Some(d) = &mut self.join_dialog {
            d.pending = Some(id);
            d.probe = None;
        }
        let (layers, tables) = self.sql_sources();
        let egui_ctx = ctx.clone();
        crate::sql::engine::spawn_query(id, sql, layers, tables, self.join_tx.clone(), move || {
            egui_ctx.request_repaint();
        });
    }

    /// Layers and tables as SQL sees them, named the same way the console
    /// names them so a query written in either place means the same thing.
    fn sql_sources(&self) -> (Vec<crate::sql::engine::SqlLayer>, Vec<crate::sql::engine::SqlTable>) {
        use crate::sql::engine::{SqlLayer, SqlTable};
        let (ln, tn) = crate::sql::console::sql_table_names(&self.layers, &self.attr_tables);
        let layers = self
            .layers
            .iter()
            .zip(ln)
            .map(|(l, table)| SqlLayer {
                table,
                store: Arc::clone(&l.store),
                crs: l.crs.clone(),
                rg_bboxes: l
                    .rg_bboxes
                    .as_ref()
                    .filter(|r| r.boxes.len() == l.store.rg_starts().len().saturating_sub(1))
                    .map(|r| Arc::new(r.boxes.clone())),
            })
            .collect();
        let tables = self
            .attr_tables
            .iter()
            .zip(tn)
            .map(|(t, table)| SqlTable {
                table,
                schema: Arc::clone(&t.schema),
                batches: Arc::clone(&t.batches),
            })
            .collect();
        (layers, tables)
    }

    fn poll_join(&mut self, ctx: &egui::Context) {
        use crate::sql::engine::SqlDone;
        while let Ok(msg) = self.join_rx.try_recv() {
            let expected = self.join_dialog.as_ref().and_then(|d| d.pending);
            let is_running = self.join_dialog.as_ref().is_some_and(|d| d.running);
            if expected != Some(msg.id) && !is_running {
                continue;
            }
            match msg.result {
                Ok(SqlDone::Query(out)) => {
                    use arrow::array::Int64Array;
                    let get = |i: usize| {
                        out.batch
                            .column(i)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .map(|a| a.value(0))
                    };
                    if let Some(d) = &mut self.join_dialog {
                        d.pending = None;
                        d.probe = match (get(0), get(1)) {
                            (Some(t), Some(m)) => Some(Ok((t, m))),
                            _ => Some(Err("could not count matches".into())),
                        };
                    }
                }
                Ok(SqlDone::Export { path, .. }) => {
                    self.finish_join(path, ctx);
                }
                Err(e) => {
                    let running = self.join_dialog.as_ref().is_some_and(|d| d.running);
                    if running {
                        self.push_error(format!("join failed: {e}"));
                        if let Some(d) = &mut self.join_dialog {
                            d.running = false;
                        }
                    } else if let Some(d) = &mut self.join_dialog {
                        d.pending = None;
                        d.probe = Some(Err(e));
                    }
                }
            }
        }
    }

    /// The joined parquet is written: load it, and stand it in for the
    /// original when that is what was asked for.
    fn finish_join(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.temp_outputs.push(path.clone());
        let replaced = self.join_replaces.take();
        let carried = replaced.and_then(|id| {
            self.layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| (l.name.clone(), l.style.clone()))
        });
        if let Some(id) = replaced {
            self.layers.retain(|l| l.id != id);
        }
        let job = self.enqueue_load(Source::Local(path), ctx);
        if let Some((name, style)) = carried {
            // The joined copy is the same layer as far as the user is
            // concerned, so it keeps what the user gave it.
            self.pending_names.insert(job, name);
            self.pending_styles.insert(job, style);
        }
        self.join_dialog = None;
    }

    /// The join builder: two keys, the columns to bring across, and what
    /// the answer becomes.
    fn join_window(&mut self, ctx: &egui::Context) {
        use crate::sql::engine::{join_sql, JoinField};
        if self.join_dialog.is_none() {
            return;
        }
        if self.layers.is_empty() {
            self.join_dialog = None;
            self.push_error("a join needs a layer to join onto".into());
            return;
        }
        let floating_area = self.floating_area(ctx);
        let layer_names: Vec<(u64, String)> =
            self.layers.iter().map(|l| (l.id, l.name.clone())).collect();
        let d = self.join_dialog.as_ref().expect("checked");
        let table_name = self
            .attr_tables
            .iter()
            .find(|t| t.id == d.table_id)
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let layer_cols = self.join_layer_columns(d.layer_id);
        let sql_names = self.join_table_names(d.layer_id, d.table_id);

        let mut open = true;
        let mut reprobe = false;
        let mut apply = false;
        let d = self.join_dialog.as_mut().expect("checked");
        egui::Window::new(format!("Join — {table_name}"))
            .id(egui::Id::new("join_dialog"))
            .open(&mut open)
            .collapsible(false)
            .default_width(560.0)
            .constrain_to(floating_area)
            .show(ctx, |ui| {
                crate::theme::compact(ui);
                ui.horizontal(|ui| {
                    ui.label("onto layer:");
                    let current = layer_names
                        .iter()
                        .find(|(id, _)| *id == d.layer_id)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("join_layer")
                        .selected_text(current)
                        .width(220.0)
                        .show_ui(ui, |ui| {
                            for (id, name) in &layer_names {
                                if ui
                                    .selectable_label(d.layer_id == *id, name)
                                    .clicked()
                                {
                                    d.layer_id = *id;
                                    d.layer_key.clear();
                                    reprobe = true;
                                }
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("match");
                    let key_combo = |ui: &mut egui::Ui,
                                     salt: &str,
                                     cur: &mut String,
                                     opts: &[String]|
                     -> bool {
                        let mut changed = false;
                        egui::ComboBox::from_id_salt(salt)
                            .selected_text(if cur.is_empty() { "…" } else { cur.as_str() })
                            .width(170.0)
                            .show_ui(ui, |ui| {
                                for o in opts {
                                    if ui.selectable_label(cur == o, o).clicked() {
                                        *cur = o.clone();
                                        changed = true;
                                    }
                                }
                            });
                        changed
                    };
                    reprobe |= key_combo(ui, "join_lkey", &mut d.layer_key, &layer_cols);
                    ui.label("to");
                    let tcols: Vec<String> =
                        d.fields.iter().map(|(n, _)| n.clone()).collect();
                    reprobe |= key_combo(ui, "join_tkey", &mut d.table_key, &tcols);
                });

                match &d.probe {
                    Some(Ok((total, matched))) => {
                        let colour = if *matched == 0 {
                            Color32::from_rgb(220, 60, 60)
                        } else if matched < total {
                            Color32::from_rgb(242, 140, 26)
                        } else {
                            ui.visuals().weak_text_color()
                        };
                        ui.label(
                            RichText::new(format!(
                                "{} of {} features matched",
                                fmt_count(*matched as usize),
                                fmt_count(*total as usize),
                            ))
                            .small()
                            .color(colour),
                        );
                        if *matched == 0 {
                            ui.label(
                                RichText::new(
                                    "Nothing matched. The two key columns usually                                      disagree on type or on leading zeros.",
                                )
                                .small()
                                .weak(),
                            );
                        }
                    }
                    Some(Err(e)) => {
                        ui.label(RichText::new(e).small().color(ui.visuals().error_fg_color));
                    }
                    None if d.pending.is_some() => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(RichText::new("counting matches…").weak().small());
                        });
                    }
                    None => {}
                }

                ui.separator();
                ui.label(RichText::new("bring across:").weak().small());
                egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                    for (name, on) in d.fields.iter_mut() {
                        let is_key = *name == d.table_key;
                        ui.add_enabled_ui(!is_key, |ui| {
                            ui.checkbox(on, name.as_str());
                        })
                        .response
                        .on_hover_text(if is_key {
                            "the key itself is already on the layer"
                        } else {
                            ""
                        });
                    }
                });

                ui.separator();
                ui.checkbox(&mut d.keep_unmatched, "keep features with no match")
                    .on_hover_text(
                        "Their new columns are NULL. Unticked, they are dropped                          from the result entirely",
                    );
                ui.horizontal(|ui| {
                    ui.radio_value(&mut d.replace, false, "as a new layer");
                    ui.radio_value(&mut d.replace, true, "into this layer");
                });
                ui.label(
                    RichText::new(
                        "The joined result is written as a new file, geometry                          included, and loaded from it.",
                    )
                    .weak()
                    .small(),
                );

                ui.separator();
                let ready = !d.layer_key.is_empty()
                    && !d.table_key.is_empty()
                    && sql_names.is_some()
                    && !d.running;
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(ready, egui::Button::new(format!("{} Join", ph::TABLE)))
                        .clicked()
                    {
                        apply = true;
                    }
                    if d.running {
                        ui.spinner();
                        ui.label(RichText::new("joining…").weak().small());
                    }
                });
            });

        if reprobe {
            let ready = {
                let d = self.join_dialog.as_ref().expect("checked");
                !d.layer_key.is_empty() && !d.table_key.is_empty()
            };
            if ready {
                self.probe_join(ctx);
            }
        }
        if apply {
            let (sql, replace) = {
                let d = self.join_dialog.as_ref().expect("checked");
                let (lt, tt) = sql_names.clone().expect("checked by `ready`");
                let taken: Vec<String> = layer_cols.clone();
                let mut fields: Vec<JoinField> = Vec::new();
                for (name, on) in &d.fields {
                    if !*on || *name == d.table_key {
                        continue;
                    }
                    // A name the layer already uses would shadow it.
                    let mut out = name.clone();
                    let mut n = 2;
                    while taken.contains(&out) || fields.iter().any(|f| f.out == out) {
                        out = format!("{name}_{n}");
                        n += 1;
                    }
                    fields.push(JoinField {
                        source: name.clone(),
                        out,
                    });
                }
                (
                    join_sql(&lt, &d.layer_key, &tt, &d.table_key, &fields, d.keep_unmatched),
                    d.replace,
                )
            };
            let id = self.next_join_id;
            self.next_join_id += 1;
            let path = std::env::temp_dir()
                .join(format!("geopq_join_{}_{}.parquet", std::process::id(), id));
            let (layers, tables) = self.sql_sources();
            if let Some(d) = &mut self.join_dialog {
                d.running = true;
                d.pending = Some(id);
            }
            self.join_replaces = replace.then(|| {
                self.join_dialog.as_ref().map(|d| d.layer_id).unwrap_or(0)
            });
            let egui_ctx = ctx.clone();
            crate::sql::engine::spawn_export(
                id,
                sql,
                layers,
                tables,
                path,
                self.join_tx.clone(),
                move || egui_ctx.request_repaint(),
            );
        } else if !open {
            self.join_dialog = None;
        }
    }

    fn about_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
        if !self.about_open {
            return;
        }
        // App icon, decoded once on first open.
        if self.about_icon.is_none()
            && let Ok(img) =
                image::load_from_memory(include_bytes!("../assets/icon-256.png"))
        {
            let rgba = img.into_rgba8();
            let (w, h) = rgba.dimensions();
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_raw(),
            );
            self.about_icon = Some(ctx.load_texture("about_icon", ci, Default::default()));
        }
        let mut open = true;
        egui::Window::new("About GeoPQ Workbench")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(320.0)
            .constrain_to(floating_area).show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(10.0);
                    if let Some(tex) = &self.about_icon {
                        ui.add(
                            egui::Image::new(tex).fit_to_exact_size(egui::vec2(64.0, 64.0)),
                        );
                        ui.add_space(8.0);
                    }
                    ui.label(RichText::new("GeoPQ Workbench").strong().size(20.0));
                    ui.label(
                        RichText::new(concat!("version ", env!("CARGO_PKG_VERSION")))
                            .weak()
                            .small(),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        "A native workbench for GeoParquet:\n\
                         inspect, display, query and optimize.",
                    );
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(format!("© {} Geomermaids", current_year())).strong(),
                    );
                    ui.hyperlink_to("www.geomermaids.com", "https://www.geomermaids.com");
                    ui.add_space(10.0);
                });
            });
        if !open {
            self.about_open = false;
        }
    }

    /// A window screenshot arrived (File → Export image…): crop it to
    /// the map panel — the deliverable is the map, not the UI chrome —
    /// then pick a destination and write the PNG.
    fn save_screenshot(&mut self, ctx: &egui::Context) {
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        let Some(img) = shot else { return };
        let img = if self.map_rect.width() >= 1.0 {
            img.region(&self.map_rect, Some(ctx.pixels_per_point()))
        } else {
            (*img).clone()
        };
        // The frame is already captured, so it travels with the request:
        // by the time a path comes back the event carrying it is long gone.
        self.spawn_pick(PickFor::Screenshot(Box::new(img)), ctx, |d| {
            awaited_path(
                d.set_file_name("geopq-map.png")
                    .add_filter("PNG image", &["png"])
                    .save_file(),
            )
        });
    }

    fn write_screenshot(&mut self, img: &egui::ColorImage, path: &std::path::Path) {
        let [w, h] = img.size;
        let raw: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
        match image::RgbaImage::from_raw(w as u32, h as u32, raw) {
            Some(png) => {
                if let Err(e) = png.save(path) {
                    self.push_error(format!("could not save {}: {e}", path.display()));
                }
            }
            None => self.push_error("screenshot buffer size mismatch".into()),
        }
    }

    /// File → Export view to SVG…: snapshot the frame, then collect the
    /// geometry on a worker.
    ///
    /// The collection reads original geometries out of the stores, which
    /// on a remote layer is a series of ranged HTTP requests — the frame
    /// cannot wait for that. The camera of *this* frame travels with the
    /// job, so panning while the work runs cannot produce a document of a
    /// view nobody asked for.
    fn begin_svg_export(&mut self, ctx: &egui::Context) {
        if self.svg_export.is_some() {
            return;
        }
        let ppp = ctx.pixels_per_point() as f64;
        let viewport_px = [
            (self.map_rect.width() as f64 * ppp) as f32,
            (self.map_rect.height() as f64 * ppp) as f32,
        ];
        if viewport_px[0] < 1.0 || viewport_px[1] < 1.0 {
            self.push_error("the map panel has no area to export".into());
            return;
        }
        let layers: Vec<SvgLayerJob> = self
            .layers
            .iter()
            // A rebuilding layer is off the map (its mesh is in the old
            // projection); a consolidating one stays, as in the frame.
            .filter(|l| {
                l.style.visible
                    && (!self.rebuilding.contains(&l.id)
                        || self.consolidating.contains(&l.id))
            })
            .map(|l| SvgLayerJob {
                name: l.name.clone(),
                style: resolve_style(&l.style),
                style_by: l.style.style_by.clone(),
                sections: l
                    .sections
                    .iter()
                    .map(|s| (Arc::clone(&s.chunks), Arc::clone(&s.rtree)))
                    .collect(),
                store: Arc::clone(&l.store),
                crs: l.crs.clone(),
                loaded: l.loaded.clone(),
                decimated: l
                    .loaded
                    .iter()
                    .any(|g| matches!(g, crate::data::layer::GroupLoad::Preview { .. })),
                attribution: l.info.attribution.as_ref().map(|a| a.credit.clone()),
            })
            .collect();
        let dark = ctx.theme() == egui::Theme::Dark;
        let job = SvgJob {
            layers,
            display: self.display.clone(),
            camera: self.camera,
            viewport_px,
            pixels_per_point: ppp,
            view_world: self.last_view_world,
            dark,
            graticule: self.show_graticule,
            coastline: self.show_coastline.then_some(self.coast_level),
        };
        let (tx, rx) = channel();
        let egui_ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(build_svg(job));
            egui_ctx.request_repaint();
        });
        self.svg_export = Some(rx);
    }

    /// A collected SVG document goes on to the save dialog.
    fn poll_svg_export(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.svg_export else { return };
        // Only one file panel at a time: leave the document in the channel
        // rather than taking it and having `spawn_pick` drop it on the
        // floor, which would lose a collection that may have taken
        // seconds of network reads.
        if self.pick_dialog.is_some() {
            return;
        }
        let out = match rx.try_recv() {
            Ok(v) => v,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            // The worker died: reopen the slot or the menu entry stays
            // disabled for the rest of the session.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.svg_export = None;
                return;
            }
        };
        self.svg_export = None;
        for w in out.warnings {
            self.push_error(w);
        }
        let name = out.name;
        self.spawn_pick(PickFor::ExportSvg(out.doc), ctx, move |d| {
            awaited_path(
                d.set_file_name(name)
                    .add_filter("SVG image", &["svg"])
                    .add_filter("Compressed SVG", &["svgz"])
                    .save_file(),
            )
        });
    }

    /// Quality-gate dialog (docs/OPEN_POLICY.md): a non-indexable file too
    /// big for a full build waits here for Optimize / Load all / Cancel.
    fn quality_gate_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
        use crate::data::info::fmt_bytes;
        use crate::data::quality::{DIRECT_MAX_GEOM_BYTES, DIRECT_MAX_ROWS};
        let Some(gate) = self.quality_gates.first() else {
            return;
        };
        let info = &gate.opened.info;
        let name = gate.opened.store.source.name();
        let (rows, geom_bytes) = (
            info.rows,
            info.quality.as_ref().map_or(0, |q| q.geom_bytes),
        );
        let too_big = rows > DIRECT_MAX_ROWS || geom_bytes > DIRECT_MAX_GEOM_BYTES;
        enum Action {
            None,
            Optimize,
            LoadAll,
            Cancel,
        }
        let mut action = Action::None;
        egui::Window::new(format!("File not optimized — {name}"))
            .id(egui::Id::new("quality_gate").with(gate.job))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(520.0)
            .constrain_to(floating_area).show(ctx, |ui| {
                ui.label(format!(
                    "{} rows in {} row groups — {}",
                    fmt_count(rows as usize),
                    info.row_groups,
                    fmt_bytes(info.file_size),
                ));
                ui.add_space(6.0);
                if let Some(q) = &info.quality {
                    for c in q.failures() {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new(ph::X_CIRCLE)
                                    .color(Color32::from_rgb(220, 60, 60))
                                    .strong(),
                            );
                            ui.label(&c.detail);
                        });
                    }
                }
                ui.add_space(6.0);
                ui.label(
                    "Viewport-based loading is not possible on this file: the \
                     viewer cannot tell which rows are on screen without \
                     decoding them.",
                );
                ui.add_space(4.0);
                ui.label(format!(
                    "Optimize rewrites it once (spatial sort + bbox index) and \
                     opens the optimized copy. Load all decodes every feature \
                     up front (~{} of geometry) — slower to open, complete \
                     and exact after that.",
                    fmt_bytes(geom_bytes),
                ));
                if gate.opened.store.source.is_remote() {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(
                            "Remote source: either choice downloads effectively \
                             the whole file.",
                        )
                        .color(Color32::from_rgb(242, 140, 26)),
                    );
                }
                if too_big {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(format!(
                            "Too large to load in full ({} rows / {} geometry — \
                             limits {} / {}). Optimize is the only path.",
                            fmt_count(rows as usize),
                            fmt_bytes(geom_bytes),
                            fmt_count(DIRECT_MAX_ROWS as usize),
                            fmt_bytes(DIRECT_MAX_GEOM_BYTES),
                        ))
                        .color(Color32::from_rgb(220, 60, 60)),
                    );
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Optimize…").clicked() {
                        action = Action::Optimize;
                    }
                    if ui
                        .add_enabled(
                            !too_big,
                            egui::Button::new(format!(
                                "Load all {}",
                                fmt_count(rows as usize)
                            )),
                        )
                        .on_hover_text("Remembered for this file: it will load fully without asking again")
                        .clicked()
                    {
                        action = Action::LoadAll;
                    }
                    if ui.button("Cancel").clicked() {
                        action = Action::Cancel;
                    }
                });
            });
        match action {
            Action::None => {}
            Action::LoadAll => {
                let gate = self.quality_gates.remove(0);
                self.direct_files.insert(gate.key());
                save_direct_files(&self.direct_files);
                self.resume_gated(gate, ctx);
            }
            Action::Cancel => {
                let gate = self.quality_gates.remove(0);
                self.drop_gated(&gate, ctx);
            }
            Action::Optimize => {
                let gate = self.quality_gates.remove(0);
                if self.optimize.as_ref().is_none_or(|o| !o.running) {
                    let o = &gate.opened;
                    let recommended = crate::data::optimize::GpVersion::preferred(
                        &o.info.geo.geometry_types,
                    );
                    self.optimize = Some(OptimizeState {
                        layer_id: u64::MAX, // no layer behind this export
                        layer_name: o.store.source.name(),
                        src: o.store.source.clone(),
                        epsg: o.crs.epsg,
                        crs: o.crs.clone(),
                        viewport_only: false,
                        opts: crate::data::optimize::OptimizeOptions {
                            xy_geom: o.store.xy_geom,
                            version: recommended,
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
                        open_result: true,
                        recommended,
                        dest_s3: false,
                        stac: true,
                        replace_remote: false,
                        s3_uri: String::new(),
                        s3_endpoint: String::new(),
                        s3_profile: None,
                        s3_profiles: crate::data::source::aws::profiles(),
                        report_s3: None,
                        merge_with: Default::default(),
                        merge_source_col: true,
                        upload_as_is: false,
                        report_as_is: None,
                    });
                }
                self.drop_gated(&gate, ctx);
            }
        }
    }

    fn info_window(&mut self, ctx: &egui::Context) {
        let floating_area = self.floating_area(ctx);
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
            .constrain_to(floating_area).show(ctx, |ui| {
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
                        // Remote layers: what this file has actually cost
                        // over the wire. Range count matters as much as
                        // the volume — against object storage the latency
                        // of each request usually dominates the transfer.
                        let fetched: Option<(u64, u64)> = layer
                            .store
                            .fragments
                            .iter()
                            .filter_map(|f| crate::data::net::for_source(&f.source.url()?))
                            .reduce(|a, b| (a.0 + b.0, a.1 + b.1));
                        if let Some((bytes, reqs)) = fetched {
                            row(
                                ui,
                                "downloaded",
                                format!(
                                    "{} in {} range requests",
                                    fmt_bytes(bytes),
                                    fmt_count(reqs as usize)
                                ),
                            );
                        }
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
                                        "(poorly clustered: consider Export…)"
                                    } else {
                                        "(well clustered)"
                                    }
                                ),
                            );
                        }
                    });

                    if let Some(q) = &info.quality {
                        ui.add_space(8.0);
                        ui.separator();
                        Self::quality_scorecard(ui, q);
                    }

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

                    if let Some(a) = &info.attribution {
                        egui::CollapsingHeader::new("Attribution")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label(RichText::new(&a.credit).strong());
                                if ui.small_button("Copy").clicked() {
                                    ui.ctx().copy_text(a.text.clone());
                                }
                                // The full notice: licences routinely ask
                                // for a citation or a link, and the user
                                // needs to be able to read and copy it.
                                ui.add(
                                    egui::TextEdit::multiline(&mut a.text.as_str())
                                        .font(egui::TextStyle::Monospace)
                                        .desired_width(f32::INFINITY),
                                );
                            });
                    }

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

    /// Display-readiness scorecard (docs/OPEN_POLICY.md): one line per
    /// check, verdict on top.
    fn quality_scorecard(ui: &mut egui::Ui, q: &crate::data::quality::QualityReport) {
        use crate::data::quality::Status;
        let green = Color32::from_rgb(80, 200, 120);
        let amber = Color32::from_rgb(242, 140, 26);
        let red = Color32::from_rgb(220, 60, 60);
        let (verdict, color) = if q.indexable {
            ("Display readiness: optimized — viewport loading active", green)
        } else {
            (
                "Display readiness: not optimized — viewport loading unavailable",
                red,
            )
        };
        ui.label(RichText::new(verdict).strong().color(color));
        ui.add_space(4.0);
        egui::Grid::new("quality_grid").num_columns(3).striped(true).show(ui, |ui| {
            for c in &q.checks {
                let (icon, color) = match c.status {
                    Status::Pass => (ph::CHECK_CIRCLE, green),
                    Status::Warn => (ph::WARNING_CIRCLE, amber),
                    Status::Fail => (ph::X_CIRCLE, red),
                };
                ui.label(RichText::new(icon).color(color).strong());
                ui.label(RichText::new(format!("{} {}", c.code, c.title)).strong());
                ui.add(egui::Label::new(&c.detail).wrap());
                ui.end_row();
            }
        });
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
        let hidden_name = layer
            .store
            .hidden_wkb
            .map(|i| layer.store.schema.field(i).name().clone());
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
                                if Some(field.name()) == hidden_name.as_ref() {
                                    continue; // display decodes the GeoArrow sibling
                                }
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
        let floating_area = self.floating_area(ctx);
        if !self.show_errors || self.errors.is_empty() {
            return;
        }
        let mut open = self.show_errors;
        egui::Window::new("Problems")
            .open(&mut open)
            .default_width(420.0)
            .constrain_to(floating_area).show(ctx, |ui| {
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

    /// Live network use, data and basemap counted apart.
    ///
    /// Which stream is moving is the first thing worth knowing when a
    /// remote layer feels slow: a busy tile stream and an idle data one
    /// says the basemap is the holdup, not the dataset.
    fn network_readout(&mut self, ui: &mut egui::Ui) {
        use crate::data::info::fmt_bytes;
        use crate::data::net::{self, Channel};
        if !net::any_traffic() {
            return;
        }
        let (data_bytes, data_reqs) = net::totals(Channel::Data);
        let (tile_bytes, tile_reqs) = net::totals(Channel::Tiles);
        let (data_rate, tile_rate) = (net::rate(Channel::Data), net::rate(Channel::Tiles));
        let live = data_rate + tile_rate > 0.0;
        let text = if live {
            let mut parts: Vec<String> = Vec::new();
            if data_rate > 0.0 {
                parts.push(format!("data {}/s", fmt_bytes(data_rate as u64)));
            }
            if tile_rate > 0.0 {
                parts.push(format!("tiles {}/s", fmt_bytes(tile_rate as u64)));
            }
            format!("· ↓ {}", parts.join(" "))
        } else {
            format!("· ↓ {} total", fmt_bytes(data_bytes + tile_bytes))
        };
        let label = if live {
            RichText::new(text).color(Color32::from_rgb(90, 160, 210))
        } else {
            RichText::new(text).weak()
        };
        ui.monospace(label).on_hover_text(format!(
            "downloaded this session\n\
             data: {} in {} range requests\n\
             basemap: {} in {} tiles",
            fmt_bytes(data_bytes),
            fmt_count(data_reqs as usize),
            fmt_bytes(tile_bytes),
            fmt_count(tile_reqs as usize),
        ));
    }

    /// How the basemap is drawn for the current view, if at all.
    ///
    /// Tiles are Web Mercator, so any other display projection has to warp
    /// them. That is fine for the imagery and wrong for the text baked into
    /// it, which is why a labelled source falls back to its label-free twin
    /// once the view deforms enough for a place name to look tilted.
    fn basemap_plan(&self, viewport_px: [f32; 2]) -> BasemapPlan {
        use crate::map::warp;
        let Some(src) = self.basemap else {
            return BasemapPlan::Off(None);
        };
        if self.display.is_mercator() {
            return BasemapPlan::Mercator(src);
        }
        let w = warp::Warp::new(&self.display);
        let plan = match warp::plan(
            &w,
            &self.camera,
            viewport_px,
            TILE_SOURCES[src].max_zoom,
        ) {
            Ok(p) => p,
            Err(e) => return BasemapPlan::Off(Some(e.reason())),
        };
        if !TILE_SOURCES[src].labels || plan.labels_survive() {
            return BasemapPlan::Warped(src, plan);
        }
        match crate::map::tiles::nolabels_twin(src) {
            Some(twin) => BasemapPlan::Warped(twin, plan),
            None => BasemapPlan::Off(Some(
                "no tiles: this source draws its place names into the pixels, \
                 which this projection would shear. Pick a \"no labels\" source.",
            )),
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(w) = self.cursor_world {
                if let Some((lon, lat)) = world_to_lonlat(&self.display, w) {
                    ui.monospace(format!("{lon:.6}, {lat:.6}"));
                }
                if !self.display.crs.is_latlong {
                    // No CRS name here: some run long enough to push the
                    // rest of the status bar off, and the projection
                    // selector in the corner already names it.
                    let (x, y) = self.display.projected_from_world(w);
                    ui.monospace(format!("| {x:.1}, {y:.1}"));
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
                self.network_readout(ui);
                if !self.errors.is_empty() {
                    let btn = egui::Button::new(
                        RichText::new(format!("⚠ {}", self.errors.len()))
                            .color(Color32::from_rgb(220, 60, 60)),
                    );
                    if ui.add(btn).clicked() {
                        self.show_errors = !self.show_errors;
                    }
                }
                if let Some(what) = &self.attr_busy {
                    ui.spinner();
                    ui.label(RichText::new(what).weak());
                }
                if self.svg_export.is_some() {
                    ui.spinner();
                    ui.label(RichText::new("collecting the view for SVG").weak());
                }
                use crate::data::info::fmt_bytes;
                for d in &self.downloads {
                    if ui
                        .small_button("✖")
                        .on_hover_text("Stop this download")
                        .clicked()
                    {
                        d.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    // An export endpoint that builds the file on the fly
                    // states no length, so there is a count but no bar to
                    // fill: a fraction of an unknown total would be a lie.
                    let (frac, text) = match d.total {
                        Some(t) => (
                            d.got as f32 / t as f32,
                            format!("{} — {} / {}", d.label, fmt_bytes(d.got), fmt_bytes(t)),
                        ),
                        None => (0.0, format!("{} — {}", d.label, fmt_bytes(d.got))),
                    };
                    ui.add(
                        egui::ProgressBar::new(frac).desired_width(220.0).text(text),
                    );
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
        self.map_rect = rect;
        let ppp = ctx.pixels_per_point();
        let vp = [rect.width() * ppp, rect.height() * ppp];

        // --- input ---
        if response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let d = response.drag_delta();
            if d != egui::Vec2::ZERO {
                self.camera_moved = true;
            }
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
                self.camera_moved = true;
            }
        }
        if response.double_clicked() {
            if let Some(cursor) = hover_px {
                self.camera.zoom_about(1.0, cursor, vp);
                self.camera_moved = true;
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
            self.camera_moved = true;
            self.pending_fit = false;
        }
        // Provisional framing wins over the empty-map fit that set_display
        // arms, which would otherwise pull the camera back out to the whole
        // projection and leave the basemap with nothing to fetch again.
        if let Some(b) = self.frame_bounds.take() {
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
                self.cam_epoch += 1;
                self.refine_hold.clear();
                self.part_hold.clear();
                self.refine_deferred.clear();
                // A moved viewport obsoletes in-flight refinements:
                // cancel them so the new view's check starts at settle
                // instead of queueing behind downloads for a viewport
                // nobody is looking at. Batches already landed stay.
                for c in self.append_cancel.values() {
                    c.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            } else if now - self.cam_changed_at > 0.35 {
                self.refine_partial_layers(&ctx);
            }
        }

        // --- background ---
        let dark = ui.visuals().dark_mode;
        let bg = map_background(dark);
        ui.painter().rect_filled(rect, 0.0, bg);

        // --- build draw call ---
        self.tiles.poll();
        let basemap = self.basemap_plan(vp);
        self.last_basemap_plan = basemap;
        let tile_draws = match basemap {
            BasemapPlan::Mercator(src) => self.tiles.draws(src, &self.camera, vp),
            BasemapPlan::Warped(src, plan) => {
                let warp = crate::map::warp::Warp::new(&self.display);
                let epoch = self.graticule_generation;
                self.tiles.draws_warped(src, &plan, &warp, epoch)
            }
            BasemapPlan::Off(_) => Vec::new(),
        };
        let tile_uploads = self.tiles.take_uploads();
        let alive_tiles = self.tiles.alive_keys();
        let alive_layers: std::collections::HashSet<(u64, u64)> = self
            .layers
            .iter()
            .flat_map(|l| {
                (0..l.sections.len())
                    .map(|si| (section_key(l.id, si), l.draw_gen))
                    .chain(std::iter::once((RG_OVERLAY_BASE | l.id, l.generation)))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut draws: Vec<LayerDraw> = Vec::new();
        if self.show_graticule && !self.graticule_chunks.is_empty() {
            draws.push(LayerDraw {
                key: (GRATICULE_KEY, self.graticule_generation),
                composite_group: GRATICULE_KEY,
                chunks: self.graticule_chunks.clone(),
                style: graticule_style(dark),
            });
        }
        if self.show_coastline && !self.coastline_chunks.is_empty() {
            draws.push(LayerDraw {
                key: (COASTLINE_KEY, self.graticule_generation),
                composite_group: COASTLINE_KEY,
                chunks: self.coastline_chunks.clone(),
                style: coastline_style(dark),
            });
        }
        for l in &self.layers {
            // A rebuilding layer is hidden because its mesh is in the
            // previous projection's coordinates and would draw in the
            // wrong place. A consolidation changes no coordinates — it
            // only merges sections — so its old mesh stays correct and
            // stays on screen until the new one lands.
            let hidden = self.rebuilding.contains(&l.id)
                && !self.consolidating.contains(&l.id);
            if !l.style.visible || hidden {
                continue;
            }
            for (si, section) in l.sections.iter().enumerate() {
                draws.push(LayerDraw {
                    key: (section_key(l.id, si), l.draw_gen),
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
                    point_shape: crate::data::layer::PointShape::Circle,
                    bin_colors: None,
                    hidden_bins: 0,
                    ..Default::default()
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
                    point_shape: crate::data::layer::PointShape::Circle,
                    bin_colors: None,
                    hidden_bins: 0,
                    ..Default::default()
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
                    point_shape: crate::data::layer::PointShape::Circle,
                    bin_colors: None,
                    hidden_bins: 0,
                    ..Default::default()
                },
            });
        }

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            MapCallback {
                camera: self.camera,
                viewport_px: vp,
                tile_opacity: self.basemap_opacity,
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
        // Credits, bottom right: the basemap's, then one per visible
        // layer that asks for one. Licences like CC BY want the credit
        // where the data is seen, not buried in a dialog.
        let mut credits: Vec<&str> = Vec::new();
        // The source actually drawn, which outside Mercator may be the
        // label-free twin of the one selected.
        if let Some(src) = basemap.drawn_source() {
            credits.push(TILE_SOURCES[src].attribution);
        }
        for l in &self.layers {
            if !l.style.visible {
                continue;
            }
            let Some(a) = l.info.attribution.as_ref() else {
                continue;
            };
            if !credits.contains(&a.credit.as_str()) {
                credits.push(&a.credit);
            }
        }
        if !credits.is_empty() {
            let (fg, bg) = credit_colors(dark);
            let font = egui::FontId::proportional(10.0);
            const PAD: egui::Vec2 = egui::vec2(4.0, 1.0);
            // Stacked upwards so a long credit never runs off the side.
            let mut y = rect.bottom() - 4.0;
            for c in credits.iter().rev() {
                let g = ui.painter().layout_no_wrap((*c).to_string(), font.clone(), fg);
                let size = g.size();
                let pos = egui::pos2(rect.right() - 6.0 - size.x, y - size.y);
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(pos, size).expand2(PAD),
                    2.0,
                    bg,
                );
                ui.painter().galley(pos, g, fg);
                // Clear the plate, plus a hair so stacked ones stay apart.
                y -= size.y + 2.0 * PAD.y + 1.0;
            }
        }
    }
}

/// The add-repository row. Open-data portals are not added here: they
/// have their own dialog, File → Data catalogs.
fn add_repo_row(ui: &mut egui::Ui, b: &mut RepoBrowser, refetch: &mut bool) {
    use crate::data::repo::{self, RepoKind};
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
        let url = b.add.1.trim().trim_end_matches('/').to_string();
        let http = url.starts_with("https://") || url.starts_with("http://");
        if ui
            .add_enabled(
                http && !b.add.0.trim().is_empty(),
                egui::Button::new("Add repository"),
            )
            .clicked()
        {
            // A URL pasted with its catalog.json marks a STAC
            // repository; the base is its directory.
            let (url, kind) = match url.strip_suffix("/catalog.json") {
                Some(base) => (base.to_string(), RepoKind::Stac),
                None => (url.clone(), RepoKind::Parquetry),
            };
            b.repos.push(repo::Repository {
                name: b.add.0.trim().to_string(),
                url,
                kind,
                // A user-added repository credits itself from its own
                // data, if at all.
                attribution: None,
                attribution_by_license: Default::default(),
            });
            b.add = (String::new(), String::new());
            b.sel_repo = b.repos.len() - 1;
            b.snapshots = vec![repo::Snapshot::latest()];
            b.sel_snapshot = 0;
            *refetch = true;
            if let Err(e) = repo::save_repos(&b.repos) {
                log::warn!("saving repositories: {e}");
            }
        }
        if b.repos.len() > 1 && ui.button("Remove current").clicked() {
            b.repos.remove(b.sel_repo);
            b.sel_repo = 0;
            if let Err(e) = repo::save_repos(&b.repos) {
                log::warn!("saving repositories: {e}");
            }
            b.snapshots = vec![repo::Snapshot::latest()];
            b.sel_snapshot = 0;
            *refetch = true;
        }
    });
}

/// The dataset list of a DCAT portal: a search box over what the catalog
/// says in words, one row per dataset with the formats it publishes, and
/// the count of entries no format here can open.
///
/// Flat and full width, unlike the two-pane repository view: a portal
/// dataset has nothing below it to expand into — the row is the dataset.
fn dcat_pane(ui: &mut egui::Ui, b: &mut CatalogBrowser, open: &mut Vec<usize>) {
    let cat = match &b.dcat {
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("reading the portal catalog…").weak());
            });
            return;
        }
        Some(Err(e)) => {
            ui.label(RichText::new(e).color(Color32::from_rgb(220, 60, 60)));
            return;
        }
        Some(Ok(c)) => c,
    };
    if cat.datasets.is_empty() {
        ui.label(RichText::new(format!(
            "none of this catalog's {} datasets publish a format this app can \
             open — GeoParquet, GeoPackage, GeoJSON or CSV. Map services and \
             web pages are all it offers.",
            cat.hidden
        )));
        return;
    }
    if cat.truncated {
        ui.label(
            RichText::new(
                "the portal cuts its catalog feed off mid-stream — only the \
                 datasets before the cut are listed",
            )
            .color(Color32::from_rgb(200, 140, 40))
            .small(),
        );
    }
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut b.filter)
                .hint_text("search titles, keywords and descriptions…")
                .desired_width((ui.available_width() - 110.0).max(120.0)),
        );
        ui.checkbox(&mut b.geo_only, "geo formats").on_hover_text(
            "Only datasets that open as a layer — GeoParquet, GeoPackage or \
             GeoJSON. A CSV-only dataset is an attribute table, not a map.",
        );
    });
    let needle = b.filter.to_lowercase();
    let geo_only = b.geo_only;
    let matches = |d: &crate::data::repo::DcatDataset| {
        let text = needle.is_empty()
            || d.title.to_lowercase().contains(&needle)
            || d.description.to_lowercase().contains(&needle)
            || d.keywords.iter().any(|k| k.to_lowercase().contains(&needle));
        let geo = !geo_only
            || d.distributions
                .iter()
                .any(|x| x.format != crate::data::repo::DcatFormat::Csv);
        text && geo
    };
    let mut shown = 0usize;
    // Whatever height is left, minus room for the Open row and the
    // hidden-count line below: they must never slip under the border.
    let list_height = (ui.available_height() - 84.0).clamp(140.0, 340.0);
    egui::ScrollArea::vertical()
        .id_salt("dcat_datasets")
        .max_height(list_height)
        .show(ui, |ui| {
            for (i, d) in cat.datasets.iter().enumerate() {
                if !matches(d) {
                    continue;
                }
                shown += 1;
                ui.horizontal(|ui| {
                    let mut on = b.dcat_checked.contains(&i);
                    if ui.checkbox(&mut on, &d.title).changed() {
                        if on {
                            b.dcat_checked.insert(i);
                        } else {
                            b.dcat_checked.remove(&i);
                        }
                    }
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if let Some(m) = &d.modified {
                                ui.label(RichText::new(m).weak().small());
                            }
                            let badges: Vec<&str> =
                                d.distributions.iter().map(|x| x.format.label()).collect();
                            ui.label(RichText::new(badges.join(" · ")).weak().small())
                                .on_hover_text(format!(
                                    "opens as {} — {}",
                                    d.distributions[0].format.label(),
                                    d.distributions[0].url,
                                ));
                        },
                    );
                })
                .response
                .on_hover_text(dcat_hover(d));
            }
        });
    if shown == 0 {
        ui.label(RichText::new("no dataset matches the search").weak());
    }
    ui.horizontal(|ui| {
        let n = b.dcat_checked.len();
        if ui
            .add_enabled(
                n > 0,
                egui::Button::new(format!(
                    "Open {n} dataset{}",
                    if n == 1 { "" } else { "s" }
                )),
            )
            .clicked()
        {
            let mut picked: Vec<usize> = b.dcat_checked.iter().copied().collect();
            picked.sort_unstable();
            open.extend(picked);
            b.dcat_checked.clear();
        }
        if ui.small_button("none").clicked() {
            b.dcat_checked.clear();
        }
        ui.label(
            RichText::new(
                "each opens as its best format: GeoPackage and GeoJSON are \
                 downloaded and imported, parquet and CSV are read where they are",
            )
            .weak()
            .small(),
        );
    });
    if cat.hidden > 0 {
        ui.label(
            RichText::new(format!(
                "{} more without an openable format",
                cat.hidden
            ))
            .weak()
            .small(),
        )
        .on_hover_text(
            "Datasets published only as a map service, a web page, a KML or a \
             zipped shapefile. Nothing in this app extracts archives yet.",
        );
    }
}

/// Everything a portal dataset says about itself, for the row's tooltip.
fn dcat_hover(d: &crate::data::repo::DcatDataset) -> String {
    let mut out = String::new();
    if !d.description.is_empty() {
        let desc: String = d.description.chars().take(400).collect();
        out.push_str(&desc);
        if d.description.chars().count() > 400 {
            out.push('…');
        }
        out.push_str("\n\n");
    }
    if let Some(p) = &d.publisher {
        out.push_str(&format!("Publisher: {p}\n"));
    }
    if let Some(l) = &d.license {
        out.push_str(&format!("License: {l}\n"));
    }
    if let Some(b) = &d.bbox {
        out.push_str(&format!(
            "Extent: {:.4}, {:.4} → {:.4}, {:.4}\n",
            b[0], b[1], b[2], b[3]
        ));
    }
    if !d.keywords.is_empty() {
        out.push_str(&format!("Keywords: {}\n", d.keywords.join(", ")));
    }
    out.trim_end().to_string()
}

/// Text and plate colours for the credit strip, as `(text, plate)`.
///
/// The credits sit over the map, not over a panel, so the theme says
/// nothing about what is behind them: light text on a white basemap
/// disappeared into it. The plate decides that background itself, and the
/// text can go to full contrast now that it has one. Both stay
/// translucent so the strip reads as an overlay rather than a widget,
/// which is why the colours are picked here and contrast-tested rather
/// than eyeballed.
fn credit_colors(dark: bool) -> (Color32, Color32) {
    if dark {
        (Color32::from_white_alpha(200), Color32::from_black_alpha(180))
    } else {
        (Color32::from_black_alpha(200), Color32::from_white_alpha(200))
    }
}

/// Overlay stroke styles. Named here rather than inlined in the frame
/// because the SVG export draws the same overlays and has to agree with
/// the map on what they look like.
fn graticule_style(dark: bool) -> DrawStyle {
    let g = if dark { 0.42 } else { 0.55 };
    DrawStyle {
        fill_color: [0.0; 4],
        line_color: [g, g, g, 0.45],
        point_color: [0.0; 4],
        line_half_width_px: 0.4,
        point_radius_px: 0.0,
        point_shape: crate::data::layer::PointShape::Circle,
        bin_colors: None,
        hidden_bins: 0,
        ..Default::default()
    }
}

fn coastline_style(dark: bool) -> DrawStyle {
    let c = if dark {
        [0.62, 0.66, 0.70]
    } else {
        [0.35, 0.38, 0.42]
    };
    DrawStyle {
        fill_color: [0.0; 4],
        line_color: [c[0], c[1], c[2], 0.85],
        point_color: [0.0; 4],
        line_half_width_px: 0.5,
        point_radius_px: 0.0,
        point_shape: crate::data::layer::PointShape::Circle,
        bin_colors: None,
        hidden_bins: 0,
        ..Default::default()
    }
}

/// The map's background, which is also the SVG's.
fn map_background(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(24, 24, 28)
    } else {
        Color32::from_rgb(244, 243, 240)
    }
}

/// Build graticule line meshes (meridians/parallels every 15°) for the given
/// display projection. Cheap: a few thousand densified vertices.
fn build_graticule(display: &DisplayCrs) -> Arc<Vec<crate::data::geometry::ChunkMesh>> {
    let wgs = Crs::wgs84();
    let tr = BulkTransformer::new(&wgs, display);
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

    for line in graticule_latlon(display) {
        add_line(&line);
    }
    drop(add_line);
    Arc::new(mb.finish())
}

/// The graticule as WGS84 polylines: meridians and parallels every 15°,
/// densified every 2° so they stay curves through any projection.
/// Mercator's ±85° cut applies to the meridians (past it the projection
/// runs to infinity).
fn graticule_latlon(display: &DisplayCrs) -> Vec<Vec<(f64, f64)>> {
    let max_lat: f64 = if display.is_mercator() { 85.0 } else { 90.0 };
    let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
    for lon_i in (-180..=180).step_by(15) {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut lat = -max_lat;
        while lat <= max_lat + 1e-9 {
            pts.push((lon_i as f64, lat));
            lat += 2.0;
        }
        out.push(pts);
    }
    for lat_i in (-75..=75).step_by(15) {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut lon = -180.0;
        while lon <= 180.0 + 1e-9 {
            pts.push((lon, lat_i as f64));
            lon += 2.0;
        }
        out.push(pts);
    }
    out
}

// ----------------------------------------------------------------------
// SVG export: frame snapshot → collected scene
// ----------------------------------------------------------------------

/// A layer section as the export sees it: the meshes for point
/// instances, the R-tree for everything else.
type SvgSection = (
    Arc<Vec<crate::data::geometry::ChunkMesh>>,
    Arc<rstar::RTree<crate::data::layer::PickItem>>,
);

/// Everything the export reads off one layer, cloned on the frame thread
/// (all of it `Arc`s or small values) so the collection can run on a
/// worker — `fetch_geoms` on a remote layer is network I/O.
struct SvgLayerJob {
    name: String,
    style: DrawStyle,
    style_by: Option<crate::data::layer::StyleBy>,
    sections: Vec<SvgSection>,
    store: Arc<crate::data::store::FeatureStore>,
    crs: Crs,
    /// Decode state per row group: groups drawn from covering boxes must
    /// export as boxes, because boxes are what the screen shows.
    loaded: Vec<crate::data::layer::GroupLoad>,
    /// Some group is on screen as a stride preview.
    decimated: bool,
    attribution: Option<String>,
}

/// One export request, complete: the frame's camera and view settings
/// travel with it so a pan while it runs cannot change the result.
struct SvgJob {
    layers: Vec<SvgLayerJob>,
    display: DisplayCrs,
    camera: Camera,
    viewport_px: [f32; 2],
    pixels_per_point: f64,
    view_world: [f64; 4],
    dark: bool,
    graticule: bool,
    /// Coastline level when the overlay is on.
    coastline: Option<crate::data::coastline::CoastLevel>,
}

/// A finished document on its way back to the frame.
struct SvgExport {
    doc: String,
    /// Default file name, from the first visible layer.
    name: String,
    /// Layers whose geometry could not be read. Reported to the user
    /// rather than swallowed: a missing layer in a figure is worse than
    /// an error dialog.
    warnings: Vec<String>,
}

/// Collect the visible scene and render it. Blocking — worker only.
fn build_svg(job: SvgJob) -> SvgExport {
    use crate::map::svg::{SvgLayer, SvgScene};

    let mut notes = vec![
        "The raster basemap is not part of this file: tiles are images, \
         not vectors."
            .to_string(),
    ];
    let mut warnings: Vec<String> = Vec::new();
    let mut layers: Vec<SvgLayer> = Vec::new();

    // Overlays are rebuilt from their sources rather than read back out
    // of the render chunks: the chunks hold LOD-simplified segments, and
    // the polylines are what the map draws at full detail.
    if job.graticule {
        let lines = graticule_latlon(&job.display);
        let mut l = SvgLayer::new("graticule", graticule_style(job.dark));
        l.features = polyline_features(crate::data::coastline::project_overlay_polylines(
            &job.display,
            lines.iter().map(|p| p.as_slice()),
        ));
        layers.push(l);
    }
    if let Some(level) = job.coastline {
        use crate::data::coastline::{detailed_lines, project_overlay_polylines, CoastLevel};
        let detail = (level == CoastLevel::Detailed).then(detailed_lines).flatten();
        let projected = match &detail {
            Some(lines) => {
                project_overlay_polylines(&job.display, lines.iter().map(|l| l.as_slice()))
            }
            None => project_overlay_polylines(
                &job.display,
                crate::data::coastline::coastline_lines()
                    .iter()
                    .map(|l| l.as_slice()),
            ),
        };
        let mut l = SvgLayer::new("coastline", coastline_style(job.dark));
        l.features = polyline_features(projected);
        layers.push(l);
    }

    let mut credits: Vec<String> = Vec::new();
    let name = job
        .layers
        .first()
        .map(|l| svg_file_name(&l.name))
        .unwrap_or_else(|| "geopq-map.svg".to_string());
    for layer in &job.layers {
        if let Some(a) = &layer.attribution
            && !credits.contains(a)
        {
            credits.push(a.clone());
        }
        if layer.decimated {
            notes.push(format!(
                "Layer \"{}\" was drawn from a decimated preview on screen; \
                 the export carries the same rows, not the whole dataset.",
                layer.name
            ));
        }
        match collect_svg_layer(layer, &job) {
            Ok(l) => layers.push(l),
            Err(e) => {
                warnings.push(format!("SVG export, layer {}: {e}", layer.name));
                notes.push(format!(
                    "Layer \"{}\" is missing from this file: {e}",
                    layer.name
                ));
            }
        }
    }

    let scene = SvgScene {
        camera: job.camera,
        viewport_px: job.viewport_px,
        pixels_per_point: job.pixels_per_point,
        background: map_background(job.dark),
        credit_colors: credit_colors(job.dark),
        layers,
        credits,
        notes,
    };
    SvgExport {
        doc: crate::map::svg::render(&scene),
        name,
        warnings,
    }
}

/// A layer name turned into a file name: the stem, plus `.svg`.
fn svg_file_name(layer: &str) -> String {
    let stem: String = layer
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(layer)
        .trim_end_matches(".parquet")
        .chars()
        .map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' })
        .collect();
    let stem = stem.trim_matches('_');
    if stem.is_empty() {
        "geopq-map.svg".to_string()
    } else {
        format!("{stem}.svg")
    }
}

fn polyline_features(
    lines: Vec<geo_types::LineString<f64>>,
) -> Vec<crate::map::svg::SvgFeature> {
    lines
        .into_iter()
        .map(|ls| crate::map::svg::SvgFeature {
            geom: geo_types::Geometry::LineString(ls),
            bin: 0,
            underlay: false,
        })
        .collect()
}

/// Collect one layer's visible content, following the pick path: the
/// R-tree says which features are in view, the store hands back their
/// original geometry, and the bins are recomputed with the same rules the
/// build used. The tessellated meshes are never touched — they are
/// triangles, and this is a vector export.
fn collect_svg_layer(
    job: &SvgLayerJob,
    ctx: &SvgJob,
) -> Result<crate::map::svg::SvgLayer, String> {
    use crate::data::geometry::spans_underlay;
    use crate::map::svg::{SvgFeature, SvgLayer};

    let v = ctx.view_world;
    let view = [v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3])];
    let env = rstar::AABB::from_corners([view[0], view[1]], [view[2], view[3]]);

    let mut items: Vec<(u32, [f64; 4])> = job
        .sections
        .iter()
        .flat_map(|(_, rtree)| {
            rtree
                .locate_in_envelope_intersecting(env)
                .map(|i| (i.feature.index, i.bbox))
        })
        .collect();
    items.sort_by_key(|(i, _)| *i);
    items.dedup_by_key(|(i, _)| *i);

    // Groups drawn from covering boxes: their features are on screen as
    // rectangles, so that is what goes in the file. The box is the
    // R-tree entry's own bbox, already in world coordinates.
    let starts = job.store.rg_starts();
    let boxed: Vec<bool> = items
        .iter()
        .map(|(row, _)| {
            let g = starts.partition_point(|s| *s <= *row as u64).saturating_sub(1);
            matches!(
                job.loaded.get(g),
                Some(crate::data::layer::GroupLoad::Boxes { .. })
            )
        })
        .collect();

    let rows: Vec<u32> = items.iter().map(|(r, _)| *r).collect();
    let geom_rows: Vec<u32> = rows
        .iter()
        .zip(&boxed)
        .filter(|(_, b)| !**b)
        .map(|(r, _)| *r)
        .collect();
    let mut geoms: HashMap<u32, geo_types::Geometry<f64>> = HashMap::new();
    if !geom_rows.is_empty() {
        for (row, g) in job.store.fetch_geoms(&geom_rows)? {
            if let Some(g) = g {
                geoms.insert(row, g);
            }
        }
    }

    let mut features: Vec<SvgFeature> = Vec::with_capacity(items.len());
    let bins = svg_bins(job, &rows, &geoms)?;
    for (i, (row, bbox)) in items.iter().enumerate() {
        let bin = bins.as_ref().map(|b| b[i]).unwrap_or(0);
        let geom = if boxed[i] {
            geo_types::Geometry::Rect(geo_types::Rect::new(
                geo_types::Coord { x: bbox[0], y: bbox[1] },
                geo_types::Coord { x: bbox[2], y: bbox[3] },
            ))
        } else {
            match geoms.remove(row) {
                Some(g) => picking::to_world_geom(g, &job.crs, &ctx.display),
                None => continue,
            }
        };
        features.push(SvgFeature {
            geom,
            bin,
            underlay: spans_underlay(*bbox),
        });
    }

    // Points carry no R-tree entries: they live in the chunk instance
    // buffers, which is also where the marker pass reads them. The
    // renderer's per-chunk decimation is deliberately not reproduced —
    // it exists to keep a frame cheap, and a file has no frame budget.
    let mut points: Vec<([f64; 2], u8)> = Vec::new();
    let reach = job.style.point_shape.reach() as f64;
    let margin = job.style.point_radius_px as f64 * reach / ctx.camera.scale();
    for (chunks, _) in &job.sections {
        for chunk in chunks.iter() {
            if chunk.point_instances.is_empty() {
                continue;
            }
            let (o, b) = (chunk.origin, chunk.bounds_local);
            if o[0] + b[2] as f64 + margin < view[0]
                || o[0] + b[0] as f64 - margin > view[2]
                || o[1] + b[3] as f64 + margin < view[1]
                || o[1] + b[1] as f64 - margin > view[3]
            {
                continue;
            }
            for p in &chunk.point_instances {
                let w = [o[0] + p[0] as f64, o[1] + p[1] as f64];
                if w[0] + margin >= view[0]
                    && w[0] - margin <= view[2]
                    && w[1] + margin >= view[1]
                    && w[1] - margin <= view[3]
                {
                    points.push((w, chunk.bin));
                }
            }
        }
    }

    Ok(SvgLayer {
        name: job.name.clone(),
        style: job.style.clone(),
        features,
        points,
    })
}

/// Style bins for the exported rows, recomputed with the build's rules
/// (`batch_bins` / `norm_bin`) rather than read back off the chunks: a
/// chunk's bin covers all its features, and this needs one per feature.
/// None when the layer is not data-styled.
fn svg_bins(
    job: &SvgLayerJob,
    rows: &[u32],
    geoms: &HashMap<u32, geo_types::Geometry<f64>>,
) -> Result<Option<Vec<u8>>, String> {
    let Some(sb) = &job.style_by else {
        return Ok(None);
    };
    let Some(sel) = loader::resolve_style(&job.store, sb) else {
        return Ok(None);
    };
    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let batches = job.store.fetch(rows, Some(&[sel.col]))?;
    let mut out: Vec<u8> = Vec::with_capacity(rows.len());
    match (&sel.binning, sel.per_area) {
        (loader::Binning::Breaks(breaks), true) => {
            let mut it = rows.iter();
            for batch in &batches {
                let vals = loader::batch_values(batch.column(0));
                for v in vals {
                    let row = it.next().ok_or("row/batch count mismatch")?;
                    // A feature whose geometry never arrived (a covering
                    // box, an undecodable row) has no area; norm_bin
                    // sends a zero area to bin 0, as the build does.
                    let area = geoms
                        .get(row)
                        .map(|g| loader::ground_area(g, job.crs.is_latlong))
                        .unwrap_or(0.0);
                    out.push(loader::norm_bin(v, area, breaks));
                }
            }
        }
        (binning, _) => {
            for batch in &batches {
                out.extend(loader::batch_bins(batch.column(0), binning));
            }
        }
    }
    if out.len() != rows.len() {
        return Err(format!(
            "style column returned {} values for {} rows",
            out.len(),
            rows.len()
        ));
    }
    Ok(Some(out))
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
/// The class list a categorical style will draw: swatch, label, and the
/// catch-all bin.
fn category_preview(ui: &mut egui::Ui, mode: &crate::data::layer::StyleMode) {
    use crate::data::layer::{palette_color, StyleMode, STYLE_BINS};
    let StyleMode::Categorical {
        values,
        colors,
        labels,
    } = mode
    else {
        return;
    };
    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
        for (i, v) in values.iter().enumerate() {
            ui.horizontal(|ui| {
                let c = match colors {
                    Some(m) if i < m.len() => Color32::from_rgb(m[i][0], m[i][1], m[i][2]),
                    _ => palette_color(i),
                };
                let label = match labels {
                    Some(l) if i < l.len() => &l[i],
                    _ => v,
                };
                let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 2.0, c);
                ui.label(label);
            });
        }
        if values.len() < STYLE_BINS - 1 {
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 2.0, Color32::from_gray(140));
                ui.label(RichText::new("(other)").weak());
            });
        }
    });
}

/// Rows, saturated to soft: Tableau 10, CARTO Vivid, ColorBrewer Set1,
/// Dark2, Set2, Pastel1, CARTO Antique (map-friendly earth tones).
const SWATCH_ROWS: &[(&str, &[[u8; 3]])] = &[
    ("Tableau", &[
        [0x4E, 0x79, 0xA7], [0xF2, 0x8E, 0x2B], [0xE1, 0x57, 0x59], [0x76, 0xB7, 0xB2],
        [0x59, 0xA1, 0x4F], [0xED, 0xC9, 0x48], [0xB0, 0x7A, 0xA1], [0xFF, 0x9D, 0xA7],
        [0x9C, 0x75, 0x5F], [0xBA, 0xB0, 0xAC],
    ]),
    ("Vivid", &[
        [0xE5, 0x86, 0x06], [0x5D, 0x69, 0xB1], [0x52, 0xBC, 0xA3], [0x99, 0xC9, 0x45],
        [0xCC, 0x61, 0xB0], [0x24, 0x79, 0x6C], [0xDA, 0xA5, 0x1B], [0x2F, 0x8A, 0xC4],
        [0x76, 0x4E, 0x9F], [0xED, 0x64, 0x5A],
    ]),
    ("Bold", &[
        [0xE4, 0x1A, 0x1C], [0x37, 0x7E, 0xB8], [0x4D, 0xAF, 0x4A], [0x98, 0x4E, 0xA3],
        [0xFF, 0x7F, 0x00], [0xFF, 0xFF, 0x33], [0xA6, 0x56, 0x28], [0xF7, 0x81, 0xBF],
        [0x99, 0x99, 0x99],
    ]),
    ("Dark", &[
        [0x1B, 0x9E, 0x77], [0xD9, 0x5F, 0x02], [0x75, 0x70, 0xB3], [0xE7, 0x29, 0x8A],
        [0x66, 0xA6, 0x1E], [0xE6, 0xAB, 0x02], [0xA6, 0x76, 0x1D], [0x66, 0x66, 0x66],
    ]),
    ("Soft", &[
        [0x66, 0xC2, 0xA5], [0xFC, 0x8D, 0x62], [0x8D, 0xA0, 0xCB], [0xE7, 0x8A, 0xC3],
        [0xA6, 0xD8, 0x54], [0xFF, 0xD9, 0x2F], [0xE5, 0xC4, 0x94], [0xB3, 0xB3, 0xB3],
    ]),
    ("Pastel", &[
        [0xFB, 0xB4, 0xAE], [0xB3, 0xCD, 0xE3], [0xCC, 0xEB, 0xC5], [0xDE, 0xCB, 0xE4],
        [0xFE, 0xD9, 0xA6], [0xFF, 0xFF, 0xCC], [0xE5, 0xD8, 0xBD], [0xFD, 0xDA, 0xEC],
    ]),
    ("Antique", &[
        [0x85, 0x5C, 0x75], [0xD9, 0xAF, 0x6B], [0xAF, 0x64, 0x58], [0x73, 0x6F, 0x4C],
        [0x52, 0x6A, 0x83], [0x62, 0x53, 0x77], [0x68, 0x85, 0x5C], [0x9C, 0x9C, 0x5E],
        [0xA0, 0x61, 0x77], [0x8C, 0x78, 0x5D],
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

/// Paint one point marker. Shapes and their area-matched sizing mirror
/// `sd_marker` in shaders.wgsl, so a picker entry previews what the map
/// will actually draw.
fn paint_marker(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    shape: crate::data::layer::PointShape,
    color: Color32,
) {
    use crate::data::layer::PointShape;
    let poly = |pts: Vec<egui::Pos2>| egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE);
    // n vertices on a circle of radius r, first one at angle `start`.
    let ring = |n: usize, r: f32, start: f32| -> Vec<egui::Pos2> {
        (0..n)
            .map(|i| {
                let a = start + std::f32::consts::TAU * i as f32 / n as f32;
                center + egui::vec2(r * a.cos(), r * a.sin())
            })
            .collect()
    };
    let up = -std::f32::consts::FRAC_PI_2;
    // reach() is the circumradius, which is what a regular polygon is
    // built from; the square is the inscribed one.
    let r = radius * shape.reach();
    match shape {
        PointShape::Circle => {
            painter.circle_filled(center, r, color);
        }
        PointShape::Square => {
            painter.rect_filled(
                egui::Rect::from_center_size(center, egui::Vec2::splat(r * std::f32::consts::SQRT_2)),
                0.0,
                color,
            );
        }
        PointShape::Triangle => {
            painter.add(poly(ring(3, r, up)));
        }
        PointShape::Diamond => {
            painter.add(poly(ring(4, r, up)));
        }
        PointShape::Hexagon => {
            painter.add(poly(ring(6, r, up)));
        }
        PointShape::Star => {
            // Concave: egui fills a closed path as a fan, which a star
            // would tear. Build it from a convex core plus its points.
            let outer = ring(5, r, up);
            let inner = ring(5, r * 0.5, up + std::f32::consts::TAU / 10.0);
            painter.add(poly(inner.clone()));
            for i in 0..5 {
                painter.add(poly(vec![inner[(i + 4) % 5], outer[i], inner[i]]));
            }
        }
    }
}

/// Stroke a horizontal dash preview the way the line pass draws it:
/// each dash a slab, the cap's shape at both of its ends. Mirrors
/// `fs_line` in `shaders.wgsl` so the picker previews what ships.
fn paint_line_style(
    painter: &egui::Painter,
    rect: egui::Rect,
    pattern: crate::data::layer::LinePattern,
    cap: crate::data::layer::LineCap,
    width: f32,
    color: Color32,
) {
    use crate::data::layer::LineCap;
    let y = rect.center().y;
    let h = width * 0.5;
    let (x0, x1) = (rect.left() + 2.0, rect.right() - 2.0);
    let slab = |a: f32, b: f32| {
        egui::Rect::from_min_max(egui::pos2(a, y - h), egui::pos2(b, y + h))
    };
    let draw = |a: f32, b: f32| {
        let (a, b) = (a.max(x0), b.min(x1));
        if b < a {
            return;
        }
        match cap {
            LineCap::Flat => {
                painter.rect_filled(slab(a, b), 0.0, color);
            }
            LineCap::Square => {
                painter.rect_filled(slab(a - h, b + h), 0.0, color);
            }
            LineCap::Round => {
                if b > a {
                    painter.rect_filled(slab(a, b), 0.0, color);
                }
                painter.circle_filled(egui::pos2(a, y), h, color);
                painter.circle_filled(egui::pos2(b, y), h, color);
            }
        }
    };
    let d = pattern.dashes_px(cap, width);
    if d[0] < 0.0 {
        draw(x0, x1);
        return;
    }
    let period = (d[0] + d[1] + d[2] + d[3]).max(1.0);
    let mut x = x0;
    while x < x1 {
        draw(x, x + d[0]);
        if d[2] > 0.0 || d[3] > 0.0 {
            draw(x + d[0] + d[1], x + d[0] + d[1] + d[2]);
        }
        x += period;
    }
}

/// Line style picker: a dash-preview button opening the pattern and cap
/// lists. Unlike the marker picker it stays open on selection, so both
/// properties can be set in one visit. Returns true when either changed.
fn line_style_button(
    ui: &mut egui::Ui,
    id_salt: &str,
    pattern: &mut crate::data::layer::LinePattern,
    cap: &mut crate::data::layer::LineCap,
    color: Color32,
) -> bool {
    use crate::data::layer::{LineCap, LinePattern};
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(26.0, 14.0), egui::Sense::click());
    let resp = resp.on_hover_text("dash pattern and line caps");
    paint_line_style(
        ui.painter(),
        rect,
        *pattern,
        *cap,
        3.0,
        ui.style().interact(&resp).fg_stroke.color,
    );
    let mut changed = false;
    egui::Popup::from_toggle_button_response(&resp)
        .id(egui::Id::new(("line_style_popup", id_salt)))
        .kind(egui::PopupKind::Popup)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(3.0, 2.0);
            let row = |ui: &mut egui::Ui,
                           selected: bool,
                           preview_pattern: LinePattern,
                           preview_cap: LineCap,
                           preview_w: f32,
                           label: &str|
             -> bool {
                let (r, hit) =
                    ui.allocate_exact_size(egui::vec2(118.0, 20.0), egui::Sense::click());
                let hit = hit.on_hover_cursor(egui::CursorIcon::PointingHand);
                if selected || hit.hovered() {
                    let bg = if selected {
                        ui.visuals().selection.bg_fill
                    } else {
                        ui.visuals().widgets.hovered.bg_fill
                    };
                    ui.painter().rect_filled(r, 3.0, bg);
                }
                let strip =
                    egui::Rect::from_min_max(r.min + egui::vec2(4.0, 0.0), egui::pos2(r.left() + 48.0, r.max.y));
                paint_line_style(ui.painter(), strip, preview_pattern, preview_cap, preview_w, color);
                ui.painter().text(
                    egui::pos2(r.left() + 54.0, r.center().y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::proportional(11.0),
                    ui.visuals().text_color(),
                );
                hit.clicked()
            };
            for p in LinePattern::ALL {
                if row(ui, *pattern == p, p, *cap, 3.0, p.label()) {
                    *pattern = p;
                    changed = true;
                }
            }
            ui.separator();
            for c in LineCap::ALL {
                // A fat solid stub: the one preview where caps differ.
                if row(ui, *cap == c, LinePattern::Solid, c, 7.0, c.label()) {
                    *cap = c;
                    changed = true;
                }
            }
        });
    changed
}

/// Point-symbol picker: a glyph button opening the shape list.
/// Returns true when the shape changed.
fn marker_shape_button(
    ui: &mut egui::Ui,
    id_salt: &str,
    shape: &mut crate::data::layer::PointShape,
    color: Color32,
) -> bool {
    use crate::data::layer::PointShape;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(16.0, 14.0), egui::Sense::click());
    let resp = resp.on_hover_text("point symbol");
    paint_marker(
        ui.painter(),
        rect.center(),
        4.5,
        *shape,
        ui.style().interact(&resp).fg_stroke.color,
    );
    let mut changed = false;
    egui::Popup::from_toggle_button_response(&resp)
        .id(egui::Id::new(("marker_popup", id_salt)))
        .kind(egui::PopupKind::Popup)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(3.0, 2.0);
            for s in PointShape::ALL {
                let (r, hit) =
                    ui.allocate_exact_size(egui::vec2(96.0, 20.0), egui::Sense::click());
                let hit = hit.on_hover_cursor(egui::CursorIcon::PointingHand);
                if *shape == s || hit.hovered() {
                    let bg = if *shape == s {
                        ui.visuals().selection.bg_fill
                    } else {
                        ui.visuals().widgets.hovered.bg_fill
                    };
                    ui.painter().rect_filled(r, 3.0, bg);
                }
                let p = ui.painter();
                paint_marker(p, egui::pos2(r.left() + 13.0, r.center().y), 6.0, s, color);
                p.text(
                    egui::pos2(r.left() + 28.0, r.center().y),
                    egui::Align2::LEFT_CENTER,
                    s.label(),
                    egui::FontId::proportional(11.0),
                    ui.visuals().text_color(),
                );
                if hit.clicked() {
                    *shape = s;
                    changed = true;
                    ui.close();
                }
            }
        });
    changed
}

/// Compact class bound: `sig` significant digits with k/M/G suffixes
/// (42200 → "42.2k", 718000 → "718k", 0.00123 → "0.00123").
fn fmt_sig(v: f64, sig: usize) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v:.0}");
    }
    let a = v.abs();
    let (scaled, suffix) = if a >= 1e9 {
        (v / 1e9, "G")
    } else if a >= 1e6 {
        (v / 1e6, "M")
    } else if a >= 1e3 {
        (v / 1e3, "k")
    } else {
        (v, "")
    };
    let e = scaled.abs().log10().floor() as i64;
    let dec = (sig as i64 - 1 - e).max(0) as usize;
    let s = format!("{scaled:.dec$}");
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    format!("{s}{suffix}")
}

/// Legend labels for the break values: rounded to 3 significant digits,
/// with precision raised until adjacent distinct breaks keep distinct
/// labels (never show "40k – 40k" for a real interval).
fn fmt_break_labels(breaks: &[f64]) -> Vec<String> {
    for sig in 3..=8 {
        let out: Vec<String> = breaks.iter().map(|b| fmt_sig(*b, sig)).collect();
        let ok = breaks
            .windows(2)
            .zip(out.windows(2))
            .all(|(bv, bl)| bv[0] == bv[1] || bl[0] != bl[1]);
        if ok {
            return out;
        }
    }
    breaks.iter().map(|b| fmt_sig(*b, 8)).collect()
}

/// Compact legend for a data-styled layer in the layers panel: one
/// swatch per class with its value boundaries (or category value).
/// Clicking an entry toggles that class on the map (draw-time filter).
fn style_legend(
    ui: &mut egui::Ui,
    layer_id: u64,
    sb: &mut crate::data::layer::StyleBy,
    fill_alpha: f32,
    area_unit: &str,
) -> bool {
    use crate::data::layer::{StyleMode, STYLE_BINS};
    let mut reclass = false;
    // Swatches composite exactly like map fills: class color at the
    // layer's effective fill alpha over the map background, so the
    // legend matches what the map actually shows.
    let bg = if ui.visuals().dark_mode {
        Color32::from_rgb(24, 24, 28)
    } else {
        Color32::from_rgb(244, 243, 240)
    };
    let a = fill_alpha.clamp(0.0, 1.0);
    let blend = move |c: Color32| -> Color32 {
        Color32::from_rgb(
            (c.r() as f32 * a + bg.r() as f32 * (1.0 - a)).round() as u8,
            (c.g() as f32 * a + bg.g() as f32 * (1.0 - a)).round() as u8,
            (c.b() as f32 * a + bg.b() as f32 * (1.0 - a)).round() as u8,
        )
    };
    let title = if sb.per_area {
        format!("legend — {} / {area_unit}", sb.column)
    } else {
        format!("legend — {}", sb.column)
    };
    egui::CollapsingHeader::new(RichText::new(title).weak().small())
    .id_salt(("style_legend", layer_id))
    .default_open(true)
    .show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        if matches!(sb.mode, StyleMode::Graduated { .. })
            && ui
                .small_button("⟳ viewport")
                .on_hover_text(
                    "Reclassify: recompute the class breaks from the data under \
                     the current viewport (row groups intersecting it), then \
                     re-render. Classes keep this stretch until you reclassify \
                     again.",
                )
                .clicked()
        {
            reclass = true;
        }
        let mut toggle: Option<u8> = None;
        let mut swatch = |ui: &mut egui::Ui, bin: u8, c: Color32, label: String| {
            let hidden = sb.hidden_bins & (1u64 << bin) != 0;
            let row = ui
                .horizontal(|ui| {
                    let (r, _) = ui
                        .allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    let c = blend(c);
                    ui.painter()
                        .rect_filled(r, 2.0, if hidden { c.gamma_multiply(0.2) } else { c });
                    let mut text = RichText::new(label).small();
                    if hidden {
                        text = text.weak().strikethrough();
                    }
                    ui.label(text);
                })
                .response
                .interact(egui::Sense::click());
            if row
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text("click to hide/show this class")
                .clicked()
            {
                toggle = Some(bin);
            }
        };
        match &sb.mode {
            StyleMode::Graduated { breaks, .. } => {
                if breaks.is_empty() {
                    return;
                }
                let colors = sb.bin_colors();
                let labels = fmt_break_labels(breaks);
                let n = breaks.len() + 1;
                for i in 0..n {
                    let c = colors[i.min(colors.len() - 1)];
                    let c = Color32::from_rgb(
                        (c[0] * 255.0) as u8,
                        (c[1] * 255.0) as u8,
                        (c[2] * 255.0) as u8,
                    );
                    let label = if i == 0 {
                        format!("< {}", labels[0])
                    } else if i == n - 1 {
                        format!("≥ {}", labels[n - 2])
                    } else {
                        format!("{} – {}", labels[i - 1], labels[i])
                    };
                    swatch(ui, i as u8, c, label);
                }
            }
            StyleMode::Categorical {
                values,
                colors,
                labels,
            } => {
                for (i, v) in values.iter().enumerate() {
                    let c = match colors {
                        Some(m) if i < m.len() => {
                            Color32::from_rgb(m[i][0], m[i][1], m[i][2])
                        }
                        _ => crate::data::layer::palette_color(i),
                    };
                    let label = match labels {
                        Some(l) if i < l.len() => l[i].clone(),
                        _ => v.clone(),
                    };
                    swatch(ui, i as u8, c, label);
                }
                swatch(
                    ui,
                    (STYLE_BINS - 1) as u8,
                    Color32::from_gray(140),
                    "(other)".into(),
                );
            }
        }
        if let Some(bin) = toggle {
            sb.hidden_bins ^= 1u64 << bin;
        }
    });
    reclass
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
        fill_color: [r, g, b, if s.fill_on { s.fill_opacity * s.opacity } else { 0.0 }],
        line_color: [
            lc.r(),
            lc.g(),
            lc.b(),
            if s.lines_on { lc.a() * s.opacity } else { 0.0 },
        ],
        point_color: [r, g, b, s.opacity],
        line_half_width_px: (s.line_width_px * 0.5).max(0.01),
        line_pattern: s.line_pattern,
        line_cap: s.line_cap,
        point_radius_px: s.point_radius_px.max(0.1),
        point_shape: s.point_shape,
        bin_colors: s.style_by.as_ref().map(|sb| Arc::new(sb.bin_colors())),
        bin_half_widths: s
            .style_by
            .as_ref()
            .and_then(|sb| sb.bin_widths())
            .map(|w| Arc::new(w.map(|x| (x * 0.5).max(0.01)))),
        hidden_bins: s.style_by.as_ref().map(|sb| sb.hidden_bins).unwrap_or(0),
        ..Default::default()
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

#[cfg(test)]
mod year_tests {
    #[test]
    fn current_year_is_sane() {
        let y = super::current_year();
        assert!((2026..2200).contains(&y), "{y}");
    }
}

/// Civil year from the system clock, no date dependency (days-from-epoch
/// conversion, Howard Hinnant's algorithm). Good enough for a © line.
fn current_year() -> i64 {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let y = yoe + era * 400;
    if doy >= 306 { y + 1 } else { y }
}

/// Settings sidecar for the quality gate's decline memory: one small
/// JSON file in the home directory (the app has no other persistence).
/// Unknown keys are preserved for forward compatibility.
pub(crate) fn settings_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".geopq-workbench.json"))
}

fn load_direct_files() -> HashSet<String> {
    let read = || -> Option<HashSet<String>> {
        let txt = std::fs::read_to_string(settings_path()?).ok()?;
        let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
        serde_json::from_value(v.get("direct_files")?.clone()).ok()
    };
    read().unwrap_or_default()
}

fn save_direct_files(files: &HashSet<String>) {
    let Some(p) = settings_path() else { return };
    let mut root = std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    root["direct_files"] = serde_json::json!(files);
    if let Ok(txt) = serde_json::to_string_pretty(&root) {
        if let Err(e) = std::fs::write(&p, txt) {
            log::warn!("could not save {}: {e}", p.display());
        }
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
                let key = (section_key(l.id, si), l.draw_gen);
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
    fn on_exit(&mut self) {
        for p in self.temp_outputs.drain(..) {
            let _ = std::fs::remove_file(&p);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_loader(&ctx);
        self.poll_optimizer(&ctx);
        self.update_coastline_detail(&ctx);
        self.poll_picks();
        self.poll_repo();
        self.poll_catalog();
        self.poll_downloads(&ctx);
        self.poll_categories();
        self.poll_classes();
        self.poll_viewport_reclass(&ctx);
        self.strip_uploaded_cpu_meshes(frame);
        if self.strip_probe > 0 {
            self.strip_probe -= 1;
            ctx.request_repaint();
        }

        // Drag & drop.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        for p in dropped {
            let importable = crate::data::import::ImportFormat::from_path(&p).is_some();
            let tabular =
                !p.is_dir() && crate::data::attrs::is_tabular(&Source::Local(p.clone()));
            if !p.is_dir() && importable {
                self.begin_import(p, &ctx);
            } else if tabular {
                self.open_attr_table(Source::Local(p));
            } else {
                let src = if p.is_dir() { Source::Dir(p) } else { Source::Local(p) };
                self.enqueue_load(src, &ctx);
            }
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
                    if let Some(a) =
                        self.sql.panel_ui(ui, &self.layers, &self.attr_tables, view_world, &display)
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
        if self.layers_open {
            egui::Panel::left("layers")
                .resizable(true)
                .default_size(260.0)
                .show(ui, |ui| self.layers_panel(ui));
        } else {
            egui::Panel::left("layers_collapsed")
                .resizable(false)
                .exact_size(18.0)
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    if ui
                        .small_button(ph::CARET_DOUBLE_RIGHT)
                        .on_hover_text("Show the layers panel")
                        .clicked()
                    {
                        self.layers_open = true;
                    }
                });
        }
        if self.selection.is_some() {
            // Floating over the map (upper right) instead of a side panel:
            // opening it must not resize the viewport.
            let map_corner =
                ui.available_rect_before_wrap().right_top() + egui::vec2(-12.0, 12.0);
            let floating_area = self.floating_area(&ctx);
            let mut open = true;
            egui::Window::new("Feature")
                .id(egui::Id::new("feature_attrs"))
                .open(&mut open)
                .pivot(egui::Align2::RIGHT_TOP)
                .default_pos(map_corner)
                .default_width(300.0)
                .resizable(true)
                .collapsible(false)
                .constrain_to(floating_area).show(&ctx, |ui| self.attributes_panel(ui));
            if !open {
                self.clear_selection();
            }
        }
        self.errors_window(&ctx);
        self.info_window(&ctx);
        self.quality_gate_window(&ctx);
        self.about_window(&ctx);
        self.reset_confirm_window(&ctx);
        self.poll_attrs(&ctx);
        self.attr_import_window(&ctx);
        self.poll_join(&ctx);
        self.join_window(&ctx);
        self.poll_pick(&ctx);
        let cookbook_area = self.floating_area(&ctx);
        crate::cookbook::window(&ctx, &mut self.cookbook_open, cookbook_area);
        self.save_screenshot(&ctx);
        self.poll_svg_export(&ctx);
        // Desktop-standard shortcuts: Cmd/Ctrl+O opens files, Ctrl+Q quits
        // (macOS handles ⌘Q natively).
        let open_files = ctx.input_mut(|i| {
            i.consume_key(egui::Modifiers::COMMAND, egui::Key::O)
        });
        if open_files {
            self.spawn_pick(PickFor::OpenFiles, &ctx, pick_parquet_files);
        }
        if !cfg!(target_os = "macos")
            && ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Q))
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.optimize_window(&ctx);
        self.gpkg_import_window(&ctx);
        self.grid_window(&ctx);
        self.url_window(&ctx);
        self.repo_window(&ctx);
        self.catalog_window(&ctx);
        self.style_window(&ctx);
        self.poll_filters(&ctx);
        self.filter_window(&ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.map_panel(ui));

        if self.attr_busy.is_some()
            || !self.loading.is_empty()
            || !self.rebuilding.is_empty()
            || self.sql.is_running()
            || !self.filter_pending.is_empty()
            || self.filter_dialog.as_ref().is_some_and(|d| d.testing)
            || self.svg_export.is_some()
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

    /// No native file dialog may be opened from the frame.
    ///
    /// macOS runs these panels with `runModal`, a nested event loop that
    /// delivers every OS event arriving while the panel is up. Opening one
    /// from `update` — inside winit's event handler — means the next event
    /// re-enters that handler, and winit aborts the process rather than
    /// unwinding: dropping a file on the window while the Open panel was up
    /// killed the app. `spawn_pick` is the only place allowed to build one,
    /// and it does so on a worker thread.
    ///
    /// A source-level check because there is no type that can express it:
    /// any call site added later is a crash that only shows up in the hands
    /// of whoever drags a file at the wrong moment.
    #[test]
    fn file_dialogs_are_only_built_off_the_frame_thread() {
        // Everything above the test modules: this test names the pattern
        // in its own assertions and would otherwise count itself.
        let src = include_str!("app.rs")
            .split_once("\n#[cfg(test)]")
            .expect("a test module")
            .0;
        let sites = src.matches("rfd::AsyncFileDialog::new()").count();
        assert_eq!(
            sites, 1,
            "rfd::AsyncFileDialog::new() must appear once, inside spawn_pick",
        );
        let spawn = src
            .split_once("fn spawn_pick")
            .expect("spawn_pick")
            .1
            .split_once("\n    fn ")
            .expect("end of spawn_pick")
            .0;
        assert!(
            spawn.contains("std::thread::spawn"),
            "spawn_pick must run the dialog on its own thread",
        );
        assert!(
            spawn.contains("rfd::AsyncFileDialog::new()"),
            "the one construction site must be the one inside spawn_pick",
        );
    }

    /// Credits are legible on whatever the map puts behind them.
    ///
    /// They used to be drawn straight onto the map in a colour chosen from
    /// the app theme, so the dark theme's light text vanished over a white
    /// basemap. The plate under them has to be opaque enough to fix that
    /// for both themes over both extremes of map, and the extremes are the
    /// worst cases: anything mid-grey behind is easier than these.
    #[test]
    fn credits_stay_legible_over_any_basemap() {
        /// Composite a translucent colour over an opaque grey level.
        /// `Color32` is premultiplied, so the source contributes its
        /// channel as-is rather than scaled by alpha again.
        fn over(c: Color32, under: f32) -> f32 {
            let a = c.a() as f32 / 255.0;
            // Colours here are white- or black-alpha, so any channel does.
            c.r() as f32 + under * (1.0 - a)
        }
        /// WCAG relative luminance of a grey level, then contrast ratio.
        fn lum(v: f32) -> f32 {
            let s = v / 255.0;
            if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        }
        fn ratio(a: f32, b: f32) -> f32 {
            let (hi, lo) = (lum(a).max(lum(b)), lum(a).min(lum(b)));
            (hi + 0.05) / (lo + 0.05)
        }

        for dark in [false, true] {
            let (fg, bg) = credit_colors(dark);
            for map in [0.0f32, 255.0] {
                let plate = over(bg, map);
                let text = over(fg, plate);
                let r = ratio(text, plate);
                assert!(
                    r >= 4.5,
                    "dark={dark} over map={map}: contrast {r:.2} \
                     (text {text:.0} on plate {plate:.0}), want >= 4.5",
                );
            }
        }
    }

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
        let Some((device, queue)) = crate::map::renderer::test_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
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
            tile_opacity: 1.0,
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
                        point_shape: crate::data::layer::PointShape::Circle,
                        bin_colors: None,
                        hidden_bins: 0,
                        ..Default::default()
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
                        point_shape: crate::data::layer::PointShape::Circle,
                        bin_colors: None,
                        hidden_bins: 0,
                        ..Default::default()
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

#[cfg(test)]
mod legend_fmt_tests {
    use super::{fmt_break_labels, fmt_sig};

    #[test]
    fn sig_formatting_and_collision_guard() {
        assert_eq!(fmt_sig(42200.0, 3), "42.2k");
        assert_eq!(fmt_sig(718000.0, 3), "718k");
        assert_eq!(fmt_sig(16188500.0, 3), "16.2M");
        assert_eq!(fmt_sig(3200.0, 3), "3.2k");
        assert_eq!(fmt_sig(0.0, 3), "0");
        assert_eq!(fmt_sig(12.3456, 3), "12.3");
        assert_eq!(fmt_sig(0.00123, 3), "0.00123");
        assert_eq!(fmt_sig(-42200.0, 3), "-42.2k");
        // Close but distinct breaks force extra precision…
        let l = fmt_break_labels(&[42210.0, 42260.0, 99000.0]);
        assert_ne!(l[0], l[1], "{l:?}");
        // …while genuinely equal breaks (quantile dupes) may share one.
        let l = fmt_break_labels(&[0.0, 0.0, 3200.0]);
        assert_eq!(l[0], l[1]);
    }
}

#[cfg(test)]
mod svg_export_tests {
    use super::*;
    use crate::data::layer::{GroupLoad, Ramp, StyleBy, StyleMode};
    use arrow::array::{ArrayRef, BinaryArray, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::fs::File;

    /// Four unit squares in a row at lon 0..8, each carrying a value that
    /// puts it in its own class, plus one point beside each of them.
    fn write_squares(path: &std::path::Path) {
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon", "Point"],
                "bbox": [0.0, 0.0, 10.0, 2.0],
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let mut wkbs: Vec<Vec<u8>> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for i in 0..4 {
            let x = i as f64 * 2.0;
            let ring = geo_types::LineString::from(vec![
                (x, 0.0),
                (x + 1.0, 0.0),
                (x + 1.0, 1.0),
                (x, 1.0),
                (x, 0.0),
            ]);
            let poly = geo_types::Geometry::Polygon(geo_types::Polygon::new(ring, vec![]));
            wkbs.push(crate::data::import::to_wkb(&poly).unwrap());
            values.push(i as f64 * 10.0);
        }
        for i in 0..4 {
            let p = geo_types::Geometry::Point(geo_types::Point::new(i as f64 * 2.0 + 0.5, 1.5));
            wkbs.push(crate::data::import::to_wkb(&p).unwrap());
            values.push(i as f64 * 10.0);
        }
        let cols: Vec<ArrayRef> = vec![
            Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
            Arc::new(Float64Array::from(values)),
        ];
        let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn styled_by_value() -> StyleBy {
        StyleBy {
            column: "value".into(),
            mode: StyleMode::Graduated {
                breaks: vec![5.0, 15.0, 25.0],
                method: crate::data::layer::ClassMethod::EqualInterval,
            },
            ramp: Ramp::Viridis,
            per_area: false,
            hidden_bins: 0,
            classified_rows: None,
            width_px: None,
        }
    }

    /// The class colours as hex, straight from the ramp: what the map
    /// paints, so what the file must carry.
    fn class_colors() -> Vec<String> {
        let lut = styled_by_value().bin_colors();
        (0..4)
            .map(|i| {
                let c = lut[i];
                let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
            })
            .collect()
    }

    /// The collector end to end: a real parquet through the R-tree, the
    /// store and the binning, out as a document that is parsed back.
    fn export(hidden_bins: u64) -> String {
        // One file per call: these tests run concurrently and the store
        // keeps the file open for the whole export.
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "geopq_svg_export_{}_{seq}.parquet",
            std::process::id()
        ));
        write_squares(&path);
        let (store, crs, _info, _rg) =
            loader::open_source_for_test(&Source::Local(path.clone())).unwrap();
        let store = Arc::new(store);
        let display = DisplayCrs::new(Crs::wgs84());
        let mut sb = styled_by_value();
        sb.hidden_bins = hidden_bins;
        let sel = loader::resolve_style(&store, &sb).expect("style column");
        let geometry = loader::build_geometry_styled_for_test(&store, &crs, &display, &sel)
            .unwrap()
            .0;

        let mut style = crate::data::layer::LayerStyle::new(Color32::from_rgb(200, 80, 40));
        style.style_by = Some(sb.clone());
        style.point_radius_px = 4.0;
        let camera = Camera { center: display.world_from_projected(4.0, 1.0), zoom: 5.0 };
        let viewport_px = [800.0, 600.0];
        let tl = camera.screen_to_world([0.0, 0.0], viewport_px);
        let br = camera.screen_to_world(viewport_px, viewport_px);
        let view_world = [tl[0], tl[1], br[0], br[1]];
        let job = SvgJob {
            layers: vec![SvgLayerJob {
                name: "squares".into(),
                style: resolve_style(&style),
                style_by: Some(sb),
                sections: vec![(Arc::clone(&geometry.chunks), Arc::clone(&geometry.rtree))],
                store: Arc::clone(&store),
                crs,
                loaded: vec![GroupLoad::Full],
                decimated: false,
                attribution: Some("© a publisher".into()),
            }],
            // Framed on the middle of the row of squares, with the whole
            // extent inside the viewport. The view rect is derived from
            // the camera the way the frame derives it.
            camera,
            display,
            viewport_px,
            pixels_per_point: 1.0,
            view_world,
            dark: true,
            graticule: false,
            coastline: None,
        };
        let out = build_svg(job);
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
        assert_eq!(out.name, "squares.svg");
        let _ = std::fs::remove_file(&path);
        out.doc
    }

    #[test]
    fn a_real_layer_exports_every_feature_with_its_own_class() {
        let svg = export(0);
        // Four polygons: four opaque fills in the composite group, four
        // stroked outlines, four markers. A count alone would also pass
        // on an exporter that emitted one feature eight times, so the
        // colours below pin which feature each element is.
        assert_eq!(svg.matches(r#"fill-rule="evenodd""#).count(), 4, "{svg}");
        assert_eq!(svg.matches(r#"fill="none""#).count(), 4);
        assert_eq!(svg.matches("<circle").count(), 4);
        // Values 0/10/20/30 against breaks 5/15/25: one polygon and one
        // marker per class, both painted in the class colour as the map
        // paints them.
        for c in &class_colors() {
            assert_eq!(
                svg.matches(&format!(r#"fill="{c}""#)).count(),
                2,
                "class colour {c}: one fill, one marker"
            );
        }
        // Every ring closes with four corners, and no path is a fragment.
        for d in svg.split(r#" d=""#).skip(1) {
            let d = d.split('"').next().unwrap();
            assert!(d.starts_with('M'), "{d}");
            assert!(d.ends_with('Z'), "{d}");
            assert_eq!(d.matches('L').count(), 3, "square ring: {d}");
        }
        // The publisher's credit is reproduced: the licence asks for it
        // where the data is seen.
        assert!(svg.contains("© a publisher") && svg.contains("<text"));
        assert!(svg.contains("raster basemap is not part of this file"));
        // Geometry came from the store, not from the tessellated mesh:
        // eight paths, one ring each. A triangulated square would arrive
        // as two triangles.
        assert_eq!(svg.matches(r#" d=""#).count(), 8, "4 fills + 4 outlines");
    }

    #[test]
    fn a_hidden_class_leaves_the_document_and_comes_back() {
        let all = export(0);
        let hidden = export(1 << 2);
        // Class 2 gone: one fill, one outline and one marker fewer.
        assert_eq!(hidden.matches(r#"fill-rule="evenodd""#).count(), 3);
        assert_eq!(hidden.matches(r#"fill="none""#).count(), 3);
        assert_eq!(hidden.matches("<circle").count(), 3);
        let gone = format!(r#"fill="{}""#, class_colors()[2]);
        assert!(all.contains(&gone));
        assert!(!hidden.contains(&gone), "hidden class still drawn");
    }


    /// The built-in overlays are rebuilt from their sources, so they carry
    /// full-detail polylines rather than the LOD-simplified segments the
    /// chunks hold — and they draw under the data layers, as on the map.
    #[test]
    fn overlays_export_as_full_detail_polylines_below_the_layers() {
        let display = DisplayCrs::hobo_dyer();
        let camera = Camera { center: [0.5, 0.5], zoom: 1.0 };
        let viewport_px = [1200.0, 800.0];
        let tl = camera.screen_to_world([0.0, 0.0], viewport_px);
        let br = camera.screen_to_world(viewport_px, viewport_px);
        let job = SvgJob {
            layers: Vec::new(),
            display,
            camera,
            viewport_px,
            pixels_per_point: 1.0,
            view_world: [tl[0], tl[1], br[0], br[1]],
            dark: true,
            graticule: true,
            coastline: Some(crate::data::coastline::CoastLevel::Embedded),
        };
        let out = build_svg(job);
        let svg = out.doc;
        assert!(svg.contains("<title>graticule</title>"));
        assert!(svg.contains("<title>coastline</title>"));
        // Graticule before coastline, matching the map's draw order.
        assert!(
            svg.find("<title>graticule</title>") < svg.find("<title>coastline</title>"),
            "overlay order"
        );
        // 25 meridians + 11 parallels, none of which fails to project in
        // Hobo–Dyer, so none splits.
        let grat = &svg[svg.find(r#"<g id="layer-0">"#).unwrap()
            ..svg.find(r#"<g id="layer-1">"#).unwrap()];
        assert_eq!(grat.matches("<path").count(), 36, "meridians + parallels");
        // The overlay styles are the map's own, not re-invented here.
        let g = graticule_style(true);
        assert!(grat.contains(&format!(
            r#"stroke-width="{}""#,
            g.line_half_width_px * 2.0
        )));
        // The coastline keeps 1:50m detail: an LOD-simplified read-back
        // would land in the low thousands of vertices.
        let coast = &svg[svg.find(r#"<g id="layer-1">"#).unwrap()..];
        assert!(coast.matches(" L").count() > 50_000, "{}", coast.matches(" L").count());
        assert!(coast.matches("<path").count() > 1_000);
        // Lines only: an overlay has no fills and no markers.
        assert_eq!(coast.matches("<g opacity=").count(), 0);
        assert_eq!(coast.matches("<circle").count(), 0);
    }

    /// Nothing in view: still a document, still a background.
    #[test]
    fn an_empty_viewport_still_writes_a_valid_document() {
        let path = std::env::temp_dir()
            .join(format!("geopq_svg_empty_{}.parquet", std::process::id()));
        write_squares(&path);
        let (store, crs, _info, _rg) =
            loader::open_source_for_test(&Source::Local(path.clone())).unwrap();
        let store = Arc::new(store);
        let display = DisplayCrs::new(Crs::wgs84());
        let geometry = loader::build_geometry_for_test(&store, &crs, &display).unwrap().0;
        // Half a world away from the data, zoomed right in.
        let camera = Camera { center: display.world_from_projected(-170.0, -60.0), zoom: 12.0 };
        let viewport_px = [640.0, 480.0];
        let tl = camera.screen_to_world([0.0, 0.0], viewport_px);
        let br = camera.screen_to_world(viewport_px, viewport_px);
        let view_world = [tl[0], tl[1], br[0], br[1]];
        let job = SvgJob {
            layers: vec![SvgLayerJob {
                name: "squares".into(),
                style: resolve_style(&crate::data::layer::LayerStyle::new(Color32::RED)),
                style_by: None,
                sections: vec![(Arc::clone(&geometry.chunks), Arc::clone(&geometry.rtree))],
                store: Arc::clone(&store),
                crs,
                loaded: vec![GroupLoad::Full],
                decimated: true,
                attribution: None,
            }],
            camera,
            display,
            viewport_px,
            pixels_per_point: 1.0,
            view_world,
            dark: false,
            graticule: true,
            coastline: None,
        };
        let out = build_svg(job);
        let _ = std::fs::remove_file(&path);
        assert!(out.doc.contains(r#"viewBox="0 0 640 480""#));
        assert!(out.doc.contains(r##"fill="#f4f3f0""##), "background rect");
        assert_eq!(out.doc.matches(r#"fill-rule="evenodd""#).count(), 0);
        assert_eq!(out.doc.matches("<circle").count(), 0);
        // A decimated layer says so in the document rather than passing
        // itself off as the whole dataset.
        assert!(out.doc.contains("decimated preview"));
        assert!(out.doc.trim_end().ends_with("</svg>"));
    }
}
