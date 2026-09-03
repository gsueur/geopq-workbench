use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::map::camera::Camera;
use crate::map::warp::{self, TileMesh, Warp, WarpPlan};

#[allow(dead_code)]
pub const TILE_PX: u32 = 256;
/// Decoded-pixel budget for the cache, in bytes.
///
/// Counting tiles was the wrong unit: a 256² tile carries a full mip
/// chain, so it is ~350 kB of pixels, and 600 of them is over 200 MB of
/// host memory with a matching set of GPU textures behind it. ~64 MB is
/// around 190 tiles, still several viewports' worth at any zoom.
const MAX_CACHED_BYTES: usize = 64 << 20;
/// Slack before eviction runs, in bytes: evicting a handful at a time
/// would sort the whole cache on most frames.
const EVICT_SLACK_BYTES: usize = 4 << 20;
const FETCH_THREADS: usize = 4;
/// CARTO's basemaps need a (free) API key since 2026: without one the
/// tiles come back watermarked "API KEY REQUIRED". The key is read once at
/// startup from `GEOPQ_CARTO_API_KEY` or the settings file and appended to
/// every cartocdn request; see <https://carto.com/basemaps/apikey>.
static CARTO_API_KEY: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

pub fn set_carto_api_key(key: Option<String>) {
    let key = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    if let Ok(mut g) = CARTO_API_KEY.write() {
        *g = key;
    }
}

pub fn carto_api_key() -> Option<String> {
    CARTO_API_KEY.read().ok().and_then(|g| g.clone())
}

/// Whether a source is served by CARTO and so needs the key.
pub fn is_carto(source_idx: usize) -> bool {
    TILE_SOURCES
        .get(source_idx)
        .is_some_and(|s| s.url.contains("cartocdn.com"))
}

/// The tile URL for a key, with the CARTO API key appended when the
/// source needs one and the user has provided one.
fn tile_url(source: &TileSource, id: TileId) -> String {
    let mut url = source
        .url
        .replace("{z}", &id.z.to_string())
        .replace("{x}", &id.x.to_string())
        .replace("{y}", &id.y.to_string());
    if source.url.contains("cartocdn.com")
        && let Some(key) = carto_api_key()
    {
        url.push_str("?key=");
        url.push_str(&key);
    }
    url
}

const USER_AGENT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/gsueur/geopq-workbench)"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl TileId {
    pub fn parent(&self) -> Option<TileId> {
        if self.z == 0 {
            return None;
        }
        Some(TileId {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// World-space rect [minx, miny, maxx, maxy] of this tile (slippy scheme).
    pub fn world_rect(&self) -> [f64; 4] {
        let n = (1u64 << self.z) as f64;
        [
            self.x as f64 / n,
            self.y as f64 / n,
            (self.x as f64 + 1.0) / n,
            (self.y as f64 + 1.0) / n,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    pub source: u8,
    pub id: TileId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSource {
    pub name: &'static str,
    pub url: &'static str,
    pub attribution: &'static str,
    pub max_zoom: u8,
    /// Whether place names are rendered into the tile pixels. Labels are the
    /// one thing reprojection cannot fix, since they shear with the raster
    /// they are baked into, so a label-free source is what makes a basemap
    /// usable outside Mercator.
    pub labels: bool,
}

pub const TILE_SOURCES: &[TileSource] = &[
    TileSource {
        name: "Carto Light",
        url: "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
        labels: true,
    },
    TileSource {
        name: "Carto Dark",
        url: "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
        labels: true,
    },
    TileSource {
        name: "Carto Voyager",
        url: "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
        labels: true,
    },
    TileSource {
        name: "OpenStreetMap",
        url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors",
        max_zoom: 19,
        labels: true,
    },
    TileSource {
        name: "OpenTopoMap",
        url: "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors, SRTM · © OpenTopoMap (CC-BY-SA)",
        max_zoom: 17,
        labels: true,
    },
    TileSource {
        name: "Esri World Topo",
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/\
               World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attribution: "Esri, TomTom, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors",
        max_zoom: 19,
        labels: true,
    },
    TileSource {
        // Imagery carries no rendered text at all, so it reprojects
        // without any of the label caveats.
        name: "Esri World Imagery",
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/\
               World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution: "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        max_zoom: 19,
        labels: false,
    },
    TileSource {
        name: "Carto Light (no labels)",
        url: "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
        labels: false,
    },
    TileSource {
        name: "Carto Dark (no labels)",
        url: "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
        labels: false,
    },
    TileSource {
        name: "Carto Voyager (no labels)",
        url: "https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
        labels: false,
    },
];

/// The label-free twin of a source, for when the display projection would
/// shear rendered text. None when the source has no such variant.
pub fn nolabels_twin(source_idx: usize) -> Option<usize> {
    let want = match TILE_SOURCES.get(source_idx)?.name {
        "Carto Light" => "Carto Light (no labels)",
        "Carto Dark" => "Carto Dark (no labels)",
        "Carto Voyager" => "Carto Voyager (no labels)",
        _ => return None,
    };
    TILE_SOURCES.iter().position(|s| s.name == want)
}

#[derive(Debug)]
enum TileState {
    /// Queued or in flight. Carries the failures so far so a retry keeps
    /// backing off instead of restarting the clock.
    Pending { attempts: u32 },
    Ready,
    /// Fetch failed; retried with exponential backoff (transient network
    /// errors must not blank tiles for the whole session).
    Failed {
        at: std::time::Instant,
        attempts: u32,
    },
}

/// Max retry attempts per tile; backoff = 2^attempts seconds.
const TILE_RETRY_MAX: u32 = 4;

struct CacheEntry {
    state: TileState,
    last_used: u64,
    /// Decoded size of the mip chain, once the tile is Ready (0 while it
    /// is pending or failed, which is what makes those free to keep).
    bytes: usize,
}

/// One level of a tile's mip chain, RGBA8 in sRGB storage.
pub struct MipLevel {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

pub struct TileUpload {
    pub key: TileKey,
    /// Level 0 first. Built on the fetch threads so the render thread only
    /// ever copies bytes.
    pub mips: Vec<MipLevel>,
}

/// sRGB storage value to linear light, one entry per byte.
fn srgb_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let u = i as f32 / 255.0;
            if u <= 0.040_45 {
                u / 12.92
            } else {
                ((u + 0.055) / 1.055).powf(2.4)
            }
        })
    })
}

fn linear_to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// Box-reduce an RGBA8-sRGB image to a full mip chain, level 0 first.
///
/// Reprojection minifies tiles by several times along one axis, which
/// without mips aliases into shimmering noise. Colour is averaged in linear
/// light: averaging sRGB bytes directly would visibly darken every reduced
/// level. Alpha is already linear and is averaged as-is.
pub fn mip_chain(base: &[u8], w: u32, h: u32) -> Vec<MipLevel> {
    let lut = srgb_lut();
    let mut out = vec![MipLevel {
        w,
        h,
        px: base.to_vec(),
    }];
    while out.last().is_some_and(|m| m.w > 1 || m.h > 1) {
        let next = {
            let src = out.last().unwrap();
            let (nw, nh) = ((src.w / 2).max(1), (src.h / 2).max(1));
            let mut px = vec![0u8; (nw * nh * 4) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    // Clamped 2x2 footprint, so odd sizes still reduce.
                    let x0 = (2 * x).min(src.w - 1);
                    let x1 = (2 * x + 1).min(src.w - 1);
                    let y0 = (2 * y).min(src.h - 1);
                    let y1 = (2 * y + 1).min(src.h - 1);
                    let mut rgb = [0f32; 3];
                    let mut a = 0f32;
                    for (sx, sy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                        let i = ((sy * src.w + sx) * 4) as usize;
                        for (c, acc) in rgb.iter_mut().enumerate() {
                            *acc += lut[src.px[i + c] as usize];
                        }
                        a += src.px[i + 3] as f32;
                    }
                    let o = ((y * nw + x) * 4) as usize;
                    for (c, acc) in rgb.iter().enumerate() {
                        px[o + c] = linear_to_srgb(acc * 0.25);
                    }
                    px[o + 3] = (a * 0.25).round() as u8;
                }
            }
            MipLevel { w: nw, h: nh, px }
        };
        out.push(next);
    }
    out
}

pub struct TileDrawCmd {
    pub key: TileKey,
    /// Mercator world rect, used when the display projection *is* Mercator
    /// and the tile is an axis-aligned quad.
    pub world_rect: [f64; 4],
    /// Set instead when the tile has to be warped: a subdivided patch in
    /// display world space (see `map::warp`).
    pub mesh: Option<Arc<TileMesh>>,
}

struct FetchResult {
    key: TileKey,
    /// Cache generation the fetch was issued under (see `Queue`).
    generation: u64,
    mips: Option<Vec<MipLevel>>,
}

/// The fetch queue, rebuilt from scratch on every frame that asks for
/// tiles.
///
/// A FIFO was the wrong shape: zooming through six levels queued every
/// level on the way, and the workers kept grinding through views the user
/// had already left. What matters is only ever the current view, so the
/// want list is replaced rather than appended to and obsolete tiles are
/// dropped before anyone spends a request on them.
///
/// Requests already in flight cannot be taken back — a blocking read is
/// not interruptible — but there are at most `FETCH_THREADS` of those, so
/// the tail this used to leave behind is gone.
#[derive(Default)]
struct Queue {
    /// Tiles the current view wants, most useful first.
    want: VecDeque<TileKey>,
    /// Handed to a worker and not yet finished; never queued twice.
    ///
    /// A key stays here until `poll` takes its result off the channel, not
    /// until the worker stops reading: in between there was a window of a
    /// frame or two where the tile was neither in flight nor known-good,
    /// and the next frame queued a second fetch for it.
    in_flight: HashSet<TileKey>,
    /// Bumped by `clear`. Workers stamp it on the result they send, so a
    /// fetch issued before the cache was emptied can be recognised and
    /// dropped rather than reinstated as a fresh entry.
    generation: u64,
    /// Set when the cache is dropped, so workers stop waiting and exit.
    stop: bool,
}

type Shared = Arc<(Mutex<Queue>, Condvar)>;

pub struct TileCache {
    entries: HashMap<TileKey, CacheEntry>,
    queue: Shared,
    /// Kept so the channel stays open even if every worker exits, and so
    /// tests can hand `poll` a result without a network.
    #[allow(dead_code)]
    res_tx: Sender<FetchResult>,
    res_rx: Receiver<FetchResult>,
    pending_uploads: Vec<TileUpload>,
    frame: u64,
    /// Warped tile geometry, valid for one display projection only. Meshes
    /// are stable while panning and zooming, so they are built once per tile
    /// and thrown away when the projection changes.
    warp_meshes: HashMap<TileId, Option<Arc<TileMesh>>>,
    warp_epoch: u64,
    /// Bumped by `clear`, stamped on every fetch (see `Queue`).
    generation: u64,
}


/// Warped meshes kept across frames. Well above any one viewport, so panning
/// back and forth never rebuilds; cleared wholesale on a projection change.
const MAX_CACHED_MESHES: usize = 2048;

impl TileCache {
    pub fn new(egui_ctx: eframe::egui::Context) -> Self {
        let (res_tx, res_rx) = channel::<FetchResult>();
        let queue: Shared = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        for _ in 0..FETCH_THREADS {
            let q = queue.clone();
            let tx = res_tx.clone();
            let ctx = egui_ctx.clone();
            std::thread::spawn(move || loop {
                let (key, generation) = {
                    let (lock, cv) = &*q;
                    let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if g.stop {
                            return;
                        }
                        if let Some(k) = g.want.pop_front() {
                            g.in_flight.insert(k);
                            break (k, g.generation);
                        }
                        g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
                    }
                };
                let mips = fetch_tile(key);
                // `in_flight` is cleared by `poll`, once the result is
                // actually in hand.
                if tx
                    .send(FetchResult {
                        key,
                        generation,
                        mips,
                    })
                    .is_err()
                {
                    return;
                }
                ctx.request_repaint();
            });
        }
        Self {
            entries: HashMap::new(),
            queue,
            res_tx,
            res_rx,
            pending_uploads: Vec::new(),
            frame: 0,
            warp_meshes: HashMap::new(),
            warp_epoch: 0,
            generation: 0,
        }
    }

    /// Replace the fetch queue with exactly what this view wants.
    ///
    /// Anything queued for an earlier view and not yet picked up is
    /// dropped here, which is the whole point: a zoom through several
    /// levels must not spend requests on the levels it passed through.
    fn set_wanted(&mut self, want: Vec<TileKey>) {
        let (lock, cv) = &*self.queue;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        let in_flight = std::mem::take(&mut g.in_flight);
        g.want = want.into_iter().filter(|k| !in_flight.contains(k)).collect();
        g.in_flight = in_flight;
        cv.notify_all();
    }

    /// Drain results from fetch threads; call once per frame.
    pub fn poll(&mut self) {
        while let Ok(res) = self.res_rx.try_recv() {
            {
                let mut g = self.queue.0.lock().unwrap_or_else(|e| e.into_inner());
                g.in_flight.remove(&res.key);
            }
            // Fetched for a cache that has since been emptied (the CARTO
            // key changed, say): the pixels are the ones `clear` was
            // called to get rid of, and reinstating them here would put
            // the watermarked tile straight back on screen.
            if res.generation != self.generation {
                continue;
            }
            let mut bytes = 0usize;
            let state = match res.mips {
                Some(mips) => {
                    bytes = mips.iter().map(|m| m.px.len()).sum();
                    self.pending_uploads.push(TileUpload { key: res.key, mips });
                    TileState::Ready
                }
                None => {
                    let attempts = match self.entries.get(&res.key).map(|e| &e.state) {
                        Some(TileState::Pending { attempts }) => attempts + 1,
                        Some(TileState::Failed { attempts, .. }) => attempts + 1,
                        _ => 1,
                    };
                    TileState::Failed {
                        at: std::time::Instant::now(),
                        attempts,
                    }
                }
            };
            if let Some(e) = self.entries.get_mut(&res.key) {
                e.state = state;
                e.bytes = bytes;
            } else {
                self.entries.insert(
                    res.key,
                    CacheEntry {
                        state,
                        last_used: self.frame,
                        bytes,
                    },
                );
            }
        }
    }

    pub fn take_uploads(&mut self) -> Vec<TileUpload> {
        std::mem::take(&mut self.pending_uploads)
    }

    /// Keys that renderers may keep GPU textures for.
    /// Forget every cached tile, so the next frame refetches. Used when
    /// the CARTO API key changes: the watermarked tiles fetched without it
    /// would otherwise stay on screen until they scrolled out of view.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.generation += 1;
        if let Ok(mut g) = self.queue.0.lock() {
            g.want.clear();
            g.generation = self.generation;
        }
    }

    /// Keys the renderer must keep a GPU texture for.
    ///
    /// Exactly the Ready entries, and that is a contract rather than a
    /// convenience: the decoded pixels are handed to the renderer once
    /// and dropped, so a tile the GPU forgets while the cache still calls
    /// it Ready is never uploaded again and stays blank for as long as
    /// the entry lives. What bounds VRAM is therefore the cache's own
    /// byte budget (see `evict`), which the texture set follows.
    pub fn alive_keys(&self) -> HashSet<TileKey> {
        self.entries
            .iter()
            .filter(|(_, e)| matches!(e.state, TileState::Ready))
            .map(|(k, _)| *k)
            .collect()
    }

    /// Compute the draw list for a Mercator view, where a tile is an
    /// axis-aligned quad and world space is the tile pyramid's own.
    pub fn draws(
        &mut self,
        source_idx: usize,
        camera: &Camera,
        viewport_px: [f32; 2],
    ) -> Vec<TileDrawCmd> {
        let source = &TILE_SOURCES[source_idx];
        let z = (camera.zoom.round() as i64).clamp(0, source.max_zoom as i64) as u8;
        let n = 1u64 << z;

        let tl = camera.screen_to_world([0.0, 0.0], viewport_px);
        let br = camera.screen_to_world(viewport_px, viewport_px);
        // Longitude wraps: a view panned past the antimeridian sits at
        // world x > 1, where clamping the column range left the basemap
        // ending in a hard edge. The index wraps into the pyramid and the
        // quad is put back on the right side of the seam below.
        let x0 = (tl[0] * n as f64).floor() as i64;
        let x1 = ((br[0] * n as f64).ceil() as i64).min(x0 + n as i64);
        let y0 = ((tl[1] * n as f64).floor() as i64).max(0);
        let y1 = ((br[1] * n as f64).ceil() as i64).min(n as i64);

        let mut wanted: Vec<TileId> = Vec::new();
        for x in x0..x1 {
            for y in y0..y1 {
                wanted.push(TileId {
                    z,
                    x: x.rem_euclid(n as i64) as u32,
                    y: y as u32,
                });
            }
        }
        // Sanity cap in case of a degenerate viewport.
        if wanted.len() > 512 {
            wanted.truncate(512);
        }
        let mut draws = self.resolve(source_idx, wanted);
        // A tile id wraps at the seam; the quad it draws must not. Put
        // every tile in the copy of the world nearest the view centre.
        let cx = camera.center[0];
        for d in &mut draws {
            let shift = (cx - (d.world_rect[0] + d.world_rect[2]) * 0.5).round();
            if shift != 0.0 {
                d.world_rect[0] += shift;
                d.world_rect[2] += shift;
            }
        }
        draws
    }

    /// Compute the draw list for a non-Mercator view: same cache and same
    /// ancestor substitution, but every tile carries a projected mesh and
    /// tiles without one are dropped rather than drawn in the wrong place.
    pub fn draws_warped(
        &mut self,
        source_idx: usize,
        plan: &WarpPlan,
        warp: &Warp,
        epoch: u64,
    ) -> Vec<TileDrawCmd> {
        if self.warp_epoch != epoch {
            self.warp_meshes.clear();
            self.warp_epoch = epoch;
        }
        let max_zoom = TILE_SOURCES[source_idx].max_zoom;
        let z = plan.zoom.min(max_zoom);
        let wanted = warp::tiles_for(plan.merc_bbox, z, 512);
        let mut draws = self.resolve(source_idx, wanted);
        if self.warp_meshes.len() > MAX_CACHED_MESHES {
            self.warp_meshes.clear();
        }
        for d in &mut draws {
            let id = d.key.id;
            d.mesh = match self.warp_meshes.get(&id) {
                Some(m) => m.clone(),
                None => {
                    let m = warp::tile_mesh(warp, id, warp::TILE_SUBDIV).map(Arc::new);
                    self.warp_meshes.insert(id, m.clone());
                    m
                }
            };
        }
        draws.retain(|d| d.mesh.is_some());
        draws
    }

    /// Turn a wanted tile list into a draw list: request what is missing,
    /// substitute loaded ancestors underneath while children load.
    fn resolve(&mut self, source_idx: usize, wanted: Vec<TileId>) -> Vec<TileDrawCmd> {
        self.frame += 1;
        let mut draws: Vec<TileDrawCmd> = Vec::new();
        let mut fallback: Vec<TileId> = Vec::new();
        // The queue is rebuilt wholesale, so this is the complete list of
        // what to fetch for this view — not a delta on the last one.
        let mut want: Vec<TileKey> = Vec::new();

        for id in &wanted {
            let key = TileKey {
                source: source_idx as u8,
                id: *id,
            };
            match self.entries.get_mut(&key) {
                Some(e) => {
                    e.last_used = self.frame;
                    match e.state {
                        TileState::Ready => draws.push(TileDrawCmd {
                            key,
                            world_rect: id.world_rect(),
                            mesh: None,
                        }),
                        TileState::Pending { .. } => {
                            want.push(key);
                            fallback.push(*id);
                        }
                        TileState::Failed { at, attempts } => {
                            let backoff =
                                std::time::Duration::from_secs(1u64 << attempts.min(6));
                            if attempts < TILE_RETRY_MAX && at.elapsed() >= backoff {
                                // Carry the count forward: resetting it here
                                // would restart the backoff every retry and
                                // it would never actually back off.
                                e.state = TileState::Pending { attempts };
                                want.push(key);
                            }
                            fallback.push(*id);
                        }
                    }
                }
                None => {
                    self.entries.insert(
                        key,
                        CacheEntry {
                            state: TileState::Pending { attempts: 0 },
                            last_used: self.frame,
                            bytes: 0,
                        },
                    );
                    want.push(key);
                    fallback.push(*id);
                }
            }
        }
        self.set_wanted(want);

        // Ancestor fallback for missing tiles (drawn first, children on top).
        let mut ancestors: Vec<TileId> = Vec::new();
        let mut seen: HashSet<TileId> = HashSet::new();
        for id in fallback {
            let mut cur = id.parent();
            for _ in 0..6 {
                let Some(p) = cur else { break };
                let key = TileKey {
                    source: source_idx as u8,
                    id: p,
                };
                if let Some(e) = self.entries.get_mut(&key) {
                    if matches!(e.state, TileState::Ready) {
                        e.last_used = self.frame;
                        if seen.insert(p) {
                            ancestors.push(p);
                        }
                        break;
                    }
                }
                cur = p.parent();
            }
        }
        ancestors.sort(); // z ascending: coarser first
        let mut all: Vec<TileDrawCmd> = ancestors
            .into_iter()
            .map(|id| TileDrawCmd {
                key: TileKey {
                    source: source_idx as u8,
                    id,
                },
                world_rect: id.world_rect(),
                mesh: None,
            })
            .collect();
        all.append(&mut draws);

        self.evict();
        all
    }

    /// Drop least-recently-used tiles until the cache is back under
    /// budget. Only runs once the overshoot is worth a full sort.
    fn evict(&mut self) {
        let mut used: usize = self.entries.values().map(|e| e.bytes).sum();
        if used <= MAX_CACHED_BYTES + EVICT_SLACK_BYTES {
            return;
        }
        let mut by_age: VecDeque<(u64, TileKey)> = self
            .entries
            .iter()
            .map(|(k, e)| (e.last_used, *k))
            .collect();
        by_age.make_contiguous().sort();
        for (_, key) in by_age.iter() {
            if used <= MAX_CACHED_BYTES {
                break;
            }
            if let Some(e) = self.entries.remove(key) {
                used -= e.bytes;
            }
        }
    }

    /// Tiles queued or in flight for the current view.
    pub fn pending_count(&self) -> usize {
        let g = self.queue.0.lock().unwrap_or_else(|e| e.into_inner());
        g.want.len() + g.in_flight.len()
    }
}

impl Drop for TileCache {
    fn drop(&mut self) {
        let (lock, cv) = &*self.queue;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        g.stop = true;
        g.want.clear();
        cv.notify_all();
    }
}

/// Fetch one tile synchronously. Only for the snapshot harness, which needs
/// real imagery to judge whether a reprojected basemap looks right.
#[cfg(test)]
pub fn fetch_tile_blocking(key: TileKey) -> Option<Vec<MipLevel>> {
    fetch_tile(key)
}

/// Shared agent for tile fetches: connection pooling across the four
/// worker threads (a basemap pan is hundreds of requests to one host, and
/// a fresh agent per tile means a fresh TLS handshake per tile), and
/// timeouts, because a stalled read on a worker blocks that quarter of
/// the fetch capacity for as long as the OS lets it. Modelled on
/// `source::http_agent`, but tighter: a tile is ~50 kB and worth
/// abandoning early — the retry backoff will come back to it.
///
/// `http_status_as_error` keeps its default, so a 404 or a 429 arrives as
/// `Err` and lands in that same backoff.
fn tile_agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(std::time::Duration::from_secs(10)))
            .timeout_connect(Some(std::time::Duration::from_secs(10)))
            .timeout_recv_response(Some(std::time::Duration::from_secs(15)))
            .timeout_recv_body(Some(std::time::Duration::from_secs(30)))
            .build()
            .into()
    })
}

fn fetch_tile(key: TileKey) -> Option<Vec<MipLevel>> {
    let source = &TILE_SOURCES[key.source as usize];
    let url = tile_url(source, key.id);
    let mut res = tile_agent()
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .call()
        .ok()?;
    let bytes = res.body_mut().read_to_vec().ok()?;
    crate::data::net::record(crate::data::net::Channel::Tiles, &url, bytes.len() as u64);
    let rgba = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Some(mip_chain(rgba.as_raw(), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every labelled Carto source needs a twin, since that is what a
    /// reprojected view falls back to.
    #[test]
    fn carto_urls_carry_the_api_key_only_when_set() {
        let carto = TILE_SOURCES.iter().find(|s| s.url.contains("cartocdn.com")).unwrap();
        let osm = TILE_SOURCES.iter().find(|s| s.name == "OpenStreetMap").unwrap();
        let id = TileId { z: 3, x: 2, y: 3 };
        set_carto_api_key(None);
        assert!(!tile_url(carto, id).contains("key="));
        set_carto_api_key(Some("  abc123 ".into()));
        assert!(tile_url(carto, id).ends_with("/3/2/3.png?key=abc123"));
        assert!(!tile_url(osm, id).contains("key="));
        set_carto_api_key(Some(String::new()));
        assert!(carto_api_key().is_none());
    }

    #[test]
    fn carto_sources_have_nolabels_twins() {
        for (i, s) in TILE_SOURCES.iter().enumerate() {
            if !s.name.starts_with("Carto") || !s.labels {
                continue;
            }
            let twin = nolabels_twin(i).expect(s.name);
            assert!(!TILE_SOURCES[twin].labels, "{} twin still labelled", s.name);
            assert_eq!(TILE_SOURCES[twin].max_zoom, s.max_zoom);
        }
        // OSM has no label-free variant, and must not claim one.
        let osm = TILE_SOURCES.iter().position(|s| s.name == "OpenStreetMap").unwrap();
        assert_eq!(nolabels_twin(osm), None);
    }

    /// Line continuations inside these URLs have eaten characters before.
    /// Every source must still be a well-formed https template.
    #[test]
    fn every_source_url_is_well_formed() {
        for s in TILE_SOURCES {
            assert!(s.url.starts_with("https://"), "{}: {}", s.name, s.url);
            assert!(!s.url.contains(' '), "{} has whitespace: {}", s.name, s.url);
            for tag in ["{z}", "{x}", "{y}"] {
                assert!(s.url.contains(tag), "{} lacks {tag}: {}", s.name, s.url);
            }
            assert!(!s.attribution.is_empty(), "{} has no attribution", s.name);
            assert!(s.max_zoom >= 15 && s.max_zoom <= 22, "{}: {}", s.name, s.max_zoom);
        }
    }

    /// A zoom that crosses several levels must not leave the earlier
    /// levels queued. This is the behaviour the FIFO got wrong: it kept
    /// fetching views the user had already left.
    #[test]
    fn a_new_view_replaces_the_queue_instead_of_extending_it() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        // Stop the workers before they can drain anything, so what the
        // queue holds is exactly what each view asked for.
        {
            let mut g = cache.queue.0.lock().unwrap();
            g.stop = true;
        }

        let ids = |z: u8, n: u32| -> Vec<TileId> {
            (0..n).map(|x| TileId { z, x, y: 0 }).collect()
        };
        cache.resolve(0, ids(8, 6));
        assert_eq!(cache.pending_count(), 6, "the first view queues its tiles");

        // Zoom in: a different level entirely.
        cache.resolve(0, ids(14, 3));
        let g = cache.queue.0.lock().unwrap();
        assert_eq!(g.want.len(), 3, "only the current view stays queued");
        assert!(
            g.want.iter().all(|k| k.id.z == 14),
            "level 8 tiles survived the zoom: {:?}",
            g.want.iter().map(|k| k.id.z).collect::<Vec<_>>()
        );
    }

    /// A tile handed to a worker must not be queued again by the next
    /// frame, or a slow fetch would be issued once per frame.
    #[test]
    fn in_flight_tiles_are_not_requeued() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        {
            let mut g = cache.queue.0.lock().unwrap();
            g.stop = true;
        }
        let want = vec![TileId { z: 5, x: 1, y: 1 }, TileId { z: 5, x: 2, y: 1 }];
        cache.resolve(0, want.clone());
        // Simulate a worker taking the first one.
        let taken = {
            let mut g = cache.queue.0.lock().unwrap();
            let k = g.want.pop_front().unwrap();
            g.in_flight.insert(k);
            k
        };
        cache.resolve(0, want);
        let g = cache.queue.0.lock().unwrap();
        assert!(
            !g.want.contains(&taken),
            "a tile already being fetched was queued again"
        );
        assert_eq!(g.in_flight.len(), 1, "in-flight state must survive a rebuild");
    }

    /// Retry backoff has to grow. Marking a failed tile Pending used to
    /// erase the attempt count, so every retry started the clock again and
    /// a dead tile was re-requested every two seconds forever.
    #[test]
    fn retry_backoff_keeps_counting() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        {
            let mut g = cache.queue.0.lock().unwrap();
            g.stop = true;
        }
        let id = TileId { z: 3, x: 1, y: 1 };
        let key = TileKey { source: 0, id };
        let failed = |ago: u64, attempts: u32| CacheEntry {
            state: TileState::Failed {
                at: std::time::Instant::now() - std::time::Duration::from_secs(ago),
                attempts,
            },
            last_used: 0,
            bytes: 0,
        };

        // Three failures, long past its 8 s backoff: retried, count kept.
        cache.entries.insert(key, failed(600, 3));
        cache.resolve(0, vec![id]);
        match cache.entries.get(&key).unwrap().state {
            TileState::Pending { attempts } => assert_eq!(attempts, 3),
            ref other => panic!("expected a retry, got {other:?}"),
        }
        assert_eq!(cache.pending_count(), 1);

        // Still inside the backoff: left alone, and nothing queued.
        cache.entries.insert(key, failed(1, 3));
        cache.resolve(0, vec![id]);
        assert!(
            matches!(cache.entries.get(&key).unwrap().state, TileState::Failed { .. }),
            "retried before its backoff elapsed"
        );
        assert_eq!(cache.pending_count(), 0);

        // Out of attempts: never retried again.
        cache.entries.insert(key, failed(6000, TILE_RETRY_MAX));
        cache.resolve(0, vec![id]);
        assert_eq!(cache.pending_count(), 0, "retried past the attempt limit");
    }

    /// Tile fetches go through one pooled agent with timeouts. A fresh
    /// agent per tile means a TLS handshake per tile, and no timeout at
    /// all means one stalled read holds a quarter of the fetch capacity
    /// for as long as the OS allows.
    #[test]
    fn tile_fetches_share_an_agent_with_timeouts() {
        let t = tile_agent().config().timeouts();
        assert_eq!(t.resolve, Some(std::time::Duration::from_secs(10)));
        assert_eq!(t.connect, Some(std::time::Duration::from_secs(10)));
        assert_eq!(t.recv_response, Some(std::time::Duration::from_secs(15)));
        assert_eq!(t.recv_body, Some(std::time::Duration::from_secs(30)));
        // Same agent every time, or the pooling is pointless.
        assert!(std::ptr::eq(tile_agent(), tile_agent()));
        // 404 and 429 must still arrive as errors, so the retry backoff
        // sees them.
        assert!(tile_agent().config().http_status_as_error());
    }

    /// A view panned across the antimeridian must get tiles on both
    /// sides of the seam: the column index wraps into the pyramid, and
    /// the quad each tile draws is put in the copy of the world the view
    /// is actually looking at.
    #[test]
    fn a_view_across_the_antimeridian_gets_tiles_on_both_sides() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        {
            let mut g = cache.queue.0.lock().unwrap();
            g.stop = true;
        }
        let cam = Camera { center: [0.999, 0.5], zoom: 3.0 };
        let vp = [512.0, 256.0];
        cache.draws(0, &cam, vp);
        let want: Vec<TileKey> = {
            let g = cache.queue.0.lock().unwrap();
            g.want.iter().copied().collect()
        };
        let n = 1u32 << 3;
        assert!(
            want.iter().any(|k| k.id.x == n - 1),
            "nothing west of the seam: {:?}",
            want.iter().map(|k| k.id.x).collect::<Vec<_>>()
        );
        assert!(
            want.iter().any(|k| k.id.x == 0),
            "nothing east of the seam: {:?}",
            want.iter().map(|k| k.id.x).collect::<Vec<_>>()
        );

        for k in &want {
            cache.entries.insert(
                *k,
                CacheEntry { state: TileState::Ready, last_used: 0, bytes: 0 },
            );
        }
        let draws = cache.draws(0, &cam, vp);
        assert!(
            draws.iter().any(|d| d.world_rect[0] >= 1.0),
            "every quad landed west of the seam: {:?}",
            draws.iter().map(|d| d.world_rect[0]).collect::<Vec<_>>()
        );
    }

    /// The cache is bounded by decoded bytes, not by a tile count: a
    /// 256² tile with its mip chain is ~350 kB, so a count that looked
    /// modest was hundreds of megabytes of pixels.
    #[test]
    fn eviction_follows_a_byte_budget() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        let per = 350_000usize;
        let count = (MAX_CACHED_BYTES + EVICT_SLACK_BYTES) / per + 20;
        for i in 0..count {
            cache.entries.insert(
                TileKey { source: 0, id: TileId { z: 10, x: i as u32, y: 0 } },
                CacheEntry {
                    state: TileState::Ready,
                    last_used: i as u64,
                    bytes: per,
                },
            );
        }
        cache.evict();
        let used: usize = cache.entries.values().map(|e| e.bytes).sum();
        assert!(used <= MAX_CACHED_BYTES, "still {used} bytes cached");
        assert!(used > MAX_CACHED_BYTES / 2, "evicted far too much: {used}");
        // The oldest go first.
        assert!(!cache.entries.contains_key(&TileKey {
            source: 0,
            id: TileId { z: 10, x: 0, y: 0 }
        }));
        assert!(cache.entries.contains_key(&TileKey {
            source: 0,
            id: TileId { z: 10, x: count as u32 - 1, y: 0 }
        }));
    }

    /// The GPU holds a texture per Ready entry and the pixels behind it
    /// are gone after the upload, so the texture set has to track the
    /// cache exactly — and the cache's byte budget is what bounds both.
    /// Under the old tile count that came to ~210 MB of VRAM.
    #[test]
    fn the_gpu_texture_set_follows_the_byte_budget() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        let per = 350_000usize;
        for i in 0..600u32 {
            cache.entries.insert(
                TileKey { source: 0, id: TileId { z: 10, x: i, y: 0 } },
                CacheEntry {
                    state: TileState::Ready,
                    last_used: i as u64,
                    bytes: per,
                },
            );
        }
        cache.evict();
        let alive = cache.alive_keys();
        assert_eq!(alive.len(), cache.entries.len(), "a Ready tile lost its texture");
        assert!(
            alive.len() * per <= MAX_CACHED_BYTES,
            "{} textures is over budget",
            alive.len()
        );
        assert!(alive.len() > 100, "far too little cached: {}", alive.len());
    }

    /// `clear` exists to get rid of what is on screen (the CARTO key
    /// changed, and every cached tile is watermarked). Fetches already in
    /// flight carry the pixels it is throwing away, so their results have
    /// to be recognised and dropped instead of reinstated.
    #[test]
    fn clearing_the_cache_drops_fetches_already_in_flight() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        {
            let mut g = cache.queue.0.lock().unwrap();
            g.stop = true;
        }
        let key = TileKey { source: 0, id: TileId { z: 4, x: 3, y: 2 } };
        let generation = {
            let mut g = cache.queue.0.lock().unwrap();
            g.in_flight.insert(key);
            g.generation
        };
        cache.clear();
        cache
            .res_tx
            .send(FetchResult {
                key,
                generation,
                mips: Some(mip_chain(&[9u8; 4 * 4 * 4], 4, 4)),
            })
            .unwrap();
        cache.poll();
        assert!(cache.entries.is_empty(), "a stale fetch came back to life");
        assert!(cache.take_uploads().is_empty(), "stale pixels were uploaded");
        assert_eq!(cache.pending_count(), 0, "the claim outlived the result");
    }

    /// A tile stays claimed until its result is actually taken off the
    /// channel. The worker used to release it before sending, which left
    /// a window of a frame or two where the tile was neither in flight
    /// nor cached — and the next frame issued a second fetch for it.
    #[test]
    fn a_finished_fetch_stays_claimed_until_it_is_polled() {
        let ctx = eframe::egui::Context::default();
        let mut cache = TileCache::new(ctx);
        {
            let mut g = cache.queue.0.lock().unwrap();
            g.stop = true;
        }
        let id = TileId { z: 5, x: 1, y: 1 };
        let key = TileKey { source: 0, id };
        cache.resolve(0, vec![id]);
        // A worker takes it and finishes, but nothing has polled yet.
        let generation = {
            let mut g = cache.queue.0.lock().unwrap();
            let k = g.want.pop_front().unwrap();
            g.in_flight.insert(k);
            g.generation
        };
        cache
            .res_tx
            .send(FetchResult {
                key,
                generation,
                mips: Some(mip_chain(&[9u8; 4 * 4 * 4], 4, 4)),
            })
            .unwrap();
        cache.resolve(0, vec![id]);
        {
            let g = cache.queue.0.lock().unwrap();
            assert!(!g.want.contains(&key), "the same tile was fetched twice");
        }
        cache.poll();
        {
            let g = cache.queue.0.lock().unwrap();
            assert!(g.in_flight.is_empty(), "the claim was never released");
        }
        assert!(matches!(
            cache.entries.get(&key).map(|e| &e.state),
            Some(TileState::Ready)
        ));
    }

    #[test]
    fn mip_chain_halves_to_one_pixel() {
        let px = vec![128u8; 256 * 256 * 4];
        let mips = mip_chain(&px, 256, 256);
        assert_eq!(mips.len(), 9);
        assert_eq!((mips[0].w, mips[0].h), (256, 256));
        assert_eq!((mips[8].w, mips[8].h), (1, 1));
        for m in &mips {
            assert_eq!(m.px.len(), (m.w * m.h * 4) as usize);
        }
    }

    /// A flat colour must survive reduction unchanged. Averaging sRGB bytes
    /// directly passes this too, which is why the next test exists.
    #[test]
    fn mip_chain_preserves_a_flat_colour() {
        let px: Vec<u8> = [37u8, 140, 201, 255].repeat(64 * 64);
        let mips = mip_chain(&px, 64, 64);
        assert_eq!(&mips.last().unwrap().px[..], &[37, 140, 201, 255]);
    }

    /// Half black, half white must reduce to mid-grey in *linear* light
    /// (188 in sRGB), not to the byte average of 128.
    #[test]
    fn mip_chain_averages_in_linear_light() {
        let mut px = Vec::new();
        for _ in 0..(2 * 2) {
            px.extend_from_slice(&[0, 0, 0, 255]);
            px.extend_from_slice(&[255, 255, 255, 255]);
        }
        // 2 columns black, 2 white would give a 2x2 footprint of 2 black +
        // 2 white; lay it out as a 4x1 strip reduced twice instead.
        let mips = mip_chain(&px, 4, 2);
        let last = mips.last().unwrap();
        assert_eq!((last.w, last.h), (1, 1));
        assert!(
            (last.px[0] as i32 - 188).abs() <= 1,
            "got {}, expected ~188 (linear mid-grey)",
            last.px[0]
        );
    }
}
