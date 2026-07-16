use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::map::camera::Camera;

#[allow(dead_code)]
pub const TILE_PX: u32 = 256;
const MAX_CACHED: usize = 600;
const FETCH_THREADS: usize = 4;
const USER_AGENT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/gsueur/geopq-viewer)"
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
}

pub const TILE_SOURCES: &[TileSource] = &[
    TileSource {
        name: "Carto Light",
        url: "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
    },
    TileSource {
        name: "Carto Dark",
        url: "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
    },
    TileSource {
        name: "Carto Voyager",
        url: "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors © CARTO",
        max_zoom: 20,
    },
    TileSource {
        name: "OpenStreetMap",
        url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution: "© OpenStreetMap contributors",
        max_zoom: 19,
    },
];

enum TileState {
    Pending,
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
}

pub struct TileUpload {
    pub key: TileKey,
    pub rgba: image::RgbaImage,
}

pub struct TileDrawCmd {
    pub key: TileKey,
    pub world_rect: [f64; 4],
}

struct FetchResult {
    key: TileKey,
    rgba: Option<image::RgbaImage>,
}

pub struct TileCache {
    entries: HashMap<TileKey, CacheEntry>,
    req_tx: Sender<TileKey>,
    res_rx: Receiver<FetchResult>,
    pending_uploads: Vec<TileUpload>,
    frame: u64,
}

impl TileCache {
    pub fn new(egui_ctx: eframe::egui::Context) -> Self {
        let (req_tx, req_rx) = channel::<TileKey>();
        let (res_tx, res_rx) = channel::<FetchResult>();
        let shared_rx = Arc::new(Mutex::new(req_rx));
        for _ in 0..FETCH_THREADS {
            let rx = shared_rx.clone();
            let tx = res_tx.clone();
            let ctx = egui_ctx.clone();
            std::thread::spawn(move || loop {
                let key = {
                    let guard = rx.lock().unwrap();
                    guard.recv()
                };
                let Ok(key) = key else { return };
                let rgba = fetch_tile(key);
                if tx.send(FetchResult { key, rgba }).is_err() {
                    return;
                }
                ctx.request_repaint();
            });
        }
        Self {
            entries: HashMap::new(),
            req_tx,
            res_rx,
            pending_uploads: Vec::new(),
            frame: 0,
        }
    }

    /// Drain results from fetch threads; call once per frame.
    pub fn poll(&mut self) {
        while let Ok(res) = self.res_rx.try_recv() {
            let state = match res.rgba {
                Some(rgba) => {
                    self.pending_uploads.push(TileUpload { key: res.key, rgba });
                    TileState::Ready
                }
                None => {
                    let attempts = match self.entries.get(&res.key).map(|e| &e.state) {
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
            } else {
                self.entries.insert(
                    res.key,
                    CacheEntry {
                        state,
                        last_used: self.frame,
                    },
                );
            }
        }
    }

    pub fn take_uploads(&mut self) -> Vec<TileUpload> {
        std::mem::take(&mut self.pending_uploads)
    }

    /// Keys that renderers may keep GPU textures for.
    pub fn alive_keys(&self) -> HashSet<TileKey> {
        self.entries
            .iter()
            .filter(|(_, e)| matches!(e.state, TileState::Ready))
            .map(|(k, _)| *k)
            .collect()
    }

    /// Compute draw list for the current view, requesting missing tiles and
    /// substituting loaded ancestors while children load.
    pub fn draws(
        &mut self,
        source_idx: usize,
        camera: &Camera,
        viewport_px: [f32; 2],
    ) -> Vec<TileDrawCmd> {
        self.frame += 1;
        let source = &TILE_SOURCES[source_idx];
        let z = (camera.zoom.round() as i64).clamp(0, source.max_zoom as i64) as u8;
        let n = 1u64 << z;

        let tl = camera.screen_to_world([0.0, 0.0], viewport_px);
        let br = camera.screen_to_world(viewport_px, viewport_px);
        let x0 = ((tl[0] * n as f64).floor() as i64).max(0);
        let x1 = ((br[0] * n as f64).ceil() as i64).min(n as i64);
        let y0 = ((tl[1] * n as f64).floor() as i64).max(0);
        let y1 = ((br[1] * n as f64).ceil() as i64).min(n as i64);

        let mut wanted: Vec<TileId> = Vec::new();
        for x in x0..x1 {
            for y in y0..y1 {
                wanted.push(TileId {
                    z,
                    x: x as u32,
                    y: y as u32,
                });
            }
        }
        // Sanity cap in case of a degenerate viewport.
        if wanted.len() > 512 {
            wanted.truncate(512);
        }

        let mut draws: Vec<TileDrawCmd> = Vec::new();
        let mut fallback: Vec<TileId> = Vec::new();

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
                        }),
                        TileState::Pending => fallback.push(*id),
                        TileState::Failed { at, attempts } => {
                            let backoff =
                                std::time::Duration::from_secs(1u64 << attempts.min(6));
                            if attempts < TILE_RETRY_MAX && at.elapsed() >= backoff {
                                e.state = TileState::Pending;
                                let _ = self.req_tx.send(key);
                            }
                            fallback.push(*id);
                        }
                    }
                }
                None => {
                    self.entries.insert(
                        key,
                        CacheEntry {
                            state: TileState::Pending,
                            last_used: self.frame,
                        },
                    );
                    let _ = self.req_tx.send(key);
                    fallback.push(*id);
                }
            }
        }

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
            })
            .collect();
        all.append(&mut draws);

        self.evict();
        all
    }

    fn evict(&mut self) {
        if self.entries.len() <= MAX_CACHED {
            return;
        }
        let mut by_age: VecDeque<(u64, TileKey)> = self
            .entries
            .iter()
            .map(|(k, e)| (e.last_used, *k))
            .collect();
        by_age.make_contiguous().sort();
        let excess = self.entries.len() - MAX_CACHED;
        for (_, key) in by_age.iter().take(excess) {
            self.entries.remove(key);
        }
    }

    /// Number of tiles currently being fetched.
    pub fn pending_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e.state, TileState::Pending))
            .count()
    }
}

fn fetch_tile(key: TileKey) -> Option<image::RgbaImage> {
    let source = &TILE_SOURCES[key.source as usize];
    let url = source
        .url
        .replace("{z}", &key.id.z.to_string())
        .replace("{x}", &key.id.x.to_string())
        .replace("{y}", &key.id.y.to_string());
    let mut res = ureq::get(&url)
        .header("User-Agent", USER_AGENT)
        .call()
        .ok()?;
    let bytes = res.body_mut().read_to_vec().ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    Some(img.to_rgba8())
}
