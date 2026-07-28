use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::map::camera::Camera;
use crate::map::warp::{self, TileMesh, Warp, WarpPlan};

#[allow(dead_code)]
pub const TILE_PX: u32 = 256;
const MAX_CACHED: usize = 600;
const FETCH_THREADS: usize = 4;
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
    mips: Option<Vec<MipLevel>>,
}

pub struct TileCache {
    entries: HashMap<TileKey, CacheEntry>,
    req_tx: Sender<TileKey>,
    res_rx: Receiver<FetchResult>,
    pending_uploads: Vec<TileUpload>,
    frame: u64,
    /// Warped tile geometry, valid for one display projection only. Meshes
    /// are stable while panning and zooming, so they are built once per tile
    /// and thrown away when the projection changes.
    warp_meshes: HashMap<TileId, Option<Arc<TileMesh>>>,
    warp_epoch: u64,
}

/// Warped meshes kept across frames. Well above any one viewport, so panning
/// back and forth never rebuilds; cleared wholesale on a projection change.
const MAX_CACHED_MESHES: usize = 2048;

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
                let mips = fetch_tile(key);
                if tx.send(FetchResult { key, mips }).is_err() {
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
            warp_meshes: HashMap::new(),
            warp_epoch: 0,
        }
    }

    /// Drain results from fetch threads; call once per frame.
    pub fn poll(&mut self) {
        while let Ok(res) = self.res_rx.try_recv() {
            let state = match res.mips {
                Some(mips) => {
                    self.pending_uploads.push(TileUpload { key: res.key, mips });
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
        self.resolve(source_idx, wanted)
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
                mesh: None,
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

/// Fetch one tile synchronously. Only for the snapshot harness, which needs
/// real imagery to judge whether a reprojected basemap looks right.
#[cfg(test)]
pub fn fetch_tile_blocking(key: TileKey) -> Option<Vec<MipLevel>> {
    fetch_tile(key)
}

fn fetch_tile(key: TileKey) -> Option<Vec<MipLevel>> {
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
