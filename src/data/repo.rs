//! External GeoParquet repositories ("parquetry" layout).
//!
//! A repository is a plain HTTPS object store, no API:
//! - `{base}/snapshots.json` — dated snapshot prefixes (`2026-07-15/`);
//!   `latest/` always exists as an alias. Optional: absent means only
//!   `latest/`.
//! - `{base}/{snapshot}country=XX/state=XX-YY/_manifest.json` — one
//!   dataset folder per region, listing its themes and feature counts.
//! - `{base}/{snapshot}country=XX/state=XX-YY/{theme}.parquet` — one
//!   GeoParquet file per theme, loaded as a layer via range requests.
//!
//! Dataset discovery: `{base}/index.json` (`{"datasets": [{"path":
//! "country=US/state=US-AR", "name": "Arkansas"}]}`) when the repository
//! publishes one; otherwise a built-in ISO 3166-2 table (US/CA/MX) is
//! probed concurrently for `_manifest.json` files.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::source::http_agent;

const USER_AGENT: &str =
    concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

/// Repository protocol.
#[derive(Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum RepoKind {
    /// snapshots.json + per-folder _manifest.json + one file per theme.
    #[default]
    Parquetry,
    /// A static STAC catalog: releases → themes → type collections whose
    /// items are the parquet part files (Overture layout).
    Stac,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Repository {
    pub name: String,
    /// Base URL, no trailing slash.
    pub url: String,
    #[serde(default)]
    pub kind: RepoKind,
}

pub fn default_repos() -> Vec<Repository> {
    vec![
        Repository {
            name: "Geomermaids Parquetry (OSM North America)".into(),
            url: "https://parquetry.geomermaids.com".into(),
            kind: RepoKind::Parquetry,
        },
        Repository {
            name: "Overture Maps (STAC)".into(),
            url: "https://stac.overturemaps.org".into(),
            kind: RepoKind::Stac,
        },
    ]
}

/// `~/.config/geopq-viewer/repositories.json`. Tests read and write a
/// per-process temp directory instead — cache tests must never touch (or
/// depend on) the developer's real config.
fn config_file() -> Option<PathBuf> {
    #[cfg(test)]
    {
        return Some(
            std::env::temp_dir()
                .join(format!("geopq_test_config_{}", std::process::id()))
                .join("repositories.json"),
        );
    }
    #[cfg(not(test))]
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| {
            let cfg = PathBuf::from(h).join(".config");
            let dir = cfg.join("geopq-workbench");
            // One-time migration from the pre-rename directory.
            let old = cfg.join("geopq-viewer");
            if !dir.exists() && old.exists() {
                let _ = std::fs::rename(&old, &dir);
            }
            dir.join("repositories.json")
        })
}

pub fn load_repos() -> Vec<Repository> {
    let list: Option<Vec<Repository>> = config_file()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok());
    match list {
        Some(l) if !l.is_empty() => l,
        _ => default_repos(),
    }
}

pub fn save_repos(repos: &[Repository]) -> Result<(), String> {
    let path = config_file().ok_or("no home directory")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(repos).map_err(|e| e.to_string())?;
    write_atomic(&path, &json).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Truncate-then-write can leave an empty or partial file if the process
/// dies mid-write, and the `.ok()` readers would then silently fall back
/// to defaults (losing user-added repositories). Write a sibling temp file
/// and rename it into place — atomic on the same filesystem.
fn write_atomic(path: &Path, data: &str) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, data)?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// GET a JSON document; Ok(None) on 404 (absent is not an error for
/// optional repository files).
fn get_json(url: &str) -> Result<Option<Value>, String> {
    let res = http_agent()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| format!("cannot reach {url}: {e}"))?;
    match res.status().as_u16() {
        200 => {
            let body = res
                .into_body()
                .read_to_string()
                .map_err(|e| format!("read {url}: {e}"))?;
            serde_json::from_str(&body)
                .map(Some)
                .map_err(|e| format!("{url}: invalid JSON: {e}"))
        }
        404 => Ok(None),
        s => Err(format!("{url}: HTTP {s}")),
    }
}

/// HEAD probe: Ok(true) on 200, Ok(false) on any other HTTP status. A
/// transport error (after one retry) is Err — a network blip must never
/// read as "dataset absent", or discovery caches a silently incomplete
/// list under a fresh timestamp.
fn exists(url: &str) -> Result<bool, String> {
    let probe = || {
        http_agent()
            .head(url)
            .header("User-Agent", USER_AGENT)
            .call()
            .map(|r| r.status() == 200)
    };
    probe()
        .or_else(|_| probe())
        .map_err(|e| format!("cannot reach {url}: {e}"))
}

#[derive(Clone, PartialEq)]
pub struct Snapshot {
    /// Display label ("latest", "2026-07-15").
    pub label: String,
    /// URL path prefix under the base ("latest/", "2026-07-15/").
    pub path: String,
}

impl Snapshot {
    pub fn latest() -> Self {
        Self {
            label: "latest".into(),
            path: "latest/".into(),
        }
    }
}

/// Snapshots of a repository, newest first, "latest" always included.
/// A repository without snapshots.json just has "latest".
pub fn fetch_snapshots(base: &str) -> Result<Vec<Snapshot>, String> {
    let mut out = vec![Snapshot::latest()];
    if let Some(v) = get_json(&format!("{base}/snapshots.json"))? {
        for s in v.get("snapshots").and_then(Value::as_array).into_iter().flatten() {
            let (Some(date), Some(path)) = (
                s.get("date").and_then(Value::as_str),
                s.get("path").and_then(Value::as_str),
            ) else {
                continue;
            };
            out.push(Snapshot {
                label: date.to_string(),
                path: path.to_string(),
            });
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    /// ISO-style code ("US-AR"), for layer naming.
    pub code: String,
    /// Human name ("Arkansas").
    pub name: String,
    /// Folder path under the snapshot ("country=US/state=US-AR").
    pub path: String,
}

/// `~/.config/geopq-viewer/repo_cache.json`: discovered dataset lists per
/// (repository, snapshot) — discovery probes ~100 URLs, worth keeping
/// across sessions.
fn cache_file() -> Option<PathBuf> {
    config_file().map(|p| p.with_file_name("repo_cache.json"))
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    /// Unix seconds at fetch time.
    fetched_at: u64,
    datasets: Vec<Dataset>,
}

fn cache_key(base: &str, snapshot: &str) -> String {
    format!("{base}|{snapshot}")
}

/// Serializes read-modify-write cycles on the two cache files. The file
/// writes are atomic renames, but two concurrent RMWs still lose one
/// side's insert (each rewrites the whole map from its own snapshot) —
/// parallel tests hit this constantly, the app at worst re-fetches.
static CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn read_cache() -> std::collections::HashMap<String, CacheEntry> {
    cache_file()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_cache(cache: &std::collections::HashMap<String, CacheEntry>) {
    let Some(path) = cache_file() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_string(cache) {
        let _ = write_atomic(&path, &json);
    }
}

/// Cached dataset list with its age, if any.
pub fn cached_datasets(base: &str, snapshot: &str) -> Option<(Vec<Dataset>, u64)> {
    let _g = CACHE_LOCK.lock().unwrap();
    let e = read_cache().remove(&cache_key(base, snapshot))?;
    Some((e.datasets, e.fetched_at))
}

pub fn store_datasets(base: &str, snapshot: &str, datasets: &[Dataset]) {
    let _g = CACHE_LOCK.lock().unwrap();
    let mut cache = read_cache();
    cache.insert(
        cache_key(base, snapshot),
        CacheEntry {
            fetched_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            datasets: datasets.to_vec(),
        },
    );
    write_cache(&cache);
}

pub fn clear_cached_datasets(base: &str, snapshot: &str) {
    let _g = CACHE_LOCK.lock().unwrap();
    let mut cache = read_cache();
    if cache.remove(&cache_key(base, snapshot)).is_some() {
        write_cache(&cache);
    }
}

/// "5 min ago" / "3 h ago" / "2 d ago" for the cache indicator.
pub fn age_label(fetched_at: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = now.saturating_sub(fetched_at);
    if s < 120 {
        "just now".into()
    } else if s < 7200 {
        format!("{} min ago", s / 60)
    } else if s < 172_800 {
        format!("{} h ago", s / 3600)
    } else {
        format!("{} d ago", s / 86_400)
    }
}

/// Dataset folders of a snapshot: from the repository's `index.json` when
/// published, else by probing the built-in region table. Network-heavy
/// (one HEAD per candidate region on the probe path) — run off the UI
/// thread.
pub fn discover_datasets(base: &str, snapshot: &str) -> Result<Vec<Dataset>, String> {
    if let Some(v) = get_json(&format!("{base}/index.json"))? {
        let mut out = Vec::new();
        for d in v.get("datasets").and_then(Value::as_array).into_iter().flatten() {
            let Some(path) = d.get("path").and_then(Value::as_str) else {
                continue;
            };
            let code = path.rsplit('=').next().unwrap_or(path).to_string();
            out.push(Dataset {
                name: d
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&code)
                    .to_string(),
                code,
                path: path.trim_matches('/').to_string(),
            });
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }

    // No index published: probe the known region grid concurrently on a
    // small dedicated set of scoped threads — blocking HEADs must not
    // occupy the global rayon pool, which also runs decode/tessellation
    // (a slow repo host would stall map refinement).
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    const PROBE_THREADS: usize = 8;
    let next = AtomicUsize::new(0);
    let found = Mutex::new(Vec::<Dataset>::new());
    let first_err = Mutex::new(None::<String>);
    std::thread::scope(|s| {
        for _ in 0..PROBE_THREADS.min(REGIONS.len()) {
            s.spawn(|| loop {
                if first_err.lock().unwrap().is_some() {
                    break; // discovery already failed; stop probing
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(country, code, name)) = REGIONS.get(i) else {
                    break;
                };
                let path = format!("country={country}/state={code}");
                match exists(&format!("{base}/{snapshot}{path}/_manifest.json")) {
                    Ok(true) => found.lock().unwrap().push(Dataset {
                        code: code.to_string(),
                        name: name.to_string(),
                        path,
                    }),
                    Ok(false) => {}
                    // Fail the whole discovery rather than return (and
                    // cache) a partial list.
                    Err(e) => {
                        first_err.lock().unwrap().get_or_insert(e);
                    }
                }
            });
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(format!("discovery aborted: {e}"));
    }
    let mut found = found.into_inner().unwrap();
    found.sort_by(|a, b| a.code.cmp(&b.code));
    if found.is_empty() {
        return Err("no datasets found (no index.json and no known regions answered)".into());
    }
    Ok(found)
}

#[derive(Clone)]
pub struct Manifest {
    pub state_name: Option<String>,
    pub total_features: Option<u64>,
    /// (theme, feature count), manifest order.
    pub themes: Vec<(String, u64)>,
}

pub fn fetch_manifest(base: &str, snapshot: &str, path: &str) -> Result<Manifest, String> {
    let url = format!("{base}/{snapshot}{path}/_manifest.json");
    let v = get_json(&url)?.ok_or_else(|| format!("{url}: not found"))?;
    let themes = v
        .get("themes")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0)))
                .collect()
        })
        .unwrap_or_default();
    Ok(Manifest {
        state_name: v
            .get("state_name")
            .and_then(Value::as_str)
            .map(String::from),
        total_features: v.get("total_features").and_then(Value::as_u64),
        themes,
    })
}

pub fn theme_url(base: &str, snapshot: &str, path: &str, theme: &str) -> String {
    format!("{base}/{snapshot}{path}/{theme}.parquet")
}

// ---------------------------------------------------------------------
// STAC repositories (Overture layout)
// ---------------------------------------------------------------------
//
// {base}/catalog.json                 releases (children), `latest` field
// {base}/{release}/catalog.json       themes (children)
// {base}/{release}/{theme}/catalog.json      types (children)
// .../{type}/collection.json          items = one parquet part each
// .../{type}/NNNNN/NNNNN.json         item: bbox + num_rows + assets

/// Child directory names of a STAC catalog ("./2026-06-17.0/catalog.json"
/// → "2026-06-17.0"), in link order.
fn stac_children(cat: &Value) -> Vec<String> {
    cat.get("links")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|l| l.get("rel").and_then(Value::as_str) == Some("child"))
        .filter_map(|l| l.get("href").and_then(Value::as_str))
        .filter_map(|h| {
            let mut segs = h.trim_start_matches("./").split('/');
            let dir = segs.next()?;
            (!dir.is_empty()).then(|| dir.to_string())
        })
        .collect()
}

/// Releases of a STAC repository, latest first ("2026-06-17.0/", …).
/// Concrete directories only — no "latest" alias exists server-side.
pub fn fetch_snapshots_stac(base: &str) -> Result<Vec<Snapshot>, String> {
    let url = format!("{base}/catalog.json");
    let cat = get_json(&url)?.ok_or_else(|| format!("{url}: not found"))?;
    let latest = cat.get("latest").and_then(Value::as_str);
    let mut dirs = stac_children(&cat);
    if let Some(l) = latest {
        if let Some(pos) = dirs.iter().position(|d| d == l) {
            let d = dirs.remove(pos);
            dirs.insert(0, d);
        }
    }
    if dirs.is_empty() {
        return Err(format!("{url}: no releases listed"));
    }
    Ok(dirs
        .into_iter()
        .map(|d| Snapshot {
            label: d.clone(),
            path: format!("{d}/"),
        })
        .collect())
}

/// Turn the browser's placeholder "latest/" into the concrete release
/// directory (STAC has no server-side alias).
fn stac_resolve_snapshot(base: &str, snapshot: &str) -> Result<String, String> {
    if snapshot != "latest/" {
        return Ok(snapshot.to_string());
    }
    let url = format!("{base}/catalog.json");
    let cat = get_json(&url)?.ok_or_else(|| format!("{url}: not found"))?;
    match cat.get("latest").and_then(Value::as_str) {
        Some(l) => Ok(format!("{l}/")),
        None => stac_children(&cat)
            .first()
            .map(|d| format!("{d}/"))
            .ok_or_else(|| format!("{url}: no releases listed")),
    }
}

/// STAC datasets = the release's themes (one row per theme).
pub fn discover_datasets_stac(base: &str, snapshot: &str) -> Result<Vec<Dataset>, String> {
    let snap = stac_resolve_snapshot(base, snapshot)?;
    let url = format!("{base}/{snap}catalog.json");
    let cat = get_json(&url)?.ok_or_else(|| format!("{url}: not found"))?;
    let themes = stac_children(&cat);
    if themes.is_empty() {
        return Err(format!("{url}: no themes listed"));
    }
    Ok(themes
        .into_iter()
        .map(|t| Dataset {
            code: t.clone(),
            name: t.clone(),
            path: t,
        })
        .collect())
}

/// "Manifest" of a STAC theme: its type collections with part counts
/// (feature totals live in the per-part items — too many to fetch here).
pub fn fetch_stac_manifest(base: &str, snapshot: &str, theme: &str) -> Result<Manifest, String> {
    let snap = stac_resolve_snapshot(base, snapshot)?;
    let url = format!("{base}/{snap}{theme}/catalog.json");
    let cat = get_json(&url)?.ok_or_else(|| format!("{url}: not found"))?;
    let mut themes = Vec::new();
    for ty in stac_children(&cat) {
        let curl = format!("{base}/{snap}{theme}/{ty}/collection.json");
        let col = get_json(&curl)?.ok_or_else(|| format!("{curl}: not found"))?;
        let parts = col
            .get("links")
            .and_then(Value::as_array)
            .map(|ls| {
                ls.iter()
                    .filter(|l| l.get("rel").and_then(Value::as_str) == Some("item"))
                    .count()
            })
            .unwrap_or(0);
        themes.push((ty, parts as u64));
    }
    Ok(Manifest {
        state_name: Some(theme.to_string()),
        total_features: None,
        themes,
    })
}

/// collection.json URL of one type — the value a `Source::Stac` carries.
pub fn stac_collection_url(base: &str, snapshot: &str, theme: &str, ty: &str) -> String {
    format!("{base}/{snapshot}{theme}/{ty}/collection.json")
}

/// One parquet part file of a STAC collection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StacPart {
    /// HTTPS asset URL.
    pub url: String,
    /// Item bbox (WGS84 lon/lat), for part-level viewport pruning.
    pub bbox: Option<[f64; 4]>,
    pub rows: u64,
}

/// The parquet asset of a STAC item: prefer the `aws` asset, else the
/// first https parquet one.
fn stac_item_asset(item: &Value) -> Option<String> {
    let assets = item.get("assets").and_then(Value::as_object)?;
    let href = |a: &Value| {
        a.get("href")
            .and_then(Value::as_str)
            .filter(|h| h.starts_with("http://") || h.starts_with("https://"))
            .map(String::from)
    };
    if let Some(a) = assets.get("aws").and_then(href) {
        return Some(a);
    }
    assets.values().find_map(|a| {
        let is_parquet = a.get("type").and_then(Value::as_str)
            == Some("application/vnd.apache.parquet")
            || a.get("href")
                .and_then(Value::as_str)
                .is_some_and(|h| h.ends_with(".parquet"));
        if is_parquet {
            href(a)
        } else {
            None
        }
    })
}

/// STAC bbox → 2D: 6 elements is [xmin, ymin, zmin, xmax, ymax, zmax].
fn stac_bbox_2d(v: &Value) -> Option<[f64; 4]> {
    let b: Vec<f64> = v.as_array()?.iter().filter_map(Value::as_f64).collect();
    match b.len() {
        4 => Some([b[0], b[1], b[2], b[3]]),
        6 => Some([b[0], b[1], b[3], b[4]]),
        _ => None,
    }
}

/// `~/.config/geopq-viewer/stac_parts_cache.json`: part lists per
/// collection URL. STAC releases live under dated, immutable prefixes,
/// so entries stay valid until the ⟳ button clears them.
fn parts_cache_file() -> Option<PathBuf> {
    config_file().map(|p| p.with_file_name("stac_parts_cache.json"))
}

fn read_parts_cache() -> std::collections::HashMap<String, Vec<StacPart>> {
    parts_cache_file()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_parts_cache(cache: &std::collections::HashMap<String, Vec<StacPart>>) {
    let Some(path) = parts_cache_file() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_string(cache) {
        let _ = write_atomic(&path, &json);
    }
}

/// Drop cached part lists of one repository (all its collections).
pub fn clear_cached_stac_parts(base: &str) {
    let _g = CACHE_LOCK.lock().unwrap();
    let mut cache = read_parts_cache();
    let n = cache.len();
    cache.retain(|url, _| !url.starts_with(base));
    if cache.len() != n {
        write_parts_cache(&cache);
    }
}

/// All parts of a type collection, served from the on-disk cache when
/// present (release prefixes are immutable), else from its item documents
/// (parallel fetch on dedicated threads; any failure aborts — a silently
/// partial part list would silently drop data).
pub fn fetch_stac_parts(collection_url: &str) -> Result<Vec<StacPart>, String> {
    {
        let _g = CACHE_LOCK.lock().unwrap();
        if let Some(parts) = read_parts_cache().remove(collection_url) {
            return Ok(parts);
        }
    }
    // Fetch outside the lock: item documents can take a while, and the
    // lock only has to make the read-modify-write below atomic.
    let parts = fetch_stac_parts_live(collection_url)?;
    let _g = CACHE_LOCK.lock().unwrap();
    let mut cache = read_parts_cache();
    cache.insert(collection_url.to_string(), parts.clone());
    write_parts_cache(&cache);
    Ok(parts)
}

fn fetch_stac_parts_live(collection_url: &str) -> Result<Vec<StacPart>, String> {
    let col = get_json(collection_url)?.ok_or_else(|| format!("{collection_url}: not found"))?;
    let dir = collection_url
        .rsplit_once('/')
        .map(|(d, _)| d)
        .unwrap_or(collection_url);
    let item_urls: Vec<String> = col
        .get("links")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|l| l.get("rel").and_then(Value::as_str) == Some("item"))
        .filter_map(|l| l.get("href").and_then(Value::as_str))
        .map(|h| format!("{dir}/{}", h.trim_start_matches("./")))
        .collect();
    if item_urls.is_empty() {
        return Err(format!("{collection_url}: no items listed"));
    }

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    const FETCH_THREADS: usize = 8;
    let next = AtomicUsize::new(0);
    let parts = Mutex::new(vec![None::<StacPart>; item_urls.len()]);
    let first_err = Mutex::new(None::<String>);
    std::thread::scope(|s| {
        for _ in 0..FETCH_THREADS.min(item_urls.len()) {
            s.spawn(|| loop {
                if first_err.lock().unwrap().is_some() {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(url) = item_urls.get(i) else { break };
                let item = match get_json(url) {
                    Ok(Some(v)) => v,
                    Ok(None) => {
                        first_err.lock().unwrap().get_or_insert(format!("{url}: not found"));
                        continue;
                    }
                    Err(e) => {
                        first_err.lock().unwrap().get_or_insert(e);
                        continue;
                    }
                };
                let Some(asset) = stac_item_asset(&item) else {
                    first_err
                        .lock()
                        .unwrap()
                        .get_or_insert(format!("{url}: no parquet asset"));
                    continue;
                };
                parts.lock().unwrap()[i] = Some(StacPart {
                    url: asset,
                    bbox: item.get("bbox").and_then(stac_bbox_2d),
                    rows: item
                        .get("properties")
                        .and_then(|p| p.get("num_rows"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                });
            });
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(format!("listing parts: {e}"));
    }
    Ok(parts.into_inner().unwrap().into_iter().flatten().collect())
}

/// ISO 3166-2 regions probed when a repository has no index.json.
/// 404s drop out, so over-listing is harmless.
const REGIONS: &[(&str, &str, &str)] = &[
    ("US", "US-AL", "Alabama"),
    ("US", "US-AK", "Alaska"),
    ("US", "US-AZ", "Arizona"),
    ("US", "US-AR", "Arkansas"),
    ("US", "US-CA", "California"),
    ("US", "US-CO", "Colorado"),
    ("US", "US-CT", "Connecticut"),
    ("US", "US-DE", "Delaware"),
    ("US", "US-FL", "Florida"),
    ("US", "US-GA", "Georgia"),
    ("US", "US-HI", "Hawaii"),
    ("US", "US-ID", "Idaho"),
    ("US", "US-IL", "Illinois"),
    ("US", "US-IN", "Indiana"),
    ("US", "US-IA", "Iowa"),
    ("US", "US-KS", "Kansas"),
    ("US", "US-KY", "Kentucky"),
    ("US", "US-LA", "Louisiana"),
    ("US", "US-ME", "Maine"),
    ("US", "US-MD", "Maryland"),
    ("US", "US-MA", "Massachusetts"),
    ("US", "US-MI", "Michigan"),
    ("US", "US-MN", "Minnesota"),
    ("US", "US-MS", "Mississippi"),
    ("US", "US-MO", "Missouri"),
    ("US", "US-MT", "Montana"),
    ("US", "US-NE", "Nebraska"),
    ("US", "US-NV", "Nevada"),
    ("US", "US-NH", "New Hampshire"),
    ("US", "US-NJ", "New Jersey"),
    ("US", "US-NM", "New Mexico"),
    ("US", "US-NY", "New York"),
    ("US", "US-NC", "North Carolina"),
    ("US", "US-ND", "North Dakota"),
    ("US", "US-OH", "Ohio"),
    ("US", "US-OK", "Oklahoma"),
    ("US", "US-OR", "Oregon"),
    ("US", "US-PA", "Pennsylvania"),
    ("US", "US-RI", "Rhode Island"),
    ("US", "US-SC", "South Carolina"),
    ("US", "US-SD", "South Dakota"),
    ("US", "US-TN", "Tennessee"),
    ("US", "US-TX", "Texas"),
    ("US", "US-UT", "Utah"),
    ("US", "US-VT", "Vermont"),
    ("US", "US-VA", "Virginia"),
    ("US", "US-WA", "Washington"),
    ("US", "US-WV", "West Virginia"),
    ("US", "US-WI", "Wisconsin"),
    ("US", "US-WY", "Wyoming"),
    ("US", "US-DC", "District of Columbia"),
    ("US", "US-PR", "Puerto Rico"),
    ("US", "US-GU", "Guam"),
    ("US", "US-VI", "U.S. Virgin Islands"),
    ("US", "US-AS", "American Samoa"),
    ("US", "US-MP", "Northern Mariana Islands"),
    ("CA", "CA-AB", "Alberta"),
    ("CA", "CA-BC", "British Columbia"),
    ("CA", "CA-MB", "Manitoba"),
    ("CA", "CA-NB", "New Brunswick"),
    ("CA", "CA-NL", "Newfoundland and Labrador"),
    ("CA", "CA-NS", "Nova Scotia"),
    ("CA", "CA-NT", "Northwest Territories"),
    ("CA", "CA-NU", "Nunavut"),
    ("CA", "CA-ON", "Ontario"),
    ("CA", "CA-PE", "Prince Edward Island"),
    ("CA", "CA-QC", "Quebec"),
    ("CA", "CA-SK", "Saskatchewan"),
    ("CA", "CA-YT", "Yukon"),
    ("MX", "MX-AGU", "Aguascalientes"),
    ("MX", "MX-BCN", "Baja California"),
    ("MX", "MX-BCS", "Baja California Sur"),
    ("MX", "MX-CAM", "Campeche"),
    ("MX", "MX-CHP", "Chiapas"),
    ("MX", "MX-CHH", "Chihuahua"),
    ("MX", "MX-CMX", "Ciudad de México"),
    ("MX", "MX-COA", "Coahuila"),
    ("MX", "MX-COL", "Colima"),
    ("MX", "MX-DUR", "Durango"),
    ("MX", "MX-GUA", "Guanajuato"),
    ("MX", "MX-GRO", "Guerrero"),
    ("MX", "MX-HID", "Hidalgo"),
    ("MX", "MX-JAL", "Jalisco"),
    ("MX", "MX-MEX", "México"),
    ("MX", "MX-MIC", "Michoacán"),
    ("MX", "MX-MOR", "Morelos"),
    ("MX", "MX-NAY", "Nayarit"),
    ("MX", "MX-NLE", "Nuevo León"),
    ("MX", "MX-OAX", "Oaxaca"),
    ("MX", "MX-PUE", "Puebla"),
    ("MX", "MX-QUE", "Querétaro"),
    ("MX", "MX-ROO", "Quintana Roo"),
    ("MX", "MX-SLP", "San Luis Potosí"),
    ("MX", "MX-SIN", "Sinaloa"),
    ("MX", "MX-SON", "Sonora"),
    ("MX", "MX-TAB", "Tabasco"),
    ("MX", "MX-TAM", "Tamaulipas"),
    ("MX", "MX-TLA", "Tlaxcala"),
    ("MX", "MX-VER", "Veracruz"),
    ("MX", "MX-YUC", "Yucatán"),
    ("MX", "MX-ZAC", "Zacatecas"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Loopback HTTP responder: answers every request with `status` and an
    /// empty body, counting requests. Returns the base URL (no path).
    fn spawn_status_server(status: u16) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                h.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 2048];
                let _ = conn.read(&mut buf);
                let _ = write!(
                    conn,
                    "HTTP/1.1 {status} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        (format!("http://127.0.0.1:{port}"), hits)
    }

    #[test]
    fn exists_distinguishes_status_from_transport() {
        let (base, _) = spawn_status_server(200);
        assert_eq!(exists(&format!("{base}/x")), Ok(true));
        let (base, _) = spawn_status_server(404);
        assert_eq!(exists(&format!("{base}/x")), Ok(false));
        let (base, _) = spawn_status_server(500);
        assert_eq!(exists(&format!("{base}/x")), Ok(false));
        // A refused connection is a transport error, never "absent".
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert!(exists(&format!("http://127.0.0.1:{port}/x")).is_err());
    }

    #[test]
    fn exists_retries_transport_error_once() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut first = true;
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                if first {
                    first = false;
                    continue; // drop without answering: transport error
                }
                let mut buf = [0u8; 2048];
                let _ = conn.read(&mut buf);
                let _ = write!(
                    conn,
                    "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        assert_eq!(exists(&format!("http://127.0.0.1:{port}/x")), Ok(true));
    }

    #[test]
    fn discover_probes_all_regions_off_the_global_pool() {
        // 404 everywhere: index.json is absent, every manifest probe says
        // "absent" — the dedicated probe pool must drain the full grid and
        // report "no datasets", not a transport failure.
        let (base, hits) = spawn_status_server(404);
        let err = discover_datasets(&base, "latest/").unwrap_err();
        assert!(err.contains("no datasets found"), "{err}");
        assert!(hits.load(Ordering::SeqCst) > REGIONS.len(), "index.json + one HEAD per region");
    }

    #[test]
    fn discover_fails_loud_on_unreachable_host() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert!(discover_datasets(&format!("http://127.0.0.1:{port}"), "latest/").is_err());
    }

    #[test]
    fn write_atomic_replaces_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("geopq_repo_atomic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("repositories.json");
        write_atomic(&path, "[1]").unwrap();
        write_atomic(&path, "[1,2]").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[1,2]");
        // No temp file left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        // A missing parent directory surfaces as an error, not a silent no-op.
        let gone = dir.join("sub").join("x.json");
        assert!(write_atomic(&gone, "x").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repo_roundtrip_serde() {
        let repos = default_repos();
        let json = serde_json::to_string(&repos).unwrap();
        let back: Vec<Repository> = serde_json::from_str(&json).unwrap();
        assert_eq!(back[0].url, "https://parquetry.geomermaids.com");
    }

    fn write_json(root: &Path, rel: &str, v: serde_json::Value) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, v.to_string()).unwrap();
    }

    /// A minimal Overture-shaped STAC tree on disk, served over loopback.
    fn spawn_stac_fixture() -> (String, std::path::PathBuf) {
        use serde_json::json;
        // Unique per call: tests run in parallel in one process.
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "geopq_stac_fix_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst),
        ));
        let _ = std::fs::remove_dir_all(&root);
        write_json(
            &root,
            "catalog.json",
            json!({"latest": "2026-06", "links": [
                {"rel": "root", "href": "./catalog.json"},
                {"rel": "child", "href": "./2026-05/catalog.json"},
                {"rel": "child", "href": "./2026-06/catalog.json"},
            ]}),
        );
        write_json(
            &root,
            "2026-06/catalog.json",
            json!({"links": [
                {"rel": "child", "href": "./buildings/catalog.json"},
                {"rel": "child", "href": "./places/catalog.json"},
            ]}),
        );
        write_json(
            &root,
            "2026-06/buildings/catalog.json",
            json!({"links": [
                {"rel": "child", "href": "./building/collection.json"},
                {"rel": "child", "href": "./building_part/collection.json"},
            ]}),
        );
        write_json(
            &root,
            "2026-06/buildings/building/collection.json",
            json!({"links": [
                {"rel": "item", "href": "./00000/00000.json"},
                {"rel": "item", "href": "./00001/00001.json"},
            ]}),
        );
        write_json(
            &root,
            "2026-06/buildings/building_part/collection.json",
            json!({"links": [{"rel": "item", "href": "./00000/00000.json"}]}),
        );
        write_json(
            &root,
            "2026-06/buildings/building/00000/00000.json",
            json!({
                "bbox": [-10.0, 40.0, -5.0, 45.0],
                "properties": {"num_rows": 123},
                "assets": {
                    "azure": {"href": "https://azure.example/p0.parquet"},
                    "aws": {"href": "https://aws.example/p0.parquet"},
                }
            }),
        );
        write_json(
            &root,
            "2026-06/buildings/building/00001/00001.json",
            json!({
                // 6-element bbox: [xmin, ymin, zmin, xmax, ymax, zmax].
                "bbox": [5.0, 40.0, 0.0, 10.0, 45.0, 100.0],
                "properties": {"num_rows": 456},
                "assets": {
                    "data": {
                        "href": "https://other.example/p1.parquet",
                        "type": "application/vnd.apache.parquet",
                    }
                }
            }),
        );
        let base = crate::data::source::testserver::spawn_dir(root.clone());
        (base, root)
    }

    #[test]
    fn stac_snapshots_latest_first_and_alias_resolution() {
        let (base, _root) = spawn_stac_fixture();
        let snaps = fetch_snapshots_stac(&base).unwrap();
        assert_eq!(
            snaps.iter().map(|s| s.path.as_str()).collect::<Vec<_>>(),
            vec!["2026-06/", "2026-05/"],
            "latest release first"
        );
        // The browser's placeholder resolves through the root catalog.
        let ds = discover_datasets_stac(&base, "latest/").unwrap();
        assert_eq!(
            ds.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
            vec!["buildings", "places"]
        );
    }

    #[test]
    fn stac_manifest_lists_types_with_part_counts() {
        let (base, _root) = spawn_stac_fixture();
        let m = fetch_stac_manifest(&base, "2026-06/", "buildings").unwrap();
        assert_eq!(m.state_name.as_deref(), Some("buildings"));
        assert_eq!(
            m.themes,
            vec![("building".to_string(), 2), ("building_part".to_string(), 1)]
        );
    }

    #[test]
    fn stac_parts_prefer_aws_and_parse_bboxes() {
        let (base, _root) = spawn_stac_fixture();
        let url = stac_collection_url(&base, "2026-06/", "buildings", "building");
        let parts = fetch_stac_parts(&url).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].url, "https://aws.example/p0.parquet");
        assert_eq!(parts[0].bbox, Some([-10.0, 40.0, -5.0, 45.0]));
        assert_eq!(parts[0].rows, 123);
        // Generic parquet asset accepted; 6-element bbox collapsed to 2D.
        assert_eq!(parts[1].url, "https://other.example/p1.parquet");
        assert_eq!(parts[1].bbox, Some([5.0, 40.0, 10.0, 45.0]));
    }

    #[test]
    fn stac_parts_come_from_cache_after_first_fetch() {
        let (base, root) = spawn_stac_fixture();
        let url = stac_collection_url(&base, "2026-06/", "buildings", "building");
        let live = fetch_stac_parts(&url).unwrap();
        assert_eq!(live.len(), 2);
        // Remove the item documents: a second fetch can only succeed from
        // the on-disk cache (release prefixes are immutable).
        std::fs::remove_dir_all(root.join("2026-06/buildings/building/00000")).unwrap();
        std::fs::remove_dir_all(root.join("2026-06/buildings/building/00001")).unwrap();
        let cached = fetch_stac_parts(&url).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].url, live[0].url);
        assert_eq!(cached[1].bbox, live[1].bbox);
        // ⟳ clears the repository's entries; the next fetch goes live again.
        clear_cached_stac_parts(&base);
        assert!(fetch_stac_parts(&url).is_err(), "cache cleared, items gone");
    }

    #[test]
    fn stac_parts_fail_loud_on_missing_item() {
        let (base, _root) = spawn_stac_fixture();
        // building_part's collection lists an item that was never written.
        let url = stac_collection_url(&base, "2026-06/", "buildings", "building_part");
        let err = fetch_stac_parts(&url).unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn theme_url_shape() {
        assert_eq!(
            theme_url(
                "https://parquetry.geomermaids.com",
                "latest/",
                "country=US/state=US-AR",
                "buildings"
            ),
            "https://parquetry.geomermaids.com/latest/country=US/state=US-AR/buildings.parquet"
        );
    }

    /// Live repository probe, opt-in:
    /// cargo test --release repo_live -- --ignored --nocapture
    #[test]
    #[ignore]
    fn repo_live() {
        let base = "https://parquetry.geomermaids.com";
        let snaps = fetch_snapshots(base).unwrap();
        eprintln!("{} snapshots (incl. latest)", snaps.len());
        assert!(snaps.len() > 1);
        let t0 = std::time::Instant::now();
        let ds = discover_datasets(base, "latest/").unwrap();
        eprintln!("{} datasets discovered in {} ms", ds.len(), t0.elapsed().as_millis());
        assert!(ds.iter().any(|d| d.code == "US-AR"));
        let m = fetch_manifest(base, "latest/", "country=US/state=US-AR").unwrap();
        eprintln!("US-AR: {:?}, {} themes", m.state_name, m.themes.len());
        assert!(m.themes.iter().any(|(t, _)| t == "buildings"));
    }
}
