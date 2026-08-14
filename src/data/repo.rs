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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::source::http_agent;

const USER_AGENT: &str =
    concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

/// Concurrency for the many-small-documents fetches a STAC tree needs.
const FETCH_THREADS: usize = 8;

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
    /// Credit the publisher requires, for data that carries none of its
    /// own. A STAC catalog has no `ATTRIBUTION.txt` and its collections
    /// declare a licence but no provider, so without this an Overture
    /// layer showed no credit at all. For a repository that does publish
    /// a sidecar this is the fallback for when it cannot be reached.
    #[serde(default)]
    pub attribution: Option<String>,
    /// Credit to use instead of `attribution` when the data declares one
    /// of these licences, by SPDX identifier.
    ///
    /// Overture applies ODbL precisely where OpenStreetMap is in the mix,
    /// and OSM's licence asks for its contributors by name — so an ODbL
    /// collection needs a different line from the CC0 one next to it.
    #[serde(default)]
    pub attribution_by_license: BTreeMap<String, String>,
}

impl Repository {
    /// The credit this repository requires for data published under
    /// `license`, most specific first.
    pub fn credit_for(&self, license: Option<&str>) -> Option<&str> {
        license
            .and_then(|l| {
                self.attribution_by_license
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(l))
                    .map(|(_, v)| v)
            })
            .or(self.attribution.as_ref())
            .map(String::as_str)
    }
}

/// The credit a configured repository requires for a URL under it, when
/// the data carries none of its own.
///
/// The longest matching base wins, and does not fall through to a shorter
/// one: `parquetry.geomermaids.com/geoboundaries` sits under the OSM
/// repository's base, and crediting geoBoundaries polygons to OpenStreetMap
/// would be worse than crediting them to nobody.
pub fn credit_for_url(url: &str, license: Option<&str>) -> Option<String> {
    load_repos()
        .into_iter()
        .filter(|r| url.starts_with(&r.url))
        .max_by_key(|r| r.url.len())
        .and_then(|r| r.credit_for(license).map(str::to_string))
}

pub fn default_repos() -> Vec<Repository> {
    vec![
        Repository {
            name: "Geomermaids Parquetry (OSM North America)".into(),
            url: "https://parquetry.geomermaids.com".into(),
            kind: RepoKind::Parquetry,
            // This one publishes an ATTRIBUTION.txt at its root saying the
            // same thing, and that is what normally supplies the credit.
            // Here it is the answer for when the sidecar cannot be
            // fetched: OSM's licence does not stop applying because a
            // request failed.
            attribution: Some("© OpenStreetMap contributors".into()),
            attribution_by_license: BTreeMap::new(),
        },
        Repository {
            name: "Geomermaids geoBoundaries (global admin)".into(),
            url: "https://parquetry.geomermaids.com/geoboundaries".into(),
            kind: RepoKind::Parquetry,
            attribution: None,
            attribution_by_license: BTreeMap::new(),
        },
        Repository {
            name: "Geomermaids CORINE Land Cover (Europe)".into(),
            url: "https://parquetry.geomermaids.com/clc".into(),
            kind: RepoKind::Parquetry,
            attribution: None,
            attribution_by_license: BTreeMap::new(),
        },
        Repository {
            name: "Overture Maps (STAC)".into(),
            url: "https://stac.overturemaps.org".into(),
            kind: RepoKind::Stac,
            // The citation Overture's own attribution page gives. The
            // licence comes from each collection, which is where the
            // themes differ: buildings are ODbL, base/land_cover CC BY
            // 4.0, base/bathymetry CC0.
            attribution: Some("Overture Maps Foundation, overturemaps.org".into()),
            // Overture's ODbL themes carry OpenStreetMap data, and their
            // attribution page asks for this line when they do.
            attribution_by_license: BTreeMap::from([(
                "ODbL-1.0".to_string(),
                "© OpenStreetMap contributors, Overture Maps Foundation".to_string(),
            )]),
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
    let mut out = Vec::new();
    if let Some(v) = get_json(&format!("{base}/snapshots.json"))? {
        // A repository that versions its data (a dataset release rather
        // than a rolling mirror) has no `latest/` prefix to offer, and
        // an entry pointing at a 404 is worse than none. `"latest"` in
        // snapshots.json aliases it onto a real prefix.
        match v.get("latest").and_then(Value::as_str) {
            Some(p) => out.push(Snapshot {
                label: "latest".into(),
                path: p.to_string(),
            }),
            None => out.push(Snapshot::latest()),
        }
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
    if out.is_empty() {
        out.push(Snapshot::latest());
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
    let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let e = read_cache().remove(&cache_key(base, snapshot))?;
    Some((e.datasets, e.fetched_at))
}

pub fn store_datasets(base: &str, snapshot: &str, datasets: &[Dataset]) {
    let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
            let code = d
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or_else(|| path.rsplit('=').next().unwrap_or(path))
                .to_string();
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

/// Prefix of a dataset's files, snapshot included. A repository holding
/// one global dataset publishes it directly under the snapshot with an
/// empty path, and an empty path segment must not become `//`, which
/// object storage reads as a different key.
fn dataset_prefix(snapshot: &str, path: &str) -> String {
    if path.is_empty() {
        snapshot.to_string()
    } else {
        format!("{snapshot}{path}/")
    }
}

pub fn fetch_manifest(base: &str, snapshot: &str, path: &str) -> Result<Manifest, String> {
    let url = format!("{base}/{}_manifest.json", dataset_prefix(snapshot, path));
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
    format!("{base}/{}{theme}.parquet", dataset_prefix(snapshot, path))
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

/// What a STAC collection document says about itself, apart from the list
/// of items: everything the app needs from it that is not a part.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StacCollection {
    /// The `license` field: an SPDX identifier ("ODbL-1.0", "CC-BY-4.0",
    /// "CC0-1.0") or the spec's "other"/"proprietary" placeholders.
    pub license: Option<String>,
    /// `providers` entries that hold the rights (licensor or producer),
    /// most authoritative first. Overture declares none; other catalogs do.
    #[serde(default)]
    pub providers: Vec<String>,
    /// How many parts the collection has, for the browser's part count.
    pub parts: u64,
}

/// `~/.config/geopq-workbench/stac_collections_cache.json`: one entry per
/// collection URL. Same immutability argument as the part cache — release
/// prefixes are dated and never rewritten — and cleared by the same ⟳.
fn collections_cache_file() -> Option<PathBuf> {
    config_file().map(|p| p.with_file_name("stac_collections_cache.json"))
}

fn read_collections_cache() -> std::collections::HashMap<String, StacCollection> {
    collections_cache_file()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_collections_cache(cache: &std::collections::HashMap<String, StacCollection>) {
    let Some(path) = collections_cache_file() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_string(cache) {
        let _ = write_atomic(&path, &json);
    }
}

fn parse_stac_collection(col: &Value) -> StacCollection {
    let items = col
        .get("links")
        .and_then(Value::as_array)
        .map(|ls| {
            ls.iter()
                .filter(|l| l.get("rel").and_then(Value::as_str) == Some("item"))
                .count()
        })
        .unwrap_or(0);
    // A collection-only document (no Items) carries its parts as assets,
    // and counting only item links reported such a dataset as empty.
    let parts = if items > 0 {
        items
    } else {
        col.get("assets")
            .and_then(Value::as_object)
            .map(|a| a.values().filter(|v| is_parquet_asset(v)).count())
            .unwrap_or(0)
    } as u64;
    // Rights holders first: a "processor" that merely reformatted the data
    // is not who the licence asks you to credit.
    let mut providers: Vec<String> = Vec::new();
    for p in col.get("providers").and_then(Value::as_array).into_iter().flatten() {
        let Some(name) = p.get("name").and_then(Value::as_str) else {
            continue;
        };
        let roles: Vec<&str> = p
            .get("roles")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if roles.is_empty() || roles.iter().any(|r| *r == "licensor" || *r == "producer") {
            providers.push(name.to_string());
        }
    }
    StacCollection {
        license: col
            .get("license")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|l| !l.is_empty()),
        providers,
        parts,
    }
}

/// A collection's own description, from the on-disk cache when present.
///
/// These documents are mostly their item list — Overture's buildings
/// collection is 123 KB of links — and the browser used to refetch every
/// type's copy on every click just to count them. Everything the app takes
/// from one is small and immutable, so it is worth keeping.
pub fn fetch_stac_collection(collection_url: &str) -> Result<StacCollection, String> {
    {
        let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = read_collections_cache().remove(collection_url) {
            return Ok(c);
        }
    }
    let col = get_json(collection_url)?
        .ok_or_else(|| format!("{collection_url}: not found"))?;
    let parsed = parse_stac_collection(&col);
    let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut cache = read_collections_cache();
    cache.insert(collection_url.to_string(), parsed.clone());
    write_collections_cache(&cache);
    Ok(parsed)
}

/// "Manifest" of a STAC theme: its type collections with part counts
/// (feature totals live in the per-part items — too many to fetch here).
pub fn fetch_stac_manifest(base: &str, snapshot: &str, theme: &str) -> Result<Manifest, String> {
    let snap = stac_resolve_snapshot(base, snapshot)?;
    let url = format!("{base}/{snap}{theme}/catalog.json");
    let cat = get_json(&url)?.ok_or_else(|| format!("{url}: not found"))?;
    let types = stac_children(&cat);
    // In parallel: a theme has up to six types and each collection is a
    // separate document, so serially this was six round trips deep.
    let out: Mutex<Vec<Option<(String, u64)>>> = Mutex::new(vec![None; types.len()]);
    let first_err = Mutex::new(None::<String>);
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..FETCH_THREADS.min(types.len().max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(ty) = types.get(i) else { break };
                let curl = format!("{base}/{snap}{theme}/{ty}/collection.json");
                match fetch_stac_collection(&curl) {
                    Ok(c) => out.lock().unwrap()[i] = Some((ty.clone(), c.parts)),
                    Err(e) => {
                        first_err.lock().unwrap().get_or_insert(e);
                        break;
                    }
                }
            });
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(Manifest {
        state_name: Some(theme.to_string()),
        total_features: None,
        themes: out.into_inner().unwrap().into_iter().flatten().collect(),
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
    /// Path of the part below the collection, when it lives under it —
    /// the only place hive `key=value` segments can come from over plain
    /// HTTPS, where there is no listing to derive a prefix from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rel: Option<String>,
}

/// Resolve an href against the document that carries it, over the subset
/// of RFC 3986 these catalogs use: an absolute http(s) href stands on its
/// own, `./x` and `x` hang off the document's directory, `/x` off the
/// origin.
///
/// `..` is refused rather than resolved: a collection describes the data
/// published under it, and an href climbing out of that prefix is either
/// broken or aimed somewhere it has no claim to.
fn join_href(doc_url: &str, href: &str) -> Result<String, String> {
    let href = href.trim();
    if href.is_empty() {
        return Err(format!("{doc_url}: an asset has an empty href"));
    }
    if href.starts_with("http://") || href.starts_with("https://") {
        return Ok(href.to_string());
    }
    if href.contains("://") {
        return Err(format!("{href}: not an http(s) href"));
    }
    if href.split('/').any(|s| s == "..") {
        return Err(format!("{href}: href escapes the collection's prefix"));
    }
    if href.starts_with('/') {
        // Root-relative: everything after the host is replaced.
        let origin = doc_url
            .split_once("://")
            .map(|(scheme, rest)| {
                let host = rest.split('/').next().unwrap_or(rest);
                format!("{scheme}://{host}")
            })
            .ok_or_else(|| format!("{doc_url}: not an http(s) URL"))?;
        return Ok(format!("{origin}{href}"));
    }
    let dir = doc_url.rsplit_once('/').map(|(d, _)| d).unwrap_or(doc_url);
    Ok(format!("{dir}/{}", href.trim_start_matches("./")))
}

/// A resolved part URL's path below the collection, or None when the
/// part lives outside it (another host, another prefix) and a path
/// fragment would describe nothing.
fn href_below(collection_url: &str, url: &str) -> Option<String> {
    let dir = collection_url
        .rsplit_once('/')
        .map(|(d, _)| d)
        .unwrap_or(collection_url);
    url.strip_prefix(&format!("{dir}/"))
        .filter(|rel| !rel.is_empty())
        .map(str::to_string)
}

/// Whether an asset is a parquet part: the media type when it is
/// declared, else the extension — third-party collections often publish
/// `.parquet` hrefs with no type at all.
fn is_parquet_asset(a: &Value) -> bool {
    a.get("type").and_then(Value::as_str) == Some("application/vnd.apache.parquet")
        || a.get("href")
            .and_then(Value::as_str)
            .is_some_and(|h| h.to_ascii_lowercase().ends_with(".parquet"))
}

/// The parquet asset of a STAC item: prefer the `aws` asset, else the
/// first http(s) parquet one. Relative hrefs resolve against the item.
fn stac_item_asset(item: &Value, item_url: &str) -> Option<String> {
    let assets = item.get("assets").and_then(Value::as_object)?;
    // A non-http(s) scheme (Overture publishes `s3://` alongside) is
    // skipped, not an error: another asset of the same item may serve
    // the same part over a protocol this reader speaks.
    let href = |a: &Value| {
        a.get("href")
            .and_then(Value::as_str)
            .and_then(|h| join_href(item_url, h).ok())
    };
    if let Some(a) = assets.get("aws").and_then(href) {
        return Some(a);
    }
    assets
        .values()
        .find_map(|a| if is_parquet_asset(a) { href(a) } else { None })
}

/// The parts a collection lists in its own `assets` map, in key order.
///
/// This is the whole manifest for a dataset published without per-part
/// Items — what this workbench's own publish flow writes, and what makes
/// an https prefix openable at all: HTTP cannot list a directory, so the
/// collection document is the listing.
fn stac_asset_parts(col: &Value, collection_url: &str) -> Result<Vec<StacPart>, String> {
    // A collection-only document says nothing per part, so every part
    // inherits the dataset extent: pruning then keeps or drops them all,
    // which is correct if coarse. Writers that state a per-asset bbox
    // (ours does) get real pruning.
    let extent = col
        .get("extent")
        .and_then(|e| e.get("spatial"))
        .and_then(|s| s.get("bbox"))
        .and_then(Value::as_array)
        .and_then(|b| b.first())
        .and_then(stac_bbox_2d);
    let Some(assets) = col.get("assets").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut entries: Vec<(&String, &Value)> =
        assets.iter().filter(|(_, a)| is_parquet_asset(a)).collect();
    // By key: the JSON object's order is not guaranteed to survive a
    // parse, and part order decides fragment order in the layer.
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    entries
        .into_iter()
        .map(|(key, a)| {
            let href = a.get("href").and_then(Value::as_str).unwrap_or(key);
            let url = join_href(collection_url, href)?;
            Ok(StacPart {
                bbox: a.get("bbox").and_then(stac_bbox_2d).or(extent),
                rows: a.get("rows").and_then(Value::as_u64).unwrap_or(0),
                rel: href_below(collection_url, &url),
                url,
            })
        })
        .collect()
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

/// `~/.config/geopq-workbench/stac_parts_cache.json`: part lists per
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

/// Drop cached part lists and collection descriptions of one repository.
pub fn clear_cached_stac_parts(base: &str) {
    let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut cache = read_parts_cache();
    let n = cache.len();
    cache.retain(|url, _| !url.starts_with(base));
    if cache.len() != n {
        write_parts_cache(&cache);
    }
    let mut cols = read_collections_cache();
    let n = cols.len();
    cols.retain(|url, _| !url.starts_with(base));
    if cols.len() != n {
        write_collections_cache(&cols);
    }
}

/// Whether a collection URL belongs to a configured STAC repository, as
/// opposed to one a user typed or dropped on the app.
fn is_repository_collection(url: &str) -> bool {
    load_repos()
        .iter()
        .any(|r| r.kind == RepoKind::Stac && url.starts_with(&r.url))
}

/// Collection URLs whose part list has been read live in this session.
fn proven_parts() -> &'static Mutex<std::collections::HashSet<String>> {
    static PROVEN: std::sync::OnceLock<Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    PROVEN.get_or_init(Mutex::default)
}

/// Whether the on-disk part list may answer for this collection.
///
/// The cache assumes immutability, which holds for the dated release
/// prefixes a repository publishes but not for a prefix a user publishes
/// to: republishing rewrites the collection in place, and a stale entry
/// would open yesterday's parts with no sign that it did. So a
/// collection outside the configured repositories is proven live once
/// per session and cached from then on — panning re-reads the part list
/// on every pass and must not pay for a round trip each time.
fn parts_cache_may_answer(collection_url: &str) -> bool {
    is_repository_collection(collection_url)
        || proven_parts()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(collection_url)
}

/// All parts of a type collection, served from the on-disk cache when
/// present (release prefixes are immutable), else from its item documents
/// or its own assets (parallel fetch on dedicated threads; any failure
/// aborts — a silently partial part list would silently drop data).
pub fn fetch_stac_parts(collection_url: &str) -> Result<Vec<StacPart>, String> {
    if parts_cache_may_answer(collection_url) {
        let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(parts) = read_parts_cache().remove(collection_url) {
            return Ok(parts);
        }
    }
    // Fetch outside the lock: item documents can take a while, and the
    // lock only has to make the read-modify-write below atomic.
    let parts = fetch_stac_parts_live(collection_url)?;
    // Proven only now: a failed fetch must leave the next attempt going
    // live too, not fall back on the entry it just refused to trust.
    proven_parts()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(collection_url.to_string());
    let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut cache = read_parts_cache();
    cache.insert(collection_url.to_string(), parts.clone());
    write_parts_cache(&cache);
    Ok(parts)
}

/// A collection document that is not there. For a URL the user pointed
/// at, "not found" alone teaches nothing: no part of an https prefix
/// says that it is opened through a manifest, or where that manifest is
/// supposed to come from.
fn no_collection(collection_url: &str) -> String {
    if is_repository_collection(collection_url) {
        return format!("{collection_url}: not found");
    }
    format!(
        "{collection_url}: not found. Plain HTTPS cannot list a directory, so a \
         prefix opens through a STAC collection.json sitting at it — the one this \
         workbench's Publish to S3 / R2 writes beside every dataset it uploads."
    )
}

fn fetch_stac_parts_live(collection_url: &str) -> Result<Vec<StacPart>, String> {
    let col = get_json(collection_url)?.ok_or_else(|| no_collection(collection_url))?;
    // The document is in hand and carries the licence and part count, so
    // record them: a layer opened from a pasted URL never went through the
    // browser, and its credit would otherwise cost a second fetch.
    {
        let parsed = parse_stac_collection(&col);
        let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cache = read_collections_cache();
        cache.insert(collection_url.to_string(), parsed);
        write_collections_cache(&cache);
    }
    let item_urls: Vec<String> = col
        .get("links")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|l| l.get("rel").and_then(Value::as_str) == Some("item"))
        .filter_map(|l| l.get("href").and_then(Value::as_str))
        .map(|h| join_href(collection_url, h))
        .collect::<Result<_, _>>()?;
    if item_urls.is_empty() {
        // No Items: the parts are the collection's own assets.
        let parts = stac_asset_parts(&col, collection_url)?;
        if parts.is_empty() {
            return Err(format!(
                "{collection_url}: this STAC collection lists no parquet parts \
                 (neither item links nor application/vnd.apache.parquet assets)"
            ));
        }
        return Ok(parts);
    }

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
                let Some(asset) = stac_item_asset(&item, url) else {
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
                    // An Item's asset URL is its own address, not a path
                    // under the collection: no hive segments to read.
                    rel: None,
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
            // No providers, like Overture's: the credit has to come from
            // the repository entry.
            json!({"license": "ODbL-1.0", "links": [
                {"rel": "item", "href": "./00000/00000.json"},
                {"rel": "item", "href": "./00001/00001.json"},
            ]}),
        );
        write_json(
            &root,
            "2026-06/buildings/building_part/collection.json",
            json!({
                "license": "CC-BY-4.0",
                "providers": [
                    {"name": "A Processor", "roles": ["processor"]},
                    {"name": "The Publisher", "roles": ["producer", "licensor"]},
                ],
                "links": [{"rel": "item", "href": "./00000/00000.json"}],
            }),
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
    /// Overture's ODbL themes carry OpenStreetMap and must name its
    /// contributors; the CC0 and CC BY ones next to them must not.
    fn odbl_collections_get_the_openstreetmap_credit() {
        let overture = default_repos()
            .into_iter()
            .find(|r| r.kind == RepoKind::Stac)
            .expect("Overture entry");
        assert_eq!(
            overture.credit_for(Some("ODbL-1.0")),
            Some("© OpenStreetMap contributors, Overture Maps Foundation"),
        );
        // SPDX identifiers are compared case-insensitively: catalogs write
        // "ODbL-1.0", the identifier is registered as "ODbL-1.0", and a
        // catalog writing "odbl-1.0" means the same licence.
        assert_eq!(
            overture.credit_for(Some("odbl-1.0")),
            overture.credit_for(Some("ODbL-1.0")),
        );
        for other in ["CC0-1.0", "CC-BY-4.0", "other"] {
            assert_eq!(
                overture.credit_for(Some(other)),
                Some("Overture Maps Foundation, overturemaps.org"),
                "licence {other}",
            );
        }
        assert_eq!(overture.credit_for(None), overture.attribution.as_deref());
    }

    #[test]
    /// The geoBoundaries and CLC repositories sit under the OSM
    /// repository's base URL. Matching the shorter base would credit their
    /// polygons to OpenStreetMap, which is worse than crediting nobody.
    fn a_nested_repository_does_not_inherit_its_parents_credit() {
        // Defaults are in play only when the user has no config of their
        // own; these tests share a per-process temp config file.
        let _ = std::fs::remove_file(config_file().unwrap());
        assert_eq!(
            credit_for_url("https://parquetry.geomermaids.com/latest/x.parquet", None),
            Some("© OpenStreetMap contributors".to_string()),
        );
        assert_eq!(
            credit_for_url(
                "https://parquetry.geomermaids.com/geoboundaries/6.0.0/a.parquet",
                None,
            ),
            None,
        );
        assert_eq!(
            credit_for_url("https://elsewhere.example/a.parquet", None),
            None,
        );
    }

    #[test]
    /// The browser used to refetch every type's collection.json on every
    /// click, just to count its item links — 123 KB of them for Overture's
    /// buildings. Once read, a collection is cached like its part list is.
    fn stac_manifest_second_call_needs_no_network() {
        let (base, root) = spawn_stac_fixture();
        let first = fetch_stac_manifest(&base, "2026-06/", "buildings").unwrap();
        // Delete the documents the counts came from: a second call can only
        // answer from the on-disk cache.
        std::fs::remove_file(root.join("2026-06/buildings/building/collection.json")).unwrap();
        std::fs::remove_file(root.join("2026-06/buildings/building_part/collection.json"))
            .unwrap();
        let cached = fetch_stac_manifest(&base, "2026-06/", "buildings").unwrap();
        assert_eq!(cached.themes, first.themes);
        // ⟳ clears collections alongside parts; the next call goes live.
        clear_cached_stac_parts(&base);
        assert!(
            fetch_stac_manifest(&base, "2026-06/", "buildings").is_err(),
            "cache cleared, collection documents gone",
        );
    }

    #[test]
    /// Opening a layer records the collection's licence on the way past, so
    /// a layer opened from a pasted URL is credited without a second fetch.
    fn opening_parts_records_the_collection() {
        let (base, root) = spawn_stac_fixture();
        let url = stac_collection_url(&base, "2026-06/", "buildings", "building");
        fetch_stac_parts(&url).unwrap();
        std::fs::remove_file(root.join("2026-06/buildings/building/collection.json")).unwrap();
        let col = fetch_stac_collection(&url).unwrap();
        assert_eq!(col.license.as_deref(), Some("ODbL-1.0"));
        assert_eq!(col.parts, 2);
        assert!(col.providers.is_empty());
    }

    #[test]
    /// Rights holders are credited; a processor that only reformatted the
    /// data is not who the licence asks for.
    fn collection_providers_keep_rights_holders_only() {
        let (base, _root) = spawn_stac_fixture();
        let url = stac_collection_url(&base, "2026-06/", "buildings", "building_part");
        let col = fetch_stac_collection(&url).unwrap();
        assert_eq!(col.providers, vec!["The Publisher".to_string()]);
        assert_eq!(col.license.as_deref(), Some("CC-BY-4.0"));
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
    /// Hrefs are resolved against the document that carries them, and a
    /// href climbing out of the collection's prefix is refused by name —
    /// silently dropping it would open a short dataset that looks whole.
    fn hrefs_resolve_against_the_document_and_refuse_escapes() {
        let col = "https://host.example/data/roads/collection.json";
        // Absolute passes through untouched, whatever host it names.
        assert_eq!(
            join_href(col, "https://other.example/p.parquet").unwrap(),
            "https://other.example/p.parquet"
        );
        // `./x`, bare `x` and a nested path all hang off the directory.
        for href in ["./part-0.parquet", "part-0.parquet"] {
            assert_eq!(
                join_href(col, href).unwrap(),
                "https://host.example/data/roads/part-0.parquet",
                "{href}"
            );
        }
        assert_eq!(
            join_href(col, "./dep=31/part-0.parquet").unwrap(),
            "https://host.example/data/roads/dep=31/part-0.parquet"
        );
        // Root-relative replaces the whole path, not the last segment.
        assert_eq!(
            join_href(col, "/other/p.parquet").unwrap(),
            "https://host.example/other/p.parquet"
        );
        for bad in ["../secrets/p.parquet", "./a/../../p.parquet"] {
            let e = join_href(col, bad).unwrap_err();
            assert!(e.contains(bad) && e.contains("escapes"), "{e}");
        }
        // A scheme the range reader cannot speak is refused, not joined.
        assert!(join_href(col, "s3://bucket/p.parquet").is_err());
        assert!(join_href(col, "  ").is_err());
    }

    #[test]
    /// A collection that lists its parts as assets (no Items) is the
    /// whole manifest of a dataset published over plain HTTPS.
    fn collection_assets_are_parts_in_key_order() {
        use serde_json::json;
        let url = "https://host.example/data/roads/collection.json";
        let col = json!({
            "type": "Collection",
            "extent": {"spatial": {"bbox": [[-10.0, 40.0, 10.0, 50.0]]}},
            "links": [],
            "assets": {
                "dep=75/part-0.parquet": {
                    "href": "./dep=75/part-0.parquet",
                    "type": "application/vnd.apache.parquet",
                    "bbox": [1.0, 48.0, 3.0, 49.0],
                    "rows": 7,
                },
                // Media type absent: third-party collections often skip
                // it, and the extension is then all there is to go on.
                "dep=31/part-0.parquet": {"href": "./dep=31/part-0.parquet"},
                "thumbnail": {"href": "./preview.png", "type": "image/png"},
                "metadata": {"href": "./stats.json", "type": "application/json"},
            },
        });
        let parts = stac_asset_parts(&col, url).unwrap();
        // Sorted by asset key, so the layer's fragment order does not
        // depend on how the JSON object happened to be parsed.
        assert_eq!(parts.len(), 2, "only the parquet assets: {parts:?}");
        assert_eq!(
            parts.iter().map(|p| p.url.as_str()).collect::<Vec<_>>(),
            vec![
                "https://host.example/data/roads/dep=31/part-0.parquet",
                "https://host.example/data/roads/dep=75/part-0.parquet",
            ]
        );
        // Hive segments come from the path below the collection.
        assert_eq!(parts[0].rel.as_deref(), Some("dep=31/part-0.parquet"));
        // No stated bbox: the collection extent, which prunes nothing —
        // a per-asset bbox is what makes viewport pruning real.
        assert_eq!(parts[0].bbox, Some([-10.0, 40.0, 10.0, 50.0]));
        assert_eq!(parts[0].rows, 0);
        assert_eq!(parts[1].bbox, Some([1.0, 48.0, 3.0, 49.0]));
        assert_eq!(parts[1].rows, 7);

        // A part hosted elsewhere has no path under the collection, so
        // no hive segments to invent.
        let col = json!({"assets": {"a": {"href": "https://cdn.example/p.parquet"}}});
        let parts = stac_asset_parts(&col, url).unwrap();
        assert_eq!(parts[0].url, "https://cdn.example/p.parquet");
        assert_eq!(parts[0].rel, None);
        assert_eq!(parts[0].bbox, None, "no extent to fall back on");

        // Nothing parquet-shaped at all: an empty list, for the caller to
        // report as such.
        let col = json!({"assets": {"thumbnail": {"href": "./preview.png"}}});
        assert!(stac_asset_parts(&col, url).unwrap().is_empty());
    }

    #[test]
    /// Over plain HTTPS the collection document is the listing, so both
    /// ways of missing it have to say what was expected and where.
    fn a_prefix_without_a_collection_teaches_the_convention() {
        let base = spawn_files(&[("collection.json", r#"{"type":"Collection","assets":{}}"#)]);
        // Present but empty: not a network problem, a content one.
        let err = fetch_stac_parts(&format!("{base}/collection.json")).unwrap_err();
        assert!(err.contains("no parquet parts"), "{err}");

        let base = spawn_files(&[("other.json", "{}")]);
        let err = fetch_stac_parts(&format!("{base}/collection.json")).unwrap_err();
        assert!(err.contains("collection.json"), "{err}");
        assert!(err.contains("cannot list a directory"), "{err}");
        assert!(err.contains("Publish"), "the app writes one itself: {err}");
        // A configured repository's own 404 stays terse: its browser
        // never asked the user to know about manifests.
        let repo_url = "https://stac.overturemaps.org/2026-06/x/collection.json";
        assert!(is_repository_collection(repo_url));
        assert_eq!(no_collection(repo_url), format!("{repo_url}: not found"));
    }

    #[test]
    /// A dated release prefix is immutable; a prefix a user publishes to
    /// is not, and a cached part list there would silently open the
    /// dataset as it was before the last publish.
    fn a_user_collection_is_proven_live_once_per_session() {
        let base = spawn_files(&[(
            "collection.json",
            r#"{"assets": {"a.parquet": {"href": "./a.parquet"}}}"#,
        )]);
        let url = format!("{base}/collection.json");
        // Seed the disk cache with a part list that is not what the
        // server says: only a live fetch can disagree with it.
        {
            let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut cache = read_parts_cache();
            cache.insert(
                url.clone(),
                vec![StacPart {
                    url: "https://stale.example/old.parquet".into(),
                    bbox: None,
                    rows: 0,
                    rel: None,
                }],
            );
            write_parts_cache(&cache);
        }
        let parts = fetch_stac_parts(&url).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].url, format!("{base}/a.parquet"), "stale entry ignored");

        // Proven once: panning re-reads the list on every pass and must
        // not pay a round trip each time, so the cache answers from here.
        {
            let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut cache = read_parts_cache();
            cache.insert(
                url.clone(),
                vec![
                    StacPart {
                        url: "https://cached.example/0.parquet".into(),
                        bbox: None,
                        rows: 0,
                        rel: None,
                    };
                    2
                ],
            );
            write_parts_cache(&cache);
        }
        let parts = fetch_stac_parts(&url).unwrap();
        assert_eq!(parts.len(), 2, "second call comes from the cache");

        // A repository collection keeps the immutability assumption from
        // its very first fetch, network or not.
        let repo_url = "https://stac.overturemaps.org/2026-06/b/building/collection.json";
        {
            let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut cache = read_parts_cache();
            cache.insert(
                repo_url.to_string(),
                vec![StacPart {
                    url: "https://aws.example/p.parquet".into(),
                    bbox: None,
                    rows: 9,
                    rel: None,
                }],
            );
            write_parts_cache(&cache);
        }
        let parts = fetch_stac_parts(repo_url).expect("answered without a network call");
        assert_eq!(parts[0].rows, 9);

        // A collection that could not be read proves nothing: the next
        // attempt goes live again rather than settling for the entry it
        // has just declined to trust.
        let gone = format!("{}/collection.json", spawn_files(&[("other.json", "{}")]));
        {
            let _g = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut cache = read_parts_cache();
            cache.insert(
                gone.clone(),
                vec![StacPart {
                    url: "https://stale.example/old.parquet".into(),
                    bbox: None,
                    rows: 0,
                    rel: None,
                }],
            );
            write_parts_cache(&cache);
        }
        assert!(fetch_stac_parts(&gone).is_err());
        assert!(fetch_stac_parts(&gone).is_err(), "still no collection to read");
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
        // A repository holding one global dataset publishes it directly
        // under the snapshot: the empty dataset path must not leave a
        // double slash, which object storage reads as another key.
        assert_eq!(
            theme_url(
                "https://parquetry.geomermaids.com/geoboundaries",
                "6.0.0/",
                "",
                "cgaz_adm0"
            ),
            "https://parquetry.geomermaids.com/geoboundaries/6.0.0/cgaz_adm0.parquet"
        );
        assert_eq!(
            dataset_prefix("6.0.0/", ""),
            "6.0.0/",
            "manifest and theme URLs share the prefix"
        );
    }

    /// A directory of literal JSON files, served over loopback.
    fn spawn_files(files: &[(&str, &str)]) -> String {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "geopq_repo_fix_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst),
        ));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, body) in files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        crate::data::source::testserver::spawn_dir(root)
    }

    #[test]
    fn snapshots_can_alias_latest_onto_a_release() {
        let base = spawn_files(&[(
            "snapshots.json",
            r#"{"latest":"6.0.0/","snapshots":[{"date":"6.0.0","path":"6.0.0/"}]}"#,
        )]);
        let snaps = fetch_snapshots(&base).unwrap();
        // "latest" resolves to the real release rather than a 404 prefix,
        // and it is not duplicated by the release entry.
        assert_eq!(snaps[0].label, "latest");
        assert_eq!(snaps[0].path, "6.0.0/");
        assert_eq!(snaps[1].label, "6.0.0");
        assert_eq!(snaps.len(), 2);

        // Without the alias, the rolling default still applies.
        let base2 = spawn_files(&[(
            "snapshots.json",
            r#"{"snapshots":[{"date":"2026-07-15","path":"2026-07-15/"}]}"#,
        )]);
        let snaps = fetch_snapshots(&base2).unwrap();
        assert_eq!(snaps[0].path, "latest/");
    }

    #[test]
    fn index_json_may_name_a_single_global_dataset() {
        let base = spawn_files(&[(
            "index.json",
            r#"{"datasets":[{"path":"","code":"","name":"geoBoundaries CGAZ (global)"}]}"#,
        )]);
        let ds = discover_datasets(&base, "6.0.0/").unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds[0].path, "");
        assert_eq!(ds[0].code, "");
        assert_eq!(ds[0].name, "geoBoundaries CGAZ (global)");
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

    /// Live probe of every single-dataset repository, opt-in:
    /// cargo test --release repo_live_geoboundaries -- --ignored --nocapture
    #[test]
    #[ignore]
    fn repo_live_clc() {
        let base = "https://parquetry.geomermaids.com/clc";
        let snaps = fetch_snapshots(base).unwrap();
        assert_eq!(snaps[0].path, "2018/", "latest aliases the release");
        let ds = discover_datasets(base, "2018/").unwrap();
        assert_eq!(ds.len(), 1);
        let m = fetch_manifest(base, "2018/", &ds[0].path).unwrap();
        eprintln!("clc: {:?} {:?}", m.state_name, m.themes);
        assert_eq!(m.themes.len(), 1);
        for (theme, rows) in &m.themes {
            let url = theme_url(base, "2018/", &ds[0].path, theme);
            eprintln!("{theme}: {rows} rows -> {url}");
            assert!(exists(&url).unwrap(), "{url}");
        }
    }

    #[test]
    #[ignore]
    fn repo_live_geoboundaries() {
        let base = "https://parquetry.geomermaids.com/geoboundaries";
        let snaps = fetch_snapshots(base).unwrap();
        eprintln!("snapshots: {:?}", snaps.iter().map(|s| (&s.label, &s.path)).collect::<Vec<_>>());
        // "latest" is aliased onto the release, not a 404 prefix.
        assert_eq!(snaps[0].label, "latest");
        assert_eq!(snaps[0].path, "6.0.0/");

        let ds = discover_datasets(base, "6.0.0/").unwrap();
        eprintln!("datasets: {:?}", ds.iter().map(|d| (&d.name, &d.path)).collect::<Vec<_>>());
        assert_eq!(ds.len(), 1);
        assert!(ds[0].path.is_empty());

        let m = fetch_manifest(base, "6.0.0/", &ds[0].path).unwrap();
        eprintln!("manifest: {:?} {:?}", m.state_name, m.themes);
        assert_eq!(m.themes.len(), 3);

        // Every theme URL must actually resolve.
        for (theme, rows) in &m.themes {
            let url = theme_url(base, "6.0.0/", &ds[0].path, theme);
            let ok = exists(&url).unwrap();
            eprintln!("{theme}: {rows} rows -> {url} [{}]", if ok { "OK" } else { "MISSING" });
            assert!(ok, "{url}");
        }
    }
}
