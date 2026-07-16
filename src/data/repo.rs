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

const USER_AGENT: &str = concat!("geopq-viewer/", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Serialize, Deserialize)]
pub struct Repository {
    pub name: String,
    /// Base URL, no trailing slash.
    pub url: String,
}

pub fn default_repos() -> Vec<Repository> {
    vec![Repository {
        name: "Geomermaids Parquetry (OSM North America)".into(),
        url: "https://parquetry.geomermaids.com".into(),
    }]
}

/// `~/.config/geopq-viewer/repositories.json`.
fn config_file() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| {
            PathBuf::from(h)
                .join(".config")
                .join("geopq-viewer")
                .join("repositories.json")
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
    let e = read_cache().remove(&cache_key(base, snapshot))?;
    Some((e.datasets, e.fetched_at))
}

pub fn store_datasets(base: &str, snapshot: &str, datasets: &[Dataset]) {
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
