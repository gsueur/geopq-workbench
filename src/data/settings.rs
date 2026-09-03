//! The settings sidecar both binaries read: one small JSON file in the
//! home directory. The GUI writes it (basemap key, declined-file memory,
//! SQL history); `geopq-cli` only reads the COGP block out of it, so the
//! two agree on what `--cogp` means without the CLI linking any UI code.
//!
//! Every writer goes through [`update`]: read-modify-write straight onto
//! the file lost whichever key the other writer was holding (the SQL
//! console saving history over the API key the settings dialog had just
//! written), and a plain `fs::write` that is interrupted leaves a
//! truncated file that every reader then treats as "no settings at all"
//! — silently dropping the key, the history and the COGP block together.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Settings sidecar for the quality gate's decline memory: one small
/// JSON file in the home directory (the app has no other persistence).
/// Unknown keys are preserved for forward compatibility.
pub fn settings_path() -> Option<PathBuf> {
    // Tests redirect the sidecar rather than $HOME: the whole suite runs
    // in one process, and moving $HOME out from under a parallel test
    // that is resolving a cache directory is its own bug.
    #[cfg(test)]
    if let Some(p) = tests::test_path() {
        return Some(p);
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".geopq-workbench.json"))
}

/// What reading the settings file found.
///
/// "Absent" and "unreadable" have to stay apart. A first run has no file
/// and must fall back to the defaults without a word; a file that exists
/// but does not parse means somebody's settings are there and we cannot
/// see them, which is worth a warning — and worth refusing a run whose
/// whole point was to honour them.
pub enum Settings {
    /// No settings file (or no home directory): defaults apply.
    Absent,
    /// Parsed. Always a JSON object; a top-level non-object is reported
    /// as `Unreadable` instead.
    Loaded(Value),
    /// The file is there but unusable — truncated by an interrupted
    /// write, hand-edited into invalid JSON, unreadable permissions.
    Unreadable(String),
}

impl Settings {
    /// The parsed object, or None when absent or unreadable. For readers
    /// that genuinely have nothing to do without a value.
    pub fn value(&self) -> Option<&Value> {
        match self {
            Settings::Loaded(v) => Some(v),
            _ => None,
        }
    }

    /// One top-level key, when the file is readable and has it.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.value()?.get(key)
    }
}

/// Read the settings file.
pub fn read() -> Settings {
    let Some(p) = settings_path() else {
        return Settings::Absent;
    };
    match std::fs::read_to_string(&p) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Settings::Absent,
        Err(e) => Settings::Unreadable(format!("cannot read {}: {e}", p.display())),
        Ok(txt) if txt.trim().is_empty() => Settings::Absent,
        Ok(txt) => match serde_json::from_str::<Value>(&txt) {
            Ok(v) if v.is_object() => Settings::Loaded(v),
            Ok(_) => Settings::Unreadable(format!("{}: not a JSON object", p.display())),
            Err(e) => Settings::Unreadable(format!("{}: invalid JSON: {e}", p.display())),
        },
    }
}

/// Read-modify-write one settings key under an advisory lock, atomically.
///
/// `f` receives the whole document (always an object) and mutates the
/// keys it owns; everything else is carried through untouched. The write
/// is temp file + `sync_all` + rename, so a reader never sees a half
/// file, and the lock keeps two writers from each basing their document
/// on the same pre-image and clobbering the other's key.
///
/// Refuses rather than overwrites when the existing file does not parse:
/// rewriting it would throw away whatever the user actually had in
/// there, which is the opposite of what a settings save should risk.
pub fn update(f: impl FnOnce(&mut Value)) -> Result<(), String> {
    let path = settings_path().ok_or("no home directory to save settings in")?;
    let _guard = FileLock::acquire(&path)?;
    let mut root = match read() {
        Settings::Loaded(v) => v,
        Settings::Absent => Value::Object(Default::default()),
        Settings::Unreadable(e) => {
            return Err(format!("{e} — not overwriting it"));
        }
    };
    f(&mut root);
    let txt = serde_json::to_string_pretty(&root).map_err(|e| format!("settings: {e}"))?;
    write_atomic(&path, &txt)
}

/// Temp file in the same directory, flushed to disk, then renamed over
/// the target — the rename is the only step a reader can observe.
fn write_atomic(path: &Path, data: &str) -> Result<(), String> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let write = || -> std::io::Result<()> {
        let mut fh = std::fs::File::create(&tmp)?;
        fh.write_all(data.as_bytes())?;
        fh.sync_all()?;
        drop(fh);
        std::fs::rename(&tmp, path)
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot write {}: {e}", path.display())
    })
}

/// Advisory lock as a `.lock` sibling created with `create_new`, which
/// is atomic on every filesystem we care about. No `flock` crate is in
/// the tree and this file is written a handful of times per session, so
/// a lock file with a short retry is the right size of solution.
struct FileLock(PathBuf);

impl FileLock {
    /// How long a lock may sit before it is assumed to belong to a
    /// process that died holding it. Long enough that a live writer is
    /// never stolen from; short enough that a crash does not wedge
    /// settings for the rest of the session.
    const STALE: std::time::Duration = std::time::Duration::from_secs(30);

    fn acquire(target: &Path) -> Result<FileLock, String> {
        let lock = target.with_extension("lock");
        for attempt in 0..50 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock)
            {
                Ok(_) => return Ok(FileLock(lock)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Break a lock left behind by a crashed writer.
                    let stale = std::fs::metadata(&lock)
                        .and_then(|m| m.modified())
                        .map(|t| t.elapsed().unwrap_or_default() > Self::STALE)
                        .unwrap_or(false);
                    if stale {
                        let _ = std::fs::remove_file(&lock);
                        continue;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2 + attempt));
                }
                // No directory, read-only home: not something retrying fixes.
                Err(e) => return Err(format!("cannot lock {}: {e}", lock.display())),
            }
        }
        Err(format!("{} stayed locked", lock.display()))
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The COGP knobs the Export dialog no longer shows: the reference
/// converter's defaults unless the settings file carries a `cogp` object
/// overriding them. Read once — these are a power user's levers, not a
/// live setting, and an object that does not validate falls back whole
/// rather than half-applying.
///
/// `Err` means the settings file itself could not be read, so we cannot
/// say whether it carried a `cogp` block: [`cogp_defaults`] treats that
/// as "use the defaults" for the GUI, while `geopq-cli --cogp` refuses,
/// because substituting reference GSDs for the ones the user configured
/// writes a pyramid that is wrong in a way nothing downstream can spot.
pub fn cogp_settings() -> Result<&'static super::optimize::CogpOptions, &'static str> {
    use super::optimize::{CogpOptions, GsdSource, RankOrder};
    static PARSED: std::sync::OnceLock<Result<CogpOptions, String>> = std::sync::OnceLock::new();
    PARSED
        .get_or_init(|| {
            let settings = read();
            if let Settings::Unreadable(e) = &settings {
                return Err(e.clone());
            }
            let Some(c) = settings.get("cogp") else {
                return Ok(CogpOptions::default());
            };
            let mut o = CogpOptions::default();
            let u32_of = |k: &str| c.get(k).and_then(Value::as_u64).map(|n| n as u32);
            if let GsdSource::WebMercator {
                minzoom,
                maxzoom,
                resolution,
            } = &mut o.gsd
            {
                *minzoom = u32_of("minzoom").unwrap_or(*minzoom);
                *maxzoom = u32_of("maxzoom").unwrap_or(*maxzoom);
                *resolution = u32_of("resolution").unwrap_or(*resolution);
            }
            // An explicit list replaces the zoom pyramid outright, for
            // renderers that are not a Web Mercator one.
            if let Some(list) = c.get("gsds").and_then(Value::as_array) {
                o.gsd = GsdSource::Explicit(list.iter().filter_map(Value::as_f64).collect());
            }
            o.line_factor = u32_of("line_factor").unwrap_or(o.line_factor);
            o.polygon_factor = u32_of("polygon_factor").unwrap_or(o.polygon_factor);
            o.point_factor = u32_of("point_factor").unwrap_or(o.point_factor);
            o.rank = c
                .get("rank")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(|n| {
                    let asc = c.get("rank_order").and_then(Value::as_str) == Some("asc");
                    let order = if asc { RankOrder::Asc } else { RankOrder::Desc };
                    (n.to_string(), order)
                });
            match o.gsds() {
                Ok(_) => Ok(o),
                Err(e) => Err(format!("settings `cogp`: {e}")),
            }
        })
        .as_ref()
        .map_err(|e| e.as_str())
}

/// [`cogp_settings`], with an unreadable settings file logged and
/// downgraded to the defaults — for the GUI, where the Export dialog
/// still has to open.
pub fn cogp_defaults() -> &'static super::optimize::CogpOptions {
    static FALLBACK: std::sync::OnceLock<super::optimize::CogpOptions> =
        std::sync::OnceLock::new();
    match cogp_settings() {
        Ok(o) => o,
        Err(e) => {
            log::warn!("{e} — using the defaults");
            FALLBACK.get_or_init(Default::default)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where `settings_path` points while a test holds `TempHome`.
    static TEST_PATH: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);
    /// One redirect at a time: the tests below each want their own file.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(super) fn test_path() -> Option<PathBuf> {
        TEST_PATH.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    struct TempHome {
        dir: PathBuf,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl TempHome {
        fn new(tag: &str) -> TempHome {
            let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir()
                .join(format!("geopq_settings_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            *TEST_PATH.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(dir.join(".geopq-workbench.json"));
            TempHome { dir, _guard: guard }
        }
        fn file(&self) -> PathBuf {
            self.dir.join(".geopq-workbench.json")
        }
    }

    impl Drop for TempHome {
        fn drop(&mut self) {
            *TEST_PATH.lock().unwrap_or_else(|e| e.into_inner()) = None;
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Every writer owns one key. Read-modify-write through `update`
    /// has to carry the others through — including keys this build has
    /// never heard of, which is how a settings file survives a
    /// downgrade.
    #[test]
    fn update_preserves_the_keys_it_does_not_own() {
        let home = TempHome::new("keys");
        std::fs::write(
            home.file(),
            r#"{"carto_api_key":"abc","from_a_newer_build":{"x":1}}"#,
        )
        .unwrap();

        update(|root| root["sql_history"] = serde_json::json!(["select 1"])).unwrap();
        update(|root| root["direct_files"] = serde_json::json!(["/tmp/a.parquet"])).unwrap();

        let v = read();
        assert_eq!(v.get("carto_api_key").unwrap(), "abc");
        assert_eq!(v.get("from_a_newer_build").unwrap()["x"], 1);
        assert_eq!(v.get("sql_history").unwrap()[0], "select 1");
        assert_eq!(v.get("direct_files").unwrap()[0], "/tmp/a.parquet");
        // The lock is not left behind for the next writer to wait on.
        assert!(!home.dir.join(".geopq-workbench.lock").exists());
    }

    /// A truncated file is somebody's settings, damaged — not an empty
    /// one. Readers used to swallow the parse error and hand back
    /// defaults, so the API key vanished without a word and the next
    /// save wrote a fresh document over whatever was left.
    #[test]
    fn a_truncated_file_is_reported_rather_than_read_as_empty() {
        let home = TempHome::new("trunc");
        // What an interrupted `fs::write` of a pretty-printed document
        // leaves behind.
        std::fs::write(home.file(), "{\n  \"carto_api_key\": \"ab").unwrap();

        match read() {
            Settings::Unreadable(e) => assert!(e.contains("invalid JSON"), "{e}"),
            Settings::Absent => panic!("a damaged file must not read as absent"),
            Settings::Loaded(_) => panic!("that is not valid JSON"),
        }
        let err = update(|root| root["sql_history"] = serde_json::json!([])).unwrap_err();
        assert!(err.contains("not overwriting"), "{err}");
        // And the damaged bytes are still there for the user to salvage.
        assert_eq!(
            std::fs::read_to_string(home.file()).unwrap(),
            "{\n  \"carto_api_key\": \"ab"
        );
    }

    /// No file at all is the first run, and has to stay silent.
    #[test]
    fn an_absent_file_is_not_an_error() {
        let home = TempHome::new("absent");
        assert!(matches!(read(), Settings::Absent));
        update(|root| root["carto_api_key"] = serde_json::json!("k")).unwrap();
        assert_eq!(read().get("carto_api_key").unwrap(), "k");
        assert!(home.file().exists());
    }

    /// Concurrent writers each keep their own key: the lock serializes
    /// the read-modify-write, and the rename means no reader in between
    /// ever sees a partial document.
    #[test]
    fn concurrent_updates_do_not_lose_each_other() {
        let _home = TempHome::new("concurrent");
        update(|root| root["seed"] = serde_json::json!(1)).unwrap();
        std::thread::scope(|s| {
            for i in 0..8 {
                s.spawn(move || {
                    update(|root| root[format!("k{i}")] = serde_json::json!(i)).unwrap();
                });
            }
        });
        let v = read();
        assert_eq!(v.get("seed").unwrap(), 1);
        for i in 0..8 {
            assert_eq!(v.get(&format!("k{i}")).unwrap(), i, "writer {i} was lost");
        }
    }
}
