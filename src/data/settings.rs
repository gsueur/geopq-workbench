//! The settings sidecar both binaries read: one small JSON file in the
//! home directory. The GUI writes it (basemap key, declined-file memory,
//! SQL history); `geopq-cli` only reads the COGP block out of it, so the
//! two agree on what `--cogp` means without the CLI linking any UI code.

use std::path::PathBuf;

/// Settings sidecar for the quality gate's decline memory: one small
/// JSON file in the home directory (the app has no other persistence).
/// Unknown keys are preserved for forward compatibility.
pub fn settings_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".geopq-workbench.json"))
}

/// The COGP knobs the Export dialog no longer shows: the reference
/// converter's defaults unless the settings file carries a `cogp` object
/// overriding them. Read once — these are a power user's levers, not a
/// live setting, and an object that does not validate falls back whole
/// rather than half-applying.
pub fn cogp_defaults() -> &'static super::optimize::CogpOptions {
    use super::optimize::{CogpOptions, GsdSource, RankOrder};
    static DEFAULTS: std::sync::OnceLock<CogpOptions> = std::sync::OnceLock::new();
    DEFAULTS.get_or_init(|| {
        let read = || -> Option<CogpOptions> {
            let txt = std::fs::read_to_string(settings_path()?).ok()?;
            let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
            let c = v.get("cogp")?;
            let mut o = CogpOptions::default();
            let u32_of = |k: &str| c.get(k).and_then(serde_json::Value::as_u64).map(|n| n as u32);
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
            if let Some(list) = c.get("gsds").and_then(serde_json::Value::as_array) {
                o.gsd = GsdSource::Explicit(
                    list.iter().filter_map(serde_json::Value::as_f64).collect(),
                );
            }
            o.line_factor = u32_of("line_factor").unwrap_or(o.line_factor);
            o.polygon_factor = u32_of("polygon_factor").unwrap_or(o.polygon_factor);
            o.point_factor = u32_of("point_factor").unwrap_or(o.point_factor);
            o.rank = c
                .get("rank")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(|n| {
                    let asc =
                        c.get("rank_order").and_then(serde_json::Value::as_str) == Some("asc");
                    let order = if asc { RankOrder::Asc } else { RankOrder::Desc };
                    (n.to_string(), order)
                });
            Some(o)
        };
        match read() {
            None => CogpOptions::default(),
            Some(o) => match o.gsds() {
                Ok(_) => o,
                Err(e) => {
                    log::warn!("settings `cogp`: {e} — using the defaults");
                    CogpOptions::default()
                }
            },
        }
    })
}
