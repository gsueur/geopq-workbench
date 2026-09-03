//! Session context: layer sources + styles + order, camera, projection and
//! overlay toggles, saved to a JSON file and restored later. Remote layers
//! store the URL / s3 URI (never presigned URLs); credentials are
//! re-resolved at load.

use eframe::egui::Color32;
use serde::{Deserialize, Serialize};

use crate::data::crs::{DisplayCrs, DisplayKind};
use crate::data::layer::LayerStyle;
use crate::data::source::Source;

pub const CONTEXT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct Context {
    pub version: u32,
    pub camera_center: [f64; 2],
    pub camera_zoom: f64,
    /// "hobo-dyer" | "winkel-tripel" | "mercator" | "epsg:NNNN" |
    /// "proj4:<string>" (auto-fit / custom projections).
    pub projection: String,
    /// Display name for proj4-token projections.
    #[serde(default)]
    pub projection_name: Option<String>,
    pub basemap: Option<usize>,
    /// Basemap opacity. Absent in contexts written before it existed,
    /// where the basemap was always fully opaque.
    #[serde(default)]
    pub basemap_opacity: Option<f32>,
    pub show_graticule: bool,
    pub show_coastline: bool,
    /// Pixel width below which a feature is drawn from its bounding box.
    /// Absent in contexts written before it was configurable.
    #[serde(default)]
    pub box_threshold_px: Option<f64>,
    /// Geometry cap for one refinement pass, in MB.
    #[serde(default)]
    pub refine_budget_mb: Option<u32>,
    /// Bottom-to-top draw order.
    pub layers: Vec<LayerCtx>,
    /// Attribute tables: sources with no map presence, restored so a
    /// saved query still has everything it referenced. Absent in contexts
    /// written before they existed.
    #[serde(default)]
    pub tables: Vec<TableCtx>,
}

#[derive(Serialize, Deserialize)]
pub struct TableCtx {
    pub source: SourceCtx,
    /// Display name, which is also what its SQL identifier derives from.
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct LayerCtx {
    pub source: SourceCtx,
    pub style: StyleCtx,
    /// Persistent layer filter (SQL predicate), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// Display label. Absent in contexts written before labels existed,
    /// and there the loader's filename-derived name is the right answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceCtx {
    Local {
        path: String,
    },
    /// A hive-partitioned dataset directory.
    Dir {
        path: String,
    },
    Remote {
        url: String,
    },
    S3 {
        uri: String,
        profile: Option<String>,
        endpoint: Option<String>,
    },
    /// A STAC type collection (parts re-selected for the viewport at load).
    Stac {
        url: String,
        name: String,
    },
    /// A fixed set of remote parts loaded as one layer.
    Multi {
        name: String,
        urls: Vec<String>,
    },
}

/// A path that cannot be written to JSON without changing it. `display()`
/// substitutes U+FFFD for the bytes it cannot decode, so a context saved
/// from a non-UTF-8 path came back pointing at a file that does not
/// exist — silently, one layer at a time. The empty string is never a
/// usable path, so it is the sentinel [`save`] refuses on.
fn path_token(p: &std::path::Path) -> String {
    p.to_str().unwrap_or_default().to_string()
}

impl SourceCtx {
    /// The path or URL to write. A local path that is not valid UTF-8
    /// comes back empty; [`save`] turns that into a refusal naming the
    /// layer rather than writing a context that cannot be reloaded.
    pub fn of(source: &Source) -> Self {
        match source {
            Source::Local(p) => SourceCtx::Local { path: path_token(p) },
            Source::Dir(p) => SourceCtx::Dir { path: path_token(p) },
            Source::Remote { url, .. } => SourceCtx::Remote { url: url.clone() },
            Source::S3 {
                uri,
                profile,
                endpoint,
                ..
            } => SourceCtx::S3 {
                uri: uri.clone(),
                profile: profile.clone(),
                endpoint: endpoint.clone(),
            },
            Source::Stac { url, name } => SourceCtx::Stac {
                url: url.clone(),
                name: name.clone(),
            },
            Source::Multi { name, urls } => SourceCtx::Multi {
                name: name.clone(),
                urls: urls.clone(),
            },
        }
    }

    /// Whether this entry can be written and read back unchanged.
    /// Only local paths can fail, and only by not being UTF-8.
    fn writable(&self) -> bool {
        match self {
            SourceCtx::Local { path } | SourceCtx::Dir { path } => !path.is_empty(),
            _ => true,
        }
    }

    /// A local path that no longer exists. Checked before the session is
    /// cleared, so a context pointing at a deleted file reports which
    /// layer went missing instead of leaving an empty map behind.
    fn missing_local(&self) -> Option<&str> {
        match self {
            SourceCtx::Local { path } | SourceCtx::Dir { path } => {
                (!std::path::Path::new(path).exists()).then_some(path.as_str())
            }
            _ => None,
        }
    }

    /// Unresolved source; the loader re-probes / re-presigns at load.
    pub fn into_source(self) -> Source {
        match self {
            SourceCtx::Local { path } => Source::Local(path.into()),
            SourceCtx::Dir { path } => Source::Dir(path.into()),
            SourceCtx::Remote { url } => Source::Remote { url, len: 0 },
            SourceCtx::S3 {
                uri,
                profile,
                endpoint,
            } => Source::S3 {
                uri,
                profile,
                endpoint,
                url: String::new(),
                len: 0,
            },
            SourceCtx::Stac { url, name } => Source::Stac { url, name },
            SourceCtx::Multi { name, urls } => Source::Multi { name, urls },
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StyleCtx {
    pub visible: bool,
    pub show_rg_bboxes: bool,
    pub color: [u8; 4],
    pub line_color: Option<[u8; 4]>,
    pub line_width_px: f32,
    /// Dash pattern name (LinePattern::label); absent in older contexts.
    #[serde(default)]
    pub line_pattern: Option<String>,
    /// Cap name (LineCap::label); absent in older contexts.
    #[serde(default)]
    pub line_cap: Option<String>,
    pub point_radius_px: f32,
    /// Point marker name (PointShape::label); absent in older contexts.
    #[serde(default)]
    pub point_shape: Option<String>,
    pub fill_opacity: f32,
    pub opacity: f32,
    #[serde(default = "default_true")]
    pub fill_on: bool,
    #[serde(default = "default_true")]
    pub lines_on: bool,
    /// Data-driven styling: (column, ramp token, mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_by: Option<StyleByCtx>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum StyleByMode {
    Graduated {
        method: String,
        breaks: Vec<f64>,
    },
    Categorical {
        values: Vec<String>,
        /// Colour map applied to those values, when the style uses one.
        /// Absent in contexts written before colour maps existed, where
        /// the frequency palette is the right answer.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        colors: Option<Vec<[u8; 3]>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        labels: Option<Vec<String>>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StyleByCtx {
    pub column: String,
    pub ramp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classified_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub hidden_bins: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub per_area: bool,
    /// Line width ramp (min, max) in px; absent = uniform width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width_px: Option<(f32, f32)>,
    #[serde(flatten)]
    pub mode: StyleByMode,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

fn default_true() -> bool {
    true
}

impl StyleCtx {
    pub fn of(s: &LayerStyle) -> Self {
        use crate::data::layer::StyleMode;
        Self {
            visible: s.visible,
            show_rg_bboxes: s.show_rg_bboxes,
            color: s.color.to_array(),
            line_color: s.line_color.map(|c| c.to_array()),
            line_width_px: s.line_width_px,
            line_pattern: Some(s.line_pattern.label().to_string()),
            line_cap: Some(s.line_cap.label().to_string()),
            point_radius_px: s.point_radius_px,
            point_shape: Some(s.point_shape.label().to_string()),
            fill_opacity: s.fill_opacity,
            opacity: s.opacity,
            fill_on: s.fill_on,
            lines_on: s.lines_on,
            style_by: s.style_by.as_ref().map(|sb| StyleByCtx {
                column: sb.column.clone(),
                ramp: sb.ramp.label().to_string(),
                classified_rows: sb.classified_rows,
                hidden_bins: sb.hidden_bins,
                per_area: sb.per_area,
                width_px: sb.width_px,
                mode: match &sb.mode {
                    StyleMode::Graduated { method, breaks } => StyleByMode::Graduated {
                        method: method.label().to_string(),
                        breaks: breaks.clone(),
                    },
                    StyleMode::Categorical {
                        values,
                        colors,
                        labels,
                    } => StyleByMode::Categorical {
                        values: values.clone(),
                        colors: colors.clone(),
                        labels: labels.clone(),
                    },
                },
            }),
        }
    }

    pub fn into_style(self) -> LayerStyle {
        use crate::data::layer::{LineCap, LinePattern, PointShape, Ramp, StyleBy, StyleMode};
        let color = |a: [u8; 4]| Color32::from_rgba_premultiplied(a[0], a[1], a[2], a[3]);
        LayerStyle {
            visible: self.visible,
            show_rg_bboxes: self.show_rg_bboxes,
            color: color(self.color),
            line_color: self.line_color.map(color),
            line_width_px: self.line_width_px,
            line_pattern: self
                .line_pattern
                .as_deref()
                .and_then(LinePattern::from_label)
                .unwrap_or_default(),
            line_cap: self
                .line_cap
                .as_deref()
                .and_then(LineCap::from_label)
                .unwrap_or_default(),
            point_radius_px: self.point_radius_px,
            point_shape: self
                .point_shape
                .as_deref()
                .and_then(PointShape::from_label)
                .unwrap_or_default(),
            fill_opacity: self.fill_opacity,
            opacity: self.opacity,
            fill_on: self.fill_on,
            lines_on: self.lines_on,
            style_by: self.style_by.map(|sb| StyleBy {
                column: sb.column,
                ramp: Ramp::ALL
                    .iter()
                    .copied()
                    .find(|r| r.label() == sb.ramp)
                    .unwrap_or(Ramp::Viridis),
                classified_rows: sb.classified_rows,
                hidden_bins: sb.hidden_bins,
                per_area: sb.per_area,
                width_px: sb.width_px,
                mode: match sb.mode {
                    StyleByMode::Graduated { method, breaks } => StyleMode::Graduated {
                        method: crate::data::layer::ClassMethod::ALL
                            .iter()
                            .copied()
                            .find(|m| m.label() == method)
                            .unwrap_or(crate::data::layer::ClassMethod::EqualInterval),
                        breaks,
                    },
                    StyleByMode::Categorical {
                        values,
                        colors,
                        labels,
                    } => StyleMode::Categorical {
                        values,
                        colors,
                        labels,
                    },
                },
            }),
        }
    }
}

/// Write the context to `path`, atomically and only if every entry can
/// be read back.
///
/// `fs::write` truncates first: interrupting it left a context file that
/// parses as nothing at all, and the session it described was gone. Temp
/// file + `sync_all` + rename means the old context survives a failure
/// intact.
pub fn save(ctx: &Context, path: &std::path::Path) -> Result<(), String> {
    let unwritable = ctx
        .layers
        .iter()
        .find(|l| !l.source.writable())
        .map(|l| l.name.clone().unwrap_or_else(|| "a layer".into()))
        .or_else(|| {
            ctx.tables
                .iter()
                .find(|t| !t.source.writable())
                .map(|t| t.name.clone())
            });
    if let Some(name) = unwritable {
        return Err(format!(
            "{name} loads from a path that is not valid UTF-8, which cannot \
             be written to a context file"
        ));
    }
    let json = serde_json::to_string_pretty(ctx).map_err(|e| e.to_string())?;
    write_atomic(path, &json)
}

fn write_atomic(path: &std::path::Path, data: &str) -> Result<(), String> {
    use std::io::Write;
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
        e.to_string()
    })
}

/// Read a context file. Returns the context and the entries that were
/// dropped, so the caller can say what it could not restore.
///
/// Three things this does that `serde_json::from_str` alone did not:
///
/// - refuse a file from a newer build, instead of writing `version` and
///   never reading it. Serde's `#[serde(default)]` fields make a future
///   context parse *successfully* into something missing whatever that
///   build added, so the silent failure is a half-restored session;
/// - parse the layers and tables one at a time, so a single entry
///   serde cannot read does not take the other nineteen with it;
/// - check local paths here, before the caller clears the session:
///   discovering a moved file after the wipe leaves an empty map.
pub fn load(path: &std::path::Path) -> Result<(Context, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut root: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let version = root.get("version").and_then(serde_json::Value::as_u64).unwrap_or(0);
    if version > CONTEXT_VERSION as u64 {
        return Err(format!(
            "context version {version} was written by a newer build \
             (this one reads up to {CONTEXT_VERSION})"
        ));
    }

    let mut warnings: Vec<String> = Vec::new();
    let layers: Vec<LayerCtx> = take_each(&mut root, "layers", "layer", &mut warnings);
    let tables: Vec<TableCtx> = take_each(&mut root, "tables", "table", &mut warnings);
    let mut ctx: Context = serde_json::from_value(root).map_err(|e| e.to_string())?;

    ctx.layers = layers
        .into_iter()
        .filter(|l| match l.source.missing_local() {
            Some(p) => {
                warnings.push(format!("{p} is gone; that layer was skipped"));
                false
            }
            None => true,
        })
        .collect();
    ctx.tables = tables
        .into_iter()
        .filter(|t| match t.source.missing_local() {
            Some(p) => {
                warnings.push(format!("{p} is gone; that table was skipped"));
                false
            }
            None => true,
        })
        .collect();
    Ok((ctx, warnings))
}

/// Take one array out of the document and parse its entries
/// independently, replacing it with an empty array so the surrounding
/// `Context` still deserializes. Entries that fail become warnings.
fn take_each<T: serde::de::DeserializeOwned>(
    root: &mut serde_json::Value,
    key: &str,
    what: &str,
    warnings: &mut Vec<String>,
) -> Vec<T> {
    let raw = match root.get_mut(key).map(serde_json::Value::take) {
        Some(serde_json::Value::Array(a)) => a,
        _ => return Vec::new(),
    };
    root[key] = serde_json::Value::Array(Vec::new());
    raw.into_iter()
        .enumerate()
        .filter_map(|(i, v)| match serde_json::from_value::<T>(v) {
            Ok(t) => Some(t),
            Err(e) => {
                warnings.push(format!("{what} {} could not be read: {e}", i + 1));
                None
            }
        })
        .collect()
}

pub fn projection_token(d: &DisplayCrs) -> String {
    if d.name.starts_with("Hobo") {
        "hobo-dyer".into()
    } else if d.kind == DisplayKind::WinkelTripel {
        "winkel-tripel".into()
    } else if d.is_mercator() {
        "mercator".into()
    } else if let Some(e) = d.crs.epsg {
        format!("epsg:{e}")
    } else {
        // Auto-fit / custom projections: persist the proj string itself.
        format!("proj4:{}", d.crs.proj4)
    }
}

pub fn projection_from_token(t: &str) -> Result<DisplayCrs, String> {
    match t {
        "hobo-dyer" => Ok(DisplayCrs::hobo_dyer()),
        "winkel-tripel" => Ok(DisplayCrs::winkel_tripel()),
        "mercator" => Ok(DisplayCrs::mercator()),
        other => {
            if let Some(proj4) = other.strip_prefix("proj4:") {
                let crs = crate::data::crs::Crs::from_proj4(proj4, None, "custom projection")?;
                return Ok(DisplayCrs::new(crs));
            }
            other
                .strip_prefix("epsg:")
                .and_then(|c| c.parse::<u32>().ok())
                .ok_or_else(|| format!("unknown projection '{other}'"))
                .and_then(DisplayCrs::from_epsg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_roundtrip() {
        let ctx = Context {
            version: CONTEXT_VERSION,
            camera_center: [0.51, 0.29],
            camera_zoom: 13.25,
            projection: "epsg:2154".into(),
            projection_name: None,
            basemap: Some(1),
            basemap_opacity: Some(1.0),
            show_graticule: false,
            show_coastline: true,
            box_threshold_px: None,
            refine_budget_mb: None,
            layers: vec![
                LayerCtx {
                    source: SourceCtx::Local {
                        path: "/data/parcels.parquet".into(),
                    },
                    style: StyleCtx {
                        style_by: None,
                        visible: true,
                        show_rg_bboxes: true,
                        color: [31, 119, 180, 255],
                        line_color: Some([10, 20, 30, 255]),
                        line_width_px: 1.5,
                        line_pattern: Some("dash".into()),
                        line_cap: Some("flat".into()),
                        point_radius_px: 3.0,
                        point_shape: Some("star".into()),
                        fill_opacity: 0.4,
                        opacity: 0.9,
                        fill_on: true,
                        lines_on: false,
                    },
                    filter: Some("status = 'active'".into()),
                    name: Some("Parcels (2024)".into()),
                },
                LayerCtx {
                    source: SourceCtx::S3 {
                        uri: "s3://parquetry/latest/buildings.parquet".into(),
                        profile: None,
                        endpoint: Some("s3.geomermaids.com".into()),
                    },
                    style: StyleCtx::of(&LayerStyle::new(Color32::RED)),
                    filter: None,
                    name: None,
                },
            ],
            tables: vec![TableCtx {
                source: SourceCtx::Local {
                    path: "/data/codes.csv".into(),
                },
                name: "codes".into(),
            }],
        };
        let json = serde_json::to_string_pretty(&ctx).unwrap();
        let back: Context = serde_json::from_str(&json).unwrap();
        assert_eq!(back.layers.len(), 2);
        // Attribute tables come back too: a saved query that joined one
        // would otherwise restore into a session missing half its FROM.
        assert_eq!(back.tables.len(), 1);
        assert_eq!(back.tables[0].name, "codes");

        // A context written before tables existed still loads, with none.
        let older = json.replace("\"tables\"", "\"tables_removed\"");
        let back: Context = serde_json::from_str(&older).unwrap();
        assert!(back.tables.is_empty(), "absent means none, not an error");
        // The display label is the one piece of layer state a user types
        // by hand; a session that forgets it loses work silently.
        assert_eq!(back.layers[0].name.as_deref(), Some("Parcels (2024)"));
        assert_eq!(back.layers[1].name, None);
        assert_eq!(back.projection, "epsg:2154");
        assert_eq!(back.camera_zoom, 13.25);
        let style = back.layers[0].style.clone().into_style();
        assert_eq!(
            style.line_color,
            Some(Color32::from_rgba_premultiplied(10, 20, 30, 255))
        );
        match back.layers[1].source {
            SourceCtx::S3 { ref endpoint, .. } => {
                assert_eq!(endpoint.as_deref(), Some("s3.geomermaids.com"));
                let src = SourceCtx::S3 {
                    uri: "s3://x/y".into(),
                    profile: None,
                    endpoint: endpoint.clone(),
                }
                .into_source();
                assert!(matches!(src, Source::S3 { len: 0, .. }));
            }
            _ => panic!("expected s3 source"),
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "geopq_ctx_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn minimal(layers: Vec<LayerCtx>) -> Context {
        Context {
            version: CONTEXT_VERSION,
            camera_center: [0.0, 0.0],
            camera_zoom: 1.0,
            projection: "mercator".into(),
            projection_name: None,
            basemap: None,
            basemap_opacity: None,
            show_graticule: false,
            show_coastline: false,
            box_threshold_px: None,
            refine_budget_mb: None,
            layers,
            tables: Vec::new(),
        }
    }

    fn layer_at(path: &std::path::Path, name: &str) -> LayerCtx {
        LayerCtx {
            source: SourceCtx::Local {
                path: path.display().to_string(),
            },
            style: StyleCtx {
                visible: true,
                show_rg_bboxes: false,
                color: [1, 2, 3, 4],
                line_color: None,
                line_width_px: 1.0,
                line_pattern: None,
                line_cap: None,
                point_radius_px: 2.0,
                point_shape: None,
                fill_opacity: 1.0,
                opacity: 1.0,
                fill_on: true,
                lines_on: true,
                style_by: None,
            },
            filter: None,
            name: Some(name.into()),
        }
    }

    /// `fs::write` truncates before it writes, so an interrupted save
    /// left a context file that parses as nothing and a session that was
    /// gone with it. The old file has to survive intact until the new one
    /// is complete.
    #[test]
    fn saving_replaces_the_file_in_one_step() {
        let dir = temp_dir("atomic");
        let data = dir.join("layer.parquet");
        std::fs::write(&data, b"x").unwrap();
        let path = dir.join("session.geopqctx");

        save(&minimal(vec![layer_at(&data, "one")]), &path).unwrap();
        let first = std::fs::read_to_string(&path).unwrap();
        save(&minimal(vec![layer_at(&data, "two")]), &path).unwrap();
        let second = std::fs::read_to_string(&path).unwrap();
        assert_ne!(first, second);
        assert!(second.contains("\"two\""));
        // No temp file left for the next save to trip over.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");

        // A save that cannot be completed leaves the old context alone.
        let nowhere = dir.join("missing").join("session.geopqctx");
        assert!(save(&minimal(Vec::new()), &nowhere).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), second);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `version` was written and never read. A file from a newer build
    /// parses *successfully* — every added field is `#[serde(default)]` —
    /// into a session missing whatever that build added, which is worse
    /// than refusing it.
    #[test]
    fn a_newer_context_is_refused_rather_than_half_read() {
        let dir = temp_dir("version");
        let path = dir.join("session.geopqctx");
        save(&minimal(Vec::new()), &path).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        v["version"] = serde_json::json!(CONTEXT_VERSION + 1);
        v["something_new"] = serde_json::json!({"a": 1});
        std::fs::write(&path, v.to_string()).unwrap();

        let err = match load(&path) {
            Err(e) => e,
            Ok(_) => panic!("a newer context must be refused"),
        };
        assert!(err.contains("newer build"), "{err}");
        // The current version still loads, unknown keys and all.
        v["version"] = serde_json::json!(CONTEXT_VERSION);
        std::fs::write(&path, v.to_string()).unwrap();
        assert!(load(&path).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One layer entry serde cannot read used to sink the whole file:
    /// twenty layers, one written by a build with a different style enum,
    /// and the restore was "context load failed".
    #[test]
    fn one_unreadable_layer_does_not_sink_the_rest() {
        let dir = temp_dir("partial");
        let data = dir.join("layer.parquet");
        std::fs::write(&data, b"x").unwrap();
        let path = dir.join("session.geopqctx");
        save(
            &minimal(vec![layer_at(&data, "one"), layer_at(&data, "two")]),
            &path,
        )
        .unwrap();

        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        v["layers"][0]["style"] = serde_json::json!("not a style object");
        std::fs::write(&path, v.to_string()).unwrap();

        let (ctx, warnings) = load(&path).unwrap();
        assert_eq!(ctx.layers.len(), 1, "the readable layer survives");
        assert_eq!(ctx.layers[0].name.as_deref(), Some("two"));
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("layer 1"), "{warnings:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A layer whose file has moved is reported by `load`, before the
    /// caller clears the session — discovering it afterwards leaves an
    /// empty map and no way back.
    #[test]
    fn a_moved_source_is_reported_by_load_not_after_the_wipe() {
        let dir = temp_dir("missing");
        let here = dir.join("here.parquet");
        std::fs::write(&here, b"x").unwrap();
        let gone = dir.join("gone.parquet");
        let path = dir.join("session.geopqctx");
        save(
            &minimal(vec![layer_at(&here, "here"), layer_at(&gone, "gone")]),
            &path,
        )
        .unwrap();

        let (ctx, warnings) = load(&path).unwrap();
        assert_eq!(ctx.layers.len(), 1);
        assert_eq!(ctx.layers[0].name.as_deref(), Some("here"));
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("gone.parquet"), "{warnings:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `display()` replaces the bytes it cannot decode, so a non-UTF-8
    /// path was saved as a path that does not exist. Refuse, and name the
    /// layer, rather than writing a context that silently loses it.
    #[test]
    fn a_non_utf8_path_is_refused_by_name() {
        // Only Unix filesystems can hold one.
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let bad = std::path::PathBuf::from(std::ffi::OsStr::from_bytes(b"/tmp/caf\xe9.parquet"));
            let src = SourceCtx::of(&Source::Local(bad));
            assert!(matches!(&src, SourceCtx::Local { path } if path.is_empty()));
            let mut layer = layer_at(std::path::Path::new("/tmp/x.parquet"), "cafe");
            layer.source = src;
            let err = save(&minimal(vec![layer]), &temp_dir("utf8").join("c.json")).unwrap_err();
            assert!(err.contains("cafe"), "{err}");
            assert!(err.contains("UTF-8"), "{err}");
        }
        // A plain path still round-trips.
        let src = SourceCtx::of(&Source::Local("/tmp/ok.parquet".into()));
        assert!(matches!(&src, SourceCtx::Local { path } if path == "/tmp/ok.parquet"));
    }

    #[test]
    fn projection_tokens_roundtrip() {
        for t in ["hobo-dyer", "winkel-tripel", "mercator", "epsg:2154", "epsg:4326"] {
            let d = projection_from_token(t).unwrap();
            assert_eq!(projection_token(&d), t, "token {t}");
        }
        assert!(projection_from_token("bogus").is_err());

        // National auto picks persist as clean EPSG tokens...
        let auto = crate::data::crs::DisplayCrs::auto_for(
            &crate::data::crs::Crs::wgs84(),
            Some([-5.0, 41.0, 10.0, 51.0]),
        )
        .unwrap();
        assert_eq!(projection_token(&auto), "epsg:2154");
        // ...and custom extent-fit ones as proj4 tokens.
        let auto = crate::data::crs::DisplayCrs::auto_for(
            &crate::data::crs::Crs::wgs84(),
            Some([60.0, 35.0, 75.0, 50.0]),
        )
        .unwrap();
        let t = projection_token(&auto);
        assert!(t.starts_with("proj4:+proj=laea"), "{t}");
        let back = projection_from_token(&t).unwrap();
        assert_eq!(back.crs.proj4, auto.crs.proj4);
    }
}
