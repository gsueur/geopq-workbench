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
    pub show_graticule: bool,
    pub show_coastline: bool,
    /// Bottom-to-top draw order.
    pub layers: Vec<LayerCtx>,
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

impl SourceCtx {
    pub fn of(source: &Source) -> Self {
        match source {
            Source::Local(p) => SourceCtx::Local {
                path: p.display().to_string(),
            },
            Source::Dir(p) => SourceCtx::Dir {
                path: p.display().to_string(),
            },
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
    pub point_radius_px: f32,
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
            point_radius_px: s.point_radius_px,
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
        use crate::data::layer::{Ramp, StyleBy, StyleMode};
        let color = |a: [u8; 4]| Color32::from_rgba_premultiplied(a[0], a[1], a[2], a[3]);
        LayerStyle {
            visible: self.visible,
            show_rg_bboxes: self.show_rg_bboxes,
            color: color(self.color),
            line_color: self.line_color.map(color),
            line_width_px: self.line_width_px,
            point_radius_px: self.point_radius_px,
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
            show_graticule: false,
            show_coastline: true,
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
                        point_radius_px: 3.0,
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
        };
        let json = serde_json::to_string_pretty(&ctx).unwrap();
        let back: Context = serde_json::from_str(&json).unwrap();
        assert_eq!(back.layers.len(), 2);
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
