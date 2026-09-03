//! File-level metadata gathered at load time for the "File info" panel.
//!
//! GeoParquet versions this viewer knows about:
//! - 1.0: `geo` key-value metadata, WKB-encoded geometry columns.
//! - 1.1: adds native GeoArrow encodings ("point", "multipolygon", ...),
//!   `covering` (bbox columns) and `edges`.
//! - 2.0: geometry carried by Parquet's native GEOMETRY/GEOGRAPHY logical
//!   types (still WKB bytes inside), `geo` metadata optional.

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ColumnInfo {
    pub name: String,
    pub arrow_type: String,
    pub compression: String,
    pub logical: Option<String>,
    pub is_geometry: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GeoParquetInfo {
    /// e.g. "GeoParquet 1.1.0", "GeoParquet 2.0.0 + native GEOMETRY logical type",
    /// "none (guessed WKB column)".
    pub version_label: String,
    pub primary_column: String,
    pub encoding: String,
    pub geometry_types: Vec<String>,
    pub bbox: Option<[f64; 4]>,
    #[allow(dead_code)]
    pub crs_summary: String,
    pub covering: Option<String>,
    pub edges: Option<String>,
    /// Cloud Optimized GeoParquet Profile levels, summarized — or why a
    /// `cogp` block was rejected.
    pub cogp: Option<String>,
    /// Pretty-printed raw `geo` metadata JSON.
    pub raw_geo_json: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FileInfo {
    pub file_size: u64,
    pub parquet_format_version: i32,
    pub created_by: Option<String>,
    pub rows: u64,
    pub row_groups: usize,
    pub rg_rows_min: u64,
    pub rg_rows_max: u64,
    /// Uncompressed bytes of the largest row group: the real decode unit.
    /// Rows alone do not measure it — a few thousand admin boundaries
    /// carry more bytes than a million points.
    pub rg_bytes_max: u64,
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
    pub columns: Vec<ColumnInfo>,
    pub geo: GeoParquetInfo,
    /// Files in the dataset (1 for a plain file, N for a directory).
    pub files: usize,
    /// Credit the data asks for, from its metadata or a neighbouring
    /// ATTRIBUTION.txt. None when the data carries none.
    pub attribution: Option<super::attribution::Attribution>,
    /// Footer-only display-readiness analysis (docs/OPEN_POLICY.md);
    /// None only for stores that never went through a parquet footer.
    pub quality: Option<super::quality::QualityReport>,
    /// One-line summary of the H3 pyramid this dataset was opened from,
    /// when it has one (`PyramidState::info_line`).
    pub pyramid: Option<String>,
    /// The file's own `geopq:pyramid` entry, when it carries one: what
    /// an overview file says about itself, so a part opened alone is
    /// still badged as derived data.
    pub pyramid_file: Option<super::pyramid::FileMeta>,
}

pub fn fmt_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.0} kB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

/// Summarize the `geo` key-value metadata for display.
pub fn summarize_geo_meta(
    geo_meta: Option<&Value>,
    primary_fallback: &str,
    crs_name: &str,
    has_native_geometry_type: bool,
) -> GeoParquetInfo {
    let mut info = GeoParquetInfo {
        crs_summary: crs_name.to_string(),
        ..Default::default()
    };
    match geo_meta {
        Some(meta) => {
            let version = meta
                .get("version")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            info.version_label = if has_native_geometry_type {
                format!("GeoParquet {version} + native GEOMETRY logical type (2.0)")
            } else {
                format!("GeoParquet {version}")
            };
            let primary = meta
                .get("primary_column")
                .and_then(Value::as_str)
                .unwrap_or(primary_fallback);
            info.primary_column = primary.to_string();
            if let Some(col) = meta.get("columns").and_then(|c| c.get(primary)) {
                info.encoding = col
                    .get("encoding")
                    .and_then(Value::as_str)
                    .unwrap_or("WKB")
                    .to_string();
                // The Parquet GEOMETRY logical type is an annotation on a
                // binary column whose values are always WKB — spell that
                // out so "native type" + "WKB" don't read as contradictory.
                if has_native_geometry_type && info.encoding == "WKB" {
                    info.encoding =
                        "WKB (the GEOMETRY logical type stores WKB bytes)".into();
                }
                if let Some(types) = col.get("geometry_types").and_then(Value::as_array) {
                    info.geometry_types = types
                        .iter()
                        .filter_map(Value::as_str)
                        .map(String::from)
                        .collect();
                }
                if let Some(bbox) = col.get("bbox").and_then(Value::as_array) {
                    // Every element must be a number, or the box is not
                    // one. Dropping the ones that are not (a stringified
                    // "-5.0", a null) would re-index the rest and hand
                    // row-group pruning a box shifted by a slot — which
                    // prunes away real data rather than failing loudly.
                    let v: Option<Vec<f64>> = bbox.iter().map(Value::as_f64).collect();
                    // 6 elements = 3D per spec: [xmin, ymin, zmin, xmax,
                    // ymax, zmax]. Taking the first four would put zmin/
                    // xmax into the xmax/ymax slots (and this bbox feeds
                    // row-group pruning as a fallback).
                    info.bbox = v.and_then(|v| match v.len() {
                        6 => Some([v[0], v[1], v[3], v[4]]),
                        4 => Some([v[0], v[1], v[2], v[3]]),
                        _ => None,
                    })
                    .filter(|b| {
                        b.iter().all(|c| c.is_finite()) && b[0] <= b[2] && b[1] <= b[3]
                    });
                }
                if let Some(cov) = col.get("covering") {
                    // Human summary instead of the raw JSON: the column the
                    // four bbox leaves live in (per spec a struct column
                    // referenced as ["<column>", "<field>"]).
                    let col_name = cov
                        .get("bbox")
                        .and_then(|b| b.get("xmin"))
                        .and_then(Value::as_array)
                        .and_then(|p| p.first())
                        .and_then(Value::as_str)
                        .unwrap_or("?");
                    info.covering = Some(format!(
                        "per-feature bbox struct \"{col_name}\" (drives row/page pruning)"
                    ));
                }
                if let Some(e) = col.get("edges").and_then(Value::as_str) {
                    info.edges = Some(e.to_string());
                }
            }
            info.raw_geo_json = serde_json::to_string_pretty(meta).ok();
        }
        None => {
            info.version_label = if has_native_geometry_type {
                "GeoParquet 2.0 (native GEOMETRY logical type, no geo metadata)".into()
            } else {
                "none (guessed WKB column, CRS assumed OGC:CRS84)".into()
            };
            info.primary_column = primary_fallback.to_string();
            info.encoding = if has_native_geometry_type {
                "WKB (the GEOMETRY logical type stores WKB bytes)".into()
            } else {
                "WKB (assumed)".into()
            };
        }
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_parses_2d_and_3d() {
        let meta = |bbox: &str| -> GeoParquetInfo {
            let m: Value = serde_json::from_str(&format!(
                r#"{{"version":"1.0.0","primary_column":"geometry",
                     "columns":{{"geometry":{{"encoding":"WKB","bbox":{bbox}}}}}}}"#
            ))
            .unwrap();
            summarize_geo_meta(Some(&m), "geometry", "WGS 84", false)
        };
        // 2D bbox passes through.
        assert_eq!(meta("[-5.0, 41.0, 10.0, 51.0]").bbox, Some([-5.0, 41.0, 10.0, 51.0]));
        // 3D bbox is [xmin, ymin, zmin, xmax, ymax, zmax] per spec: the
        // 2D box must be [xmin, ymin, xmax, ymax], not the first four.
        assert_eq!(
            meta("[-5.0, 41.0, 0.0, 10.0, 51.0, 200.0]").bbox,
            Some([-5.0, 41.0, 10.0, 51.0])
        );
    }

    /// A bbox that is not four (or six) numbers is no bbox at all.
    /// Skipping the non-numeric elements re-indexed the rest, so a
    /// stringified xmin silently produced [ymin, xmax, ymax, ?] — and
    /// this box is a row-group pruning fallback, so a shifted one prunes
    /// away data that is really in view.
    #[test]
    fn a_malformed_bbox_is_rejected_rather_than_re_indexed() {
        let meta = |bbox: &str| -> GeoParquetInfo {
            let m: Value = serde_json::from_str(&format!(
                r#"{{"version":"1.0.0","primary_column":"geometry",
                     "columns":{{"geometry":{{"encoding":"WKB","bbox":{bbox}}}}}}}"#
            ))
            .unwrap();
            summarize_geo_meta(Some(&m), "geometry", "WGS 84", false)
        };
        // A stringified xmin used to shift every other element down one.
        assert_eq!(meta(r#"["-5.0", 41.0, 10.0, 51.0]"#).bbox, None);
        assert_eq!(meta("[null, 41.0, 10.0, 51.0]").bbox, None);
        // Too few / too many elements.
        assert_eq!(meta("[-5.0, 41.0, 10.0]").bbox, None);
        assert_eq!(meta("[-5.0, 41.0, 0.0, 10.0, 51.0]").bbox, None);
        // Non-finite and inverted boxes are not usable either.
        assert_eq!(meta(r#"[-5.0, 41.0, 10.0, "NaN"]"#).bbox, None);
        assert_eq!(meta("[10.0, 41.0, -5.0, 51.0]").bbox, None);
        // The well-formed one still passes.
        assert_eq!(meta("[-5.0, 41.0, 10.0, 51.0]").bbox, Some([-5.0, 41.0, 10.0, 51.0]));
    }
}
