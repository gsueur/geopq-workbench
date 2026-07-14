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
    /// e.g. "GeoParquet 1.1.0", "GeoParquet 2.0 (native GEOMETRY type)",
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
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
    pub columns: Vec<ColumnInfo>,
    pub geo: GeoParquetInfo,
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
                format!("GeoParquet {version} + native GEOMETRY type (2.0)")
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
                if let Some(types) = col.get("geometry_types").and_then(Value::as_array) {
                    info.geometry_types = types
                        .iter()
                        .filter_map(Value::as_str)
                        .map(String::from)
                        .collect();
                }
                if let Some(bbox) = col.get("bbox").and_then(Value::as_array) {
                    let v: Vec<f64> = bbox.iter().filter_map(Value::as_f64).collect();
                    if v.len() >= 4 {
                        info.bbox = Some([v[0], v[1], v[2], v[3]]);
                    }
                }
                if col.get("covering").is_some() {
                    info.covering = Some(
                        serde_json::to_string(col.get("covering").unwrap()).unwrap_or_default(),
                    );
                }
                if let Some(e) = col.get("edges").and_then(Value::as_str) {
                    info.edges = Some(e.to_string());
                }
            }
            info.raw_geo_json = serde_json::to_string_pretty(meta).ok();
        }
        None => {
            info.version_label = if has_native_geometry_type {
                "GeoParquet 2.0 (native GEOMETRY type, no geo metadata)".into()
            } else {
                "none (guessed WKB column, CRS assumed OGC:CRS84)".into()
            };
            info.primary_column = primary_fallback.to_string();
            info.encoding = "WKB (assumed)".into();
        }
    }
    info
}
