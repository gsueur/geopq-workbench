//! Shared machinery for vector-format imports (GeoPackage, Shapefile,
//! GeoJSON). Every importer writes the same thing: plain WKB GeoParquet
//! 1.1 in source order — a faithful raw export whose quality scorecard
//! then shows what is missing, with Optimize one click away.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder,
};
use arrow::datatypes::DataType;
use geo::BoundingRect;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use serde_json::{json, Value};

pub const IMPORT_BATCH_ROWS: usize = 65_536;

/// Formats the File → Import vector file… dialog accepts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImportFormat {
    Gpkg,
    Shapefile,
    GeoJson,
}

impl ImportFormat {
    pub fn from_path(p: &Path) -> Option<Self> {
        match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "gpkg" => Some(Self::Gpkg),
            "shp" => Some(Self::Shapefile),
            "geojson" | "json" => Some(Self::GeoJson),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Gpkg => "GeoPackage",
            Self::Shapefile => "Shapefile",
            Self::GeoJson => "GeoJSON",
        }
    }
}

/// Writer properties every importer uses: zstd (fast level — imports are
/// intermediates, Optimize writes the distribution-grade file), 65k row
/// groups, page statistics.
pub(crate) fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_max_row_group_row_count(Some(IMPORT_BATCH_ROWS))
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_created_by(format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")))
        .build()
}

/// GeoParquet 1.1 `geo` metadata for an import. `crs`: `None` omits the
/// key entirely (CRS84 per spec — right for GeoJSON), `Some(Value::Null)`
/// records an explicitly unknown CRS, anything else is written verbatim.
pub(crate) fn geo_meta(primary: &str, stats: &GeomStats, crs: Option<Value>) -> Value {
    let mut col = json!({
        "encoding": "WKB",
        "geometry_types": stats.types.iter().collect::<Vec<_>>(),
    });
    if let Some(c) = crs {
        col["crs"] = c;
    }
    let b = stats.bbox;
    if b[0].is_finite() && b[2].is_finite() {
        col["bbox"] = json!([b[0], b[1], b[2], b[3]]);
    }
    json!({
        "version": "1.1.0",
        "primary_column": primary,
        "columns": { primary: col },
    })
}

/// Geometry bookkeeping every importer needs: observed geometry types
/// (for the metadata and the GeoArrow recommendation downstream) and the
/// dataset bbox.
pub(crate) struct GeomStats {
    pub types: BTreeSet<String>,
    pub bbox: [f64; 4],
}

impl GeomStats {
    pub fn new() -> Self {
        Self {
            types: BTreeSet::new(),
            bbox: [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        }
    }

    pub fn add(&mut self, g: &geo_types::Geometry<f64>) {
        use geo_types::Geometry::*;
        self.types.insert(
            match g {
                Point(_) => "Point",
                Line(_) | LineString(_) => "LineString",
                Polygon(_) | Rect(_) | Triangle(_) => "Polygon",
                MultiPoint(_) => "MultiPoint",
                MultiLineString(_) => "MultiLineString",
                MultiPolygon(_) => "MultiPolygon",
                GeometryCollection(_) => "GeometryCollection",
            }
            .to_string(),
        );
        if let Some(r) = g.bounding_rect() {
            self.bbox[0] = self.bbox[0].min(r.min().x);
            self.bbox[1] = self.bbox[1].min(r.min().y);
            self.bbox[2] = self.bbox[2].max(r.max().x);
            self.bbox[3] = self.bbox[3].max(r.max().y);
        }
    }
}

/// ISO WKB bytes of a geometry.
pub(crate) fn to_wkb(g: &geo_types::Geometry<f64>) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    wkb::writer::write_geometry(&mut buf, g, &wkb::writer::WriteOptions::default())
        .map_err(|e| format!("WKB encode: {e}"))?;
    Ok(buf)
}

/// One dynamically-typed cell — the lingua franca between source rows
/// (SQLite values, JSON properties, DBF fields) and the arrow builders.
pub(crate) enum Cell<'a> {
    Null,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(Cow<'a, str>),
    Bytes(&'a [u8]),
}

/// Typed column builder with per-cell coercion: source cells that don't
/// fit the column's declared type become nulls (numbers widen, anything
/// stringifies into text columns) rather than errors.
pub(crate) enum AttrBuilder {
    Int(Int64Builder),
    Float(Float64Builder),
    Bool(BooleanBuilder),
    Text(StringBuilder),
    Blob(BinaryBuilder),
}

impl AttrBuilder {
    pub fn new(dt: &DataType) -> Self {
        match dt {
            DataType::Int64 => Self::Int(Int64Builder::new()),
            DataType::Float64 => Self::Float(Float64Builder::new()),
            DataType::Boolean => Self::Bool(BooleanBuilder::new()),
            DataType::Binary => Self::Blob(BinaryBuilder::new()),
            _ => Self::Text(StringBuilder::new()),
        }
    }

    pub fn push(&mut self, v: Cell<'_>) {
        match self {
            Self::Int(b) => match v {
                Cell::Int(i) => b.append_value(i),
                _ => b.append_null(),
            },
            Self::Float(b) => match v {
                Cell::Float(f) => b.append_value(f),
                Cell::Int(i) => b.append_value(i as f64),
                _ => b.append_null(),
            },
            Self::Bool(b) => match v {
                Cell::Bool(x) => b.append_value(x),
                Cell::Int(i) => b.append_value(i != 0),
                _ => b.append_null(),
            },
            Self::Text(b) => match v {
                Cell::Str(s) => b.append_value(s),
                Cell::Int(i) => b.append_value(i.to_string()),
                Cell::Float(f) => b.append_value(f.to_string()),
                Cell::Bool(x) => b.append_value(x.to_string()),
                _ => b.append_null(),
            },
            Self::Blob(b) => match v {
                Cell::Bytes(x) => b.append_value(x),
                _ => b.append_null(),
            },
        }
    }

    pub fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Int(b) => Arc::new(b.finish()),
            Self::Float(b) => Arc::new(b.finish()),
            Self::Bool(b) => Arc::new(b.finish()),
            Self::Text(b) => Arc::new(b.finish()),
            Self::Blob(b) => Arc::new(b.finish()),
        }
    }
}
