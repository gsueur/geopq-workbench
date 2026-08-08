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
use arrow::datatypes::{DataType, Field};
use geo::BoundingRect;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use serde_json::{json, Value};

pub const IMPORT_BATCH_ROWS: usize = 65_536;

/// Formats the File → Import vector file… dialog accepts. `Gdb` only
/// converts when built with `--features gdal-import` (it needs a system
/// GDAL — the one format nothing pure-Rust can read); the variant still
/// exists in default builds so match arms stay exhaustive, `from_path`
/// just never produces it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImportFormat {
    Gpkg,
    Shapefile,
    GeoJson,
    Gdb,
}

impl ImportFormat {
    pub fn from_path(p: &Path) -> Option<Self> {
        match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "gpkg" => Some(Self::Gpkg),
            "shp" => Some(Self::Shapefile),
            "geojson" | "json" => Some(Self::GeoJson),
            #[cfg(feature = "gdal-import")]
            "gdb" => Some(Self::Gdb),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Gpkg => "GeoPackage",
            Self::Shapefile => "Shapefile",
            Self::GeoJson => "GeoJSON",
            Self::Gdb => "File Geodatabase",
        }
    }

    /// Whether the source path is a directory rather than a single file
    /// (a File Geodatabase is a `.gdb` folder, not a file).
    pub fn is_directory(&self) -> bool {
        matches!(self, Self::Gdb)
    }
}

/// One selectable layer inside a multi-layer import source (GeoPackage
/// feature tables, File Geodatabase layers). Kept independent of the
/// `gdal` crate so `ImportState` compiles the same with or without the
/// `gdal-import` feature; only `data::gdb` (cfg-gated) actually builds
/// these.
#[derive(Clone)]
pub struct GdbLayer {
    pub name: String,
    pub rows: u64,
    /// EPSG code, when the layer's spatial reference maps to one.
    pub epsg: Option<u32>,
    pub srs_name: String,
}

/// Byte cap per row group. An import of heavy polygons (land cover,
/// cadastre) filled 65k-row groups with 700 MB each, and a row group is
/// the unit a reader must fetch and decode whole.
pub(crate) const IMPORT_ROW_GROUP_BYTES: usize = 16 << 20;
/// Rows per `write` call. The byte cap can only close a row group
/// between writes, so the batch has to be a fraction of the budget
/// rather than equal to the row cap.
pub(crate) const IMPORT_WRITE_ROWS: usize = 8_192;
/// Geometry bytes per `write` call. Row counts do not bound a batch of
/// land-cover or cadastral polygons: a few thousand of them are hundreds
/// of megabytes, and the row group cannot close mid-batch.
pub(crate) const IMPORT_BATCH_BYTES: usize = 8 << 20;

/// Writer properties every importer uses: zstd (fast level — imports are
/// intermediates, Optimize writes the distribution-grade file), row
/// groups capped by rows *and* bytes, page statistics.
pub(crate) fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_max_row_group_row_count(Some(IMPORT_BATCH_ROWS))
        .set_max_row_group_bytes(Some(IMPORT_ROW_GROUP_BYTES))
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_created_by(format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")))
        .build()
}

/// The covering bbox column every import writes beside the geometry.
///
/// Without it a reader has no way to select features spatially: WKB
/// column statistics are meaningless, and 1.1 has no native geometry
/// statistics to fall back on. Four f64 leaves cost 32 bytes a row and
/// turn a file that must be decoded whole into one that can be read by
/// viewport.
pub(crate) fn bbox_field() -> Field {
    Field::new(
        "bbox",
        DataType::Struct(
            vec![
                Field::new("xmin", DataType::Float64, true),
                Field::new("ymin", DataType::Float64, true),
                Field::new("xmax", DataType::Float64, true),
                Field::new("ymax", DataType::Float64, true),
            ]
            .into(),
        ),
        true,
    )
}

/// Per-feature bbox accumulator, finished into the struct column.
#[derive(Default)]
pub(crate) struct BboxBuilder {
    xmin: Vec<Option<f64>>,
    ymin: Vec<Option<f64>>,
    xmax: Vec<Option<f64>>,
    ymax: Vec<Option<f64>>,
}

impl BboxBuilder {
    pub fn push(&mut self, env: Option<[f64; 4]>) {
        let e = env.filter(|e| e.iter().all(|v| v.is_finite()));
        self.xmin.push(e.map(|e| e[0]));
        self.ymin.push(e.map(|e| e[1]));
        self.xmax.push(e.map(|e| e[2]));
        self.ymax.push(e.map(|e| e[3]));
    }

    pub fn finish(&mut self) -> ArrayRef {
        let f = |v: &mut Vec<Option<f64>>| {
            Arc::new(arrow::array::Float64Array::from(std::mem::take(v))) as ArrayRef
        };
        let DataType::Struct(fields) = bbox_field().data_type().clone() else {
            unreachable!("bbox_field is a struct")
        };
        Arc::new(arrow::array::StructArray::new(
            fields,
            vec![f(&mut self.xmin), f(&mut self.ymin), f(&mut self.xmax), f(&mut self.ymax)],
            None,
        ))
    }
}

/// GeoParquet 1.1 `geo` metadata for an import. `crs`: `None` omits the
/// key entirely (CRS84 per spec — right for GeoJSON), `Some(Value::Null)`
/// records an explicitly unknown CRS, anything else is written verbatim.
pub(crate) fn geo_meta(primary: &str, stats: &GeomStats, crs: Option<Value>) -> Value {
    geo_meta_with_proj4(primary, stats, crs, None)
}

/// `geo_meta` with an optional vendor CRS: when the source CRS has no
/// EPSG identity but does parse to a proj4 string (ESRI .prj files),
/// the spec `crs` stays honest (`null` = unknown) and the proj4 string
/// rides in a `geopq:crs` extension key that this app's reader uses
/// for correct display. Other readers ignore it.
pub(crate) fn geo_meta_with_proj4(
    primary: &str,
    stats: &GeomStats,
    crs: Option<Value>,
    proj4: Option<(String, String)>,
) -> Value {
    let mut col = json!({
        "encoding": "WKB",
        "geometry_types": stats.types.iter().collect::<Vec<_>>(),
    });
    if let Some(c) = crs {
        col["crs"] = c;
    }
    if let Some((p4, name)) = proj4 {
        col["geopq:crs"] = json!({ "proj4": p4, "name": name });
    }
    let b = stats.bbox;
    if b[0].is_finite() && b[2].is_finite() {
        col["bbox"] = json!([b[0], b[1], b[2], b[3]]);
    }
    col["covering"] = json!({"bbox": {
        "xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
        "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"],
    }});
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

    /// Record a feature and return its envelope, which the covering
    /// column needs and `bounding_rect` has already computed.
    pub fn add(&mut self, g: &geo_types::Geometry<f64>) -> Option<[f64; 4]> {
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
        let r = g.bounding_rect()?;
        let e = [r.min().x, r.min().y, r.max().x, r.max().y];
        self.bbox[0] = self.bbox[0].min(e[0]);
        self.bbox[1] = self.bbox[1].min(e[1]);
        self.bbox[2] = self.bbox[2].max(e[2]);
        self.bbox[3] = self.bbox[3].max(e[3]);
        Some(e)
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
