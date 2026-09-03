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
    covering_field_named("bbox")
}

fn covering_field_named(name: &str) -> Field {
    Field::new(
        name,
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

/// The covering column's name and field, dodging the attribute names
/// already in `taken`.
///
/// A source attribute genuinely called `bbox` is data and must survive
/// the import: appending a second field of that name produced a file
/// whose two `bbox` columns no reader could tell apart, and whose
/// `covering` metadata pointed at whichever one the reader resolved
/// first. Optimize has always renamed around the collision; the
/// importers pushed `bbox` unconditionally, so this is the same rule in
/// one place. The chosen name has to reach `geo_meta`, which is why it
/// comes back beside the field rather than being re-derived there.
pub(crate) fn covering_field(taken: &[Field]) -> (String, Field) {
    let mut name = "bbox".to_string();
    let mut k = 0usize;
    while taken.iter().any(|f| f.name() == &name) {
        k += 1;
        name = format!("bbox_{k}");
    }
    let field = covering_field_named(&name);
    (name, field)
}

/// Per-feature bbox accumulator, finished into the struct column.
#[derive(Default)]
pub(crate) struct BboxBuilder {
    xmin: Vec<Option<f64>>,
    ymin: Vec<Option<f64>>,
    xmax: Vec<Option<f64>>,
    ymax: Vec<Option<f64>>,
}

/// A bbox only if every corner is finite.
///
/// NaN and infinity are legal in a double column and meaningless in a
/// covering one: `xmin` statistics that include a NaN make a row group
/// unprunable, and an infinite corner claims the whole plane, so one bad
/// vertex in a source file silently disables spatial pruning for
/// everything beside it. A null covering value says "no box", which is
/// what a reader knows how to handle.
pub(crate) fn finite_bbox(b: [f64; 4]) -> Option<[f64; 4]> {
    b.iter().all(|v| v.is_finite()).then_some(b)
}

impl BboxBuilder {
    pub fn push(&mut self, env: Option<[f64; 4]>) {
        let e = env.and_then(finite_bbox);
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
pub(crate) fn geo_meta(
    primary: &str,
    covering: &str,
    stats: &GeomStats,
    crs: Option<Value>,
) -> Value {
    geo_meta_with_proj4(primary, covering, stats, crs, None)
}

/// `geo_meta` with an optional vendor CRS: when the source CRS has no
/// EPSG identity but does parse to a proj4 string (ESRI .prj files),
/// the spec `crs` stays honest (`null` = unknown) and the proj4 string
/// rides in a `geopq:crs` extension key that this app's reader uses
/// for correct display. Other readers ignore it.
pub(crate) fn geo_meta_with_proj4(
    primary: &str,
    // The covering column's actual name: `bbox` unless an attribute of
    // that name kept it (see `covering_field`).
    covering: &str,
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
    col["covering"] = covering_json(covering);
    json!({
        "version": "1.1.0",
        "primary_column": primary,
        "columns": { primary: col },
    })
}

/// The `covering` block naming one column's four bbox leaves.
pub(crate) fn covering_json(covering: &str) -> Value {
    json!({"bbox": {
        "xmin": [covering, "xmin"], "ymin": [covering, "ymin"],
        "xmax": [covering, "xmax"], "ymax": [covering, "ymax"],
    }})
}

/// Geometry bookkeeping every importer needs: observed geometry types
/// (for the metadata and the GeoArrow recommendation downstream) and the
/// dataset bbox.
pub(crate) struct GeomStats {
    pub types: BTreeSet<String>,
    pub bbox: [f64; 4],
    /// Features whose geometry could not be read or encoded and were
    /// therefore written as null. A silent null is indistinguishable
    /// from a source row that genuinely had no geometry, so the count
    /// is what lets the import result say "5 of 200k skipped" instead
    /// of nothing at all.
    pub skipped: u64,
}

impl GeomStats {
    pub fn new() -> Self {
        Self {
            types: BTreeSet::new(),
            bbox: [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            skipped: 0,
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

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    /// Every row group of an import stays inside the byte cap the
    /// writer was given, plus the one batch the cap cannot cut inside.
    ///
    /// The quality scorecard's own ceiling (`quality::RG_BYTES_MAX`,
    /// 128 MB) is the number that matters to a reader, but asserting on
    /// it needs a fixture larger than a unit test should write; the
    /// importer's own 16 MB target catches the same bug an order of
    /// magnitude sooner, so both are checked.
    pub(crate) fn assert_row_groups_bounded(path: &Path) {
        use parquet::file::reader::FileReader;
        let f = std::fs::File::open(path).unwrap();
        let r = parquet::file::reader::SerializedFileReader::new(f).unwrap();
        let sizes: Vec<i64> =
            r.metadata().row_groups().iter().map(|g| g.total_byte_size()).collect();
        let max = sizes.iter().copied().max().unwrap_or(0) as usize;
        let ceiling = super::IMPORT_ROW_GROUP_BYTES + super::IMPORT_BATCH_BYTES;
        assert!(
            max <= ceiling,
            "largest row group {max} B over the {ceiling} B ceiling ({} groups: {sizes:?})",
            sizes.len()
        );
        assert!(
            max as u64 <= crate::data::quality::RG_BYTES_MAX,
            "largest row group {max} B over the scorecard's ceiling"
        );
    }
}
