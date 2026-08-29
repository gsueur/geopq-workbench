//! GeoParquet optimization export.
//!
//! Rewrites a GeoParquet file spatially sorted (Hilbert curve over feature
//! bbox centers) with tuned row groups, so that per-row-group bboxes become
//! compact and metadata-based pruning works. Output is one of:
//!
//! - **GeoParquet 1.1 (WKB)**: WKB column + `geo` metadata + `bbox` covering
//!   struct column (per-row-group min/max column statistics drive pruning).
//! - **GeoParquet 1.1 (GeoArrow)**: geometry as nested coordinate arrays
//!   (single geometry family; singles promote to their multi variant) — the
//!   x/y leaves carry ordinary parquet statistics, so any reader prunes
//!   without covering support.
//! - **GeoParquet 2.0**: parquet-native `GEOMETRY` logical type; the writer
//!   computes native geospatial statistics per column chunk (bbox + types).
//!
//! Geometry transcodes in any direction (WKB ↔ GeoArrow).
//!
//! Bloom filters found on source columns are reproduced on the output, or
//! can be added to all attribute columns.
//!
//! The rewrite is two passes over a re-openable source, so its peak memory
//! follows the row count rather than the file size. The key pass reads
//! geometry (plus whatever the partition keys need) and keeps a fixed-width
//! side table per row; the gather pass re-reads source row groups through a
//! byte-budgeted cache and writes the output in sorted order. A file whose
//! selected row groups fit the decode budget is simply the case where
//! nothing is ever evicted.

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StructArray, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::compute::{concat_batches, interleave_record_batch, take_record_batch};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use serde_json::{json, Value};

use super::geoarrow::{self, GaBuilder, GeomCol, GeomEncoding};
use super::source::Source;

/// Decoded source row groups the gather pass may hold at once. Hilbert
/// order correlates with source order on anything that was already loosely
/// spatial, so a handful of resident groups serve most gathers; an
/// adversarial order costs re-decodes, never unbounded memory.
const DECODE_BUDGET_BYTES: usize = 768 << 20;
/// Output writers open at the same time. Each holds its in-flight row group
/// (bounded by `row_group_bytes`), so this is what a partitioned export
/// pays on top of the decode budget; more partitions than this are written
/// in several sweeps of the sorted order.
const MAX_OPEN_WRITERS: usize = 32;
/// Ceiling on the key pass's side tables. They are the one thing left that
/// grows with the input, so this is where a rewrite that cannot fit says so
/// instead of being OS-killed halfway through.
const MAX_KEY_PASS_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const READ_BATCH: usize = 64 * 1024;
/// Hilbert curve order: 2^16 cells per axis.
const HILBERT_ORDER: u32 = 16;
/// Uncompressed page-size cap for covering bbox leaves: ~4096 f64 rows per
/// page, so the page index can prune well below row-group granularity.
const BBOX_LEAF_PAGE_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpVersion {
    /// WKB + `geo` metadata + covering bbox column.
    V1_1,
    /// GeoArrow native coordinate arrays (single geometry family; singles
    /// promote to their multi variant).
    V1_1GeoArrow,
    /// Parquet-native GEOMETRY logical type + geospatial statistics.
    V2_0,
}

impl GpVersion {
    pub fn label(&self) -> &'static str {
        match self {
            GpVersion::V1_1 => "GeoParquet 1.1 (WKB + covering bbox)",
            GpVersion::V1_1GeoArrow => "GeoParquet 1.1 (GeoArrow coordinate arrays)",
            GpVersion::V2_0 => "GeoParquet 2.0 (native GEOMETRY + geo stats)",
        }
    }

    /// Recommended output format for a source with these declared
    /// geometry types: GeoArrow whenever a single geometry family can
    /// hold them — the display-optimal habit this workbench wants to
    /// teach — and WKB + covering when the families are mixed, unknown
    /// or not GeoArrow-storable (maximum interoperability instead).
    pub fn preferred(geometry_types: &[String]) -> Self {
        let fits_geoarrow = !geometry_types.is_empty()
            && super::geoarrow::target_encoding(geometry_types.iter().map(String::as_str))
                .is_ok();
        if fits_geoarrow {
            GpVersion::V1_1GeoArrow
        } else {
            GpVersion::V1_1
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    Zstd,
    Snappy,
    Uncompressed,
}

impl Codec {
    pub fn label(&self) -> &'static str {
        match self {
            Codec::Zstd => "zstd",
            Codec::Snappy => "snappy",
            Codec::Uncompressed => "none",
        }
    }
    fn compression(&self) -> Compression {
        match self {
            // Level 15 per the GeoParquet distribution best practices:
            // decompression cost is flat across zstd levels, so readers
            // never pay for it and the write is one-time.
            Codec::Zstd => Compression::ZSTD(ZstdLevel::try_new(15).unwrap()),
            Codec::Snappy => Compression::SNAPPY,
            Codec::Uncompressed => Compression::UNCOMPRESSED,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BloomMode {
    /// Enable bloom filters on the columns that have them in the source.
    Preserve,
    /// Enable on every string/binary/integer attribute column.
    AllAttributes,
    /// Write no bloom filters.
    None,
}

impl BloomMode {
    pub fn label(&self) -> &'static str {
        match self {
            BloomMode::Preserve => "preserve source",
            BloomMode::AllAttributes => "all attribute columns",
            BloomMode::None => "none",
        }
    }
}

/// Web Mercator equatorial circumference (2π · 6_378_137 m), the constant
/// the COGP reference converter derives its zoom-based GSDs from.
const WEB_MERCATOR_CIRCUMFERENCE_M: f64 = 40_075_016.685_578_49;

/// Where a COGP export's level GSDs come from.
#[derive(Clone, Debug, PartialEq)]
pub enum GsdSource {
    /// A Web Mercator tile pyramid: `gsd(z) = circumference / (resolution ·
    /// 2^z)` metres for z in `minzoom..=maxzoom`. `resolution` is base units
    /// per tile side; the reference default of 1024 is ~4× a 256 px tile, so
    /// features that collapse within a few subpixels defer to a finer level.
    WebMercator {
        minzoom: u32,
        maxzoom: u32,
        resolution: u32,
    },
    /// Explicit GSDs in metres, coarse to fine (strictly decreasing).
    Explicit(Vec<f64>),
}

/// Which end of a rank column wins a point-thinning cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RankOrder {
    /// Largest value wins.
    Desc,
    /// Smallest value wins.
    Asc,
}

impl RankOrder {
    pub fn label(&self) -> &'static str {
        match self {
            RankOrder::Desc => "largest wins",
            RankOrder::Asc => "smallest wins",
        }
    }
}

/// Cloud Optimized GeoParquet Profile levels (see `super::cogp`).
///
/// The heuristic reproduces the reference converter's, so a file this app
/// writes behaves like one `cogp-rs convert` writes: a feature lands in the
/// coarsest level at which it is *independently renderable*. Lines and
/// polygons qualify by bbox diagonal against a multiple of the level's GSD;
/// points have no extent, so they qualify by grid thinning instead — one
/// point per cell survives per level and the rest defer to finer ones.
#[derive(Clone, Debug, PartialEq)]
pub struct CogpOptions {
    pub gsd: GsdSource,
    /// A line is renderable at a level once its bbox diagonal reaches
    /// `line_factor · gsd`. A diagonal of exactly one GSD is a hairline.
    pub line_factor: u32,
    /// Same for polygons, defaulting higher: a shape under ~4 cells across
    /// is not a shape yet, and letting them in crowds the coarse levels.
    pub polygon_factor: u32,
    /// Point-thinning grid pitch, in multiples of the level's GSD. Each
    /// step up thins by roughly its square.
    pub point_factor: u32,
    /// Attribute column deciding which point wins a contested cell, and
    /// which end of it wins. Without one the winner is the largest bbox,
    /// then a deterministic hash of the row — arbitrary but stable.
    pub rank: Option<(String, RankOrder)>,
}

impl Default for CogpOptions {
    fn default() -> Self {
        // The reference converter's defaults, verbatim.
        Self {
            gsd: GsdSource::WebMercator {
                minzoom: 0,
                maxzoom: 16,
                resolution: 1024,
            },
            line_factor: 2,
            polygon_factor: 4,
            point_factor: 4,
            rank: None,
        }
    }
}

impl CogpOptions {
    /// The level GSDs in metres, coarse to fine. Validated here rather than
    /// at write time: a bad zoom range or a non-decreasing explicit list is
    /// a dialog mistake, and it should not surface as a half-written file.
    pub fn gsds(&self) -> Result<Vec<f64>, String> {
        let list: Vec<f64> = match &self.gsd {
            GsdSource::WebMercator {
                minzoom,
                maxzoom,
                resolution,
            } => {
                if minzoom > maxzoom {
                    return Err(format!("COGP: min zoom {minzoom} is past max {maxzoom}"));
                }
                if *maxzoom > 30 {
                    return Err(format!("COGP: max zoom {maxzoom} is past 30"));
                }
                if *resolution == 0 {
                    return Err("COGP: resolution must be positive".into());
                }
                let z0 = WEB_MERCATOR_CIRCUMFERENCE_M / *resolution as f64;
                (*minzoom..=*maxzoom)
                    .map(|z| z0 / (1u64 << z) as f64)
                    .collect()
            }
            GsdSource::Explicit(v) => v.clone(),
        };
        if list.is_empty() {
            return Err("COGP: no levels".into());
        }
        // One level per byte of the per-row level code, and long before 255
        // the levels have stopped meaning anything.
        if list.len() > 255 {
            return Err(format!("COGP: {} levels is past the 255 limit", list.len()));
        }
        for (i, g) in list.iter().enumerate() {
            if !(g.is_finite() && *g > 0.0) {
                return Err(format!("COGP: level {i} has a non-positive gsd {g}"));
            }
            if i > 0 && *g >= list[i - 1] {
                return Err(format!(
                    "COGP: gsd must strictly decrease, got {} then {g}",
                    list[i - 1]
                ));
            }
        }
        if self.line_factor == 0 || self.polygon_factor == 0 || self.point_factor == 0 {
            return Err("COGP: the visibility factors must be at least 1".into());
        }
        Ok(list)
    }
}

/// Geometry family of one feature, for the COGP visibility rules. One byte
/// per exported row, which is why it is not `geometry::GeomKind` (that one
/// carries Mixed/Unknown, which have no rule here).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CogpKind {
    Point,
    Line,
    Polygon,
}

impl CogpKind {
    /// A geometry type name from `geom_type_name`. A GeometryCollection is
    /// spatially extended and has no rule of its own, so it takes the
    /// strictest one rather than being let into level 0 as a "point".
    fn from_type_name(name: &str) -> Self {
        match name {
            "Point" | "MultiPoint" => CogpKind::Point,
            "LineString" | "MultiLineString" => CogpKind::Line,
            _ => CogpKind::Polygon,
        }
    }
}

/// Metres per degree of longitude at the equator, and per degree of
/// latitude. Rendering-grade constants: COGP thresholds decide which level
/// a feature is *drawn* at, so a percent of sphere-vs-ellipsoid error moves
/// nothing a viewer can see.
const M_PER_DEG_LON: f64 = 111_320.0;
const M_PER_DEG_LAT: f64 = 110_540.0;

/// How the layer's coordinates convert to the metres COGP measures in.
///
/// The spec is explicit (§4.2) that `gsd` is metres on the ground whatever
/// the file's CRS, so degrees have to be converted rather than compared
/// directly — a threshold of "1000" against degrees would defer every
/// feature on Earth to the finest level.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CogpUnits {
    /// Geographic degrees. Metres per degree of longitude shrink with the
    /// cosine of latitude, so the conversion is per feature rather than the
    /// flat scale factor the reference converter uses.
    Degrees,
    /// Projected linear units, this many metres each (1.0 for metres).
    Linear(f64),
}

impl CogpUnits {
    fn label(&self) -> String {
        match self {
            CogpUnits::Degrees => "degrees (converted at each feature's latitude)".into(),
            CogpUnits::Linear(f) if *f == 1.0 => "metres".into(),
            CogpUnits::Linear(f) => format!("projected units of {f} m"),
        }
    }

    /// A bbox's width and height in metres.
    fn extent_m(self, b: &[f64; 4]) -> (f64, f64) {
        match self {
            CogpUnits::Linear(f) => ((b[2] - b[0]) * f, (b[3] - b[1]) * f),
            CogpUnits::Degrees => {
                let lat = ((b[1] + b[3]) * 0.5).to_radians();
                (
                    (b[2] - b[0]) * M_PER_DEG_LON * lat.cos().abs(),
                    (b[3] - b[1]) * M_PER_DEG_LAT,
                )
            }
        }
    }

    /// The thinning-grid cell of a bbox centre, at `pitch` metres.
    ///
    /// Degrees go through a sinusoidal projection (x = R·λ·cos φ, y = R·φ)
    /// rather than a per-feature scale factor: the projection is continuous,
    /// so two neighbouring points always agree on where the cell boundary
    /// between them is. Scaling each feature by the cosine of *its own*
    /// latitude does not, and would let a cluster spanning a boundary keep
    /// two winners at the same level.
    fn cell(self, b: &[f64; 4], pitch: f64) -> (i64, i64) {
        let (cx, cy) = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5);
        let (mx, my) = match self {
            CogpUnits::Linear(f) => (cx * f, cy * f),
            CogpUnits::Degrees => (
                cx * M_PER_DEG_LON * cy.to_radians().cos(),
                cy * M_PER_DEG_LAT,
            ),
        };
        ((mx / pitch).floor() as i64, (my / pitch).floor() as i64)
    }
}

/// Metres per linear unit of a proj string (`+to_meter`, or the unit names
/// proj4 abbreviates). Unknown units read as metres, which is what proj
/// itself defaults to.
fn proj4_meters_per_unit(proj4: &str) -> f64 {
    for tok in proj4.split_whitespace() {
        if let Some(v) = tok.strip_prefix("+to_meter=")
            && let Ok(f) = v.parse::<f64>()
            && f.is_finite()
            && f > 0.0
        {
            return f;
        }
        match tok {
            "+units=us-ft" => return 0.304_800_609_601_219_2,
            "+units=ft" => return 0.3048,
            "+units=km" => return 1000.0,
            _ => {}
        }
    }
    1.0
}

/// What the exported coordinates measure in, best evidence first.
///
/// A CRS the app can actually build answers directly (`is_latlong` plus the
/// proj string's unit); otherwise the PROJJSON `type` is enough to tell a
/// projected CRS from a geographic one, which is the distinction that
/// matters. GeoParquet's default when `crs` is absent is OGC:CRS84, so an
/// absent CRS means degrees, not metres.
fn cogp_units(crs: Option<&Value>, vendor_crs: Option<&Value>) -> CogpUnits {
    let from_crs = |c: &super::crs::Crs| {
        if c.is_latlong {
            CogpUnits::Degrees
        } else {
            CogpUnits::Linear(proj4_meters_per_unit(&c.proj4))
        }
    };
    if let Some(p4) = vendor_crs.and_then(|v| v.get("proj4")?.as_str())
        && let Ok(c) = super::crs::Crs::from_proj4(p4, None, "vendor")
    {
        return from_crs(&c);
    }
    let Some(v) = crs else {
        return CogpUnits::Degrees; // absent crs means CRS84
    };
    if v.is_null() {
        // Declared-unknown. The data still has to be placed somewhere and
        // the app renders it as CRS84, so measure it the same way.
        return CogpUnits::Degrees;
    }
    if let Ok(c) = super::crs::Crs::from_geoparquet_crs(Some(v)) {
        return from_crs(&c);
    }
    // Last resort: the PROJJSON type alone, the way the reference converter
    // decides it.
    fn classify(v: &Value) -> Option<CogpUnits> {
        let t = v.get("type")?.as_str()?;
        if t.contains("Projected") {
            return Some(CogpUnits::Linear(1.0));
        }
        if t.contains("Geographic") {
            return Some(CogpUnits::Degrees);
        }
        if t == "BoundCRS" {
            return v.get("source_crs").and_then(classify);
        }
        None
    }
    classify(v).unwrap_or(CogpUnits::Degrees)
}

/// Winner priority inside a point-thinning cell: the rank column leads, a
/// larger bbox breaks ties (a MultiPoint spanning ground beats a single
/// point), then a hash of the row index so the choice is deterministic and
/// not simply "whichever came first in the file".
fn cogp_priority(b: &[f64; 4], rank: u32, row: u32) -> (u32, u64, u64) {
    let (w, h) = ((b[2] - b[0]).max(0.0), (b[3] - b[1]).max(0.0));
    let sq = w * w + h * h;
    let sq_bits = if sq.is_finite() { sq.to_bits() } else { 0 };
    let mut hash = row as u64;
    hash = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    hash ^= hash >> 30;
    (rank, sq_bits, hash)
}

/// Assign every exported row to a COGP level, coarse (0) to fine.
///
/// Lines and polygons enter at the coarsest level whose GSD their bbox
/// diagonal clears — a hard gate, so a reader that stops after the coarse
/// prefix never pays for sub-resolution features. Points have no extent and
/// so cannot be gated that way; they compete for grid cells instead, one
/// winner per cell per level, with the cells a coarser level already used
/// blocked so a tight cluster does not surface a near-identical neighbour
/// at every level. Whatever is left over lands in the finest level, which
/// is what makes the assignment total: every feature is placed exactly once.
fn assign_cogp_levels(
    bboxes: &[Option<[f64; 4]>],
    kinds: &[CogpKind],
    gsds: &[f64],
    units: CogpUnits,
    opts: &CogpOptions,
    ranks: Option<&[u32]>,
) -> Vec<u8> {
    use std::collections::HashMap;
    let n = bboxes.len();
    let last = (gsds.len() - 1) as u8;
    let rank_of = |i: usize| ranks.map_or(0, |r| r[i]);

    // Coarsest level each feature could possibly enter at. Points: 0.
    // Null geometry renders nothing, so it defers all the way.
    let mut floor: Vec<u8> = vec![0; n];
    for i in 0..n {
        let Some(b) = bboxes[i] else {
            floor[i] = last;
            continue;
        };
        if kinds[i] == CogpKind::Point {
            continue;
        }
        let (dx, dy) = units.extent_m(&b);
        let sq = dx * dx + dy * dy;
        if sq <= 0.0 {
            // A zero-extent line or polygon is degenerate, not small: it has
            // no diagonal to gate on, so it is treated as point-like. Same
            // call the reference converter makes.
            continue;
        }
        let factor = if kinds[i] == CogpKind::Line {
            opts.line_factor
        } else {
            opts.polygon_factor
        } as f64;
        // Squared throughout: a per-row sqrt buys nothing here.
        floor[i] = gsds
            .iter()
            .position(|g| {
                let t = g * factor;
                sq >= t * t
            })
            .unwrap_or(gsds.len() - 1) as u8;
    }

    let mut level: Vec<u8> = vec![u8::MAX; n];
    let mut remaining: Vec<u32> = (0..n as u32)
        .filter(|&i| bboxes[i as usize].is_some())
        .collect();
    for i in 0..n {
        if bboxes[i].is_none() {
            level[i] = last;
        }
    }
    // Points already placed, re-projected onto each new level's grid to
    // block its cells.
    let mut placed_points: Vec<u32> = Vec::new();
    for (li, gsd) in gsds.iter().enumerate() {
        if remaining.is_empty() {
            break;
        }
        let pitch = gsd * opts.point_factor as f64;
        let blocked: HashSet<(i64, i64)> = placed_points
            .iter()
            .map(|&r| units.cell(&bboxes[r as usize].unwrap(), pitch))
            .collect();
        let mut best: HashMap<(i64, i64), u32> = HashMap::new();
        let mut picked: Vec<u32> = Vec::new();
        for &r in &remaining {
            let i = r as usize;
            if floor[i] as usize > li {
                continue;
            }
            if kinds[i] != CogpKind::Point {
                // Extended geometry contributes across many cells; deciding
                // it by its centre would hide a long river behind a pond.
                picked.push(r);
                continue;
            }
            let b = bboxes[i].unwrap();
            let key = units.cell(&b, pitch);
            if blocked.contains(&key) {
                continue;
            }
            let prio = cogp_priority(&b, rank_of(i), r);
            match best.get(&key) {
                Some(&cur) => {
                    let c = bboxes[cur as usize].unwrap();
                    if prio > cogp_priority(&c, rank_of(cur as usize), cur) {
                        best.insert(key, r);
                    }
                }
                None => {
                    best.insert(key, r);
                }
            }
        }
        let winners: Vec<u32> = best.into_values().collect();
        placed_points.extend(winners.iter().copied());
        picked.extend(winners);
        if picked.is_empty() {
            continue;
        }
        for &r in &picked {
            level[r as usize] = li as u8;
        }
        let taken: HashSet<u32> = picked.into_iter().collect();
        remaining.retain(|r| !taken.contains(r));
    }
    for r in remaining {
        level[r as usize] = last;
    }
    level
}

#[derive(Clone)]
pub struct OptimizeOptions {
    pub version: GpVersion,
    /// Row cap per row group.
    pub row_group_size: usize,
    /// Byte cap per row group, whichever limit is reached first. Rows
    /// alone size a row group only when features are small: a few
    /// hundred administrative boundaries carry more bytes than a million
    /// points, and without this they land in one group that has to be
    /// fetched and decoded whole.
    ///
    /// Measured on the *encoded* (compressed) estimate, which is what a
    /// reader actually downloads for the group.
    pub row_group_bytes: usize,
    pub codec: Codec,
    /// Sort features along a Hilbert curve over bbox centers.
    pub hilbert_sort: bool,
    /// Write a `bbox` covering struct column (always useful for 1.1
    /// readers; redundant but allowed alongside 2.0 native stats).
    pub covering: bool,
    /// 2.0 flavor: also write a `{primary}_geoarrow` coordinate-array
    /// column next to the native GEOMETRY primary. The file stays
    /// conformant 2.0 (the sibling is a plain data column, not declared
    /// in `geo`); GeoArrow-aware readers decode it directly. Geometry is
    /// stored twice. Needs a single geometry family; ignored for 1.1.
    pub geoarrow_aux: bool,
    pub bloom: BloomMode,
    /// Export only features whose bbox intersects this rect (data CRS).
    pub filter_rect: Option<[f64; 4]>,
    /// Add an `h3_r{n}` UInt64 cell column (centroid-based).
    pub h3_resolution: Option<u8>,
    /// Split the output into hive directories / adaptive H3 cells.
    pub partition: super::partition::PartitionBy,
    /// Order the output coarse to fine and write the `cogp` level metadata
    /// (Cloud Optimized GeoParquet Profile). Implies the Hilbert sort and,
    /// on 1.1, the covering column; cannot combine with partitioning
    /// (levels and hive parts both want to own the file layout) or with the
    /// GeoArrow flavour.
    pub cogp: Option<CogpOptions>,
    /// Source has no geometry column: synthesize WKB points from these
    /// coordinate columns (x/lon, y/lat) — materializes x/y layers into
    /// real GeoParquet.
    pub xy_geom: Option<(usize, usize)>,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            version: GpVersion::V1_1,
            row_group_size: 65_536,
            // In the same range as 65k rows of ordinary features, so
            // light data keeps its current layout and only heavy
            // features are split.
            row_group_bytes: 16 << 20,
            codec: Codec::Zstd,
            hilbert_sort: true,
            covering: true,
            geoarrow_aux: false,
            bloom: BloomMode::Preserve,
            filter_rect: None,
            h3_resolution: None,
            partition: super::partition::PartitionBy::None,
            cogp: None,
            xy_geom: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptimizeReport {
    pub rows: u64,
    pub size_before: u64,
    pub size_after: u64,
    pub rg_before: usize,
    pub rg_after: usize,
    /// Avg row-group bbox overlap (see `bbox_overlap_metric`).
    pub overlap_before: f64,
    pub overlap_after: f64,
    pub bloom_columns: Vec<String>,
    pub version_label: String,
    pub elapsed_ms: u64,
    /// Output files written (1 unless partitioned).
    pub files: usize,
    /// COGP levels actually written, coarse to fine (empty when off).
    pub cogp_levels: Vec<CogpLevelReport>,
}

/// One written COGP level, for the completion summary.
#[derive(Clone, Debug)]
pub struct CogpLevelReport {
    pub gsd: f64,
    pub rows: u64,
    /// Row groups this level owns, inclusive.
    pub rg_start: usize,
    pub rg_end: usize,
}

impl OptimizeReport {
    /// Overlap as a fraction of the possible overlaps (comparable across
    /// different row-group counts, unlike the raw metric).
    pub fn overlap_frac_before(&self) -> f64 {
        self.overlap_before / (self.rg_before.max(2) - 1) as f64
    }
    pub fn overlap_frac_after(&self) -> f64 {
        self.overlap_after / (self.rg_after.max(2) - 1) as f64
    }
}

/// Bytes of fixed-width side tables the key pass keeps per exported row:
/// the feature bbox (40), its source row and sort position (8), the Hilbert
/// key (8), and whatever the chosen options derive on top — lon/lat
/// centroids, H3 cells, joined admin values, partition codes and plan.
/// None of it scales with feature size, which is the point: the 314k
/// million-vertex polygons of issue #12 cost the same here as 314k points.
///
/// Every line is what the code actually allocates *while the others are
/// still live*, not the steady state of one table at a time — an estimate
/// that bills the peak too low is worse than no estimate, because it
/// promises a run the machine cannot finish.
fn key_pass_bytes_per_row(
    opts: &OptimizeOptions,
    admin_join: bool,
    part_fields: &[String],
) -> u64 {
    use super::partition::PartitionBy;
    let mut n = 56;
    if opts.h3_resolution.is_some() || matches!(opts.partition, PartitionBy::AdaptiveH3 { .. }) {
        n += 24; // lon/lat centroids
    }
    if opts.h3_resolution.is_some() {
        n += 16; // cell per row
    }
    if admin_join {
        // Dictionary codes (4) and the centroids they are looked up with
        // (24). The joined strings themselves belong to the boundary
        // layer, not to the exported rows.
        n += 28;
    }
    match &opts.partition {
        PartitionBy::None => {}
        PartitionBy::Fields(_) => {
            // Key codes per field, the plan's index lists, and the routing
            // table, all live together while the sweeps run.
            n += 8 + 4 * part_fields.len() as u64;
        }
        PartitionBy::AdaptiveH3 { .. } => {
            // `split_adaptive_h3` holds a fine cell per row (8) and
            // re-buckets rows between its work and done lists (8) while
            // the sort order, the plan it returns and the routing table
            // (12) are all live.
            n += 28;
        }
    }
    if let Some(c) = &opts.cogp {
        // Geometry family (1), the visibility floor (1) and the assigned
        // level (1), plus the `remaining` and `placed_points` lists (8).
        // Dominating all of it: the per-level winner map and the blocked
        // set it is rebuilt from — a hashbrown entry for a 16-byte cell key
        // and a 4-byte row, at the load factor, and in the worst case
        // (every point alone in its cell at level 0) there is one per row.
        n += 3 + 8 + 48;
        if c.rank.is_some() {
            // The u32 rank, and slack for the ranked column's own values
            // while they are concatenated. A wide string column costs more
            // than this; it is read once and dropped before the sort.
            n += 8;
        }
    }
    n
}

/// The row groups of a selection worth reading. An empty group is legal
/// parquet and carries nothing; keeping one would put duplicate offsets in
/// the gatherer's row map, where a binary search has no defined answer,
/// and would hand the key pass a group that decodes to no batch at all.
fn readable_groups(groups: Vec<usize>, rg_rows: &[usize]) -> Vec<usize> {
    groups.into_iter().filter(|&g| rg_rows[g] > 0).collect()
}

/// What the key pass needs out of every decoded batch.
struct ScanPlan<'a> {
    map: &'a ColMap,
    geom_idx: usize,
    xy: Option<(usize, usize)>,
    encoding: GeomEncoding,
    filter: Option<[f64; 4]>,
    /// (slot in `part_names`, source column) for keys read from the data.
    part_src: &'a [(usize, usize)],
    part_names: &'a [String],
    /// COGP: record each feature's geometry family, and keep the values of
    /// this source column for the cell-winner rank.
    cogp_kinds: bool,
    cogp_rank_src: Option<usize>,
}

/// What it accumulates. Everything here is per exported row and fixed
/// width — the partition values are dictionary codes rather than a
/// formatted `String` per row, which is what a partitioned export of a
/// large layer used to spend most of its memory on.
#[derive(Default)]
struct ScanOut {
    row_bboxes: Vec<Option<[f64; 4]>>,
    /// Source row of each exported row, in this gatherer's addressing.
    kept: Vec<u32>,
    geom_types: HashSet<String>,
    interners: Vec<super::partition::Interner>,
    /// COGP: geometry family per exported row (one byte each).
    row_kinds: Vec<CogpKind>,
    /// COGP: the rank column's exported values, concatenated and ranked
    /// once the scan is over. Dropped before the sort order is built.
    rank_parts: Vec<ArrayRef>,
    /// Union over the row group being scanned (reset by the caller).
    group_box: Option<[f64; 4]>,
    scanned: u64,
}

fn scan_batch(batch: &RecordBatch, plan: &ScanPlan, out: &mut ScanOut) -> Result<(), String> {
    let geom = chunk_geometry(batch, plan.map, plan.geom_idx, plan.xy)?;
    let mut boxes: Vec<Option<[f64; 4]>> = Vec::with_capacity(batch.num_rows());
    let mut kinds: Vec<CogpKind> = Vec::new();
    scan_bboxes(
        &geom,
        plan.encoding,
        &mut boxes,
        &mut out.geom_types,
        plan.cogp_kinds.then_some(&mut kinds),
    )?;
    out.group_box = union_bboxes(out.group_box.iter().chain(boxes.iter().flatten()));
    // Viewport exports drop rows here rather than after the read, so
    // nothing downstream carries a row that will not be written.
    let keep: Vec<usize> = match plan.filter {
        None => (0..boxes.len()).collect(),
        Some(r) => (0..boxes.len())
            .filter(|&i| {
                boxes[i].is_some_and(|b| {
                    b[0] <= r[2] && b[2] >= r[0] && b[1] <= r[3] && b[3] >= r[1]
                })
            })
            .collect(),
    };
    out.kept
        .extend(keep.iter().map(|&i| (out.scanned + i as u64) as u32));
    out.row_bboxes.extend(keep.iter().map(|&i| boxes[i]));
    if plan.cogp_kinds {
        out.row_kinds.extend(keep.iter().map(|&i| kinds[i]));
    }
    if let Some(col_idx) = plan.cogp_rank_src {
        // Kept rows only, so the ranks line up with `row_bboxes` without a
        // second mapping — and a viewport export does not rank rows it
        // will not write.
        let col = batch.column(plan.map.pos(col_idx));
        let idx = UInt32Array::from_iter_values(keep.iter().map(|&i| i as u32));
        out.rank_parts.push(
            arrow::compute::take(col.as_ref(), &idx, None)
                .map_err(|e| format!("COGP rank column: {e}"))?,
        );
    }

    use arrow::util::display::{ArrayFormatter, FormatOptions};
    let fopts = FormatOptions::default().with_display_error(true);
    for (slot, &(field, col_idx)) in plan.part_src.iter().enumerate() {
        let name = &plan.part_names[field];
        let col = batch.column(plan.map.pos(col_idx));
        let f = ArrayFormatter::try_new(col.as_ref(), &fopts)
            .map_err(|e| format!("partition field '{name}': {e}"))?;
        for &i in &keep {
            let v = (!col.is_null(i)).then(|| f.value(i).to_string());
            out.interners[slot].push(v.as_deref())?;
        }
    }
    out.scanned += batch.num_rows() as u64;
    Ok(())
}

// Test knobs. A tiny budget forces the gather pass to evict and re-decode;
// the stats let a test assert the cache honoured it, which is the only
// observable a cache bug has short of an OOM report.
#[cfg(test)]
thread_local! {
    static DECODE_BUDGET: std::cell::Cell<usize> =
        const { std::cell::Cell::new(DECODE_BUDGET_BYTES) };
    static GATHER_STATS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
fn decode_budget() -> usize {
    DECODE_BUDGET.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn decode_budget() -> usize {
    DECODE_BUDGET_BYTES
}

#[cfg(test)]
fn record_gather_stats(cache: &GroupCache) {
    GATHER_STATS.with(|c| c.set((cache.live_high_water, cache.decodes)));
}

#[cfg(not(test))]
#[inline]
fn record_gather_stats(_cache: &GroupCache) {}

/// One partition's output while the sweep feeding it is in flight. The
/// `StagedFile` behind it deliberately lives elsewhere and outlives this:
/// the writer must close before anything removes or renames the file it
/// holds open, and on Windows the reverse order fails outright.
struct PartWriter {
    writer: ArrowWriter<File>,
    part: usize,
}

/// Suffix of the sibling every output is written through.
const PARTIAL_SUFFIX: &str = ".partial";

/// `<path>.partial`, the sibling an output is built as.
fn partial_path(dst: &Path) -> Result<PathBuf, String> {
    let mut name = dst
        .file_name()
        .ok_or_else(|| format!("{} is not a file path", dst.display()))?
        .to_os_string();
    name.push(PARTIAL_SUFFIX);
    Ok(dst.with_file_name(name))
}

/// Everything one export writes, published by a single rename.
///
/// `File::create` on the target truncates it the moment writing starts, so
/// an interrupted rewrite used to replace the user's file with a plausible
/// prefix of a parquet — unreadable, undetectable by size, silently wrong.
/// Everything is therefore built as a `<dst>.partial` sibling and renamed
/// into place at the end.
///
/// A partitioned dataset is one artifact and gets one rename too: the whole
/// tree is built inside `<dst>.partial/` and the *directory* is what moves.
/// Renaming forty part files one by one has a middle, and a rename failing
/// at the seventeenth would publish a dataset that reads perfectly and is
/// missing rows — the failure this staging exists to prevent, arrived at by
/// a different road.
struct StagedOutputs {
    /// `<dst>.partial`: the file itself, or the directory the parts are
    /// built inside.
    staging: PathBuf,
    dst: PathBuf,
    partitioned: bool,
    /// What has been written, for the flush that precedes the rename.
    files: Vec<PathBuf>,
    committed: bool,
}

impl StagedOutputs {
    fn new(dst: &Path, partitioned: bool) -> Result<Self, String> {
        let staging = partial_path(dst)?;
        if partitioned {
            // A staging tree left by a dead process is not this run's work
            // and must never be published as if it were.
            let _ = std::fs::remove_dir_all(&staging);
            std::fs::create_dir_all(&staging)
                .map_err(|e| format!("cannot create {}: {e}", staging.display()))?;
        }
        Ok(Self {
            staging,
            dst: dst.to_path_buf(),
            partitioned,
            files: Vec::new(),
            committed: false,
        })
    }

    /// The file for one output part. `rel` is its hive path inside a
    /// partitioned dataset, and is ignored for a single-file output.
    fn create(&mut self, rel: &str) -> Result<File, String> {
        let path = if self.partitioned {
            let dir = self.staging.join(rel);
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
            dir.join("part-0.parquet")
        } else {
            self.staging.clone()
        };
        let file = File::create(&path).map_err(|e| format!("cannot create output: {e}"))?;
        self.files.push(path);
        Ok(file)
    }

    /// Flush everything to the device, then publish. Returns the paths the
    /// output now lives at.
    fn commit_all(&mut self) -> Result<Vec<PathBuf>, String> {
        // A rename is metadata. Publishing a name that points at unsynced
        // bytes is a worse failure than the truncation this replaced, so
        // all the real I/O happens here, before anything is published.
        for p in &self.files {
            std::fs::OpenOptions::new()
                .write(true)
                .open(p)
                .and_then(|f| f.sync_all())
                .map_err(|e| format!("cannot flush {}: {e}", p.display()))?;
        }
        for dir in self.files.iter().filter_map(|p| p.parent()).collect::<HashSet<_>>() {
            if let Ok(d) = File::open(dir) {
                let _ = d.sync_all();
            }
        }

        // --- publishing starts here: renames only, and as few as one ---
        fail_commit(0)?;
        if !self.partitioned {
            std::fs::rename(&self.staging, &self.dst)
                .map_err(|e| format!("cannot finalize {}: {e}", self.dst.display()))?;
            self.committed = true;
            return Ok(vec![self.dst.clone()]);
        }
        // A directory rename cannot replace a directory, so an existing
        // dataset steps aside first — removed once the new one is in
        // place, put back if it never gets there.
        let aside = self.dst.with_file_name(format!(
            "{}.replaced-{}",
            self.dst.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        ));
        let displaced = self.dst.exists();
        if displaced {
            std::fs::rename(&self.dst, &aside)
                .map_err(|e| format!("cannot replace {}: {e}", self.dst.display()))?;
        }
        if let Err(e) = std::fs::rename(&self.staging, &self.dst) {
            if displaced && std::fs::rename(&aside, &self.dst).is_err() {
                // The forward rename and the restore failed the same way;
                // the previous dataset is intact, but the error must say
                // where, or the user cannot find their own data.
                return Err(format!(
                    "cannot finalize {}: {e}\n(the previous dataset is at {})",
                    self.dst.display(),
                    aside.display()
                ));
            }
            return Err(format!("cannot finalize {}: {e}", self.dst.display()));
        }
        self.committed = true;
        if displaced {
            let _ = std::fs::remove_dir_all(&aside);
        }
        let staging = self.staging.clone();
        Ok(self
            .files
            .iter()
            .filter_map(|p| Some(self.dst.join(p.strip_prefix(&staging).ok()?)))
            .collect())
    }
}

impl Drop for StagedOutputs {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // The staging name never touches the target, so removing it takes
        // the whole failed attempt — part files and hive skeleton alike.
        if self.partitioned {
            let _ = std::fs::remove_dir_all(&self.staging);
        } else {
            let _ = std::fs::remove_file(&self.staging);
        }
    }
}

/// Fail before the passes rather than after them if the target cannot be
/// written. `File::create` used to prove this in the first millisecond by
/// truncating the target; staging removed that, and on Windows a rename
/// cannot replace a file another process holds open — a check the old code
/// got for free and this one has to make. Publishing is a create and a
/// rename in the target's own directory, so that is what gets probed,
/// whether the output is one file or a whole dataset.
fn probe_output(dst: &Path, partitioned: bool) -> Result<(), String> {
    if let Some(parent) = dst.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    if !partitioned && dst.exists() {
        std::fs::OpenOptions::new()
            .write(true)
            .open(dst)
            .map_err(|e| format!("cannot replace {}: {e}", dst.display()))?;
    }
    let probe = partial_path(dst)?.with_extension(format!("probe-{}", std::process::id()));
    let moved = probe.with_extension(format!("probe-{}-moved", std::process::id()));
    File::create(&probe).map_err(|e| format!("cannot write next to {}: {e}", dst.display()))?;
    let renamed = std::fs::rename(&probe, &moved);
    let _ = std::fs::remove_file(&probe);
    let _ = std::fs::remove_file(&moved);
    renamed.map_err(|e| format!("cannot publish into {}: {e}", dst.display()))
}

// Fail the write loop once this many rows are out, so a test can prove
// what an interrupted export leaves behind. Nothing the pipeline reaches
// on its own errors this late, and the crash the users hit (an OOM kill)
// is not reproducible in-process at all.
#[cfg(test)]
thread_local! {
    static FAIL_AFTER_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    /// Fail at this publishing step. There is exactly one — which is the
    /// property under test, so a test that sets this past the first step
    /// asserts that the export either published everything or nothing.
    static FAIL_AT_COMMIT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

#[cfg(test)]
fn fail_commit(step: usize) -> Result<(), String> {
    if step == FAIL_AT_COMMIT.with(std::cell::Cell::get) {
        return Err("injected failure: commit interrupted".into());
    }
    Ok(())
}

#[cfg(not(test))]
#[inline]
fn fail_commit(_step: usize) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
fn fail_point(written: usize) -> Result<(), String> {
    if written > FAIL_AFTER_ROWS.with(std::cell::Cell::get) {
        return Err("injected failure: write interrupted".into());
    }
    Ok(())
}

#[cfg(not(test))]
#[inline]
fn fail_point(_written: usize) -> Result<(), String> {
    Ok(())
}

/// Rewrite `src` into `dst` per `opts`. `epsg_hint`: CRS to record when the
/// source has no usable CRS metadata (e.g. the already-loaded layer's CRS).
/// `progress(frac, stage)` is called from the worker thread.
/// A WKB point column synthesized from two coordinate columns (null when
/// either coordinate is null).
fn xy_wkb_array(x: &ArrayRef, y: &ArrayRef) -> Result<ArrayRef, String> {
    use arrow::array::BinaryBuilder;
    let as_f64 = |a: &ArrayRef| -> Result<Float64Array, String> {
        arrow::compute::cast(a, &DataType::Float64)
            .map_err(|e| format!("coordinate cast: {e}"))
            .map(|a| a.as_any().downcast_ref::<Float64Array>().unwrap().clone())
    };
    let (xs, ys) = (as_f64(x)?, as_f64(y)?);
    let n = xs.len();
    let mut b = BinaryBuilder::with_capacity(n, n * 21);
    for i in 0..n {
        if xs.is_null(i) || ys.is_null(i) {
            b.append_null();
            continue;
        }
        let mut wkb = [0u8; 21];
        wkb[0] = 1; // little endian
        wkb[1..5].copy_from_slice(&1u32.to_le_bytes());
        wkb[5..13].copy_from_slice(&xs.value(i).to_le_bytes());
        wkb[13..21].copy_from_slice(&ys.value(i).to_le_bytes());
        b.append_value(wkb);
    }
    Ok(Arc::new(b.finish()))
}

/// Source arrow field index → column position in a projected batch.
struct ColMap(Vec<usize>);

impl ColMap {
    fn pos(&self, src_col: usize) -> usize {
        self.0
            .binary_search(&src_col)
            .expect("column is part of the projection")
    }
    fn mask(&self, meta: &ParquetMetaData) -> ProjectionMask {
        ProjectionMask::roots(meta.file_metadata().schema_descr(), self.0.iter().copied())
    }
}

/// The chunk's geometry in the source encoding: the projected column, or a
/// WKB point column synthesized from the x/y pair for an x/y source.
fn chunk_geometry(
    batch: &RecordBatch,
    map: &ColMap,
    geom_idx: usize,
    xy: Option<(usize, usize)>,
) -> Result<ArrayRef, String> {
    match xy {
        Some((xi, yi)) => {
            xy_wkb_array(batch.column(map.pos(xi)), batch.column(map.pos(yi)))
        }
        None => Ok(batch.column(map.pos(geom_idx)).clone()),
    }
}

/// Decoded source row groups, evicted least-recently-used against a byte
/// budget. The budget is in bytes rather than groups because a row group is
/// a few kB of points or a gigabyte of boundaries and only the second one
/// can take a machine down.
///
/// Everything a gather can reach is charged here, the groups the chunk in
/// flight holds included — those are *pinned* for the length of the gather
/// and never evicted under it. A cache that only bounded what survives
/// *between* chunks would let one chunk touching sixty groups hold all
/// sixty at once while reporting itself well inside its budget, which is
/// the exact shape of the crash this rewrite exists to remove.
struct GroupCache {
    budget: usize,
    bytes: usize,
    /// (selected group, rows, size, last touch).
    slots: Vec<(usize, Arc<RecordBatch>, usize, u64)>,
    clock: u64,
    /// Groups the gather in flight has claimed; never evicted under it.
    pinned: Vec<usize>,
    /// Peak decoded source bytes actually reachable at once, settled at the
    /// end of every gather from what the residents held plus what the cache
    /// kept behind them. The cache's own total is not that number — it
    /// cannot exceed the budget by construction, so asserting on it proves
    /// nothing.
    live_high_water: usize,
    decodes: usize,
}

impl GroupCache {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            slots: Vec::new(),
            clock: 0,
            pinned: Vec::new(),
            live_high_water: 0,
            decodes: 0,
        }
    }

    /// Take a resident group for the gather in flight, pinning it.
    fn claim(&mut self, gi: usize) -> Option<Arc<RecordBatch>> {
        self.clock += 1;
        let clock = self.clock;
        let batch = {
            let slot = self.slots.iter_mut().find(|s| s.0 == gi)?;
            slot.3 = clock;
            Arc::clone(&slot.1)
        };
        self.pin(gi);
        Some(batch)
    }

    fn pin(&mut self, gi: usize) {
        if !self.pinned.contains(&gi) {
            self.pinned.push(gi);
        }
    }

    /// Drop the pins once a gather has handed its batch back.
    fn release(&mut self) {
        self.pinned.clear();
    }

    fn pinned_slot_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| self.pinned.contains(&s.0))
            .map(|s| s.2)
            .sum()
    }

    /// The most this cache could offer without disturbing what the gather
    /// in flight holds. Asked before evicting anything, so a group that
    /// cannot fit even an emptied cache does not empty it on the way to
    /// finding that out — one oversized group would otherwise flush fifty
    /// small ones and then take the row path anyway, leaving every chunk
    /// that follows to re-decode them.
    fn free_ceiling(&self) -> usize {
        let unpinned = self.bytes - self.pinned_slot_bytes();
        self.budget.saturating_sub(self.bytes) + unpinned
    }

    /// Evict unpinned entries, least recently used first, until `want`
    /// bytes are free or nothing else can go. Returns the bytes free.
    fn reserve(&mut self, want: usize) -> usize {
        while self.budget.saturating_sub(self.bytes) < want {
            let lru = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !self.pinned.contains(&s.0))
                .min_by_key(|(_, s)| s.3)
                .map(|(i, _)| i);
            match lru {
                Some(i) => self.bytes -= self.slots.swap_remove(i).2,
                None => break,
            }
        }
        self.budget.saturating_sub(self.bytes)
    }

    /// Keep a freshly decoded group. The caller has already reserved room.
    fn admit(&mut self, gi: usize, batch: Arc<RecordBatch>, size: usize) {
        self.clock += 1;
        self.bytes += size;
        self.slots.push((gi, batch, size, self.clock));
        self.pin(gi);
    }
}

/// Where a gathered row comes from: a whole resident group, or a decode of
/// just the rows one chunk wants out of a group there is no room to hold.
enum Resident {
    Group(Arc<RecordBatch>, u64),
    Rows(Arc<RecordBatch>, Vec<u32>),
}

impl Resident {
    fn batch(&self) -> &RecordBatch {
        match self {
            Resident::Group(b, _) | Resident::Rows(b, _) => b,
        }
    }
    fn bytes(&self) -> usize {
        self.batch().get_array_memory_size()
    }
    fn offset(&self, row: u32) -> usize {
        match self {
            Resident::Group(_, start) => (row as u64 - start) as usize,
            Resident::Rows(_, rows) => rows.binary_search(&row).expect("row was decoded"),
        }
    }
}

/// Random access to the source rows the export keeps, addressed as indices
/// into the concatenation of the selected row groups — the same numbering
/// the key pass hands out.
struct Gatherer<'a> {
    src: &'a Source,
    /// A remote reader kept warm across decodes. The parquet builders take
    /// the reader by value and this pass opens one row group at a time, so
    /// without it every decode would start with a cold streaming window
    /// and re-request ranges the previous one had already paid for. Local
    /// files open again for free, and a duplicated descriptor would share
    /// its cursor, so they stay `None`.
    warm: Option<super::source::SourceReader>,
    meta: ArrowReaderMetadata,
    groups: Vec<usize>,
    /// `starts[i]` is the first row of `groups[i]`; one extra entry closes
    /// the last group.
    starts: Vec<u64>,
    projection: ProjectionMask,
    /// Columns the projection reads, checked against the first batch.
    columns: usize,
    schema: Option<SchemaRef>,
    /// Arrow bytes a full decode of a group actually took, once seen. The
    /// parquet-side estimate is only a starting guess.
    measured: std::collections::HashMap<usize, usize>,
    cache: GroupCache,
}

impl<'a> Gatherer<'a> {
    fn new(
        src: &'a Source,
        meta: ArrowReaderMetadata,
        groups: &[usize],
        map: &ColMap,
        budget: usize,
    ) -> Result<Self, String> {
        let mut starts = Vec::with_capacity(groups.len() + 1);
        let mut acc = 0u64;
        for &g in groups {
            starts.push(acc);
            acc += meta.metadata().row_groups()[g].num_rows().max(0) as u64;
        }
        starts.push(acc);
        let projection = map.mask(meta.metadata());
        let warm = match src.is_remote() {
            true => src.open()?.share_remote(),
            false => None,
        };
        Ok(Self {
            src,
            warm,
            meta,
            groups: groups.to_vec(),
            starts,
            projection,
            columns: map.0.len(),
            schema: None,
            measured: std::collections::HashMap::new(),
            cache: GroupCache::new(budget),
        })
    }

    fn reader(&self) -> Result<super::source::SourceReader, String> {
        match self.warm.as_ref().and_then(super::source::SourceReader::share_remote) {
            Some(r) => Ok(r),
            None => self.src.open(),
        }
    }

    fn group_of(&self, row: u32) -> usize {
        match self.starts.binary_search(&(row as u64)) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }

    /// A reader over one selected group, optionally restricted to the given
    /// rows (ascending, unique, in this gatherer's addressing).
    fn read_masked(
        &self,
        gi: usize,
        rows: Option<&[u32]>,
        mask: &ProjectionMask,
        batch_rows: usize,
    ) -> Result<ParquetRecordBatchReader, String> {
        let batch_size = batch_rows.clamp(1, READ_BATCH);
        let mut b = ParquetRecordBatchReaderBuilder::new_with_metadata(
            self.reader()?,
            self.meta.clone(),
        )
        .with_projection(mask.clone())
        .with_row_groups(vec![self.groups[gi]])
        .with_batch_size(batch_size);
        if let Some(rows) = rows {
            // Runs, not one selector per row: a gather chunk is usually a
            // handful of contiguous stretches, and the reader skips a run
            // far more cheaply than it skips a thousand single rows.
            let mut sel: Vec<RowSelector> = Vec::new();
            let mut pos = 0u64;
            let mut run = 0usize;
            for &r in rows {
                let p = r as u64 - self.starts[gi];
                if p == pos {
                    run += 1;
                } else {
                    if run > 0 {
                        sel.push(RowSelector::select(run));
                    }
                    sel.push(RowSelector::skip((p - pos) as usize));
                    run = 1;
                }
                pos = p + 1;
            }
            if run > 0 {
                sel.push(RowSelector::select(run));
            }
            b = b.with_row_selection(RowSelection::from(sel));
        }
        b.build().map_err(|e| format!("parquet read error: {e}"))
    }

    /// Concatenate what one decode produced, remembering the projected
    /// schema. The column count is checked once against the projection:
    /// `ProjectionMask::roots` indexes parquet roots, and everything here
    /// assumes those line up one-for-one with arrow's top-level fields.
    fn join(&mut self, mut batches: Vec<RecordBatch>, cols: usize) -> Result<RecordBatch, String> {
        let schema = match (&self.schema, batches.first()) {
            (Some(s), _) => s.clone(),
            (None, Some(b)) => {
                if b.num_columns() != cols {
                    return Err(format!(
                        "projection read {} columns for {cols} source fields — \
                         this file's arrow schema does not map one-for-one onto \
                         its parquet columns",
                        b.num_columns()
                    ));
                }
                self.schema = Some(b.schema());
                b.schema()
            }
            (None, None) => return Err("row group decoded to nothing".into()),
        };
        if batches.len() == 1 {
            return Ok(batches.pop().unwrap());
        }
        concat_batches(&schema, &batches).map_err(|e| format!("gather failed: {e}"))
    }

    /// Bytes a full decode of this group is expected to take: what an
    /// earlier decode measured, or twice its parquet-side uncompressed
    /// size — arrow's form runs wider on offsets, capacity and expanded
    /// dictionaries, and deciding on the parquet number alone let a group
    /// be decoded, rejected and re-decoded on every chunk that touched it.
    fn group_estimate(&self, gi: usize) -> usize {
        if let Some(&m) = self.measured.get(&gi) {
            return m;
        }
        (self.meta.metadata().row_groups()[self.groups[gi]]
            .total_byte_size()
            .max(0) as usize)
            .saturating_mul(2)
    }

    /// Decode a whole group, giving up if it goes past `limit` bytes.
    /// `None` means it did not fit; the caller reads the rows it wants
    /// instead.
    fn decode_group(&mut self, gi: usize, limit: usize) -> Result<Option<RecordBatch>, String> {
        self.cache.decodes += 1;
        // Several batches, not one. The size check can only stop a decode
        // between batches, so a group that arrives whole — which every
        // group of 64k rows or fewer does at the default batch size — has
        // already been materialized by the time it is measured, and the
        // limit polices nothing. A quarter of the room per batch bounds
        // the overshoot at one batch.
        let rows_in_group = (self.starts[gi + 1] - self.starts[gi]).max(1) as usize;
        let per_row = self.group_estimate(gi).div_ceil(rows_in_group).max(1);
        let batch_rows = (limit / per_row / 4).clamp(1, rows_in_group);
        let reader = self.read_masked(gi, None, &self.projection, batch_rows)?;
        let cols = self.columns;
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut bytes = 0usize;
        for res in reader {
            let b = res.map_err(|e| format!("parquet decode error: {e}"))?;
            bytes += b.get_array_memory_size();
            if bytes > limit {
                // Wider than its metadata predicted. Record what it has
                // already shown, doubled for the part not yet read, so the
                // next visit does not pay for the same discovery.
                self.measured.insert(gi, bytes.saturating_mul(2));
                return Ok(None);
            }
            batches.push(b);
        }
        let one = self.join(batches, cols)?;
        self.measured.insert(gi, one.get_array_memory_size());
        Ok(Some(one))
    }

    /// Decode exactly these rows of a group (ascending, unique).
    fn decode_rows(&mut self, gi: usize, rows: &[u32]) -> Result<RecordBatch, String> {
        self.cache.decodes += 1;
        let reader = self.read_masked(gi, Some(rows), &self.projection, rows.len())?;
        let cols = self.columns;
        let mut batches: Vec<RecordBatch> = Vec::new();
        for res in reader {
            batches.push(res.map_err(|e| format!("parquet decode error: {e}"))?);
        }
        self.join(batches, cols)
    }

    /// Decode a whole group for the key pass, keeping it for the gather
    /// pass when the budget allows. The scan needs the batch either way,
    /// and this path only runs when the whole selection was measured to
    /// fit, so the decode itself is not speculative.
    fn decode_for_scan(&mut self, gi: usize) -> Result<Arc<RecordBatch>, String> {
        let batch = self
            .decode_group(gi, usize::MAX)?
            .ok_or("row group decoded to nothing")?;
        let size = batch.get_array_memory_size();
        let batch = Arc::new(batch);
        if self.cache.reserve(size) >= size {
            self.cache.admit(gi, Arc::clone(&batch), size);
            self.cache.release();
        }
        Ok(batch)
    }

    /// The given source rows, in the given (output) order, as one batch.
    fn gather(&mut self, rows: &[u32]) -> Result<RecordBatch, String> {
        let of: Vec<usize> = rows.iter().map(|&r| self.group_of(r)).collect();
        let mut distinct = of.clone();
        distinct.sort_unstable();
        distinct.dedup();

        let mut residents: Vec<Resident> = Vec::with_capacity(distinct.len());
        for &gi in &distinct {
            if let Some(b) = self.cache.claim(gi) {
                residents.push(Resident::Group(b, self.starts[gi]));
                continue;
            }
            // Half the free budget, because `concat_batches` holds its
            // inputs and its output at once. A group that does not fit is
            // read row by row: that costs re-decodes, never memory, and it
            // is what keeps a chunk touching sixty groups from holding
            // sixty groups.
            let want = self.group_estimate(gi);
            let need = want.saturating_mul(2);
            let room = match need <= self.cache.free_ceiling() {
                true => self.cache.reserve(need) / 2,
                false => 0,
            };
            if let Some(b) = (want <= room)
                .then(|| self.decode_group(gi, room))
                .transpose()?
                .flatten()
            {
                let size = b.get_array_memory_size();
                let b = Arc::new(b);
                self.cache.admit(gi, Arc::clone(&b), size);
                residents.push(Resident::Group(b, self.starts[gi]));
                continue;
            }
            let mut wanted: Vec<u32> = rows
                .iter()
                .zip(&of)
                .filter(|&(_, &g)| g == gi)
                .map(|(&r, _)| r)
                .collect();
            wanted.sort_unstable();
            let b = Arc::new(self.decode_rows(gi, &wanted)?);
            residents.push(Resident::Rows(b, wanted));
        }

        // Settle the high-water mark on what is actually reachable, and
        // measure the residents themselves rather than trusting that they
        // are still the cache's pinned slots. Counting a claimed group as
        // zero "because the cache already has it" would make this number
        // depend on the pinning it is here to police: delete the pin
        // filter from `reserve` and every group would be evicted out from
        // under a live `Arc`, with the metric reporting nothing wrong.
        let live = residents.iter().map(Resident::bytes).sum::<usize>()
            + self.cache.bytes.saturating_sub(self.cache.pinned_slot_bytes());
        self.cache.live_high_water = self.cache.live_high_water.max(live);

        let indices: Vec<(usize, usize)> = rows
            .iter()
            .zip(&of)
            .map(|(&r, &gi)| {
                let si = distinct.binary_search(&gi).expect("group was resolved");
                (si, residents[si].offset(r))
            })
            .collect();
        let refs: Vec<&RecordBatch> = residents.iter().map(Resident::batch).collect();
        let out = interleave_record_batch(&refs, &indices)
            .map_err(|e| format!("gather failed: {e}"))?;
        drop(residents);
        self.cache.release();
        Ok(out)
    }
}

pub fn optimize(
    src: &Source,
    dst: &Path,
    opts: &OptimizeOptions,
    epsg_hint: Option<u32>,
    admin: Option<&super::partition::AdminJoinSpec>,
    progress: &dyn Fn(f32, &str),
) -> Result<OptimizeReport, String> {
    let t0 = std::time::Instant::now();
    progress(0.0, "reading metadata");

    // The footer is read once and handed to every reader of both passes.
    let reader = src.open()?;
    let arrow_meta = ArrowReaderMetadata::load(&reader, Default::default())
        .map_err(|e| format!("not a parquet file: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(reader, arrow_meta.clone());
    let src_schema = builder.schema().clone();
    let meta = builder.metadata().clone();
    let fmd = meta.file_metadata();

    // --- source geo metadata ---
    let kv = fmd.key_value_metadata().cloned().unwrap_or_default();
    let geo_meta: Option<Value> = kv
        .iter()
        .find(|kv| kv.key == "geo")
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| serde_json::from_str(v).ok());
    let (primary, geom_idx, src_encoding) = match opts.xy_geom {
        // x/y source: a WKB point column is synthesized during the read
        // and appended after the source fields.
        Some(_) => {
            let name = if src_schema.index_of("geometry").is_ok() {
                "xy_geometry"
            } else {
                "geometry"
            };
            (name.to_string(), src_schema.fields().len(), GeomEncoding::Wkb)
        }
        None => {
            let primary = geo_meta
                .as_ref()
                .and_then(|m| m.get("primary_column"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    guess_geom_column(&src_schema).unwrap_or_else(|| "geometry".into())
                });
            let geom_idx = src_schema
                .index_of(&primary)
                .map_err(|_| format!("geometry column '{primary}' not found"))?;
            let src_encoding = geo_meta
                .as_ref()
                .and_then(|m| m.get("columns")?.get(&primary)?.get("encoding")?.as_str())
                .map(|e| {
                    GeomEncoding::parse(e).ok_or_else(|| format!("encoding '{e}' not supported"))
                })
                .transpose()?
                .unwrap_or_default();
            (primary, geom_idx, src_encoding)
        }
    };

    // CRS, best source first: geo metadata PROJJSON, then the GEOMETRY
    // logical type's crs string (2.0 sources), then the caller's hint.
    let crs_value: Option<Value> = geo_meta
        .as_ref()
        .and_then(|m| m.get("columns")?.get(&primary)?.get("crs").cloned())
        .filter(|v| !v.is_null())
        .or_else(|| logical_type_crs(&meta, &primary))
        .or_else(|| {
            epsg_hint.filter(|&e| e != 4326).map(|e| {
                json!({"id": {"authority": "EPSG", "code": e}})
            })
        });
    // A carried-but-unidentified CRS (shapefile imports from ESRI .prj
    // without an EPSG authority): the source declares `crs: null` with
    // the proj4 string in our `geopq:crs` extension. Both must survive
    // the rewrite — omitting `crs` would falsely claim the CRS84
    // default and put the output thousands of km off the map.
    let src_col_meta =
        geo_meta.as_ref().and_then(|m| m.get("columns")?.get(&primary).cloned());
    let vendor_crs: Option<Value> = crs_value
        .is_none()
        .then(|| src_col_meta.as_ref()?.get("geopq:crs").cloned())
        .flatten();
    let crs_explicit_null = crs_value.is_none()
        && (vendor_crs.is_some()
            || src_col_meta
                .as_ref()
                .and_then(|c| c.get("crs"))
                .is_some_and(Value::is_null));

    // Source covering bbox column (dropped and rebuilt if we write our own).
    let src_covering_root: Option<String> = geo_meta
        .as_ref()
        .and_then(|m| {
            m.get("columns")?
                .get(&primary)?
                .get("covering")?
                .get("bbox")?
                .get("xmin")?
                .as_array()?
                .first()?
                .as_str()
                .map(str::to_string)
        });

    // Columns carrying bloom filters in the source (leaf paths).
    let src_bloom: Vec<Vec<String>> = meta
        .row_groups()
        .first()
        .map(|rg| {
            rg.columns()
                .iter()
                .filter(|c| c.bloom_filter_offset().is_some())
                .map(|c| c.column_descr().path().parts().to_vec())
                .collect()
        })
        .unwrap_or_default();

    let rg_before = meta.num_row_groups();
    let all_rg_rows: Vec<usize> = meta
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as usize)
        .collect();

    // --- row groups the export can touch ---
    // A viewport export reads only the row groups the viewport can touch.
    // Filtering after the read was correct but cost the whole file: on a
    // remote source that is the entire download for a handful of
    // features. Row-group boxes come from the same metadata the viewer
    // prunes with, so this needs no extra request.
    let mut keep_groups: Vec<usize> = (0..rg_before).collect();
    if let Some(rect) = opts.filter_rect {
        let boxes = super::loader::rg_bboxes_from_metadata(
            &builder,
            geo_meta.as_ref(),
            Some(geom_idx),
            &primary,
            src_encoding,
            // Only used to sanity-check degenerate lon/lat boxes; a
            // wrong guess here cannot select the wrong groups.
            false,
        );
        if let Some((source, boxes)) = boxes {
            let keep: Vec<usize> = super::loader::intersecting_rgs(&boxes, rect)
                .into_iter()
                .map(|g| g as usize)
                .collect();
            if keep.is_empty() {
                return Err("no features intersect the current viewport".into());
            }
            log::info!(
                "viewport export: {} of {} row groups intersect ({source})",
                keep.len(),
                rg_before
            );
            keep_groups = keep;
        }
    }
    // Nothing reads through this handle again; both passes open their own.
    drop(builder);
    let keep_groups = readable_groups(keep_groups, &all_rg_rows);
    let total_rows: usize = keep_groups.iter().map(|&g| all_rg_rows[g]).sum();
    // Over the selected groups, not the file: a viewport export of a large
    // layer must not be sized by bytes it never touches.
    let uncompressed: u64 = keep_groups
        .iter()
        .map(|&g| meta.row_groups()[g].total_byte_size().max(0) as u64)
        .sum();

    // x/y source: extend the schema with the synthesized point column.
    let src_base = src_schema.fields().len();
    let src_schema = match opts.xy_geom {
        Some(_) => {
            let mut fields: Vec<arrow::datatypes::Field> = src_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            fields.push(arrow::datatypes::Field::new(&primary, DataType::Binary, true));
            Arc::new(arrow::datatypes::Schema::new_with_metadata(
                fields,
                src_schema.metadata().clone(),
            ))
        }
        None => src_schema,
    };

    // --- partition keys, resolved before either pass picks its columns ---
    use super::partition::{self, PartitionBy};
    let part_fields: Vec<String> = match &opts.partition {
        PartitionBy::Fields(f) if f.is_empty() => {
            return Err("partitioning by fields: none selected".into())
        }
        PartitionBy::Fields(f) => f.clone(),
        _ => Vec::new(),
    };
    let partitioned = !matches!(opts.partition, PartitionBy::None);
    let h3_name = opts.h3_resolution.map(|r| format!("h3_r{r}"));
    let need_lonlat = opts.h3_resolution.is_some()
        || matches!(opts.partition, PartitionBy::AdaptiveH3 { .. });
    let data_crs = if need_lonlat || admin.is_some() {
        // A vendor proj4 CRS resolves for centroid math like any other.
        let vendor = vendor_crs.as_ref().and_then(|v| {
            let p4 = v.get("proj4")?.as_str()?;
            let name = v.get("name").and_then(Value::as_str).unwrap_or("from .prj");
            super::crs::Crs::from_proj4(p4, None, name).ok()
        });
        Some(match vendor {
            Some(c) => c,
            None => super::crs::Crs::from_geoparquet_crs(crs_value.as_ref())
                .map_err(|e| format!("H3/admin needs a resolvable CRS: {e}"))?,
        })
    } else {
        None
    };
    // Partition fields that come from the data rather than from a derived
    // column: (slot in `part_fields`, source column).
    let mut part_src: Vec<(usize, usize)> = Vec::new();
    for (slot, name) in part_fields.iter().enumerate() {
        if Some(name.as_str()) == h3_name.as_deref()
            || Some(name.as_str()) == admin.map(|a| a.out_name.as_str())
        {
            continue;
        }
        let idx = src_schema
            .index_of(name)
            .map_err(|_| format!("partition field '{name}' not found"))?;
        if idx == geom_idx {
            return Err("cannot partition by the geometry column".into());
        }
        part_src.push((slot, idx));
    }

    // --- COGP levels ---
    // The profile owns the physical layout of the file, which is why it
    // refuses to share it: hive parts would each need their own level list
    // (the spec has no notion of a partitioned dataset), and the GeoArrow
    // flavour is left out deliberately to keep the surface small. What it
    // *takes over* rather than refuses is the sort and the covering column,
    // both of which it requires.
    let cogp_gsds: Option<Vec<f64>> = match &opts.cogp {
        None => None,
        Some(c) => {
            if partitioned {
                return Err("COGP levels and partitioned output cannot combine: \
                            the profile describes one file's row groups"
                    .into());
            }
            if opts.version == GpVersion::V1_1GeoArrow {
                return Err("COGP needs the WKB 1.1 or the native 2.0 flavour, \
                            not GeoArrow"
                    .into());
            }
            Some(c.gsds()?)
        }
    };
    let cogp_on = cogp_gsds.is_some();
    // COGP v0.1 requires the covering bbox with row-group statistics
    // (§5.1). On 2.0 the native GEOMETRY statistics are the pruning signal
    // instead, so the covering column stays the user's choice there.
    let cogp_needs_covering = cogp_on && opts.version == GpVersion::V1_1;
    // §5.2 asks producers to cluster spatially within each level; the
    // Hilbert order is what this app has, so COGP simply switches it on.
    let hilbert_sort = opts.hilbert_sort || cogp_on;
    let cogp_units = cogp_on.then(|| cogp_units(crs_value.as_ref(), vendor_crs.as_ref()));
    if let Some(u) = cogp_units {
        log::info!("COGP: measuring gsd thresholds in {}", u.label());
    }
    // The rank column is read by the key pass like a partition key is.
    let cogp_rank_src: Option<usize> = match opts.cogp.as_ref().and_then(|c| c.rank.as_ref()) {
        Some((name, _)) => Some(
            src_schema
                .index_of(name)
                .map_err(|_| format!("COGP rank column '{name}' not found"))?,
        ),
        None => None,
    };

    // --- output columns, and what each pass therefore reads ---
    let write_covering = opts.covering || cogp_needs_covering;
    let drop_covering = write_covering.then_some(src_covering_root.as_deref()).flatten();
    let mut kept_src_indices: Vec<usize> = Vec::new();
    for (i, f) in src_schema.fields().iter().enumerate() {
        // Only a covering-style struct named `bbox` is dropped for the
        // rebuilt covering column; a plain attribute that happens to be
        // called `bbox` is data and must survive (the new covering column
        // is renamed around it below).
        if Some(f.name().as_str()) == drop_covering
            || (write_covering
                && f.name() == "bbox"
                && i != geom_idx
                && is_covering_struct(f.data_type()))
            // Hive convention: partition columns live in the path only.
            || (part_fields.iter().any(|p| p == f.name()) && i != geom_idx)
        {
            continue;
        }
        kept_src_indices.push(i);
    }
    // The key pass reads geometry and the partition keys; the gather pass
    // reads the output's own columns. Both go through one projection so a
    // source small enough to stay resident is decoded once rather than
    // once per pass — on a remote source that difference is the download.
    let mut key_cols: Vec<usize> = match opts.xy_geom {
        Some((xi, yi)) => vec![xi, yi],
        None => vec![geom_idx],
    };
    key_cols.extend(part_src.iter().map(|&(_, i)| i));
    key_cols.extend(cogp_rank_src);
    key_cols.sort_unstable();
    key_cols.dedup();
    let mut read_cols: Vec<usize> = kept_src_indices
        .iter()
        .copied()
        .filter(|&i| i < src_base)
        .chain(key_cols.iter().copied())
        .collect();
    read_cols.sort_unstable();
    read_cols.dedup();
    let (read_map, key_map) = (ColMap(read_cols), ColMap(key_cols));

    // What still scales with the input is the row count, not the file
    // size: the key pass keeps a fixed-width row of side tables and the
    // gather pass keeps a byte-budgeted slice of the source. Say so before
    // the allocator does — what issue #12 reported was an OS kill with no
    // message and a truncated output file in its place.
    let key_pass = total_rows as u64 * key_pass_bytes_per_row(opts, admin.is_some(), &part_fields);
    if key_pass > MAX_KEY_PASS_BYTES {
        return Err(format!(
            "{total_rows} rows need about {} for the sort index alone, past the {} \
             this build allows — export a viewport, or split the layer first",
            super::info::fmt_bytes(key_pass),
            super::info::fmt_bytes(MAX_KEY_PASS_BYTES)
        ));
    }

    // The output target has to be writable before minutes of scanning, not
    // after: on Windows the rename that finishes the job cannot replace a
    // file another process holds open, and finding that out at the end
    // throws the whole rewrite away.
    probe_output(dst, partitioned)?;

    // --- key pass: geometry bboxes, sort keys, partition keys ---
    progress(0.02, "scanning geometry");
    let budget = decode_budget();
    let mut gather = Gatherer::new(src, arrow_meta, &keep_groups, &read_map, budget)?;
    // Doubled because `total_byte_size` is the parquet-side uncompressed
    // size and arrow's decoded form runs wider on strings and binaries.
    // Guessing high only costs a re-decode; the cache enforces the budget.
    let resident = uncompressed.saturating_mul(2) <= budget as u64;
    let key_mask = key_map.mask(&meta);
    // Sized up front: doubling these while they are the largest thing in
    // memory would briefly need half again as much as the estimate allows.
    let mut scan = ScanOut {
        row_bboxes: Vec::with_capacity(total_rows),
        kept: Vec::with_capacity(total_rows),
        interners: part_src.iter().map(|_| partition::Interner::default()).collect(),
        ..ScanOut::default()
    };
    let mut rg_boxes_before: Vec<[f64; 4]> = Vec::with_capacity(keep_groups.len());
    let base_plan = ScanPlan {
        map: &read_map,
        geom_idx,
        xy: opts.xy_geom,
        encoding: src_encoding,
        filter: opts.filter_rect,
        part_src: &part_src,
        part_names: &part_fields,
        cogp_kinds: cogp_on,
        cogp_rank_src,
    };
    let key_plan = ScanPlan { map: &key_map, ..base_plan };
    for gi in 0..keep_groups.len() {
        scan.group_box = None;
        if resident {
            let batch = gather.decode_for_scan(gi)?;
            scan_batch(&batch, &base_plan, &mut scan)?;
        } else {
            for res in gather.read_masked(gi, None, &key_mask, READ_BATCH)? {
                let batch = res.map_err(|e| format!("parquet decode error: {e}"))?;
                scan_batch(&batch, &key_plan, &mut scan)?;
            }
        }
        rg_boxes_before.extend(scan.group_box);
        progress(
            0.02 + 0.38 * ((gi + 1) as f32 / keep_groups.len().max(1) as f32),
            "scanning geometry",
        );
    }
    if scan.scanned == 0 {
        return Err("file has no rows".into());
    }
    let ScanOut {
        row_bboxes,
        kept,
        geom_types,
        interners,
        row_kinds,
        rank_parts,
        ..
    } = scan;
    let rows = row_bboxes.len();
    if rows == 0 {
        return Err("no features intersect the current viewport".into());
    }
    let overlap_before = super::loader::bbox_overlap_metric(&rg_boxes_before);
    // Metadata bbox reflects what is actually exported.
    let file_bbox = union_bboxes(row_bboxes.iter().flatten());

    // --- COGP level per row ---
    // Before the sort, because the sort is by (level, Hilbert key): the
    // levels decide the coarse-to-fine order of the file and the Hilbert
    // key clusters within each of them.
    let cogp_level: Option<Vec<u8>> = match (&cogp_gsds, opts.cogp.as_ref()) {
        (Some(gsds), Some(c)) => {
            progress(0.39, "assigning COGP levels");
            let ranks: Option<Vec<u32>> = match &c.rank {
                None => None,
                Some((name, order)) => {
                    let parts: Vec<&dyn Array> = rank_parts.iter().map(|a| a.as_ref()).collect();
                    let col = arrow::compute::concat(&parts)
                        .map_err(|e| format!("COGP rank column '{name}': {e}"))?;
                    // nulls_first keeps nulls at the lowest rank whichever
                    // way round the order is; `descending` only decides
                    // which end of the values earns the winning rank.
                    let sort = arrow::compute::SortOptions {
                        descending: *order == RankOrder::Asc,
                        nulls_first: true,
                    };
                    Some(
                        arrow::compute::rank(col.as_ref(), Some(sort))
                            .map_err(|e| format!("COGP rank column '{name}': {e}"))?,
                    )
                }
            };
            Some(assign_cogp_levels(
                &row_bboxes,
                &row_kinds,
                gsds,
                cogp_units.unwrap_or(CogpUnits::Linear(1.0)),
                c,
                ranks.as_deref(),
            ))
        }
        _ => None,
    };
    drop(rank_parts);

    // --- sort order ---
    progress(0.40, "sorting (Hilbert)");
    let mut order: Vec<u32> = (0..rows as u32).collect();
    if hilbert_sort {
        let fb = file_bbox.unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let sx = (fb[2] - fb[0]).max(f64::MIN_POSITIVE);
        let sy = (fb[3] - fb[1]).max(f64::MIN_POSITIVE);
        let n = (1u64 << HILBERT_ORDER) as f64;
        let codes: Vec<u64> = row_bboxes
            .iter()
            .map(|b| match b {
                Some(b) => {
                    let cx = ((b[0] + b[2]) * 0.5 - fb[0]) / sx;
                    let cy = ((b[1] + b[3]) * 0.5 - fb[1]) / sy;
                    let gx = ((cx * n) as u64).min((1 << HILBERT_ORDER) - 1) as u32;
                    let gy = ((cy * n) as u64).min((1 << HILBERT_ORDER) - 1) as u32;
                    hilbert_xy2d(HILBERT_ORDER, gx, gy)
                }
                None => u64::MAX, // null geometries sort last
            })
            .collect();
        // COGP: level first, Hilbert key within it. The file then reads
        // coarse to fine, and each level is still spatially clustered, which
        // is what keeps the row-group bboxes tight enough for §5.1 pruning
        // to be worth anything inside a level.
        match &cogp_level {
            Some(lv) => order.sort_by_key(|&i| (lv[i as usize], codes[i as usize])),
            None => order.sort_by_key(|&i| codes[i as usize]),
        }
    }
    // (No `else` branch for levels: `hilbert_sort` is forced on above
    // whenever COGP is, so an unsorted COGP export cannot happen.)

    // --- derived columns ---
    progress(0.43, "computing derived columns");
    let lonlat: Option<Vec<Option<(f64, f64)>>> = need_lonlat.then(|| {
        partition::centroids_in(&row_bboxes, data_crs.as_ref().unwrap(), &super::crs::Crs::wgs84())
    });
    let h3_vals: Option<Vec<Option<u64>>> = match (opts.h3_resolution, &lonlat) {
        (Some(res), Some(ll)) => Some(partition::h3_cells(ll, res)?),
        _ => None,
    };
    // Interned as it is joined: the values come from a boundary layer of
    // at most 200k polygons however many features are exported, so a
    // `String` per row would be the largest allocation of the run and
    // every one of them a duplicate.
    let admin_vals: Option<partition::FieldCodes> = match admin {
        Some(spec) => {
            let cb = partition::centroids_in(&row_bboxes, data_crs.as_ref().unwrap(), &spec.crs);
            Some(partition::admin_join(spec, &spec.out_name, &cb)?)
        }
        None => None,
    };
    // Per-row values for each hive partition field, interned: the data
    // ones were read by the key pass, the derived ones come from the
    // columns just computed.
    let mut coded: Vec<Option<partition::FieldCodes>> = part_fields.iter().map(|_| None).collect();
    for (&(slot, _), it) in part_src.iter().zip(interners) {
        coded[slot] = Some(it.finish(&part_fields[slot]));
    }
    for (slot, name) in part_fields.iter().enumerate() {
        if coded[slot].is_some() {
            continue;
        }
        if Some(name.as_str()) == h3_name.as_deref() {
            let vals = h3_vals
                .as_ref()
                .ok_or("internal: h3 partition field without h3 column")?;
            let mut it = partition::Interner::default();
            for v in vals {
                let s = v
                    .and_then(|v| h3o::CellIndex::try_from(v).ok())
                    .map(|c| c.to_string());
                it.push(s.as_deref())?;
            }
            coded[slot] = Some(it.finish(name));
            continue;
        }
        // The admin join already produced codes; partitioning on it needs
        // no second pass over the values, only the cardinality check the
        // hive split does anyway.
        let vals = admin_vals
            .as_ref()
            .ok_or("internal: admin partition field without a join")?;
        coded[slot] = Some(partition::FieldCodes {
            name: name.clone(),
            dict: vals.dict.clone(),
            codes: vals.codes.clone(),
        });
    }
    let field_codes: Vec<partition::FieldCodes> = coded.into_iter().flatten().collect();

    // --- geometry output form ---
    let geom_out: GeomOut = match opts.version {
        GpVersion::V1_1GeoArrow => {
            // GeoArrow coordinate arrays are 2D here: Z is dropped by the
            // rebuild, so the family choice ignores the " Z" suffix.
            let target =
                geoarrow::target_encoding(geom_types.iter().map(|t| t.trim_end_matches(" Z")))?;
            if target == src_encoding {
                GeomOut::PassThrough
            } else {
                GeomOut::ToGa(target)
            }
        }
        GpVersion::V1_1 | GpVersion::V2_0 if !src_encoding.is_wkb() => GeomOut::ToWkb,
        _ => GeomOut::PassThrough,
    };
    let out_encoding = match (&geom_out, opts.version) {
        (GeomOut::ToGa(t), _) => *t,
        (GeomOut::PassThrough, GpVersion::V1_1GeoArrow) => src_encoding,
        _ => GeomEncoding::Wkb,
    };
    // Auxiliary GeoArrow column (2.0 flavor): coordinate arrays next to
    // the native GEOMETRY primary, for GeoArrow-aware readers.
    let aux_target: Option<GeomEncoding> = if opts.geoarrow_aux
        && opts.version == GpVersion::V2_0
    {
        Some(geoarrow::target_encoding(
            geom_types.iter().map(|t| t.trim_end_matches(" Z")),
        )?)
    } else {
        None
    };

    // --- output schema ---
    let mut fields: Vec<Field> = Vec::new();
    for &i in &kept_src_indices {
        let f = src_schema.field(i);
        let mut field = f.clone();
        if i == geom_idx {
            // Rebuild the geometry field per the output form, then apply
            // version-specific typing.
            match &geom_out {
                GeomOut::ToGa(t) => {
                    field = Field::new(f.name(), geoarrow::data_type(*t), true);
                }
                GeomOut::ToWkb => {
                    field = Field::new(f.name(), DataType::Binary, true);
                }
                GeomOut::PassThrough => {}
            }
            match opts.version {
                GpVersion::V2_0 => {
                    let crs_str = crs_value.as_ref().map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    });
                    let md = parquet_geospatial::WkbMetadata::new(crs_str.as_deref(), None);
                    field
                        .try_with_extension_type(parquet_geospatial::WkbType::new(Some(md)))
                        .map_err(|e| format!("cannot tag geometry column: {e}"))?;
                }
                GpVersion::V1_1 | GpVersion::V1_1GeoArrow => {
                    // A native-GEOMETRY source propagates its extension type
                    // through the arrow schema; strip it so an explicit 1.1
                    // export carries no logical type (max reader
                    // compatibility — the point of choosing 1.1).
                    let mut md = field.metadata().clone();
                    md.remove("ARROW:extension:name");
                    md.remove("ARROW:extension:metadata");
                    field.set_metadata(md);
                }
            }
        }
        fields.push(field);
    }
    let bbox_fields: Fields = ["xmin", "ymin", "xmax", "ymax"]
        .iter()
        .map(|n| Field::new(*n, DataType::Float64, true))
        .collect();
    // The conventional `bbox` name may be taken by a surviving source
    // attribute; readers follow the geo metadata's covering section, so
    // any non-colliding name works.
    let covering_name = {
        let mut name = "bbox".to_string();
        let mut k = 0usize;
        while fields.iter().any(|f| f.name() == &name) {
            k += 1;
            name = format!("bbox_{k}");
        }
        name
    };
    if write_covering {
        fields.push(Field::new(
            covering_name.as_str(),
            DataType::Struct(bbox_fields.clone()),
            true,
        ));
    }
    if let Some(t) = aux_target {
        let mut name = format!("{primary}_geoarrow");
        let mut k = 0usize;
        while fields.iter().any(|f| f.name() == &name) {
            k += 1;
            name = format!("{primary}_geoarrow_{k}");
        }
        let mut f = Field::new(name.as_str(), geoarrow::data_type(t), true);
        f.set_metadata(std::collections::HashMap::from([(
            "ARROW:extension:name".to_string(),
            format!("geoarrow.{}", t.geo_name()),
        )]));
        fields.push(f);
    }
    // Derived columns (skipped when hive uses them as path-only keys).
    let write_h3 = h3_name
        .as_deref()
        .filter(|n| !part_fields.iter().any(|p| p == n));
    if let Some(n) = write_h3 {
        fields.push(Field::new(n, DataType::UInt64, true));
    }
    let write_admin = admin
        .map(|a| a.out_name.as_str())
        .filter(|n| !part_fields.iter().any(|p| p == n));
    if let Some(n) = write_admin {
        fields.push(Field::new(n, DataType::Utf8, true));
    }

    let out_schema = Arc::new(Schema::new(fields));

    // --- writer properties (rebuilt per output file) ---
    let mut bloom_columns: Vec<String> = Vec::new();
    match opts.bloom {
        BloomMode::Preserve => {
            for parts in &src_bloom {
                if parts.first().map(String::as_str) == drop_covering {
                    continue; // rebuilt bbox column gets no bloom filter
                }
                if parts
                    .first()
                    .is_some_and(|r| part_fields.iter().any(|p| p == r))
                {
                    continue; // partition columns are path-only
                }
                bloom_columns.push(parts.join("."));
            }
        }
        BloomMode::AllAttributes => {
            for f in out_schema.fields() {
                if f.name() == &primary || (write_covering && f.name() == &covering_name) {
                    continue;
                }
                let eligible = matches!(
                    f.data_type(),
                    DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                        | DataType::Binary
                        | DataType::LargeBinary
                        | DataType::BinaryView
                        | DataType::Int8
                        | DataType::Int16
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64
                        | DataType::Date32
                        | DataType::Date64
                );
                if eligible {
                    bloom_columns.push(f.name().clone());
                }
            }
        }
        BloomMode::None => {}
    }
    let src_bloom_paths: Vec<Vec<String>> = match opts.bloom {
        BloomMode::Preserve => bloom_columns
            .iter()
            .map(|c| c.split('.').map(str::to_string).collect())
            .collect(),
        _ => bloom_columns.iter().map(|c| vec![c.clone()]).collect(),
    };
    let make_props = || {
        let mut props = WriterProperties::builder()
            // The parquet envelope follows the GeoParquet flavour. 2.0
            // means the native GEOMETRY logical type, which only a recent
            // reader understands, and such a reader certainly handles V2
            // data pages — announcing the newest geo spec inside the
            // oldest envelope is a mismatch of intent, and inspection
            // tools rightly flag it. Older flavours stay on V1 pages,
            // where being readable by everything is the whole point.
            // Measured on CORINE and on 47-column parcels: no size
            // difference either way once zstd has run, so this is about
            // what the file says it is, not about bytes.
            .set_writer_version(if opts.version == GpVersion::V2_0 {
                parquet::file::properties::WriterVersion::PARQUET_2_0
            } else {
                parquet::file::properties::WriterVersion::PARQUET_1_0
            })
            .set_compression(opts.codec.compression())
            .set_max_row_group_row_count(Some(opts.row_group_size))
            .set_max_row_group_bytes(Some(opts.row_group_bytes))
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_created_by(format!(
                "{} {}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            ));
        if write_covering {
            // Small pages on the bbox leaves (~4k rows at 8 B/value) give
            // the page index sub-row-group granularity, so readers can
            // prune at page level instead of whole row groups. Dictionary
            // encoding is disabled there: coordinates are mostly unique
            // (no dict win) and the page-size cap applies to the encoded
            // size, which tiny dict indices would defeat.
            for leaf in ["xmin", "ymin", "xmax", "ymax"] {
                let path = ColumnPath::new(vec![covering_name.clone(), leaf.into()]);
                props = props
                    .set_column_data_page_size_limit(path.clone(), BBOX_LEAF_PAGE_BYTES)
                    .set_column_dictionary_enabled(path, false);
            }
        }
        for parts in &src_bloom_paths {
            props = props
                .set_column_bloom_filter_enabled(ColumnPath::new(parts.clone()), true)
                .set_column_bloom_filter_fpp(ColumnPath::new(parts.clone()), 0.01);
        }
        props.build()
    };

    // --- partition plan ---
    let parts: Vec<(String, Vec<u32>)> = match &opts.partition {
        PartitionBy::None => vec![(String::new(), order.clone())],
        PartitionBy::Fields(_) => partition::split_by_field_codes(&order, &field_codes)?,
        PartitionBy::AdaptiveH3 { target_rows, max_res } => partition::split_adaptive_h3(
            &order,
            lonlat.as_ref().ok_or("internal: adaptive H3 without centroids")?,
            (*target_rows).max(1),
            *max_res,
        )?,
    };

    // --- gather pass: re-read the source in sorted order and write ---
    progress(0.45, "writing");
    // The writer can only close a row group between `write` calls, so a
    // byte cap is worth exactly the granularity we hand it: one chunk of
    // 65k heavy features would be one 200 MB group however low the cap.
    // Size the chunk from the source's own bytes per row, aiming at half
    // the budget so a group lands near it rather than at twice it.
    let bytes_per_row = (uncompressed / total_rows.max(1) as u64).max(1);
    let rows_per_budget = (opts.row_group_bytes as u64 / 2 / bytes_per_row).max(1);
    let chunk_rows = opts
        .row_group_size
        .min(READ_BATCH)
        .min(rows_per_budget.min(usize::MAX as u64) as usize)
        .max(1);
    let out_geom_pos = kept_src_indices
        .iter()
        .position(|&i| i == geom_idx)
        .ok_or("geometry column dropped from output")?;

    // COGP level boundaries as (gsd, end offset in `order`), exclusive.
    // A level that received no feature is dropped rather than written: it
    // cannot own a row group, and an entry repeating the previous
    // `row_group_end` would break the strictly-increasing rule of §5.3.
    // Its GSD simply disappears from the list, which stays legal — the
    // remaining ones are still strictly decreasing.
    let cogp_bounds: Option<Vec<(f64, usize)>> = match (&cogp_gsds, &cogp_level) {
        (Some(gsds), Some(lv)) => {
            let mut counts = vec![0usize; gsds.len()];
            for &r in &order {
                counts[lv[r as usize] as usize] += 1;
            }
            let mut end = 0usize;
            let mut out: Vec<(f64, usize)> = Vec::new();
            for (li, n) in counts.iter().enumerate() {
                end += n;
                if *n > 0 {
                    out.push((gsds[li], end));
                }
            }
            Some(out)
        }
        _ => None,
    };

    // Which partition each row belongs to, so one sweep of the sorted
    // order can feed every open writer.
    let row_part: Vec<u32> = if parts.len() > 1 {
        let mut v = vec![0u32; rows];
        for (pi, (_, part_rows)) in parts.iter().enumerate() {
            for &r in part_rows {
                v[r as usize] = pi as u32;
            }
        }
        v
    } else {
        Vec::new()
    };

    // One file per partition; everything (covering, bloom, ordering, geo
    // metadata with per-file bbox) applies inside each file. The writers
    // of one batch of partitions stay open across a single sweep of the
    // sorted order, so the source is re-read once per batch rather than
    // once per file.
    //
    // Nothing is renamed into place until the last sweep has closed: a
    // dataset that published as each sweep finished would answer a failure
    // in sweep two with a directory of thirty-two valid partitions and
    // eight missing ones, readable and wrong, with nothing saying so.
    let mut staged = StagedOutputs::new(dst, partitioned)?;
    let mut rg_after_boxes: Vec<[f64; 4]> = Vec::new();
    let mut rg_after = 0usize;
    let mut written = 0usize;
    // Row group each COGP level ended on, in the writer's own count.
    let mut cogp_rg_ends: Vec<usize> = Vec::new();
    let mut cogp_meta: Option<super::cogp::Cogp> = None;
    for lo in (0..parts.len()).step_by(MAX_OPEN_WRITERS) {
        let hi = (lo + MAX_OPEN_WRITERS).min(parts.len());
        let mut open: Vec<PartWriter> = Vec::new();
        for (pi, (rel, part_order)) in parts.iter().enumerate().take(hi).skip(lo) {
            let part_bbox =
                union_bboxes(part_order.iter().filter_map(|&r| row_bboxes[r as usize].as_ref()));
            let out_file = staged.create(rel)?;
            let mut writer =
                ArrowWriter::try_new(out_file, out_schema.clone(), Some(make_props()))
                    .map_err(|e| format!("writer init: {e}"))?;
            // `geo` is rebuilt; other source key-value metadata passes through.
            writer.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
                "geo".to_string(),
                build_geo_meta(
                    opts,
                    &primary,
                    crs_value.as_ref(),
                    vendor_crs.as_ref(),
                    crs_explicit_null,
                    &geom_types,
                    out_encoding,
                    part_bbox,
                    &covering_name,
                    write_covering,
                )
                .to_string(),
            ));
            // `cogp` joins `geo` in being rebuilt rather than copied: the
            // source's level boundaries describe the source's row groups,
            // and carrying them onto a file this pass has just re-laid out
            // would publish a confidently wrong index.
            for entry in kv.iter().filter(|kv| {
                kv.key != "geo" && kv.key != "ARROW:schema" && kv.key != super::cogp::KEY
            }) {
                writer.append_key_value_metadata(entry.clone());
            }
            open.push(PartWriter { writer, part: pi });
        }

        let sweep: Vec<u32> = if parts.len() == 1 {
            order.clone()
        } else {
            order
                .iter()
                .copied()
                .filter(|&r| (lo..hi).contains(&(row_part[r as usize] as usize)))
                .collect()
        };
        // Chunk boundaries, cut at every COGP level end. A row group must
        // hold rows from one level only (§5.3), and the writer can only
        // close a group between `write` calls — so a level end has to be a
        // chunk end too, or the tail of one level and the head of the next
        // land in the same group and no flush can separate them after the
        // fact. Without COGP this is one run of `chunk_rows` slices, as
        // before.
        let level_ends: Vec<usize> = match &cogp_bounds {
            Some(b) => b.iter().map(|&(_, end)| end).collect(),
            None => vec![sweep.len()],
        };
        let mut chunks: Vec<(std::ops::Range<usize>, bool)> = Vec::new();
        let mut seg_lo = 0usize;
        for end in level_ends {
            while seg_lo < end {
                let hi = (seg_lo + chunk_rows).min(end);
                chunks.push((seg_lo..hi, hi == end));
                seg_lo = hi;
            }
        }
        for (range, at_level_end) in chunks {
            let chunk = &sweep[range];
            let src_rows: Vec<u32> = chunk.iter().map(|&r| kept[r as usize]).collect();
            let gathered = gather.gather(&src_rows)?;
            let geom_array = chunk_geometry(&gathered, &read_map, geom_idx, opts.xy_geom)?;
            let mut cols: Vec<ArrayRef> = kept_src_indices
                .iter()
                .map(|&i| {
                    if i == geom_idx {
                        Arc::clone(&geom_array)
                    } else {
                        gathered.column(read_map.pos(i)).clone()
                    }
                })
                .collect();
            if !matches!(geom_out, GeomOut::PassThrough) {
                cols[out_geom_pos] =
                    transcode_geometry(cols[out_geom_pos].as_ref(), src_encoding, &geom_out)?;
            }
            if write_covering {
                cols.push(build_bbox_column(chunk, &row_bboxes, &bbox_fields));
            }
            if let Some(t) = aux_target {
                cols.push(transcode_geometry(
                    geom_array.as_ref(),
                    src_encoding,
                    &GeomOut::ToGa(t),
                )?);
            }
            if write_h3.is_some() {
                let vals = h3_vals.as_ref().unwrap();
                cols.push(Arc::new(arrow::array::UInt64Array::from_iter(
                    chunk.iter().map(|&r| vals[r as usize]),
                )));
            }
            if write_admin.is_some() {
                let vals = admin_vals.as_ref().unwrap();
                cols.push(Arc::new(arrow::array::StringArray::from_iter(
                    chunk.iter().map(|&r| vals.value(r)),
                )));
            }
            let out = RecordBatch::try_new(out_schema.clone(), cols)
                .map_err(|e| format!("batch assembly failed: {e}"))?;
            if open.len() == 1 {
                open[0].writer.write(&out).map_err(|e| format!("write failed: {e}"))?;
            } else {
                for w in open.iter_mut() {
                    let route: Vec<u32> = chunk
                        .iter()
                        .enumerate()
                        .filter(|&(_, &r)| row_part[r as usize] as usize == w.part)
                        .map(|(i, _)| i as u32)
                        .collect();
                    if route.is_empty() {
                        continue;
                    }
                    let sub = take_record_batch(&out, &UInt32Array::from(route))
                        .map_err(|e| format!("partition routing failed: {e}"))?;
                    w.writer.write(&sub).map_err(|e| format!("write failed: {e}"))?;
                }
            }
            written += chunk.len();
            fail_point(written)?;
            progress(0.45 + 0.55 * (written as f32 / rows.max(1) as f32), "writing");
            if at_level_end && cogp_bounds.is_some() {
                // COGP never partitions, so there is exactly one writer and
                // this is its level boundary. The row-group index is read
                // back from the writer rather than derived from row counts:
                // the byte cap can close a group anywhere inside a level,
                // and arithmetic that assumed otherwise would drift from
                // the file without ever saying so.
                let w = &mut open[0];
                w.writer
                    .flush()
                    .map_err(|e| format!("row group flush failed: {e}"))?;
                let end = w
                    .writer
                    .flushed_row_groups()
                    .len()
                    .checked_sub(1)
                    .ok_or("internal: COGP level closed with no row group")?;
                cogp_rg_ends.push(end);
            }
        }

        for mut w in open {
            if let Some(bounds) = &cogp_bounds {
                let levels: Vec<super::cogp::Level> = bounds
                    .iter()
                    .zip(&cogp_rg_ends)
                    .map(|(&(gsd, _), &row_group_end)| super::cogp::Level { row_group_end, gsd })
                    .collect();
                cogp_meta = Some(super::cogp::Cogp::new(levels));
                w.writer
                    .append_key_value_metadata(parquet::file::metadata::KeyValue::new(
                        super::cogp::KEY.to_string(),
                        cogp_meta.as_ref().unwrap().to_json(),
                    ));
            }
            let closed = w.writer.close().map_err(|e| format!("finalize failed: {e}"))?;
            if cogp_meta.is_some() {
                // Read back out of the closed footer rather than checked in
                // memory: the profile is only worth writing if it is true of
                // the file that came out, and that includes the key having
                // survived serialisation. Anything failing here is an
                // internal bug, and a silently non-conformant file is worse
                // than a failed export.
                let json = closed
                    .file_metadata()
                    .key_value_metadata()
                    .and_then(|kv| kv.iter().find(|k| k.key == super::cogp::KEY))
                    .and_then(|k| k.value.clone())
                    .ok_or("internal: the cogp key did not reach the file")?;
                super::cogp::Cogp::parse(&json)?.validate(closed.num_row_groups())?;
            }
            let part_order = &parts[w.part].1;
            let mut off = 0usize;
            for rg in closed.row_groups() {
                let n = rg.num_rows().max(0) as usize;
                rg_after_boxes.extend(union_bboxes(
                    part_order[off..off + n].iter().flat_map(|&r| &row_bboxes[r as usize]),
                ));
                off += n;
                rg_after += 1;
            }
        }
    }
    let out_paths = staged.commit_all()?;
    let size_after: u64 = out_paths
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    let overlap_after = super::loader::bbox_overlap_metric(&rg_after_boxes);
    log::info!(
        "optimize: {rows} rows, {} source groups, {} decodes, live peak {}",
        keep_groups.len(),
        gather.cache.decodes,
        super::info::fmt_bytes(gather.cache.live_high_water as u64)
    );
    record_gather_stats(&gather.cache);

    // Per-level feature counts and the row groups each level owns, so the
    // completion summary can show what the levels actually cost.
    let cogp_levels: Vec<CogpLevelReport> = match (&cogp_bounds, &cogp_meta) {
        (Some(bounds), Some(meta)) => {
            let mut start_row = 0usize;
            let mut rg_start = 0usize;
            let mut out = Vec::with_capacity(bounds.len());
            for (&(gsd, end_row), l) in bounds.iter().zip(&meta.levels) {
                out.push(CogpLevelReport {
                    gsd,
                    rows: (end_row - start_row) as u64,
                    rg_start,
                    rg_end: l.row_group_end,
                });
                start_row = end_row;
                rg_start = l.row_group_end + 1;
            }
            out
        }
        _ => Vec::new(),
    };
    if !cogp_levels.is_empty() {
        log::info!(
            "COGP: {} levels, {} row groups",
            cogp_levels.len(),
            rg_after
        );
    }

    Ok(OptimizeReport {
        rows: rows as u64,
        size_before: src.size(),
        size_after,
        rg_before,
        rg_after,
        overlap_before,
        overlap_after,
        bloom_columns,
        version_label: if aux_target.is_some() {
            format!("{} + GeoArrow column", opts.version.label())
        } else {
            opts.version.label().into()
        },
        elapsed_ms: t0.elapsed().as_millis() as u64,
        files: parts.len(),
        cogp_levels,
    })
}

/// How the geometry column is rewritten.
enum GeomOut {
    /// Values move with the row gather unchanged.
    PassThrough,
    /// Decode (WKB or GeoArrow) and re-encode as WKB bytes.
    ToWkb,
    /// Decode and rebuild as GeoArrow coordinate arrays.
    ToGa(GeomEncoding),
}

/// Re-encode one gathered geometry column.
fn transcode_geometry(
    col: &dyn arrow::array::Array,
    src_encoding: GeomEncoding,
    out: &GeomOut,
) -> Result<ArrayRef, String> {
    let geoms =
        GeomCol::new(col, src_encoding).ok_or("geometry column does not match its encoding")?;
    match out {
        GeomOut::PassThrough => unreachable!(),
        GeomOut::ToGa(target) => {
            let mut b = GaBuilder::new(*target);
            for i in 0..col.len() {
                b.push(geoms.geometry(i).as_ref())?;
            }
            Ok(b.finish())
        }
        GeomOut::ToWkb => {
            let opts = wkb::writer::WriteOptions::default();
            let mut buf: Vec<u8> = Vec::new();
            let values: Vec<Option<Vec<u8>>> = (0..col.len())
                .map(|i| -> Result<Option<Vec<u8>>, String> {
                    let Some(g) = geoms.geometry(i) else {
                        return Ok(None);
                    };
                    buf.clear();
                    wkb::writer::write_geometry(&mut buf, &g, &opts)
                        .map_err(|e| format!("WKB encode: {e}"))?;
                    Ok(Some(buf.clone()))
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(arrow::array::BinaryArray::from_iter(values)))
        }
    }
}

/// GeoParquet `geo` file metadata for the output.
#[allow(clippy::too_many_arguments)]
fn build_geo_meta(
    opts: &OptimizeOptions,
    primary: &str,
    crs: Option<&Value>,
    vendor_crs: Option<&Value>,
    crs_explicit_null: bool,
    geom_types: &HashSet<String>,
    out_encoding: GeomEncoding,
    file_bbox: Option<[f64; 4]>,
    covering_name: &str,
    // Not `opts.covering`: a COGP 1.1 export writes the column whether the
    // user ticked it or not, and the metadata has to describe the file that
    // was actually written.
    covering: bool,
) -> Value {
    // GeoArrow columns store exactly one type (singles promoted, Z dropped
    // by the coordinate rebuild).
    let types: Vec<&str> = if out_encoding.is_wkb() {
        let mut t: Vec<&str> = geom_types.iter().map(String::as_str).collect();
        t.sort_unstable();
        t
    } else {
        vec![out_encoding.geometry_type_name()]
    };
    let mut col = json!({
        "encoding": out_encoding.geo_name(),
        "geometry_types": types,
    });
    if let Some(c) = crs {
        col["crs"] = c.clone();
    } else if crs_explicit_null {
        // Unknown stays declared-unknown, never an implied CRS84.
        col["crs"] = Value::Null;
    }
    if let Some(v) = vendor_crs {
        col["geopq:crs"] = v.clone();
    }
    if let Some(b) = file_bbox {
        col["bbox"] = json!([b[0], b[1], b[2], b[3]]);
    }
    if covering {
        col["covering"] = json!({"bbox": {
            "xmin": [covering_name, "xmin"], "ymin": [covering_name, "ymin"],
            "xmax": [covering_name, "xmax"], "ymax": [covering_name, "ymax"],
        }});
    }
    json!({
        "version": match opts.version {
            GpVersion::V1_1 | GpVersion::V1_1GeoArrow => "1.1.0",
            GpVersion::V2_0 => "2.0.0",
        },
        "primary_column": primary,
        "columns": { primary: col },
    })
}

/// CRS string recorded in a GEOMETRY/GEOGRAPHY logical type (2.0 sources).
/// Looks the geometry column up by name: parquet leaf indices don't line up
/// with arrow root indices once nested columns exist.
fn logical_type_crs(
    meta: &parquet::file::metadata::ParquetMetaData,
    geom_name: &str,
) -> Option<Value> {
    use parquet::basic::LogicalType;
    let col = meta
        .row_groups()
        .first()?
        .columns()
        .iter()
        .find(|c| c.column_descr().path().parts().first().map(String::as_str) == Some(geom_name))?;
    let crs = match col.column_descr().logical_type_ref() {
        Some(LogicalType::Geometry { crs }) => crs.clone(),
        Some(LogicalType::Geography { crs, .. }) => crs.clone(),
        _ => None,
    }?;
    serde_json::from_str(&crs)
        .ok()
        .or(Some(Value::String(crs)))
}

fn guess_geom_column(schema: &Schema) -> Option<String> {
    ["geometry", "geom", "wkb_geometry", "wkb"]
        .iter()
        .find(|n| schema.index_of(n).is_ok())
        .map(|n| n.to_string())
        .or_else(|| {
            schema
                .fields()
                .iter()
                .find(|f| {
                    matches!(
                        f.data_type(),
                        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
                    )
                })
                .map(|f| f.name().clone())
        })
}

/// A covering-style bbox column: a struct with numeric
/// xmin/ymin/xmax/ymax fields.
fn is_covering_struct(dt: &DataType) -> bool {
    let DataType::Struct(fs) = dt else {
        return false;
    };
    ["xmin", "ymin", "xmax", "ymax"].iter().all(|n| {
        fs.iter().any(|f| {
            f.name() == n && matches!(f.data_type(), DataType::Float32 | DataType::Float64)
        })
    })
}

/// Does this WKB value carry Z coordinates? Header-only check — the
/// decoder drops Z, but pass-through outputs keep the original bytes, so
/// the reported geometry types need the " Z" suffix. Recognizes ISO codes
/// (1000s = Z, 3000s = ZM) and the EWKB 0x80000000 flag.
pub(crate) fn wkb_has_z(buf: &[u8]) -> bool {
    if buf.len() < 5 {
        return false;
    }
    let code = match buf[0] {
        1 => u32::from_le_bytes(buf[1..5].try_into().unwrap()),
        0 => u32::from_be_bytes(buf[1..5].try_into().unwrap()),
        _ => return false,
    };
    if code & 0x8000_0000 != 0 {
        return true; // EWKB Z flag
    }
    // Mask EWKB M/SRID flags before reading the ISO dimension digit.
    matches!((code & 0x0FFF_FFFF) / 1000, 1 | 3)
}

/// Append per-row bboxes (None for null/undecodable geometries).
fn scan_bboxes(
    col: &ArrayRef,
    encoding: GeomEncoding,
    out: &mut Vec<Option<[f64; 4]>>,
    geom_types: &mut HashSet<String>,
    // COGP needs the family of each individual feature, not just the set of
    // families in the file; it comes free from the type name already read.
    mut kinds: Option<&mut Vec<CogpKind>>,
) -> Result<(), String> {
    use geo::BoundingRect;
    let geoms = GeomCol::new(col.as_ref(), encoding)
        .ok_or("geometry column does not match its declared encoding")?;
    let insert_type = |set: &mut HashSet<String>, name: &str, z: bool| {
        let full = if z { format!("{name} Z") } else { name.to_string() };
        set.insert(full);
    };
    for i in 0..col.len() {
        if geoms.is_null(i) {
            out.push(None);
            // A null geometry has no family; it renders nothing and the
            // level assignment defers it on the missing bbox alone.
            if let Some(k) = kinds.as_deref_mut() {
                k.push(CogpKind::Point);
            }
            continue;
        }
        // GeoArrow arrays are 2D; only WKB values can carry Z.
        let z = match &geoms {
            GeomCol::Wkb(b) => b.value(i).is_some_and(wkb_has_z),
            GeomCol::Ga(_) => false,
        };
        if let Some((x, y)) = geoms.point2(i) {
            insert_type(geom_types, "Point", z);
            out.push((x.is_finite() && y.is_finite()).then_some([x, y, x, y]));
            if let Some(k) = kinds.as_deref_mut() {
                k.push(CogpKind::Point);
            }
            continue;
        }
        match geoms.geometry(i) {
            Some(geom) => {
                let name = geom_type_name(&geom);
                insert_type(geom_types, name, z);
                if let Some(k) = kinds.as_deref_mut() {
                    k.push(CogpKind::from_type_name(name));
                }
                out.push(geom.bounding_rect().map(|r| {
                    let (min, max) = (r.min(), r.max());
                    [min.x, min.y, max.x, max.y]
                }));
            }
            None => {
                out.push(None);
                if let Some(k) = kinds.as_deref_mut() {
                    k.push(CogpKind::Point);
                }
            }
        }
    }
    Ok(())
}

fn geom_type_name(g: &geo_types::Geometry<f64>) -> &'static str {
    use geo_types::Geometry::*;
    match g {
        Point(_) => "Point",
        Line(_) | LineString(_) => "LineString",
        Polygon(_) | Rect(_) | Triangle(_) => "Polygon",
        MultiPoint(_) => "MultiPoint",
        MultiLineString(_) => "MultiLineString",
        MultiPolygon(_) => "MultiPolygon",
        GeometryCollection(_) => "GeometryCollection",
    }
}

fn union_bboxes<'a>(boxes: impl Iterator<Item = &'a [f64; 4]>) -> Option<[f64; 4]> {
    let mut out: Option<[f64; 4]> = None;
    for b in boxes {
        out = Some(match out {
            None => *b,
            Some(a) => [
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ],
        });
    }
    out
}

/// Covering `bbox` struct column for the given (sorted) global row indices.
fn build_bbox_column(
    rows: &[u32],
    row_bboxes: &[Option<[f64; 4]>],
    bbox_fields: &Fields,
) -> ArrayRef {
    let get = |k: usize| -> ArrayRef {
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|&r| row_bboxes[r as usize].map(|b| b[k]))
                .collect::<Vec<_>>(),
        ))
    };
    let validity = NullBuffer::from(
        rows.iter()
            .map(|&r| row_bboxes[r as usize].is_some())
            .collect::<Vec<_>>(),
    );
    Arc::new(StructArray::new(
        bbox_fields.clone(),
        vec![get(0), get(1), get(2), get(3)],
        Some(validity),
    ))
}

/// Hilbert curve distance of cell (x, y) on a 2^order × 2^order grid.
fn hilbert_xy2d(order: u32, mut x: u32, mut y: u32) -> u64 {
    let mut d: u64 = 0;
    let mut s: u32 = 1 << (order - 1);
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        // Rotate the quadrant so the curve connects.
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, StringArray};
    use parquet::file::properties::WriterProperties;

    /// A source whose CRS exists only as the `geopq:crs` proj4 vendor
    /// key (ESRI .prj import without an EPSG authority): the rewrite
    /// must keep `crs: null` + the vendor key, and the reopened output
    /// must resolve to the same projected CRS — not silently claim
    /// CRS84 (the MassGIS statewide-parcels regression).
    #[test]
    fn vendor_proj4_crs_survives_optimize() {
        let dir = std::env::temp_dir().join("geopq_optimize_vendor_crs");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let p4 = "+proj=lcc +lat_1=41.71666666666667 +lat_2=42.68333333333333 \
                  +lat_0=41 +lon_0=-71.5 +x_0=200000 +y_0=750000 \
                  +a=6378137.0 +rf=298.257222101 +towgs84=0,0,0,0,0,0,0 +units=m";
        let geo = serde_json::json!({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": ["Point"],
                "crs": null,
                "geopq:crs": {"proj4": p4, "name": "Mass Mainland"},
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let wkbs: Vec<Vec<u8>> = (0..100)
            .map(|i| wkb_point(200_000.0 + i as f64 * 100.0, 890_000.0))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from((0..100i64).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let src = dir.join("mass.parquet");
        let mut w =
            ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let dst = dir.join("mass_optimized.parquet");
        let opts = OptimizeOptions::default();
        optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();

        let (_store, crs, info, _rg) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, None);
        assert!(!crs.is_latlong, "projected CRS, not an implied CRS84");
        assert!(crs.proj4.contains("+proj=lcc"), "{}", crs.proj4);
        assert!(crs.name.contains("Mass"), "{}", crs.name);
        // The metadata says unknown-but-carried, never nothing.
        let geo: serde_json::Value =
            serde_json::from_str(info.geo.raw_geo_json.as_ref().expect("geo metadata"))
                .unwrap();
        let col = &geo["columns"]["geometry"];
        assert!(col.get("crs").is_some_and(serde_json::Value::is_null));
        assert!(col.get("geopq:crs").is_some());
    }

    /// Partitioned exports: hive fields (incl. an admin join column),
    /// adaptive H3, and the H3 cell column — every partition file must be
    /// a loadable GeoParquet with the partition columns path-only.
    #[test]
    fn partitioned_export_roundtrips() {
        use crate::data::partition::{AdminJoinSpec, PartitionBy};
        let dir = std::env::temp_dir().join("geopq_optimize_partition");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Four spatial clusters, one per "quad" value (EPSG:2154 meters,
        // clusters ~500 km apart so H3 splits them cleanly).
        let quads = ["A", "B", "C", "D"];
        let centers = [
            (200_000.0, 6_200_000.0),
            (700_000.0, 6_200_000.0),
            (200_000.0, 6_700_000.0),
            (700_000.0, 6_700_000.0),
        ];
        let n_per = 1000usize;
        let (mut wkbs, mut ids, mut quad_vals) = (Vec::new(), Vec::new(), Vec::new());
        for (q, (cx, cy)) in quads.iter().zip(centers) {
            for i in 0..n_per {
                let (dx, dy) = ((i % 32) as f64 * 50.0, (i / 32) as f64 * 50.0);
                wkbs.push(wkb_point(cx + dx, cy + dy));
                ids.push((quad_vals.len()) as i64);
                quad_vals.push(q.to_string());
            }
        }
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": ["Point"],
                "crs": {"id": {"authority": "EPSG", "code": 2154}},
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("quad", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(quad_vals)),
            ],
        )
        .unwrap();
        let src = dir.join("clusters.parquet");
        let mut w =
            ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        // --- hive partition by the quad column ---
        let dst = dir.join("by_quad");
        let opts = OptimizeOptions {
            row_group_size: 2048,
            partition: PartitionBy::Fields(vec!["quad".into()]),
            ..Default::default()
        };
        let rep =
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(rep.files, 4);
        assert_eq!(rep.rows, 4 * n_per as u64);
        for q in quads {
            let f = dst.join(format!("quad={q}")).join("part-0.parquet");
            let (store, crs, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            assert_eq!(store.total_rows(), n_per as u64, "quad={q}");
            assert_eq!(crs.epsg, Some(2154));
            assert!(
                store.schema.index_of("quad").is_err(),
                "partition column must be path-only"
            );
            assert!(store.covering.is_some(), "covering survives partitioning");
        }

        // --- adaptive H3 + H3 column ---
        let dst2 = dir.join("adaptive");
        let opts2 = OptimizeOptions {
            row_group_size: 2048,
            h3_resolution: Some(7),
            partition: PartitionBy::AdaptiveH3 { target_rows: 1500, max_res: 10 },
            ..Default::default()
        };
        let rep2 =
            optimize(&Source::Local(src.clone()), &dst2, &opts2, None, None, &|_, _| {}).unwrap();
        assert!(rep2.files >= 4, "clusters must split: {} files", rep2.files);
        let mut total = 0u64;
        for entry in std::fs::read_dir(&dst2).unwrap() {
            let d = entry.unwrap().path();
            assert!(d.file_name().unwrap().to_str().unwrap().starts_with("h3="));
            let f = d.join("part-0.parquet");
            let (store, _, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            total += store.total_rows();
            let idx = store.schema.index_of("h3_r7").expect("h3 column present");
            assert_eq!(
                store.schema.field(idx).data_type(),
                &DataType::UInt64
            );
            // Values are valid H3 cells at res 7.
            let rows: Vec<u32> = (0..store.total_rows().min(5) as u32).collect();
            let b = store.fetch(&rows, Some(&[idx])).unwrap();
            let col = b[0]
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            for i in 0..col.len() {
                let cell = h3o::CellIndex::try_from(col.value(i)).expect("valid cell");
                assert_eq!(u8::from(cell.resolution()), 7);
            }
        }
        assert_eq!(total, 4 * n_per as u64);

        // --- admin join, partitioned by the joined column ---
        // Two boundary polygons splitting the clusters west/east.
        let mut poly_wkbs: Vec<Vec<u8>> = Vec::new();
        for (x0, x1) in [(0.0, 450_000.0), (450_000.0, 1_000_000.0)] {
            let poly = geo_types::Polygon::new(
                geo_types::LineString::from(vec![
                    (x0, 6_000_000.0),
                    (x1, 6_000_000.0),
                    (x1, 7_000_000.0),
                    (x0, 7_000_000.0),
                    (x0, 6_000_000.0),
                ]),
                vec![],
            );
            let mut buf = Vec::new();
            wkb::writer::write_geometry(
                &mut buf,
                &geo_types::Geometry::Polygon(poly),
                &wkb::writer::WriteOptions::default(),
            )
            .unwrap();
            poly_wkbs.push(buf);
        }
        let bschema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("zone", DataType::Utf8, false),
        ]));
        let bbatch = RecordBatch::try_new(
            bschema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(poly_wkbs.iter())),
                Arc::new(StringArray::from(vec!["west", "east"])),
            ],
        )
        .unwrap();
        let bounds = dir.join("zones.parquet");
        let mut w =
            ArrowWriter::try_new(File::create(&bounds).unwrap(), bschema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&bbatch).unwrap();
        w.close().unwrap();
        let (bstore, bcrs, _, _) = crate::data::loader::open_store_for_test(&bounds).unwrap();
        let admin = AdminJoinSpec {
            out_name: "zone".into(),
            store: Arc::new(bstore),
            value_column: "zone".into(),
            crs: bcrs,
        };

        let dst3 = dir.join("by_zone");
        let opts3 = OptimizeOptions {
            row_group_size: 2048,
            partition: PartitionBy::Fields(vec!["zone".into()]),
            ..Default::default()
        };
        let rep3 = optimize(
            &Source::Local(src.clone()),
            &dst3,
            &opts3,
            None,
            Some(&admin),
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(rep3.files, 2, "west/east");
        for (zone, expect) in [("west", 2 * n_per as u64), ("east", 2 * n_per as u64)] {
            let f = dst3.join(format!("zone={zone}")).join("part-0.parquet");
            let (store, _, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            assert_eq!(store.total_rows(), expect, "zone={zone}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run with a gather-pass decode budget of `bytes`, then restore the
    /// shipping one.
    fn with_budget<T>(bytes: usize, f: impl FnOnce() -> T) -> T {
        DECODE_BUDGET.with(|c| c.set(bytes));
        let out = f();
        DECODE_BUDGET.with(|c| c.set(DECODE_BUDGET_BYTES));
        out
    }

    /// (cache high-water bytes, source row-group decodes) of the last run.
    fn gather_stats() -> (usize, usize) {
        GATHER_STATS.with(std::cell::Cell::get)
    }

    /// A whole parquet file as one batch, for exact output comparison.
    fn read_all(path: &Path) -> RecordBatch {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .with_batch_size(1 << 20)
            .build()
            .unwrap();
        let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
        concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    fn assert_same_output(a: &Path, b: &Path) {
        let (x, y) = (read_all(a), read_all(b));
        assert_eq!(x.schema(), y.schema(), "schemas differ");
        assert_eq!(x.num_rows(), y.num_rows(), "row counts differ");
        for i in 0..x.num_columns() {
            assert_eq!(
                x.column(i),
                y.column(i),
                "column '{}' differs between runs",
                x.schema().field(i).name()
            );
        }
    }

    /// The gather pass must produce the same file whether the whole source
    /// stays resident or every row group has to be re-decoded. Equal row
    /// counts alone would also be the symptom of a gather that returned
    /// the right number of wrong rows, so this compares every column
    /// value, and the cache instrumentation proves the tight run really
    /// was under pressure rather than quietly resident.
    #[test]
    fn streaming_gather_equals_resident_gather() {
        let dir = std::env::temp_dir().join("geopq_optimize_stream");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let opts = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };

        let resident = dir.join("resident.parquet");
        optimize(&Source::Local(src.clone()), &resident, &opts, None, None, &|_, _| {}).unwrap();
        let (_, resident_decodes) = gather_stats();

        // Room for a couple of the 20 source groups: the cache has to
        // evict, and the Hilbert order walks back into evicted ones.
        let budget = 512 * 1024;
        let tight = dir.join("tight.parquet");
        let report = with_budget(budget, || {
            optimize(&Source::Local(src.clone()), &tight, &opts, None, None, &|_, _| {}).unwrap()
        });
        let (high_water, decodes) = gather_stats();
        assert!(
            high_water <= budget * 2,
            "held {high_water} B live against a {budget} B budget"
        );
        assert!(
            decodes > resident_decodes,
            "budget never bit: {decodes} decodes vs {resident_decodes} resident"
        );
        assert_eq!(report.rg_after, 40_000_usize.div_ceil(2048));
        assert_same_output(&resident, &tight);
        assert_rows_consistent(&tight);

        // Budget below one row group: nothing is cacheable and every
        // gather decodes exactly the rows it wants out of the source. What
        // stays live is then one chunk's worth of rows and nothing else.
        let starved = dir.join("starved.parquet");
        with_budget(4096, || {
            optimize(&Source::Local(src.clone()), &starved, &opts, None, None, &|_, _| {}).unwrap()
        });
        let (high_water, _) = gather_stats();
        assert!(
            high_water < 1 << 20,
            "no group fits, so only the gathered rows may be live: {high_water} B"
        );
        assert_same_output(&resident, &starved);
        assert_rows_consistent(&starved);

        // Same for the transcoding path, where the gathered chunk is
        // rebuilt rather than copied.
        let ga_opts = OptimizeOptions {
            version: GpVersion::V1_1GeoArrow,
            row_group_size: 2048,
            ..Default::default()
        };
        let ga_res = dir.join("ga_resident.parquet");
        let ga_tight = dir.join("ga_tight.parquet");
        optimize(&Source::Local(src.clone()), &ga_res, &ga_opts, None, None, &|_, _| {}).unwrap();
        with_budget(budget, || {
            optimize(&Source::Local(src.clone()), &ga_tight, &ga_opts, None, None, &|_, _| {})
                .unwrap()
        });
        assert_same_output(&ga_res, &ga_tight);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The refusal is sized on what a run will actually hold. A flat
    /// number would either turn away work that fits or wave through work
    /// that does not: a plain rewrite and an admin-joined, H3-partitioned
    /// one of the same layer are a factor apart.
    #[test]
    fn key_pass_estimate_tracks_the_options() {
        use crate::data::partition::PartitionBy;
        let plain = key_pass_bytes_per_row(&OptimizeOptions::default(), false, &[]);
        assert_eq!(plain, 56, "bbox + source row + sort position + hilbert key");
        let fields = vec!["state".to_string(), "county".to_string()];
        let heavy = key_pass_bytes_per_row(
            &OptimizeOptions {
                h3_resolution: Some(8),
                partition: PartitionBy::Fields(fields.clone()),
                ..Default::default()
            },
            true,
            &fields,
        );
        assert!(
            heavy > plain * 2,
            "derived columns must move the estimate: {heavy} vs {plain}"
        );
        // Nothing in it reads feature size, so the ceiling is a row count:
        // the million-vertex polygons of issue #12 cost what points cost.
        let ceiling = MAX_KEY_PASS_BYTES / plain;
        assert!(
            (50_000_000..150_000_000).contains(&ceiling),
            "a plain rewrite should clear tens of millions of rows, got {ceiling}"
        );
    }

    /// A group that cannot fit even an emptied cache must not empty it on
    /// the way to finding that out. Evicting first and testing after meant
    /// one oversized group flushed fifty small residents and then took the
    /// row path anyway, leaving every chunk after it to re-decode them.
    #[test]
    fn an_unfittable_group_does_not_flush_the_cache() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = |n: i64| {
            Arc::new(
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int64Array::from((0..n).collect::<Vec<_>>()))],
                )
                .unwrap(),
            )
        };
        let mut cache = GroupCache::new(64 * 1024);
        for gi in 0..4 {
            let b = batch(512);
            let size = b.get_array_memory_size();
            cache.admit(gi, b, size);
        }
        cache.release();
        let before = cache.bytes;
        assert!(before > 0, "cache should hold the four groups");

        // What the gather asks before it decides. It is an upper bound, so
        // an oversized group fails the test without a byte being dropped.
        assert!(cache.free_ceiling() <= cache.budget);
        assert_eq!(cache.bytes, before, "asking must not evict");
        // A request that can be met does evict, so the gate is not simply
        // refusing everything.
        assert!(cache.reserve(cache.budget) >= before);
        assert!(cache.bytes < before, "a satisfiable request evicts");
    }

    /// A row group with no rows is legal parquet and carries nothing.
    /// Keeping one would give two groups the same starting row, leaving
    /// the gatherer's binary search over that map without a defined
    /// answer, and would hand the key pass a group that decodes to no
    /// batch at all. (Arrow's own writer coalesces empty batches away, so
    /// the file that provokes this cannot be built through it — the guard
    /// is tested on the selection instead.)
    #[test]
    fn empty_row_groups_are_never_selected() {
        let rows = [0usize, 512, 0, 0, 300, 0];
        let kept = readable_groups((0..rows.len()).collect(), &rows);
        assert_eq!(kept, vec![1, 4]);
        // Which is what makes the row map a strictly increasing sequence,
        // the property the gatherer's binary search rests on.
        let mut starts = vec![0u64];
        for &g in &kept {
            starts.push(starts.last().unwrap() + rows[g] as u64);
        }
        assert!(
            starts.windows(2).all(|w| w[0] < w[1]),
            "row map must be strictly increasing: {starts:?}"
        );
        assert_eq!(readable_groups(vec![0, 2, 5], &rows), Vec::<usize>::new());
    }

    /// A scrambled source with many small row groups: Hilbert order makes
    /// every gather chunk reach into all of them at once. What the gather
    /// holds must still be its budget plus the chunk it is building —
    /// bounding only what *survives between* chunks would let this run
    /// hold all 78 groups live while reporting itself comfortably inside
    /// its budget, which is the shape of the crash in issue #12. The
    /// fixture is sized so that the unbudgeted set is an order of
    /// magnitude past the bound asserted here.
    #[test]
    fn gather_holds_only_its_budget() {
        let dir = std::env::temp_dir().join("geopq_optimize_live_bytes");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let n = 40_000usize;
        let side = 200usize;
        let stride = (n / 2 + 1) | 1;
        let (mut wkbs, mut ids, mut names) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..n {
            let i = (k * stride) % n;
            let (gx, gy) = (i % side, i / side);
            wkbs.push(wkb_point(gx as f64 * 0.05, gy as f64 * 0.05));
            ids.push(i as i64);
            names.push(format!("cell_{gx}_{gy}"));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();
        let src = dir.join("many_groups.parquet");
        // 512-row groups: 79 of them, so one 2048-row gather chunk pulls
        // from every group in the file.
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(512))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&src).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
        let groups = ParquetRecordBatchReaderBuilder::try_new(File::open(&src).unwrap())
            .unwrap()
            .metadata()
            .num_row_groups();
        assert!(groups > 70, "fixture needs many groups, got {groups}");

        // What one group costs decoded, so the claim below is in units of
        // the thing that would pile up.
        let budget = 256 * 1024;
        let dst = dir.join("out.parquet");
        let opts = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        with_budget(budget, || {
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap()
        });
        let (live, decodes) = gather_stats();
        assert!(
            decodes > groups,
            "budget never bit: {decodes} decodes for {groups} groups"
        );
        // Measured: with the in-flight set unbudgeted this reports around
        // 39 MB here, 150x the budget it claims to respect.
        assert!(
            live <= budget * 2,
            "gather held {live} B live against a {budget} B budget across \
             {groups} source groups"
        );
        assert_rows_consistent(&dst);

        // And the output is still the one the unbounded run produces.
        let free = dir.join("free.parquet");
        optimize(&Source::Local(src), &free, &opts, None, None, &|_, _| {}).unwrap();
        assert_same_output(&free, &dst);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No option combination falls back to holding the source: each one
    /// must give the same file with a budget that fits nothing as with one
    /// that fits everything. Run over the option matrix rather than the
    /// default, because it is the derived columns (transcode, covering,
    /// native stats, synthesized points) that read the gathered chunk.
    #[test]
    fn every_option_streams_identically() {
        let dir = std::env::temp_dir().join("geopq_optimize_stream_matrix");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(20_000, &dir);

        let cases: Vec<(&str, OptimizeOptions)> = vec![
            (
                "v2_aux",
                OptimizeOptions {
                    version: GpVersion::V2_0,
                    row_group_size: 1024,
                    geoarrow_aux: true,
                    bloom: BloomMode::AllAttributes,
                    ..Default::default()
                },
            ),
            (
                "geoarrow",
                OptimizeOptions {
                    version: GpVersion::V1_1GeoArrow,
                    row_group_size: 1024,
                    covering: false,
                    ..Default::default()
                },
            ),
            (
                "viewport",
                OptimizeOptions {
                    row_group_size: 1024,
                    filter_rect: Some([0.0, 0.0, 4.999, 4.999]),
                    ..Default::default()
                },
            ),
            (
                "h3",
                OptimizeOptions {
                    row_group_size: 1024,
                    h3_resolution: Some(8),
                    ..Default::default()
                },
            ),
            (
                "unsorted",
                OptimizeOptions {
                    row_group_size: 1024,
                    hilbert_sort: false,
                    ..Default::default()
                },
            ),
        ];
        for (label, opts) in cases {
            let a = dir.join(format!("{label}_resident.parquet"));
            let b = dir.join(format!("{label}_starved.parquet"));
            let ra =
                optimize(&Source::Local(src.clone()), &a, &opts, None, None, &|_, _| {}).unwrap();
            let rb = with_budget(4096, || {
                optimize(&Source::Local(src.clone()), &b, &opts, None, None, &|_, _| {}).unwrap()
            });
            assert_eq!(ra.rows, rb.rows, "{label}: row count");
            assert_eq!(ra.rg_after, rb.rg_after, "{label}: row groups");
            assert_same_output(&a, &b);
            assert_eq!(read_geo_meta(&a), read_geo_meta(&b), "{label}: geo metadata");
        }

        // An x/y source has no geometry column at all: the point column is
        // synthesized from the gathered coordinates in both passes.
        let n = 5_000usize;
        let stride = n / 2 + 1;
        let (mut xs, mut ys, mut ids) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..n {
            let i = (k * stride) % n;
            xs.push((i % 70) as f64 * 0.5);
            ys.push((i / 70) as f64 * 0.5);
            ids.push(i as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("lon", DataType::Float64, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(Int64Array::from(ids)),
            ],
        )
        .unwrap();
        let xy_src = dir.join("xy_src.parquet");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(512))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&xy_src).unwrap(), schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let xy_opts = OptimizeOptions {
            row_group_size: 1024,
            xy_geom: Some((0, 1)),
            ..Default::default()
        };
        let a = dir.join("xy_resident.parquet");
        let b = dir.join("xy_starved.parquet");
        optimize(&Source::Local(xy_src.clone()), &a, &xy_opts, None, None, &|_, _| {}).unwrap();
        with_budget(4096, || {
            optimize(&Source::Local(xy_src), &b, &xy_opts, None, None, &|_, _| {}).unwrap()
        });
        assert_same_output(&a, &b);
        let (store, _, _, _) = crate::data::loader::open_store_for_test(&a).unwrap();
        assert_eq!(store.total_rows(), n as u64);
        assert_eq!(store.encoding, GeomEncoding::Wkb, "synthesized WKB points");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Partitioned exports go through the same two passes: the plan comes
    /// from the key pass, the rows from the gather pass, and both the hive
    /// and adaptive-H3 routes must survive a budget that forces eviction.
    #[test]
    fn partitioned_export_streams() {
        use crate::data::partition::PartitionBy;
        let dir = std::env::temp_dir().join("geopq_optimize_stream_parts");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // 40 hive values: more partitions than writers stay open at once,
        // so the sorted order is swept more than once.
        let n = 20_000usize;
        let side = 200usize;
        let stride = (n / 2 + 1) | 1;
        let (mut wkbs, mut ids, mut zones) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..n {
            let i = (k * stride) % n;
            let (gx, gy) = (i % side, i / side);
            wkbs.push(wkb_point(
                2.0 + gx as f64 * 1e-3,
                48.0 + gy as f64 * 1e-3,
            ));
            ids.push(i as i64);
            zones.push(format!("z{:02}", i % 40));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("zone", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(zones)),
            ],
        )
        .unwrap();
        let src = dir.join("zones_src.parquet");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1024))
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&src).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {
                    "encoding": "WKB", "geometry_types": ["Point"],
                    "crs": {"id": {"authority": "OGC", "code": "CRS84"}},
                }},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let hive = OptimizeOptions {
            row_group_size: 512,
            partition: PartitionBy::Fields(vec!["zone".into()]),
            ..Default::default()
        };
        let dst = dir.join("by_zone");
        let report = with_budget(256 * 1024, || {
            optimize(&Source::Local(src.clone()), &dst, &hive, None, None, &|_, _| {}).unwrap()
        });
        assert_eq!(report.files, 40, "one file per zone across several sweeps");
        assert_eq!(report.rows, n as u64);
        // Sized from where the parts ended up, not from the staging tree
        // they were built in — a path the directory rename invalidates.
        let on_disk: u64 = walk_files(&dst)
            .iter()
            .map(|p| std::fs::metadata(p).unwrap().len())
            .sum();
        assert_eq!(report.size_after, on_disk, "report must size the published files");
        let mut seen = 0u64;
        for z in 0..40 {
            let f = dst.join(format!("zone=z{z:02}")).join("part-0.parquet");
            let (store, _, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            assert_eq!(store.total_rows(), (n / 40) as u64, "zone z{z:02}");
            assert!(
                store.schema.index_of("zone").is_err(),
                "partition column must stay path-only"
            );
            // Every id in this file really belongs to the zone: routing a
            // chunk to the wrong writer would keep the counts and move the
            // rows.
            let rows: Vec<u32> = (0..store.total_rows() as u32).collect();
            let b = store
                .fetch(&rows, Some(&[store.schema.index_of("id").unwrap()]))
                .unwrap();
            for batch in &b {
                let a = Int64Array::from(batch.column(0).to_data());
                for i in 0..batch.num_rows() {
                    assert_eq!(a.value(i) % 40, z as i64, "id {} in zone z{z:02}", a.value(i));
                }
                seen += batch.num_rows() as u64;
            }
        }
        assert_eq!(seen, n as u64);

        // Adaptive H3 needs global cell counts from the key pass before
        // the gather pass can route anything.
        let dst2 = dir.join("adaptive");
        let adaptive = OptimizeOptions {
            row_group_size: 512,
            h3_resolution: Some(9),
            partition: PartitionBy::AdaptiveH3 { target_rows: 3000, max_res: 11 },
            ..Default::default()
        };
        let report2 = with_budget(256 * 1024, || {
            optimize(&Source::Local(src.clone()), &dst2, &adaptive, None, None, &|_, _| {})
                .unwrap()
        });
        assert!(report2.files > 1, "clusters must split: {}", report2.files);
        let mut total = 0u64;
        for entry in std::fs::read_dir(&dst2).unwrap() {
            let d = entry.unwrap().path();
            let f = d.join("part-0.parquet");
            let (store, _, _, _) = crate::data::loader::open_store_for_test(&f).unwrap();
            let idx = store.schema.index_of("h3_r9").expect("h3 column present");
            let rows: Vec<u32> = (0..store.total_rows() as u32).collect();
            for b in store.fetch(&rows, Some(&[idx])).unwrap() {
                let col = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .unwrap();
                for i in 0..col.len() {
                    let cell = h3o::CellIndex::try_from(col.value(i)).expect("valid cell");
                    assert_eq!(u8::from(cell.resolution()), 9);
                }
            }
            total += store.total_rows();
        }
        assert_eq!(total, n as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every regular file under `root`, recursively (root itself if it is
    /// a file). Used to prove nothing is left behind.
    fn walk_files(root: &Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        if root.is_file() {
            out.push(root.to_path_buf());
            return out;
        }
        let Ok(rd) = std::fs::read_dir(root) else {
            return out;
        };
        for e in rd.flatten() {
            out.extend(walk_files(&e.path()));
        }
        out
    }

    fn partials_under(root: &Path) -> Vec<std::path::PathBuf> {
        walk_files(root)
            .into_iter()
            .filter(|p| p.to_string_lossy().ends_with(PARTIAL_SUFFIX))
            .collect()
    }

    /// An export that dies mid-write must leave nothing that looks like a
    /// dataset: no truncated `.parquet`, and an overwritten target still
    /// holding its previous bytes. The reported crash in issue #12 left a
    /// half-written file with a plausible name and no error at all.
    #[test]
    fn interrupted_write_leaves_no_plausible_output() {
        let dir = std::env::temp_dir().join("geopq_optimize_partial");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(20_000, &dir);
        let opts = OptimizeOptions {
            row_group_size: 1024,
            ..Default::default()
        };

        // The target already holds a file the user cares about.
        let dst = dir.join("out.parquet");
        std::fs::write(&dst, b"previous contents").unwrap();

        FAIL_AFTER_ROWS.with(|c| c.set(2048));
        let err =
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap_err();
        FAIL_AFTER_ROWS.with(|c| c.set(usize::MAX));
        assert!(err.contains("injected"), "{err}");
        assert_eq!(
            std::fs::read(&dst).unwrap(),
            b"previous contents",
            "an interrupted rewrite must not touch the target"
        );
        assert!(
            partials_under(&dir).is_empty(),
            "partials left: {:?}",
            partials_under(&dir)
        );

        // Partitioned: the same guarantee per part file.
        let pdst = dir.join("parted");
        let popts = OptimizeOptions {
            row_group_size: 1024,
            partition: super::super::partition::PartitionBy::AdaptiveH3 {
                target_rows: 4000,
                max_res: 8,
            },
            ..Default::default()
        };
        FAIL_AFTER_ROWS.with(|c| c.set(2048));
        let err = optimize(&Source::Local(src.clone()), &pdst, &popts, None, None, &|_, _| {})
            .unwrap_err();
        FAIL_AFTER_ROWS.with(|c| c.set(usize::MAX));
        assert!(err.contains("injected"), "{err}");
        let left = walk_files(&pdst);
        assert!(left.is_empty(), "interrupted partitioned export left {left:?}");

        // Success path: the output exists and no sibling survives it.
        optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert!(partials_under(&dir).is_empty());
        assert_eq!(
            crate::data::loader::open_store_for_test(&dst).unwrap().0.total_rows(),
            20_000,
            "the committed file is a readable parquet"
        );
        optimize(&Source::Local(src), &pdst, &popts, None, None, &|_, _| {}).unwrap();
        assert!(partials_under(&pdst).is_empty());
        assert!(
            walk_files(&pdst).iter().all(|p| p.extension().is_some_and(|e| e == "parquet")),
            "{:?}",
            walk_files(&pdst)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// More partitions than writers stay open at once, so the sorted order
    /// is swept more than once. A failure in a later sweep must leave
    /// nothing: committing sweep by sweep would answer it with a directory
    /// of complete-looking partitions missing the ones never written —
    /// readable, wrong, and nothing about it saying so.
    #[test]
    fn interrupted_later_sweep_commits_nothing() {
        use crate::data::partition::PartitionBy;
        let dir = std::env::temp_dir().join("geopq_optimize_sweep_partial");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // 40 zones: two sweeps at 32 writers each.
        let n = 8_000usize;
        let per_zone = n / 40;
        let (mut wkbs, mut ids, mut zones) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            wkbs.push(wkb_point((i % 100) as f64 * 0.01, (i / 100) as f64 * 0.01));
            ids.push(i as i64);
            zones.push(format!("z{:02}", i % 40));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("zone", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(zones)),
            ],
        )
        .unwrap();
        let src = dir.join("sweeps_src.parquet");
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let opts = OptimizeOptions {
            row_group_size: 256,
            partition: PartitionBy::Fields(vec!["zone".into()]),
            ..Default::default()
        };
        let dst = dir.join("by_zone");
        // The first sweep writes 32 of the 40 partitions, so failing just
        // past that lands in the second one.
        let first_sweep = MAX_OPEN_WRITERS * per_zone;
        FAIL_AFTER_ROWS.with(|c| c.set(first_sweep + per_zone / 2));
        let err =
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap_err();
        FAIL_AFTER_ROWS.with(|c| c.set(usize::MAX));
        assert!(err.contains("injected"), "{err}");
        assert!(
            walk_files(&dst).is_empty(),
            "a failed second sweep left {:?}",
            walk_files(&dst)
        );
        assert!(!dst.exists(), "and no hive skeleton either");
        assert!(partials_under(&dir).is_empty(), "nor a staging tree");

        // Failing at the publishing step itself: the dataset has been
        // written in full by then, and still nothing may appear.
        FAIL_AT_COMMIT.with(|c| c.set(0));
        let err =
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap_err();
        FAIL_AT_COMMIT.with(|c| c.set(usize::MAX));
        assert!(err.contains("injected"), "{err}");
        assert!(!dst.exists(), "a failed commit published {:?}", walk_files(&dst));
        assert!(partials_under(&dir).is_empty());

        // And one step further in: publishing is a *single* rename, so
        // there is no step 1 to fail at and the export completes. A commit
        // that renamed part by part would stop between two of them, and
        // this is the assertion that catches it — the dataset is all of it
        // or none of it, never the seventeen renames that succeeded.
        FAIL_AT_COMMIT.with(|c| c.set(1));
        let mid = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {});
        FAIL_AT_COMMIT.with(|c| c.set(usize::MAX));
        let published = walk_files(&dst).len();
        assert!(
            published == 0 || published == 40,
            "commit published {published} of 40 parts"
        );
        assert_eq!(mid.is_ok(), published == 40, "a failed commit must publish nothing");

        // The same export, uninterrupted, does produce all 40.
        let report =
            optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.files, 40);
        assert_eq!(walk_files(&dst).len(), 40);
        assert!(partials_under(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![1u8, 1, 0, 0, 0];
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    /// ISO WKB POINT Z (type code 1001).
    fn wkb_point_z(x: f64, y: f64, z: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&1001u32.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b.extend_from_slice(&z.to_le_bytes());
        b
    }

    fn read_geo_meta(path: &Path) -> Value {
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        let kv = b
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .iter()
            .find(|kv| kv.key == "geo")
            .expect("geo metadata");
        serde_json::from_str(kv.value.as_deref().unwrap()).unwrap()
    }

    #[test]
    fn wkb_z_header_detection() {
        assert!(!wkb_has_z(&wkb_point(1.0, 2.0)));
        assert!(wkb_has_z(&wkb_point_z(1.0, 2.0, 3.0))); // ISO Z
        // ISO ZM (3001).
        let mut zm = wkb_point_z(1.0, 2.0, 3.0);
        zm[1..5].copy_from_slice(&3001u32.to_le_bytes());
        assert!(wkb_has_z(&zm));
        // ISO M only (2001): no Z.
        let mut m = wkb_point_z(1.0, 2.0, 3.0);
        m[1..5].copy_from_slice(&2001u32.to_le_bytes());
        assert!(!wkb_has_z(&m));
        // EWKB Z flag (with SRID flag set too).
        let mut ewkb = wkb_point_z(1.0, 2.0, 3.0);
        ewkb[1..5].copy_from_slice(&(1u32 | 0x8000_0000 | 0x2000_0000).to_le_bytes());
        assert!(wkb_has_z(&ewkb));
        // EWKB SRID-only 2D point: no Z.
        let mut srid = wkb_point(1.0, 2.0);
        srid[1..5].copy_from_slice(&(1u32 | 0x2000_0000).to_le_bytes());
        assert!(!wkb_has_z(&srid));
    }

    /// A plain attribute that happens to be named `bbox` must survive the
    /// covering rewrite; the covering struct takes a non-colliding name
    /// that the geo metadata points at.
    #[test]
    fn plain_bbox_attribute_survives_covering() {
        let dir = std::env::temp_dir().join("geopq_optimize_bbox_attr");
        std::fs::create_dir_all(&dir).unwrap();
        let n = 4000usize;
        let stride = n / 2 + 1;
        let (mut wkbs, mut ids, mut bbox_attr) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..n {
            let i = (k * stride) % n;
            wkbs.push(wkb_point((i % 64) as f64, (i / 64) as f64));
            ids.push(i as i64);
            bbox_attr.push(i as f64 * 2.0);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("bbox", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(Float64Array::from(bbox_attr)),
            ],
        )
        .unwrap();
        let src = dir.join("bbox_attr_src.parquet");
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let dst = dir.join("bbox_attr_out.parquet");
        let opts = OptimizeOptions {
            row_group_size: 1024,
            ..Default::default() // covering on
        };
        optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();

        // Output keeps the Float64 attribute and adds the renamed struct.
        let geo = read_geo_meta(&dst);
        let covering = &geo["columns"]["geometry"]["covering"]["bbox"];
        assert_eq!(covering["xmin"][0], "bbox_1", "{covering}");
        let (store, _crs, _info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        let attr_idx = store.schema.index_of("bbox").expect("attribute kept");
        assert_eq!(store.schema.field(attr_idx).data_type(), &DataType::Float64);
        let struct_idx = store.schema.index_of("bbox_1").expect("covering struct");
        assert!(matches!(
            store.schema.field(struct_idx).data_type(),
            DataType::Struct(_)
        ));
        // The loader follows the metadata to the renamed covering column.
        assert!(store.covering.is_some(), "covering usable via metadata");
        let (source, _boxes) = rg_meta.expect("rg bboxes");
        assert!(source.contains("covering"), "{source}");
        // Attribute values stay attached to their row through the reorder.
        let rows: Vec<u32> = (0..n as u32).step_by(197).collect();
        let batches = store
            .fetch(&rows, Some(&[store.schema.index_of("id").unwrap(), attr_idx]))
            .unwrap();
        for b in &batches {
            let ids = Int64Array::from(b.column(0).to_data());
            let vals = Float64Array::from(b.column(1).to_data());
            for i in 0..b.num_rows() {
                assert_eq!(vals.value(i), ids.value(i) as f64 * 2.0);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pass-through WKB keeps Z coordinates, so the written geometry_types
    /// must carry the spec's " Z" suffix.
    #[test]
    fn z_wkb_types_get_suffix() {
        let dir = std::env::temp_dir().join("geopq_optimize_z");
        std::fs::create_dir_all(&dir).unwrap();
        let n = 500usize;
        let wkbs: Vec<Vec<u8>> =
            (0..n).map(|i| wkb_point_z((i % 25) as f64, (i / 25) as f64, i as f64)).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let src = dir.join("z_src.parquet");
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point Z"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        for version in [GpVersion::V1_1, GpVersion::V2_0] {
            let dst = dir.join(format!("z_out_{version:?}.parquet"));
            let opts = OptimizeOptions {
                version,
                row_group_size: 256,
                covering: version == GpVersion::V1_1,
                ..Default::default()
            };
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {})
                .unwrap();
            let geo = read_geo_meta(&dst);
            assert_eq!(
                geo["columns"]["geometry"]["geometry_types"],
                serde_json::json!(["Point Z"]),
                "{version:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn heavy_features_split_row_groups_by_bytes() {
        use arrow::array::BinaryArray;
        use arrow::datatypes::{DataType, Field, Schema};
        // 400 polygons of 2,000 vertices each: far under any row cap,
        // ~13 MB of geometry. This is the shape administrative
        // boundaries have, and the reason a row-only cap left them in
        // one indivisible row group.
        let dir = std::env::temp_dir().join(format!("geopq_rgbytes_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("heavy.parquet");
        let mut wkbs: Vec<Vec<u8>> = Vec::new();
        for i in 0..400usize {
            let cx = (i % 20) as f64;
            let cy = (i / 20) as f64;
            // Jittered, so the coordinates do not compress away to
            // nothing: the writer's byte budget counts encoded bytes,
            // and a smooth circle costs almost none of them.
            let mut seed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut rnd = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            let ring: Vec<(f64, f64)> = (0..2_000)
                .map(|k| {
                    let a = k as f64 / 2_000.0 * std::f64::consts::TAU;
                    let r = 0.3 + 0.2 * rnd();
                    (cx + r * a.cos(), cy + r * a.sin())
                })
                .collect();
            let mut b = vec![1u8];
            b.extend_from_slice(&3u32.to_le_bytes()); // Polygon
            b.extend_from_slice(&1u32.to_le_bytes()); // one ring
            b.extend_from_slice(&((ring.len() + 1) as u32).to_le_bytes());
            for (x, y) in ring.iter().chain(std::iter::once(&ring[0])) {
                b.extend_from_slice(&x.to_le_bytes());
                b.extend_from_slice(&y.to_le_bytes());
            }
            wkbs.push(b);
        }
        let schema = Arc::new(Schema::new(vec![Field::new(
            "geometry",
            DataType::Binary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BinaryArray::from_iter_values(wkbs.iter()))],
        )
        .unwrap();
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        // Row cap alone: everything lands in one group, which is exactly
        // the file the app used to produce.
        let one = dir.join("one.parquet");
        let opts = OptimizeOptions {
            row_group_size: 65_536,
            row_group_bytes: usize::MAX,
            ..Default::default()
        };
        optimize(&Source::Local(src.clone()), &one, &opts, None, None, &|_, _| {}).unwrap();
        let (_, _, info, _) = crate::data::loader::open_store_for_test(&one).unwrap();
        assert_eq!(info.row_groups, 1, "row cap never fires on 400 rows");
        let heavy = info.rg_bytes_max;
        assert!(heavy > 4 << 20, "fixture should be chunky, got {heavy} B");

        // Byte cap on: the same features split, and each group is under
        // the limit.
        let split = dir.join("split.parquet");
        let opts = OptimizeOptions {
            row_group_size: 65_536,
            row_group_bytes: 2 << 20,
            ..Default::default()
        };
        optimize(&Source::Local(src.clone()), &split, &opts, None, None, &|_, _| {}).unwrap();
        let (store, _, info, _) = crate::data::loader::open_store_for_test(&split).unwrap();
        assert!(
            info.row_groups > 2,
            "expected several groups, got {}",
            info.row_groups
        );
        assert!(
            info.rg_bytes_max < heavy,
            "largest group {} should be under the single-group {heavy}",
            info.rg_bytes_max
        );
        // No feature lost or duplicated in the split.
        assert_eq!(store.total_rows(), 400);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write a spatially-scrambled GeoParquet 1.0 file: `rows` points on a
    /// grid, visited in an order that destroys spatial locality, with a
    /// bloom filter on `name`. Returns (path, expected names by grid pos).
    fn write_scrambled(rows: usize, dir: &Path) -> std::path::PathBuf {
        let path = dir.join(format!("scrambled_{rows}.parquet"));
        let side = (rows as f64).sqrt().ceil() as usize;
        // A large stride coprime with `rows` scrambles spatial order.
        let stride = (rows / 2 + 1) | 1;
        let idx: Vec<usize> = (0..rows).map(|i| (i * stride) % rows).collect();
        let (mut wkbs, mut ids, mut names) = (Vec::new(), Vec::new(), Vec::new());
        for &i in &idx {
            let (gx, gy) = (i % side, i / side);
            let (x, y) = (gx as f64 / side as f64 * 10.0, gy as f64 / side as f64 * 10.0);
            wkbs.push(wkb_point(x, y));
            ids.push(i as i64);
            names.push(format!("cell_{gx}_{gy}"));
        }
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": ["Point"],
                "crs": {"id": {"authority": "EPSG", "code": 2154}},
            }},
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2048))
            .set_column_bloom_filter_enabled(ColumnPath::new(vec!["name".into()]), true)
            .build();
        let mut w =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
        path
    }

    fn bloom_paths(path: &Path) -> Vec<String> {
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        b.metadata().row_groups()[0]
            .columns()
            .iter()
            .filter(|c| c.bloom_filter_offset().is_some())
            .map(|c| c.column_descr().path().string())
            .collect()
    }

    /// Names must stay attached to their geometry after reordering.
    /// Names must stay attached to their geometry after reordering,
    /// whatever the geometry encoding.
    fn assert_rows_consistent(path: &std::path::PathBuf) {
        use geo::BoundingRect;
        let (store, _crs, _info, _rg) =
            crate::data::loader::open_store_for_test(path).unwrap();
        let rows: Vec<u32> = (0..store.total_rows() as u32).step_by(997).collect();
        let geoms = store.fetch_geoms(&rows).unwrap();
        let batches = store.fetch(&rows, None).unwrap();
        let mut names: Vec<String> = Vec::new();
        for batch in batches {
            let col = batch.column(batch.schema().index_of("name").unwrap()).clone();
            let arr = StringArray::from(col.to_data());
            names.extend((0..batch.num_rows()).map(|i| arr.value(i).to_string()));
        }
        assert_eq!(names.len(), geoms.len());
        for ((_, g), name) in geoms.iter().zip(&names) {
            let c = g.as_ref().unwrap().bounding_rect().unwrap().min();
            let side = 200usize; // matches write_scrambled(40_000)
            let gx = (c.x / 10.0 * side as f64).round() as usize;
            let gy = (c.y / 10.0 * side as f64).round() as usize;
            assert_eq!(name, &format!("cell_{gx}_{gy}"));
        }
    }

    #[test]
    fn v1_1_roundtrip_sorts_and_preserves() {
        let dir = std::env::temp_dir().join("geopq_optimize_v11");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_v11.parquet");

        let opts = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        assert_eq!(report.rg_after, 40_000_usize.div_ceil(2048));
        // Scrambled input: every row group spans the full extent. Sorted
        // output: near-disjoint boxes.
        assert!(
            report.overlap_after < report.overlap_before * 0.25,
            "overlap {} -> {}",
            report.overlap_before,
            report.overlap_after
        );
        assert_eq!(report.bloom_columns, vec!["name".to_string()]);
        assert_eq!(bloom_paths(&dst), vec!["name".to_string()]);

        // Our own loader must see a 1.1 covering file with EPSG:2154.
        let (store, crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        assert_eq!(store.total_rows(), 40_000);
        let (source, boxes) = rg_meta.expect("covering stats");
        assert!(source.contains("covering"), "{source}");
        assert_eq!(boxes.len(), report.rg_after);
        assert!(info.geo.version_label.contains("1.1"), "{}", info.geo.version_label);
        assert_rows_consistent(&dst);

        // The 1.1 flavour keeps the widely readable envelope: V1 data
        // pages, so anything that reads parquet at all can read it.
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&dst).unwrap()).unwrap();
        assert_eq!(b.metadata().file_metadata().version(), 1);
    }

    #[test]
    fn v2_0_writes_native_geometry_and_stats() {
        let dir = std::env::temp_dir().join("geopq_optimize_v20");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_v20.parquet");

        let opts = OptimizeOptions {
            version: GpVersion::V2_0,
            row_group_size: 2048,
            covering: false,
            bloom: BloomMode::AllAttributes,
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        let mut bloom = report.bloom_columns.clone();
        bloom.sort();
        assert_eq!(bloom, vec!["id".to_string(), "name".to_string()]);

        // Native GEOMETRY logical type with the CRS, plus geo statistics.
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&dst).unwrap()).unwrap();
        let meta = b.metadata();
        // The parquet envelope matches the flavour: a file announcing the
        // 2.0 geo spec must not report the legacy format version.
        assert_eq!(
            meta.file_metadata().version(),
            2,
            "GeoParquet 2.0 output must be written as Parquet 2.0"
        );
        let geom = meta.row_groups()[0]
            .columns()
            .iter()
            .find(|c| c.column_descr().name() == "geometry")
            .expect("geometry column");
        match geom.column_descr().logical_type_ref() {
            Some(parquet::basic::LogicalType::Geometry { crs }) => {
                let crs = crs.as_deref().expect("crs recorded");
                assert!(crs.contains("2154"), "{crs}");
            }
            other => panic!("expected GEOMETRY logical type, got {other:?}"),
        }
        for rg in meta.row_groups() {
            let col = rg
                .columns()
                .iter()
                .find(|c| c.column_descr().name() == "geometry")
                .unwrap();
            let stats = col.geo_statistics().expect("native geo statistics");
            let bb = stats.bounding_box().expect("bbox");
            assert!(bb.get_xmax() >= bb.get_xmin());
        }

        // Loader path: native stats detected as the rg-bbox source, strong
        // clustering, and the file info panel flags 2.0.
        let (_store, crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        let (source, boxes) = rg_meta.expect("native stats");
        assert!(source.contains("geospatial"), "{source}");
        let overlap = crate::data::loader::bbox_overlap_metric(&boxes);
        assert!(overlap < boxes.len() as f64 * 0.25, "clustered: {overlap}");
        assert!(info.geo.version_label.contains("2.0"), "{}", info.geo.version_label);
        assert_rows_consistent(&dst);

        // Downgrade path: re-optimizing a native-GEOMETRY file as 1.1 must
        // strip the logical type (plain WKB) while keeping the CRS.
        let dst11 = dir.join("out_v20_to_v11.parquet");
        let opts11 = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        optimize(&Source::Local(dst.clone()), &dst11, &opts11, None, None, &|_, _| {}).unwrap();
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&dst11).unwrap()).unwrap();
        let geom = b.metadata().row_groups()[0]
            .columns()
            .iter()
            .find(|c| c.column_descr().name() == "geometry")
            .unwrap();
        assert!(
            geom.column_descr().logical_type_ref().is_none(),
            "1.1 export must be plain WKB, got {:?}",
            geom.column_descr().logical_type_ref()
        );
        let (_s, crs11, info11, _rg) =
            crate::data::loader::open_store_for_test(&dst11).unwrap();
        assert_eq!(crs11.epsg, Some(2154));
        assert!(info11.geo.version_label.contains("1.1"), "{}", info11.geo.version_label);
    }

    /// 2.0 flavor: auxiliary GeoArrow column next to the native GEOMETRY
    /// primary. The file stays conformant 2.0 (sibling undeclared in geo
    /// metadata) and the reader adopts the sibling for decode.
    #[test]
    fn v2_0_geoarrow_aux_column_round_trip() {
        let dir = std::env::temp_dir().join("geopq_optimize_v20_aux");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_v20_aux.parquet");

        let opts = OptimizeOptions {
            version: GpVersion::V2_0,
            row_group_size: 2048,
            covering: true,
            geoarrow_aux: true,
            ..Default::default()
        };
        let report =
            optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        assert!(
            report.version_label.contains("GeoArrow column"),
            "{}",
            report.version_label
        );

        // Conformant 2.0: the primary keeps the GEOMETRY logical type, the
        // geo metadata declares only the primary, and the sibling is a
        // plain column tagged with the GeoArrow extension name.
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(&dst).unwrap()).unwrap();
        let geom = b.metadata().row_groups()[0]
            .columns()
            .iter()
            .find(|c| c.column_descr().name() == "geometry")
            .expect("primary");
        assert!(matches!(
            geom.column_descr().logical_type_ref(),
            Some(parquet::basic::LogicalType::Geometry { .. })
        ));
        let geo: serde_json::Value = serde_json::from_str(
            b.metadata()
                .file_metadata()
                .key_value_metadata()
                .unwrap()
                .iter()
                .find(|kv| kv.key == "geo")
                .unwrap()
                .value
                .as_deref()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(geo["version"], "2.0.0");
        assert!(geo["columns"].get("geometry_geoarrow").is_none());
        let aux = b.schema().field_with_name("geometry_geoarrow").expect("aux field");
        assert_eq!(
            aux.metadata().get("ARROW:extension:name").map(String::as_str),
            Some("geoarrow.point")
        );

        // Reader adopts the sibling: native encoding, WKB primary hidden.
        let (store, _crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.encoding, GeomEncoding::Point);
        assert_eq!(store.schema.field(store.geom_col).name(), "geometry_geoarrow");
        assert_eq!(
            store.hidden_wkb.map(|i| store.schema.field(i).name().clone()).as_deref(),
            Some("geometry")
        );
        assert!(info.geo.encoding.contains("GeoArrow column"), "{}", info.geo.encoding);
        assert!(info.quality.expect("quality").indexable);
        rg_meta.expect("rg bboxes");
        assert_rows_consistent(&dst);
    }

    /// Viewport-only export: only features intersecting the rect survive,
    /// rows stay consistent, and the metadata bbox shrinks to the export.
    #[test]
    fn viewport_only_export() {
        let dir = std::env::temp_dir().join("geopq_optimize_vp");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_vp.parquet");
        // Grid spans 0..10 in both axes; keep the lower-left quadrant.
        let rect = [0.0, 0.0, 4.999, 4.999];
        let opts = OptimizeOptions {
            row_group_size: 2048,
            filter_rect: Some(rect),
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        let quarter = 40_000 / 4;
        assert!(
            (report.rows as i64 - quarter as i64).unsigned_abs() < 500,
            "≈ one quadrant: {}",
            report.rows
        );
        let (store, _crs, info, _rg) = crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.total_rows(), report.rows);
        let b = info.geo.bbox.expect("metadata bbox");
        assert!(b[2] <= 5.1 && b[3] <= 5.1, "bbox shrank to the export: {b:?}");
        assert_rows_consistent(&dst);

        // The export must READ only what the viewport can touch, not read
        // the file and filter afterwards. Over http that difference is
        // the whole download, and it is why a viewport export of a remote
        // layer used to pull gigabytes for a handful of features.
        {
            use crate::data::net;
            // Serve a *sorted* source: row-group pruning is what makes a
            // viewport export cheap, and on scrambled input every group
            // holds a few of the wanted rows, so nothing can be skipped.
            let sorted = dir.join("sorted_src.parquet");
            optimize(
                &Source::Local(src.clone()),
                &sorted,
                &OptimizeOptions { row_group_size: 2048, ..Default::default() },
                None,
                None,
                &|_, _| {},
            )
            .unwrap();
            let server = crate::data::source::testserver::spawn(sorted.clone());
            let file_len = std::fs::metadata(&sorted).unwrap().len();
            let remote = Source::Remote { url: server.url.clone(), len: file_len };
            let before = net::for_source(&server.url).map_or(0, |(b, _)| b);
            let report_r = optimize(
                &remote,
                &dir.join("out_vp_remote.parquet"),
                &opts,
                None,
                None,
                &|_, _| {},
            )
            .unwrap();
            assert!(
                (report_r.rows as i64 - report.rows as i64).unsigned_abs() < 500,
                "same quadrant as the local export: {} vs {}",
                report_r.rows,
                report.rows
            );
            let read = net::for_source(&server.url).map_or(0, |(b, _)| b) - before;
            eprintln!(
                "viewport export read {read} of {file_len} B ({:.0}%)",
                read as f64 / file_len as f64 * 100.0
            );
            // A quadrant of a sorted file: most row groups never open.
            assert!(
                read < file_len / 2,
                "read {read} of {file_len} B for a quadrant"
            );
        }

        // Empty viewport errors instead of writing a rowless file.
        let err = optimize(
            &Source::Local(src),
            &dir.join("out_empty.parquet"),
            &OptimizeOptions {
                filter_rect: Some([1000.0, 1000.0, 1001.0, 1001.0]),
                ..Default::default()
            },
            None,
            None,
            &|_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("viewport"), "{err}");
    }

    /// GeoArrow points: WKB → coordinate arrays, loader reads them, and the
    /// x/y leaf statistics replace covering/geo stats as the pruning source.
    #[test]
    fn geoarrow_points_roundtrip() {
        let dir = std::env::temp_dir().join("geopq_optimize_ga_pts");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(40_000, &dir);
        let dst = dir.join("out_ga.parquet");
        let opts = OptimizeOptions {
            version: GpVersion::V1_1GeoArrow,
            row_group_size: 2048,
            covering: false, // force the coordinate-stats pruning source
            ..Default::default()
        };
        let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
        assert_eq!(report.rows, 40_000);
        assert!(
            report.overlap_frac_after() < 0.25,
            "sorted: {}",
            report.overlap_frac_after()
        );

        let (store, crs, info, rg_meta) = crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(crs.epsg, Some(2154));
        assert_eq!(store.encoding, GeomEncoding::Point);
        assert_eq!(info.geo.encoding, "point");
        let (source, boxes) = rg_meta.expect("rg bboxes from coordinate stats");
        assert!(source.contains("coordinate"), "{source}");
        assert_eq!(boxes.len(), report.rg_after);
        assert_rows_consistent(&dst);

        // Loader builds it: same rows, Point kind, and the mesh path used
        // the GeoArrow fast path (no WKB anywhere).
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (geom, rows, bad) =
            crate::data::loader::build_geometry_for_test(&store, &crs, &display).unwrap();
        assert_eq!((rows, bad), (40_000, 0));
        assert_eq!(geom.kind, crate::data::geometry::GeomKind::Point);

        // Attribute-panel path: a full-row fetch must expose the geometry
        // through the encoding-aware accessor (regression: the panel
        // assumed WKB and showed "<invalid>" on GeoArrow layers).
        let row = store.fetch_row(7).unwrap();
        let g = crate::data::geoarrow::GeomCol::new(
            row.column(store.geom_col).as_ref(),
            store.encoding,
        )
        .and_then(|g| g.geometry(0));
        assert!(
            matches!(g, Some(geo_types::Geometry::Point(_))),
            "geometry accessible from a full-row fetch"
        );

        // And back: GeoArrow → plain 1.1 WKB.
        let dst_back = dir.join("back_wkb.parquet");
        let opts_back = OptimizeOptions {
            row_group_size: 2048,
            ..Default::default()
        };
        optimize(&Source::Local(dst.clone()), &dst_back, &opts_back, None, None, &|_, _| {}).unwrap();
        let (store_b, _crs, info_b, _rg) =
            crate::data::loader::open_store_for_test(&dst_back).unwrap();
        assert_eq!(store_b.encoding, GeomEncoding::Wkb);
        assert!(info_b.geo.encoding.starts_with("WKB"), "{}", info_b.geo.encoding);
        assert_rows_consistent(&dst_back);
    }

    /// GeoArrow polygons with single→multi promotion: a WKB source mixing
    /// Polygon and MultiPolygon becomes one multipolygon-encoded column.
    #[test]
    fn geoarrow_polygon_promotion() {
        use geo_types::{polygon, Geometry, MultiPolygon};
        let dir = std::env::temp_dir().join("geopq_optimize_ga_poly");
        std::fs::create_dir_all(&dir).unwrap();

        // WKB source: squares on a grid, every 5th row a MultiPolygon.
        let n = 5000usize;
        let square = |x0: f64, y0: f64| {
            polygon![(x: x0, y: y0), (x: x0 + 1.0, y: y0), (x: x0 + 1.0, y: y0 + 1.0), (x: x0, y: y0 + 1.0), (x: x0, y: y0)]
        };
        let stride = n / 2 + 1;
        let (mut wkbs, mut ids) = (Vec::new(), Vec::new());
        let wopts = wkb::writer::WriteOptions::default();
        for k in 0..n {
            let i = (k * stride) % n;
            let (gx, gy) = ((i % 71) as f64 * 2.0 - 71.0, (i / 71) as f64 * 1.2);
            let g: Geometry<f64> = if i % 5 == 0 {
                MultiPolygon(vec![square(gx, gy), square(gx + 1.5, gy + 1.5)]).into()
            } else {
                square(gx, gy).into()
            };
            let mut buf = Vec::new();
            wkb::writer::write_geometry(&mut buf, &g, &wopts).unwrap();
            wkbs.push(buf);
            ids.push(i as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(wkbs.iter())),
                Arc::new(Int64Array::from(ids)),
            ],
        )
        .unwrap();
        let src = dir.join("poly_src.parquet");
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            serde_json::json!({
                "version": "1.0.0", "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon", "MultiPolygon"]}},
            })
            .to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let dst = dir.join("poly_ga.parquet");
        let opts = OptimizeOptions {
            version: GpVersion::V1_1GeoArrow,
            row_group_size: 1024,
            ..Default::default()
        };
        optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();

        let (store, _crs, info, rg_meta) =
            crate::data::loader::open_store_for_test(&dst).unwrap();
        assert_eq!(store.encoding, GeomEncoding::MultiPolygon);
        assert_eq!(info.geo.encoding, "multipolygon");
        assert_eq!(info.geo.geometry_types, vec!["MultiPolygon".to_string()]);
        // Covering written by default → per-feature selection still works.
        assert!(store.covering.is_some());
        let (source, _boxes) = rg_meta.unwrap();
        assert!(source.contains("covering"), "{source}");

        // Every geometry survives promotion: id encodes the grid slot, so
        // the bbox min corner must match the id-derived square origin.
        use geo::BoundingRect;
        let rows: Vec<u32> = (0..5000u32).step_by(97).collect();
        let geoms = store.fetch_geoms(&rows).unwrap();
        let batches = store.fetch(&rows, Some(&[1])).unwrap();
        let mut ids: Vec<i64> = Vec::new();
        for b in &batches {
            let a = Int64Array::from(b.column(0).to_data());
            ids.extend((0..b.num_rows()).map(|i| a.value(i)));
        }
        for ((_, g), id) in geoms.iter().zip(&ids) {
            let g = g.as_ref().unwrap();
            assert!(matches!(g, Geometry::MultiPolygon(_)), "promoted to multi");
            let min = g.bounding_rect().unwrap().min();
            let (gx, gy) = ((id % 71) as f64 * 2.0 - 71.0, (id / 71) as f64 * 1.2);
            assert_eq!((min.x, min.y), (gx, gy), "id {id}");
        }

        // Loader tessellates it end to end (bulk GeoArrow path)...
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let (geom, rows_n, bad) =
            crate::data::loader::build_geometry_for_test(&store, &_crs, &display).unwrap();
        assert_eq!((rows_n, bad), (5000, 0));
        assert_eq!(geom.kind, crate::data::geometry::GeomKind::Polygon);

        // ...and produces the same mesh as the per-feature WKB path on the
        // same shapes: identical tessellated area, segments and pick items.
        let mesh_stats = |g: &crate::data::layer::LayerGeometry| {
            let mut area = 0.0f64;
            let mut segs = 0usize;
            for c in g.chunks.iter() {
                for t in c.fill_indices.chunks_exact(3) {
                    let a = c.fill_vertices[t[0] as usize];
                    let b = c.fill_vertices[t[1] as usize];
                    let d = c.fill_vertices[t[2] as usize];
                    area += 0.5
                        * ((b[0] - a[0]) as f64 * (d[1] - a[1]) as f64
                            - (d[0] - a[0]) as f64 * (b[1] - a[1]) as f64)
                            .abs();
                }
                segs += c.lines[0].segments.len();
            }
            (area, segs, g.rtree.size())
        };
        let (src_store, src_crs, _i, _r) =
            crate::data::loader::open_store_for_test(&src).unwrap();
        assert_eq!(src_store.encoding, GeomEncoding::Wkb);
        let (geom_wkb, _rows, _bad) =
            crate::data::loader::build_geometry_for_test(&src_store, &src_crs, &display)
                .unwrap();
        let (a_ga, s_ga, r_ga) = mesh_stats(&geom);
        let (a_wkb, s_wkb, r_wkb) = mesh_stats(&geom_wkb);
        assert_eq!((s_ga, r_ga), (s_wkb, r_wkb), "segment / pick-item counts");
        assert!(
            (a_ga - a_wkb).abs() <= a_wkb * 1e-9,
            "tessellated area differs: {a_ga} vs {a_wkb}"
        );
    }

    /// The covering bbox leaves must be written in small pages so the page
    /// index (ColumnIndex/OffsetIndex) prunes below row-group granularity.
    #[test]
    fn bbox_leaves_get_fine_grained_pages() {
        let dir = std::env::temp_dir().join("geopq_optimize_pages");
        std::fs::create_dir_all(&dir).unwrap();
        let src = write_scrambled(150_000, &dir);
        let dst = dir.join("out_pages.parquet");
        let report =
            optimize(&Source::Local(src.clone()), &dst, &OptimizeOptions::default(), None, None, &|_, _| {})
                .unwrap();
        assert_eq!(report.rg_after, 150_000_usize.div_ceil(65_536));

        let options = parquet::arrow::arrow_reader::ArrowReaderOptions::new()
            .with_page_index_policy(parquet::file::metadata::PageIndexPolicy::Optional);
        let b = ParquetRecordBatchReaderBuilder::try_new_with_options(
            File::open(&dst).unwrap(),
            options,
        )
        .unwrap();
        let leaf_idx = |name: &str| {
            b.parquet_schema()
                .columns()
                .iter()
                .position(|c| c.path().string() == name)
                .unwrap_or_else(|| panic!("leaf {name} not found"))
        };
        let offset_index = b.metadata().offset_index().expect("offset index written");
        // First (full 65k-row) group: expect ~16 pages per bbox leaf
        // (65536 rows / ~4096 rows per 32 KB page), and far fewer for a
        // default-page-size column like the geometry.
        for leaf in ["bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax"] {
            let pages = offset_index[0][leaf_idx(leaf)].page_locations().len();
            assert!(pages >= 8, "{leaf}: {pages} pages, expected fine-grained");
        }
        // Sanity: page row ranges are recoverable via first_row_index.
        let locs = offset_index[0][leaf_idx("bbox.xmin")].page_locations();
        assert_eq!(locs[0].first_row_index, 0);
        assert!(locs[1].first_row_index > 0);
    }

    /// WKB vs GeoArrow decode speed on the real fixtures, opt-in:
    /// cargo test --release geoarrow_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn geoarrow_speed() {
        let dir = std::env::temp_dir().join("geopq_ga_speed");
        std::fs::create_dir_all(&dir).unwrap();
        let display = crate::data::crs::DisplayCrs::hobo_dyer();
        let bench = |label: &str, fixture: &str| {
            let src = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/"))
                .join(fixture);
            if !src.exists() {
                eprintln!("{label}: fixture missing, skipping");
                return;
            }
            // Matched pair: same codec, order, row groups, no covering —
            // the only difference is the geometry encoding.
            let base = OptimizeOptions {
                hilbert_sort: false,
                covering: false,
                ..Default::default()
            };
            let wkb = dir.join(format!("{label}_wkb.parquet"));
            let ga = dir.join(format!("{label}_ga.parquet"));
            optimize(&Source::Local(src.clone()), &wkb, &base, None, None, &|_, _| {}).unwrap();
            let opts = OptimizeOptions {
                version: GpVersion::V1_1GeoArrow,
                ..base
            };
            optimize(&Source::Local(src.clone()), &ga, &opts, None, None, &|_, _| {}).unwrap();
            let time = |path: &std::path::PathBuf| {
                let (store, crs, _i, _r) =
                    crate::data::loader::open_store_for_test(path).unwrap();
                let t = std::time::Instant::now();
                let (_g, rows, _b) =
                    crate::data::loader::build_geometry_for_test(&store, &crs, &display)
                        .unwrap();
                (t.elapsed().as_millis(), rows)
            };
            let size = |p: &std::path::PathBuf| std::fs::metadata(p).unwrap().len() / (1 << 20);
            let (wkb_ms, rows) = time(&wkb);
            let (ga_ms, rows2) = time(&ga);
            assert_eq!(rows, rows2);
            eprintln!(
                "{label}: {rows} rows — load: WKB {wkb_ms} ms vs GeoArrow {ga_ms} ms ({:.2}x) — size: {} MB vs {} MB",
                wkb_ms as f64 / ga_ms as f64,
                size(&wkb),
                size(&ga),
            );
        };
        bench("points_3m75", "points_3m75.parquet");
        bench("parcels", "parcels_hilbert.parquet");
    }

    /// Real-file benchmark, opt-in:
    /// GEOPQ_OPT_FILE=... cargo test --release optimize_real_file -- --ignored --nocapture
    #[test]
    #[ignore]
    fn optimize_real_file() {
        let Ok(src) = std::env::var("GEOPQ_OPT_FILE") else {
            return;
        };
        let src = std::path::PathBuf::from(src);
        let dst = std::env::temp_dir().join("geopq_opt_real.parquet");
        for version in [GpVersion::V1_1, GpVersion::V2_0] {
            let opts = OptimizeOptions {
                version,
                covering: version == GpVersion::V1_1,
                ..Default::default()
            };
            let report = optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();
            eprintln!("{report:#?}");
            eprintln!(
                "overlap fraction: {:.0}% -> {:.0}%",
                report.overlap_frac_before() * 100.0,
                report.overlap_frac_after() * 100.0
            );
            // Normalized: raw overlap counts aren't comparable when the
            // row-group count changes.
            assert!(report.overlap_frac_after() <= report.overlap_frac_before());
            let (_store, _crs, info, rg_meta) =
                crate::data::loader::open_store_for_test(&dst).unwrap();
            let (source, boxes) = rg_meta.expect("metadata bboxes on output");
            eprintln!(
                "loader sees: {} ({} boxes), version: {}",
                source,
                boxes.len(),
                info.geo.version_label
            );
        }
        let _ = std::fs::remove_file(&dst);
    }

    /// Exhaustive check on a small grid: xy2d must be a bijection and
    /// consecutive curve positions must be grid-adjacent (unit steps).
    #[test]
    fn hilbert_is_a_space_filling_curve() {
        const ORDER: u32 = 4;
        let n = 1u32 << ORDER;
        let mut cell_of = vec![None; (n * n) as usize];
        for x in 0..n {
            for y in 0..n {
                let d = hilbert_xy2d(ORDER, x, y);
                assert!(d < (n * n) as u64, "code {d} out of range");
                assert!(cell_of[d as usize].is_none(), "duplicate code {d}");
                cell_of[d as usize] = Some((x, y));
            }
        }
        for w in cell_of.windows(2) {
            let (x0, y0) = w[0].unwrap();
            let (x1, y1) = w[1].unwrap();
            let step = x0.abs_diff(x1) + y0.abs_diff(y1);
            assert_eq!(step, 1, "curve jumps from ({x0},{y0}) to ({x1},{y1})");
        }
    }

    /// GeoArrow is the recommended default whenever one geometry family
    /// fits; mixed, exotic or undeclared types fall back to WKB.
    #[test]
    fn preferred_version_prefers_geoarrow() {
        let types = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            GpVersion::preferred(&types(&["MultiPolygon"])),
            GpVersion::V1_1GeoArrow
        );
        assert_eq!(
            GpVersion::preferred(&types(&["Polygon", "MultiPolygon"])),
            GpVersion::V1_1GeoArrow,
            "singles promote into their multi variant"
        );
        assert_eq!(
            GpVersion::preferred(&types(&["Point", "Polygon"])),
            GpVersion::V1_1,
            "mixed families cannot be GeoArrow"
        );
        assert_eq!(
            GpVersion::preferred(&types(&["GeometryCollection"])),
            GpVersion::V1_1
        );
        assert_eq!(GpVersion::preferred(&[]), GpVersion::V1_1, "unknown types");
    }

    // --- COGP (Cloud Optimized GeoParquet Profile) ---

    fn wkb_line(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&2u32.to_le_bytes()); // LineString
        b.extend_from_slice(&2u32.to_le_bytes());
        for (x, y) in [(x0, y0), (x1, y1)] {
            b.extend_from_slice(&x.to_le_bytes());
            b.extend_from_slice(&y.to_le_bytes());
        }
        b
    }

    /// An axis-aligned square, `half` units either side of its centre.
    fn wkb_square(cx: f64, cy: f64, half: f64) -> Vec<u8> {
        let ring = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
            (cx - half, cy - half),
        ];
        let mut b = vec![1u8];
        b.extend_from_slice(&3u32.to_le_bytes()); // Polygon
        b.extend_from_slice(&1u32.to_le_bytes()); // one ring
        b.extend_from_slice(&(ring.len() as u32).to_le_bytes());
        for (x, y) in ring {
            b.extend_from_slice(&x.to_le_bytes());
            b.extend_from_slice(&y.to_le_bytes());
        }
        b
    }

    /// Write a GeoParquet 1.0 fixture in metres (EPSG:2154) from
    /// (wkb, id, rank) triples, unsorted.
    fn write_metric_fixture(path: &Path, rows: Vec<(Vec<u8>, i64, i64)>, types: &[&str]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("rank", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(rows.iter().map(|r| &r.0))),
                Arc::new(Int64Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>())),
                Arc::new(Int64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB", "geometry_types": types,
                // A projected CRS, so the metre thresholds apply directly.
                "crs": {"id": {"authority": "EPSG", "code": 2154}},
            }},
        });
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// The `cogp` metadata of a written file, plus its row-group count.
    fn read_cogp(path: &Path) -> (crate::data::cogp::Cogp, usize) {
        let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        let md = b.metadata().clone();
        let json = md
            .file_metadata()
            .key_value_metadata()
            .expect("file has key-value metadata")
            .iter()
            .find(|k| k.key == crate::data::cogp::KEY)
            .expect("cogp key present")
            .value
            .clone()
            .expect("cogp key has a value");
        (
            crate::data::cogp::Cogp::parse(&json).unwrap(),
            md.num_row_groups(),
        )
    }

    /// One decoded batch per row group, in file order.
    fn read_per_row_group(path: &Path) -> Vec<RecordBatch> {
        let n = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .metadata()
            .num_row_groups();
        (0..n)
            .map(|g| {
                let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
                    .unwrap()
                    .with_row_groups(vec![g])
                    .with_batch_size(1 << 20)
                    .build()
                    .unwrap();
                let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
                concat_batches(&batches[0].schema(), &batches).unwrap()
            })
            .collect()
    }

    /// One exported feature: the level it landed in, its id, its bbox and
    /// its geometry family.
    type CogpFeature = (usize, i64, [f64; 4], CogpKind);

    /// Every exported feature with the level it landed in, read back from
    /// the file alone — so it validates the output rather than repeating
    /// the writer's bookkeeping.
    fn cogp_features(path: &Path) -> (crate::data::cogp::Cogp, Vec<CogpFeature>) {
        let (meta, n_rg) = read_cogp(path);
        meta.validate(n_rg).expect("cogp metadata conforms");
        let mut out = Vec::new();
        for (gi, batch) in read_per_row_group(path).into_iter().enumerate() {
            let level = meta
                .levels
                .iter()
                .position(|l| gi <= l.row_group_end)
                .expect("every row group belongs to a level");
            let geom = batch.column(batch.schema().index_of("geometry").unwrap()).clone();
            let ids = batch
                .column(batch.schema().index_of("id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let (mut boxes, mut kinds, mut types) = (Vec::new(), Vec::new(), HashSet::new());
            scan_bboxes(&geom, GeomEncoding::Wkb, &mut boxes, &mut types, Some(&mut kinds)).unwrap();
            for i in 0..batch.num_rows() {
                out.push((level, ids.value(i), boxes[i].unwrap(), kinds[i]));
            }
        }
        (meta, out)
    }

    /// Points at three densities, lines and polygons at three sizes, all in
    /// metres. Sized against gsds [10000, 1000, 100] with the default
    /// factors so each extended feature has an unambiguous level.
    fn cogp_mixed_fixture(path: &Path) -> usize {
        let mut rows: Vec<(Vec<u8>, i64, i64)> = Vec::new();
        let mut id = 0i64;
        let mut push = |rows: &mut Vec<_>, wkb: Vec<u8>, rank: i64| {
            rows.push((wkb, id, rank));
            id += 1;
        };
        // Polygons: diagonal 84853 m (level 0), 4243 m (level 1), 14 m (last).
        for k in 0..3 {
            push(&mut rows, wkb_square(k as f64 * 120_000.0, 0.0, 30_000.0), 0);
        }
        for k in 0..4 {
            push(&mut rows, wkb_square(k as f64 * 9_000.0, 200_000.0, 1_500.0), 0);
        }
        for k in 0..5 {
            push(&mut rows, wkb_square(k as f64 * 300.0, 400_000.0, 5.0), 0);
        }
        // Lines: 50000 m (level 0), 3000 m (level 1), 50 m (last).
        for k in 0..3 {
            let y = 600_000.0 + k as f64 * 1_000.0;
            push(&mut rows, wkb_line(0.0, y, 50_000.0, y), 0);
        }
        for k in 0..4 {
            let y = 700_000.0 + k as f64 * 5_000.0;
            push(&mut rows, wkb_line(0.0, y, 3_000.0, y), 0);
        }
        for k in 0..5 {
            let y = 800_000.0 + k as f64 * 200.0;
            push(&mut rows, wkb_line(0.0, y, 50.0, y), 0);
        }
        // Points: a sparse 100 km spread plus two dense clusters.
        for k in 0..40 {
            let (gx, gy) = (k % 8, k / 8);
            push(
                &mut rows,
                wkb_point(gx as f64 * 12_500.0, 900_000.0 + gy as f64 * 12_500.0),
                k as i64,
            );
        }
        for k in 0..60 {
            push(
                &mut rows,
                wkb_point(1_000_000.0 + k as f64 * 30.0, 1_000_000.0),
                k as i64,
            );
        }
        let n = rows.len();
        // Scramble, so nothing passes by accident of input order.
        let stride = n / 2 + 1;
        let scrambled: Vec<_> = (0..n).map(|i| rows[(i * stride) % n].clone()).collect();
        write_metric_fixture(
            path,
            scrambled,
            &["Point", "LineString", "Polygon"],
        );
        n
    }

    fn cogp_opts(gsds: &[f64]) -> CogpOptions {
        CogpOptions {
            gsd: GsdSource::Explicit(gsds.to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn cogp_levels_validate_and_place_every_feature() {
        let dir = std::env::temp_dir().join(format!("geopq_cogp_mixed_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mixed.parquet");
        let n = cogp_mixed_fixture(&src);
        let gsds = [10_000.0, 1_000.0, 100.0];

        // Both flavours COGP is offered on: 1.1 with the covering column the
        // profile requires, and 2.0 leaning on native geospatial statistics.
        for (name, version, covering) in [
            ("v11", GpVersion::V1_1, true),
            ("v20", GpVersion::V2_0, false),
        ] {
            let dst = dir.join(format!("{name}.parquet"));
            let opts = OptimizeOptions {
                version,
                covering,
                // Small groups, so every level owns several of them and the
                // boundary logic is actually exercised.
                row_group_size: 16,
                cogp: Some(cogp_opts(&gsds)),
                ..Default::default()
            };
            let rep =
                optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();

            let (meta, feats) = cogp_features(&dst);
            let (_, n_rg) = read_cogp(&dst);
            assert_eq!(n_rg, rep.rg_after, "{name}: report agrees with the file");
            assert_eq!(
                meta.levels.last().unwrap().row_group_end,
                n_rg - 1,
                "{name}: the last level owns the last row group"
            );

            // Every input row, exactly once.
            let mut ids: Vec<i64> = feats.iter().map(|f| f.1).collect();
            ids.sort_unstable();
            assert_eq!(ids, (0..n as i64).collect::<Vec<_>>(), "{name}: rows");

            // COGP 1.1 must declare the covering; 2.0 is this app's
            // extension and relies on the native statistics instead.
            let geo = read_geo_meta(&dst);
            let col = &geo["columns"]["geometry"];
            if version == GpVersion::V1_1 {
                assert!(!col["covering"].is_null(), "{name}: covering declared");
            }

            // Semantics, checked from the output alone: every extended
            // feature is renderable at its own level's gsd, and was not
            // renderable at the previous one (or it would have been placed
            // there). Together that is "one level per row group", stated in
            // terms a reader can verify.
            for &(level, id, b, kind) in &feats {
                if kind == CogpKind::Point {
                    continue;
                }
                let factor = if kind == CogpKind::Line { 2.0 } else { 4.0 };
                let diag = ((b[2] - b[0]).powi(2) + (b[3] - b[1]).powi(2)).sqrt();
                let gsd = meta.levels[level].gsd;
                if level + 1 < meta.levels.len() {
                    assert!(
                        diag >= factor * gsd - 1e-6,
                        "{name}: feature {id} ({diag} m) is below level {level}'s {gsd} m gsd"
                    );
                }
                if level > 0 {
                    let coarser = meta.levels[level - 1].gsd;
                    assert!(
                        diag < factor * coarser,
                        "{name}: feature {id} ({diag} m) belonged in level {}",
                        level - 1
                    );
                }
            }

            // Levels appear coarse to fine in the file and the report says
            // the same thing.
            assert_eq!(rep.cogp_levels.len(), meta.levels.len(), "{name}: report");
            let total: u64 = rep.cogp_levels.iter().map(|l| l.rows).sum();
            assert_eq!(total, n as u64, "{name}: report row counts");
            for (i, l) in rep.cogp_levels.iter().enumerate() {
                assert_eq!(l.rg_end, meta.levels[i].row_group_end);
                assert!(l.rg_start <= l.rg_end && l.rows > 0);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cogp_puts_the_big_polygon_first_and_the_tiny_one_last() {
        let dir = std::env::temp_dir().join(format!("geopq_cogp_size_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mixed.parquet");
        cogp_mixed_fixture(&src);
        let dst = dir.join("out.parquet");
        let opts = OptimizeOptions {
            row_group_size: 16,
            cogp: Some(cogp_opts(&[10_000.0, 1_000.0, 100.0])),
            ..Default::default()
        };
        optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();
        let (meta, feats) = cogp_features(&dst);
        assert_eq!(meta.levels.len(), 3, "all three levels received features");
        let level_of = |id: i64| feats.iter().find(|f| f.1 == id).unwrap().0;
        // id 0: a 60 km square. id 7: a 10 m one.
        assert_eq!(level_of(0), 0, "the 60 km polygon is in the coarse level");
        assert_eq!(level_of(7), 2, "the 10 m polygon is in the finest level");
        // id 12: a 50 km line. id 19: a 50 m one.
        assert_eq!(level_of(12), 0);
        assert_eq!(level_of(19), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cogp_thins_points_by_cell_and_honours_the_rank_column() {
        let dir = std::env::temp_dir().join(format!("geopq_cogp_thin_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("points.parquet");
        // Four clusters of five points each, every cluster inside one
        // 1000 m cell and the clusters far apart. Ranks ascend within a
        // cluster, so the winner is knowable: id 4, 9, 14, 19.
        let mut rows: Vec<(Vec<u8>, i64, i64)> = Vec::new();
        for c in 0..4 {
            for k in 0..5 {
                let id = (c * 5 + k) as i64;
                rows.push((
                    wkb_point(c as f64 * 50_000.0 + k as f64 * 10.0, 0.0),
                    id,
                    k as i64,
                ));
            }
        }
        write_metric_fixture(&src, rows, &["Point"]);

        let dst = dir.join("out.parquet");
        let opts = OptimizeOptions {
            row_group_size: 4,
            cogp: Some(CogpOptions {
                gsd: GsdSource::Explicit(vec![1_000.0, 100.0]),
                // One winner per gsd-sized cell, no extra coarsening.
                point_factor: 1,
                rank: Some(("rank".to_string(), RankOrder::Desc)),
                ..Default::default()
            }),
            ..Default::default()
        };
        optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();
        let (_, feats) = cogp_features(&dst);

        let coarse: Vec<i64> = feats
            .iter()
            .filter(|f| f.0 == 0)
            .map(|f| f.1)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(
            coarse,
            vec![4, 9, 14, 19],
            "one winner per 1000 m cell, the highest rank in each"
        );
        // And no two coarse points share a cell.
        let cells: HashSet<(i64, i64)> = feats
            .iter()
            .filter(|f| f.0 == 0)
            .map(|f| CogpUnits::Linear(1.0).cell(&f.2, 1_000.0))
            .collect();
        assert_eq!(cells.len(), coarse.len(), "one point per cell per level");
        assert_eq!(feats.len(), 20, "the other sixteen deferred, none lost");

        // Ascending order flips which end of the column wins.
        let asc = dir.join("asc.parquet");
        let opts = OptimizeOptions {
            row_group_size: 4,
            cogp: Some(CogpOptions {
                gsd: GsdSource::Explicit(vec![1_000.0, 100.0]),
                point_factor: 1,
                rank: Some(("rank".to_string(), RankOrder::Asc)),
                ..Default::default()
            }),
            ..Default::default()
        };
        let src2 = dir.join("points.parquet");
        optimize(&Source::Local(src2), &asc, &opts, None, None, &|_, _| {}).unwrap();
        let (_, feats) = cogp_features(&asc);
        let coarse: Vec<i64> = feats
            .iter()
            .filter(|f| f.0 == 0)
            .map(|f| f.1)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(coarse, vec![0, 5, 10, 15], "smallest rank wins instead");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cogp_drops_levels_that_received_nothing() {
        let dir = std::env::temp_dir().join(format!("geopq_cogp_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("tiny.parquet");
        // Nothing here clears even the finest gsd, so every feature falls
        // through to the last level and the three coarse ones stay empty.
        let rows: Vec<(Vec<u8>, i64, i64)> = (0..40)
            .map(|k| (wkb_square(k as f64 * 10.0, 0.0, 1.0), k as i64, 0))
            .collect();
        write_metric_fixture(&src, rows, &["Polygon"]);
        let dst = dir.join("out.parquet");
        let opts = OptimizeOptions {
            row_group_size: 8,
            cogp: Some(cogp_opts(&[1e6, 1e5, 1e4, 1e3])),
            ..Default::default()
        };
        optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();
        let (meta, n_rg) = read_cogp(&dst);
        meta.validate(n_rg).unwrap();
        assert_eq!(meta.levels.len(), 1, "the three empty levels collapsed");
        assert_eq!(meta.levels[0].gsd, 1e3);
        assert_eq!(meta.levels[0].row_group_end, n_rg - 1);
        assert!(n_rg > 1, "the fixture should span several row groups");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cogp_refuses_the_combinations_it_cannot_honour() {
        let dir = std::env::temp_dir().join(format!("geopq_cogp_refuse_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mixed.parquet");
        cogp_mixed_fixture(&src);
        let dst = dir.join("out.parquet");

        let partitioned = OptimizeOptions {
            partition: crate::data::partition::PartitionBy::AdaptiveH3 {
                target_rows: 10,
                max_res: 6,
            },
            cogp: Some(cogp_opts(&[1_000.0, 100.0])),
            ..Default::default()
        };
        let err = optimize(
            &Source::Local(src.clone()),
            &dst,
            &partitioned,
            None,
            None,
            &|_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("partitioned"), "got {err}");

        let geoarrow = OptimizeOptions {
            version: GpVersion::V1_1GeoArrow,
            cogp: Some(cogp_opts(&[1_000.0, 100.0])),
            ..Default::default()
        };
        let err = optimize(&Source::Local(src.clone()), &dst, &geoarrow, None, None, &|_, _| {})
            .unwrap_err();
        assert!(err.contains("GeoArrow"), "got {err}");

        // A gsd list that is not strictly decreasing is caught before any
        // file is touched.
        let bad = OptimizeOptions {
            cogp: Some(cogp_opts(&[100.0, 1_000.0])),
            ..Default::default()
        };
        let err = optimize(&Source::Local(src), &dst, &bad, None, None, &|_, _| {}).unwrap_err();
        assert!(err.contains("strictly decrease"), "got {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn web_mercator_gsds_halve_per_zoom() {
        let g = CogpOptions {
            gsd: GsdSource::WebMercator {
                minzoom: 0,
                maxzoom: 3,
                resolution: 1024,
            },
            ..Default::default()
        }
        .gsds()
        .unwrap();
        assert_eq!(g.len(), 4);
        assert!((g[0] - 40_075_016.685_578_49 / 1024.0).abs() < 1e-6);
        for w in g.windows(2) {
            assert!((w[0] / w[1] - 2.0).abs() < 1e-9, "each zoom halves the gsd");
        }
    }

    /// Write a GeoParquet 1.0 fixture in degrees from (wkb, id, rank)
    /// triples. `crs` is omitted, which per the spec means OGC:CRS84 — the
    /// case a geographic layer usually arrives in.
    fn write_wgs84_fixture(path: &Path, rows: Vec<(Vec<u8>, i64, i64)>, types: &[&str]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("rank", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(rows.iter().map(|r| &r.0))),
                Arc::new(Int64Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>())),
                Arc::new(Int64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let geo = serde_json::json!({
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "geometry_types": types}},
        });
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// The mixed fixture in degrees near the equator: points at three
    /// densities, lines and squares at three sizes, each sized in metres
    /// against gsds [10000, 1000, 100] so all three levels receive
    /// something. Degrees make the writer convert and the reader's
    /// `view_gsd` convert back, which is the half of the round trip that
    /// a metric fixture cannot exercise.
    fn cogp_geographic_fixture(path: &Path) -> usize {
        // Sizes below are degrees at the equator, where one degree is
        // ~111 km; the writer converts them the same way.
        let mut rows: Vec<(Vec<u8>, i64, i64)> = Vec::new();
        let mut id = 0i64;
        let mut push = |rows: &mut Vec<_>, wkb: Vec<u8>| {
            rows.push((wkb, id, id));
            id += 1;
        };
        // Squares: diagonals of ~94 km, ~9.4 km and ~0.9 km, which at
        // polygon factor 4 land in levels 0, 1 and 2 respectively.
        for (i, half) in [0.3, 0.03, 0.003].iter().enumerate() {
            push(&mut rows, wkb_square(i as f64 * 2.0, 0.0, *half));
        }
        // Lines: ~56 km, ~5.6 km and ~0.6 km long, factor 2.
        for (i, len) in [0.5, 0.05, 0.005].iter().enumerate() {
            let x = 10.0 + i as f64 * 2.0;
            push(&mut rows, wkb_line(x, 1.0, x + len, 1.0));
        }
        // Points, three densities: a degree apart (one per 40 km cell),
        // ~2 km apart, and ~55 m apart.
        for k in 0..9 {
            push(&mut rows, wkb_point(k as f64, 5.0));
        }
        for k in 0..12 {
            push(&mut rows, wkb_point(20.0 + k as f64 * 0.02, 0.0));
        }
        for k in 0..24 {
            push(&mut rows, wkb_point(30.0 + k as f64 * 0.000_5, 0.0));
        }
        let n = rows.len();
        // Scramble, so nothing passes by accident of input order.
        let stride = n / 2 + 1;
        let scrambled: Vec<_> = (0..n).map(|i| rows[(i * stride) % n].clone()).collect();
        write_wgs84_fixture(path, scrambled, &["Point", "LineString", "Polygon"]);
        n
    }

    /// The writer stamps `cogp`, the reader parses it: the two halves have
    /// to agree about the same bytes, in both flavours the profile is
    /// offered on.
    #[test]
    fn cogp_round_trips_from_writer_to_reader() {
        use crate::data::cogp::Pruning;

        let dir = std::env::temp_dir().join(format!("geopq_cogp_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("mixed_wgs84.parquet");
        cogp_geographic_fixture(&src);
        let gsds = [10_000.0, 1_000.0, 100.0];

        // 1.1 carries the covering column the published profile requires;
        // 2.0 with the covering opt-in off leans on native geospatial
        // statistics instead, which is this app's labelled extension.
        for (name, version, covering, want) in [
            ("v11", GpVersion::V1_1, true, Pruning::Covering),
            ("v20", GpVersion::V2_0, false, Pruning::NativeStats),
        ] {
            let dst = dir.join(format!("rt_{name}.parquet"));
            let opts = OptimizeOptions {
                version,
                covering,
                // Small groups, so each level owns several of them.
                row_group_size: 16,
                cogp: Some(cogp_opts(&gsds)),
                ..Default::default()
            };
            let rep =
                optimize(&Source::Local(src.clone()), &dst, &opts, None, None, &|_, _| {}).unwrap();

            let (store, _crs, info, _rg) =
                crate::data::loader::open_store_for_test(&dst).unwrap();

            // (1) The reader recognises the profile and names the
            // statistics this flavour actually offers.
            let levels = store
                .cogp
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: the reader found no COGP levels"));
            assert_eq!(levels.pruning, want, "{name}");
            assert_eq!(levels.version, crate::data::cogp::VERSION, "{name}");
            let summary = info.geo.cogp.as_deref().expect("summary line");
            assert_eq!(
                summary.contains("(2.0 extension)"),
                want == Pruning::NativeStats,
                "{name}: {summary}"
            );

            // (2) Level for level, the numbers the writer reported.
            assert_eq!(levels.levels.len(), gsds.len(), "{name}: every gsd used");
            assert_eq!(levels.levels.len(), rep.cogp_levels.len(), "{name}");
            for (i, (parsed, written)) in levels.levels.iter().zip(&rep.cogp_levels).enumerate() {
                assert_eq!(parsed.row_group_end, written.rg_end, "{name}: level {i} end");
                assert_eq!(parsed.gsd, written.gsd, "{name}: level {i} gsd");
            }
            let n_rg = store.rg_starts().len() - 1;
            assert_eq!(levels.levels.last().unwrap().row_group_end, n_rg - 1, "{name}");

            // (3) A view between the two coarsest levels reads level 0's
            // prefix: the finest level whose gsd still covers it.
            let between = (gsds[0] + gsds[1]) / 2.0;
            assert_eq!(levels.level_for_gsd(between), 0, "{name}");
            assert_eq!(
                levels.row_group_end_for_gsd(between),
                levels.levels[0].row_group_end,
                "{name}"
            );

            // (4) C8 grades it, advisory and passing, naming the same
            // statistics — and the verdict itself passes, because C2
            // measures clustering inside the levels rather than across
            // a file whose levels overlap each other by construction.
            let q = info.quality.as_ref().expect("quality report");
            let c2 = q.checks.iter().find(|c| c.code == "C2").expect("C2");
            assert_eq!(
                c2.status,
                crate::data::quality::Status::Pass,
                "{name}: {}",
                c2.detail
            );
            assert!(c2.detail.starts_with("within COGP levels:"), "{name}: {}", c2.detail);
            assert!(q.indexable, "{name}: a COGP file we wrote must open ungated");
            let c8 = q.checks.iter().find(|c| c.code == "C8").expect("C8");
            assert_eq!(c8.status, crate::data::quality::Status::Pass, "{name}: {}", c8.detail);
            assert!(!c8.gating, "{name}");
            assert!(c8.detail.contains("COGP 0.1.0"), "{name}: {}", c8.detail);
            assert!(c8.detail.contains(want.label()), "{name}: {}", c8.detail);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Degrees convert to metres per feature; a projected CRS in metres does
    /// not convert at all. Same layer, same shapes, same levels.
    #[test]
    fn cogp_measures_degrees_in_metres() {
        let dir = std::env::temp_dir().join(format!("geopq_cogp_deg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // ~1 degree square at the equator is ~111 km across: comfortably
        // over a 10 km gsd at polygon factor 4, and it must be placed
        // there rather than compared against a raw "1" of longitude.
        let rows: Vec<(Vec<u8>, i64, i64)> = vec![
            (wkb_square(0.0, 0.0, 0.5), 0, 0),
            (wkb_square(10.0, 0.0, 0.000_01), 1, 0),
        ];
        let src = dir.join("deg.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("geometry", DataType::Binary, false),
            Field::new("id", DataType::Int64, false),
            Field::new("rank", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_iter_values(rows.iter().map(|r| &r.0))),
                Arc::new(Int64Array::from(vec![0i64, 1])),
                Arc::new(Int64Array::from(vec![0i64, 0])),
            ],
        )
        .unwrap();
        // No `crs`, which per GeoParquet means OGC:CRS84 — degrees.
        let geo = serde_json::json!({
            "version": "1.0.0", "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon"]}},
        });
        let mut w = ArrowWriter::try_new(File::create(&src).unwrap(), schema, None).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        ));
        w.write(&batch).unwrap();
        w.close().unwrap();

        let dst = dir.join("out.parquet");
        let opts = OptimizeOptions {
            row_group_size: 1,
            cogp: Some(cogp_opts(&[10_000.0, 100.0])),
            ..Default::default()
        };
        optimize(&Source::Local(src), &dst, &opts, None, None, &|_, _| {}).unwrap();
        let (meta, feats) = cogp_features(&dst);
        assert_eq!(meta.levels.len(), 2);
        assert_eq!(feats.iter().find(|f| f.1 == 0).unwrap().0, 0);
        assert_eq!(feats.iter().find(|f| f.1 == 1).unwrap().0, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

}
