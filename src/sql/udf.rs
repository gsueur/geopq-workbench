//! Spatial scalar UDFs (ST_*), our own implementation on geo-types.
//!
//! Geometries travel as WKB in `Binary` columns — the layer tables expose
//! that encoding natively and every constructor here returns it, so the
//! functions compose freely. Measures are planar, in the data CRS units;
//! null or undecodable geometries yield NULL rather than failing the query.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanBuilder, Float64Array, Float64Builder,
    Int64Builder, StringBuilder,
};
use arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::{create_udf, ColumnarValue, ScalarUDF, Volatility};
use datafusion::prelude::SessionContext;
use geo::{
    Area, BooleanOps, BoundingRect, Buffer, Centroid, Contains, ConvexHull, CoordsIter, Distance,
    Euclidean, Intersects, Length, OpType, Simplify, Within,
};
use geo_types::{Geometry, MultiPolygon, Point, Polygon, Rect};
use wkt::ToWkt;

use crate::data::crs::Crs;
use crate::data::loader::decode_wkb;

/// Parse a user CRS argument: `EPSG:nnnn`, a bare numeric code, or a
/// full proj4 string.
fn parse_crs(s: &str) -> Result<Crs, String> {
    let t = s.trim();
    if t.starts_with('+') {
        return Crs::from_proj4(t, None, t);
    }
    let num = t
        .strip_prefix("EPSG:")
        .or_else(|| t.strip_prefix("epsg:"))
        .unwrap_or(t);
    let code: u32 = num
        .trim()
        .parse()
        .map_err(|_| format!("invalid CRS {t:?} — use 'EPSG:nnnn' or a proj4 string"))?;
    Crs::from_epsg(code)
}

/// Register every spatial UDF on the session.
pub fn register_all(ctx: &SessionContext) {
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }
}

/// Bare function names, for autocomplete.
pub const NAMES: &[&str] = &[
    "st_area",
    "st_length",
    "st_centroid",
    "st_x",
    "st_y",
    "st_xmin",
    "st_ymin",
    "st_xmax",
    "st_ymax",
    "st_astext",
    "st_geomfromtext",
    "st_point",
    "st_makeenvelope",
    "st_intersects",
    "st_contains",
    "st_within",
    "st_distance",
    "st_dwithin",
    "st_geometrytype",
    "st_npoints",
    "st_buffer",
    "st_simplify",
    "st_convexhull",
    "st_transform",
    "st_union",
    "st_intersection",
    "st_difference",
    "st_symdifference",
];

/// Names and signatures for the console's help panel.
pub fn catalog() -> &'static [(&'static str, &'static str)] {
    &[
        ("st_area(geom)", "planar area, CRS units²"),
        ("st_length(geom)", "planar length, CRS units"),
        ("st_centroid(geom)", "centroid point"),
        ("st_x(geom) / st_y(geom)", "point coordinates"),
        ("st_xmin/st_ymin/st_xmax/st_ymax(geom)", "bounding box edges"),
        ("st_astext(geom)", "WKT string"),
        ("st_geomfromtext(wkt)", "geometry from WKT"),
        ("st_point(x, y)", "point from coordinates"),
        ("st_makeenvelope(xmin, ymin, xmax, ymax)", "rectangle polygon"),
        ("st_intersects(a, b)", "true if a and b intersect"),
        ("st_contains(a, b)", "true if a contains b"),
        ("st_within(a, b)", "true if a is within b"),
        ("st_distance(a, b)", "planar distance, CRS units"),
        ("st_dwithin(a, b, dist)", "true if within dist"),
        ("st_geometrytype(geom)", "type name, e.g. MultiPolygon"),
        ("st_npoints(geom)", "number of vertices"),
        ("st_buffer(geom, dist)", "buffered polygon"),
        ("st_simplify(geom, tol)", "Douglas-Peucker simplification"),
        ("st_convexhull(geom)", "convex hull polygon"),
        (
            "st_transform(geom, 'EPSG:4326', 'EPSG:2154')",
            "reproject between CRSs (EPSG codes or proj4 strings)",
        ),
        ("st_union(a, b)", "areal union of two geometries"),
        ("st_intersection(a, b)", "the area both cover"),
        ("st_difference(a, b)", "a with b cut out of it"),
        ("st_symdifference(a, b)", "the area exactly one of them covers"),
    ]
}

fn all_udfs() -> Vec<ScalarUDF> {
    use DataType::{Binary, Boolean, Float64, Int64, Utf8};
    vec![
        // Measures.
        geom_to_f64("st_area", |g| Some(g.unsigned_area())),
        geom_to_f64("st_length", |g| Some(linear_length(g))),
        geom_to_f64("st_x", |g| match g {
            Geometry::Point(p) => Some(p.x()),
            _ => None,
        }),
        geom_to_f64("st_y", |g| match g {
            Geometry::Point(p) => Some(p.y()),
            _ => None,
        }),
        geom_to_f64("st_xmin", |g| g.bounding_rect().map(|r| r.min().x)),
        geom_to_f64("st_ymin", |g| g.bounding_rect().map(|r| r.min().y)),
        geom_to_f64("st_xmax", |g| g.bounding_rect().map(|r| r.max().x)),
        geom_to_f64("st_ymax", |g| g.bounding_rect().map(|r| r.max().y)),
        // Accessors.
        make_udf("st_geometrytype", vec![Binary], Utf8, |args| {
            let g = geom_array(&args[0])?;
            let mut b = StringBuilder::new();
            for i in 0..g.len() {
                match geom_at(&g, i) {
                    Some(g) => b.append_value(type_name(&g)),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }),
        make_udf("st_npoints", vec![Binary], Int64, |args| {
            let g = geom_array(&args[0])?;
            let mut b = Int64Builder::new();
            for i in 0..g.len() {
                match geom_at(&g, i) {
                    Some(g) => b.append_value(g.coords_count() as i64),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }),
        make_udf("st_astext", vec![Binary], Utf8, |args| {
            let g = geom_array(&args[0])?;
            let mut b = StringBuilder::new();
            for i in 0..g.len() {
                match geom_at(&g, i) {
                    Some(g) => b.append_value(g.wkt_string()),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }),
        // Constructors.
        // A literal the user typed and a column of stored text are not
        // the same kind of input. Bad WKT in the query is a typo and has
        // to be reported; one bad row out of a million in a column is
        // data, and killing the whole query over it contradicts the rule
        // every other function here follows — an undecodable geometry is
        // NULL, not a failure.
        make_udf_cv("st_geomfromtext", vec![Utf8], Binary, |args, rows| {
            let a = StrArg::new(&args[0])?;
            let literal = matches!(args[0], ColumnarValue::Scalar(_));
            let parse = |t: &str| -> Option<Geometry<f64>> {
                std::str::FromStr::from_str(t)
                    .ok()
                    .and_then(|w: wkt::Wkt<f64>| Geometry::try_from(w).ok())
            };
            let mut b = WkbOut::new();
            if literal {
                // Parsed once, not once per row.
                let g = match a.get(0) {
                    None => None,
                    Some(t) => Some(parse(t).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "st_geomfromtext: invalid WKT {t:?}"
                        ))
                    })?),
                };
                for _ in 0..rows {
                    b.push(g.as_ref())?;
                }
                return Ok(b.finish());
            }
            for i in 0..rows {
                match a.get(i).and_then(parse) {
                    Some(g) => b.push(Some(&g))?,
                    None => b.push(None)?,
                }
            }
            Ok(b.finish())
        }),
        make_udf("st_point", vec![Float64, Float64], Binary, |args| {
            let x = f64_array(&args[0])?;
            let y = f64_array(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..x.len() {
                if x.is_null(i) || y.is_null(i) {
                    b.push(None)?;
                } else {
                    b.push(Some(&Geometry::Point(Point::new(x.value(i), y.value(i)))))?;
                }
            }
            Ok(b.finish())
        }),
        make_udf(
            "st_makeenvelope",
            vec![Float64, Float64, Float64, Float64],
            Binary,
            |args| {
                let c: Vec<&Float64Array> =
                    args.iter().map(f64_array).collect::<DfResult<_>>()?;
                let mut b = WkbOut::new();
                for i in 0..c[0].len() {
                    if c.iter().any(|a| a.is_null(i)) {
                        b.push(None)?;
                    } else {
                        let r = Rect::new(
                            (c[0].value(i), c[1].value(i)),
                            (c[2].value(i), c[3].value(i)),
                        );
                        b.push(Some(&Geometry::Polygon(r.to_polygon())))?;
                    }
                }
                Ok(b.finish())
            },
        ),
        // Unary transforms.
        geom_to_geom("st_centroid", |g| g.centroid().map(Geometry::Point)),
        geom_to_geom("st_convexhull", |g| {
            Some(Geometry::Polygon(g.convex_hull()))
        }),
        // Set operations. Always a MultiPolygon out, even for a single
        // part: cutting one polygon with another can split it in two, so
        // the type cannot depend on the values.
        geom_pair_to_geom("st_union", |a, b| set_op(a, b, OpType::Union)),
        geom_pair_to_geom("st_intersection", |a, b| {
            set_op(a, b, OpType::Intersection)
        }),
        geom_pair_to_geom("st_difference", |a, b| set_op(a, b, OpType::Difference)),
        geom_pair_to_geom("st_symdifference", |a, b| set_op(a, b, OpType::Xor)),
        make_udf_cv("st_buffer", vec![Binary, Float64], Binary, |args, rows| {
            let g = GeomArg::new(&args[0])?;
            let d = F64Arg::new(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..rows {
                match (g.get(i), d.get(i)) {
                    (Some(g), Some(d)) => {
                        b.push(Some(&Geometry::MultiPolygon(g.buffer(d))))?
                    }
                    _ => b.push(None)?,
                }
            }
            Ok(b.finish())
        }),
        make_udf_cv("st_simplify", vec![Binary, Float64], Binary, |args, rows| {
            let g = GeomArg::new(&args[0])?;
            let t = F64Arg::new(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..rows {
                match (g.get(i), t.get(i)) {
                    (Some(g), Some(t)) => b.push(Some(&simplify_geom(&g, t)))?,
                    _ => b.push(None)?,
                }
            }
            Ok(b.finish())
        }),
        // Reprojection. WKB carries no SRID here, so both CRSs are
        // explicit; strings parse per distinct value, not per row.
        make_udf_cv("st_transform", vec![Binary, Utf8, Utf8], Binary, |args, rows| {
            use geo::MapCoords;
            let g = GeomArg::new(&args[0])?;
            let from = StrArg::new(&args[1])?;
            let to = StrArg::new(&args[2])?;
            let mut cache: Option<(String, String, Crs, Crs)> = None;
            let mut b = WkbOut::new();
            for i in 0..rows {
                let (Some(fs), Some(ts)) = (from.get(i), to.get(i)) else {
                    b.push(None)?;
                    continue;
                };
                if cache.as_ref().map(|(f, t, ..)| (f.as_str(), t.as_str()))
                    != Some((fs, ts))
                {
                    let src = parse_crs(fs).map_err(DataFusionError::Execution)?;
                    let dst = parse_crs(ts).map_err(DataFusionError::Execution)?;
                    cache = Some((fs.to_string(), ts.to_string(), src, dst));
                }
                let (_, _, src, dst) = cache.as_ref().unwrap();
                let out = g.get(i).and_then(|geom| {
                    geom.try_map_coords(|c| {
                        crate::data::crs::transform_point(src, dst, c.x, c.y)
                            .map(|(x, y)| geo_types::Coord { x, y })
                    })
                    .ok()
                });
                b.push(out.as_ref())?;
            }
            Ok(b.finish())
        }),
        // Predicates and relations.
        geom_pair_to_bool("st_intersects", |a, b| a.intersects(b)),
        geom_pair_to_bool("st_contains", |a, b| a.contains(b)),
        geom_pair_to_bool("st_within", |a, b| a.is_within(b)),
        make_udf_cv("st_distance", vec![Binary, Binary], Float64, |args, rows| {
            let a = GeomArg::new(&args[0])?;
            let c = GeomArg::new(&args[1])?;
            let mut b = Float64Builder::new();
            for i in 0..rows {
                match (a.get(i), c.get(i)) {
                    (Some(ga), Some(gb)) => b.append_value(Euclidean.distance(&*ga, &*gb)),
                    _ => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }),
        make_udf_cv(
            "st_dwithin",
            vec![Binary, Binary, Float64],
            Boolean,
            |args, rows| {
                let a = GeomArg::new(&args[0])?;
                let c = GeomArg::new(&args[1])?;
                let d = F64Arg::new(&args[2])?;
                let mut b = BooleanBuilder::new();
                for i in 0..rows {
                    match (a.get(i), c.get(i), d.get(i)) {
                        (Some(ga), Some(gb), Some(d)) => {
                            b.append_value(Euclidean.distance(&*ga, &*gb) <= d)
                        }
                        _ => b.append_null(),
                    }
                }
                Ok(Arc::new(b.finish()))
            },
        ),
    ]
}

// ---------------------------------------------------------------------------
// Plumbing

/// Wrap an array-in/array-out kernel as a DataFusion scalar UDF.
fn make_udf(
    name: &str,
    args: Vec<DataType>,
    ret: DataType,
    f: impl Fn(&[ArrayRef]) -> DfResult<ArrayRef> + Send + Sync + 'static,
) -> ScalarUDF {
    create_udf(
        name,
        args,
        ret,
        Volatility::Immutable,
        Arc::new(move |cols: &[ColumnarValue]| {
            let arrays = ColumnarValue::values_to_arrays(cols)?;
            f(&arrays).map(ColumnarValue::Array)
        }),
    )
}

/// Like [`make_udf`], but the implementation sees the arguments as
/// DataFusion handed them over — `Scalar` for a literal, `Array` for a
/// column — plus the row count.
///
/// `values_to_arrays` expands a scalar argument into one copy per row,
/// and the row loops here then decoded that copy again for every one of
/// them. The console's own viewport filter is exactly that shape —
/// `st_intersects(geometry, st_makeenvelope(…))`, whose second argument
/// folds to a literal — so browsing a 2M-row layer decoded the same
/// envelope two million times.
fn make_udf_cv(
    name: &str,
    args: Vec<DataType>,
    ret: DataType,
    f: impl Fn(&[ColumnarValue], usize) -> DfResult<ArrayRef> + Send + Sync + 'static,
) -> ScalarUDF {
    create_udf(
        name,
        args,
        ret,
        Volatility::Immutable,
        Arc::new(move |cols: &[ColumnarValue]| {
            // A column fixes the row count; all-scalar arguments mean one.
            let rows = cols
                .iter()
                .filter_map(|c| match c {
                    ColumnarValue::Array(a) => Some(a.len()),
                    ColumnarValue::Scalar(_) => None,
                })
                .max()
                .unwrap_or(1);
            f(cols, rows).map(ColumnarValue::Array)
        }),
    )
}

/// The single-row array behind one argument, whatever form it arrived in.
fn as_one(v: &ColumnarValue) -> DfResult<ArrayRef> {
    Ok(ColumnarValue::values_to_arrays(std::slice::from_ref(v))?.remove(0))
}

/// A geometry argument: a column of WKB, or one value shared by every row
/// and therefore decoded once.
enum GeomArg {
    Const(Option<Geometry<f64>>),
    Col(BinaryArray),
}

impl GeomArg {
    fn new(v: &ColumnarValue) -> DfResult<Self> {
        let arr = as_one(v)?;
        let b = geom_array(&arr)?;
        match v {
            ColumnarValue::Scalar(_) => Ok(GeomArg::Const(geom_at(&b, 0))),
            ColumnarValue::Array(_) => Ok(GeomArg::Col(b)),
        }
    }

    fn get(&self, i: usize) -> Option<std::borrow::Cow<'_, Geometry<f64>>> {
        use std::borrow::Cow;
        match self {
            GeomArg::Const(g) => g.as_ref().map(Cow::Borrowed),
            GeomArg::Col(a) => geom_at(a, i).map(Cow::Owned),
        }
    }
}

/// A f64 argument, constant or per row.
enum F64Arg {
    Const(Option<f64>),
    Col(Float64Array),
}

impl F64Arg {
    fn new(v: &ColumnarValue) -> DfResult<Self> {
        let arr = as_one(v)?;
        let a = f64_array(&arr)?.clone();
        match v {
            ColumnarValue::Scalar(_) => {
                Ok(F64Arg::Const((!a.is_null(0)).then(|| a.value(0))))
            }
            ColumnarValue::Array(_) => Ok(F64Arg::Col(a)),
        }
    }

    fn get(&self, i: usize) -> Option<f64> {
        match self {
            F64Arg::Const(v) => *v,
            F64Arg::Col(a) => (!a.is_null(i)).then(|| a.value(i)),
        }
    }
}

/// A string argument, constant or per row.
enum StrArg {
    Const(Option<String>),
    Col(arrow::array::StringArray),
}

impl StrArg {
    fn new(v: &ColumnarValue) -> DfResult<Self> {
        let arr = as_one(v)?;
        let a = arrow::array::cast::as_string_array(&arr).clone();
        match v {
            ColumnarValue::Scalar(_) => {
                Ok(StrArg::Const((!a.is_null(0)).then(|| a.value(0).to_string())))
            }
            ColumnarValue::Array(_) => Ok(StrArg::Col(a)),
        }
    }

    fn get(&self, i: usize) -> Option<&str> {
        match self {
            StrArg::Const(v) => v.as_deref(),
            StrArg::Col(a) => (!a.is_null(i)).then(|| a.value(i)),
        }
    }
}

fn geom_to_f64(name: &str, f: fn(&Geometry<f64>) -> Option<f64>) -> ScalarUDF {
    make_udf(name, vec![DataType::Binary], DataType::Float64, move |args| {
        let g = geom_array(&args[0])?;
        let mut b = Float64Builder::new();
        for i in 0..g.len() {
            match geom_at(&g, i).and_then(|g| f(&g)) {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        Ok(Arc::new(b.finish()))
    })
}

fn geom_to_geom(name: &str, f: fn(&Geometry<f64>) -> Option<Geometry<f64>>) -> ScalarUDF {
    make_udf_cv(
        name,
        vec![DataType::Binary],
        DataType::Binary,
        move |args, rows| {
            let g = GeomArg::new(&args[0])?;
            let mut b = WkbOut::new();
            for i in 0..rows {
                match g.get(i).and_then(|g| f(&g)) {
                    Some(g) => b.push(Some(&g))?,
                    None => b.push(None)?,
                }
            }
            Ok(b.finish())
        },
    )
}

fn geom_pair_to_geom(
    name: &str,
    f: fn(&Geometry<f64>, &Geometry<f64>) -> Option<Geometry<f64>>,
) -> ScalarUDF {
    make_udf_cv(
        name,
        vec![DataType::Binary, DataType::Binary],
        DataType::Binary,
        move |args, rows| {
            let a = GeomArg::new(&args[0])?;
            let c = GeomArg::new(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..rows {
                match (a.get(i), c.get(i)) {
                    (Some(ga), Some(gb)) => b.push(f(&ga, &gb).as_ref())?,
                    _ => b.push(None)?,
                }
            }
            Ok(b.finish())
        },
    )
}

/// The areal part of a geometry, as the `MultiPolygon` the set operations
/// work on.
///
/// `BooleanOps` is implemented for `Polygon` and `MultiPolygon` only, which
/// is not a gap in the library: a line and a point have no area, so there
/// is no answer to give for "the region both cover". A collection
/// contributes whatever polygons it holds. Anything else yields None, and
/// the caller turns that into NULL rather than a wrong answer.
fn areal(g: &Geometry<f64>) -> Option<MultiPolygon<f64>> {
    match g {
        Geometry::Polygon(p) => Some(MultiPolygon::new(vec![p.clone()])),
        Geometry::MultiPolygon(m) => Some(m.clone()),
        Geometry::Rect(r) => Some(MultiPolygon::new(vec![r.to_polygon()])),
        Geometry::Triangle(t) => Some(MultiPolygon::new(vec![t.to_polygon()])),
        Geometry::GeometryCollection(c) => {
            let polys: Vec<Polygon<f64>> =
                c.iter().filter_map(areal).flat_map(|m| m.0).collect();
            (!polys.is_empty()).then(|| MultiPolygon::new(polys))
        }
        _ => None,
    }
}

fn set_op(a: &Geometry<f64>, b: &Geometry<f64>, op: OpType) -> Option<Geometry<f64>> {
    let (a, b) = (areal(a)?, areal(b)?);
    Some(Geometry::MultiPolygon(a.boolean_op(&b, op)))
}

fn geom_pair_to_bool(name: &str, f: fn(&Geometry<f64>, &Geometry<f64>) -> bool) -> ScalarUDF {
    make_udf_cv(
        name,
        vec![DataType::Binary, DataType::Binary],
        DataType::Boolean,
        move |args, rows| {
            let a = GeomArg::new(&args[0])?;
            let c = GeomArg::new(&args[1])?;
            let mut b = BooleanBuilder::new();
            for i in 0..rows {
                match (a.get(i), c.get(i)) {
                    (Some(ga), Some(gb)) => b.append_value(f(&ga, &gb)),
                    _ => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        },
    )
}

fn geom_array(arr: &ArrayRef) -> DfResult<BinaryArray> {
    arr.as_any()
        .downcast_ref::<BinaryArray>()
        .cloned()
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "expected WKB binary geometry, got {}",
                arr.data_type()
            ))
        })
}

fn f64_array(arr: &ArrayRef) -> DfResult<&Float64Array> {
    arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        DataFusionError::Execution(format!("expected double, got {}", arr.data_type()))
    })
}

// How many WKB blobs this thread has decoded. Test-only, and per thread
// so a parallel test run cannot perturb the count: it is what pins the
// "a constant argument is decoded once, not once per row" property,
// which is otherwise invisible from the outside.
#[cfg(test)]
thread_local! {
    static WKB_DECODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn geom_at(arr: &BinaryArray, i: usize) -> Option<Geometry<f64>> {
    if arr.is_null(i) {
        return None;
    }
    #[cfg(test)]
    WKB_DECODES.with(|c| c.set(c.get() + 1));
    decode_wkb(arr.value(i))
}

fn type_name(g: &Geometry<f64>) -> &'static str {
    match g {
        Geometry::Point(_) => "Point",
        Geometry::Line(_) => "Line",
        Geometry::LineString(_) => "LineString",
        Geometry::Polygon(_) => "Polygon",
        Geometry::MultiPoint(_) => "MultiPoint",
        Geometry::MultiLineString(_) => "MultiLineString",
        Geometry::MultiPolygon(_) => "MultiPolygon",
        Geometry::GeometryCollection(_) => "GeometryCollection",
        Geometry::Rect(_) => "Rect",
        Geometry::Triangle(_) => "Triangle",
    }
}

/// Planar length of linear geometries; 0 for points and areas, matching
/// PostGIS ST_Length.
fn linear_length(g: &Geometry<f64>) -> f64 {
    match g {
        Geometry::Line(l) => Euclidean.length(l),
        Geometry::LineString(l) => Euclidean.length(l),
        Geometry::MultiLineString(l) => Euclidean.length(l),
        Geometry::GeometryCollection(c) => c.iter().map(linear_length).sum(),
        _ => 0.0,
    }
}

fn simplify_geom(g: &Geometry<f64>, tol: f64) -> Geometry<f64> {
    match g {
        Geometry::LineString(l) => Geometry::LineString(l.simplify(tol)),
        Geometry::MultiLineString(l) => Geometry::MultiLineString(l.simplify(tol)),
        Geometry::Polygon(p) => Geometry::Polygon(p.simplify(tol)),
        Geometry::MultiPolygon(p) => Geometry::MultiPolygon(p.simplify(tol)),
        other => other.clone(),
    }
}

/// WKB-encoding binary column builder.
struct WkbOut {
    b: BinaryBuilder,
    buf: Vec<u8>,
    opts: wkb::writer::WriteOptions,
}

impl WkbOut {
    fn new() -> Self {
        Self {
            b: BinaryBuilder::new(),
            buf: Vec::new(),
            opts: wkb::writer::WriteOptions::default(),
        }
    }

    fn push(&mut self, g: Option<&Geometry<f64>>) -> DfResult<()> {
        match g {
            Some(g) => {
                self.buf.clear();
                wkb::writer::write_geometry(&mut self.buf, g, &self.opts)
                    .map_err(|e| DataFusionError::Execution(format!("wkb encode: {e}")))?;
                self.b.append_value(&self.buf);
            }
            None => self.b.append_null(),
        }
        Ok(())
    }

    fn finish(mut self) -> ArrayRef {
        Arc::new(self.b.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::scalar::ScalarValue;

    fn wkb(g: &Geometry<f64>) -> Vec<u8> {
        let mut buf = Vec::new();
        wkb::writer::write_geometry(&mut buf, g, &wkb::writer::WriteOptions::default()).unwrap();
        buf
    }

    fn decodes<T>(f: impl FnOnce() -> T) -> (T, usize) {
        WKB_DECODES.with(|c| c.set(0));
        let out = f();
        (out, WKB_DECODES.with(|c| c.get()))
    }

    /// `st_intersects(geometry, st_makeenvelope(…))` — the console's own
    /// viewport filter — reaches the UDF with the envelope folded to a
    /// literal. `values_to_arrays` copied it once per row and the row
    /// loop decoded every copy, so the cost of the constant scaled with
    /// the layer. It is decoded once now.
    #[test]
    fn a_constant_geometry_argument_is_decoded_once() {
        const ROWS: usize = 1000;
        let point = wkb(&Geometry::Point(Point::new(1.0, 1.0)));
        let envelope = wkb(&Geometry::Polygon(
            Rect::new((0.0, 0.0), (10.0, 10.0)).to_polygon(),
        ));
        let col: BinaryArray = (0..ROWS).map(|_| Some(point.clone())).collect();
        let column = ColumnarValue::Array(Arc::new(col));
        let literal = ColumnarValue::Scalar(ScalarValue::Binary(Some(envelope)));

        let (hits, n) = decodes(|| {
            let a = GeomArg::new(&column).unwrap();
            let b = GeomArg::new(&literal).unwrap();
            (0..ROWS)
                .filter(|&i| match (a.get(i), b.get(i)) {
                    (Some(x), Some(y)) => x.intersects(&*y),
                    _ => false,
                })
                .count()
        });
        assert_eq!(hits, ROWS, "every point is inside the envelope");
        assert_eq!(n, ROWS + 1, "one decode per row plus one for the literal");

        // A null literal is still one lookup, and yields NULL per row.
        let (_, n) = decodes(|| {
            let b = GeomArg::new(&ColumnarValue::Scalar(ScalarValue::Binary(None))).unwrap();
            assert!((0..ROWS).all(|i| b.get(i).is_none()));
        });
        assert_eq!(n, 0, "a NULL literal decodes nothing");
    }

    /// A typo in the query is worth an error; one unparseable row out of
    /// a million is data, and the rest of this module turns that into
    /// NULL rather than failing the query.
    #[test]
    fn bad_wkt_fails_a_literal_and_nulls_a_column() {
        let ctx = SessionContext::new();
        register_all(&ctx);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();

        let err = rt.block_on(async {
            match ctx.sql("select st_geomfromtext('POINT(oops)')").await {
                Err(e) => e.to_string(),
                Ok(df) => match df.collect().await {
                    Err(e) => e.to_string(),
                    Ok(_) => panic!("a bad literal must be reported"),
                },
            }
        });
        assert!(err.contains("invalid WKT"), "{err}");

        // The same text arriving in a column: NULL for the bad row, the
        // good rows unaffected.
        let batches = rt.block_on(async {
            ctx.sql(
                "select st_astext(st_geomfromtext(w)) g from ( \
                   select 'POINT(1 2)' w union all select 'POINT(oops)' \
                   union all select 'POINT(3 4)') order by g nulls last",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
        });
        let col = batches[0].column(0);
        let vals: Vec<Option<String>> = (0..col.len())
            .map(|i| {
                (!col.is_null(i))
                    .then(|| arrow::util::display::array_value_to_string(col, i).unwrap())
            })
            .collect();
        assert_eq!(
            vals,
            vec![
                Some("POINT(1 2)".to_string()),
                Some("POINT(3 4)".to_string()),
                None
            ],
            "one bad row must not sink the query"
        );
    }

    /// The scalar arguments still have to produce the right answers.
    #[test]
    fn constant_arguments_still_give_the_right_answers() {
        let ctx = SessionContext::new();
        register_all(&ctx);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let one = |sql: &str| -> String {
            let batches = rt
                .block_on(async { ctx.sql(sql).await.unwrap().collect().await.unwrap() });
            let b = &batches[0];
            arrow::util::display::array_value_to_string(b.column(0), 0).unwrap()
        };
        assert_eq!(
            one("select st_intersects(st_geomfromtext('POINT(1 1)'), \
                 st_makeenvelope(0, 0, 10, 10))"),
            "true"
        );
        assert_eq!(
            one("select st_intersects(st_geomfromtext('POINT(50 50)'), \
                 st_makeenvelope(0, 0, 10, 10))"),
            "false"
        );
        assert_eq!(
            one("select st_astext(st_transform(st_geomfromtext('POINT(0 0)'), \
                 'EPSG:4326', 'EPSG:4326'))"),
            "POINT(0 0)"
        );
        assert_eq!(
            one("select round(st_area(st_buffer(st_geomfromtext('POINT(0 0)'), 1)))"),
            "3.0"
        );
        assert_eq!(one("select st_distance(st_geomfromtext('POINT(0 0)'), \
             st_geomfromtext('POINT(3 4)'))"), "5.0");
        assert_eq!(one("select st_dwithin(st_geomfromtext('POINT(0 0)'), \
             st_geomfromtext('POINT(3 4)'), 5)"), "true");
    }
}
