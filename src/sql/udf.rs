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
        make_udf("st_geomfromtext", vec![Utf8], Binary, |args| {
            let arr = arrow::array::cast::as_string_array(&args[0]);
            let mut b = WkbOut::new();
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    b.push(None)?;
                    continue;
                }
                let g: Option<Geometry<f64>> = std::str::FromStr::from_str(arr.value(i))
                    .ok()
                    .and_then(|w: wkt::Wkt<f64>| Geometry::try_from(w).ok());
                let Some(g) = g else {
                    return Err(DataFusionError::Execution(format!(
                        "st_geomfromtext: invalid WKT {:?}",
                        arr.value(i)
                    )));
                };
                b.push(Some(&g))?;
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
        make_udf("st_buffer", vec![Binary, Float64], Binary, |args| {
            let g = geom_array(&args[0])?;
            let d = f64_array(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..g.len() {
                match (geom_at(&g, i), d.is_null(i)) {
                    (Some(g), false) => b.push(Some(&Geometry::MultiPolygon(
                        g.buffer(d.value(i)),
                    )))?,
                    _ => b.push(None)?,
                }
            }
            Ok(b.finish())
        }),
        make_udf("st_simplify", vec![Binary, Float64], Binary, |args| {
            let g = geom_array(&args[0])?;
            let t = f64_array(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..g.len() {
                match (geom_at(&g, i), t.is_null(i)) {
                    (Some(g), false) => {
                        let s = simplify_geom(&g, t.value(i));
                        b.push(Some(&s))?
                    }
                    _ => b.push(None)?,
                }
            }
            Ok(b.finish())
        }),
        // Reprojection. WKB carries no SRID here, so both CRSs are
        // explicit; strings parse per distinct value, not per row.
        make_udf("st_transform", vec![Binary, Utf8, Utf8], Binary, |args| {
            use geo::MapCoords;
            let g = geom_array(&args[0])?;
            let from = arrow::array::cast::as_string_array(&args[1]);
            let to = arrow::array::cast::as_string_array(&args[2]);
            let mut cache: Option<(String, String, Crs, Crs)> = None;
            let mut b = WkbOut::new();
            for i in 0..g.len() {
                if g.is_null(i) || from.is_null(i) || to.is_null(i) {
                    b.push(None)?;
                    continue;
                }
                let (fs, ts) = (from.value(i), to.value(i));
                if cache.as_ref().map(|(f, t, ..)| (f.as_str(), t.as_str()))
                    != Some((fs, ts))
                {
                    let src = parse_crs(fs).map_err(DataFusionError::Execution)?;
                    let dst = parse_crs(ts).map_err(DataFusionError::Execution)?;
                    cache = Some((fs.to_string(), ts.to_string(), src, dst));
                }
                let (_, _, src, dst) = cache.as_ref().unwrap();
                let out = geom_at(&g, i).and_then(|geom| {
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
        make_udf("st_distance", vec![Binary, Binary], Float64, |args| {
            let a = geom_array(&args[0])?;
            let c = geom_array(&args[1])?;
            let mut b = Float64Builder::new();
            for i in 0..a.len() {
                match (geom_at(&a, i), geom_at(&c, i)) {
                    (Some(ga), Some(gb)) => b.append_value(Euclidean.distance(&ga, &gb)),
                    _ => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }),
        make_udf(
            "st_dwithin",
            vec![Binary, Binary, Float64],
            Boolean,
            |args| {
                let a = geom_array(&args[0])?;
                let c = geom_array(&args[1])?;
                let d = f64_array(&args[2])?;
                let mut b = BooleanBuilder::new();
                for i in 0..a.len() {
                    match (geom_at(&a, i), geom_at(&c, i), d.is_null(i)) {
                        (Some(ga), Some(gb), false) => {
                            b.append_value(Euclidean.distance(&ga, &gb) <= d.value(i))
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
    make_udf(name, vec![DataType::Binary], DataType::Binary, move |args| {
        let g = geom_array(&args[0])?;
        let mut b = WkbOut::new();
        for i in 0..g.len() {
            match geom_at(&g, i).and_then(|g| f(&g)) {
                Some(g) => b.push(Some(&g))?,
                None => b.push(None)?,
            }
        }
        Ok(b.finish())
    })
}

fn geom_pair_to_geom(
    name: &str,
    f: fn(&Geometry<f64>, &Geometry<f64>) -> Option<Geometry<f64>>,
) -> ScalarUDF {
    make_udf(
        name,
        vec![DataType::Binary, DataType::Binary],
        DataType::Binary,
        move |args| {
            let a = geom_array(&args[0])?;
            let c = geom_array(&args[1])?;
            let mut b = WkbOut::new();
            for i in 0..a.len() {
                match (geom_at(&a, i), geom_at(&c, i)) {
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
    make_udf(
        name,
        vec![DataType::Binary, DataType::Binary],
        DataType::Boolean,
        move |args| {
            let a = geom_array(&args[0])?;
            let c = geom_array(&args[1])?;
            let mut b = BooleanBuilder::new();
            for i in 0..a.len() {
                match (geom_at(&a, i), geom_at(&c, i)) {
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

fn geom_at(arr: &BinaryArray, i: usize) -> Option<Geometry<f64>> {
    if arr.is_null(i) {
        return None;
    }
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
