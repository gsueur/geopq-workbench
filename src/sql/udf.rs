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
    Area, BoundingRect, Buffer, Centroid, Contains, ConvexHull, CoordsIter, Distance, Euclidean,
    Intersects, Length, Simplify, Within,
};
use geo_types::{Geometry, Point, Rect};
use wkt::ToWkt;

use crate::data::loader::decode_wkb;

/// Register every spatial UDF on the session.
pub fn register_all(ctx: &SessionContext) {
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }
}

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
