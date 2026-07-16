//! GeoArrow native geometry encodings (GeoParquet 1.1 `encoding` values
//! "point", "linestring", "polygon", "multipoint", "multilinestring",
//! "multipolygon"): geometry stored as nested arrow coordinate arrays
//! (`list<...<struct<x, y>>>`) instead of WKB blobs — no per-feature
//! parsing on read, and the x/y leaf columns get ordinary parquet
//! statistics (row-group and page level) for free.
//!
//! [`GeomCol`] is the unified per-batch accessor over WKB and GeoArrow
//! columns; [`GaBuilder`] builds GeoArrow arrays for the optimize export.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, ListArray, StructArray};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Fields};
use geo_types::{
    Coord, Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
};

use super::loader::{decode_wkb, parse_wkb_point_2d, BinCol};

/// Geometry column encoding, per GeoParquet `geo` metadata.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GeomEncoding {
    #[default]
    Wkb,
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
}

impl GeomEncoding {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "wkb" => Self::Wkb,
            "point" => Self::Point,
            "linestring" => Self::LineString,
            "polygon" => Self::Polygon,
            "multipoint" => Self::MultiPoint,
            "multilinestring" => Self::MultiLineString,
            "multipolygon" => Self::MultiPolygon,
            _ => return None,
        })
    }

    /// GeoParquet metadata `encoding` value.
    pub fn geo_name(&self) -> &'static str {
        match self {
            Self::Wkb => "WKB",
            Self::Point => "point",
            Self::LineString => "linestring",
            Self::Polygon => "polygon",
            Self::MultiPoint => "multipoint",
            Self::MultiLineString => "multilinestring",
            Self::MultiPolygon => "multipolygon",
        }
    }

    /// GeoParquet `geometry_types` entry for what this encoding stores.
    pub fn geometry_type_name(&self) -> &'static str {
        match self {
            Self::Wkb => "Geometry",
            Self::Point => "Point",
            Self::LineString => "LineString",
            Self::Polygon => "Polygon",
            Self::MultiPoint => "MultiPoint",
            Self::MultiLineString => "MultiLineString",
            Self::MultiPolygon => "MultiPolygon",
        }
    }

    pub fn is_wkb(&self) -> bool {
        matches!(self, Self::Wkb)
    }
}

/// Pick the GeoArrow encoding able to hold all observed geometry types
/// (singles promote into their multi variant).
pub fn target_encoding<'a>(
    types: impl Iterator<Item = &'a str> + Clone,
) -> Result<GeomEncoding, String> {
    let mut seen = [false; 3]; // point / line / polygon family
    let mut multi = false;
    let mut any = false;
    for t in types.clone() {
        any = true;
        match t {
            "Point" => seen[0] = true,
            "MultiPoint" => (seen[0], multi) = (true, true),
            "LineString" => seen[1] = true,
            "MultiLineString" => (seen[1], multi) = (true, true),
            "Polygon" => seen[2] = true,
            "MultiPolygon" => (seen[2], multi) = (true, true),
            other => {
                return Err(format!(
                    "GeoArrow encoding cannot store '{other}' geometries"
                ))
            }
        }
    }
    if !any {
        return Err("no geometries found".into());
    }
    match seen {
        [true, false, false] => Ok(if multi {
            GeomEncoding::MultiPoint
        } else {
            GeomEncoding::Point
        }),
        [false, true, false] => Ok(if multi {
            GeomEncoding::MultiLineString
        } else {
            GeomEncoding::LineString
        }),
        [false, false, true] => Ok(if multi {
            GeomEncoding::MultiPolygon
        } else {
            GeomEncoding::Polygon
        }),
        _ => Err(format!(
            "GeoArrow encoding needs a single geometry family, file mixes: {}",
            types.collect::<Vec<_>>().join(", ")
        )),
    }
}

// ---------------------------------------------------------------------
// Read side
// ---------------------------------------------------------------------

/// Separated coordinates: the `struct<x, y>` leaves.
struct Coords<'a> {
    x: &'a Float64Array,
    y: &'a Float64Array,
}

impl<'a> Coords<'a> {
    fn new(st: &'a StructArray) -> Option<Self> {
        let get = |name: &str, idx: usize| -> Option<&'a Float64Array> {
            st.column_by_name(name)
                .or_else(|| st.columns().get(idx).map(|c| c as _))?
                .as_any()
                .downcast_ref::<Float64Array>()
        };
        Some(Self {
            x: get("x", 0)?,
            y: get("y", 1)?,
        })
    }

    fn coord(&self, i: usize) -> Coord<f64> {
        Coord {
            x: self.x.value(i),
            y: self.y.value(i),
        }
    }

    fn line(&self, range: std::ops::Range<usize>) -> LineString<f64> {
        LineString::new(range.map(|i| self.coord(i)).collect())
    }
}

/// GeoArrow column accessor: nesting arrays + declared encoding.
pub struct GaCol<'a> {
    enc: GeomEncoding,
    top: &'a dyn Array,
    /// Outermost-to-innermost list arrays (0–3 of them).
    lists: Vec<&'a ListArray>,
    coords: Coords<'a>,
}

impl<'a> GaCol<'a> {
    fn new(col: &'a dyn Array, enc: GeomEncoding) -> Option<Self> {
        let depth = match enc {
            GeomEncoding::Wkb => return None,
            GeomEncoding::Point => 0,
            GeomEncoding::LineString | GeomEncoding::MultiPoint => 1,
            GeomEncoding::Polygon | GeomEncoding::MultiLineString => 2,
            GeomEncoding::MultiPolygon => 3,
        };
        let mut lists: Vec<&'a ListArray> = Vec::with_capacity(depth);
        let mut cur: &'a dyn Array = col;
        for _ in 0..depth {
            let l = cur.as_any().downcast_ref::<ListArray>()?;
            lists.push(l);
            cur = l.values().as_ref();
        }
        let st = cur.as_any().downcast_ref::<StructArray>()?;
        Some(Self {
            enc,
            top: col,
            lists,
            coords: Coords::new(st)?,
        })
    }

    fn range(l: &ListArray, i: usize) -> std::ops::Range<usize> {
        let o = l.value_offsets();
        o[i] as usize..o[i + 1] as usize
    }

    fn polygon(&self, rings: std::ops::Range<usize>, ring_list: &ListArray) -> Polygon<f64> {
        let mut it = rings.map(|r| self.coords.line(Self::range(ring_list, r)));
        let exterior = it.next().unwrap_or_else(|| LineString::new(vec![]));
        Polygon::new(exterior, it.collect())
    }

    fn geometry(&self, i: usize) -> Option<Geometry<f64>> {
        if self.top.is_null(i) {
            return None;
        }
        Some(match self.enc {
            GeomEncoding::Wkb => unreachable!(),
            GeomEncoding::Point => Geometry::Point(Point(self.coords.coord(i))),
            GeomEncoding::LineString => {
                Geometry::LineString(self.coords.line(Self::range(self.lists[0], i)))
            }
            GeomEncoding::MultiPoint => Geometry::MultiPoint(MultiPoint(
                Self::range(self.lists[0], i)
                    .map(|c| Point(self.coords.coord(c)))
                    .collect(),
            )),
            GeomEncoding::Polygon => {
                Geometry::Polygon(self.polygon(Self::range(self.lists[0], i), self.lists[1]))
            }
            GeomEncoding::MultiLineString => Geometry::MultiLineString(MultiLineString(
                Self::range(self.lists[0], i)
                    .map(|l| self.coords.line(Self::range(self.lists[1], l)))
                    .collect(),
            )),
            GeomEncoding::MultiPolygon => Geometry::MultiPolygon(MultiPolygon(
                Self::range(self.lists[0], i)
                    .map(|p| self.polygon(Self::range(self.lists[1], p), self.lists[2]))
                    .collect(),
            )),
        })
    }
}

/// Unified geometry column accessor: WKB bytes or GeoArrow arrays.
pub enum GeomCol<'a> {
    Wkb(BinCol<'a>),
    Ga(GaCol<'a>),
}

impl<'a> GeomCol<'a> {
    pub fn new(col: &'a dyn Array, enc: GeomEncoding) -> Option<Self> {
        match enc {
            GeomEncoding::Wkb => BinCol::new(col).map(GeomCol::Wkb),
            _ => GaCol::new(col, enc).map(GeomCol::Ga),
        }
    }

    /// Fast path: plain 2D point coordinates without geometry allocation.
    pub fn point2(&self, i: usize) -> Option<(f64, f64)> {
        match self {
            GeomCol::Wkb(b) => parse_wkb_point_2d(b.value(i)?),
            GeomCol::Ga(g) if g.enc == GeomEncoding::Point => {
                (!g.top.is_null(i)).then(|| (g.coords.x.value(i), g.coords.y.value(i)))
            }
            GeomCol::Ga(_) => None,
        }
    }

    pub fn geometry(&self, i: usize) -> Option<Geometry<f64>> {
        match self {
            GeomCol::Wkb(b) => decode_wkb(b.value(i)?),
            GeomCol::Ga(g) => g.geometry(i),
        }
    }

    pub fn is_null(&self, i: usize) -> bool {
        match self {
            GeomCol::Wkb(b) => b.value(i).is_none(),
            GeomCol::Ga(g) => g.top.is_null(i),
        }
    }
}

// ---------------------------------------------------------------------
// Bulk loading path
// ---------------------------------------------------------------------

impl<'a> GaCol<'a> {
    /// Innermost coordinate span of one row (walks all list levels).
    fn coord_span(&self, i: usize) -> std::ops::Range<usize> {
        let mut r = i..i + 1;
        for l in &self.lists {
            let o = l.value_offsets();
            r = o[r.start] as usize..o[r.end] as usize;
        }
        r
    }
}

/// Bulk-load one GeoArrow batch: reproject the entire coordinate buffer in
/// a single linear pass (no per-feature geometry allocation), then emit
/// features into the mesh builder straight from the arrow offsets.
///
/// Transform failures become NaN coordinates and fail the per-feature
/// world-bbox check (counted in `bad`). `raw_bbox(global_row, bbox)` gets
/// each feature's data-CRS bbox for row-group stats accumulation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_bulk(
    ga: &GaCol,
    rows: usize,
    global_of: &dyn Fn(usize) -> u64,
    tr: &super::crs::BulkTransformer,
    display: &super::crs::DisplayCrs,
    mb: &mut super::geometry::MeshBuilder,
    items: &mut Vec<super::layer::PickItem>,
    bad: &mut usize,
    raw_bbox: &mut dyn FnMut(u64, [f64; 4]),
    bins: Option<&[u8]>,
) -> usize {
    use super::geometry::FeatureRef;
    use super::layer::PickItem;

    let (xs, ys) = (ga.coords.x.values(), ga.coords.y.values());
    let n = xs.len().min(ys.len());
    let mut world: Vec<Coord<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let (mut x, mut y) = (xs[i], ys[i]);
        if tr.apply(&mut x, &mut y) {
            let w = display.world_from_projected(x, y);
            world.push(Coord { x: w[0], y: w[1] });
        } else {
            world.push(Coord {
                x: f64::NAN,
                y: f64::NAN,
            });
        }
    }

    let mut ring_refs: Vec<&[Coord<f64>]> = Vec::new();
    for i in 0..rows {
        if ga.top.is_null(i) {
            continue;
        }
        let span = ga.coord_span(i);
        if span.is_empty() {
            continue;
        }
        let global = global_of(i);
        let fref = FeatureRef {
            index: global as u32,
        };

        // Data-CRS bbox for row-group stats.
        let mut rb = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for c in span.clone() {
            rb = [rb[0].min(xs[c]), rb[1].min(ys[c]), rb[2].max(xs[c]), rb[3].max(ys[c])];
        }
        if rb[0].is_finite() && rb[3].is_finite() {
            raw_bbox(global, rb);
        }

        // World bbox; NaN anywhere in the feature ⇒ projection failed.
        let mut wb = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for c in &world[span.clone()] {
            wb = [wb[0].min(c.x), wb[1].min(c.y), wb[2].max(c.x), wb[3].max(c.y)];
        }
        if !(wb[0].is_finite() && wb[1].is_finite() && wb[2].is_finite() && wb[3].is_finite()) {
            *bad += 1;
            continue;
        }

        if let Some(b) = bins {
            mb.bin = b[i];
        }
        mb.expand_feature(wb);
        let mut needs_rtree = true;
        match ga.enc {
            GeomEncoding::Wkb => unreachable!(),
            GeomEncoding::Point => {
                mb.add_point_xy(world[span.start].x, world[span.start].y, fref);
                needs_rtree = false;
            }
            GeomEncoding::MultiPoint => {
                mb.add_points_world(&world[span.clone()], wb, fref);
                needs_rtree = false;
            }
            GeomEncoding::LineString => {
                mb.add_polyline_world(&world[span.clone()], wb, false);
            }
            GeomEncoding::MultiLineString => {
                for part in GaCol::range(ga.lists[0], i) {
                    let s = GaCol::range(ga.lists[1], part);
                    mb.add_polyline_world(&world[s], wb, false);
                }
            }
            GeomEncoding::Polygon => {
                ring_refs.clear();
                for ring in GaCol::range(ga.lists[0], i) {
                    let s = GaCol::range(ga.lists[1], ring);
                    ring_refs.push(&world[s]);
                }
                mb.add_polygon_parts(&ring_refs, wb);
            }
            GeomEncoding::MultiPolygon => {
                for poly in GaCol::range(ga.lists[0], i) {
                    ring_refs.clear();
                    for ring in GaCol::range(ga.lists[1], poly) {
                        let s = GaCol::range(ga.lists[2], ring);
                        ring_refs.push(&world[s]);
                    }
                    mb.add_polygon_parts(&ring_refs, wb);
                }
            }
        }
        if needs_rtree {
            items.push(PickItem {
                bbox: wb,
                feature: fref,
            });
        }
    }
    rows
}

// ---------------------------------------------------------------------
// Write side
// ---------------------------------------------------------------------

fn coord_fields() -> Fields {
    [
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]
    .into_iter()
    .collect()
}

/// Arrow data type of a GeoArrow-encoded geometry column (separated
/// coordinates; canonical child names).
pub fn data_type(enc: GeomEncoding) -> DataType {
    let coords = DataType::Struct(coord_fields());
    let list = |name: &str, inner: DataType| {
        DataType::List(Arc::new(Field::new(name, inner, false)))
    };
    match enc {
        GeomEncoding::Wkb => DataType::Binary,
        GeomEncoding::Point => coords,
        GeomEncoding::LineString => list("vertices", coords),
        GeomEncoding::MultiPoint => list("points", coords),
        GeomEncoding::Polygon => list("rings", list("vertices", coords)),
        GeomEncoding::MultiLineString => list("linestrings", list("vertices", coords)),
        GeomEncoding::MultiPolygon => {
            list("polygons", list("rings", list("vertices", coords)))
        }
    }
}

/// Builds a GeoArrow geometry column; singles promote into multi targets
/// (e.g. Polygon rows in a MultiPolygon column).
pub struct GaBuilder {
    enc: GeomEncoding,
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Offsets per nesting level, outermost first (each starts with 0).
    offsets: [Vec<i32>; 3],
    valid: Vec<bool>,
    len: usize,
}

impl GaBuilder {
    pub fn new(enc: GeomEncoding) -> Self {
        Self {
            enc,
            xs: Vec::new(),
            ys: Vec::new(),
            offsets: [vec![0], vec![0], vec![0]],
            valid: Vec::new(),
            len: 0,
        }
    }

    fn depth(&self) -> usize {
        match self.enc {
            GeomEncoding::Wkb | GeomEncoding::Point => 0,
            GeomEncoding::LineString | GeomEncoding::MultiPoint => 1,
            GeomEncoding::Polygon | GeomEncoding::MultiLineString => 2,
            GeomEncoding::MultiPolygon => 3,
        }
    }

    fn push_coord(&mut self, c: Coord<f64>) {
        self.xs.push(c.x);
        self.ys.push(c.y);
    }

    fn push_line(&mut self, l: &LineString<f64>) {
        for c in &l.0 {
            self.push_coord(*c);
        }
    }

    fn close(&mut self, level: usize) {
        let end = if level + 1 < self.depth() {
            self.offsets[level + 1].len() as i32 - 1
        } else {
            self.xs.len() as i32
        };
        self.offsets[level].push(end);
    }

    /// Append one geometry (or a null row).
    pub fn push(&mut self, g: Option<&Geometry<f64>>) -> Result<(), String> {
        use Geometry as G;
        let normalized: Option<G<f64>>; // owns promoted Rect/Triangle forms
        let g = match g {
            Some(G::Rect(r)) => {
                normalized = Some(G::Polygon(r.to_polygon()));
                normalized.as_ref()
            }
            Some(G::Triangle(t)) => {
                normalized = Some(G::Polygon(t.to_polygon()));
                normalized.as_ref()
            }
            other => other,
        };
        self.len += 1;
        self.valid.push(g.is_some());
        let mismatch = |got: &str| -> Result<(), String> {
            Err(format!(
                "cannot store {got} in a '{}' GeoArrow column",
                self.enc.geo_name()
            ))
        };
        match (self.enc, g) {
            (GeomEncoding::Point, None) => self.push_coord(Coord { x: 0.0, y: 0.0 }),
            (GeomEncoding::Point, Some(G::Point(p))) => self.push_coord(p.0),
            (GeomEncoding::Point, Some(other)) => return mismatch(kind_name(other)),

            (GeomEncoding::MultiPoint, g) => {
                match g {
                    None => {}
                    Some(G::Point(p)) => self.push_coord(p.0),
                    Some(G::MultiPoint(mp)) => {
                        for p in &mp.0 {
                            self.push_coord(p.0);
                        }
                    }
                    Some(other) => return mismatch(kind_name(other)),
                }
                self.close(0);
            }
            (GeomEncoding::LineString, g) => {
                match g {
                    None => {}
                    Some(G::LineString(l)) => self.push_line(l),
                    Some(other) => return mismatch(kind_name(other)),
                }
                self.close(0);
            }
            (GeomEncoding::MultiLineString, g) => {
                match g {
                    None => {}
                    Some(G::LineString(l)) => {
                        self.push_line(l);
                        self.close(1);
                    }
                    Some(G::MultiLineString(ml)) => {
                        for l in &ml.0 {
                            self.push_line(l);
                            self.close(1);
                        }
                    }
                    Some(other) => return mismatch(kind_name(other)),
                }
                self.close(0);
            }
            (GeomEncoding::Polygon, g) => {
                match g {
                    None => {}
                    Some(G::Polygon(p)) => self.push_polygon_rings(p),
                    Some(other) => return mismatch(kind_name(other)),
                }
                self.close(0);
            }
            (GeomEncoding::MultiPolygon, g) => {
                match g {
                    None => {}
                    Some(G::Polygon(p)) => {
                        self.push_polygon_rings(p);
                        self.close(1);
                    }
                    Some(G::MultiPolygon(mp)) => {
                        for p in &mp.0 {
                            self.push_polygon_rings(p);
                            self.close(1);
                        }
                    }
                    Some(other) => return mismatch(kind_name(other)),
                }
                self.close(0);
            }
            (GeomEncoding::Wkb, _) => return Err("GaBuilder cannot build WKB".into()),
        }
        Ok(())
    }

    /// Rings of one polygon, closing the innermost list level per ring.
    fn push_polygon_rings(&mut self, p: &Polygon<f64>) {
        let inner = self.depth() - 1;
        self.push_line(p.exterior());
        self.close(inner);
        for hole in p.interiors() {
            self.push_line(hole);
            self.close(inner);
        }
    }

    pub fn finish(self) -> ArrayRef {
        let depth = self.depth();
        let Self {
            enc,
            xs,
            ys,
            offsets,
            valid,
            ..
        } = self;
        let coords: ArrayRef = Arc::new(StructArray::new(
            coord_fields(),
            vec![
                Arc::new(Float64Array::from(xs)) as ArrayRef,
                Arc::new(Float64Array::from(ys)) as ArrayRef,
            ],
            None,
        ));
        let nulls = NullBuffer::from(valid);
        if depth == 0 {
            // Point: struct with top-level validity.
            let st = coords.as_any().downcast_ref::<StructArray>().unwrap();
            return Arc::new(StructArray::new(
                coord_fields(),
                st.columns().to_vec(),
                Some(nulls),
            ));
        }
        // Wrap lists innermost-first; child fields follow data_type().
        let dt = data_type(enc);
        let mut fields = Vec::with_capacity(depth);
        let mut cur = &dt;
        for _ in 0..depth {
            let DataType::List(f) = cur else { unreachable!() };
            fields.push(f.clone());
            cur = f.data_type();
        }
        let mut arr: ArrayRef = coords;
        for level in (0..depth).rev() {
            let o = OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(offsets[level].clone()));
            let top_nulls = (level == 0).then(|| nulls.clone());
            arr = Arc::new(ListArray::new(fields[level].clone(), o, arr, top_nulls));
        }
        arr
    }
}

fn kind_name(g: &Geometry<f64>) -> &'static str {
    use Geometry::*;
    match g {
        Point(_) => "Point",
        Line(_) => "Line",
        LineString(_) => "LineString",
        Polygon(_) => "Polygon",
        MultiPoint(_) => "MultiPoint",
        MultiLineString(_) => "MultiLineString",
        MultiPolygon(_) => "MultiPolygon",
        GeometryCollection(_) => "GeometryCollection",
        Rect(_) => "Rect",
        Triangle(_) => "Triangle",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{polygon, MultiPolygon};

    /// Round-trip every encoding through GaBuilder → GeomCol.
    #[test]
    fn builder_accessor_roundtrip() {
        let cases: Vec<(GeomEncoding, Vec<Option<Geometry<f64>>>)> = vec![
            (
                GeomEncoding::Point,
                vec![
                    Some(Point::new(1.0, 2.0).into()),
                    None,
                    Some(Point::new(-3.5, 4.25).into()),
                ],
            ),
            (
                GeomEncoding::MultiPoint,
                vec![
                    Some(Point::new(1.0, 2.0).into()), // promoted
                    Some(MultiPoint(vec![Point::new(3.0, 4.0), Point::new(5.0, 6.0)]).into()),
                    None,
                ],
            ),
            (
                GeomEncoding::LineString,
                vec![Some(
                    LineString::from(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]).into(),
                )],
            ),
            (
                GeomEncoding::MultiLineString,
                vec![
                    Some(LineString::from(vec![(0.0, 0.0), (1.0, 1.0)]).into()),
                    Some(
                        MultiLineString(vec![
                            LineString::from(vec![(0.0, 0.0), (1.0, 0.0)]),
                            LineString::from(vec![(2.0, 2.0), (3.0, 3.0), (4.0, 2.0)]),
                        ])
                        .into(),
                    ),
                ],
            ),
            (
                GeomEncoding::Polygon,
                vec![
                    Some(
                        polygon!(exterior: [(x: 0.0, y: 0.0), (x: 4.0, y: 0.0), (x: 4.0, y: 4.0), (x: 0.0, y: 0.0)],
                                 interiors: [[(x: 1.0, y: 1.0), (x: 2.0, y: 1.0), (x: 2.0, y: 2.0), (x: 1.0, y: 1.0)]])
                        .into(),
                    ),
                    None,
                ],
            ),
            (
                GeomEncoding::MultiPolygon,
                vec![
                    Some(polygon![(x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 1.0, y: 1.0), (x: 0.0, y: 0.0)].into()), // promoted
                    Some(
                        MultiPolygon(vec![
                            polygon![(x: 5.0, y: 5.0), (x: 6.0, y: 5.0), (x: 6.0, y: 6.0), (x: 5.0, y: 5.0)],
                            polygon!(exterior: [(x: 8.0, y: 8.0), (x: 12.0, y: 8.0), (x: 12.0, y: 12.0), (x: 8.0, y: 8.0)],
                                     interiors: [[(x: 9.0, y: 9.0), (x: 10.0, y: 9.0), (x: 10.0, y: 10.0), (x: 9.0, y: 9.0)]]),
                        ])
                        .into(),
                    ),
                ],
            ),
        ];
        for (enc, geoms) in cases {
            let mut b = GaBuilder::new(enc);
            for g in &geoms {
                b.push(g.as_ref()).unwrap();
            }
            let arr = b.finish();
            assert_eq!(&data_type(enc), arr.data_type(), "{enc:?} data type");
            let col = GeomCol::new(arr.as_ref(), enc).expect("accessor");
            for (i, g) in geoms.iter().enumerate() {
                let got = col.geometry(i);
                match g {
                    None => assert!(got.is_none(), "{enc:?} row {i} should be null"),
                    Some(want) => {
                        let got = got.unwrap_or_else(|| panic!("{enc:?} row {i} null"));
                        // Promotion wraps singles into multis.
                        let promoted = match (enc, want.clone()) {
                            (GeomEncoding::MultiPoint, Geometry::Point(p)) => {
                                Geometry::MultiPoint(MultiPoint(vec![p]))
                            }
                            (GeomEncoding::MultiLineString, Geometry::LineString(l)) => {
                                Geometry::MultiLineString(MultiLineString(vec![l]))
                            }
                            (GeomEncoding::MultiPolygon, Geometry::Polygon(p)) => {
                                Geometry::MultiPolygon(MultiPolygon(vec![p]))
                            }
                            (_, w) => w,
                        };
                        assert_eq!(got, promoted, "{enc:?} row {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn target_encoding_selection() {
        let t = |v: &[&'static str]| target_encoding(v.iter().copied());
        assert_eq!(t(&["Point"]).unwrap(), GeomEncoding::Point);
        assert_eq!(t(&["Point", "MultiPoint"]).unwrap(), GeomEncoding::MultiPoint);
        assert_eq!(t(&["Polygon"]).unwrap(), GeomEncoding::Polygon);
        assert_eq!(
            t(&["Polygon", "MultiPolygon"]).unwrap(),
            GeomEncoding::MultiPolygon
        );
        assert_eq!(t(&["LineString"]).unwrap(), GeomEncoding::LineString);
        assert!(t(&["Point", "Polygon"]).is_err());
        assert!(t(&["GeometryCollection"]).is_err());
        assert!(t(&[]).is_err());
    }
}
