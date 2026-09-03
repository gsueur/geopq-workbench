//! Spatial aggregate UDFs (ST_* over a group), on geo-types.
//!
//! These are what turn the console from something that filters a layer into
//! something that produces one: a dissolve by attribute, an extent per
//! category, a set of points gathered into one geometry. Each returns WKB
//! in a `Binary` column like the scalar functions do, so the result feeds
//! "add as layer" unchanged.
//!
//! Cost is worth stating plainly. `st_union_agg` holds every input polygon
//! until the group closes, then overlays them in one pass — that is what
//! makes a dissolve exact, and it means memory scales with the geometry in
//! the group, not with the answer. `st_extent` holds four numbers.
//! Partial aggregation helps: each partition dissolves its own share and
//! only the result crosses to the merge.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryArray, Float64Array};
use arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, Result as DfResult, ScalarValue};
use datafusion::logical_expr::{create_udaf, Accumulator, AggregateUDF, Volatility};
use datafusion::prelude::SessionContext;
use geo::{CoordsIter, unary_union};
use geo_types::{Geometry, GeometryCollection, MultiPolygon, Polygon, Rect, coord};

use crate::data::loader::decode_wkb;

/// Register every spatial aggregate on the session.
pub fn register_all(ctx: &SessionContext) {
    for udaf in all_udafs() {
        ctx.register_udaf(udaf);
    }
}

/// Bare names, for autocomplete.
pub const NAMES: &[&str] = &["st_union_agg", "st_extent", "st_collect"];

/// Names and signatures for the console's help panel.
pub fn catalog() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "st_union_agg(geom)",
            "dissolve a group's polygons into one (aggregate)",
        ),
        (
            "st_extent(geom)",
            "bounding rectangle of a group, as a polygon (aggregate)",
        ),
        (
            "st_collect(geom)",
            "gather a group into one collection, no merging (aggregate)",
        ),
    ]
}

fn all_udafs() -> Vec<AggregateUDF> {
    vec![
        // `st_union` is the two-argument scalar; the aggregate takes the
        // `_agg` suffix rather than overloading it. PostGIS overloads by
        // arity, but DataFusion keeps scalar and aggregate functions in
        // separate registries resolved by name, so one name cannot be both
        // without depending on which registry the planner consults first.
        create_udaf(
            "st_union_agg",
            vec![DataType::Binary],
            Arc::new(DataType::Binary),
            Volatility::Immutable,
            Arc::new(|_| Ok(Box::<UnionAcc>::default())),
            // The partial state is the partial union, so merging is the
            // same operation as accumulating.
            Arc::new(vec![DataType::Binary]),
        ),
        create_udaf(
            "st_extent",
            vec![DataType::Binary],
            Arc::new(DataType::Binary),
            Volatility::Immutable,
            Arc::new(|_| Ok(Box::<ExtentAcc>::default())),
            Arc::new(vec![
                DataType::Float64,
                DataType::Float64,
                DataType::Float64,
                DataType::Float64,
            ]),
        ),
        create_udaf(
            "st_collect",
            vec![DataType::Binary],
            Arc::new(DataType::Binary),
            Volatility::Immutable,
            Arc::new(|_| Ok(Box::<CollectAcc>::default())),
            Arc::new(vec![DataType::Binary]),
        ),
    ]
}

fn wkb(g: &Geometry<f64>) -> DfResult<Vec<u8>> {
    let mut buf = Vec::new();
    wkb::writer::write_geometry(&mut buf, g, &wkb::writer::WriteOptions::default())
        .map_err(|e| DataFusionError::Execution(format!("wkb encode: {e}")))?;
    Ok(buf)
}

fn binary_array(arr: &ArrayRef) -> DfResult<&BinaryArray> {
    arr.as_any().downcast_ref::<BinaryArray>().ok_or_else(|| {
        DataFusionError::Execution(format!("expected WKB binary geometry, got {}", arr.data_type()))
    })
}

/// Every geometry in a WKB column, nulls and undecodable values skipped —
/// the same "a bad geometry is absent, not fatal" rule the scalar
/// functions follow.
fn geoms(arr: &ArrayRef) -> DfResult<Vec<Geometry<f64>>> {
    let a = binary_array(arr)?;
    Ok((0..a.len())
        .filter(|&i| !a.is_null(i))
        .filter_map(|i| decode_wkb(a.value(i)))
        .collect())
}

/// The polygons inside a geometry. Lines and points contribute none: a
/// dissolve is an areal operation, and silently treating a line as a
/// zero-width sliver would put it in the output as nothing at all.
fn polygons(g: &Geometry<f64>, out: &mut Vec<Polygon<f64>>) {
    match g {
        Geometry::Polygon(p) => out.push(p.clone()),
        Geometry::MultiPolygon(m) => out.extend(m.0.iter().cloned()),
        Geometry::Rect(r) => out.push(r.to_polygon()),
        Geometry::Triangle(t) => out.push(t.to_polygon()),
        Geometry::GeometryCollection(c) => c.iter().for_each(|g| polygons(g, out)),
        _ => {}
    }
}

/// Dissolve: `unary_union` over everything the group held.
#[derive(Debug, Default)]
struct UnionAcc {
    polys: Vec<Polygon<f64>>,
}

impl UnionAcc {
    fn dissolved(&self) -> Option<MultiPolygon<f64>> {
        (!self.polys.is_empty()).then(|| unary_union(self.polys.iter()))
    }

    fn value(&self) -> DfResult<ScalarValue> {
        match self.dissolved() {
            Some(m) => Ok(ScalarValue::Binary(Some(wkb(&Geometry::MultiPolygon(m))?))),
            // An empty group is NULL, not an empty polygon: nothing was
            // there to union.
            None => Ok(ScalarValue::Binary(None)),
        }
    }
}

impl Accumulator for UnionAcc {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DfResult<()> {
        for g in geoms(&values[0])? {
            polygons(&g, &mut self.polys);
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DfResult<()> {
        // States are partial unions, which are just more polygons.
        self.update_batch(states)
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        // Dissolve before handing the partition's share over: the merge
        // then re-unions far fewer rings than it otherwise would, and the
        // result is the same either way.
        Ok(vec![self.value()?])
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        self.value()
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self
                .polys
                .iter()
                .map(|p| {
                    (p.exterior().0.len()
                        + p.interiors().iter().map(|r| r.0.len()).sum::<usize>())
                        * size_of::<geo_types::Coord<f64>>()
                })
                .sum::<usize>()
    }
}

/// Bounding rectangle of the group, returned as a polygon so it can be
/// drawn or added as a layer.
#[derive(Debug, Default)]
struct ExtentAcc {
    bbox: Option<[f64; 4]>,
}

impl ExtentAcc {
    fn extend(&mut self, r: [f64; 4]) {
        self.bbox = Some(match self.bbox {
            Some(b) => [b[0].min(r[0]), b[1].min(r[1]), b[2].max(r[2]), b[3].max(r[3])],
            None => r,
        });
    }
}

impl Accumulator for ExtentAcc {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DfResult<()> {
        use geo::BoundingRect;
        for g in geoms(&values[0])? {
            if let Some(r) = g.bounding_rect() {
                self.extend([r.min().x, r.min().y, r.max().x, r.max().y]);
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DfResult<()> {
        let f = |i: usize| -> DfResult<&Float64Array> {
            states[i].as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                DataFusionError::Execution("st_extent state should be four doubles".into())
            })
        };
        let (x0, y0, x1, y1) = (f(0)?, f(1)?, f(2)?, f(3)?);
        for i in 0..x0.len() {
            if x0.is_null(i) {
                continue;
            }
            self.extend([x0.value(i), y0.value(i), x1.value(i), y1.value(i)]);
        }
        Ok(())
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        let b = self.bbox;
        Ok((0..4)
            .map(|i| ScalarValue::Float64(b.map(|b| b[i])))
            .collect())
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        match self.bbox {
            Some(b) => {
                let r = Rect::new(coord! { x: b[0], y: b[1] }, coord! { x: b[2], y: b[3] });
                Ok(ScalarValue::Binary(Some(wkb(&Geometry::Polygon(
                    r.to_polygon(),
                ))?)))
            }
            None => Ok(ScalarValue::Binary(None)),
        }
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// Gather without merging: the counterpart to `st_union_agg` for data that
/// is not areal, and for when the parts must stay distinguishable.
#[derive(Debug, Default)]
struct CollectAcc {
    geoms: Vec<Geometry<f64>>,
}

impl CollectAcc {
    fn value(&self) -> DfResult<ScalarValue> {
        if self.geoms.is_empty() {
            return Ok(ScalarValue::Binary(None));
        }
        let g = Geometry::GeometryCollection(GeometryCollection(self.geoms.clone()));
        Ok(ScalarValue::Binary(Some(wkb(&g)?)))
    }
}

impl Accumulator for CollectAcc {
    fn update_batch(&mut self, values: &[ArrayRef]) -> DfResult<()> {
        self.geoms.extend(geoms(&values[0])?);
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DfResult<()> {
        // A partial state is a collection; flatten it back rather than
        // nesting collections inside collections.
        for g in geoms(&states[0])? {
            match g {
                Geometry::GeometryCollection(c) => self.geoms.extend(c.0),
                other => self.geoms.push(other),
            }
        }
        Ok(())
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        Ok(vec![self.value()?])
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        self.value()
    }

    /// The real footprint, not the size of the enum headers.
    ///
    /// A `Geometry` is a fat pointer to heap coordinate vectors, so
    /// counting `size_of::<Geometry>()` per element reported a few
    /// hundred bytes for an accumulator holding tens of megabytes of
    /// rings — and DataFusion's memory pool sizes spills from this
    /// number, so `st_collect` over a parcel layer never spilled and the
    /// process grew until it was killed.
    fn size(&self) -> usize {
        size_of_val(self)
            + self
                .geoms
                .iter()
                .map(|g| {
                    size_of::<Geometry<f64>>()
                        + g.coords_count() * size_of::<geo_types::Coord<f64>>()
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DataFusion sizes its spills from `Accumulator::size`, and a
    /// `Geometry` is a fat pointer: counting `size_of::<Geometry>()` per
    /// element reported a few hundred bytes for an accumulator holding
    /// tens of megabytes of rings, so `st_collect` over a parcel layer
    /// never spilled and the process grew until it was killed.
    #[test]
    fn collect_reports_the_memory_it_actually_holds() {
        let ring = |n: usize| {
            geo_types::LineString(
                (0..n)
                    .map(|i| coord! { x: i as f64, y: i as f64 })
                    .collect(),
            )
        };
        let mut small = CollectAcc::default();
        small.geoms.push(Geometry::Polygon(Polygon::new(ring(4), vec![])));
        let mut big = CollectAcc::default();
        big.geoms.push(Geometry::Polygon(Polygon::new(ring(100_000), vec![])));

        assert!(
            big.size() > 100_000 * size_of::<geo_types::Coord<f64>>(),
            "a 100k-vertex ring must report megabytes, not {} bytes",
            big.size()
        );
        assert!(big.size() > small.size() * 1000, "{} vs {}", big.size(), small.size());
    }
}
