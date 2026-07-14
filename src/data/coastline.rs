//! Embedded Natural Earth 1:50m coastline, and the shared machinery for
//! projecting built-in WGS84 overlay polylines (coastline, graticule) into
//! the current display projection.
//!
//! Data: Natural Earth (public domain), converted to a compact binary:
//! `"NEC1" u32 line_count { u32 n, n × (f32 lon, f32 lat) }`, little endian.

use std::sync::{Arc, OnceLock};

use crate::data::crs::{BulkTransformer, Crs, DisplayCrs};
use crate::data::geometry::{ChunkMesh, FeatureRef, MeshBuilder};

static COASTLINE_BIN: &[u8] = include_bytes!("../../assets/ne_50m_coastline.bin");

/// Parse the embedded coastline once.
pub fn coastline_lines() -> &'static [Vec<(f64, f64)>] {
    static LINES: OnceLock<Vec<Vec<(f64, f64)>>> = OnceLock::new();
    LINES.get_or_init(|| parse_nec1(COASTLINE_BIN).expect("embedded coastline is valid"))
}

fn parse_nec1(b: &[u8]) -> Option<Vec<Vec<(f64, f64)>>> {
    let mut off = 0usize;
    let take_u32 = |off: &mut usize| -> Option<u32> {
        let v = u32::from_le_bytes(b.get(*off..*off + 4)?.try_into().ok()?);
        *off += 4;
        Some(v)
    };
    if b.get(0..4)? != b"NEC1" {
        return None;
    }
    off += 4;
    let n_lines = take_u32(&mut off)? as usize;
    let mut lines = Vec::with_capacity(n_lines);
    for _ in 0..n_lines {
        let n = take_u32(&mut off)? as usize;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            let x = f32::from_le_bytes(b.get(off..off + 4)?.try_into().ok()?);
            let y = f32::from_le_bytes(b.get(off + 4..off + 8)?.try_into().ok()?);
            off += 8;
            pts.push((x as f64, y as f64));
        }
        lines.push(pts);
    }
    Some(lines)
}

/// Project WGS84 polylines into world space under `display` and build
/// render chunks. Vertices that fail to project split the polyline.
pub fn project_overlay_lines<'a>(
    display: &DisplayCrs,
    lines: impl Iterator<Item = &'a [(f64, f64)]>,
) -> Arc<Vec<ChunkMesh>> {
    let wgs = Crs::wgs84();
    let tr = BulkTransformer::new(&wgs, display);
    let mut mb = MeshBuilder::default();
    let mut coords: Vec<geo_types::Coord<f64>> = Vec::new();
    for line in lines {
        coords.clear();
        let flush = |coords: &mut Vec<geo_types::Coord<f64>>, mb: &mut MeshBuilder| {
            if coords.len() >= 2 {
                mb.add(
                    &geo_types::Geometry::LineString(geo_types::LineString(coords.clone())),
                    FeatureRef::INVALID,
                );
            }
            coords.clear();
        };
        for &(lon, lat) in line {
            let (mut x, mut y) = (lon, lat);
            let ok = tr.apply(&mut x, &mut y);
            if !ok {
                flush(&mut coords, &mut mb);
                continue;
            }
            let w = display.world_from_projected(x, y);
            if w[0].is_finite() && w[1].is_finite() {
                coords.push(geo_types::Coord { x: w[0], y: w[1] });
            } else {
                flush(&mut coords, &mut mb);
            }
        }
        flush(&mut coords, &mut mb);
    }
    Arc::new(mb.finish())
}

/// Build the coastline overlay for a display projection.
pub fn build_coastline(display: &DisplayCrs) -> Arc<Vec<ChunkMesh>> {
    project_overlay_lines(display, coastline_lines().iter().map(|l| l.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_coastline_parses() {
        let lines = coastline_lines();
        let pts: usize = lines.iter().map(|l| l.len()).sum();
        assert!(lines.len() > 1000, "{} lines", lines.len());
        assert!(pts > 50_000, "{pts} points");
        // All coordinates are plausible lon/lat.
        for l in lines.iter() {
            for &(lon, lat) in l {
                assert!((-180.01..=180.01).contains(&lon), "lon {lon}");
                assert!((-90.01..=90.01).contains(&lat), "lat {lat}");
            }
        }
    }

    #[test]
    fn coastline_builds_in_all_projections() {
        for display in [
            DisplayCrs::hobo_dyer(),
            DisplayCrs::winkel_tripel(),
            DisplayCrs::mercator(),
            DisplayCrs::new(Crs::wgs84()),
        ] {
            let chunks = build_coastline(&display);
            let segs: usize = chunks.iter().map(|c| c.lines[0].segments.len()).sum();
            assert!(segs > 50_000, "{}: only {segs} segments", display.name);
        }
    }
}
