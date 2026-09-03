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
    // Counts come from the file, and the detail cache is a downloaded
    // file that can be truncated or corrupt. Reserving from a bad u32
    // asks for gigabytes and aborts the process on the allocation
    // failure — instead of returning None and taking the refetch path
    // two lines down in `load_detail`. Every line costs at least its own
    // 4-byte count, every point exactly 8 bytes.
    let n_lines = take_u32(&mut off)? as usize;
    let mut lines = Vec::with_capacity(n_lines.min(b.len().saturating_sub(off) / 4));
    for _ in 0..n_lines {
        let n = take_u32(&mut off)? as usize;
        if n > b.len().saturating_sub(off) / 8 {
            return None;
        }
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

/// Project WGS84 polylines into world space under `display`. Vertices
/// that fail to project split the polyline: joining across the gap would
/// draw a segment through a region the projection does not cover.
///
/// The map takes these as meshes and the SVG export takes them as
/// polylines, so the splitting rule lives here and both get the same
/// lines.
pub fn project_overlay_polylines<'a>(
    display: &DisplayCrs,
    lines: impl Iterator<Item = &'a [(f64, f64)]>,
) -> Vec<geo_types::LineString<f64>> {
    let wgs = Crs::wgs84();
    let tr = BulkTransformer::new(&wgs, display);
    let mut out: Vec<geo_types::LineString<f64>> = Vec::new();
    let mut coords: Vec<geo_types::Coord<f64>> = Vec::new();
    for line in lines {
        coords.clear();
        let flush = |coords: &mut Vec<geo_types::Coord<f64>>,
                     out: &mut Vec<geo_types::LineString<f64>>| {
            if coords.len() >= 2 {
                out.push(geo_types::LineString(std::mem::take(coords)));
            }
            coords.clear();
        };
        for &(lon, lat) in line {
            let (mut x, mut y) = (lon, lat);
            let ok = tr.apply(&mut x, &mut y);
            if !ok {
                flush(&mut coords, &mut out);
                continue;
            }
            let w = display.world_from_projected(x, y);
            if w[0].is_finite() && w[1].is_finite() {
                coords.push(geo_types::Coord { x: w[0], y: w[1] });
            } else {
                flush(&mut coords, &mut out);
            }
        }
        flush(&mut coords, &mut out);
    }
    out
}

/// The same polylines, built into render chunks.
pub fn project_overlay_lines<'a>(
    display: &DisplayCrs,
    lines: impl Iterator<Item = &'a [(f64, f64)]>,
) -> Arc<Vec<ChunkMesh>> {
    let mut mb = MeshBuilder::default();
    for ls in project_overlay_polylines(display, lines) {
        mb.add(&geo_types::Geometry::LineString(ls), FeatureRef::INVALID);
    }
    Arc::new(mb.finish())
}

/// Build the coastline overlay for a display projection.
pub fn build_coastline(display: &DisplayCrs) -> Arc<Vec<ChunkMesh>> {
    project_overlay_lines(display, coastline_lines().iter().map(|l| l.as_slice()))
}

// --- zoom-dependent detail: Natural Earth 1:10m, fetched on demand ------

/// NEC1-packed 1:10m coastline (~3.3 MB), published as a release asset
/// (regenerate with `assets/make_coastline_bin.py 10m`).
pub const DETAIL_URL: &str =
    "https://github.com/gsueur/geopq-workbench/releases/download/assets/ne_10m_coastline.bin";

/// Which coastline generation the overlay chunks are built from.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CoastLevel {
    /// Embedded 1:50m — always available.
    #[default]
    Embedded,
    /// Fetched 1:10m — used when zoomed in, once available.
    Detailed,
}

/// Parsed WGS84 polylines, shared between the fetch thread and builds.
type SharedLines = Arc<Vec<Vec<(f64, f64)>>>;

enum DetailState {
    Idle,
    Fetching,
    Ready(SharedLines),
    /// Fetch failed: stay on the embedded coastline for this session
    /// (no retry storm; a restart tries again).
    Failed,
}

fn detail_state() -> &'static std::sync::Mutex<DetailState> {
    static S: OnceLock<std::sync::Mutex<DetailState>> = OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(DetailState::Idle))
}

/// `~/.config/geopq-workbench/ne_10m_coastline.bin` (tests: temp dir).
fn detail_cache_path() -> Option<std::path::PathBuf> {
    #[cfg(test)]
    {
        return Some(
            std::env::temp_dir()
                .join(format!("geopq_test_coast_{}", std::process::id()))
                .join("ne_10m_coastline.bin"),
        );
    }
    #[cfg(not(test))]
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| {
            std::path::PathBuf::from(h)
                .join(".config")
                .join("geopq-workbench")
                .join("ne_10m_coastline.bin")
        })
}

/// The 1:10m lines, when already fetched.
pub fn detailed_lines() -> Option<SharedLines> {
    match &*detail_state().lock().unwrap() {
        DetailState::Ready(l) => Some(l.clone()),
        _ => None,
    }
}

/// Kick off the one-time background fetch (disk cache first, then the
/// release asset). `on_ready` fires once on success — use it to request
/// a repaint. No-op while fetching, after success, or after a failure.
pub fn request_detailed(on_ready: impl FnOnce() + Send + 'static) {
    {
        let mut st = detail_state().lock().unwrap();
        match *st {
            DetailState::Idle => *st = DetailState::Fetching,
            _ => return,
        }
    }
    std::thread::spawn(move || {
        let loaded = load_detail();
        let mut st = detail_state().lock().unwrap();
        match loaded {
            Some(lines) => {
                log::info!(
                    "coastline: 1:10m detail ready ({} lines)",
                    lines.len()
                );
                *st = DetailState::Ready(Arc::new(lines));
                drop(st);
                on_ready();
            }
            None => {
                log::warn!("coastline: 1:10m fetch failed, staying on 1:50m");
                *st = DetailState::Failed;
            }
        }
    });
}

fn load_detail() -> Option<Vec<Vec<(f64, f64)>>> {
    let cache = detail_cache_path();
    if let Some(p) = &cache
        && let Ok(bytes) = std::fs::read(p)
    {
        if let Some(lines) = parse_nec1(&bytes) {
            return Some(lines);
        }
        let _ = std::fs::remove_file(p); // corrupt cache: refetch
    }
    log::info!("coastline: fetching 1:10m detail");
    let res = crate::data::source::http_agent().get(DETAIL_URL).call().ok()?;
    if res.status().as_u16() != 200 {
        return None;
    }
    let mut bytes = Vec::new();
    use std::io::Read;
    res.into_body().into_reader().read_to_end(&mut bytes).ok()?;
    let lines = parse_nec1(&bytes)?;
    if let Some(p) = &cache {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Temp + rename: a plain write interrupted at exit leaves a
        // half file that the next run reads, fails to parse, deletes and
        // refetches — a 20 MB download every start until it happens to
        // finish cleanly.
        let tmp = p.with_extension(format!("tmp-{}", std::process::id()));
        let wrote = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, p));
        if wrote.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Some(lines)
}

/// Build the coastline overlay at a given level; a not-yet-fetched
/// Detailed level falls back to the embedded lines.
pub fn build_coastline_at(display: &DisplayCrs, level: CoastLevel) -> Arc<Vec<ChunkMesh>> {
    match level {
        CoastLevel::Detailed => match detailed_lines() {
            Some(lines) => {
                project_overlay_lines(display, lines.iter().map(|l| l.as_slice()))
            }
            None => build_coastline(display),
        },
        CoastLevel::Embedded => build_coastline(display),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The detail cache is a downloaded file, so its counts are hostile
    /// input. Reserving from a corrupt u32 asks the allocator for
    /// gigabytes and aborts the process, instead of returning None and
    /// letting `load_detail` delete the cache and refetch.
    #[test]
    fn a_corrupt_cache_is_rejected_rather_than_reserved_for() {
        // Header claims 4 billion lines; the file holds none.
        let mut bad = b"NEC1".to_vec();
        bad.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_nec1(&bad).is_none());

        // One line claiming 4 billion points, with 8 bytes to give.
        let mut bad = b"NEC1".to_vec();
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&u32::MAX.to_le_bytes());
        bad.extend_from_slice(&[0u8; 8]);
        assert!(parse_nec1(&bad).is_none());

        // Truncated mid-point: still None, still no huge reservation.
        let mut bad = b"NEC1".to_vec();
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&3u32.to_le_bytes());
        bad.extend_from_slice(&[0u8; 12]);
        assert!(parse_nec1(&bad).is_none());

        // A well-formed one still parses.
        let mut ok = b"NEC1".to_vec();
        ok.extend_from_slice(&1u32.to_le_bytes());
        ok.extend_from_slice(&2u32.to_le_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            ok.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(parse_nec1(&ok).unwrap(), vec![vec![(1.0, 2.0), (3.0, 4.0)]]);
    }

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
    fn detail_falls_back_and_reads_cache() {
        // Before any fetch, the Detailed level builds the embedded lines.
        let display = DisplayCrs::hobo_dyer();
        let segs = |c: &Arc<Vec<ChunkMesh>>| -> usize {
            c.iter().map(|c| c.lines[0].segments.len()).sum()
        };
        let a = build_coastline(&display);
        let b = build_coastline_at(&display, CoastLevel::Detailed);
        assert_eq!(segs(&a), segs(&b));

        // A valid cached NEC1 file loads without touching the network.
        let p = detail_cache_path().unwrap();
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let mut bytes: Vec<u8> = b"NEC1".to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(3u32.to_le_bytes());
        for (x, y) in [(0.0f32, 0.0f32), (1.0, 1.0), (2.0, 0.5)] {
            bytes.extend(x.to_le_bytes());
            bytes.extend(y.to_le_bytes());
        }
        std::fs::write(&p, &bytes).unwrap();
        let lines = load_detail().expect("cache hit");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), 3);
        let _ = std::fs::remove_file(&p);
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
