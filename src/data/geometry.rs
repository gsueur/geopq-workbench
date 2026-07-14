use std::collections::HashMap;

use geo_types::{Geometry, LineString, Point, Polygon};
use lyon::math::point as lpoint;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};

/// Coordinate scale applied while tessellating with lyon: chunk-local world
/// coordinates are ~1e-6..5e-4, far below the ~unit scale lyon's internal
/// fixed-point representation expects, and the quantization shows up as
/// chord/sliver artifacts in fills. Scale up for tessellation, back down
/// for storage.
const TESS_SCALE: f32 = 1.0e7;

/// World-space chunk grid size. Vertices are stored as f32 offsets from the
/// chunk origin, keeping precision at deep zoom (1/2048 world ≈ 19.6 km at
/// the equator on Mercator; f32 offsets give millimeter resolution there).
/// Small cells also make per-chunk viewport culling effective at city zoom.
pub const CHUNK_WORLD: f64 = 1.0 / 2048.0;

/// Reference to one feature: its global row index in the source file.
/// 4 bytes, so the per-point pick index stays cheap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeatureRef {
    pub index: u32,
}

impl FeatureRef {
    /// Sentinel for meshes that are not pickable (highlight, graticule,
    /// points nested in geometry collections).
    pub const INVALID: FeatureRef = FeatureRef { index: u32::MAX };

    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

/// CPU-side geometry for one spatial chunk, in chunk-local f32 coordinates.
#[derive(Clone)]
pub struct ChunkMesh {
    pub origin: [f64; 2],
    pub fill_vertices: Vec<[f32; 2]>,
    pub fill_indices: Vec<u32>,
    /// Line segments per LOD; index 0 is full detail. Within each LOD the
    /// segments are sorted by their feature's extent (descending) so a
    /// prefix draw shows exactly the features larger than a pixel cutoff.
    pub lines: [LineLod; LINE_LODS_TOTAL],
    /// Per-LOD redirect: levels that would nearly duplicate a finer kept
    /// level are left empty and alias to it (identity when kept).
    pub line_alias: [u8; LINE_LODS_TOTAL],
    /// One instance per point: [x, y].
    pub point_instances: Vec<[f32; 2]>,
    /// Feature reference per point instance (parallel to `point_instances`).
    /// This doubles as the point pick index: points are found by scanning
    /// the chunks near the cursor instead of a per-feature R-tree, which
    /// keeps the index at 4 bytes/point.
    pub point_refs: Vec<FeatureRef>,
    /// Tight chunk-local bounds over all vertices ([minx, miny, maxx, maxy]),
    /// for per-chunk viewport culling. Computed in `MeshBuilder::finish`.
    pub bounds_local: [f32; 4],
}

/// One LOD level of line segments plus the feature-size prefix index.
#[derive(Default, Clone)]
pub struct LineLod {
    /// [x0, y0, x1, y1] per segment, sorted by feature extent descending
    /// (sorting happens in `MeshBuilder::finish`).
    pub segments: Vec<[f32; 4]>,
    /// Feature extent (world units) per segment; transient during build,
    /// cleared once `size_index` is computed.
    pub extents: Vec<f32>,
    /// (extent_threshold, prefix_count) pairs, thresholds descending:
    /// drawing `prefix_count` segments shows all features with
    /// extent ≥ threshold.
    pub size_index: Vec<(f32, u32)>,
}

impl LineLod {
    /// Number of segments to draw so that only features with extent ≥
    /// `cutoff` (world units) appear.
    #[allow(dead_code)] // exercised by the opt-in real-file benchmark
    pub fn count_for_cutoff(&self, cutoff: f64) -> u32 {
        let mut count = 0u32;
        for &(threshold, prefix) in &self.size_index {
            if (threshold as f64) >= cutoff {
                count = prefix;
            } else {
                break;
            }
        }
        count
    }
}

/// Number of simplified line LODs in addition to full detail.
pub const LINE_LOD_COUNT: usize = 9;
/// Total line arrays per chunk (full detail + LODs).
pub const LINE_LODS_TOTAL: usize = 1 + LINE_LOD_COUNT;

/// Features below `BUILD_CUTOFF_FACTOR × tolerance` are omitted from a LOD
/// at build time: with the renderer's 4 px minimum feature size they could
/// never be drawn from that level anyway (4 px at the band's deepest zoom
/// equals ≈2.83 × tolerance).
const BUILD_CUTOFF_FACTOR: f64 = 2.8;

/// If a LOD has more than this fraction of the previous kept level's
/// segments, it is aliased to that level instead of storing a near-copy.
const LOD_ALIAS_FRACTION: f64 = 0.7;

/// World-space simplification tolerance per LOD level k (1-based in the
/// chunk arrays): 1 px at zoom 16−k, the center of the one-zoom-wide band
/// that uses it. Max error at band edges ≈1.4 px, and drawn segments stay
/// around a pixel long — sub-pixel blended triangles are what kill the GPU.
pub const LINE_LOD_TOLERANCE: [f64; LINE_LOD_COUNT] = [
    1.0 / (256.0 * 32768.0), // k=1, z15 band (~4.8 m)
    1.0 / (256.0 * 16384.0), // k=2, z14 (~9.5 m)
    1.0 / (256.0 * 8192.0),  // k=3, z13 (~19 m)
    1.0 / (256.0 * 4096.0),  // k=4, z12 (~38 m)
    1.0 / (256.0 * 2048.0),  // k=5, z11 (~76 m)
    1.0 / (256.0 * 1024.0),  // k=6, z10 (~153 m)
    1.0 / (256.0 * 512.0),   // k=7, z9 (~306 m)
    1.0 / (256.0 * 256.0),   // k=8, z8 (~611 m)
    1.0 / (256.0 * 128.0),   // k=9, z7 and below (~1.2 km)
];

impl Default for ChunkMesh {
    fn default() -> Self {
        let mut line_alias = [0u8; LINE_LODS_TOTAL];
        for (i, a) in line_alias.iter_mut().enumerate() {
            *a = i as u8;
        }
        Self {
            origin: [0.0; 2],
            fill_vertices: Vec::new(),
            fill_indices: Vec::new(),
            lines: Default::default(),
            line_alias,
            point_instances: Vec::new(),
            point_refs: Vec::new(),
            bounds_local: [0.0; 4],
        }
    }
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.fill_indices.is_empty()
            && self.lines.iter().all(|l| l.segments.is_empty())
            && self.point_instances.is_empty()
    }
}

/// Result of adding a feature to the builder.
pub struct Added {
    pub bbox: [f64; 4],
    pub needs_rtree: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeomKind {
    Point,
    Line,
    Polygon,
    Mixed,
    Unknown,
}

impl GeomKind {
    pub fn label(&self) -> &'static str {
        match self {
            GeomKind::Point => "points",
            GeomKind::Line => "lines",
            GeomKind::Polygon => "polygons",
            GeomKind::Mixed => "mixed",
            GeomKind::Unknown => "empty",
        }
    }

    fn merge(self, other: GeomKind) -> GeomKind {
        match (self, other) {
            (GeomKind::Unknown, o) => o,
            (s, GeomKind::Unknown) => s,
            (s, o) if s == o => s,
            _ => GeomKind::Mixed,
        }
    }
}

/// Accumulates tessellated features into spatial chunks. One builder per
/// worker thread; merge with `merge_into`.
pub struct MeshBuilder {
    chunks: HashMap<(i64, i64), ChunkMesh>,
    tess: FillTessellator,
    scratch: VertexBuffers<[f32; 2], u32>,
    /// Reused LOD decimation buffer (one allocation per builder, not per ring).
    lod_scratch: Vec<geo_types::Coord<f64>>,
    pub bounds: [f64; 4],
    pub kind: GeomKind,
    pub fill_errors: usize,
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            tess: FillTessellator::new(),
            scratch: VertexBuffers::new(),
            lod_scratch: Vec::new(),
            bounds: [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            kind: GeomKind::Unknown,
            fill_errors: 0,
        }
    }
}

fn geom_bbox(geom: &Geometry<f64>) -> Option<[f64; 4]> {
    use geo::BoundingRect;
    let r = geom.bounding_rect()?;
    let (min, max) = (r.min(), r.max());
    if !(min.x.is_finite() && min.y.is_finite() && max.x.is_finite() && max.y.is_finite()) {
        return None;
    }
    Some([min.x, min.y, max.x, max.y])
}

fn expand(a: &mut [f64; 4], b: [f64; 4]) {
    a[0] = a[0].min(b[0]);
    a[1] = a[1].min(b[1]);
    a[2] = a[2].max(b[2]);
    a[3] = a[3].max(b[3]);
}

impl MeshBuilder {
    fn chunk_for(&mut self, bbox: [f64; 4]) -> &mut ChunkMesh {
        let cx = (bbox[0] + bbox[2]) * 0.5;
        let cy = (bbox[1] + bbox[3]) * 0.5;
        let key = (
            (cx / CHUNK_WORLD).floor() as i64,
            (cy / CHUNK_WORLD).floor() as i64,
        );
        self.chunks.entry(key).or_insert_with(|| ChunkMesh {
            origin: [key.0 as f64 * CHUNK_WORLD, key.1 as f64 * CHUNK_WORLD],
            ..Default::default()
        })
    }

    /// Add one feature (coordinates already in world space).
    /// Returns its world bbox and whether it must go into the R-tree
    /// (point-only features are indexed by their chunk instead).
    pub fn add(&mut self, geom: &Geometry<f64>, fref: FeatureRef) -> Option<Added> {
        let bbox = geom_bbox(geom)?;
        expand(&mut self.bounds, bbox);
        let needs_rtree = self.add_inner(geom, bbox, fref);
        Some(Added { bbox, needs_rtree })
    }

    fn add_inner(&mut self, geom: &Geometry<f64>, bbox: [f64; 4], fref: FeatureRef) -> bool {
        match geom {
            Geometry::Point(p) => {
                self.add_point(p, bbox, fref);
                false
            }
            Geometry::MultiPoint(mp) => {
                self.kind = self.kind.merge(GeomKind::Point);
                let chunk = self.chunk_for(bbox);
                let origin = chunk.origin;
                for p in &mp.0 {
                    chunk.point_instances.push(local(origin, p.x(), p.y()));
                    chunk.point_refs.push(fref);
                }
                false
            }
            Geometry::Line(l) => {
                let _ = fref;
                self.kind = self.kind.merge(GeomKind::Line);
                self.add_polyline(&[l.start, l.end], bbox, false);
                true
            }
            Geometry::LineString(ls) => {
                self.kind = self.kind.merge(GeomKind::Line);
                self.add_linestring(ls, bbox);
                true
            }
            Geometry::MultiLineString(mls) => {
                self.kind = self.kind.merge(GeomKind::Line);
                for ls in &mls.0 {
                    self.add_linestring(ls, bbox);
                }
                true
            }
            Geometry::Polygon(poly) => {
                self.kind = self.kind.merge(GeomKind::Polygon);
                self.add_polygon(poly, bbox);
                true
            }
            Geometry::MultiPolygon(mp) => {
                self.kind = self.kind.merge(GeomKind::Polygon);
                for poly in &mp.0 {
                    self.add_polygon(poly, bbox);
                }
                true
            }
            Geometry::GeometryCollection(gc) => {
                for g in &gc.0 {
                    if let Some(b) = geom_bbox(g) {
                        // Collections are picked via the R-tree; inner points
                        // get sentinel refs so the chunk scan skips them.
                        self.add_inner(g, b, FeatureRef::INVALID);
                    }
                }
                true
            }
            Geometry::Rect(r) => {
                self.add_inner(&Geometry::Polygon(r.to_polygon()), bbox, fref)
            }
            Geometry::Triangle(t) => {
                self.add_inner(&Geometry::Polygon(t.to_polygon()), bbox, fref)
            }
        }
    }

    fn add_point(&mut self, p: &Point<f64>, bbox: [f64; 4], fref: FeatureRef) {
        self.kind = self.kind.merge(GeomKind::Point);
        let chunk = self.chunk_for(bbox);
        let o = chunk.origin;
        chunk.point_instances.push(local(o, p.x(), p.y()));
        chunk.point_refs.push(fref);
    }

    // ------------------------------------------------------------------
    // Slice entry points (bulk GeoArrow path — no geo-types intermediate).
    // The caller is responsible for expanding bounds via `expand_feature`.
    // ------------------------------------------------------------------

    pub(crate) fn expand_feature(&mut self, bbox: [f64; 4]) {
        expand(&mut self.bounds, bbox);
    }

    pub(crate) fn add_point_xy(&mut self, x: f64, y: f64, fref: FeatureRef) {
        self.kind = self.kind.merge(GeomKind::Point);
        let bbox = [x, y, x, y];
        let chunk = self.chunk_for(bbox);
        let o = chunk.origin;
        chunk.point_instances.push(local(o, x, y));
        chunk.point_refs.push(fref);
    }

    /// MultiPoint: every coordinate is one point instance of the feature.
    pub(crate) fn add_points_world(
        &mut self,
        pts: &[geo_types::Coord<f64>],
        bbox: [f64; 4],
        fref: FeatureRef,
    ) {
        self.kind = self.kind.merge(GeomKind::Point);
        let chunk = self.chunk_for(bbox);
        let o = chunk.origin;
        for p in pts {
            chunk.point_instances.push(local(o, p.x, p.y));
            chunk.point_refs.push(fref);
        }
    }

    pub(crate) fn add_polyline_world(
        &mut self,
        pts: &[geo_types::Coord<f64>],
        bbox: [f64; 4],
        closed: bool,
    ) {
        self.kind = self.kind.merge(GeomKind::Line);
        self.add_polyline(pts, bbox, closed);
    }

    fn add_linestring(&mut self, ls: &LineString<f64>, bbox: [f64; 4]) {
        self.add_polyline(&ls.0, bbox, false);
    }

    /// Emit a polyline into the chunk at full detail plus each LOD level
    /// (radial-distance decimation; whole feature dropped at levels whose
    /// tolerance exceeds its extent). Each segment carries its feature's
    /// extent so the renderer can hide sub-pixel features.
    fn add_polyline(&mut self, pts: &[geo_types::Coord<f64>], bbox: [f64; 4], closed: bool) {
        if pts.len() < 2 {
            return;
        }
        let extent = ((bbox[2] - bbox[0]).max(bbox[3] - bbox[1])) as f32;
        let chunk = self.chunk_for(bbox);
        let o = chunk.origin;
        for w in pts.windows(2) {
            let a = local(o, w[0].x, w[0].y);
            let b = local(o, w[1].x, w[1].y);
            chunk.lines[0].segments.push([a[0], a[1], b[0], b[1]]);
            chunk.lines[0].extents.push(extent);
        }
        let mut kept = std::mem::take(&mut self.lod_scratch);
        for (lvl, &tol) in LINE_LOD_TOLERANCE.iter().enumerate() {
            if (extent as f64) < tol * BUILD_CUTOFF_FACTOR {
                continue; // could never pass the 4 px draw cutoff from here
            }
            decimate_polyline(pts, tol, &mut kept);
            let min_pts = if closed { 4 } else { 2 };
            if kept.len() < min_pts {
                continue;
            }
            let chunk = self.chunk_for(bbox);
            for w in kept.windows(2) {
                let a = local(o, w[0].x, w[0].y);
                let b = local(o, w[1].x, w[1].y);
                chunk.lines[1 + lvl].segments.push([a[0], a[1], b[0], b[1]]);
                chunk.lines[1 + lvl].extents.push(extent);
            }
        }
        self.lod_scratch = kept;
    }

    fn add_polygon(&mut self, poly: &Polygon<f64>, bbox: [f64; 4]) {
        let rings: Vec<&[geo_types::Coord<f64>]> = std::iter::once(poly.exterior())
            .chain(poly.interiors())
            .map(|r| r.0.as_slice())
            .collect();
        self.add_polygon_parts(&rings, bbox);
    }

    /// Rings of one polygon (first = exterior), coordinates in world space.
    /// Ring winding is normalized here (exterior CCW, holes CW): real-world
    /// parcel data has mixed winding, and the non-zero fill rule needs
    /// consistency to fill correctly.
    pub(crate) fn add_polygon_parts(
        &mut self,
        rings: &[&[geo_types::Coord<f64>]],
        bbox: [f64; 4],
    ) {
        self.kind = self.kind.merge(GeomKind::Polygon);
        let origin = {
            let chunk = self.chunk_for(bbox);
            chunk.origin
        };

        // Fill via lyon, in chunk-local coordinates.
        let mut path = Path::builder();
        let mut any = false;
        for (ri, ring) in rings.iter().enumerate() {
            if ring.len() < 4 {
                continue;
            }
            // Shoelace winding; rings carry the duplicate closing point.
            let n = ring.len() - 1;
            let mut area2 = 0.0f64;
            for w in ring.windows(2) {
                area2 += w[0].x * w[1].y - w[1].x * w[0].y;
            }
            let want_ccw = ri == 0;
            let emit = |path: &mut lyon::path::path::Builder, idx: &mut dyn Iterator<Item = usize>| {
                let first = idx.next().expect("ring has vertices");
                let p = local(origin, ring[first].x, ring[first].y);
                path.begin(lpoint(p[0] * TESS_SCALE, p[1] * TESS_SCALE));
                for i in idx {
                    let p = local(origin, ring[i].x, ring[i].y);
                    path.line_to(lpoint(p[0] * TESS_SCALE, p[1] * TESS_SCALE));
                }
                path.end(true);
            };
            if (area2 > 0.0) == want_ccw {
                emit(&mut path, &mut (0..n));
            } else {
                emit(&mut path, &mut (0..n).rev());
            }
            any = true;
        }
        if any {
            let path = path.build();
            self.scratch.vertices.clear();
            self.scratch.indices.clear();
            // Non-zero rule: slightly overlapping/self-touching rings
            // (ubiquitous in parcel data) must not punch even-odd holes.
            // Tolerance is in TESS_SCALE units: 0.1 here ≈ 1e-8 world ≈ 40 cm.
            let opts = FillOptions::default()
                .with_tolerance(0.1)
                .with_fill_rule(FillRule::NonZero);
            let res = self.tess.tessellate_path(
                &path,
                &opts,
                &mut BuffersBuilder::new(&mut self.scratch, |v: FillVertex| {
                    let p = v.position();
                    [p.x / TESS_SCALE, p.y / TESS_SCALE]
                }),
            );
            match res {
                Ok(()) => {
                    let chunk = self.chunks.get_mut(&key_of(origin)).expect("chunk exists");
                    let base = chunk.fill_vertices.len() as u32;
                    chunk.fill_vertices.extend_from_slice(&self.scratch.vertices);
                    chunk
                        .fill_indices
                        .extend(self.scratch.indices.iter().map(|i| i + base));
                }
                Err(_) => self.fill_errors += 1,
            }
        }

        // Outlines as line segments (with LODs).
        for ring in rings {
            self.add_polyline(ring, bbox, true);
        }
    }

    /// Merge another builder's output into this one (rayon reduce step).
    pub fn merge_into(&mut self, other: MeshBuilder) {
        for (key, mesh) in other.chunks {
            match self.chunks.entry(key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(mesh);
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let dst = e.get_mut();
                    let base = dst.fill_vertices.len() as u32;
                    dst.fill_vertices.extend_from_slice(&mesh.fill_vertices);
                    dst.fill_indices
                        .extend(mesh.fill_indices.iter().map(|i| i + base));
                    for (d, m) in dst.lines.iter_mut().zip(&mesh.lines) {
                        d.segments.extend_from_slice(&m.segments);
                        d.extents.extend_from_slice(&m.extents);
                    }
                    dst.point_instances.extend_from_slice(&mesh.point_instances);
                    dst.point_refs.extend_from_slice(&mesh.point_refs);
                }
            }
        }
        expand(&mut self.bounds, other.bounds);
        self.kind = self.kind.merge(other.kind);
        self.fill_errors += other.fill_errors;
    }

    pub fn finish(self) -> Vec<ChunkMesh> {
        let mut chunks: Vec<ChunkMesh> = self
            .chunks
            .into_values()
            .filter(|c| !c.is_empty())
            .flat_map(|c| split_oversized(c, &SplitCaps::default()))
            .collect();
        for c in &mut chunks {
            // Shuffle points (with refs) and line segments so a buffer
            // prefix is a uniform random sample: the renderer draws only as
            // much as the chunk's on-screen area warrants.
            let mut state =
                (c.origin[0].to_bits() ^ c.origin[1].to_bits().rotate_left(17)) | 1;
            let mut rand = move || {
                // xorshift64*
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_F491_4F6C_DD1D)
            };
            let n = c.point_instances.len();
            if n > 1 {
                for i in (1..n).rev() {
                    let j = (rand() % (i as u64 + 1)) as usize;
                    c.point_instances.swap(i, j);
                    c.point_refs.swap(i, j);
                }
            }
            let _ = &mut rand;
            // Alias LODs that barely improve on the previous kept level.
            let mut last_kept = 0usize;
            for k in 1..LINE_LODS_TOTAL {
                let prev_len = c.lines[last_kept].segments.len();
                let len = c.lines[k].segments.len();
                if prev_len > 0 && len as f64 > prev_len as f64 * LOD_ALIAS_FRACTION {
                    c.lines[k] = LineLod::default();
                    c.line_alias[k] = last_kept as u8;
                } else {
                    c.line_alias[k] = k as u8;
                    last_kept = k;
                }
            }
            for lod in &mut c.lines {
                sort_and_index_lod(lod);
            }
            // Tight local bounds over all vertex data.
            let mut b = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
            let mut grow = |x: f32, y: f32| {
                b[0] = b[0].min(x);
                b[1] = b[1].min(y);
                b[2] = b[2].max(x);
                b[3] = b[3].max(y);
            };
            for v in &c.fill_vertices {
                grow(v[0], v[1]);
            }
            for l in &c.lines[0].segments {
                grow(l[0], l[1]);
                grow(l[2], l[3]);
            }
            for p in &c.point_instances {
                grow(p[0], p[1]);
            }
            c.bounds_local = b;
        }
        chunks
    }
}

fn key_of(origin: [f64; 2]) -> (i64, i64) {
    (
        (origin[0] / CHUNK_WORLD).round() as i64,
        (origin[1] / CHUNK_WORLD).round() as i64,
    )
}

#[inline]
fn local(origin: [f64; 2], x: f64, y: f64) -> [f32; 2] {
    [(x - origin[0]) as f32, (y - origin[1]) as f32]
}


/// Per-buffer element caps so no single GPU buffer exceeds ~96 MB
/// (wgpu's default max_buffer_size is 256 MB; dense datasets can pile
/// hundreds of MB of segments into one spatial chunk).
pub struct SplitCaps {
    pub fill_vertices: usize,
    pub fill_indices: usize,
    pub line_instances: usize,
    pub point_instances: usize,
}

impl Default for SplitCaps {
    fn default() -> Self {
        Self {
            fill_vertices: 12_000_000,  // 96 MB @ 8 B
            fill_indices: 24_000_000,   // 96 MB @ 4 B
            line_instances: 6_000_000,  // 96 MB @ 16 B
            point_instances: 12_000_000, // 96 MB @ 8 B
        }
    }
}

/// Split a chunk whose arrays exceed the caps into several chunks sharing
/// the same origin. Fill meshes are re-emitted triangle by triangle with a
/// vertex remap so indices stay valid within each part.
pub fn split_oversized(c: ChunkMesh, caps: &SplitCaps) -> Vec<ChunkMesh> {
    let within = c.fill_vertices.len() <= caps.fill_vertices
        && c.fill_indices.len() <= caps.fill_indices
        && c.point_instances.len() <= caps.point_instances
        && c.lines.iter().all(|l| l.segments.len() <= caps.line_instances);
    if within {
        return vec![c];
    }

    let origin = c.origin;
    let alias = c.line_alias;
    let blank = move || ChunkMesh {
        origin,
        line_alias: alias,
        ..Default::default()
    };
    let mut out: Vec<ChunkMesh> = Vec::new();
    let mut cur = blank();

    // Fills: triangle-wise re-emission with per-part vertex remap.
    let mut remap: Vec<u32> = vec![u32::MAX; c.fill_vertices.len()];
    for tri in c.fill_indices.chunks_exact(3) {
        if cur.fill_indices.len() + 3 > caps.fill_indices
            || cur.fill_vertices.len() + 3 > caps.fill_vertices
        {
            out.push(std::mem::replace(&mut cur, blank()));
            remap.fill(u32::MAX);
        }
        for &i in tri {
            let slot = &mut remap[i as usize];
            if *slot == u32::MAX {
                *slot = cur.fill_vertices.len() as u32;
                cur.fill_vertices.push(c.fill_vertices[i as usize]);
            }
            cur.fill_indices.push(*slot);
        }
    }
    drop(remap);

    // Lines and points: plain slicing (instances are independent; extents
    // stay parallel and the size index is rebuilt in finish()).
    for lvl in 0..LINE_LODS_TOTAL {
        let mut i = 0;
        while i < c.lines[lvl].segments.len() {
            let room = caps
                .line_instances
                .saturating_sub(cur.lines[lvl].segments.len());
            if room == 0 {
                out.push(std::mem::replace(&mut cur, blank()));
                continue;
            }
            let take = room.min(c.lines[lvl].segments.len() - i);
            cur.lines[lvl]
                .segments
                .extend_from_slice(&c.lines[lvl].segments[i..i + take]);
            cur.lines[lvl]
                .extents
                .extend_from_slice(&c.lines[lvl].extents[i..i + take]);
            i += take;
        }
    }
    let mut i = 0;
    while i < c.point_instances.len() {
        let room = caps.point_instances.saturating_sub(cur.point_instances.len());
        if room == 0 {
            out.push(std::mem::replace(&mut cur, blank()));
            continue;
        }
        let take = room.min(c.point_instances.len() - i);
        cur.point_instances
            .extend_from_slice(&c.point_instances[i..i + take]);
        cur.point_refs.extend_from_slice(&c.point_refs[i..i + take]);
        i += take;
    }

    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Sort a LOD's segments by feature extent (descending) and build the
/// prefix index used for the per-pixel feature-size cutoff at draw time.
fn sort_and_index_lod(lod: &mut LineLod) {
    let n = lod.segments.len();
    if n == 0 {
        lod.extents = Vec::new();
        return;
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        lod.extents[b as usize]
            .partial_cmp(&lod.extents[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    lod.segments = order
        .iter()
        .map(|&i| lod.segments[i as usize])
        .collect();
    let extents: Vec<f32> = order.iter().map(|&i| lod.extents[i as usize]).collect();

    // Geometric thresholds from the largest extent downward.
    lod.size_index.clear();
    let max_e = extents[0].max(1e-12);
    let min_e = extents[n - 1].max(1e-12);
    let mut threshold = max_e;
    let mut idx = 0usize;
    for _ in 0..28 {
        threshold *= 0.5;
        while idx < n && extents[idx] >= threshold {
            idx += 1;
        }
        lod.size_index.push((threshold, idx as u32));
        if threshold <= min_e {
            break;
        }
    }
    if lod.size_index.last().map(|&(_, c)| c) != Some(n as u32) {
        lod.size_index.push((0.0, n as u32));
    }
    lod.extents = Vec::new();
}

/// Radial-distance decimation: keep a vertex only if it is at least `tol`
/// away from the last kept one; endpoints always kept.
fn decimate_polyline(
    pts: &[geo_types::Coord<f64>],
    tol: f64,
    out: &mut Vec<geo_types::Coord<f64>>,
) {
    out.clear();
    let tol2 = tol * tol;
    let mut last = pts[0];
    out.push(last);
    for p in &pts[1..pts.len() - 1] {
        let (dx, dy) = (p.x - last.x, p.y - last.y);
        if dx * dx + dy * dy >= tol2 {
            out.push(*p);
            last = *p;
        }
    }
    out.push(pts[pts.len() - 1]);
}

#[cfg(test)]
mod lod_tests {
    use super::*;

    #[test]
    fn polyline_lods_simplify_and_drop() {
        let mut mb = MeshBuilder::default();
        // A ring with extent 6e-5 world (~2.4 km) and vertices ~9.4e-7 apart.
        let n = 200;
        let ring: Vec<geo_types::Coord<f64>> = (0..=n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                geo_types::Coord {
                    x: 0.5 + 3e-5 * a.cos(),
                    y: 0.5 + 3e-5 * a.sin(),
                }
            })
            .collect();
        let poly = geo_types::Polygon::new(geo_types::LineString(ring), vec![]);
        mb.add(&geo_types::Geometry::Polygon(poly), FeatureRef::INVALID);
        let chunks = mb.finish();
        let count = |lvl: usize| -> usize {
            chunks
                .iter()
                .map(|c| {
                    let target = c.line_alias[lvl] as usize;
                    c.lines[target].segments.len()
                })
                .sum()
        };
        assert_eq!(count(0), 200);
        // Effective segment counts never increase with coarseness.
        for k in 1..LINE_LODS_TOTAL {
            assert!(
                count(k) <= count(k - 1),
                "lod {k}: {} > {}",
                count(k),
                count(k - 1)
            );
        }
        // A coarse level with tolerance well above vertex spacing must
        // simplify substantially while keeping a closed ring.
        assert!(count(5) >= 4 && count(5) < 100, "lod5 {}", count(5));
        // Levels whose build cutoff exceeds the feature extent store nothing
        // (extent 6e-5 < 2.8 × tol for the coarsest levels).
        let direct: usize = chunks.iter().map(|c| c.lines[9].segments.len()).sum();
        let aliased_target: usize = chunks
            .iter()
            .map(|c| c.line_alias[9] as usize)
            .max()
            .unwrap_or(9);
        assert!(
            direct == 0 || aliased_target < 9,
            "coarsest level should be empty or aliased"
        );

        // A tiny ring (extent 8e-6) drops from every level with
        // 2.8 × tol > 8e-6 but stays at full detail.
        let mut mb = MeshBuilder::default();
        let tiny: Vec<geo_types::Coord<f64>> = (0..=20)
            .map(|i| {
                let a = i as f64 / 20.0 * std::f64::consts::TAU;
                geo_types::Coord {
                    x: 0.5 + 4e-6 * a.cos(),
                    y: 0.5 + 4e-6 * a.sin(),
                }
            })
            .collect();
        let poly = geo_types::Polygon::new(geo_types::LineString(tiny), vec![]);
        mb.add(&geo_types::Geometry::Polygon(poly), FeatureRef::INVALID);
        let chunks = mb.finish();
        let full: usize = chunks.iter().map(|c| c.lines[0].segments.len()).sum();
        assert_eq!(full, 20);
        // 2.8 × tol(k=5, 76 m ≈ 1.9e-6) = 5.4e-6 < 8e-6 keeps it;
        // 2.8 × tol(k=6, 153 m) = 1.1e-5 > 8e-6 drops it.
        let l6: usize = chunks
            .iter()
            .map(|c| c.lines[c.line_alias[6] as usize].segments.len())
            .sum();
        let l6_direct: usize = chunks.iter().map(|c| c.lines[6].segments.len()).sum();
        assert!(
            l6_direct == 0,
            "tiny feature must be absent from coarse level, got {l6_direct}"
        );
        let _ = l6;
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn oversized_chunk_splits_and_preserves_data() {
        let mut c = ChunkMesh {
            origin: [0.5, 0.5],
            ..Default::default()
        };
        // 10 quads: 4 verts + 6 indices each (verts shared within a quad).
        for q in 0..10u32 {
            let base = c.fill_vertices.len() as u32;
            let x = q as f32;
            c.fill_vertices.extend_from_slice(&[
                [x, 0.0],
                [x + 1.0, 0.0],
                [x + 1.0, 1.0],
                [x, 1.0],
            ]);
            c.fill_indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        for i in 0..25u32 {
            c.lines[0].segments.push([i as f32, 0.0, i as f32, 1.0]);
            c.lines[0].extents.push(1.0);
        }
        for i in 0..30u32 {
            c.point_instances.push([i as f32, 0.0]);
            c.point_refs.push(FeatureRef { index: i });
        }

        let caps = SplitCaps {
            fill_vertices: 12,
            fill_indices: 18,
            line_instances: 10,
            point_instances: 8,
        };
        let parts = split_oversized(c, &caps);
        assert!(parts.len() > 1);

        let tri_count: usize = parts.iter().map(|p| p.fill_indices.len() / 3).sum();
        assert_eq!(tri_count, 20);
        let lines: usize = parts.iter().map(|p| p.lines[0].segments.len()).sum();
        assert_eq!(lines, 25);
        let pts: usize = parts.iter().map(|p| p.point_instances.len()).sum();
        assert_eq!(pts, 30);
        for p in &parts {
            assert!(p.fill_vertices.len() <= caps.fill_vertices);
            assert!(p.fill_indices.len() <= caps.fill_indices);
            assert!(p.lines[0].segments.len() <= caps.line_instances);
            assert!(p.point_instances.len() <= caps.point_instances);
            assert_eq!(p.point_instances.len(), p.point_refs.len());
            assert_eq!(p.origin, [0.5, 0.5]);
            // Every index must reference an existing vertex.
            for &i in &p.fill_indices {
                assert!((i as usize) < p.fill_vertices.len());
            }
        }
        // Point refs preserved in order.
        let refs: Vec<u32> = parts
            .iter()
            .flat_map(|p| p.point_refs.iter().map(|r| r.index))
            .collect();
        assert_eq!(refs, (0..30).collect::<Vec<_>>());
    }
}
