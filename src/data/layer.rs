use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::Color32;
use rstar::{RTree, RTreeObject, AABB};

use super::crs::Crs;
use super::geometry::{ChunkMesh, GeomKind};
use super::info::FileInfo;
use super::store::FeatureStore;

pub use super::geometry::FeatureRef;

/// R-tree entry: feature bbox in world space.
#[derive(Clone, Debug)]
pub struct PickItem {
    pub bbox: [f64; 4],
    pub feature: FeatureRef,
}

impl RTreeObject for PickItem {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[3]])
    }
}

/// Per-row-group spatial extents, in the layer's data CRS.
#[derive(Clone, Debug)]
pub struct RgBboxes {
    /// Where the boxes come from: parquet native geospatial statistics,
    /// GeoParquet 1.1 covering-column statistics, or computed at load.
    pub source: String,
    pub boxes: Vec<[f64; 4]>,
    /// Average number of *other* row-group boxes each box intersects.
    /// Raw counts aren't comparable across row-group counts; judge
    /// clustering with [`Self::overlap_frac`].
    pub avg_overlap: f64,
}

impl RgBboxes {
    /// Overlap as a fraction of the possible overlaps (0 = disjoint boxes,
    /// 1 = every box intersects every other). Comparable across row-group
    /// counts, unlike the raw average. Reference points: Hilbert-sorted
    /// 65k-row groups land at 13–25% (adjacent groups necessarily touch),
    /// attribute-ordered data ~35%, spatially random data ~100%.
    pub fn overlap_frac(&self) -> f64 {
        self.avg_overlap / (self.boxes.len().max(2) - 1) as f64
    }

    /// Heuristic: would a spatial-order rewrite (or finer row groups)
    /// improve pruning?
    pub fn poorly_clustered(&self) -> bool {
        self.overlap_frac() > 0.3
    }
}

#[derive(Clone, Debug)]
pub struct LayerStyle {
    pub visible: bool,
    /// Draw the row-group bounding boxes overlay.
    pub show_rg_bboxes: bool,
    pub color: Color32,
    pub line_width_px: f32,
    pub point_radius_px: f32,
    pub fill_opacity: f32,
    pub opacity: f32,
}

impl LayerStyle {
    pub fn new(color: Color32) -> Self {
        Self {
            visible: true,
            show_rg_bboxes: false,
            color,
            line_width_px: 1.2,
            point_radius_px: 3.0,
            fill_opacity: 0.35,
            opacity: 1.0,
        }
    }
}

const PALETTE: [Color32; 8] = [
    Color32::from_rgb(31, 119, 180),
    Color32::from_rgb(255, 127, 14),
    Color32::from_rgb(44, 160, 44),
    Color32::from_rgb(214, 39, 40),
    Color32::from_rgb(148, 103, 189),
    Color32::from_rgb(23, 190, 207),
    Color32::from_rgb(227, 119, 194),
    Color32::from_rgb(188, 189, 34),
];

pub fn palette_color(i: usize) -> Color32 {
    PALETTE[i % PALETTE.len()]
}

#[derive(Clone, Debug, Default)]
pub struct LoadStats {
    pub read_ms: u64,
    pub build_ms: u64,
    #[allow(dead_code)]
    pub rows: usize,
    pub bad_geoms: usize,
}

/// Geometry + spatial index produced for one display projection.
pub struct LayerGeometry {
    pub chunks: Arc<Vec<ChunkMesh>>,
    pub rtree: Arc<RTree<PickItem>>,
    /// World-space bounds in the current display projection.
    pub bounds_world: [f64; 4],
    pub kind: GeomKind,
}

pub struct VectorLayer {
    pub id: u64,
    /// Bumped whenever geometry is rebuilt (projection change) so the
    /// renderer knows to re-upload GPU buffers.
    pub generation: u64,
    pub name: String,
    pub path: PathBuf,
    /// Lazy row access to the source file (attributes, WKB, rebuilds).
    pub store: Arc<FeatureStore>,
    pub crs: Crs,
    /// Geometry sections: index 0 from the initial load, later entries from
    /// row-group refinement appends. Projection rebuilds consolidate back
    /// to a single section.
    pub sections: Vec<LayerGeometry>,
    pub style: LayerStyle,
    pub feature_count: usize,
    pub stats: LoadStats,
    pub info: FileInfo,
    /// Row-group spatial extents (data CRS); None only if nothing could be
    /// derived (e.g. zero-row file).
    pub rg_bboxes: Option<RgBboxes>,
    /// Row groups currently decoded into sections.
    pub loaded_rgs: Vec<u32>,
}

impl VectorLayer {
    /// Union of section bounds (world space).
    pub fn bounds_world(&self) -> [f64; 4] {
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for s in &self.sections {
            b[0] = b[0].min(s.bounds_world[0]);
            b[1] = b[1].min(s.bounds_world[1]);
            b[2] = b[2].max(s.bounds_world[2]);
            b[3] = b[3].max(s.bounds_world[3]);
        }
        if b[0].is_finite() {
            b
        } else {
            [0.0, 0.0, 1.0, 1.0]
        }
    }

    pub fn kind(&self) -> super::geometry::GeomKind {
        self.sections
            .first()
            .map(|s| s.kind)
            .unwrap_or(super::geometry::GeomKind::Unknown)
    }

    pub fn total_rgs(&self) -> usize {
        self.rg_bboxes.as_ref().map(|r| r.boxes.len()).unwrap_or(0)
    }

    pub fn is_partial(&self) -> bool {
        let total = self.total_rgs();
        total > 0 && self.loaded_rgs.len() < total
    }
}
