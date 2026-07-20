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

/// Decode state of one row group.
#[derive(Clone, Debug, Default)]
pub enum GroupLoad {
    /// Nothing decoded.
    #[default]
    None,
    /// Per-feature selection: only rows whose covering bbox intersected
    /// `rect` (data CRS) were decoded; `ranges` are the group-relative
    /// [start, end) row spans that cover them.
    Rows { ranges: Vec<(u32, u32)>, rect: [f64; 4] },
    /// Decimated preview: every `stride`-th row was decoded (a load that
    /// would have exceeded the row budget), restricted to the features
    /// intersecting `rect` (data CRS) first when the load was
    /// viewport-filtered. Never covers a viewport, so zooming in refines
    /// it with real rows.
    Preview { stride: u32, rect: Option<[f64; 4]> },
    /// Whole group decoded.
    Full,
}

impl GroupLoad {
    pub fn is_full(&self) -> bool {
        matches!(self, GroupLoad::Full)
    }

    /// Does the loaded state already cover features intersecting `rect`?
    pub fn covers(&self, rect: [f64; 4]) -> bool {
        match self {
            GroupLoad::None | GroupLoad::Preview { .. } => false,
            GroupLoad::Full => true,
            GroupLoad::Rows { rect: r, .. } => {
                r[0] <= rect[0] && r[1] <= rect[1] && r[2] >= rect[2] && r[3] >= rect[3]
            }
        }
    }
}

/// Number of style bins (chunk meshes carry a u8 bin; 0 is the default
/// bin used when no data-driven styling is active).
pub const STYLE_BINS: usize = 16;

/// Data-driven styling: color features by an attribute column.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleBy {
    pub column: String,
    pub ramp: Ramp,
    pub mode: StyleMode,
    /// Rows that were loaded when data-dependent classes were computed
    /// (quantiles/std-dev/Jenks classify only what is loaded — panning or
    /// zooming afterwards can make the classes stale).
    pub classified_rows: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StyleMode {
    /// Numeric values binned by ascending break values (STYLE_BINS - 1
    /// breaks; bin = number of breaks ≤ value).
    Graduated { method: ClassMethod, breaks: Vec<f64> },
    /// Explicit category values, one bin each (bin STYLE_BINS-1 = other).
    Categorical { values: Vec<String> },
}

/// Classification method for graduated styling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassMethod {
    /// Equal-width intervals over [min, max] (from column statistics —
    /// no data read needed).
    EqualInterval,
    /// Equal feature counts per class (from loaded rows).
    Quantile,
    /// Classes of half a standard deviation around the mean (loaded rows).
    StdDev,
    /// Jenks natural breaks (Fisher's algorithm on a loaded-rows sample).
    Jenks,
}

impl ClassMethod {
    pub const ALL: &[ClassMethod] = &[
        ClassMethod::EqualInterval,
        ClassMethod::Quantile,
        ClassMethod::StdDev,
        ClassMethod::Jenks,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ClassMethod::EqualInterval => "Equal interval",
            ClassMethod::Quantile => "Quantiles",
            ClassMethod::StdDev => "Std deviation",
            ClassMethod::Jenks => "Jenks natural breaks",
        }
    }

    /// Does this method classify from data values (vs metadata only)?
    pub fn needs_values(&self) -> bool {
        !matches!(self, ClassMethod::EqualInterval)
    }
}

/// Equal-interval breaks over [min, max].
pub fn equal_interval_breaks(min: f64, max: f64) -> Vec<f64> {
    let span = if (max - min).abs() < f64::EPSILON { 1.0 } else { max - min };
    (1..STYLE_BINS)
        .map(|i| min + span * i as f64 / STYLE_BINS as f64)
        .collect()
}

/// Data-driven breaks from sampled values. `values` gets sorted.
pub fn classify_breaks(method: ClassMethod, values: &mut Vec<f64>) -> Vec<f64> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return equal_interval_breaks(0.0, 1.0);
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    match method {
        ClassMethod::EqualInterval => equal_interval_breaks(values[0], values[n - 1]),
        ClassMethod::Quantile => (1..STYLE_BINS)
            .map(|i| values[(i * n / STYLE_BINS).min(n - 1)])
            .collect(),
        ClassMethod::StdDev => {
            let mean = values.iter().sum::<f64>() / n as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let sd = var.sqrt().max(f64::EPSILON);
            // Half-sd classes centered on the mean: 15 breaks spanning
            // mean ± 3.75 σ.
            (1..STYLE_BINS)
                .map(|i| mean + sd * 0.5 * (i as f64 - STYLE_BINS as f64 / 2.0))
                .collect()
        }
        ClassMethod::Jenks => jenks_breaks(values),
    }
}

/// Fisher-Jenks natural breaks on a bounded sample (O(k·n²) exact DP —
/// the caller keeps n small).
fn jenks_breaks(sorted: &[f64]) -> Vec<f64> {
    const MAX_N: usize = 1500;
    let sample: Vec<f64> = if sorted.len() > MAX_N {
        (0..MAX_N)
            .map(|i| sorted[i * sorted.len() / MAX_N])
            .collect()
    } else {
        sorted.to_vec()
    };
    let n = sample.len();
    let k = STYLE_BINS.min(n.max(1));
    if n <= k {
        return equal_interval_breaks(sample[0], sample[n - 1]);
    }
    // Prefix sums for O(1) within-class variance.
    let mut ps = vec![0.0f64; n + 1];
    let mut ps2 = vec![0.0f64; n + 1];
    for (i, v) in sample.iter().enumerate() {
        ps[i + 1] = ps[i] + v;
        ps2[i + 1] = ps2[i] + v * v;
    }
    let ssd = |a: usize, b: usize| -> f64 {
        // Sum of squared deviations of sample[a..b].
        let m = (b - a) as f64;
        let s = ps[b] - ps[a];
        (ps2[b] - ps2[a]) - s * s / m
    };
    // dp[c][i]: best cost splitting sample[0..i] into c classes.
    let mut dp = vec![vec![f64::INFINITY; n + 1]; k + 1];
    let mut cut = vec![vec![0usize; n + 1]; k + 1];
    for i in 1..=n {
        dp[1][i] = ssd(0, i);
    }
    for c in 2..=k {
        for i in c..=n {
            for j in (c - 1)..i {
                let cost = dp[c - 1][j] + ssd(j, i);
                if cost < dp[c][i] {
                    dp[c][i] = cost;
                    cut[c][i] = j;
                }
            }
        }
    }
    // Backtrack the class boundaries → break values.
    let mut bounds = Vec::with_capacity(k - 1);
    let (mut c, mut i) = (k, n);
    while c > 1 {
        let j = cut[c][i];
        bounds.push(sample[j]);
        i = j;
        c -= 1;
    }
    bounds.reverse();
    // Pad if degenerate (shouldn't happen with n > k).
    while bounds.len() < STYLE_BINS - 1 {
        bounds.push(*bounds.last().unwrap_or(&0.0));
    }
    bounds
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ramp {
    Viridis,
    Turbo,
    /// Blue → white → red diverging.
    BuRd,
}

impl Ramp {
    pub const ALL: &[Ramp] = &[Ramp::Viridis, Ramp::Turbo, Ramp::BuRd];

    pub fn label(&self) -> &'static str {
        match self {
            Ramp::Viridis => "Viridis",
            Ramp::Turbo => "Turbo",
            Ramp::BuRd => "Blue–Red",
        }
    }

    /// Color at t in [0, 1].
    // A viridis stop happens to sit near 1/pi; these are colors, not math.
    #[allow(clippy::approx_constant)]
    pub fn sample(&self, t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        match self {
            // Polynomial fits (Nathan H. Matplotlib-style approximations,
            // good to a few percent — plenty for 16 bins).
            Ramp::Viridis => {
                let stops: [[f32; 3]; 6] = [
                    [0.267, 0.005, 0.329],
                    [0.254, 0.265, 0.530],
                    [0.164, 0.471, 0.558],
                    [0.128, 0.567, 0.551],
                    [0.478, 0.821, 0.318],
                    [0.993, 0.906, 0.144],
                ];
                lerp_stops(&stops, t)
            }
            Ramp::Turbo => {
                let stops: [[f32; 3]; 7] = [
                    [0.190, 0.072, 0.232],
                    [0.276, 0.408, 0.977],
                    [0.100, 0.746, 0.698],
                    [0.635, 0.923, 0.235],
                    [0.973, 0.729, 0.222],
                    [0.937, 0.334, 0.075],
                    [0.480, 0.016, 0.011],
                ];
                lerp_stops(&stops, t)
            }
            Ramp::BuRd => {
                let stops: [[f32; 3]; 3] = [
                    [0.129, 0.400, 0.674],
                    [0.969, 0.966, 0.965],
                    [0.792, 0.086, 0.113],
                ];
                lerp_stops(&stops, t)
            }
        }
    }
}

fn lerp_stops<const N: usize>(stops: &[[f32; 3]; N], t: f32) -> [f32; 3] {
    let x = t * (N - 1) as f32;
    let i = (x.floor() as usize).min(N - 2);
    let f = x - i as f32;
    let (a, b) = (stops[i], stops[i + 1]);
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

impl StyleBy {
    /// Per-bin RGB lookup for the renderer.
    pub fn bin_colors(&self) -> [[f32; 3]; STYLE_BINS] {
        let mut out = [[0.0; 3]; STYLE_BINS];
        match &self.mode {
            StyleMode::Graduated { .. } => {
                for (i, c) in out.iter_mut().enumerate() {
                    *c = self.ramp.sample(i as f32 / (STYLE_BINS - 1) as f32);
                }
            }
            StyleMode::Categorical { values } => {
                for (i, c) in out.iter_mut().enumerate() {
                    if i < values.len().min(STYLE_BINS - 1) {
                        let p = palette_color(i);
                        *c = [
                            p.r() as f32 / 255.0,
                            p.g() as f32 / 255.0,
                            p.b() as f32 / 255.0,
                        ];
                    } else {
                        *c = [0.55, 0.55, 0.55]; // "other"
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct LayerStyle {
    pub visible: bool,
    /// Draw the row-group bounding boxes overlay.
    pub show_rg_bboxes: bool,
    pub color: Color32,
    /// Border / line color; None = derived from `color` (darkened).
    pub line_color: Option<Color32>,
    pub line_width_px: f32,
    pub point_radius_px: f32,
    pub fill_opacity: f32,
    pub opacity: f32,
    /// Data-driven styling; None = single color. Changing column/breaks
    /// rebuilds the layer meshes (features are binned at build time).
    pub style_by: Option<StyleBy>,
}

impl LayerStyle {
    pub fn new(color: Color32) -> Self {
        Self {
            visible: true,
            show_rg_bboxes: false,
            color,
            line_color: None,
            line_width_px: 1.2,
            point_radius_px: 3.0,
            fill_opacity: 0.35,
            opacity: 1.0,
            style_by: None,
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

/// Thematic default color guessed from a layer name ("rivers_france" →
/// water blue, "building" → warm gray); None falls back to the rotating
/// palette. Keywords match name tokens by prefix (so plurals work and
/// "research" does not read as "sea"); keys containing '_' match the
/// whole lowercased name, letting `land_cover`/`land_use` win over the
/// bare `land` entry below them. First hit decides.
pub fn name_color(name: &str) -> Option<Color32> {
    type Keys = &'static [&'static str];
    const TABLE: &[(Keys, (u8, u8, u8))] = &[
        (&["bathymetry"], (48, 90, 148)),
        (
            &[
                "water", "river", "lake", "stream", "ocean", "sea", "hydro", "wetland",
                "reservoir", "canal", "flood", "precip", "rain", "coast",
            ],
            (66, 120, 179),
        ),
        (&["burn", "fire", "wildfire"], (196, 89, 48)),
        // Before the vegetation group: "natural" is a prefix of both OSM
        // natural themes, and only the areas are green.
        (&["natural_feature"], (146, 116, 91)),
        (
            &[
                "forest", "wood", "tree", "vegetation", "grass", "meadow", "park", "green",
                "natural_area", "natural",
            ],
            (96, 138, 74),
        ),
        (&["land_cover", "landcover"], (140, 155, 90)),
        (&["land_use", "landuse"], (165, 152, 84)),
        (&["land"], (188, 178, 140)),
        (&["building"], (146, 130, 120)),
        (&["parcel", "cadastr", "lot"], (189, 157, 105)),
        (&["rail", "train", "tram", "metro"], (94, 74, 96)),
        (&["public_transport", "transit"], (58, 112, 132)),
        (
            &["road", "street", "highway", "motorway", "segment", "transportation"],
            (86, 88, 94),
        ),
        (&["aeroway", "airport", "runway", "aerodrome", "heliport"], (154, 136, 170)),
        (&["connector"], (128, 128, 128)),
        (&["boundar", "division", "border", "admin"], (141, 94, 176)),
        (&["place", "poi", "pois", "amenit", "shop"], (226, 138, 48)),
        (&["address"], (186, 96, 125)),
        (&["power", "pipeline", "utilit", "energy"], (168, 144, 62)),
        (&["barrier", "fence", "wall"], (110, 96, 88)),
        (&["infrastructure"], (108, 122, 148)),
        (&["snow", "ice", "glacier"], (168, 200, 222)),
        (&["sand", "beach", "desert", "bare"], (214, 189, 138)),
    ];
    let lower = name.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    for (keys, (r, g, b)) in TABLE {
        let hit = keys.iter().any(|k| {
            if k.contains('_') {
                lower.contains(k)
            } else {
                // Short keys match whole tokens only: "sea" must not
                // claim "seattle", while "building" still covers
                // "buildings".
                tokens
                    .iter()
                    .any(|t| if k.len() < 4 { t == k } else { t.starts_with(k) })
            }
        });
        if hit {
            return Some(Color32::from_rgb(*r, *g, *b));
        }
    }
    None
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
    /// Lazy row access to the source file/URL (attributes, WKB, rebuilds).
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
    /// Decode state per row group (len = row groups in the file).
    pub loaded: Vec<GroupLoad>,
    /// Active layer filter: the persistent working subset. While set,
    /// `loaded` holds exactly the matching row ranges (with an infinite
    /// coverage rect, so viewport refinement never adds rows back).
    pub filter: Option<LayerFilter>,
    pub mode: LayerMode,
}

/// How this layer loads and displays (docs/OPEN_POLICY.md). Decided once
/// at open; never changes for the life of the layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LayerMode {
    /// Viewport machinery: row-group pruning, per-feature selection,
    /// preview fallback, refinement on camera settle.
    #[default]
    Indexed,
    /// Everything decoded up front (user's choice for a non-indexable
    /// file): no previews, no refinement, no viewport reloads.
    Direct,
}

/// A persistent SQL predicate restricting the layer to a subset of rows.
#[derive(Clone, Debug)]
pub struct LayerFilter {
    pub sql: String,
    pub matched: usize,
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
        self.loaded.len()
    }

    pub fn full_rgs(&self) -> usize {
        self.loaded.iter().filter(|g| g.is_full()).count()
    }

    /// Row groups with per-feature (viewport rect) selection only.
    pub fn partial_rgs(&self) -> usize {
        self.loaded
            .iter()
            .filter(|g| matches!(g, GroupLoad::Rows { .. }))
            .count()
    }

    /// Row groups holding a decimated preview.
    pub fn preview_rgs(&self) -> usize {
        self.loaded
            .iter()
            .filter(|g| matches!(g, GroupLoad::Preview { .. }))
            .count()
    }

    pub fn is_partial(&self) -> bool {
        self.loaded.iter().any(|g| !g.is_full())
    }

    /// Rows currently decoded into the map (drives the "classes may be
    /// stale" hint for data-classified styling).
    pub fn loaded_rows(&self) -> u64 {
        let starts = self.store.rg_starts();
        self.loaded
            .iter()
            .enumerate()
            .map(|(g, st)| match st {
                GroupLoad::Full => starts[g + 1] - starts[g],
                GroupLoad::Rows { ranges, .. } => {
                    ranges.iter().map(|&(s, e)| (e - s) as u64).sum()
                }
                // Rect-filtered previews load fewer rows; this upper bound
                // only drives the staleness hint.
                GroupLoad::Preview { stride, .. } => {
                    (starts[g + 1] - starts[g]).div_ceil(*stride as u64)
                }
                GroupLoad::None => 0,
            })
            .sum()
    }
}

#[cfg(test)]
mod name_color_tests {
    use super::*;

    #[test]
    fn thematic_names_get_thematic_colors() {
        let water = name_color("water").unwrap();
        // Prefix and token handling: plurals and composites match…
        assert_eq!(name_color("rivers_france"), Some(water));
        assert_eq!(name_color("EU_Hydro_Network"), Some(water));
        assert_eq!(name_color("north-sea-wrecks"), Some(water));
        // …but "sea" must not claim Seattle, nor "research".
        let parcels = name_color("parcels").unwrap();
        assert_eq!(name_color("seattle_parcels"), Some(parcels));
        assert_ne!(name_color("research_sites"), Some(water));

        // Overture types land on distinct entries.
        let land = name_color("land").unwrap();
        assert_ne!(name_color("land_cover"), Some(land));
        assert_ne!(name_color("land_use"), Some(land));
        assert_ne!(name_color("land_use"), name_color("land_cover"));
        assert_eq!(name_color("buildings"), name_color("building"));
        // Woodland reads as forest, wetland as water.
        assert_eq!(name_color("woodland"), name_color("forest"));
        assert_eq!(name_color("wetlands"), Some(water));

        // No keyword: fall through to the rotating palette.
        assert_eq!(name_color("mystery_dataset_42"), None);
    }

    /// Every theme of the geomermaids parquetry repositories resolves to
    /// a thematic color, with the ambiguous pairs kept distinct.
    #[test]
    fn geomermaids_themes_are_all_covered() {
        let themes = [
            "buildings", "roads", "railways", "waterways", "water", "landuse",
            "natural_areas", "natural_features", "places", "boundaries", "pois",
            "amenities_polygons", "power", "aeroways", "barriers", "public_transport",
        ];
        for t in themes {
            assert!(name_color(t).is_some(), "{t} has no thematic color");
        }
        // The pairs that must not collapse into one color.
        assert_ne!(name_color("natural_areas"), name_color("natural_features"));
        assert_eq!(name_color("natural_areas"), name_color("forest"));
        assert_ne!(name_color("power"), name_color("infrastructure"));
        assert_ne!(name_color("railways"), name_color("public_transport"));
        assert_ne!(name_color("roads"), name_color("railways"));
        // Both amenity spellings land with places/POIs.
        assert_eq!(name_color("amenities_polygons"), name_color("pois"));
        assert_eq!(name_color("amenity"), name_color("places"));
        // Waterways and water share the water blue.
        assert_eq!(name_color("waterways"), name_color("water"));
    }
}

#[cfg(test)]
mod class_tests {
    use super::*;

    #[test]
    fn classification_breaks_sane() {
        // Skewed data: 900 small values + 100 large ones.
        let mut vals: Vec<f64> = (0..900).map(|i| i as f64 / 900.0).collect();
        vals.extend((0..100).map(|i| 100.0 + i as f64));

        let eq = classify_breaks(ClassMethod::EqualInterval, &mut vals.clone());
        assert_eq!(eq.len(), STYLE_BINS - 1);
        assert!(eq.windows(2).all(|w| w[0] <= w[1]), "ascending");
        // Equal interval spreads over the full range: most breaks above 1.
        assert!(eq.iter().filter(|b| **b > 1.0).count() > 10);

        let q = classify_breaks(ClassMethod::Quantile, &mut vals.clone());
        assert_eq!(q.len(), STYLE_BINS - 1);
        assert!(q.windows(2).all(|w| w[0] <= w[1]));
        // Quantiles follow the mass: most breaks inside the dense [0,1).
        assert!(q.iter().filter(|b| **b < 1.0).count() >= 12, "{q:?}");

        let j = classify_breaks(ClassMethod::Jenks, &mut vals.clone());
        assert_eq!(j.len(), STYLE_BINS - 1);
        assert!(j.windows(2).all(|w| w[0] <= w[1]));
        // Natural breaks must isolate the [100, 200) cluster from [0, 1).
        assert!(j.iter().any(|b| *b > 1.0 && *b <= 101.0), "{j:?}");

        let sd = classify_breaks(ClassMethod::StdDev, &mut vals.clone());
        assert_eq!(sd.len(), STYLE_BINS - 1);
        assert!(sd.windows(2).all(|w| w[0] <= w[1]));

        // Degenerate inputs don't panic.
        assert_eq!(
            classify_breaks(ClassMethod::Jenks, &mut vec![5.0; 10]).len(),
            STYLE_BINS - 1
        );
        assert_eq!(
            classify_breaks(ClassMethod::Quantile, &mut Vec::new()).len(),
            STYLE_BINS - 1
        );
    }

    #[test]
    fn breaks_binning_matches_partition() {
        // partition_point semantics: value below first break -> bin 0,
        // above last -> last bin.
        let breaks = equal_interval_breaks(0.0, 16.0);
        let bin = |v: f64| breaks.partition_point(|b| v >= *b);
        assert_eq!(bin(-1.0), 0);
        assert_eq!(bin(0.5), 0);
        assert_eq!(bin(1.0), 1);
        assert_eq!(bin(15.5), 15);
        assert_eq!(bin(99.0), 15);
    }
}
