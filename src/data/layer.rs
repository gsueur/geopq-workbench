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
    /// Average number of *other* row-group boxes each box intersects,
    /// over the run C2 judges the layer on. Raw counts aren't comparable
    /// across run sizes; judge clustering with [`Self::overlap_frac`].
    pub avg_overlap: f64,
    /// Boxes that average was measured over: every box normally, and on
    /// a COGP layer only the worst level's, because levels overlap each
    /// other by construction.
    measured_over: usize,
    /// C2's verdict on that run, so the panel and the scorecard cannot
    /// disagree about the same file.
    well_clustered: bool,
}

impl RgBboxes {
    /// Measure the boxes as they are built.
    ///
    /// `cogp_levels` makes the measurement per COGP level instead of per
    /// file — the same run C2 grades, so the panel and the scorecard
    /// never disagree about whether a layer is well clustered.
    pub fn new(
        source: String,
        boxes: Vec<[f64; 4]>,
        cogp_levels: Option<&[crate::data::cogp::LevelRun]>,
    ) -> Self {
        let (_, m) = crate::data::quality::Clustering::worst(&boxes, cogp_levels);
        Self {
            source,
            boxes,
            avg_overlap: m.avg,
            measured_over: m.n,
            well_clustered: m.passes(),
        }
    }

    /// Overlap as a fraction of the possible overlaps (0 = disjoint boxes,
    /// 1 = every box intersects every other). Comparable across row-group
    /// counts, unlike the raw average. Reference points: Hilbert-sorted
    /// 65k-row groups land at 13–25% (adjacent groups necessarily touch),
    /// attribute-ordered data ~35%, spatially random data ~100%.
    pub fn overlap_frac(&self) -> f64 {
        self.avg_overlap / (self.measured_over.max(2) - 1) as f64
    }

    /// Would a spatial-order rewrite (or finer row groups) improve
    /// pruning? This is C2's own verdict over the same run, not a second
    /// threshold: a file the scorecard passes must not be labelled
    /// poorly clustered a panel away.
    pub fn poorly_clustered(&self) -> bool {
        !self.well_clustered
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
    /// `ranges`: the in-rect rows the load's covering scan resolved,
    /// before decimation — kept so a rebuild replays the selection
    /// instead of paying for that scan again. None when the load was not
    /// rect-filtered, or when the state predates the scan.
    Preview {
        stride: u32,
        rect: Option<[f64; 4]>,
        ranges: Option<Vec<(u32, u32)>>,
    },
    /// Every feature of the group is on the map, drawn from its covering
    /// bounding box instead of its geometry (restricted to the features
    /// intersecting `rect` when the load was viewport-filtered).
    ///
    /// The alternative for a polygon coverage over budget is a stride
    /// preview, which removes most of the polygons: on data that tiles
    /// the plane, that reads as holes rather than as detail. Boxes keep
    /// the coverage complete at a resolution the screen can show, for
    /// four doubles a feature and no geometry decode at all. Like a
    /// preview it never covers a viewport, so zooming in refines it.
    /// `ranges` as in [`GroupLoad::Preview`].
    Boxes {
        rect: Option<[f64; 4]>,
        ranges: Option<Vec<(u32, u32)>>,
    },
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
            GroupLoad::None | GroupLoad::Preview { .. } | GroupLoad::Boxes { .. } => false,
            GroupLoad::Full => true,
            GroupLoad::Rows { rect: r, .. } => {
                r[0] <= rect[0] && r[1] <= rect[1] && r[2] >= rect[2] && r[3] >= rect[3]
            }
        }
    }
}

/// Number of style bins (chunk meshes carry a u8 bin; 0 is the default
/// bin used when no data-driven styling is active).
pub const STYLE_BINS: usize = 64;

/// Data-driven styling: color features by an attribute column.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleBy {
    pub column: String,
    pub ramp: Ramp,
    pub mode: StyleMode,
    /// Bitmask of classes hidden from the map (bit i = bin i), toggled
    /// by clicking legend entries. Draw-time filter only; meshes keep
    /// every feature.
    pub hidden_bins: u64,
    /// Graduated styling: classify and render value / polygon area
    /// (data-CRS units) instead of the absolute value, so large
    /// polygons don't dominate a choropleth.
    pub per_area: bool,
    /// Rows that were loaded when data-dependent classes were computed
    /// (quantiles/std-dev/Jenks classify only what is loaded — panning or
    /// zooming afterwards can make the classes stale).
    pub classified_rows: Option<usize>,
    /// Line width ramp (min, max) in px: class widths interpolate
    /// linearly from first to last bin, the way colors sample the ramp.
    /// None = the layer-wide width slider applies to every class.
    pub width_px: Option<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StyleMode {
    /// Numeric values binned by ascending break values (STYLE_BINS - 1
    /// breaks; bin = number of breaks ≤ value).
    Graduated { method: ClassMethod, breaks: Vec<f64> },
    /// Explicit category values, one bin each (bin STYLE_BINS-1 = other).
    ///
    /// `colors` and `labels`, when set, are aligned with `values` and
    /// come from a colour map — a dataset whose classes have an official
    /// palette (CORINE land cover, soil types, a QGIS export shipped
    /// beside the data) must be drawn in it, not in whatever the
    /// frequency order happened to assign.
    Categorical {
        values: Vec<String>,
        #[allow(clippy::type_complexity)]
        colors: Option<Vec<[u8; 3]>>,
        labels: Option<Vec<String>>,
    },
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
    /// Class widths grow linearly (1w, 2w, 3w…): breaks concentrate
    /// near the minimum. Metadata-only, like equal interval.
    Arithmetic,
    /// Class bounds follow min·q^i (equal intervals on a log scale);
    /// the base is the smallest positive value, class 1 absorbs ≤ 0.
    Geometric,
    /// Head/tail breaks (Jiang 2013): split at the mean, recurse into
    /// the head while it stays a minority. Made for heavy-tailed data;
    /// may produce fewer classes than requested.
    HeadTail,
}

impl ClassMethod {
    pub const ALL: &[ClassMethod] = &[
        ClassMethod::EqualInterval,
        ClassMethod::Quantile,
        ClassMethod::StdDev,
        ClassMethod::Jenks,
        ClassMethod::Arithmetic,
        ClassMethod::Geometric,
        ClassMethod::HeadTail,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ClassMethod::EqualInterval => "Equal interval",
            ClassMethod::Quantile => "Quantiles",
            ClassMethod::StdDev => "Std deviation",
            ClassMethod::Jenks => "Jenks natural breaks",
            ClassMethod::Arithmetic => "Arithmetic progression",
            ClassMethod::Geometric => "Geometric progression",
            ClassMethod::HeadTail => "Head/tail breaks",
        }
    }

    /// Does this method classify from data values (vs metadata only)?
    pub fn needs_values(&self) -> bool {
        !matches!(self, ClassMethod::EqualInterval | ClassMethod::Arithmetic)
    }
}

/// Breaks computable from bounds alone, for the editable min/max path
/// (methods where `needs_values()` is false).
pub fn bounds_breaks(method: ClassMethod, min: f64, max: f64, classes: usize) -> Vec<f64> {
    match method {
        ClassMethod::Arithmetic => arithmetic_breaks(min, max, classes),
        _ => equal_interval_breaks(min, max, classes),
    }
}

/// Arithmetic-progression breaks: the i-th class is i units wide, the
/// unit chosen so `classes` classes exactly cover [min, max].
pub fn arithmetic_breaks(min: f64, max: f64, classes: usize) -> Vec<f64> {
    let classes = classes.clamp(2, STYLE_BINS);
    let span = if (max - min).abs() < f64::EPSILON { 1.0 } else { max - min };
    let unit = span / (classes * (classes + 1) / 2) as f64;
    let mut acc = min;
    (1..classes)
        .map(|i| {
            acc += unit * i as f64;
            acc
        })
        .collect()
}

/// Equal-interval breaks over [min, max], `classes - 1` of them.
pub fn equal_interval_breaks(min: f64, max: f64, classes: usize) -> Vec<f64> {
    let classes = classes.clamp(2, STYLE_BINS);
    let span = if (max - min).abs() < f64::EPSILON { 1.0 } else { max - min };
    (1..classes)
        .map(|i| min + span * i as f64 / classes as f64)
        .collect()
}

/// Data-driven breaks from sampled values (`classes - 1` of them, so the
/// class count is always recoverable as `breaks.len() + 1`). `values`
/// gets sorted.
pub fn classify_breaks(method: ClassMethod, values: &mut Vec<f64>, classes: usize) -> Vec<f64> {
    let classes = classes.clamp(2, STYLE_BINS);
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return equal_interval_breaks(0.0, 1.0, classes);
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    match method {
        ClassMethod::EqualInterval => equal_interval_breaks(values[0], values[n - 1], classes),
        ClassMethod::Quantile => (1..classes)
            .map(|i| values[(i * n / classes).min(n - 1)])
            .collect(),
        ClassMethod::StdDev => {
            let mean = values.iter().sum::<f64>() / n as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let sd = var.sqrt().max(f64::EPSILON);
            // Half-sd classes centered on the mean (16 classes span
            // mean ± 3.75 σ).
            (1..classes)
                .map(|i| mean + sd * 0.5 * (i as f64 - classes as f64 / 2.0))
                .collect()
        }
        ClassMethod::Jenks => jenks_breaks(values, classes),
        ClassMethod::Arithmetic => arithmetic_breaks(values[0], values[n - 1], classes),
        ClassMethod::Geometric => {
            // Base = smallest positive value: zeros/negatives all land in
            // class 1 and the ratio spans the positive range.
            let max = values[n - 1];
            let base = values.iter().copied().find(|v| *v > 0.0).unwrap_or(0.0);
            if base <= 0.0 || max / base <= 1.0 + 1e-12 {
                return equal_interval_breaks(values[0], max, classes);
            }
            let q = (max / base).powf(1.0 / classes as f64);
            (1..classes).map(|i| base * q.powi(i as i32)).collect()
        }
        ClassMethod::HeadTail => {
            let mut out: Vec<f64> = Vec::new();
            let mut slice: &[f64] = values;
            while out.len() < classes - 1 && slice.len() > 1 {
                let mean = slice.iter().sum::<f64>() / slice.len() as f64;
                let head_start = slice.partition_point(|v| *v < mean);
                let head = &slice[head_start..];
                if head.is_empty() || head.len() == slice.len() {
                    break;
                }
                out.push(mean);
                // Recurse only while the head stays a minority (< 40%).
                if head.len() as f64 / slice.len() as f64 >= 0.4 {
                    break;
                }
                slice = head;
            }
            if out.is_empty() {
                return equal_interval_breaks(values[0], values[n - 1], classes);
            }
            out
        }
    }
}

/// Fisher-Jenks natural breaks on a bounded sample (O(k·n²) exact DP —
/// the caller keeps n small).
fn jenks_breaks(sorted: &[f64], classes: usize) -> Vec<f64> {
    const MAX_N: usize = 1500;
    let sample: Vec<f64> = if sorted.len() > MAX_N {
        (0..MAX_N)
            .map(|i| sorted[i * sorted.len() / MAX_N])
            .collect()
    } else {
        sorted.to_vec()
    };
    let n = sample.len();
    let k = classes.min(n.max(1));
    if n <= k {
        return equal_interval_breaks(sample[0], sample[n - 1], classes);
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
    while bounds.len() < classes - 1 {
        bounds.push(*bounds.last().unwrap_or(&0.0));
    }
    bounds
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ramp {
    Viridis,
    Plasma,
    /// Perceptually uniform blue → gray → yellow, color-blind safe.
    Cividis,
    Turbo,
    /// Blue → white → red diverging.
    BuRd,
    /// Brown → white → teal diverging (ColorBrewer BrBG): dry/wet,
    /// loss/gain — diverging data where red/blue reads as politics.
    BrBg,
}

impl Ramp {
    pub const ALL: &[Ramp] = &[
        Ramp::Viridis,
        Ramp::Plasma,
        Ramp::Cividis,
        Ramp::Turbo,
        Ramp::BuRd,
        Ramp::BrBg,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Ramp::Viridis => "Viridis",
            Ramp::Plasma => "Plasma",
            Ramp::Cividis => "Cividis",
            Ramp::Turbo => "Turbo",
            Ramp::BuRd => "Blue–Red",
            Ramp::BrBg => "Brown–Teal",
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
            Ramp::Plasma => {
                let stops: [[f32; 3]; 6] = [
                    [0.051, 0.031, 0.529],
                    [0.416, 0.000, 0.659],
                    [0.694, 0.165, 0.565],
                    [0.882, 0.392, 0.384],
                    [0.988, 0.651, 0.212],
                    [0.941, 0.976, 0.129],
                ];
                lerp_stops(&stops, t)
            }
            Ramp::Cividis => {
                let stops: [[f32; 3]; 6] = [
                    [0.000, 0.133, 0.306],
                    [0.208, 0.271, 0.424],
                    [0.400, 0.408, 0.439],
                    [0.580, 0.557, 0.467],
                    [0.784, 0.722, 0.400],
                    [0.992, 0.918, 0.271],
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
            Ramp::BrBg => {
                let stops: [[f32; 3]; 5] = [
                    [0.549, 0.318, 0.039],
                    [0.847, 0.702, 0.396],
                    [0.961, 0.961, 0.961],
                    [0.353, 0.706, 0.675],
                    [0.004, 0.400, 0.369],
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
            StyleMode::Graduated { breaks, .. } => {
                // Class count derives from the breaks; bins past it never
                // occur (bin = number of breaks ≤ value) but get the last
                // color anyway.
                let n = (breaks.len() + 1).clamp(2, STYLE_BINS);
                for (i, c) in out.iter_mut().enumerate() {
                    *c = self.ramp.sample(i.min(n - 1) as f32 / (n - 1) as f32);
                }
            }
            StyleMode::Categorical { values, colors, .. } => {
                for (i, c) in out.iter_mut().enumerate() {
                    if i < values.len().min(STYLE_BINS - 1) {
                        let rgb = match colors {
                            Some(m) if i < m.len() => m[i],
                            _ => {
                                let p = palette_color(i);
                                [p.r(), p.g(), p.b()]
                            }
                        };
                        *c = [
                            rgb[0] as f32 / 255.0,
                            rgb[1] as f32 / 255.0,
                            rgb[2] as f32 / 255.0,
                        ];
                    } else {
                        *c = [0.55, 0.55, 0.55]; // "other"
                    }
                }
            }
        }
        out
    }

    /// Full line widths per bin in px, when a width ramp is set: linear
    /// from min to max across the classes, mirroring how `bin_colors`
    /// samples the color ramp. Bins past the class count keep the max.
    pub fn bin_widths(&self) -> Option<[f32; STYLE_BINS]> {
        let (min, max) = self.width_px?;
        let n = match &self.mode {
            StyleMode::Graduated { breaks, .. } => (breaks.len() + 1).clamp(2, STYLE_BINS),
            StyleMode::Categorical { values, .. } => values.len().clamp(2, STYLE_BINS),
        };
        let mut out = [max; STYLE_BINS];
        for (i, w) in out.iter_mut().enumerate().take(n) {
            *w = min + (max - min) * i as f32 / (n - 1) as f32;
        }
        Some(out)
    }
}

/// Dash pattern of a line layer's stroke (or a polygon's outline).
///
/// Lengths are in multiples of the full line width, converted to px at
/// draw time, so thicker lines get proportionally longer dashes. A dash
/// of length 0 is a dot: the cap alone gives it its shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LinePattern {
    #[default]
    Solid,
    Dash,
    LongDash,
    Dot,
    DashDot,
}

impl LinePattern {
    pub const ALL: [LinePattern; 5] = [
        LinePattern::Solid,
        LinePattern::Dash,
        LinePattern::LongDash,
        LinePattern::Dot,
        LinePattern::DashDot,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LinePattern::Solid => "solid",
            LinePattern::Dash => "dash",
            LinePattern::LongDash => "long dash",
            LinePattern::Dot => "dot",
            LinePattern::DashDot => "dash dot",
        }
    }

    pub fn from_label(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.label() == name)
    }

    /// (dash, gap, dash, gap) in width multiples; dash < 0 = solid.
    fn spec(self) -> [f32; 4] {
        match self {
            LinePattern::Solid => [-1.0, 0.0, 0.0, 0.0],
            LinePattern::Dash => [4.0, 2.0, 0.0, 0.0],
            LinePattern::LongDash => [8.0, 3.0, 0.0, 0.0],
            LinePattern::Dot => [0.0, 3.0, 0.0, 0.0],
            LinePattern::DashDot => [4.0, 2.0, 0.0, 2.0],
        }
    }

    /// The pattern in px for a given full line width, shaped by the cap:
    /// a flat cap adds nothing past a dash's end, which would erase
    /// zero-length dots entirely, so dots grow to one width under it.
    /// The unit is floored at 3 px: dashes scale with the stroke, but on
    /// a hairline an unfloored gap sinks into the AA feather (round caps
    /// eat half a width from each end on top) and the pattern reads as
    /// solid.
    pub fn dashes_px(self, cap: LineCap, width_px: f32) -> [f32; 4] {
        let mut d = self.spec();
        if d[0] < 0.0 {
            return d;
        }
        let w = width_px.max(3.0);
        if cap == LineCap::Flat {
            if d[0] == 0.0 {
                d[0] = 1.0;
            }
            if d[2] == 0.0 && d[3] > 0.0 {
                d[2] = 1.0;
            }
        }
        [d[0] * w, d[1] * w, d[2] * w, d[3] * w]
    }
}

/// Line end treatment, applied at run ends and both ends of every dash.
/// Interior joints of a polyline always stay round: any other join
/// would crack at angles with the segment-instanced line pass. The
/// codes are duplicated in `shaders.wgsl` and must stay in step.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LineCap {
    #[default]
    Round,
    Square,
    Flat,
}

impl LineCap {
    pub const ALL: [LineCap; 3] = [LineCap::Round, LineCap::Square, LineCap::Flat];

    /// Shader id.
    pub fn code(self) -> u32 {
        match self {
            LineCap::Round => 0,
            LineCap::Square => 1,
            LineCap::Flat => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LineCap::Round => "round",
            LineCap::Square => "square",
            LineCap::Flat => "flat",
        }
    }

    pub fn from_label(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.label() == name)
    }
}

/// Marker drawn for each feature of a point layer.
///
/// Every shape is sized to the area of the circle it replaces, so
/// switching symbol keeps the ink weight of the layer and one radius
/// slider still means the same thing. `reach()` is how far the shape
/// extends past that radius (a square's corner, a star's tip); the
/// shader needs it to size the marker quad. Both tables are duplicated
/// in `shaders.wgsl` and must stay in step with `code()`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PointShape {
    #[default]
    Circle,
    Square,
    Triangle,
    Diamond,
    Hexagon,
    Star,
}

impl PointShape {
    pub const ALL: [PointShape; 6] = [
        PointShape::Circle,
        PointShape::Square,
        PointShape::Triangle,
        PointShape::Diamond,
        PointShape::Hexagon,
        PointShape::Star,
    ];

    /// Shader id, and the token used in a saved context.
    pub fn code(self) -> u32 {
        match self {
            PointShape::Circle => 0,
            PointShape::Square => 1,
            PointShape::Triangle => 2,
            PointShape::Diamond => 3,
            PointShape::Hexagon => 4,
            PointShape::Star => 5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PointShape::Circle => "circle",
            PointShape::Square => "square",
            PointShape::Triangle => "triangle",
            PointShape::Diamond => "diamond",
            PointShape::Hexagon => "hexagon",
            PointShape::Star => "star",
        }
    }

    pub fn from_label(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.label() == name)
    }

    /// Circumradius of the marker: its farthest point from the centre,
    /// as a multiple of the layer's point radius.
    pub fn reach(self) -> f32 {
        match self {
            PointShape::Circle => 1.0,
            // corner of an equal-area square, half-side 0.8862 r
            PointShape::Square | PointShape::Diamond => 1.2534,
            PointShape::Triangle => 1.5535,
            PointShape::Hexagon => 1.1000,
            PointShape::Star => 1.4620,
        }
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
    pub line_pattern: LinePattern,
    pub line_cap: LineCap,
    pub point_radius_px: f32,
    pub point_shape: PointShape,
    pub fill_opacity: f32,
    pub opacity: f32,
    /// Master switches for polygon rendition (the panel's clickable
    /// "fill:" / "w:" labels): borders-only or fill-only display.
    pub fill_on: bool,
    pub lines_on: bool,
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
            line_pattern: LinePattern::Solid,
            line_cap: LineCap::Round,
            point_radius_px: 3.0,
            point_shape: PointShape::Circle,
            fill_opacity: 0.35,
            fill_on: true,
            lines_on: true,
            opacity: 1.0,
            style_by: None,
        }
    }

    /// Render as a published palette does: opaque fills, no outlines.
    ///
    /// A nomenclature like CORINE defines one thing per class, a fill
    /// colour. It has no outline, and its colours are exact RGB values
    /// that only come out right at full opacity — at the default 35% over
    /// a basemap, every class lands on a colour the palette never
    /// specified. Outlines are worse than wrong here: on a coverage that
    /// tiles the plane they draw a border between every neighbouring
    /// parcel, and the map turns into a dark mesh with colour showing
    /// through it.
    ///
    /// Applied when a colour map is adopted. Both switches stay under the
    /// layer row, so a user who wants outlines back just clicks them on.
    pub fn adopt_palette(&mut self) {
        self.lines_on = false;
        self.fill_opacity = 1.0;
    }
}

impl StyleMode {
    /// Does this styling come from a colour map (explicit per-class
    /// colours) rather than from a ramp or the generic palette?
    pub fn is_color_map(&self) -> bool {
        matches!(self, StyleMode::Categorical { colors: Some(_), .. })
    }
}

/// Tableau 20, darks first then their light variants: the first ten
/// layers/classes stay maximally distinct, and only busier categorical
/// legends dip into the pastels before cycling.
const PALETTE: [Color32; 20] = [
    Color32::from_rgb(31, 119, 180),
    Color32::from_rgb(255, 127, 14),
    Color32::from_rgb(44, 160, 44),
    Color32::from_rgb(214, 39, 40),
    Color32::from_rgb(148, 103, 189),
    Color32::from_rgb(140, 86, 75),
    Color32::from_rgb(227, 119, 194),
    Color32::from_rgb(127, 127, 127),
    Color32::from_rgb(188, 189, 34),
    Color32::from_rgb(23, 190, 207),
    Color32::from_rgb(174, 199, 232),
    Color32::from_rgb(255, 187, 120),
    Color32::from_rgb(152, 223, 138),
    Color32::from_rgb(255, 152, 150),
    Color32::from_rgb(197, 176, 213),
    Color32::from_rgb(196, 156, 148),
    Color32::from_rgb(247, 182, 210),
    Color32::from_rgb(199, 199, 199),
    Color32::from_rgb(219, 219, 141),
    Color32::from_rgb(158, 218, 229),
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
    /// This layer draws its unrefined groups from covering boxes. Set at
    /// open when the plan chose them, and used by rebuilds to keep the
    /// coverage complete: a group refined for one viewport must still
    /// show boxes everywhere else, or it leaves a row-group-shaped hole.
    pub box_layer: bool,
    /// Bumped whenever `sections` changes, and used as the GPU cache key.
    ///
    /// Distinct from `generation`: that advances when a rebuild is
    /// *requested*, so keying uploads by it drops the resident buffers
    /// the moment work starts, and the CPU meshes behind them have
    /// usually been freed after their first upload. The layer would draw
    /// nothing until the rebuild landed. Keying by what is actually in
    /// `sections` keeps the old mesh on screen until the new one
    /// replaces it.
    pub draw_gen: u64,
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

    /// Every count the layers panel shows about the load state, in one
    /// pass over `loaded`.
    ///
    /// The panel used to ask for them one at a time — full, partial,
    /// preview, boxes, rows — which is five scans of a vector with one
    /// entry per row group, per layer, on every frame the panel is up.
    pub fn load_summary(&self) -> LoadSummary {
        load_summary(&self.loaded, self.store.rg_starts())
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

    /// Row groups drawn from covering boxes rather than geometry.
    pub fn boxes_rgs(&self) -> usize {
        self.loaded
            .iter()
            .filter(|g| matches!(g, GroupLoad::Boxes { .. }))
            .count()
    }

    pub fn is_partial(&self) -> bool {
        self.loaded.iter().any(|g| !g.is_full())
    }

    /// Typical feature span in data-CRS units: the side of the square a
    /// feature would occupy if the layer's features tiled their row
    /// group's extent evenly.
    ///
    /// Derived from the row-group boxes and their row counts, so it costs
    /// nothing and needs no data read. It is what makes the box/geometry
    /// decision a property of the *scale* rather than of the area: a
    /// budget on bytes or counts would refine a viewport over farmland
    /// and refuse the same viewport over a city, at the same zoom, which
    /// no user can predict or interpret.
    pub fn feature_span(&self) -> f64 {
        let Some(rg) = &self.rg_bboxes else { return 0.0 };
        let starts = self.store.rg_starts();
        let (mut area, mut rows) = (0.0f64, 0u64);
        for (g, b) in rg.boxes.iter().enumerate() {
            if b[0] > b[2] || g + 1 >= starts.len() {
                continue; // sentinel box
            }
            let n = starts[g + 1] - starts[g];
            if n == 0 {
                continue;
            }
            area += (b[2] - b[0]) * (b[3] - b[1]);
            rows += n;
        }
        if rows == 0 || !area.is_finite() || area <= 0.0 {
            return 0.0;
        }
        (area / rows as f64).sqrt()
    }

    /// Rows currently decoded into the map (drives the "classes may be
    /// stale" hint for data-classified styling).
    pub fn loaded_rows(&self) -> u64 {
        self.load_summary().rows
    }
}

/// `VectorLayer::load_summary` over the parts it needs, so the counting
/// can be tested without a store behind it.
pub fn load_summary(loaded: &[GroupLoad], starts: &[u64]) -> LoadSummary {
    let mut s = LoadSummary { total: loaded.len(), ..Default::default() };
    for (g, st) in loaded.iter().enumerate() {
        // A store replaced under an in-flight job can leave `loaded`
        // longer than the row groups it describes for a frame; count
        // what is there and read no rows for the rest.
        let group_rows = match (starts.get(g), starts.get(g + 1)) {
            (Some(a), Some(b)) => b - a,
            _ => 0,
        };
        match st {
            GroupLoad::Full => {
                s.full += 1;
                s.rows += group_rows;
            }
            GroupLoad::Rows { ranges, .. } => {
                s.partial += 1;
                s.rows += ranges.iter().map(|&(a, b)| (b - a) as u64).sum::<u64>();
            }
            // Rect-filtered previews load fewer rows; this upper bound
            // only drives the staleness hint.
            GroupLoad::Preview { stride, .. } => {
                s.preview += 1;
                s.rows += group_rows.div_ceil(*stride as u64);
            }
            // Boxes carry every feature of the group, so the count is
            // the group's; only their shape is approximate.
            GroupLoad::Boxes { .. } => {
                s.boxes += 1;
                s.rows += group_rows;
            }
            GroupLoad::None => {}
        }
    }
    s
}

/// What one pass over a layer's `loaded` state says about it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadSummary {
    /// Row groups the layer has state for (`loaded.len()`).
    pub total: usize,
    pub full: usize,
    /// Row groups with per-feature (viewport rect) selection only.
    pub partial: usize,
    /// Row groups holding a decimated preview.
    pub preview: usize,
    /// Row groups drawn from covering boxes rather than geometry.
    pub boxes: usize,
    /// Rows currently decoded into the map.
    pub rows: u64,
}

#[cfg(test)]
mod name_color_tests {
    use super::*;

    /// The box/geometry decision must depend on the scale alone, so the
    /// same zoom behaves the same over a city and over farmland.
    #[test]
    fn the_representation_follows_the_scale_not_the_density() {
        // CORINE: a 25 ha minimum mapping unit, so a typical feature is
        // roughly 500 m across.
        let span = 500.0f64;
        const BOX_ENOUGH_PX: f64 = 3.0;
        let boxes_suffice = |metres_per_px: f64| span < BOX_ENOUGH_PX * metres_per_px;

        // Europe-wide: a pixel is kilometres, the feature is invisible.
        assert!(boxes_suffice(3000.0), "continent view draws boxes");
        // Regional: still under three pixels.
        assert!(boxes_suffice(200.0));
        // Street level (z≈14.7 is ~6 m/px): the feature is ~80 px, so
        // real geometry, whatever sits under the viewport.
        assert!(!boxes_suffice(6.0), "street zoom draws geometry");
        assert!(!boxes_suffice(60.0));
        // The switch is at span / BOX_ENOUGH_PX, and nothing about the
        // amount of data enters into it.
        let threshold = span / BOX_ENOUGH_PX;
        assert!(boxes_suffice(threshold + 1.0));
        assert!(!boxes_suffice(threshold - 1.0));
    }

    /// The GPU cache key must follow what is in `sections`, not what a
    /// rebuild intends to put there.
    ///
    /// Uploaded meshes are freed on the CPU side, so a key that changes
    /// when a rebuild *starts* leaves the layer with nothing to re-upload
    /// and the map goes blank until the rebuild lands. An append must not
    /// disturb the keys of the sections already up, either.
    #[test]
    fn the_draw_key_tracks_sections_not_pending_rebuilds() {
        let mut draw_gen = 0u64; // stands in for VectorLayer::draw_gen
        let key = |section: usize, g: u64| (section as u64 | ((section as u64 + 1) << 40), g);

        // Section 0 uploaded.
        let first = key(0, draw_gen);
        // A rebuild is requested: generation advances, draw_gen does not.
        let requested = key(0, draw_gen);
        assert_eq!(first, requested, "a pending rebuild keeps the mesh on screen");

        // An append adds section 1 and leaves section 0 alone.
        let appended = key(1, draw_gen);
        assert_ne!(appended, first, "the new section takes its own key");
        assert_eq!(key(0, draw_gen), first, "the resident section keeps its key");

        // The rebuild lands and replaces every section: now the key moves.
        draw_gen += 1;
        assert_ne!(key(0, draw_gen), first, "replaced content is a new upload");
    }

    /// A colour map is a fill palette: outlines off, fills opaque.
    #[test]
    fn a_colour_map_brings_its_own_rendition() {
        let map_mode = crate::data::colormap::categorical_mode(
            Vec::new(),
            crate::data::colormap::match_column("Code_18").as_ref(),
        );
        assert!(map_mode.is_color_map());
        let mut style = LayerStyle::new(Color32::GRAY);
        assert!(style.lines_on, "outlines are on by default");
        style.adopt_palette();
        assert!(!style.lines_on, "CORINE defines no outline");
        assert_eq!(style.fill_opacity, 1.0, "its RGB values are exact");

        // The generic categorical palette is not a colour map: a layer
        // styled by frequency keeps the ordinary rendition.
        let generic = crate::data::colormap::categorical_mode(
            vec!["a".to_string(), "b".to_string()],
            None,
        );
        assert!(!generic.is_color_map());
        // Neither is a graduated ramp.
        assert!(!StyleMode::Graduated {
            method: ClassMethod::Quantile,
            breaks: vec![1.0, 2.0],
        }
        .is_color_map());
    }

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

        let eq = classify_breaks(ClassMethod::EqualInterval, &mut vals.clone(), STYLE_BINS);
        assert_eq!(eq.len(), STYLE_BINS - 1);
        assert!(eq.windows(2).all(|w| w[0] <= w[1]), "ascending");
        // Equal interval spreads over the full range: most breaks above 1.
        assert!(eq.iter().filter(|b| **b > 1.0).count() > 10);

        let q = classify_breaks(ClassMethod::Quantile, &mut vals.clone(), STYLE_BINS);
        assert_eq!(q.len(), STYLE_BINS - 1);
        assert!(q.windows(2).all(|w| w[0] <= w[1]));
        // Quantiles follow the mass: most breaks inside the dense [0,1).
        assert!(q.iter().filter(|b| **b < 1.0).count() >= 12, "{q:?}");

        let j = classify_breaks(ClassMethod::Jenks, &mut vals.clone(), STYLE_BINS);
        assert_eq!(j.len(), STYLE_BINS - 1);
        assert!(j.windows(2).all(|w| w[0] <= w[1]));
        // Natural breaks must isolate the [100, 200) cluster from [0, 1).
        assert!(j.iter().any(|b| *b > 1.0 && *b <= 101.0), "{j:?}");

        let sd = classify_breaks(ClassMethod::StdDev, &mut vals.clone(), STYLE_BINS);
        assert_eq!(sd.len(), STYLE_BINS - 1);
        assert!(sd.windows(2).all(|w| w[0] <= w[1]));

        // Degenerate inputs don't panic.
        assert_eq!(
            classify_breaks(ClassMethod::Jenks, &mut vec![5.0; 10], STYLE_BINS).len(),
            STYLE_BINS - 1
        );
        assert_eq!(
            classify_breaks(ClassMethod::Quantile, &mut Vec::new(), STYLE_BINS).len(),
            STYLE_BINS - 1
        );
    }

    #[test]
    fn class_count_flows_through_breaks() {
        let mut vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        for classes in [2usize, 5, 8, STYLE_BINS] {
            for m in ClassMethod::ALL {
                let b = classify_breaks(*m, &mut vals.clone(), classes);
                if *m == ClassMethod::HeadTail {
                    // Head/tail decides its own depth; never more than asked.
                    assert!(!b.is_empty() && b.len() < classes, "{m:?}");
                } else {
                    assert_eq!(b.len(), classes - 1, "{m:?} classes={classes}");
                }
                assert!(b.windows(2).all(|w| w[0] <= w[1]), "{m:?} sorted");
            }
        }
        // bin_colors spreads the ramp over exactly the class count.
        let sb = StyleBy {
            column: "v".into(),
            ramp: Ramp::Viridis,
            hidden_bins: 0,
            per_area: false,
            mode: StyleMode::Graduated {
                method: ClassMethod::EqualInterval,
                breaks: equal_interval_breaks(0.0, 100.0, 5),
            },
            classified_rows: None,
            width_px: None,
        };
        let colors = sb.bin_colors();
        assert_eq!(colors[4], Ramp::Viridis.sample(1.0), "last class = ramp end");
        assert_eq!(colors[5], colors[4], "bins past the class count reuse the last color");
        assert_ne!(colors[0], colors[1]);
    }

    #[test]
    fn progression_and_headtail_breaks() {
        // Arithmetic over [0, 100], 4 classes: widths 10/20/30/40.
        assert_eq!(arithmetic_breaks(0.0, 100.0, 4), vec![10.0, 30.0, 60.0]);
        // Geometric spans decades; zeros fold into class 1 without
        // breaking the ratio.
        let mut v = vec![0.0, 0.0, 1.0, 10.0, 100.0, 1000.0];
        let g = classify_breaks(ClassMethod::Geometric, &mut v, 3);
        assert!((g[0] - 10.0).abs() < 1e-9 && (g[1] - 100.0).abs() < 1e-9, "{g:?}");
        // Head/tail on a heavy tail: first break at the overall mean,
        // deeper breaks climb into the head.
        let mut v: Vec<f64> = vec![1.0; 900];
        v.extend([10.0; 90]);
        v.extend([100.0; 9]);
        v.push(1000.0);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let ht = classify_breaks(ClassMethod::HeadTail, &mut v, 8);
        assert!((ht[0] - mean).abs() < 1e-9, "{ht:?} vs mean {mean}");
        assert!(ht.len() >= 2 && ht.windows(2).all(|w| w[0] < w[1]), "{ht:?}");
        // Uniform data cannot split at the mean: falls back, never empty.
        let flat = classify_breaks(ClassMethod::HeadTail, &mut vec![5.0; 40], 6);
        assert_eq!(flat.len(), 5);
    }

    #[test]
    fn breaks_binning_matches_partition() {
        // partition_point semantics: value below first break -> bin 0,
        // above last -> last bin. Sixteen unit-wide classes, stated here
        // rather than taken from STYLE_BINS: the property is about the
        // binning rule, not about how many bins the GPU allows.
        let breaks = equal_interval_breaks(0.0, 16.0, 16);
        let bin = |v: f64| breaks.partition_point(|b| v >= *b);
        assert_eq!(bin(-1.0), 0);
        assert_eq!(bin(0.5), 0);
        assert_eq!(bin(1.0), 1);
        assert_eq!(bin(15.5), 15);
        assert_eq!(bin(99.0), 15);
    }
}

#[cfg(test)]
mod load_state_tests {
    use super::*;

    /// One pass answers every count the layers panel asks for, and it
    /// survives a `loaded` that describes more row groups than the store
    /// has.
    ///
    /// The panel used to ask five separate scans and `loaded_rows`
    /// indexed `rg_starts[g + 1]` directly, which panicked in exactly the
    /// case a pyramid level switch creates: the store shrinks under a
    /// layer whose decode state is still the old level's.
    #[test]
    fn load_summary_counts_in_one_pass_and_survives_a_shrunken_store() {
        let loaded = vec![
            GroupLoad::Full,
            GroupLoad::Rows { ranges: vec![(0, 3), (7, 10)], rect: [0.0; 4] },
            GroupLoad::Preview { stride: 4, rect: None, ranges: None },
            GroupLoad::Boxes { rect: None, ranges: None },
            GroupLoad::None,
        ];
        let starts = [0u64, 10, 20, 30, 40, 50];
        let s = load_summary(&loaded, &starts);
        assert_eq!((s.total, s.full, s.partial, s.preview, s.boxes), (5, 1, 1, 1, 1));
        // 10 full + 6 selected + ceil(10 / 4) preview + 10 boxes.
        assert_eq!(s.rows, 10 + 6 + 3 + 10);
        // The store is now one row group; the state has not caught up.
        let shrunk = load_summary(&loaded, &[0, 10]);
        assert_eq!(shrunk.total, 5);
        assert_eq!(shrunk.rows, 10 + 6, "groups the store no longer has read no rows");
    }
}
