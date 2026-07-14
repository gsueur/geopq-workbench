# geopq-viewer

Fast native GeoParquet cartographic viewer, written in Rust (egui + wgpu).

Loads GeoParquet directly (parquet → arrow → WKB), reprojects with pure-Rust
proj4rs, tessellates on all cores with rayon/lyon, and renders through custom
wgpu pipelines. Attributes stay on disk: the file streams through the build
once and only GPU-ready geometry is kept, so memory scales with vertex count
(~12 bytes/point), not file size. 3.75M points load in ~180 ms and pan/zoom
at vsync.

## Run

```bash
cargo run --release [file.parquet ...]
```

Or drag & drop `.parquet` files onto the window / use **Open…**.

Test fixtures (DuckDB spatial) live in `testdata/` but are not committed
(~600 MB, generated). Rebuild them with:

```bash
./testdata/regenerate.sh [/path/to/massgis_parcels.parquet]
cargo run --release testdata/points_1m_wgs84.parquet testdata/polygons_5k_l93.parquet
```

Fixture-dependent tests self-skip when the files are absent.

## Features

- **GeoParquet 1.0 / 1.1 / 2.0**: `geo` metadata (primary column, WKB encoding,
  PROJJSON CRS); 2.0 files with native GEOMETRY logical types are detected and
  load through the WKB path. Files without `geo` metadata fall back to a
  guessed WKB column + CRS84. GeoArrow-native 1.1 encodings not yet supported.
- **File info panel**: per-layer Info button — detected GeoParquet version,
  encoding, CRS, metadata bbox, covering/edges, column types + compression,
  row-group layout, raw `geo` JSON with copy.
- **Row-group bbox visualization**: per-layer checkbox drawing each row
  group's spatial extent (labeled by index), sourced from parquet native
  geospatial statistics (GeoParquet 2.0), `covering` column statistics
  (1.1), or computed during the load stream (1.0/untagged) — plus an
  average-overlap clustering metric that tells you whether the file would
  benefit from a spatial-order rewrite.
- **Row-group pruning on load**: when metadata bboxes exist, only row groups
  intersecting the viewport are decoded; panning/zooming out streams the
  missing groups in the background (debounced, appended as sections without
  re-uploading loaded geometry). Projection switches consolidate sections.
  Layer panel shows "partial: N/M row groups" with a Load-all button.
  Global row indices are preserved, so picking/attributes work on partial
  layers. Fixture: `testdata/parcels_hilbert.parquet` (Hilbert-sorted, 1.1
  covering).
- **Optimize / export**: per-layer "Optimize…" rewrites any loaded file as a
  pruning-friendly GeoParquet — features Hilbert-sorted by bbox center, tuned
  row-group size, zstd/snappy — in your choice of:
  - **GeoParquet 1.1**: plain WKB column (native GEOMETRY logical types from
    2.0 sources are stripped for reader compatibility) + `bbox` covering
    struct column, so covering column statistics drive pruning;
  - **GeoParquet 2.0**: parquet-native GEOMETRY logical type (CRS carried in
    the type), with native geospatial statistics written per row group — no
    covering column needed, smaller file.
  Bloom filters found on source columns are reproduced (or added to all
  string/integer attribute columns, or dropped). The report shows size,
  row-group count and the bbox-overlap metric before/after, and the result
  can be loaded as a layer directly. Real-world effect on the 1.89M-parcel
  MassGIS file: row-group bbox overlap drops from 35% to 25% of possible
  overlaps at 65k-row groups, in ~7 s. In-memory rewrite, capped at 8 GB
  uncompressed.
- **CRS handling**: data CRS read from PROJJSON (EPSG id), transformed via
  proj4rs + the embedded `crs-definitions` EPSG database. Mixed-CRS layers
  coexist on one map (e.g. Lambert-93 polygons over WGS84 points).
- **Display projection switching**: Hobo–Dyer cylindrical equal-area (default
  world projection), Winkel Tripel (implemented natively — analytic forward,
  Newton inverse), Web Mercator (with basemap), WGS84 plate carrée, or any
  EPSG code typed in the toolbar. Layers re-tessellate in the background on
  switch.
- **Graticule + coastline**: 15° meridians/parallels and the Natural Earth
  1:50m coastline (public domain, ~0.5 MB embedded in the binary), both
  rendered through the active projection (toggleable).
- **Basemaps**: Carto Light/Dark/Voyager, OSM raster tiles with LRU cache and
  ancestor fallback while loading (EPSG:3857 only).
- **Attributes, lazily**: click a feature → highlight + full attribute panel.
  Values are fetched from the parquet file on demand (row-group + page
  selection via `FeatureStore`), nothing is cached in RAM. Picking uses an
  R-tree over line/polygon bboxes, chunk-local scans for points, and an exact
  geometry test in the data CRS.
- **Styling**: per-layer color, opacity, fill opacity, line width, point radius.
- **Status bar**: cursor in WGS84 + display CRS, zoom, frame time, feature counts.

## Architecture

| Module | Role |
|---|---|
| `data/loader.rs` | streaming parquet read, `geo` metadata, WKB decode (fast path for 2D points), rayon-parallel build |
| `data/store.rs` | `FeatureStore`: lazy row fetch (attributes, WKB) via row-group/RowSelection reads |
| `data/optimize.rs` | Hilbert-sort rewrite to GeoParquet 1.1 (covering) or 2.0 (native GEOMETRY + geo stats), bloom filters |
| `data/crs.rs` | PROJJSON → EPSG → proj4rs; native Winkel Tripel; `DisplayCrs` maps projected coords to normalized world space |
| `data/geometry.rs` | lyon fill tessellation, line segments, points, accumulated into spatial chunks with per-chunk f64 origins (f32 GPU precision at deep zoom) |
| `map/renderer.rs` | wgpu pipelines (fill / SDF lines / SDF points / tiles), per-chunk dynamic uniform offsets, drawn inside egui's render pass |
| `map/tiles.rs` | XYZ fetch worker pool, tile cache, ancestor fallback |
| `picking.rs` | R-tree candidates → exact test in data CRS |
| `app.rs` | egui UI: toolbar, layer panel, attribute panel, status bar |

Rendering happens in normalized "world" space (`[0,1]²` slippy convention for
Mercator). Vertices are stored as f32 offsets from per-chunk f64 origins, and
the camera-to-chunk translation is computed in f64 on the CPU each frame, so
deep zoom stays jitter-free.

## Roadmap ideas

- Native GeoArrow encodings (`geoarrow.point` etc.) without WKB decode
- Streaming (external-sort) optimize for files beyond the 8 GB in-memory cap
- Zoom-dependent coastline detail (embed 1:110m + fetch 1:10m on demand)
- HTTP range-request streaming of remote GeoParquet (row-group pyramids)
- Data-driven styling (color by attribute, graduated/categorical)
- Zoom-dependent decimation / LOD for very dense line/polygon layers
- Attribute table view + filtering (DataFusion over the parquet file)
- Label rendering from attribute columns
