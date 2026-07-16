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
cargo run --release [file.parquet | dataset_dir/ | https://host/file.parquet ...]
```

Or drag & drop `.parquet` files (or a dataset directory) onto the window /
use **Open…** / **Open folder…**.

Test fixtures (DuckDB spatial) live in `testdata/` but are not committed
(~600 MB, generated). Rebuild them with:

```bash
./testdata/regenerate.sh [/path/to/massgis_parcels.parquet]
cargo run --release testdata/points_1m_wgs84.parquet testdata/polygons_5k_l93.parquet
```

Fixture-dependent tests self-skip when the files are absent.

## Features

- **GeoParquet 1.0 / 1.1 / 2.0**: `geo` metadata (primary column, PROJJSON
  CRS); 2.0 files with native GEOMETRY logical types are detected and load
  through the WKB path; GeoArrow-native 1.1 encodings ("point",
  "multipolygon", ... — nested coordinate arrays) load through a dedicated
  accessor, with per-row-group bboxes derived from the x/y coordinate
  column statistics when no covering/geo stats exist. Files without `geo`
  metadata fall back to a guessed WKB column + CRS84. Files with no
  geometry column at all but lon/lat (or x/y, lng, latitude/longitude…)
  float columns load as point layers: a virtual `Struct{x,y}` geometry is
  synthesized at read time (CRS assumed CRS84, said so in the CRS name),
  per-row-group bboxes come from the coordinate columns' ordinary min/max
  statistics — so viewport pruning works — and the virtual column is
  queryable in SQL like any geometry (WKB via the usual normalization).
  Optimize… materializes such layers into real GeoParquet (synthesized
  WKB points + the full Hilbert/covering treatment; lon/lat stay as
  attribute columns).
- **Hive-partitioned datasets as one layer**: open a directory (File →
  Open folder…, drag & drop, or a CLI argument) holding a multi-file
  GeoParquet dataset — including this viewer's own partitioned exports and
  Overture-style `key=value` layouts. All files present as a single layer
  with one global row-group space, so viewport pruning, refinement,
  picking, layer filters and the SQL console work unchanged across file
  boundaries. Hive path keys (`state=MA/…`) become virtual nullable Utf8
  columns: visible in the info and feature panels, queryable and groupable
  in SQL. Equality filters on them (`where state = 'MA'`) prune whole
  files before any IO, and partition-column-only projections are answered
  from metadata without touching the files. Files lacking per-row-group
  bboxes fall back to their file-level `geo` bbox for pruning. All files
  must share schema, CRS and geometry encoding.
- **Repository browser** (File → Repositories…): browse external
  GeoParquet repositories and load their layers directly, no URL typing.
  A repository is a plain HTTPS object store ("parquetry" layout):
  `snapshots.json` at the root (dated prefixes + `latest/`), one folder
  per region with a `_manifest.json` listing its themes and feature
  counts, one GeoParquet file per theme. The browser picks a snapshot,
  discovers datasets (via the repository's `index.json` — `{"datasets":
  [{"path": "country=US/state=US-AR", "name": "Arkansas"}]}` — or, when
  absent, by probing a built-in ISO 3166-2 grid concurrently), filters
  regions by country or text, and loads any set of checked themes as
  layers over range requests with meaningful names (`US-AR buildings`).
  Layers stack by geometry category — polygons at the bottom, lines,
  then points, each new layer on top of its own category (the layer
  rows show a small type glyph; manual reorder still wins) — so loading
  buildings + roads + pois renders sensibly without shuffling.
  Geomermaids Parquetry (OSM North
  America: 101 regions × 16 themes, daily snapshots) is preconfigured;
  add or remove repositories in the dialog (persisted in
  `~/.config/geopq-viewer/repositories.json`). Discovered dataset lists
  are cached across sessions (`repo_cache.json`, "cached N h ago" shown
  in the dialog); the ⟳ button clears the cache and re-discovers.
- **STAC repositories** (Overture Maps preconfigured): a static STAC
  catalog is browsed the same way — releases as snapshots, themes as
  datasets, type collections as the checkable rows. Loading a type opens
  it as one multi-fragment remote layer built from the collection's part
  files: the STAC items carry each part's bbox, so only the parts
  intersecting the current viewport are opened (capped at 16 files per
  load — zoom in for the dense types like `building`, 512 parts / ~2.5B
  rows), then the usual covering-column row-group/feature pruning takes
  over inside each part. Adding a repository whose URL ends in
  `catalog.json` selects the STAC protocol automatically.
- **Data-driven styling** (layer ☰ → Style by value…): color features by
  an attribute — numeric columns get a graduated ramp (Viridis / Turbo /
  Blue–Red) with a classification method: equal interval (bounds from the
  parquet column statistics, editable), quantiles, half-std-dev classes,
  or Jenks natural breaks. Data-dependent methods classify **only the
  already-loaded rows** (a capped background sample — the app never
  fetches the whole dataset to classify), and the layer panel flags the
  styling as stale when the loaded extent has drifted >25% since
  classification. Text columns get a categorical palette from the 15
  most frequent values (fetched through the SQL engine). Features are
  binned into 16 style bins at tessellation time (chunk meshes split per
  bin, each drawn with its own uniform color), so ramp swaps are free and
  only column/break changes rebuild. Fills, outlines (darkened ramp) and
  points all follow; persisted in saved contexts.
- **File info panel**: per-layer Info button — detected GeoParquet version,
  encoding, CRS, metadata bbox, covering/edges, column types + compression,
  row-group layout, raw `geo` JSON with copy.
- **Row-group bbox visualization**: per-layer checkbox drawing each row
  group's spatial extent (labeled by index), sourced from parquet native
  geospatial statistics (GeoParquet 2.0), `covering` column statistics
  (1.1), or computed during the load stream (1.0/untagged) — plus an
  average-overlap clustering metric that tells you whether the file would
  benefit from a spatial-order rewrite.
- **Remote GeoParquet over HTTP range requests and S3**: open an
  `https://` URL or `s3://bucket/key` (toolbar "URL…" button, or as a CLI
  argument) — the footer, metadata and
  page indexes are fetched with bounded range requests, and the same
  pruning stack below then applies over the network: a 304 MB GeoParquet
  2.0 file (2.7M MA buildings) opens in ~1.3 s, and a city-scale viewport
  (307k buildings, 6 of 53 row groups via native geo statistics) renders
  after ~5 s of parallel per-group range requests. Sequential reads use
  growing bounded windows (256 KB → 8 MB) instead of open-ended ranges;
  files with a page index get exact per-page fetches. Lazy attributes and
  picking work identically (each click is a small ranged read); the whole
  pick pipeline (hit test + attribute row) runs off the UI thread with a
  status-bar spinner, so a slow remote click never freezes the app, and
  very wide schemas (time series in columns) fetch only the first 256
  attribute columns for the info panel (noted in the panel). The parquet
  footer is parsed once and shared by every reader (remote footers can be
  MBs), and response bodies are fully drained so the connection pool
  actually pools (a missing EOF read cost a TLS handshake per range
  request — 3× slower against S3). For `s3://`: pick an AWS profile from
  `~/.aws` in the dialog (static keys + session tokens), and optionally a
  custom S3-compatible endpoint (path-style; e.g. `s3.example.com` or
  `https://minio:9000` — per-profile `endpoint_url` and
  `AWS_ENDPOINT_URL` are honored too), or leave on auto — env credentials, then the default
  profile, then anonymous for public buckets (validated against Overture's
  518 MB building parts: open ~2 s, one covering-pruned row group and
  ~10k buildings in ~8 s). Layers display the `s3://` URI, never the
  presigned URL. Note: opening a remote file while zoomed out to the whole
  extent legitimately downloads all row groups — zoom in first for large
  files.
- **Row-group + per-feature pruning on load**: when metadata bboxes exist,
  only row groups intersecting the viewport are decoded — and inside the
  surviving groups of a covering-column (1.1) file, a cheap scan of the four
  bbox leaves builds an exact RowSelection so only viewport-intersecting
  features are decoded (verified row-for-row identical to the equivalent SQL
  predicate). Only the geometry column is read at load; attributes stay
  lazy. Panning/zooming streams missing rows in the background (debounced):
  unseen groups get a viewport selection, already-selected groups are
  completed with their complement rows (no duplicates). Projection switches
  consolidate sections; the layer panel shows full/viewport-filtered group
  counts with a Load-all button. Global row indices are preserved, so
  picking/attributes work on partial layers. Loads that would decode
  more than ~2.5 M rows switch to a **decimated preview** (every Nth
  row, ~1.2 M features) instead of exhausting memory: the layer panel
  says so, zooming in replaces preview groups with real viewport rows,
  and refinement is row-budgeted (it waits for a tight enough zoom
  rather than re-downloading a world view). Fixture:
  `testdata/parcels_hilbert.parquet` (Hilbert-sorted, 1.1 covering).
- **Optimize / export**: per-layer "Optimize…" rewrites any loaded file as a
  pruning-friendly GeoParquet — features Hilbert-sorted by bbox center, tuned
  row-group size, zstd/snappy — in your choice of:
  - **GeoParquet 1.1**: plain WKB column (native GEOMETRY logical types from
    2.0 sources are stripped for reader compatibility) + `bbox` covering
    struct column, so covering column statistics drive pruning;
  - **GeoParquet 1.1 (GeoArrow)**: geometry as raw coordinate arrays
    (requires a single geometry family; Polygon rows promote into a
    multipolygon column, etc.). The x/y leaves get ordinary parquet
    statistics, so any engine prunes with zero geo-awareness; GDAL reads
    the output natively. GeoArrow sources load through a bulk path: the
    whole coordinate buffer is reprojected in one linear pass and features
    are meshed straight from the arrow offsets, no per-feature geometry
    allocation (verified mesh-identical to the WKB path). Measured
    (matched codec/order): points load ~1.2x faster than WKB, polygons
    are parity — lyon tessellation dominates polygon builds — and files
    run ~7-8% larger.
  - **GeoParquet 2.0**: parquet-native GEOMETRY logical type (CRS carried in
    the type), with native geospatial statistics written per row group — no
    covering column needed, smaller file.
  Geometry transcodes in any direction (WKB ↔ GeoArrow ↔ native type).
  A "viewport only" option exports just the features intersecting the
  current map viewport (bbox-based, metadata bbox shrinks accordingly).
  **Derived columns**: an `h3_r{n}` UInt64 cell column (centroid-based,
  resolution picker — pure-Rust h3o) and admin attribution from any loaded
  boundary polygon layer (state, county, ... via centroid point-in-polygon,
  R-tree accelerated, cross-CRS). **Partitioned output** per the OGC
  distribution guidance: hive directories by chosen fields
  (`state=MA/part-0.parquet`, live distinct-count next to each candidate
  field, partition columns path-only) or adaptive H3 cells that split until
  each file is under a row target (balanced, non-overlapping — for datasets
  with no natural key). Every partition file keeps the full treatment:
  Hilbert order, covering, bloom filters, per-file `geo` bbox.
  The covering bbox leaves are written in small (~4k-row) pages with
  dictionary encoding off, so the parquet page index (ColumnIndex/
  OffsetIndex) prunes well below row-group granularity in any reader.
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
- **Auto projection (default)**: the first loaded layer picks the display
  projection. Projected data CRS wins outright (the mapping agency already
  chose the best local projection — MassGIS parcels display in EPSG:26986);
  geographic data gets a Snyder-style equal-area choice from its extent:
  Albers conic for mid-latitude east–west bands (1/6-rule parallels),
  azimuthal equal-area on the centroid for square-ish regions, equal-area
  cylindrical for equatorial bands, polar azimuthal near the poles, and the
  Hobo–Dyer world default for global data. A curated table of ~30 official
  national CRSs (Wikipedia's list, validated against the embedded EPSG
  database) takes precedence: France-extent data displays in Lambert-93,
  UK in British National Grid, CONUS in EPSG:5070 Albers, tightest country
  wins on overlap. The national grids are also directly selectable in the
  projection combo. Selected before geometry builds
  when metadata bboxes exist (no double build, remote-safe); any manual
  projection choice turns auto off for the session.
- **Display projection switching**: Auto (above), Hobo–Dyer cylindrical
  equal-area (world default), Winkel Tripel (implemented natively — analytic
  forward, Newton inverse), Web Mercator (with basemap), WGS84 plate carrée,
  or any EPSG code typed in the toolbar. Layers re-tessellate in the
  background on switch.
- **Graticule + coastline**: 15° meridians/parallels and the Natural Earth
  1:50m coastline (public domain, ~0.5 MB embedded in the binary), both
  rendered through the active projection (toggleable).
- **Basemaps**: Carto Light/Dark/Voyager, OSM raster tiles with LRU cache and
  ancestor fallback while loading (EPSG:3857 only).
- **SQL console** (toolbar "SQL", pure Rust — no DuckDB): query loaded layers
  with DataFusion plus our own ST_* spatial UDFs (19 functions: measures,
  predicates, constructors, buffer/simplify/convexhull/centroid, WKT
  conversions — WKB in/out on geo-types). Two modes, TablePlus-style:
  **Browse** (table picker + WHERE bar + limit, Enter applies) and **Query**
  (free-form SQL, Ctrl+Enter). Layer tables stream row groups through the
  same `FeatureStore`/range-request stack as map loading — local, HTTP and
  S3 layers all queryable — with projection pushdown, GeoArrow geometry
  normalized to WKB, and column names lowercased so unquoted identifiers
  match any file. **Spatial predicate pushdown**: `st_intersects /
  st_within / st_contains / st_dwithin` against a literal geometry
  (`st_makeenvelope(...)`, `st_geomfromtext(...)` — constant-folded) prune
  row groups via metadata bboxes and rows via the covering bbox-leaf scan,
  then DataFusion re-applies the exact predicate (verified row-identical to
  the unpruned scan). The results grid: checkbox per row builds a
  highlighted map selection, a magnifier zooms to a feature, any cell click
  copies its full value (geometry as WKT), checked rows copy as TSV with
  headers. Results with geometry export to a temp GeoParquet and load back
  as regular layers (whole result or checked rows only), carrying the
  source layer's CRS.
- **Layer filters**: a persistent SQL predicate per layer (⋮ → Filter…) —
  the working subset, not a query: only matching rows are decoded, shown,
  picked and queried until cleared. The predicate runs through the SQL
  engine (spatial predicates pruned via row-group/page statistics, so a
  location filter on a huge remote file reads only the relevant groups),
  survives projection switches, and persists in saved contexts. The dialog
  has column-name autocomplete and a Test button (validates + counts
  matches without touching the layer; Apply reuses the tested result).
- **Attributes, lazily**: click a feature → highlight + a floating attribute
  window (upper right, draggable — the map never resizes). Values are
  fetched from the parquet file on demand (row-group + page selection via
  `FeatureStore`), nothing is cached in RAM. Picking uses an R-tree over
  line/polygon bboxes, chunk-local scans for points, and an exact geometry
  test in the data CRS.
- **Styling**: per-layer fill/point color, separate border/line color (auto-derived
  dark shade by default, ↺ resets), opacity, fill opacity, line width, point
  radius; per-layer menu for info/optimize/reorder/remove (list order =
  draw order) with an inline zoom-to-layer button.
- **Session context**: File → "Save context…" writes layers (sources incl.
  remote URLs and s3 URIs + profile/endpoint — never presigned URLs), styles,
  order, camera, projection and toggles to a JSON file; "Load context…"
  restores it, re-resolving credentials and re-pruning against the restored
  viewport.
- **Status bar**: cursor in WGS84 + display CRS, projection picker (built-ins,
  national grids, EPSG entry), zoom, frame time, feature counts, load
  progress with per-stage percentages, error badge.

## Architecture

| Module | Role |
|---|---|
| `data/loader.rs` | streaming parquet read, `geo` metadata, WKB decode (fast path for 2D points), rayon-parallel build |
| `data/store.rs` | `FeatureStore`: lazy row fetch (attributes, WKB) via row-group/RowSelection reads |
| `data/source.rs` | local file / HTTP / S3 `ChunkReader` (windowed range reads, presigned S3 URLs, ~/.aws profiles) |
| `data/optimize.rs` | Hilbert-sort rewrite to GeoParquet 1.1 (covering) or 2.0 (native GEOMETRY + geo stats), bloom filters |
| `data/crs.rs` | PROJJSON → EPSG → proj4rs; native Winkel Tripel; `DisplayCrs` maps projected coords to normalized world space |
| `data/geometry.rs` | lyon fill tessellation, line segments, points, accumulated into spatial chunks with per-chunk f64 origins (f32 GPU precision at deep zoom) |
| `map/renderer.rs` | wgpu pipelines (fill / SDF lines / SDF points / tiles), per-chunk dynamic uniform offsets, drawn inside egui's render pass |
| `map/tiles.rs` | XYZ fetch worker pool, tile cache, ancestor fallback |
| `picking.rs` | R-tree candidates → exact test in data CRS |
| `sql/table.rs` | DataFusion `TableProvider` over `FeatureStore` (partition per row group, projection + spatial predicate pushdown) |
| `sql/udf.rs` | ST_* scalar UDFs on geo-types, WKB in/out |
| `sql/engine.rs` | session per query on a worker thread, result cap, geometry detection |
| `sql/console.rs` + `sql/export.rs` | console UI (browse/query, results grid) and result → GeoParquet export |
| `app.rs` | egui UI: menu bar, toolbar, layer panel, floating attributes, status bar |

Rendering happens in normalized "world" space (`[0,1]²` slippy convention for
Mercator). Vertices are stored as f32 offsets from per-chunk f64 origins, and
the camera-to-chunk translation is computed in f64 on the CPU each frame, so
deep zoom stays jitter-free.

## Roadmap ideas

### Features

- Overview/pyramid levels for zoomed-out views of huge remote files
  (decimated previews bound memory today, but a world view of a huge
  remote file still downloads every candidate row group's coordinates)
- Remote hive datasets (s3:// / https:// prefix listing; local dirs and
  STAC-listed collections work)
- STAC V2: lazy part-append on camera move (today the part selection is
  fixed at load; pan past it and reload instead)
- Optimize over a multi-file dataset (single files only today)
- Streaming (external-sort) optimize for files beyond the 8 GB in-memory cap
- Page-index (footer-only) pruning for 2.0 native-stats files without a
  covering column (per-feature selection needs the bbox column today)
- SQL: `st_transform` (proj4rs), polygon set ops (`st_union` /
  `st_intersection` via geo bool_ops), aggregates (`st_extent`,
  `st_collect`), query history, save result to a chosen path
- Zoom-dependent decimation / LOD for very dense line/polygon layers
- Zoom-dependent coastline detail (embed 1:110m + fetch 1:10m on demand)
- Basemap tiles warped to non-Mercator projections
- Label rendering from attribute columns
- GeoParquet 1.1 `edges: spherical`; 2.0 CRS read from the GEOMETRY
  logical type when `geo` metadata is absent
