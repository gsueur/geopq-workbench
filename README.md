<p align="center"><img src="assets/icon-256.png" width="100" alt="GeoPQ Workbench icon"/></p>

# GeoPQ Workbench

**Open, inspect, fix and query GeoParquet. A fast native desktop app, no
server, no GDAL.**

GeoPQ Workbench is a desktop application (Rust, egui + wgpu) for working
with GeoParquet at any size. It renders millions of features at
interactive frame rates, tells you exactly why a file is slow, rewrites
it into a fast one, and lets you query everything in SQL. It opens local
files, plain HTTPS URLs, S3 buckets, and whole catalogs like Overture
Maps.

It is also opinionated on purpose: most GeoParquet in the wild is a raw
database export that performs far below what the format allows. The
workbench shows you what your file is missing and fixes it in one click,
so it doubles as a hands-on guide to GeoParquet best practices.

## Highlights

- **Fast**: 3.75M points load in ~180 ms and pan/zoom at vsync. Memory
  scales with what is on screen, not with file size; attributes stay on
  disk until you click a feature.
- **Opens everything GeoParquet**: versions 1.0, 1.1 (WKB and GeoArrow
  coordinate arrays), 2.0 with native GEOMETRY types, files with no geo
  metadata at all, even bare tables with lon/lat columns.
- **Quality scorecard**: every file is graded at open (spatial index,
  spatial ordering, row groups, encoding, compression, metadata). Green
  means fast; anything else comes with an explanation.
- **One-click Optimize**: rewrite any layer as a best-practice
  GeoParquet (spatially sorted, indexed, zstd), in your choice of 1.1
  WKB, 1.1 GeoArrow, or 2.0 native, with optional H3 / admin-boundary
  columns and partitioned output.
- **Remote-native**: open `https://` and `s3://` files in place over
  range requests. A 304 MB file opens in ~1.3 s; only the row groups
  under your viewport are downloaded.
- **Catalog browser**: Overture Maps and Geomermaids Parquetry are
  preconfigured; check the layers you want and they load straight onto
  the map.
- **SQL console**: DataFusion with 19 spatial functions over every
  loaded layer, local or remote, with spatial predicate pushdown.
  Results can be highlighted on the map or exported as new layers.
- **Cartography that defaults well**: automatic equal-area projection
  choice per dataset, ~30 official national grids, data-driven styling
  with Viridis/Turbo ramps, Jenks/quantile classification, basemaps.
- **No barriers to entry**: built-in sample datasets and a pure-Rust
  GeoPackage importer (no GDAL) if your data is not in GeoParquet yet.

## Install

Prebuilt binaries for all four desktop targets are on the
[GitHub Releases](https://github.com/gsueur/geopq-workbench/releases) page,
each with a SHA-256 checksum. No installer, no runtime dependencies —
unpack the archive and run the binary inside. The builds are not
code-signed (no Apple notarization, no Windows certificate), so each OS
needs its one-time "yes, I trust this" step, spelled out below.

| Platform | Archive |
|---|---|
| macOS Apple Silicon | `geopq-workbench-<version>-aarch64-apple-darwin.tar.gz` |
| macOS Intel | `geopq-workbench-<version>-x86_64-apple-darwin.tar.gz` |
| Linux x86_64 | `geopq-workbench-<version>-x86_64-unknown-linux-gnu.tar.gz` |
| Windows x86_64 | `geopq-workbench-<version>-x86_64-pc-windows-msvc.zip` |

### macOS

```bash
# 1. Extract (if starting from the .tar)
tar -xf geopq-workbench-v0.2.0-aarch64-apple-darwin.tar

# 2. Fix the missing execute bit on the extracted directory
#    (tarball packed dirs as 0666 -> can't traverse without +x)
#    Capital X adds x only to dirs / already-executable files
chmod -R u+rwX geopq-workbench-v0.2.0-aarch64-apple-darwin

# 3. Strip macOS quarantine so Gatekeeper doesn't block the binary
xattr -dr com.apple.quarantine geopq-workbench-v0.2.0-aarch64-apple-darwin

# 4. Launch
cd geopq-workbench-v0.2.0-aarch64-apple-darwin
./geopq-workbench
```

Intel Macs: same steps with the `x86_64-apple-darwin` archive. If
Gatekeeper still complains, right-click the binary → Open once, or allow
it under System Settings → Privacy & Security.

### Windows

Unzip (right-click → Extract All, or `Expand-Archive` in PowerShell) and
run `geopq-workbench.exe`. SmartScreen will warn about an unrecognized
app the first time: click **More info → Run anyway**. If the window opens
blank or not at all, update your GPU driver — the renderer needs a
working Direct3D 12 or Vulkan stack.

### Linux

```bash
tar xzf geopq-workbench-v0.2.0-x86_64-unknown-linux-gnu.tar.gz
cd geopq-workbench-v0.2.0-x86_64-unknown-linux-gnu
./geopq-workbench
```

Needs a Wayland or X11 session and Vulkan drivers (`mesa-vulkan-drivers`
on Debian/Ubuntu). File dialogs go through the XDG desktop portal —
present on any mainstream desktop, may need
`xdg-desktop-portal-gtk`/`-kde` on minimal setups. The binary is built on
Ubuntu 22.04, so it runs on distributions with glibc 2.35 or newer; for
older ones, build from source.

### Verify a download

```bash
shasum -a 256 -c geopq-workbench-<version>-<target>.tar.gz.sha256
```

### Build from source

Any platform with stable Rust:

```bash
git clone https://github.com/gsueur/geopq-workbench
cd geopq-workbench
cargo build --release   # binary in target/release/geopq-workbench
```

## Quick start

1. **Launch it.** Nothing to configure.
2. **Get data on the map.** Drag & drop `.parquet` files or a dataset
   folder onto the window, or use File → Open… / Open folder… / Open
   URL… No GeoParquet at hand? **File → Open sample dataset** streams
   small hosted samples, and **File → Import GeoPackage…** converts a
   `.gpkg` feature table on the spot.
3. **Click around.** Click any feature for its attributes (fetched from
   the file on demand, nothing preloaded). The status bar shows
   coordinates, zoom, frame time and load progress.
4. **Open the file info** (layer ℹ button) to see the quality scorecard:
   what the file does well, what it is missing, and what that costs you.
5. **If a big unoptimized file pauses at a dialog**, that is the quality
   gate: choose **Optimize** (recommended: rewrites the file properly,
   then opens the fast copy) or **Load all** (QGIS-style full decode,
   slower but complete; remembered per file).

The app also takes paths and URLs as command-line arguments:

```bash
geopq-workbench file.parquet dataset_dir/ https://host/data.parquet
```

## Opening data

| Source | How |
|---|---|
| Local GeoParquet | drag & drop, File → Open…, or CLI argument |
| Multi-file / hive-partitioned dataset | File → Open folder… — the directory loads as a single layer; `key=value` path segments become queryable columns |
| HTTPS | File → Open URL… — needs range-request support on the server |
| S3 (`s3://bucket/key`) | File → Open URL… — profiles from `~/.aws`, custom endpoints (MinIO etc.), anonymous access to public buckets |
| Catalogs (Overture Maps, Parquetry, your own) | File → Repositories… — browse snapshots, check layers, they load with sensible names and stacking order |
| GeoPackage | File → Import GeoPackage… — pure-Rust conversion to GeoParquet, then opens normally |
| Sample data | File → Open sample dataset |

Format-wise, anything in the GeoParquet family works: 1.0, 1.1 with WKB
or GeoArrow coordinate arrays, 2.0 with native GEOMETRY logical types,
plus untagged parquet with a guessable geometry column, and tables with
only lon/lat (or x/y) columns, which load as point layers directly.
Mixed CRS across layers is fine; everything reprojects onto one map.

One tip for very large remote files: zoom to your area of interest
before or right after opening. The viewport drives what gets
downloaded, and a whole-world view of a country-sized file legitimately
wants every row group.

## Understanding a file

- **Quality scorecard** (layer ℹ → File info): seven checks run against
  the parquet footer alone, before any data is read. The three gating
  ones: is there a spatial index (bbox covering column, native geo
  statistics, or coordinate statistics)? is the file spatially ordered
  (row-group bbox overlap)? are the row groups a sensible size? Plus
  advisory checks on encoding, page index, compression and metadata
  hygiene. Each line explains itself.
- **Row-group bboxes on the map** (per-layer checkbox): draws each row
  group's spatial extent, so you can *see* clustering. A well-sorted
  file shows a neat mosaic; a raw export shows every box covering the
  whole extent, which is exactly why it cannot be displayed
  incrementally.
- **File info** also lists version, encoding, CRS, per-column types and
  compression, row-group layout, and the raw `geo` metadata JSON.
- **The quality gate**: files that fail the gating checks and are too
  big to just decode (over ~2.5M rows) pause at a dialog instead of
  silently rendering a decimated view that can never refine. You choose:
  Optimize now, or load everything the slow way. Your choice is
  remembered per file in `~/.geopq-workbench.json`.

The full policy is written up in
[docs/OPEN_POLICY.md](docs/OPEN_POLICY.md).

## Making files fast

Per-layer **Optimize…** rewrites any loaded file (local or remote) as a
best-practice GeoParquet: features sorted along a Hilbert curve, tuned
row groups, zstd compression at the level the
[GeoParquet distribution guidance](https://github.com/opengeospatial/geoparquet/blob/main/format-specs/distributing-geoparquet.md)
recommends, page indexes, bloom filters. Three output flavors:

- **GeoParquet 1.1 (WKB + bbox column)**: opens everywhere, and the
  bbox covering column gives readers per-feature spatial selection.
  The safe choice for files that travel.
- **GeoParquet 1.1 (GeoArrow)**: geometry as raw coordinate arrays,
  the fastest to decode and display; needs a single geometry family
  (Polygon promotes into MultiPolygon and so on). Recommended by the
  dialog whenever your data fits it.
- **GeoParquet 2.0 (native GEOMETRY)**: the new Parquet-native
  geospatial type with built-in statistics; picking it applies the
  official recommended settings. Two optional extras: keep the bbox
  column (still the only way to get page-level spatial pruning and
  exact viewport selection) and/or add an auxiliary GeoArrow column so
  GeoArrow-aware readers get the fast decode path from a fully
  conformant 2.0 file.

Beyond the rewrite itself:

- **Viewport only**: export just the features intersecting the current
  view.
- **Derived columns**: an H3 cell column at your chosen resolution, and
  admin attribution (state, county, …) from any loaded boundary layer.
- **Partitioned output**: hive directories by chosen fields
  (`state=MA/part-0.parquet`) or adaptive H3 cells that split until each
  file is under a row target. Every part keeps the full treatment.
- **Before/after report**: size, row groups, and a spatial-clustering
  metric, with a button to load the result directly. A 1.89M-parcel
  cadastral file rewrites in ~7 s.

The optimizer holds the dataset in memory (capped at 8 GB uncompressed);
larger files are on the roadmap.

## Exploring the map

- **Projections**: the first layer picks a sensible display projection
  automatically — projected data keeps its own CRS, geographic data gets
  an equal-area choice fitted to its extent (Albers, azimuthal, or
  cylindrical), a curated table of ~30 official national grids takes
  precedence (France displays in Lambert-93, UK in British National
  Grid, CONUS in EPSG:5070), and world-scale data uses Hobo–Dyer. Or
  pick any projection / EPSG code in the status bar; layers re-project
  in the background.
- **Styling**: fill, border and point colors with sensible auto-derived
  defaults, opacity, line width, point radius. **Style by value** colors
  by any attribute: numeric columns get Viridis/Turbo/Blue–Red ramps
  with equal-interval, quantile, half-std-dev or Jenks classification;
  text columns get a categorical palette of the most frequent values.
- **Basemaps**: Carto Light/Dark/Voyager and OSM raster tiles (Web
  Mercator), plus an embedded Natural Earth coastline and graticule that
  follow any projection.
- **Layer filters** (⋮ → Filter…): a persistent SQL predicate per layer;
  only matching rows are decoded, drawn, picked and queried until you
  clear it. Spatial predicates prune at the file level, so a location
  filter on a huge remote file only reads the relevant row groups.
- **Reload to viewport** (View menu): drop everything loaded and reload
  just the current view — reclaims memory after a continent-sized
  detour.
- **Attributes on click**, fetched lazily from the file; a floating
  window that never resizes the map. Works identically on remote files
  (each click is a small ranged read, off the UI thread).
- **Session contexts**: File → Save context… stores layers (including
  remote URLs and S3 sources), styles, order, camera and projection as a
  JSON file; Load context… restores the whole session.

Large files never lock you out: views that would decode more than
~2.5M rows render as a live decimated preview, and zooming in streams
the real rows for your viewport in the background.

## Querying with SQL

The SQL console (toolbar → SQL) runs DataFusion with 19 `ST_*` spatial
functions (measures, predicates, constructors, buffer / simplify /
convex hull / centroid, WKT in/out) over every loaded layer — local,
HTTPS and S3 alike. Two modes, TablePlus-style: **Browse** (pick a
table, type a WHERE clause) and **Query** (free-form SQL, Ctrl+Enter).

```sql
select name, st_area(geometry) as m2
from buildings
where st_intersects(geometry, st_makeenvelope(-71.1, 42.3, -71.0, 42.4))
order by m2 desc limit 100;
```

Spatial predicates against literal geometries push down into the
parquet reads: row groups are pruned by their bboxes and rows by the
covering column, so the query above touches only the files' relevant
parts, then the exact predicate is re-applied. Results can highlight
features on the map, zoom to a feature, copy as TSV, or export back as
a regular layer (whole result or checked rows).

## Performance notes

Measured on real datasets, release build:

- 3.75M points: ~180 ms load, vsync pan/zoom.
- 304 MB remote GeoParquet 2.0 (2.7M buildings): opens in ~1.3 s; a
  city-scale viewport (307k buildings, 6 of 53 row groups) renders
  after ~5 s of parallel range requests.
- Overture buildings on S3 (518 MB part): open ~2 s, ~10k buildings for
  a city viewport in ~8 s, anonymously.
- Memory is bounded by what is displayed (~12 bytes per point of GPU
  geometry), not by file size, and the row budget (~2.5M rows per load)
  keeps worst-case files from exhausting RAM.

## Under the hood

Pure-Rust stack end to end: parquet/arrow readers, proj4rs
reprojection, rayon + lyon tessellation, custom wgpu render pipelines
inside an egui shell. No GDAL, no web view, no server process.

| Module | Role |
|---|---|
| `data/loader.rs` | streaming reads, metadata, quality analysis, parallel geometry build |
| `data/store.rs` | lazy row access (attributes, geometry) over row-group/page selections |
| `data/source.rs` | local / HTTPS / S3 readers with windowed range requests |
| `data/optimize.rs` | the Optimize rewrite (Hilbert sort, covering, 1.1/2.0 flavors, partitioning) |
| `data/gpkg.rs` | GeoPackage import (bundled SQLite) |
| `data/crs.rs` | PROJJSON → EPSG → proj4rs, projection selection |
| `map/` | wgpu pipelines, tiles, chunked f64-origin geometry for jitter-free deep zoom |
| `sql/` | DataFusion integration, ST_* UDFs, spatial pushdown, console UI |
| `app.rs` | the egui application |

Design notes live in [docs/OPEN_POLICY.md](docs/OPEN_POLICY.md) (file
quality gate and display policy).

## Roadmap

- Overview/pyramid levels for zoomed-out views of huge remote files
- Remote hive datasets (prefix listing over `s3://` / `https://`)
- Streaming optimize beyond the 8 GB in-memory cap; multi-file optimize
- Lazy part-append while panning across STAC collections
- More SQL: `st_transform`, polygon set operations, spatial aggregates,
  query history
- Zoom-dependent level of detail for very dense layers; label rendering
- Basemap tiles warped to non-Mercator projections

## Development

```bash
cargo test          # 130+ tests, no fixtures needed for most
cargo run --release testdata/points_1m_wgs84.parquet
```

Test fixtures are generated, not committed: `./testdata/regenerate.sh`
rebuilds them with the DuckDB CLI (fixture-dependent tests self-skip
when absent). CI runs the suite on Linux, macOS and Windows, including
GPU render tests on a software rasterizer. Releases are tag-driven:
bump the version in `Cargo.toml`, tag `vX.Y.Z`, push the tag, and the
workflow builds and publishes all four platform archives with
checksums.

## Credits

GeoPQ Workbench is created by [Geomermaids](https://www.geomermaids.com).

## License

MIT OR Apache-2.0, at your option. Copyright Guillaume Sueur,
Geomermaids.
