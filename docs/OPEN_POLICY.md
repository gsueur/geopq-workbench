# File quality gate and display policy

Status: implemented (dc1d453). Checks and optimizer defaults are
aligned with the upstream [best practices for distributing
GeoParquet](https://github.com/opengeospatial/geoparquet/blob/main/format-specs/distributing-geoparquet.md):
zstd 15, 50k–150k-row row groups, spatial ordering, page index.
Selecting 2.0 in the optimizer applies the official recommended
settings (native GEOMETRY + native stats only); two opt-in flavor
extras exist: the bbox covering column (page-level spatial pruning —
which that document lists as unsolved — and the cheapest exact viewport
selection; without it, WKB files fall back to a per-group WKB envelope
scan for rect resolution) and
an auxiliary GeoArrow coordinate-array sibling column
(`{primary}_geoarrow`, undeclared in `geo` so the file stays
conformant 2.0; the loader adopts it for decode and hides the
redundant WKB primary from attribute UIs).

## 1. Problem

The viewer's display machinery (row-group pruning, per-feature viewport
selection, refinement, reload-to-viewport) requires per-row-group spatial
bboxes and benefits from spatial ordering. Files that lack these
(e.g. a plain DuckDB export: WKB, no covering column, insertion order)
fall into a trap: the initial load decimates to a stride preview, and
refinement is structurally impossible because every row group's bbox
spans the whole extent — every viewport selects more rows than the
budget, forever. The user sees silent, permanent decimation at every
zoom with no explanation (measured on `parcelles_quebec.parquet`: a
1.5 km viewport intersects 37/37 row groups = 4.45M rows > 2.5M budget).

## 2. Policy

Two display modes, decided once at open, stored on the layer:

- **Indexed** — the current machinery, unchanged: pruning, rect
  selection, stride preview at wide zoom, refinement on camera settle,
  reload-to-viewport. Allowed only for files where refinement provably
  works.
- **Direct** — QGIS-style: decode and tessellate every row once, with
  progress and cancel. No preview, no decimation, no refinement, no
  reload-to-viewport. Wait once, then seamless.

There is no middle state. A stride preview may exist only in Indexed
mode, where zooming in provably replaces it with exact rows.

### Invariants

1. Decimated geometry is never on screen without a visible badge saying
   so, and never in a state that cannot refine to exact rows.
2. Direct mode never decodes fewer than all rows (modulo an explicit
   user filter) and never runs viewport-driven work on camera moves.
3. The quality dialog appears exactly when, without user action, the
   user would otherwise get a permanent preview: verdict = fail AND
   `total_rows > MAX_BUILD_ROWS`. Small files never nag.
4. Analysis is footer-only. No data pages are read to produce the
   verdict (matters for remote sources).
5. A declined offer is remembered per file; the next open goes straight
   to Direct with a passive banner instead of a dialog.

## 3. Quality analysis

Runs in `open_store` on the already-fetched parquet footer + geo
metadata. Produces a `QualityReport` attached to `FileInfo`.

### Checks

| # | Check | Pass | Warn | Fail | Gate? |
|---|-------|------|------|------|-------|
| C1 | Row-group bboxes available from metadata (native geo stats, covering column stats, GeoArrow coord stats, or x/y column stats) | any source | — | none | **yes** |
| C2 | Spatial clustering: `overlap_frac` over the C1 boxes | ≤ `OVERLAP_FRAC_MAX` | — | > `OVERLAP_FRAC_MAX` | **yes** |
| C3 | Row-group granularity: max rows per group | ≤ `RG_ROWS_MAX` | < `RG_ROWS_MIN` (footer bloat) | > `RG_ROWS_MAX` | **yes** |
| C4 | Geometry encoding | GeoArrow / native GEOMETRY | WKB | — | no |
| C5 | Page index (offset + column index) present | yes | no | — | no |
| C6 | Compression | zstd/snappy/lz4 (non-zstd notes "zstd recommended for distribution") | uncompressed | — | no |
| C7 | Metadata hygiene: geo version, declared `geometry_types`, CRS present, file bbox | all present | any missing | — | no |

Verdict = C1 ∧ C2 ∧ C3. C4–C7 are advisory lines on the scorecard
(the "good practices" education), never gate.

Early pass: if `total_rows ≤ MAX_BUILD_ROWS` the verdict is forced to
pass (mode Indexed, which degenerates to a plain full load). The gate
only exists to prevent the permanent-preview trap, and small files
cannot fall into it.

### Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `OVERLAP_FRAC_MAX` | 0.30 | Fraction of possible pairwise bbox overlaps (`RgBboxes::overlap_frac`, the existing `poorly_clustered` threshold). Measured: Hilbert-sorted files 11–25%, attribute-ordered ~35%, spatially random ~100%. An earlier draft used a raw average-overlap cap of 4.0; that was miscalibrated — the known-good optimized Quebec file measures avg 7.3 (frac 10.9%) and would have failed it. |
| `CHAIN_OVERLAP_MIN` | 2.0 | Absolute floor on the C2 pass: sorted files form a bbox chain where each box touches ~2 neighbors regardless of group count, so the fraction test alone spuriously fails small files (4 sorted groups measure 50% on adjacency alone). C2 passes when `avg_overlap ≤ max(CHAIN_OVERLAP_MIN, OVERLAP_FRAC_MAX × (n−1))`. |
| `RG_ROWS_MAX` | 512_000 | Pruning granularity and refine decode unit. 4 × 512k = 2.05M ≤ 2.5M budget. |
| `RG_ROWS_MIN` | 16_000 | Below this, footer size and per-group overhead dominate. Warn only. |
| `DIRECT_MAX_ROWS` | 12_000_000 | Hard ceiling for "Load all anyway". Provisional; calibrate. |
| `DIRECT_MAX_GEOM_BYTES` | 8 GiB | Uncompressed geometry-column bytes from footer (`total_byte_size` of geometry leaves) as a decode-size proxy. Whichever ceiling hits first. Provisional; calibrate. |

Calibration references: `parcelles_quebec_optimized.parquet` (65k
rows/rg, covering, Hilbert) must pass; `parcelles_quebec.parquet` (no
bboxes → C1 fail) must fail and be Direct-loadable (4.45M rows,
~1.9 GiB uncompressed geometry).

Multi-fragment stores (directory / hive / S3 prefix): analysis runs per
fragment; the aggregate verdict is pass only if every fragment passes.
The Optimize offer is disabled for multi-fragment sources (existing
optimizer limitation) — the dialog offers only Direct or Cancel, with a
note.

## 4. Open flow

```
open_store ──► QualityReport
    │
    ├─ verdict pass (or total_rows ≤ MAX_BUILD_ROWS)
    │      └─► Indexed mode, load as today. Scorecard visible in the
    │          file-info panel (advisory warns included).
    │
    └─ verdict fail AND total_rows > MAX_BUILD_ROWS
           ├─ remembered decline for this file?
           │      └─► Direct mode immediately + passive banner
           │          "Unoptimized file — loaded fully (N rows)".
           └─► modal dialog with scorecard summary and:
                  [Optimize…]   pre-filled optimize dialog (existing).
                                On completion, open the optimized file
                                as the layer (original not added).
                  [Load all]    Direct mode. Disabled above
                                DIRECT_MAX_* with the reason shown;
                                then Optimize is the only path.
                  [Cancel]      nothing is added.
```

Decline memory: keyed by canonical path + file size + mtime, stored in
egui app storage. Any of the three changing re-arms the dialog.

Remote sources: the dialog states the cost explicitly — Direct means
downloading effectively the whole file; Optimize additionally writes a
local optimized copy (which is the recommendation for remote work
anyway).

### Dialog copy

> **This file is not optimized for display.**
> `parcelles_quebec.parquet` — 4,452,455 rows, 37 row groups.
> ✗ No spatial index: no row-group bboxes in metadata (C1)
> ✗ Not spatially sorted: every viewport touches all 37 row groups (C2)
> ✓/… advisory lines
>
> Viewport-based loading is not possible. GeoPQ can rewrite it
> (Hilbert sort + bbox covering, ~size estimate), or load all 4.5M
> features up front (~est. memory, no partial loading).
> `[Optimize…] [Load all] [Cancel]`

## 5. Mode × event matrix

| Event | Indexed | Direct |
|-------|---------|--------|
| Initial load | plan_viewport_selection as today (prune → rect/all → preview fallback) | `GroupSel::All` for every group, no budget, progress + cancel |
| Camera settle | refine_partial_layers as today | skipped (mode check; also `is_partial()` is false) |
| Zoom/pan | draw only | draw only |
| Reload-to-viewport | as today | disabled for the layer (tooltip: no spatial index — reload cannot prune) |
| Display CRS change | rebuild as today (loaded states preserved) | rebuild re-decodes all rows (one-shot, progress) |
| Style-by change | rebuild as today | rebuild all rows |
| Row filter | decode selected ranges (existing) | same (existing filter machinery, ranges of the full set) |
| Optimize finishes | n/a | replace layer with optimized file in Indexed mode (new store, fresh load) |
| SQL console | unchanged | unchanged |
| Badges | "Using preview 1/N" / partial row-group line, as today | none, or the passive "loaded fully" banner line in the layer block |

`RefineDeferred` (Indexed only) additionally gets a visible one-line
badge — "viewport too dense to refine (≥N rows), zoom in" — instead of
today's debug-level log. On a passing file this state is reachable but
always escapable by zooming; the badge makes it honest.

## 6. Implementation plan

1. `src/data/quality.rs` (new, ~200 lines): `QualityReport { checks:
   Vec<Check>, verdict, geom_bytes_uncompressed }`, built from
   `ParquetMetaData` + geo meta + the existing `rg_bboxes_from_metadata`
   / `xy_rg_boxes` output and `bbox_overlap_metric`. Unit tests with
   synthesized files (sorted/unsorted, covering/none, tiny/huge groups).
2. `open_store` returns the report inside `FileInfo`; multi-fragment
   aggregation in the store-open path.
3. `LayerMode { Indexed, Direct }` on `VectorLayer`; `spawn_load` takes
   the mode: Direct plans `All` for every group with no budget.
   Progress messages already flow through the loader channel.
4. App: pre-add dialog (blocks the layer add, not the UI), decline
   memory in app storage, banner lines, reload-to-viewport gating,
   optimize-dialog handoff (pre-filled dst, auto-open on Done, swap
   layer).
5. File-info panel: scorecard section rendering all seven checks.
6. Tests: analyzer verdicts (unit); "Direct never spawns refine"
   (invariant); manual/ignored integration on the Quebec pair.

Order: 1–2 land first (report visible in info panel, no behavior
change), then 3 (Direct mode reachable via a debug toggle), then 4–5
(the gate). Each step is shippable and testable alone.

## 7. Out of scope

- Overview sidecars (separate effort, later).
- Page-level pruning in the viewer (C5 is advisory only for now).
- Optimizing multi-fragment sources.
- Any change to Indexed-mode behavior beyond the RefineDeferred badge.
