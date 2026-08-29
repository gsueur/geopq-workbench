//! The GeoParquet cookbook: an in-app guide to the format's versions,
//! geometry encodings and performance levers — the "why" behind the
//! quality scorecard and the Optimize dialog.

use eframe::egui::{self, RichText};

use crate::theme;

fn h(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(
        RichText::new(text)
            .heading()
            .color(if ui.visuals().dark_mode { theme::ACCENT } else { theme::ACCENT_DARK }),
    );
    ui.add_space(2.0);
}

fn p(ui: &mut egui::Ui, text: &str) {
    ui.label(text);
    ui.add_space(4.0);
}

fn row(ui: &mut egui::Ui, w: f32, name: &str, text: &str) {
    ui.label(RichText::new(name).strong());
    cell(ui, w, text);
    ui.end_row();
}

/// A fixed-width, wrapping cell: forces its grid column to `w`, so the
/// comparison chart fills the window instead of hugging its content.
fn cell(ui: &mut egui::Ui, w: f32, text: &str) {
    ui.vertical(|ui| {
        ui.set_width(w);
        ui.label(text);
    });
}

fn row4(ui: &mut egui::Ui, w: f32, name: &str, a: &str, b: &str, c: &str) {
    ui.label(RichText::new(name).strong());
    cell(ui, w, a);
    cell(ui, w, b);
    cell(ui, w, c);
    ui.end_row();
}

pub fn window(ctx: &egui::Context, open: &mut bool, area: egui::Rect) {
    egui::Window::new("GeoParquet cookbook")
        .open(open)
        .default_width(780.0)
        .default_height(560.0)
        .vscroll(true)
        .constrain_to(area).show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            let spacing_x = ui.spacing().item_spacing.x;
            // Text column width for the two-column sections: fill the
            // window (110 px label column), capped for readability.
            let wide = (ui.available_width() - 110.0 - spacing_x).clamp(220.0, 620.0);

            p(ui,
                "GeoParquet is Apache Parquet with geometry columns and a \
                 little spatial metadata. Everything that makes Parquet fast \
                 for analytics — columnar layout, statistics, compression, \
                 selective reads — applies to spatial data too, but only if \
                 the file is written with care. This page is the map.");

            h(ui, "Versions");
            egui::Grid::new(("cb_versions", wide as i32)).num_columns(2).striped(true).show(ui, |ui| {
                row(ui, wide, "1.0",
                    "The baseline: WKB geometry + `geo` file metadata (CRS, \
                     geometry types, bbox). Universally readable, no indexing \
                     story.");
                row(ui, wide, "1.1",
                    "Adds the bbox covering column (a per-feature spatial \
                     index) and GeoArrow native encodings. Files can be \
                     pruned and read incrementally.");
                row(ui, wide, "2.0",
                    "Geometry becomes a first-class Parquet type: GEOMETRY / \
                     GEOGRAPHY logical types with the CRS inside and native \
                     geospatial statistics per row group. Any Parquet reader \
                     can prune spatially, no geo-awareness required. \
                     Reader support is still catching up: DuckDB spatial up \
                     to 1.4 — and everything that bundles it, e.g. \
                     duckdb-wasm 1.31 behind maplibre-gl-vector — refuses a \
                     file whose `geo` block declares version 2.0.0 outright \
                     (\"Geoparquet version 2.0.0 is not supported\"). DuckDB \
                     1.5 reads them.");
            });

            h(ui, "Geometry encodings");
            p(ui,
                "The same polygon can be stored three ways, and the choice \
                 drives read performance:");
            egui::Grid::new(("cb_enc", wide as i32)).num_columns(2).striped(true).show(ui, |ui| {
                row(ui, wide, "WKB",
                    "One binary blob per feature. Opens everywhere, but every \
                     read parses features byte by byte. The 2.0 GEOMETRY \
                     logical type is WKB with statistics on top.");
                row(ui, wide, "GeoArrow",
                    "Raw coordinate arrays (nested lists of doubles). No \
                     per-feature parsing: readers decode whole columns in \
                     bulk, and the x/y leaves get ordinary Parquet statistics \
                     for free. Needs a single geometry family per column. \
                     The fastest to display.");
                row(ui, wide, "Native GEOMETRY (2.0)",
                    "WKB storage + built-in spatial statistics + CRS in the \
                     type. The interoperability choice: pruning works in any \
                     2.0-aware reader without extra columns. Writers vary on \
                     the CRS half: DuckDB 1.5 emits GeometryType(crs=<null>) \
                     and states the CRS only in the `geo` block, so a reader \
                     that trusts the logical type alone finds none.");
            });

            h(ui, "What makes a file fast");
            p(ui,
                "Parquet readers skip data using metadata. A file is fast \
                 when its layout lets them skip a lot:");
            egui::Grid::new(("cb_fast", wide as i32)).num_columns(2).striped(true).show(ui, |ui| {
                row(ui, wide, "Spatial ordering",
                    "Features sorted along a space-filling curve (Hilbert) \
                     make each row group cover a small area. Without it every \
                     row group spans the whole extent and nothing can be \
                     skipped — the #1 reason files feel slow.");
                row(ui, wide, "Row-group size",
                    "The unit of skipping and of decoding. 50k–150k rows is \
                     the sweet spot: big enough for scan throughput, small \
                     enough to prune.");
                row(ui, wide, "bbox covering column",
                    "Four little floats per feature. Row-group statistics on \
                     them prune groups; reading them selects exact features \
                     for a viewport; small pages make the page index prune \
                     below row-group level. ~32 bytes/feature well spent.");
                row(ui, wide, "Page index",
                    "Per-page min/max in the footer. Lets readers fetch only \
                     the pages that matter — essential over HTTP.");
                row(ui, wide, "Compression",
                    "zstd. Decompression speed is flat across levels, so \
                     distribution files should use a high level (15+): \
                     smaller downloads, same read speed.");
            });

            h(ui, "How this workbench plays it");
            p(ui,
                "Every file you open is graded against exactly these levers — \
                 that is the quality scorecard in the File info panel. Files \
                 that cannot be displayed incrementally pause at the quality \
                 gate instead of silently decimating. And Optimize rewrites \
                 any layer with the whole recipe: Hilbert order, tuned row \
                 groups, covering column, page index, zstd, in your choice of \
                 1.1 WKB, 1.1 GeoArrow or 2.0 native.");
            p(ui,
                "Rules of thumb: GeoArrow for files you display a lot, WKB + \
                 covering for files that travel to unknown readers, 2.0 \
                 native for modern data platforms. When in doubt, let the \
                 Optimize dialog recommend.");

            h(ui, "Writing GeoParquet: this workbench vs DuckDB vs GDAL");
            p(ui,
                "Most GeoParquet in the wild comes out of DuckDB (spatial) \
                 or GDAL's ogr2ogr. Both can produce good files, but the \
                 display-grade layout is opt-in flags and hand-written SQL; \
                 here it is the default:");
            let col_w =
                ((ui.available_width() - 110.0 - 3.0 * spacing_x) / 3.0).clamp(150.0, 280.0);
            // Grid state remembers column widths per id and never shrinks
            // them; salting the id with the width makes resize re-layout.
            egui::Grid::new(("cb_tools", col_w as i32))
                .num_columns(4)
                .striped(true)
                .show(ui, |ui| {
                    let head = |t: &str| {
                        RichText::new(t)
                            .family(egui::FontFamily::Name(theme::MEDIUM.into()))
                            .strong()
                    };
                    ui.label("");
                    ui.label(head("GeoPQ Workbench").color(
                        if ui.visuals().dark_mode { theme::ACCENT } else { theme::ACCENT_DARK },
                    ));
                    ui.label(head("DuckDB (spatial)"));
                    ui.label(head("GDAL / ogr2ogr"));
                    ui.end_row();
                    row4(ui, col_w, "Output format",
                        "1.1 WKB + covering, 1.1 GeoArrow, or 2.0 native \
                         GEOMETRY — pick per file",
                        "1.0 from a plain COPY; 2.0 native via \
                         GEOPARQUET_VERSION 'V2'. No 1.1 covering or \
                         GeoArrow flavors. Reads 2.0 only from 1.5: up to \
                         1.4 a `geo` version of 2.0.0 is rejected",
                        "1.1 by default (3.9+); GeoArrow via \
                         GEOMETRY_ENCODING; 2.0 native via \
                         USE_PARQUET_GEO_TYPES (3.12+)");
                    row4(ui, col_w, "Spatial ordering",
                        "Automatic Hilbert sort, extent computed for you",
                        "Manual: ORDER BY ST_Hilbert(geom, extent), extent \
                         subquery written by hand",
                        "Opt-in: SORT_BY_BBOX=YES (3.9+)");
                    row4(ui, col_w, "bbox covering",
                        "Written and declared; small pages give page-level \
                         pruning",
                        "DIY struct column, never declared — readers won't \
                         discover it. V2 row-group statistics prune without \
                         one",
                        "Written and declared by default (3.9+)");
                    row4(ui, col_w, "CRS",
                        "Preserved across rewrites (PROJJSON / EPSG)",
                        "Preserved since DuckDB 1.5; older versions dropped \
                         it on write. Its 2.0 writer leaves the logical \
                         type's crs null and states the CRS in `geo` only",
                        "Preserved (PROJJSON)");
                    row4(ui, col_w, "Row groups",
                        "Tuned presets (16k–131k rows)",
                        "ROW_GROUP_SIZE option (default ~122k)",
                        "ROW_GROUP_SIZE option (default 65,536)");
                    row4(ui, col_w, "Compression",
                        "zstd level 15 by default (distribution guidance)",
                        "COMPRESSION 'zstd' + COMPRESSION_LEVEL; snappy by \
                         default",
                        "COMPRESSION=ZSTD; snappy by default");
                    row4(ui, col_w, "Partitioning",
                        "Hive directories by fields, or adaptive H3 cells \
                         balanced to a row target",
                        "PARTITION_BY columns (hive); spatial cells need a \
                         precomputed key column",
                        "Not supported by the Parquet driver");
                    row4(ui, col_w, "Derived columns",
                        "H3 cell and admin-boundary attribution built in",
                        "Any SQL expression — DuckDB's home turf",
                        "OGR SQL / SQLite dialect via -sql");
                });
            p(ui, "");
            p(ui,
                "They are complementary, not rivals: shape and join your \
                 data in DuckDB or convert it with ogr2ogr, then run it \
                 through Optimize (or import it here) to get the spatial \
                 layout the exporters leave to you. Open a raw export in \
                 this workbench and the scorecard shows the difference line \
                 by line. (DuckDB 1.5 and GDAL 3.12, July 2026 — both \
                 evolve quickly.)");

            ui.add_space(6.0);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Further reading:").weak());
                ui.hyperlink_to(
                    "GeoParquet specification",
                    "https://github.com/opengeospatial/geoparquet",
                );
                ui.label(RichText::new("·").weak());
                ui.hyperlink_to(
                    "Distributing GeoParquet (best practices)",
                    "https://github.com/opengeospatial/geoparquet/blob/main/format-specs/distributing-geoparquet.md",
                );
                ui.label(RichText::new("·").weak());
                ui.hyperlink_to("GeoArrow", "https://geoarrow.org");
            });
        });
}
