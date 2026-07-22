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

fn row(ui: &mut egui::Ui, name: &str, text: &str) {
    ui.label(RichText::new(name).strong());
    ui.label(text);
    ui.end_row();
}

pub fn window(ctx: &egui::Context, open: &mut bool, area: egui::Rect) {
    egui::Window::new("GeoParquet cookbook")
        .open(open)
        .default_width(560.0)
        .default_height(520.0)
        .vscroll(true)
        .constrain_to(area).show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;

            p(ui,
                "GeoParquet is Apache Parquet with geometry columns and a \
                 little spatial metadata. Everything that makes Parquet fast \
                 for analytics — columnar layout, statistics, compression, \
                 selective reads — applies to spatial data too, but only if \
                 the file is written with care. This page is the map.");

            h(ui, "Versions");
            egui::Grid::new("cb_versions").num_columns(2).striped(true).show(ui, |ui| {
                row(ui, "1.0",
                    "The baseline: WKB geometry + `geo` file metadata (CRS, \
                     geometry types, bbox). Universally readable, no indexing \
                     story.");
                row(ui, "1.1",
                    "Adds the bbox covering column (a per-feature spatial \
                     index) and GeoArrow native encodings. Files can be \
                     pruned and read incrementally.");
                row(ui, "2.0",
                    "Geometry becomes a first-class Parquet type: GEOMETRY / \
                     GEOGRAPHY logical types with the CRS inside and native \
                     geospatial statistics per row group. Any Parquet reader \
                     can prune spatially, no geo-awareness required.");
            });

            h(ui, "Geometry encodings");
            p(ui,
                "The same polygon can be stored three ways, and the choice \
                 drives read performance:");
            egui::Grid::new("cb_enc").num_columns(2).striped(true).show(ui, |ui| {
                row(ui, "WKB",
                    "One binary blob per feature. Opens everywhere, but every \
                     read parses features byte by byte. The 2.0 GEOMETRY \
                     logical type is WKB with statistics on top.");
                row(ui, "GeoArrow",
                    "Raw coordinate arrays (nested lists of doubles). No \
                     per-feature parsing: readers decode whole columns in \
                     bulk, and the x/y leaves get ordinary Parquet statistics \
                     for free. Needs a single geometry family per column. \
                     The fastest to display.");
                row(ui, "Native GEOMETRY (2.0)",
                    "WKB storage + built-in spatial statistics + CRS in the \
                     type. The interoperability choice: pruning works in any \
                     2.0-aware reader without extra columns.");
            });

            h(ui, "What makes a file fast");
            p(ui,
                "Parquet readers skip data using metadata. A file is fast \
                 when its layout lets them skip a lot:");
            egui::Grid::new("cb_fast").num_columns(2).striped(true).show(ui, |ui| {
                row(ui, "Spatial ordering",
                    "Features sorted along a space-filling curve (Hilbert) \
                     make each row group cover a small area. Without it every \
                     row group spans the whole extent and nothing can be \
                     skipped — the #1 reason files feel slow.");
                row(ui, "Row-group size",
                    "The unit of skipping and of decoding. 50k–150k rows is \
                     the sweet spot: big enough for scan throughput, small \
                     enough to prune.");
                row(ui, "bbox covering column",
                    "Four little floats per feature. Row-group statistics on \
                     them prune groups; reading them selects exact features \
                     for a viewport; small pages make the page index prune \
                     below row-group level. ~32 bytes/feature well spent.");
                row(ui, "Page index",
                    "Per-page min/max in the footer. Lets readers fetch only \
                     the pages that matter — essential over HTTP.");
                row(ui, "Compression",
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
