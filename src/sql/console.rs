//! SQL console panel: editor, background execution, results table, and
//! "add as layer" export back through the loader.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use arrow::array::{Array, BinaryArray};
use eframe::egui::{self, RichText};
use egui_extras::{Column, TableBuilder};

use super::engine::{self, QueryOutput, SqlLayer, SqlMsg, MAX_RESULT_ROWS};

/// Above this many checked rows the map highlight is skipped: decoding and
/// meshing every geometry happens on the UI thread.
const MAX_HIGHLIGHT_ROWS: usize = 20_000;
use super::{export, udf};
use crate::data::crs::Crs;
use crate::data::layer::VectorLayer;
use crate::data::loader::decode_wkb;

/// What the app should do after a console interaction.
pub enum ConsoleAction {
    /// Load an exported result file as a new layer.
    LoadLayer(PathBuf),
    /// Replace the map selection with the checked rows' geometries
    /// (data CRS).
    Select {
        crs: Crs,
        geoms: Vec<geo_types::Geometry<f64>>,
    },
    /// Zoom the map to one feature (🔍 button), leaving selection alone.
    Zoom {
        crs: Crs,
        geom: geo_types::Geometry<f64>,
    },
    /// No rows checked anymore: clear the map selection.
    ClearSelection,
}

/// Console mode, TablePlus-style: browse a table with a WHERE bar, or
/// write free-form SQL.
#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Browse,
    Query,
}

pub struct SqlConsole {
    pub open: bool,
    mode: Mode,
    /// Browse mode: selected table, WHERE clause, row limit.
    browse_table: String,
    browse_where: String,
    browse_limit: usize,
    /// Query mode: free-form SQL.
    query: String,
    tx: Sender<SqlMsg>,
    rx: Receiver<SqlMsg>,
    next_id: u64,
    running: Option<u64>,
    result: Option<QueryOutput>,
    /// Cumulative row offset of each result batch (len = batches + 1).
    row_offsets: Vec<usize>,
    /// Rows checked in the results grid (map selection + TSV copy).
    selected_rows: BTreeSet<usize>,
    /// Wall-clock time of the last clipboard copy, for brief feedback.
    copied_at: Option<f64>,
    error: Option<String>,
    show_help: bool,
    export_n: u64,
}

impl SqlConsole {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            open: false,
            mode: Mode::Browse,
            browse_table: String::new(),
            browse_where: String::new(),
            browse_limit: 1000,
            query: String::new(),
            tx,
            rx,
            next_id: 0,
            running: None,
            result: None,
            row_offsets: vec![0],
            selected_rows: BTreeSet::new(),
            copied_at: None,
            error: None,
            show_help: false,
            export_n: 0,
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Open the console with the ST_* reference visible (Help menu).
    pub fn open_with_help(&mut self) {
        self.open = true;
        self.show_help = true;
    }

    /// Drain finished queries. Call every frame before the panel.
    pub fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            if Some(msg.id) != self.running {
                continue; // stale result from a superseded query
            }
            self.running = None;
            self.selected_rows.clear();
            match msg.result {
                Ok(out) => {
                    self.row_offsets = std::iter::once(0)
                        .chain(out.batches.iter().scan(0usize, |acc, b| {
                            *acc += b.num_rows();
                            Some(*acc)
                        }))
                        .collect();
                    self.result = Some(out);
                    self.error = None;
                }
                Err(e) => {
                    self.error = Some(e);
                    self.result = None;
                }
            }
        }
    }

    /// Render the console. Returns an action for the app to apply (load an
    /// exported result as a layer, zoom to a clicked feature).
    pub fn panel_ui(
        &mut self,
        ui: &mut egui::Ui,
        layers: &[VectorLayer],
    ) -> Option<ConsoleAction> {
        let mut action = None;
        let tables = table_names(layers);

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("SQL");
            ui.selectable_value(&mut self.mode, Mode::Browse, "Browse");
            ui.selectable_value(&mut self.mode, Mode::Query, "Query");
            if self.running.is_some() {
                ui.spinner();
                ui.label(RichText::new("running…").weak());
            }
            ui.toggle_value(&mut self.show_help, "ST_* help");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("✕").clicked() {
                    self.open = false;
                }
            });
        });

        match self.mode {
            Mode::Browse => self.browse_bar(ui, layers, &tables),
            Mode::Query => self.query_editor(ui, layers, &tables),
        }

        if self.show_help {
            egui::ScrollArea::vertical()
                .id_salt("sql_help")
                .max_height(90.0)
                .show(ui, |ui| {
                    egui::Grid::new("sql_help_grid").num_columns(2).striped(true).show(
                        ui,
                        |ui| {
                            for (sig, desc) in udf::catalog() {
                                ui.label(RichText::new(*sig).monospace().small());
                                ui.label(RichText::new(*desc).weak().small());
                                ui.end_row();
                            }
                        },
                    );
                });
            ui.separator();
        }

        if let Some(err) = &self.error {
            ui.label(RichText::new(err).color(ui.visuals().error_fg_color));
        }

        let mut clear = false;
        let mut export = false;
        let mut export_sel = false;
        let mut copy_tsv = false;
        if let Some(out) = &self.result {
            let mut status = format!("{} rows · {} ms", out.total_rows, out.elapsed_ms);
            if out.truncated {
                status.push_str(&format!(" · truncated at {MAX_RESULT_ROWS}"));
            }
            let geom_name = out
                .geom
                .as_ref()
                .map(|(i, _)| out.schema.field(*i).name().clone());
            let has_geom = geom_name.is_some();
            let n_sel = self.selected_rows.len();
            let copied_recently = self
                .copied_at
                .is_some_and(|t| ui.input(|i| i.time) - t < 2.0);
            let total = out.total_rows;
            ui.horizontal(|ui| {
                ui.label(RichText::new(status).weak().small());
                if let Some(gname) = geom_name {
                    export = ui
                        .button(format!("🗺 Result as layer ({total} rows)"))
                        .on_hover_text(format!(
                            "Write the whole result grid — exactly the {total} rows \
                             shown here, after any LIMIT/truncation — to a temporary \
                             GeoParquet ({gname} as geometry) and load it on the map"
                        ))
                        .clicked();
                }
                if n_sel > 0 {
                    ui.label(RichText::new(format!("{n_sel} checked")).small());
                    if n_sel > MAX_HIGHLIGHT_ROWS {
                        ui.label(
                            RichText::new("(too many to highlight on map)").weak().small(),
                        );
                    }
                    copy_tsv = ui
                        .button("📋 Copy TSV")
                        .on_hover_text(
                            "Copy the checked rows (with headers, geometry as WKT) \
                             for pasting into a spreadsheet",
                        )
                        .clicked();
                    if has_geom {
                        export_sel = ui
                            .button(format!("🗺 Checked as layer ({n_sel})"))
                            .on_hover_text(
                                "Write only the checked rows to a temporary \
                                 GeoParquet and load them on the map",
                            )
                            .clicked();
                    }
                }
                clear = ui.button("Clear").clicked();
                if copied_recently {
                    ui.label(RichText::new("✔ copied").weak().small());
                }
            });
        }
        if export || export_sel {
            let rows = export_sel.then(|| self.selected_rows.clone());
            match self.export_result(rows.as_ref()) {
                Ok(path) => action = Some(ConsoleAction::LoadLayer(path)),
                Err(e) => self.error = Some(e),
            }
        }
        if copy_tsv {
            if let Some(out) = &self.result {
                ui.ctx()
                    .copy_text(selection_tsv(out, &self.row_offsets, &self.selected_rows));
                self.copied_at = Some(ui.input(|i| i.time));
            }
        }
        if clear {
            self.result = None;
            self.row_offsets = vec![0];
            if !self.selected_rows.is_empty() {
                action = Some(ConsoleAction::ClearSelection);
            }
            self.selected_rows.clear();
        }

        if let Some(out) = &self.result {
            ui.separator();
            let ev = results_table(ui, out, &self.row_offsets, &mut self.selected_rows);
            if ev.copied {
                self.copied_at = Some(ui.input(|i| i.time));
            }
            if ev.toggled {
                action = Some(self.selection_action());
            } else if let Some(row) = ev.zoom_clicked {
                if let (Some((_, crs)), Some(geom)) = (
                    out.geom.clone(),
                    geometry_at(out, &self.row_offsets, row),
                ) {
                    action = Some(ConsoleAction::Zoom { crs, geom });
                }
            }
        }
        action
    }

    /// Rebuild the map-selection action after a checkbox toggle.
    fn selection_action(&self) -> ConsoleAction {
        let Some(out) = &self.result else {
            return ConsoleAction::ClearSelection;
        };
        let Some((_, crs)) = out.geom.clone() else {
            return ConsoleAction::ClearSelection;
        };
        // Select-all on a huge result must not freeze the UI decoding
        // every geometry; the selection still works for TSV/export.
        if self.selected_rows.len() > MAX_HIGHLIGHT_ROWS {
            return ConsoleAction::ClearSelection;
        }
        let geoms: Vec<geo_types::Geometry<f64>> = self
            .selected_rows
            .iter()
            .filter_map(|&r| geometry_at(out, &self.row_offsets, r))
            .collect();
        if geoms.is_empty() {
            return ConsoleAction::ClearSelection;
        }
        ConsoleAction::Select { crs, geoms }
    }

    /// TablePlus-style bar: pick a table, type a WHERE clause, Enter (or ▶)
    /// applies it.
    fn browse_bar(&mut self, ui: &mut egui::Ui, layers: &[VectorLayer], tables: &[String]) {
        if !tables.contains(&self.browse_table) {
            self.browse_table = tables.first().cloned().unwrap_or_default();
        }
        let mut run = false;
        ui.horizontal(|ui| {
            let mut picked = self.browse_table.clone();
            egui::ComboBox::from_id_salt("sql_browse_table")
                .selected_text(if picked.is_empty() {
                    "no layers".to_string()
                } else {
                    picked.clone()
                })
                .show_ui(ui, |ui| {
                    for (l, t) in layers.iter().zip(tables) {
                        ui.selectable_value(&mut picked, t.clone(), t)
                            .on_hover_text(&l.name);
                    }
                });
            if picked != self.browse_table {
                self.browse_table = picked;
                run = true; // load on select, like TablePlus
            }

            ui.label(RichText::new("WHERE").weak().monospace());
            let where_id = egui::Id::new("sql_browse_where");
            ctrl_z_alias(ui, where_id);
            let w = ui.add(
                egui::TextEdit::singleline(&mut self.browse_where)
                    .id(where_id)
                    .desired_width(ui.available_width() - 170.0)
                    .hint_text("st_area(geometry) > 1000 — Enter applies"),
            );
            if w.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                run = true;
            }

            ui.add(
                egui::DragValue::new(&mut self.browse_limit)
                    .range(1..=MAX_RESULT_ROWS)
                    .prefix("limit "),
            );
            if ui
                .add_enabled(
                    self.running.is_none() && !self.browse_table.is_empty(),
                    egui::Button::new("▶"),
                )
                .on_hover_text("Apply (Enter in the WHERE box)")
                .clicked()
            {
                run = true;
            }
        });
        if run && self.running.is_none() && !self.browse_table.is_empty() {
            let mut sql = format!("select * from {}", self.browse_table);
            let w = self.browse_where.trim();
            if !w.is_empty() {
                sql.push_str(&format!(" where {w}"));
            }
            sql.push_str(&format!(" limit {}", self.browse_limit));
            self.run(ui.ctx().clone(), layers, sql);
        }
    }

    /// Free-form SQL editor with table-name chips and Ctrl+Enter.
    fn query_editor(&mut self, ui: &mut egui::Ui, layers: &[VectorLayer], tables: &[String]) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("tables:").weak().small());
            if layers.is_empty() {
                ui.label(RichText::new("none (load a layer first)").weak().small());
            }
            for (l, t) in layers.iter().zip(tables) {
                if ui
                    .add(egui::Button::new(RichText::new(t).monospace().small()).frame(false))
                    .on_hover_text(format!("{} — click to insert", l.name))
                    .clicked()
                {
                    if self.query.trim().is_empty() {
                        self.query = format!("select * from {t} limit 100");
                    } else {
                        self.query.push_str(t);
                    }
                }
            }
        });

        let query_id = egui::Id::new("sql_query_edit");
        ctrl_z_alias(ui, query_id);
        ui.add(
            egui::TextEdit::multiline(&mut self.query)
                .id(query_id)
                .code_editor()
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("select * from <table> where st_area(geometry) > 1000"),
        );

        let run_clicked = ui
            .add_enabled(
                self.running.is_none() && !self.query.trim().is_empty(),
                egui::Button::new("▶ Run"),
            )
            .on_hover_text("Ctrl+Enter")
            .clicked();
        let hotkey = ui
            .input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));
        if (run_clicked || hotkey) && self.running.is_none() && !self.query.trim().is_empty() {
            let sql = self.query.clone();
            self.run(ui.ctx().clone(), layers, sql);
        }
    }

    fn run(&mut self, egui_ctx: egui::Context, layers: &[VectorLayer], sql: String) {
        let sql_layers: Vec<SqlLayer> = layers
            .iter()
            .zip(table_names(layers))
            .map(|(l, table)| SqlLayer {
                table,
                store: Arc::clone(&l.store),
                crs: l.crs.clone(),
                // Only index-aligned boxes can prune: partial viewport
                // loads compute boxes for a subset of groups.
                rg_bboxes: l
                    .rg_bboxes
                    .as_ref()
                    .filter(|r| {
                        r.boxes.len() == l.store.rg_starts().len().saturating_sub(1)
                    })
                    .map(|r| Arc::new(r.boxes.clone())),
            })
            .collect();
        let id = self.next_id;
        self.next_id += 1;
        self.running = Some(id);
        self.error = None;
        engine::spawn_query(id, sql, sql_layers, self.tx.clone(), move || {
            egui_ctx.request_repaint();
        });
    }

    /// Export the result (or only `rows`, when given) to a temp GeoParquet.
    fn export_result(&mut self, rows: Option<&BTreeSet<usize>>) -> Result<PathBuf, String> {
        let out = self.result.as_ref().ok_or("no result")?;
        let (gcol, crs) = out.geom.clone().ok_or("no geometry column in result")?;
        let filtered;
        let batches: &[arrow::record_batch::RecordBatch] = match rows {
            Some(rows) => {
                filtered = filter_rows(out, &self.row_offsets, rows)?;
                &filtered
            }
            None => &out.batches,
        };
        self.export_n += 1;
        let path = std::env::temp_dir().join(format!(
            "geopq_query_{}_{}.parquet",
            std::process::id(),
            self.export_n
        ));
        export::write_result(&path, &out.schema, batches, gcol, &crs)?;
        Ok(path)
    }
}

/// Route Ctrl+Z / Ctrl+Shift+Z to the focused text field's native undoer
/// (egui binds undo to the platform command key — Cmd on macOS — so a
/// PC-habit Ctrl+Z does nothing there). Consumes the Ctrl event and
/// re-injects it with the command modifier before the widget reads input.
/// On Windows/Linux Ctrl IS the command key, so consume + re-inject is a
/// net no-op.
fn ctrl_z_alias(ui: &mut egui::Ui, field: egui::Id) {
    if !ui.ctx().memory(|m| m.has_focus(field)) {
        return;
    }
    let redo = ui.input_mut(|i| {
        i.consume_key(egui::Modifiers::CTRL | egui::Modifiers::SHIFT, egui::Key::Z)
    });
    let undo = ui.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Z));
    let inject = |modifiers: egui::Modifiers| {
        ui.input_mut(|i| {
            i.events.push(egui::Event::Key {
                key: egui::Key::Z,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            });
        });
    };
    if redo {
        inject(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT);
    }
    if undo {
        inject(egui::Modifiers::COMMAND);
    }
}

/// SQL identifier per layer; same-named layers get `_2`, `_3`, ...
/// suffixes so both stay queryable.
fn table_names(layers: &[VectorLayer]) -> Vec<String> {
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    layers
        .iter()
        .map(|l| {
            let base = engine::table_name(&l.name);
            let n = seen.entry(base.clone()).or_insert(0);
            *n += 1;
            if *n > 1 {
                format!("{base}_{n}")
            } else {
                base
            }
        })
        .collect()
}

/// Locate (batch index, row within batch) for a global result row.
fn locate(row_offsets: &[usize], r: usize) -> (usize, usize) {
    let b = match row_offsets.binary_search(&r) {
        Ok(i) if i + 1 < row_offsets.len() => i,
        Ok(i) => i - 1,
        Err(i) => i - 1,
    };
    (b, r - row_offsets[b])
}

/// Decode the geometry of one result row.
fn geometry_at(
    out: &QueryOutput,
    row_offsets: &[usize],
    r: usize,
) -> Option<geo_types::Geometry<f64>> {
    let (gcol, _) = out.geom.as_ref()?;
    let (b, local) = locate(row_offsets, r);
    let arr = out.batches[b]
        .column(*gcol)
        .as_any()
        .downcast_ref::<BinaryArray>()?;
    if arr.is_null(local) {
        return None;
    }
    decode_wkb(arr.value(local))
}

/// The full (untruncated) value of one result cell; geometry as WKT.
fn cell_value(out: &QueryOutput, row_offsets: &[usize], r: usize, c: usize) -> String {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    if out.geom.as_ref().is_some_and(|(gcol, _)| *gcol == c) {
        return match geometry_at(out, row_offsets, r) {
            Some(g) => {
                use wkt::ToWkt;
                g.wkt_string()
            }
            None => String::new(),
        };
    }
    let (b, local) = locate(row_offsets, r);
    let opts = FormatOptions::default().with_display_error(true);
    ArrayFormatter::try_new(out.batches[b].column(c).as_ref(), &opts)
        .map(|f| f.value(local).to_string())
        .unwrap_or_default()
}

/// Only the given rows of the result, as new record batches.
fn filter_rows(
    out: &QueryOutput,
    row_offsets: &[usize],
    rows: &BTreeSet<usize>,
) -> Result<Vec<arrow::record_batch::RecordBatch>, String> {
    use arrow::array::UInt32Array;
    let mut per_batch: Vec<Vec<u32>> = vec![Vec::new(); out.batches.len()];
    for &r in rows {
        let (b, local) = locate(row_offsets, r);
        per_batch[b].push(local as u32);
    }
    per_batch
        .iter()
        .enumerate()
        .filter(|(_, idx)| !idx.is_empty())
        .map(|(b, idx)| {
            let indices = UInt32Array::from(idx.clone());
            let cols = out.batches[b]
                .columns()
                .iter()
                .map(|c| arrow::compute::take(c.as_ref(), &indices, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("row filter: {e}"))?;
            arrow::record_batch::RecordBatch::try_new(out.schema.clone(), cols)
                .map_err(|e| format!("row filter: {e}"))
        })
        .collect()
}

/// Checked rows as tab-separated text with a header line, for spreadsheets.
fn selection_tsv(out: &QueryOutput, row_offsets: &[usize], rows: &BTreeSet<usize>) -> String {
    let n_cols = out.schema.fields().len();
    let mut s = out
        .schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect::<Vec<_>>()
        .join("\t");
    for &r in rows {
        s.push('\n');
        for c in 0..n_cols {
            if c > 0 {
                s.push('\t');
            }
            let v = cell_value(out, row_offsets, r, c);
            // Keep the grid rectangular for the paste target.
            s.push_str(&v.replace(['\t', '\n'], " "));
        }
    }
    s
}

/// Outcome of one results-grid frame.
#[derive(Default)]
struct TableEvent {
    /// A row checkbox changed.
    toggled: bool,
    /// The 🔍 button of this row was clicked.
    zoom_clicked: Option<usize>,
    /// A cell value was copied to the clipboard.
    copied: bool,
}

/// Virtualized result grid. A checkbox per row adds the feature to the map
/// selection, 🔍 zooms to it, clicking any cell copies its full value.
fn results_table(
    ui: &mut egui::Ui,
    out: &QueryOutput,
    row_offsets: &[usize],
    selected: &mut BTreeSet<usize>,
) -> TableEvent {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    let n_cols = out.schema.fields().len();
    if n_cols == 0 {
        return TableEvent::default();
    }
    let geom_col = out.geom.as_ref().map(|(i, _)| *i);
    let has_geom = geom_col.is_some();
    let opts = FormatOptions::default().with_display_error(true);
    let mut ev = TableEvent::default();

    // No vertical gap between rows: dead space for clicks, and selected
    // rows should read as solid bands.
    ui.spacing_mut().item_spacing.y = 0.0;
    let mut builder = TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .sense(egui::Sense::click())
        .column(Column::exact(22.0));
    if has_geom {
        builder = builder.column(Column::exact(22.0));
    }
    builder
        .columns(Column::auto().at_least(60.0).clip(true), n_cols)
        .header(20.0, |mut header| {
            header.col(|ui| {
                // Select all / none; shows the mixed state when partial.
                let n_sel = selected.len();
                let mut all = n_sel == out.total_rows && out.total_rows > 0;
                if ui
                    .add(
                        egui::Checkbox::without_text(&mut all)
                            .indeterminate(n_sel > 0 && n_sel < out.total_rows),
                    )
                    .on_hover_text("Check / uncheck all rows")
                    .changed()
                {
                    selected.clear();
                    if all {
                        selected.extend(0..out.total_rows);
                    }
                    ev.toggled = true;
                }
            });
            if has_geom {
                header.col(|_| {});
            }
            for f in out.schema.fields() {
                header.col(|ui| {
                    ui.label(RichText::new(f.name()).strong().small());
                });
            }
        })
        .body(|body| {
            body.rows(22.0, out.total_rows, |mut row| {
                let r = row.index();
                row.set_selected(selected.contains(&r));
                row.col(|ui| {
                    let mut checked = selected.contains(&r);
                    if ui
                        .checkbox(&mut checked, "")
                        .on_hover_text("Add to the map selection")
                        .changed()
                    {
                        if checked {
                            selected.insert(r);
                        } else {
                            selected.remove(&r);
                        }
                        ev.toggled = true;
                    }
                });
                if has_geom {
                    row.col(|ui| {
                        if ui
                            .add(egui::Button::new(RichText::new("🔍").small()).frame(false))
                            .on_hover_text("Zoom to this feature")
                            .clicked()
                        {
                            ev.zoom_clicked = Some(r);
                        }
                    });
                }
                let (b, local) = locate(row_offsets, r);
                let batch = &out.batches[b];
                for c in 0..n_cols {
                    row.col(|ui| {
                        let text = if Some(c) == geom_col {
                            geom_cell(batch.column(c).as_ref(), local)
                        } else {
                            ArrayFormatter::try_new(batch.column(c).as_ref(), &opts)
                                .map(|f| f.value(local).to_string())
                                .unwrap_or_else(|_| "<?>".into())
                        };
                        let resp = ui.add(
                            egui::Label::new(RichText::new(text).small())
                                .sense(egui::Sense::click()),
                        );
                        if resp.clicked() {
                            ui.ctx().copy_text(cell_value(out, row_offsets, r, c));
                            ev.copied = true;
                        }
                    });
                }
            });
        });
    ev
}

fn geom_cell(col: &dyn Array, i: usize) -> String {
    let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() else {
        return "<?>".into();
    };
    if arr.is_null(i) {
        return "∅".into();
    }
    let buf = arr.value(i);
    match decode_wkb(buf) {
        Some(g) => {
            use geo::CoordsIter;
            let t = match &g {
                geo_types::Geometry::Point(_) => "Point",
                geo_types::Geometry::Line(_) => "Line",
                geo_types::Geometry::LineString(_) => "LineString",
                geo_types::Geometry::Polygon(_) => "Polygon",
                geo_types::Geometry::MultiPoint(_) => "MultiPoint",
                geo_types::Geometry::MultiLineString(_) => "MultiLineString",
                geo_types::Geometry::MultiPolygon(_) => "MultiPolygon",
                geo_types::Geometry::GeometryCollection(_) => "GeometryCollection",
                geo_types::Geometry::Rect(_) => "Rect",
                geo_types::Geometry::Triangle(_) => "Triangle",
            };
            format!("{t} ({} pts)", g.coords_count())
        }
        None => format!("WKB {} B", buf.len()),
    }
}
