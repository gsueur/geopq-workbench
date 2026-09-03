//! SQL console panel: editors with autocomplete, background execution,
//! paginated/sortable results grid, and "add as layer" export back through
//! the loader.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use arrow::array::{Array, BinaryArray};
use arrow::record_batch::RecordBatch;
use eframe::egui::{self, RichText};
use egui_extras::{Column, TableBuilder};

use super::engine::{self, QueryOutput, SqlDone, SqlLayer, SqlMsg, MAX_RESULT_ROWS};
use super::{export, udf};
use crate::data::attrs::AttrTable;
use crate::data::crs::{Crs, DisplayCrs};
use crate::data::layer::VectorLayer;
use crate::data::loader::{decode_wkb, viewport_to_data_bbox};

/// Above this many checked rows the map highlight is skipped: decoding and
/// meshing every geometry happens on the UI thread.
const MAX_HIGHLIGHT_ROWS: usize = 20_000;

const PAGE_SIZES: [usize; 3] = [100, 1_000, 10_000];

const SQL_KEYWORDS: &[&str] = &[
    "select", "from", "where", "and", "or", "not", "order by", "group by", "having", "limit",
    "offset", "as", "join", "left join", "inner join", "on", "distinct", "count", "sum", "avg",
    "min", "max", "between", "like", "in", "is null", "is not null", "case", "when", "then",
    "else", "end", "cast", "asc", "desc", "union all",
];

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
    /// Zoom the map to one feature and highlight it together with the
    /// current checked selection.
    Zoom {
        crs: Crs,
        zoom: geo_types::Geometry<f64>,
        highlight: Vec<geo_types::Geometry<f64>>,
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

/// Autocomplete popup state for one text field.
#[derive(Default)]
pub(crate) struct AcState {
    open: bool,
    items: Vec<String>,
    /// Byte range of the token being completed.
    token: std::ops::Range<usize>,
    selected: usize,
    /// A token the popup must stay shut for. `open` is recomputed from
    /// the cursor every frame, so without this a dismissal lasts until
    /// the next frame and no further.
    dismissed: Option<Dismissed>,
}

/// A token the completion popup was dismissed for.
pub(crate) struct Dismissed {
    /// Start byte of the token, so the same word typed elsewhere in the
    /// query still completes.
    start: usize,
    /// The token, lowercased.
    word: String,
    /// Whether it also covers whatever is typed after it. Escape means
    /// "leave this word alone" and stays shut as the word grows;
    /// accepting a candidate only suppresses the reopen it would cause
    /// by itself, so typing on re-triggers the popup. There is no
    /// keystroke to summon it back, which is why accepting does not
    /// shut it for the rest of the word.
    sticky: bool,
}

/// Should the popup stay shut for the token now under the cursor?
fn suppressed(d: Option<&Dismissed>, start: usize, prefix: &str) -> bool {
    d.is_some_and(|d| {
        d.start == start
            && if d.sticky {
                prefix.starts_with(&d.word)
            } else {
                prefix == d.word
            }
    })
}

pub struct SqlConsole {
    pub open: bool,
    mode: Mode,
    /// Browse mode: selected table, WHERE clause, viewport filter.
    browse_table: String,
    browse_where: String,
    viewport_only: bool,
    /// Query mode: free-form SQL.
    query: String,
    /// The SQL actually executed last and the layer set it ran against
    /// (for full-result re-run exports).
    last_sql: String,
    last_layers: Vec<SqlLayer>,
    /// Attribute tables the last run saw, kept for the same reason
    /// `last_layers` is: a streaming re-export must query what the result
    /// on screen came from.
    last_tables: Vec<engine::SqlTable>,
    tx: Sender<SqlMsg>,
    rx: Receiver<SqlMsg>,
    next_id: u64,
    running: Option<u64>,
    /// Stop flag shared with the running query/export thread. One per
    /// job: a stopped job's flag must not stop the next one.
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Table names and the completion dictionary, rebuilt only when the
    /// layer/table set changes.
    sources: Option<Arc<Sources>>,
    sources_key: Option<u64>,
    exporting: Option<u64>,
    result: Option<QueryOutput>,
    /// Rows checked in the results grid — underlying (unsorted) indices.
    selected_rows: BTreeSet<usize>,
    /// Result view: current page and rows per page.
    page: usize,
    page_size: usize,
    /// Sort state: (column, ascending); `perm[view] = underlying`.
    sort: Option<(usize, bool)>,
    perm: Option<Vec<u32>>,
    /// Wall-clock time of the last clipboard copy, for brief feedback.
    copied_at: Option<f64>,
    error: Option<String>,
    show_help: bool,
    export_n: u64,
    ac_query: AcState,
    ac_where: AcState,
    /// Executed free-form queries, most recent first. Persisted in the
    /// settings sidecar; navigated with Ctrl+Up / Ctrl+Down.
    history: Vec<String>,
    /// Current position while browsing history (None = editing draft).
    hist_pos: Option<usize>,
    /// The in-progress query stashed while browsing history.
    hist_draft: String,
}

const HISTORY_CAP: usize = 50;

fn load_history() -> Vec<String> {
    crate::data::settings::read()
        .get("sql_history")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default()
}

fn save_history(history: &[String]) {
    if let Err(e) =
        crate::data::settings::update(|root| root["sql_history"] = serde_json::json!(history))
    {
        log::warn!("could not save the SQL history: {e}");
    }
}

impl SqlConsole {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            open: false,
            mode: Mode::Browse,
            browse_table: String::new(),
            browse_where: String::new(),
            viewport_only: false,
            query: String::new(),
            last_sql: String::new(),
            last_layers: Vec::new(),
            last_tables: Vec::new(),
            tx,
            rx,
            next_id: 0,
            running: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            sources: None,
            sources_key: None,
            exporting: None,
            result: None,
            selected_rows: BTreeSet::new(),
            page: 0,
            page_size: 1_000,
            sort: None,
            perm: None,
            copied_at: None,
            error: None,
            show_help: false,
            export_n: 0,
            ac_query: AcState::default(),
            ac_where: AcState::default(),
            history: load_history(),
            hist_pos: None,
            hist_draft: String::new(),
        }
    }

    /// Record an executed query: dedupe, most recent first, capped.
    fn push_history(&mut self, sql: &str) {
        let sql = sql.trim();
        if sql.is_empty() {
            return;
        }
        self.history.retain(|h| h != sql);
        self.history.insert(0, sql.to_string());
        self.history.truncate(HISTORY_CAP);
        self.hist_pos = None;
        save_history(&self.history);
    }

    pub fn is_running(&self) -> bool {
        self.running.is_some() || self.exporting.is_some()
    }

    /// Open the console with the ST_* reference visible (Help menu).
    pub fn open_with_help(&mut self) {
        self.open = true;
        self.show_help = true;
    }

    /// Drain finished jobs. Call every frame; may return an action (e.g. a
    /// finished full-result export to load).
    pub fn poll(&mut self) -> Option<ConsoleAction> {
        let mut action = None;
        while let Ok(msg) = self.rx.try_recv() {
            if Some(msg.id) == self.running {
                self.running = None;
                self.selected_rows.clear();
                self.page = 0;
                self.sort = None;
                self.perm = None;
                match msg.result {
                    Ok(SqlDone::Query(out)) => {
                        self.result = Some(out);
                        self.error = None;
                    }
                    Ok(SqlDone::Export { .. }) => {}
                    // The user pressing Stop is not a failure to report.
                    Err(e) if e == engine::CANCELLED => self.error = None,
                    Err(e) => {
                        self.error = Some(e);
                        self.result = None;
                    }
                }
            } else if Some(msg.id) == self.exporting {
                self.exporting = None;
                match msg.result {
                    Ok(SqlDone::Export { path, .. }) => {
                        action = Some(ConsoleAction::LoadLayer(path));
                    }
                    Ok(SqlDone::Query(_)) => {}
                    Err(e) if e == engine::CANCELLED => self.error = None,
                    Err(e) => self.error = Some(e),
                }
            }
        }
        action
    }

    /// Render the console. Returns an action for the app to apply.
    pub fn panel_ui(
        &mut self,
        ui: &mut egui::Ui,
        layers: &[VectorLayer],
        attrs: &[AttrTable],
        view_world: [f64; 4],
        display: &DisplayCrs,
    ) -> Option<ConsoleAction> {
        let mut action = None;
        let sources = self.sources(layers, attrs);
        let (tables, attr_tables) = (&sources.tables, &sources.attr_tables);
        let dict = &sources.dict;

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("SQL");
            ui.selectable_value(&mut self.mode, Mode::Browse, "Browse");
            ui.selectable_value(&mut self.mode, Mode::Query, "Query");
            if self.running.is_some() {
                ui.spinner();
                ui.label(RichText::new("running…").weak());
            }
            if self.exporting.is_some() {
                ui.spinner();
                ui.label(RichText::new("exporting full result…").weak());
            }
            if (self.running.is_some() || self.exporting.is_some())
                && ui
                    .button("Stop")
                    .on_hover_text("Stop the running query at the next batch")
                    .clicked()
            {
                self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            ui.toggle_value(&mut self.show_help, "ST_* help");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // The phosphor glyph, like every other close button: the
                // bare ✕ codepoint is not in the font and draws as a box.
                if ui
                    .button(egui_phosphor::regular::X)
                    .on_hover_text("Close the SQL console")
                    .clicked()
                {
                    self.open = false;
                }
            });
        });

        match self.mode {
            Mode::Browse => {
                // Browsing works on anything registered, geometry or not.
                // The two lists stay apart: the viewport filter needs the
                // layer they index into, and merging them made the two
                // index spaces silently different.
                self.browse_bar(
                    ui,
                    layers,
                    attrs,
                    tables,
                    attr_tables,
                    dict,
                    view_world,
                    display,
                )
            }
            Mode::Query => {
                self.query_editor(ui, layers, attrs, tables, attr_tables, dict)
            }
        }

        if self.show_help {
            egui::ScrollArea::vertical()
                .id_salt("sql_help")
                .max_height(90.0)
                .show(ui, |ui| {
                    egui::Grid::new("sql_help_grid").num_columns(2).striped(true).show(
                        ui,
                        |ui| {
                            for (sig, desc) in
                                udf::catalog().iter().chain(super::agg::catalog())
                            {
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
        let mut export_all = false;
        let mut export_sel = false;
        let mut copy_tsv = false;
        if let Some(out) = &self.result {
            let total = out.total_rows;
            let n_sel = self.selected_rows.len();
            let has_geom = out.geom.is_some();
            let copied_recently = self
                .copied_at
                .is_some_and(|t| ui.input(|i| i.time) - t < 2.0);
            let mut status = format!("{total} rows · {} ms", out.elapsed_ms);
            if out.truncated {
                status.push_str(&format!(" · showing first {MAX_RESULT_ROWS}"));
            }

            let mut new_page = self.page;
            let mut new_size = self.page_size;
            ui.horizontal(|ui| {
                ui.label(RichText::new(status).weak().small());

                // Pagination.
                let n_pages = total.div_ceil(self.page_size).max(1);
                if n_pages > 1 {
                    ui.separator();
                    if ui.add_enabled(self.page > 0, egui::Button::new("⏮")).clicked() {
                        new_page = 0;
                    }
                    if ui.add_enabled(self.page > 0, egui::Button::new("⏴")).clicked() {
                        new_page = self.page - 1;
                    }
                    let lo = self.page * self.page_size + 1;
                    let hi = ((self.page + 1) * self.page_size).min(total);
                    ui.label(RichText::new(format!("{lo}–{hi} of {total}")).small());
                    if ui
                        .add_enabled(self.page + 1 < n_pages, egui::Button::new("⏵"))
                        .clicked()
                    {
                        new_page = self.page + 1;
                    }
                    if ui
                        .add_enabled(self.page + 1 < n_pages, egui::Button::new("⏭"))
                        .clicked()
                    {
                        new_page = n_pages - 1;
                    }
                }
                egui::ComboBox::from_id_salt("sql_page_size")
                    .selected_text(format!("{}/page", self.page_size))
                    .width(80.0)
                    .show_ui(ui, |ui| {
                        for s in PAGE_SIZES {
                            ui.selectable_value(&mut new_size, s, format!("{s}/page"));
                        }
                    });

                ui.separator();
                if has_geom {
                    let (label, hover) = if out.truncated {
                        (
                            "🗺 Result as layer (all rows)".to_string(),
                            "Re-runs the query and streams EVERY matching row to a \
                             temporary GeoParquet (not just the rows shown), then \
                             loads it on the map"
                                .to_string(),
                        )
                    } else {
                        (
                            format!("🗺 Result as layer ({total} rows)"),
                            format!(
                                "Write the whole result ({total} rows) to a temporary \
                                 GeoParquet and load it on the map"
                            ),
                        )
                    };
                    export_all = ui
                        .add_enabled(self.exporting.is_none(), egui::Button::new(label))
                        .on_hover_text(hover)
                        .clicked();
                } else if let Some(w) = &out.geom_warning {
                    // There *is* a WKB column; we just cannot say what
                    // CRS it is in. Saying so beats an absent button.
                    ui.label(
                        RichText::new("⚠ not loadable as a layer")
                            .weak()
                            .small(),
                    )
                    .on_hover_text(w);
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
            if new_size != self.page_size {
                // Keep the first visible row stable across page-size change.
                let first = self.page * self.page_size;
                self.page_size = new_size;
                self.page = first / new_size;
            } else {
                self.page = new_page;
            }
        }

        if export_all {
            action = self.export_full().or(action);
        }
        if export_sel {
            let rows = self.selected_rows.clone();
            match self.export_materialized(Some(&rows)) {
                Ok(path) => action = Some(ConsoleAction::LoadLayer(path)),
                Err(e) => self.error = Some(e),
            }
        }
        if copy_tsv {
            if let Some(out) = &self.result {
                ui.ctx().copy_text(selection_tsv(out, &self.selected_rows));
                self.copied_at = Some(ui.input(|i| i.time));
            }
        }
        if clear {
            self.result = None;
            self.perm = None;
            self.sort = None;
            self.page = 0;
            if !self.selected_rows.is_empty() {
                action = Some(ConsoleAction::ClearSelection);
            }
            self.selected_rows.clear();
        }

        if let Some(out) = &self.result {
            ui.separator();
            let ev = results_table(
                ui,
                out,
                &mut self.selected_rows,
                self.page,
                self.page_size,
                self.sort,
                self.perm.as_deref(),
            );
            if ev.copied {
                self.copied_at = Some(ui.input(|i| i.time));
            }
            if let Some(col) = ev.sort_clicked {
                self.cycle_sort(col);
            }
            if ev.toggled {
                action = Some(self.selection_action());
            } else if let Some(row) = ev.zoom_clicked {
                if let Some(zoom) = self.zoom_action(row) {
                    action = Some(zoom);
                }
            }
        }
        action
    }

    /// ASC → DESC → NONE cycle on a header click.
    fn cycle_sort(&mut self, col: usize) {
        self.sort = match self.sort {
            Some((c, true)) if c == col => Some((col, false)),
            Some((c, false)) if c == col => None,
            _ => Some((col, true)),
        };
        self.perm = match self.sort {
            None => None,
            Some((c, asc)) => {
                let out = self.result.as_ref().unwrap();
                let opts = arrow::compute::SortOptions {
                    descending: !asc,
                    nulls_first: false,
                };
                match arrow::compute::sort_to_indices(out.batch.column(c), Some(opts), None) {
                    Ok(idx) => Some(idx.values().to_vec()),
                    Err(e) => {
                        self.error = Some(format!("sort: {e}"));
                        self.sort = None;
                        None
                    }
                }
            }
        };
        self.page = 0;
    }

    /// TablePlus-style bar: pick a table, type a WHERE clause, Enter (or ▶)
    /// applies it; optionally restrict to the current viewport.
    #[allow(clippy::too_many_arguments)]
    fn browse_bar(
        &mut self,
        ui: &mut egui::Ui,
        layers: &[VectorLayer],
        attrs: &[AttrTable],
        tables: &[String],
        attr_tables: &[String],
        dict: &CompletionDict,
        view_world: [f64; 4],
        display: &DisplayCrs,
    ) {
        // Everything registered can be browsed; the dropdown has to
        // offer the attribute tables too, or the default below picks one
        // the user has no way to see or change.
        let browsable: Vec<&String> = tables.iter().chain(attr_tables).collect();
        if !browsable.contains(&&self.browse_table) {
            self.browse_table = browsable.first().map(|t| (*t).clone()).unwrap_or_default();
        }
        let mut run = false;
        ui.horizontal(|ui| {
            let mut picked = self.browse_table.clone();
            egui::ComboBox::from_id_salt("sql_browse_table")
                .selected_text(if picked.is_empty() {
                    "no tables".to_string()
                } else {
                    picked.clone()
                })
                .show_ui(ui, |ui| {
                    for (l, t) in layers.iter().zip(tables) {
                        ui.selectable_value(&mut picked, t.clone(), t)
                            .on_hover_text(&l.name);
                    }
                    for (a, t) in attrs.iter().zip(attr_tables) {
                        ui.selectable_value(&mut picked, t.clone(), t)
                            .on_hover_text(format!("{} (attribute table)", a.name));
                    }
                });
            // Selecting a table only targets it — no implicit full-table
            // query (expensive on large/remote layers); ▶ or Enter runs.
            if picked != self.browse_table {
                self.browse_table = picked;
            }

            ui.label(RichText::new("WHERE").weak().monospace());
            let where_id = egui::Id::new("sql_browse_where");
            ctrl_z_alias(ui, where_id);
            let width = ui.available_width() - 200.0;
            let w = autocomplete_edit(
                ui,
                where_id,
                &mut self.browse_where,
                &mut self.ac_where,
                dict,
                move |text| {
                    egui::TextEdit::singleline(text)
                        .id(where_id)
                        .desired_width(width)
                        .hint_text("st_area(geometry) > 1000 — Enter applies")
                },
            );
            if w.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                run = true;
            }

            if ui
                .checkbox(&mut self.viewport_only, "viewport")
                .on_hover_text(
                    "Only rows intersecting the current map viewport \
                     (adds st_intersects against the view envelope; pruned \
                     via row-group/page statistics)",
                )
                .changed()
            {
                run = true;
            }
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
            let mut preds: Vec<String> = Vec::new();
            let w = self.browse_where.trim();
            if !w.is_empty() {
                preds.push(format!("({w})"));
            }
            if self.viewport_only {
                match self.viewport_predicate(layers, tables, view_world, display) {
                    Ok(p) => preds.push(p),
                    Err(e) => {
                        self.error = Some(e);
                        return;
                    }
                }
            }
            let mut sql = format!("select * from {}", self.browse_table);
            if !preds.is_empty() {
                sql.push_str(&format!(" where {}", preds.join(" and ")));
            }
            self.run(ui.ctx().clone(), layers, attrs, sql);
        }
    }

    /// `st_intersects(<geom>, st_makeenvelope(...))` for the current
    /// viewport, in the browsed layer's data CRS.
    ///
    /// `tables` is the *layer* table list only. It used to be the merged
    /// layer + attribute list while `layers` was not, so browsing an
    /// attribute table (the default pick with no layers loaded) indexed
    /// past the end of `layers` and panicked instead of saying that a
    /// table with no geometry cannot be clipped to the viewport.
    fn viewport_predicate(
        &self,
        layers: &[VectorLayer],
        tables: &[String],
        view_world: [f64; 4],
        display: &DisplayCrs,
    ) -> Result<String, String> {
        let no_geom = || {
            format!(
                "{} has no geometry column, so it cannot be limited to the viewport",
                self.browse_table
            )
        };
        let idx = tables
            .iter()
            .position(|t| *t == self.browse_table)
            .ok_or_else(no_geom)?;
        let layer = layers.get(idx).ok_or_else(no_geom)?;
        let b = viewport_to_data_bbox(view_world, display, &layer.crs)
            .ok_or("viewport does not transform into the layer CRS")?;
        let geom = layer
            .store
            .schema
            .field(layer.store.geom_col)
            .name()
            .to_lowercase();
        Ok(format!(
            "st_intersects({geom}, st_makeenvelope({}, {}, {}, {}))",
            b[0], b[1], b[2], b[3]
        ))
    }

    /// Free-form SQL editor with table-name chips and Ctrl+Enter.
    fn query_editor(
        &mut self,
        ui: &mut egui::Ui,
        layers: &[VectorLayer],
        attrs: &[AttrTable],
        tables: &[String],
        attr_tables: &[String],
        dict: &CompletionDict,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("tables:").weak().small());
            if layers.is_empty() && attrs.is_empty() {
                ui.label(RichText::new("none (load a layer first)").weak().small());
            }
            let chips: Vec<(&str, &String, bool)> = layers
                .iter()
                .map(|l| l.name.as_str())
                .zip(tables)
                .map(|(n, t)| (n, t, false))
                .chain(
                    attrs
                        .iter()
                        .map(|a| a.name.as_str())
                        .zip(attr_tables)
                        .map(|(n, t)| (n, t, true)),
                )
                .collect();
            for (name, t, is_attr) in chips {
                let label = RichText::new(t).monospace().small();
                // Tables read weaker than layers: same namespace, but one
                // of them is on the map and the other is not.
                let label = if is_attr { label.weak() } else { label };
                let hover = if is_attr {
                    format!("{name} (attribute table) — click to insert")
                } else {
                    format!("{name} — click to insert")
                };
                if ui
                    .add(egui::Button::new(label).frame(false))
                    .on_hover_text(hover)
                    .clicked()
                {
                    if self.query.trim().is_empty() {
                        self.query = format!("select * from {t}");
                    } else {
                        self.query.push_str(t);
                    }
                }
            }
        });

        let query_id = egui::Id::new("sql_query_edit");
        ctrl_z_alias(ui, query_id);
        autocomplete_edit(
            ui,
            query_id,
            &mut self.query,
            &mut self.ac_query,
            dict,
            |text| {
                egui::TextEdit::multiline(text)
                    .id(query_id)
                    .code_editor()
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .hint_text("select * from <table> where st_area(geometry) > 1000")
            },
        );

        // History: Ctrl+Up / Ctrl+Down step through past queries; the
        // in-progress draft is stashed and restored when stepping back.
        let (older, newer) = ui.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::ArrowUp),
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::ArrowDown),
            )
        });
        if older && !self.history.is_empty() {
            let next = match self.hist_pos {
                None => {
                    self.hist_draft = self.query.clone();
                    0
                }
                Some(p) => (p + 1).min(self.history.len() - 1),
            };
            self.hist_pos = Some(next);
            self.query = self.history[next].clone();
        }
        if newer {
            match self.hist_pos {
                Some(0) => {
                    self.hist_pos = None;
                    self.query = std::mem::take(&mut self.hist_draft);
                }
                Some(p) => {
                    self.hist_pos = Some(p - 1);
                    self.query = self.history[p - 1].clone();
                }
                None => {}
            }
        }

        ui.horizontal(|ui| {
            let run_clicked = ui
                .add_enabled(
                    self.running.is_none() && !self.query.trim().is_empty(),
                    egui::Button::new("▶ Run"),
                )
                .on_hover_text("Ctrl+Enter")
                .clicked();
            let hotkey = ui
                .input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));
            if (run_clicked || hotkey) && self.running.is_none() && !self.query.trim().is_empty()
            {
                let sql = self.query.clone();
                self.push_history(&sql);
                self.run(ui.ctx().clone(), layers, attrs, sql);
            }

            if !self.history.is_empty() {
                ui.menu_button(
                    format!("{} History", egui_phosphor::regular::CLOCK_COUNTER_CLOCKWISE),
                    |ui| {
                        ui.set_max_width(420.0);
                        let mut pick: Option<String> = None;
                        for h in &self.history {
                            let label = h.replace(['\n', '\t'], " ");
                            let label = if label.chars().count() > 70 {
                                let cut: String = label.chars().take(70).collect();
                                format!("{cut}…")
                            } else {
                                label
                            };
                            if ui
                                .button(RichText::new(label).monospace().small())
                                .on_hover_text(h)
                                .clicked()
                            {
                                pick = Some(h.clone());
                                ui.close();
                            }
                        }
                        if let Some(h) = pick {
                            self.hist_pos = None;
                            self.query = h;
                        }
                    },
                )
                .response
                .on_hover_text("Recent queries (Ctrl+Up / Ctrl+Down)");
            }
        });
    }

    fn sql_layers(layers: &[VectorLayer], names: Vec<String>) -> Vec<SqlLayer> {
        layers
            .iter()
            .zip(names)
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
            .collect()
    }

    fn run(
        &mut self,
        egui_ctx: egui::Context,
        layers: &[VectorLayer],
        attrs: &[AttrTable],
        sql: String,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        self.running = Some(id);
        self.error = None;
        self.last_sql = sql.clone();
        let (layer_names, attr_names) = sql_names(layers, attrs);
        self.last_layers = Self::sql_layers(layers, layer_names);
        self.last_tables = attrs
            .iter()
            .zip(attr_names)
            .map(|(t, table)| engine::SqlTable {
                table,
                schema: Arc::clone(&t.schema),
                batches: Arc::clone(&t.batches),
            })
            .collect();
        // A fresh flag per job; the previous one may still be set from a
        // Stop the user pressed a moment ago.
        self.cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        engine::spawn_query(
            id,
            sql,
            self.last_layers.clone(),
            self.last_tables.clone(),
            self.tx.clone(),
            Arc::clone(&self.cancel),
            move || {
                egui_ctx.request_repaint();
            },
        );
    }

    /// "Result as layer": materialized fast path when complete, streaming
    /// re-run when the display cap truncated the result.
    fn export_full(&mut self) -> Option<ConsoleAction> {
        let out = self.result.as_ref()?;
        if !out.truncated {
            return match self.export_materialized(None) {
                Ok(path) => Some(ConsoleAction::LoadLayer(path)),
                Err(e) => {
                    self.error = Some(e);
                    None
                }
            };
        }
        // Truncated: re-run the exact SQL against the same layer set and
        // stream everything.
        let id = self.next_id;
        self.next_id += 1;
        self.exporting = Some(id);
        self.export_n += 1;
        let path = std::env::temp_dir().join(format!(
            "geopq_query_{}_{}.parquet",
            std::process::id(),
            self.export_n
        ));
        self.cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        engine::spawn_export(
            id,
            self.last_sql.clone(),
            self.last_layers.clone(),
            self.last_tables.clone(),
            path,
            self.tx.clone(),
            Arc::clone(&self.cancel),
            || {},
        );
        None
    }

    /// Export the materialized result (or only `rows`) to a temp GeoParquet.
    fn export_materialized(&mut self, rows: Option<&BTreeSet<usize>>) -> Result<PathBuf, String> {
        let out = self.result.as_ref().ok_or("no result")?;
        let (gcol, crs) = out.geom.clone().ok_or("no geometry column in result")?;
        let filtered;
        let batches: &[RecordBatch] = match rows {
            Some(rows) => {
                filtered = vec![filter_rows(out, rows)?];
                &filtered
            }
            None => std::slice::from_ref(&out.batch),
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
            .filter_map(|&r| geometry_at(out, r))
            .collect();
        if geoms.is_empty() {
            return ConsoleAction::ClearSelection;
        }
        ConsoleAction::Select { crs, geoms }
    }

    /// Zoom to a row's feature, highlighting it along with the checked set.
    fn zoom_action(&self, row: usize) -> Option<ConsoleAction> {
        let out = self.result.as_ref()?;
        let (_, crs) = out.geom.clone()?;
        let zoom = geometry_at(out, row)?;
        let mut highlight: Vec<geo_types::Geometry<f64>> =
            if self.selected_rows.len() <= MAX_HIGHLIGHT_ROWS {
                self.selected_rows
                    .iter()
                    .filter(|&&r| r != row)
                    .filter_map(|&r| geometry_at(out, r))
                    .collect()
            } else {
                Vec::new()
            };
        highlight.push(zoom.clone());
        Some(ConsoleAction::Zoom {
            crs,
            zoom,
            highlight,
        })
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

/// Names offered by autocomplete, split by context: inside a FROM/JOIN
/// clause only table names make sense; everywhere else the full set
/// (column names lowercased, table names, ST_* functions, SQL keywords).
pub(crate) struct CompletionDict {
    pub tables: Vec<String>,
    pub all: Vec<String>,
    /// Column names per table, for qualified completion.
    pub columns: std::collections::HashMap<String, Vec<String>>,
}

impl CompletionDict {
    /// The candidate list for a token whose preceding text is `before`,
    /// within the query `sql`.
    ///
    /// `sql` is the whole text, not just what precedes the cursor: the
    /// FROM clause that names an alias is usually typed after the select
    /// list that uses it, so resolving `c.` from `before` alone would
    /// work only when the query is written backwards.
    fn for_context(&self, sql: &str, before: &str) -> &[String] {
        if let Some(q) = qualifier(before) {
            let by = aliases(sql);
            let table = by.get(&q).cloned().unwrap_or(q);
            if let Some(cols) = self.columns.get(&table) {
                return cols;
            }
            // An unknown qualifier is a typo or a table not loaded;
            // everything is a better answer than nothing.
            return &self.all;
        }
        if from_context(before) {
            &self.tables
        } else {
            &self.all
        }
    }
}

/// The table qualifier a token is written under: the `c` of `c.nom`.
///
/// `before` is the text up to the token, so a qualified token leaves it
/// ending in `<ident>.`.
fn qualifier(before: &str) -> Option<String> {
    let rest = before.strip_suffix('.')?;
    let ident: String = rest
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    // A dot after a digit is a decimal point, not a qualifier.
    if ident.is_empty() || ident.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    Some(ident.to_lowercase())
}

/// Table aliases declared in a query, mapping alias to table.
///
/// `from communes c`, `join codes as k`, and the bare `from communes`
/// which lets `communes.` qualify itself.
fn aliases(sql: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let words: Vec<String> = sql
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let mut i = 0;
    while i < words.len() {
        if words[i] != "from" && words[i] != "join" {
            i += 1;
            continue;
        }
        let Some(table) = words.get(i + 1) else { break };
        if is_sql_word(table) {
            i += 1;
            continue;
        }
        out.insert(table.clone(), table.clone());
        // `as` is optional, and the next word is only an alias if it is
        // not the keyword that ends the clause.
        let mut j = i + 2;
        if words.get(j).is_some_and(|w| w == "as") {
            j += 1;
        }
        if let Some(alias) = words.get(j).filter(|w| !is_sql_word(w)) {
            out.insert(alias.clone(), table.clone());
            i = j + 1;
        } else {
            i += 2;
        }
    }
    out
}

/// A word that ends a table reference rather than aliasing it.
fn is_sql_word(w: &str) -> bool {
    SQL_KEYWORDS
        .iter()
        .any(|k| k.split_whitespace().any(|kw| kw == w))
        || udf::NAMES.contains(&w)
        || super::agg::NAMES.contains(&w)
}

fn completion_dict(
    layers: &[VectorLayer],
    attrs: &[AttrTable],
    tables: &[String],
    attr_tables: &[String],
) -> CompletionDict {
    let mut dict: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut push = |s: String| {
        if seen.insert(s.clone()) {
            dict.push(s);
        }
    };
    let mut columns: std::collections::HashMap<String, Vec<String>> = Default::default();
    for (l, name) in layers.iter().zip(tables) {
        // The SQL-visible names, which is what a qualified token has to
        // match: the layer table lowercases and dedupes them.
        let cols = super::table::sql_column_names(&l.store.schema);
        for c in &cols {
            push(c.clone());
        }
        columns.insert(name.clone(), cols);
    }
    for (t, name) in attrs.iter().zip(attr_tables) {
        let cols: Vec<String> = t
            .schema
            .fields()
            .iter()
            .map(|f| f.name().to_lowercase())
            .collect();
        for c in &cols {
            push(c.clone());
        }
        columns.insert(name.clone(), cols);
    }
    for t in tables.iter().chain(attr_tables) {
        push(t.clone());
    }
    for n in udf::NAMES.iter().chain(super::agg::NAMES) {
        push((*n).to_string());
    }
    for k in SQL_KEYWORDS {
        push((*k).to_string());
    }
    CompletionDict {
        tables: tables.iter().chain(attr_tables).cloned().collect(),
        all: dict,
        columns,
    }
}

/// Is a token whose preceding text is `before` part of a FROM/JOIN table
/// list? Walk backwards over identifiers, dots, commas and whitespace
/// (aliases and multi-table lists); the first SQL keyword or function
/// name reached decides. Any other character (parens, operators) means
/// an expression, e.g. a subquery's interior.
fn from_context(before: &str) -> bool {
    let is_keyword = |w: &str| {
        SQL_KEYWORDS
            .iter()
            .any(|k| k.split_whitespace().any(|kw| kw == w))
            || udf::NAMES.contains(&w)
            || super::agg::NAMES.contains(&w)
    };
    let chars: Vec<char> = before.chars().collect();
    let mut i = chars.len();
    while i > 0 {
        let c = chars[i - 1];
        if c.is_ascii_alphanumeric() || c == '_' {
            let end = i;
            while i > 0 && (chars[i - 1].is_ascii_alphanumeric() || chars[i - 1] == '_') {
                i -= 1;
            }
            let word: String = chars[i..end].iter().collect::<String>().to_lowercase();
            match word.as_str() {
                "from" | "join" => return true,
                w if is_keyword(w) => return false,
                // A plain identifier (earlier table, alias): keep walking.
                _ => {}
            }
        } else if c.is_whitespace() || c == ',' || c == '.' {
            i -= 1;
        } else {
            return false;
        }
    }
    false
}

/// A text edit with a completion popup: candidates from `dict` matching the
/// identifier under the cursor; Up/Down navigate, Tab/Enter accept, Esc
/// closes.
pub(crate) fn autocomplete_edit(
    ui: &mut egui::Ui,
    id: egui::Id,
    text: &mut String,
    ac: &mut AcState,
    dict: &CompletionDict,
    make_edit: impl FnOnce(&mut String) -> egui::TextEdit<'_>,
) -> egui::Response {
    // Handle navigation/acceptance keys before the editor consumes them.
    let mut accept: Option<String> = None;
    if ac.open && !ac.items.is_empty() {
        ui.input_mut(|i| {
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
                ac.selected = (ac.selected + 1) % ac.items.len();
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
                ac.selected = (ac.selected + ac.items.len() - 1) % ac.items.len();
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Escape) {
                ac.open = false;
                ac.dismissed = Some(Dismissed {
                    start: ac.token.start,
                    word: text.get(ac.token.clone()).unwrap_or_default().to_lowercase(),
                    sticky: true,
                });
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Tab)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
            {
                accept = Some(ac.items[ac.selected].clone());
            }
        });
    }
    if let Some(word) = &accept {
        apply_completion(ui.ctx(), id, text, ac, word);
    }

    let edit = make_edit(text);
    let output = edit.show(ui);
    let response = output.response.response.clone();

    // Recompute the token under the cursor and the candidate list. When
    // the pointer is over the popup, keep it open even though the click
    // just defocused the field — otherwise the popup vanishes between
    // press and release and candidates can never be clicked.
    let pointer_over_popup = ui
        .ctx()
        .memory(|m| m.area_rect(id.with("ac_popup")))
        .zip(ui.input(|i| i.pointer.latest_pos()))
        .is_some_and(|(rect, pos)| rect.expand(4.0).contains(pos));
    if !response.has_focus() && !pointer_over_popup {
        ac.open = false;
    }
    if response.has_focus() {
        ac.open = false;
        if let Some(range) = output.state.cursor.char_range() {
            let cursor_char = range.primary.index.0;
            let chars: Vec<char> = text.chars().collect();
            let cur = cursor_char.min(chars.len());
            let mut start = cur;
            while start > 0 && (chars[start - 1].is_ascii_alphanumeric() || chars[start - 1] == '_')
            {
                start -= 1;
            }
            if start < cur {
                let prefix: String = chars[start..cur].iter().collect();
                let prefix_l = prefix.to_lowercase();
                let before: String = chars[..start].iter().collect();
                let full: String = chars.iter().collect();
                let items: Vec<String> = dict
                    .for_context(&full, &before)
                    .iter()
                    .filter(|c| c.starts_with(&prefix_l) && **c != prefix_l)
                    .take(8)
                    .cloned()
                    .collect();
                if !items.is_empty() {
                    let byte_start: usize =
                        chars[..start].iter().map(|c| c.len_utf8()).sum();
                    let byte_end: usize = chars[..cur].iter().map(|c| c.len_utf8()).sum();
                    if suppressed(ac.dismissed.as_ref(), byte_start, &prefix_l) {
                        ac.token = byte_start..byte_end;
                    } else {
                        ac.dismissed = None;
                        ac.token = byte_start..byte_end;
                        ac.selected = ac.selected.min(items.len() - 1);
                        ac.items = items;
                        ac.open = true;
                    }
                }
            }
        }
    }

    if ac.open {
        let mut clicked: Option<String> = None;
        egui::Area::new(id.with("ac_popup"))
            .order(egui::Order::Foreground)
            .fixed_pos(response.rect.left_bottom() + egui::vec2(8.0, 2.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(260.0);
                    for (i, item) in ac.items.iter().enumerate() {
                        if ui
                            .selectable_label(i == ac.selected, RichText::new(item).monospace())
                            .clicked()
                        {
                            clicked = Some(item.clone());
                        }
                    }
                    ui.label(
                        RichText::new("Tab/Enter to complete · Esc to dismiss")
                            .weak()
                            .small(),
                    );
                });
            });
        if let Some(word) = clicked {
            apply_completion(ui.ctx(), id, text, ac, &word);
            response.request_focus();
        }
    }
    response
}

/// Replace the token under completion with `word` and move the cursor to
/// its end.
fn apply_completion(
    ctx: &egui::Context,
    id: egui::Id,
    text: &mut String,
    ac: &mut AcState,
    word: &str,
) {
    let range = ac.token.clone();
    // Both ends: the token is a byte range into the text as it was when
    // the popup opened, and an edit since then can leave `end` inside a
    // multi-byte character — `replace_range` panics on that, not errors.
    if range.end > text.len()
        || !text.is_char_boundary(range.start)
        || !text.is_char_boundary(range.end)
    {
        ac.open = false;
        return;
    }
    text.replace_range(range.clone(), word);
    // The cursor now sits at the end of the word just inserted, which is
    // itself a prefix of every longer candidate. Without this the popup
    // reopens on the completion it just applied, and only a character
    // that breaks the prefix (a comma, a space) closes it.
    ac.dismissed = Some(Dismissed {
        start: range.start,
        word: word.to_lowercase(),
        sticky: false,
    });
    let cursor_chars = text[..range.start + word.len()].chars().count();
    if let Some(mut state) = egui::text_edit::TextEditState::load(ctx, id) {
        state.cursor.set_char_range(Some(egui::text::CCursorRange::one(
            egui::text::CCursor::new(cursor_chars),
        )));
        state.store(ctx, id);
    }
    ac.open = false;
}

/// SQL identifier per layer; same-named layers get `_2`, `_3`, ...
/// suffixes so both stay queryable.
/// SQL identifiers for everything queryable: layers first, then attribute
/// tables, deduped against each other.
///
/// One namespace, because a query joining a layer to a table writes both
/// names in the same FROM clause. Layers keep their existing names when a
/// table would collide, so adding a table never renames a layer out from
/// under a query the user already wrote.
pub fn attr_sql_names(layers: &[VectorLayer], attrs: &[AttrTable]) -> Vec<String> {
    sql_names(layers, attrs).1
}

/// SQL identifiers for layers and attribute tables, in their own order.
pub fn sql_table_names(
    layers: &[VectorLayer],
    attrs: &[AttrTable],
) -> (Vec<String>, Vec<String>) {
    sql_names(layers, attrs)
}

/// The SQL table names and the autocomplete dictionary for one layer
/// and attribute-table set.
///
/// `panel_ui` used to build both every frame: `completion_dict` walks
/// every column of every layer through `sql_column_names` and fills two
/// hash maps, sixty times a second, for a set that changes when a layer
/// is loaded or renamed. Cached behind a key over the sources instead.
pub(crate) struct Sources {
    tables: Vec<String>,
    attr_tables: Vec<String>,
    dict: CompletionDict,
}

impl SqlConsole {
    /// Cached [`Sources`], rebuilt when the layer/table set changes.
    /// Returns an `Arc` so the caller holds no borrow of `self`.
    fn sources(&mut self, layers: &[VectorLayer], attrs: &[AttrTable]) -> Arc<Sources> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for l in layers {
            l.id.hash(&mut h);
            l.name.hash(&mut h);
            // A rebuilt store (a reopened file) can change the columns
            // under the same id and name.
            (Arc::as_ptr(&l.store) as usize).hash(&mut h);
        }
        for t in attrs {
            t.id.hash(&mut h);
            t.name.hash(&mut h);
            (Arc::as_ptr(&t.schema) as *const u8 as usize).hash(&mut h);
        }
        let key = h.finish();
        if self.sources_key != Some(key) || self.sources.is_none() {
            let (tables, attr_tables) = sql_names(layers, attrs);
            let dict = completion_dict(layers, attrs, &tables, &attr_tables);
            self.sources = Some(Arc::new(Sources {
                tables,
                attr_tables,
                dict,
            }));
            self.sources_key = Some(key);
        }
        Arc::clone(self.sources.as_ref().expect("just built"))
    }
}

fn sql_names(layers: &[VectorLayer], attrs: &[AttrTable]) -> (Vec<String>, Vec<String>) {
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    let mut dedupe = |name: &str| {
        let base = engine::table_name(name);
        let n = seen.entry(base.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            format!("{base}_{n}")
        } else {
            base
        }
    };
    let l: Vec<String> = layers.iter().map(|l| dedupe(&l.name)).collect();
    let a: Vec<String> = attrs.iter().map(|t| dedupe(&t.name)).collect();
    (l, a)
}

/// Decode the geometry of one result row (underlying index).
fn geometry_at(out: &QueryOutput, r: usize) -> Option<geo_types::Geometry<f64>> {
    let (gcol, _) = out.geom.as_ref()?;
    let arr = out.batch.column(*gcol).as_any().downcast_ref::<BinaryArray>()?;
    if r >= arr.len() || arr.is_null(r) {
        return None;
    }
    decode_wkb(arr.value(r))
}

/// The full (untruncated) value of one result cell; geometry as WKT.
fn cell_value(out: &QueryOutput, r: usize, c: usize) -> String {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    if out.geom.as_ref().is_some_and(|(gcol, _)| *gcol == c) {
        return match geometry_at(out, r) {
            Some(g) => {
                use wkt::ToWkt;
                g.wkt_string()
            }
            None => String::new(),
        };
    }
    let opts = FormatOptions::default().with_display_error(true);
    ArrayFormatter::try_new(out.batch.column(c).as_ref(), &opts)
        .map(|f| f.value(r).to_string())
        .unwrap_or_default()
}

/// Only the given rows of the result, as one new record batch.
fn filter_rows(out: &QueryOutput, rows: &BTreeSet<usize>) -> Result<RecordBatch, String> {
    use arrow::array::UInt32Array;
    let indices = UInt32Array::from(rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
    let cols = out
        .batch
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("row filter: {e}"))?;
    RecordBatch::try_new(out.schema.clone(), cols).map_err(|e| format!("row filter: {e}"))
}

/// Checked rows as tab-separated text with a header line, for spreadsheets.
fn selection_tsv(out: &QueryOutput, rows: &BTreeSet<usize>) -> String {
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
            let v = cell_value(out, r, c);
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
    /// The 🔍 button of this row (underlying index) was clicked.
    zoom_clicked: Option<usize>,
    /// A sortable header was clicked (column index).
    sort_clicked: Option<usize>,
    /// A cell value was copied to the clipboard.
    copied: bool,
}

/// Virtualized, paginated result grid. A checkbox per row adds the feature
/// to the map selection, 🔍 zooms to it, clicking any cell copies its full
/// value, clicking a header cycles the sort.
fn results_table(
    ui: &mut egui::Ui,
    out: &QueryOutput,
    selected: &mut BTreeSet<usize>,
    page: usize,
    page_size: usize,
    sort: Option<(usize, bool)>,
    perm: Option<&[u32]>,
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

    let total = out.total_rows;
    let page_start = (page * page_size).min(total);
    let page_rows = page_size.min(total - page_start);
    let underlying = |view: usize| -> usize {
        perm.map_or(view, |p| p[view] as usize)
    };

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
                // Select all / none (the WHOLE result, not just this page);
                // shows the mixed state when partial.
                let n_sel = selected.len();
                let mut all = n_sel == total && total > 0;
                if ui
                    .add(
                        egui::Checkbox::without_text(&mut all)
                            .indeterminate(n_sel > 0 && n_sel < total),
                    )
                    .on_hover_text("Check / uncheck all result rows")
                    .changed()
                {
                    selected.clear();
                    if all {
                        selected.extend(0..total);
                    }
                    ev.toggled = true;
                }
            });
            if has_geom {
                header.col(|_| {});
            }
            for (c, f) in out.schema.fields().iter().enumerate() {
                header.col(|ui| {
                    if Some(c) == geom_col {
                        ui.label(RichText::new(f.name()).strong().small());
                        return;
                    }
                    let arrow = match sort {
                        Some((sc, true)) if sc == c => " ⏶",
                        Some((sc, false)) if sc == c => " ⏷",
                        _ => "",
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(format!("{}{arrow}", f.name())).strong().small(),
                            )
                            .frame(false),
                        )
                        .on_hover_text("Sort: ascending / descending / none")
                        .clicked()
                    {
                        ev.sort_clicked = Some(c);
                    }
                });
            }
        })
        .body(|body| {
            body.rows(22.0, page_rows, |mut row| {
                let u = underlying(page_start + row.index());
                row.set_selected(selected.contains(&u));
                row.col(|ui| {
                    let mut checked = selected.contains(&u);
                    if ui
                        .checkbox(&mut checked, "")
                        .on_hover_text("Add to the map selection")
                        .changed()
                    {
                        if checked {
                            selected.insert(u);
                        } else {
                            selected.remove(&u);
                        }
                        ev.toggled = true;
                    }
                });
                if has_geom {
                    row.col(|ui| {
                        if ui
                            .add(egui::Button::new(RichText::new("🔍").small()).frame(false))
                            .on_hover_text("Zoom to this feature (and highlight it)")
                            .clicked()
                        {
                            ev.zoom_clicked = Some(u);
                        }
                    });
                }
                for c in 0..n_cols {
                    row.col(|ui| {
                        let text = if Some(c) == geom_col {
                            geom_cell(out.batch.column(c).as_ref(), u)
                        } else {
                            ArrayFormatter::try_new(out.batch.column(c).as_ref(), &opts)
                                .map(|f| f.value(u).to_string())
                                .unwrap_or_else(|_| "<?>".into())
                        };
                        let resp = ui.add(
                            egui::Label::new(RichText::new(text).small())
                                .sense(egui::Sense::click()),
                        );
                        if resp.clicked() {
                            ui.ctx().copy_text(cell_value(out, u, c));
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

#[cfg(test)]
mod tests {
    use super::{aliases, from_context, qualifier, suppressed, CompletionDict, Dismissed};

    /// Browse mode offers the attribute tables as well as the layers,
    /// and with no layers loaded the default pick is one of them. The
    /// viewport filter used to look that name up in the merged list and
    /// index the layer list with the result: `&layers[0]` on an empty
    /// slice, i.e. a panic on a plain checkbox click.
    #[test]
    fn the_viewport_filter_refuses_a_table_without_geometry() {
        use crate::data::crs::{Crs, DisplayCrs};
        let mut c = super::SqlConsole::new();
        c.browse_table = "attr_1".into();
        let display = DisplayCrs::new(Crs::wgs84());
        // No layers at all: what a session with only an imported CSV has.
        let err = c
            .viewport_predicate(&[], &[], [0.0, 0.0, 1.0, 1.0], &display)
            .unwrap_err();
        assert!(err.contains("attr_1"), "{err}");
        assert!(err.contains("no geometry"), "{err}");
        // And with layers loaded, a name that is not one of them is
        // still an explanation rather than an out-of-bounds index.
        let err = c
            .viewport_predicate(&[], &["roads".into()], [0.0, 0.0, 1.0, 1.0], &display)
            .unwrap_err();
        assert!(err.contains("no geometry"), "{err}");
    }

    /// Accepting a candidate must not reopen the popup on the word it
    /// just inserted, and typing on must bring it back.
    #[test]
    fn an_accepted_completion_shuts_the_popup_until_the_word_changes() {
        let d = Dismissed {
            start: 7,
            word: "nom".into(),
            sticky: false,
        };
        assert!(suppressed(Some(&d), 7, "nom"));
        // one more character: the user is typing again, so offer again
        assert!(!suppressed(Some(&d), 7, "nom_"));
        // the same word elsewhere in the query is a different token
        assert!(!suppressed(Some(&d), 20, "nom"));
    }

    /// Escape means "leave this word alone", so it has to survive the
    /// next keystroke: the popup is reopened from the cursor every frame.
    #[test]
    fn escape_shuts_the_popup_for_the_rest_of_the_word() {
        let d = Dismissed {
            start: 7,
            word: "nom".into(),
            sticky: true,
        };
        assert!(suppressed(Some(&d), 7, "nom"));
        assert!(suppressed(Some(&d), 7, "nom_off"));
        // deleting back past what was dismissed starts a new word
        assert!(!suppressed(Some(&d), 7, "no"));
        assert!(!suppressed(None, 7, "nom"));
    }

    fn dict() -> CompletionDict {
        let mut columns = std::collections::HashMap::new();
        columns.insert(
            "communes".to_string(),
            vec!["code_insee".into(), "nom_officiel".into(), "geometrie".into()],
        );
        columns.insert(
            "ec_nais_com".to_string(),
            vec!["code".into(), "_2025".into()],
        );
        CompletionDict {
            tables: vec!["communes".into(), "ec_nais_com".into()],
            all: vec!["code_insee".into(), "nom_officiel".into(), "code".into(), "select".into()],
            columns,
        }
    }

    #[test]
    fn a_qualifier_is_an_identifier_before_a_dot() {
        assert_eq!(qualifier("select c."), Some("c".into()));
        assert_eq!(qualifier("select a.b, ec_nais_com."), Some("ec_nais_com".into()));
        assert_eq!(qualifier("select C."), Some("c".into()), "case folded");
        assert_eq!(qualifier("select "), None);
        // A decimal point is not a qualifier.
        assert_eq!(qualifier("where x > 3."), None);
    }

    #[test]
    fn aliases_are_read_from_from_and_join() {
        let a = aliases("select * from communes c join ec_nais_com as n on c.code_insee = n.code");
        assert_eq!(a.get("c"), Some(&"communes".to_string()));
        assert_eq!(a.get("n"), Some(&"ec_nais_com".to_string()));
        // A table with no alias qualifies itself.
        assert_eq!(a.get("communes"), Some(&"communes".to_string()));
        // A keyword after the table is not an alias.
        let a = aliases("select * from communes where x = 1");
        assert_eq!(a.get("communes"), Some(&"communes".to_string()));
        assert!(!a.contains_key("where"));
        let a = aliases("select * from communes order by nom");
        assert!(!a.contains_key("order"));
    }

    /// The point of the whole thing: after `from communes c`, `c.` offers
    /// that table's columns and nothing else.
    #[test]
    fn a_qualified_token_offers_only_that_tables_columns() {
        let d = dict();
        let sql = "select c.nom, n.code from communes c join ec_nais_com n on c.code_insee = n.code";
        assert_eq!(
            d.for_context(sql, "select c."),
            &["code_insee".to_string(), "nom_officiel".into(), "geometrie".into()],
        );
        assert_eq!(
            d.for_context(sql, "select c.nom, n."),
            &["code".to_string(), "_2025".into()],
        );
        // The alias is declared after the cursor in a query being written
        // front to back, which is how they are actually typed.
        assert_eq!(
            d.for_context("select c. from communes c", "select c."),
            &["code_insee".to_string(), "nom_officiel".into(), "geometrie".into()],
        );
        // An unqualified token still gets everything.
        assert_eq!(d.for_context(sql, "select "), &d.all[..]);
        // A qualifier naming nothing loaded falls back rather than
        // offering an empty list.
        assert_eq!(d.for_context(sql, "select zz."), &d.all[..]);
    }

    #[test]
    fn from_context_detects_table_position() {
        // Completing a table name.
        assert!(from_context("select * from "));
        assert!(from_context("select a, b from "));
        assert!(from_context("select * from roads, "));
        assert!(from_context("select * from roads r join "));
        assert!(from_context("SELECT * FROM "));
        // Completing anything else.
        assert!(!from_context(""));
        assert!(!from_context("select "));
        assert!(!from_context("select * from t where "));
        assert!(!from_context("select * from t where pop > 10 and "));
        assert!(!from_context("select * from t order by "));
        // Expressions and subquery interiors.
        assert!(!from_context("select * from t where st_area("));
        assert!(!from_context("select * from (select "));
    }
}
