//! Attribute tables: sources opened for their columns alone.
//!
//! These carry no geometry and no map presence. They exist to be queried
//! and, mostly, to be joined against a layer — a lookup table of codes and
//! names, a spreadsheet of measurements keyed by an identifier the layer
//! also carries.
//!
//! Opening one is two steps. [`inspect`] reads a sample and proposes a
//! plan: a delimiter, a name and a type per column. [`import`] applies the
//! plan the user approved. The split exists because inference is a guess
//! and only the user knows the answer — one `NA` in ten thousand rows
//! silently turns a column of counts into text, and a column of INSEE
//! codes reads as integers until the first Corsican `2A004` arrives. Both
//! are invisible in the data and obvious in a list of columns.
//!
//! Values are read as text and cast, rather than parsed straight into
//! their type. That is what makes an unparseable value a NULL that gets
//! counted and reported instead of an error that fails the whole import,
//! and it is what lets the preview say what a type choice would cost
//! before it is made.
//!
//! Unlike layers, tables are read whole into memory. That is a deliberate
//! departure from how the rest of the app treats data, and geometry is
//! what makes the difference: a layer's bulk is its coordinates, which is
//! why it pays to page them in by viewport. A table with no geometry is a
//! few columns of scalars, the small side of a join, and there is no
//! viewport to prune it by — nothing about a join is spatial.
//! [`MAX_BYTES`] keeps that honest.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow::compute::{cast_with_options, CastOptions};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::ChunkReader;

use super::source::Source;

/// Most an attribute table may occupy once decoded.
pub const MAX_BYTES: usize = 512 * 1024 * 1024;

/// Rows read per batch. Matches the layer scan path.
const BATCH_SIZE: usize = 8192;

/// Rows sampled for the preview. Enough that a stray marker late in a
/// column still shows up, cheap enough that opening the dialog is instant.
const SAMPLE_ROWS: usize = 10_000;

/// Distinct values shown per column in the preview.
const SAMPLE_VALUES: usize = 3;

/// Delimiters tried when detecting one.
const DELIMITERS: [u8; 4] = [b',', b';', b'\t', b'|'];

// ---------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------

/// The types a column can be imported as.
///
/// Deliberately few. These are join keys and values to style by, and every
/// extra option is one more thing to get wrong in a dialog meant to be
/// read at a glance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColType {
    Text,
    Integer,
    Float,
    Boolean,
}

impl ColType {
    pub fn label(self) -> &'static str {
        match self {
            ColType::Text => "text",
            ColType::Integer => "integer",
            ColType::Float => "float",
            ColType::Boolean => "boolean",
        }
    }

    pub const ALL: [ColType; 4] = [
        ColType::Text,
        ColType::Integer,
        ColType::Float,
        ColType::Boolean,
    ];

    fn arrow(self) -> DataType {
        match self {
            ColType::Text => DataType::Utf8,
            ColType::Integer => DataType::Int64,
            ColType::Float => DataType::Float64,
            ColType::Boolean => DataType::Boolean,
        }
    }

    fn of(t: &DataType) -> Self {
        match t {
            DataType::Boolean => ColType::Boolean,
            t if t.is_integer() => ColType::Integer,
            t if t.is_floating() => ColType::Float,
            _ => ColType::Text,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColumnPlan {
    /// Name as the file spells it, kept so the import can find the column
    /// again after a rename.
    pub source_name: String,
    /// Name to import it under. Editable.
    pub name: String,
    pub ty: ColType,
    pub include: bool,
}

/// How numbers are written in the file.
///
/// Not a "strip the separators" switch, because a comma means opposite
/// things either side of the Channel: `3,739` is three thousand seven
/// hundred and thirty-nine in an English-formatted export and three point
/// seven three nine in a French one. Nothing in the value says which, so
/// the file says which.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum NumberFormat {
    /// `1234.56` — no grouping at all.
    #[default]
    Plain,
    /// `1,234.56` — comma groups thousands, dot is the decimal point.
    GroupComma,
    /// `1 234,56` or `1.234,56` — comma is the decimal point.
    DecimalComma,
}

impl NumberFormat {
    pub fn label(self) -> &'static str {
        match self {
            NumberFormat::Plain => "1234.56",
            NumberFormat::GroupComma => "1,234.56",
            NumberFormat::DecimalComma => "1 234,56",
        }
    }

    pub const ALL: [NumberFormat; 3] = [
        NumberFormat::Plain,
        NumberFormat::GroupComma,
        NumberFormat::DecimalComma,
    ];
}

/// Grouping characters that appear between digits: plain space and the two
/// non-breaking ones French exports use, plus the Swiss apostrophe.
const GROUPING: [char; 4] = [' ', '\u{00A0}', '\u{202F}', '\''];

/// Rewrite a number as Rust's parsers expect it. Returns the input
/// unchanged when there is nothing to do, which is the common case.
pub fn normalize_number(v: &str, fmt: NumberFormat) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    let t = v.trim();
    match fmt {
        NumberFormat::Plain => {
            if t.len() == v.len() {
                Cow::Borrowed(t)
            } else {
                Cow::Owned(t.to_string())
            }
        }
        NumberFormat::GroupComma => {
            if !t.contains(',') && !t.contains(GROUPING) {
                return Cow::Borrowed(t);
            }
            Cow::Owned(t.chars().filter(|c| *c != ',' && !GROUPING.contains(c)).collect())
        }
        NumberFormat::DecimalComma => {
            if !t.contains(',') && !t.contains('.') && !t.contains(GROUPING) {
                return Cow::Borrowed(t);
            }
            // The dot groups here, so it goes; the comma becomes the point.
            Cow::Owned(
                t.chars()
                    .filter(|c| *c != '.' && !GROUPING.contains(c))
                    .map(|c| if c == ',' { '.' } else { c })
                    .collect(),
            )
        }
    }
}

/// Whether the file describes places or only values.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum GeometryPlan {
    /// Columns only: a table, with no presence on the map.
    #[default]
    None,
    /// A point per row, built from two coordinate columns.
    Points {
        x: String,
        y: String,
        /// What the coordinates are in. Guessed only when the values
        /// leave no doubt; see [`suggest_points`].
        epsg: u32,
    },
}

#[derive(Clone, Debug)]
pub struct ImportPlan {
    /// Ignored for parquet.
    pub delimiter: u8,
    /// Ignored for parquet.
    pub has_header: bool,
    /// Ignored for parquet, where numbers are already numbers.
    pub numbers: NumberFormat,
    pub geometry: GeometryPlan,
    pub columns: Vec<ColumnPlan>,
}

/// What the sample says about one column, for the dialog to show.
#[derive(Clone, Debug)]
pub struct ColumnPreview {
    /// What inference proposed, before any edit.
    pub inferred: ColType,
    pub samples: Vec<String>,
    /// Sampled rows that are not empty but would not survive the currently
    /// planned type.
    pub bad: usize,
    pub bad_examples: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Preview {
    pub plan: ImportPlan,
    pub columns: Vec<ColumnPreview>,
    pub sampled_rows: usize,
    /// True for parquet, where types are declared rather than guessed and
    /// there is no delimiter to choose.
    pub typed_source: bool,
    /// Whether the sample values have arrived. False for the first,
    /// footer-only look at a parquet file: names and types are in the
    /// footer and cost one small read, while the values behind them cost
    /// a page of every column and can take half a minute over a network.
    /// The dialog is worth showing before that finishes.
    pub sampled: bool,
}

#[derive(Debug)]
pub struct AttrData {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
    pub rows: usize,
    /// Decoded size, for the panel and for the cap.
    pub bytes: usize,
    /// Values that did not survive their column's chosen type, by column
    /// name. Reported after the import rather than blocking it.
    pub nulled: Vec<(String, usize)>,
}

/// A loaded attribute table, as the app holds it.
pub struct AttrTable {
    pub id: u64,
    /// Display name, editable like a layer's.
    pub name: String,
    /// The file it came from, which is what the user recognizes it by —
    /// not `parquet`, which is an artifact of the import.
    pub source: Source,
    pub schema: SchemaRef,
    pub batches: Arc<Vec<RecordBatch>>,
    pub rows: usize,
    pub bytes: usize,
    /// The imported copy on disk. Temporary and removed at exit unless the
    /// user saves it somewhere.
    pub parquet: Option<std::path::PathBuf>,
}

impl AttrTable {
    pub fn new(id: u64, name: String, source: Source, data: AttrData) -> Self {
        Self {
            id,
            name,
            source,
            schema: data.schema,
            batches: Arc::new(data.batches),
            rows: data.rows,
            bytes: data.bytes,
            parquet: None,
        }
    }
}

// ---------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------

/// A column name that can be written in SQL without quoting.
///
/// Lowercased, because unquoted identifiers arrive lowercased from the
/// parser; everything outside `[a-z0-9_]` collapsed to `_`; and a leading
/// digit prefixed, since `2025` parses as a number and `n.2025` is a
/// syntax error rather than a column reference.
pub fn sanitize(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 1);
    let mut last_us = false;
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_us = false;
        } else if !last_us {
            out.push('_');
            last_us = true;
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        return "column".into();
    }
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{out}")
    } else {
        out
    }
}

/// Make every name unique, in order, with `_2`, `_3`, … suffixes.
pub fn dedupe(names: &mut [String]) {
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    for n in names.iter_mut() {
        let e = seen.entry(n.clone()).or_insert(0);
        *e += 1;
        if *e > 1 {
            *n = format!("{n}_{}", *e);
        }
    }
}

// ---------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------

/// The extension a source's path or URL ends in, lowercased and without
/// the dot. `Source::name` is the display name and drops it.
fn extension(source: &Source) -> String {
    let label = source.label();
    let path = label.split(['?', '#']).next().unwrap_or(&label);
    path.rsplit(['/', '\\'])
        .next()
        .and_then(|file| file.rsplit_once('.'))
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// Is this a source that can only be an attribute table?
///
/// A CSV has no geometry to draw, so opening one as a layer can only
/// fail. Parquet is not included: most parquet handed to this app is
/// GeoParquet, and a plain one is better opened explicitly than guessed at.
pub fn is_tabular(source: &Source) -> bool {
    is_csv(source)
}

fn is_csv(source: &Source) -> bool {
    matches!(extension(source).as_str(), "csv" | "tsv" | "txt")
}

fn whole(source: &Source) -> Result<Box<dyn std::io::Read + Send>, String> {
    source
        .open()?
        .get_read(0)
        .map_err(|e| format!("{}: cannot read ({e})", source.name()))
}

/// Guess the delimiter from the first line: whichever candidate splits it
/// into the most fields.
///
/// Worth doing rather than assuming a comma: most European public data is
/// semicolon separated, and a wrong guess loads the whole file as one
/// column, which looks like a corrupt file rather than a settings problem.
fn detect_delimiter(source: &Source) -> u8 {
    use std::io::Read;
    let mut head = vec![0u8; 64 * 1024];
    let Ok(mut r) = whole(source) else {
        return b',';
    };
    let n = r.read(&mut head).unwrap_or(0);
    let line: Vec<u8> = head[..n]
        .split(|b| *b == b'\n')
        .next()
        .unwrap_or_default()
        .to_vec();
    // Counted outside quotes, so a comma inside "Paris, France" does not
    // win a file that is really semicolon separated.
    let count = |d: u8| {
        let mut in_q = false;
        line.iter().fold(0usize, |acc, &b| {
            if b == b'"' {
                in_q = !in_q;
                acc
            } else if b == d && !in_q {
                acc + 1
            } else {
                acc
            }
        })
    };
    DELIMITERS
        .iter()
        .copied()
        .max_by_key(|&d| count(d))
        .filter(|&d| count(d) > 0)
        .unwrap_or(b',')
}

fn csv_format(delimiter: u8, has_header: bool) -> arrow::csv::reader::Format {
    arrow::csv::reader::Format::default()
        .with_header(has_header)
        .with_delimiter(delimiter)
}

/// Column names and sampled values, all as text.
fn sample_text(source: &Source, plan: &ImportPlan) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    if is_csv(source) {
        sample_csv_text(source, plan.delimiter, plan.has_header)
    } else {
        sample_parquet_text(source)
    }
}

fn text_schema(names: &[String]) -> SchemaRef {
    Arc::new(Schema::new(
        names
            .iter()
            .map(|n| Field::new(n, DataType::Utf8, true))
            .collect::<Vec<_>>(),
    ))
}

fn columns_as_text(batch: &RecordBatch, cols: &mut [Vec<String>]) {
    for (i, col) in cols.iter_mut().enumerate() {
        if let Some(a) = batch.column(i).as_any().downcast_ref::<StringArray>() {
            for r in 0..a.len() {
                col.push(if a.is_null(r) {
                    String::new()
                } else {
                    a.value(r).to_string()
                });
            }
        }
    }
}

fn sample_csv_text(
    source: &Source,
    delimiter: u8,
    has_header: bool,
) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let format = csv_format(delimiter, has_header);
    // One record is enough to learn the column count and header names;
    // the types this pass would guess are thrown away.
    let (probe, _) = format
        .infer_schema(whole(source)?, Some(1))
        .map_err(|e| format!("{}: cannot read as CSV ({e})", source.name()))?;
    let names: Vec<String> = probe.fields().iter().map(|f| f.name().clone()).collect();
    let rdr = arrow::csv::ReaderBuilder::new(text_schema(&names))
        .with_format(format)
        .with_batch_size(BATCH_SIZE)
        .build(whole(source)?)
        .map_err(|e| format!("{}: cannot read as CSV ({e})", source.name()))?;
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); names.len()];
    let mut seen = 0usize;
    for batch in rdr {
        let batch = batch.map_err(|e| format!("{}: CSV parse error ({e})", source.name()))?;
        columns_as_text(&batch, &mut cols);
        seen += batch.num_rows();
        if seen >= SAMPLE_ROWS {
            break;
        }
    }
    Ok((names, cols))
}

fn sample_parquet_text(source: &Source) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let (schema, batches) = read_parquet(source, SAMPLE_ROWS, &[])?;
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); names.len()];
    for batch in &batches {
        let as_text = to_text(batch, &text_schema(&names))?;
        columns_as_text(&as_text, &mut cols);
    }
    Ok((names, cols))
}

/// Would this text survive being read as `ty`?
fn parses_as(v: &str, ty: ColType, fmt: NumberFormat) -> bool {
    let v = v.trim();
    if v.is_empty() {
        return true; // absent, not wrong
    }
    match ty {
        ColType::Text => true,
        ColType::Integer => normalize_number(v, fmt).parse::<i64>().is_ok(),
        ColType::Float => normalize_number(v, fmt).parse::<f64>().is_ok(),
        ColType::Boolean => matches!(
            v.to_ascii_lowercase().as_str(),
            "true" | "false" | "t" | "f" | "1" | "0" | "yes" | "no"
        ),
    }
}

/// `01001` is an INSEE code, not one thousand and one.
///
/// A suggestion, not a rule: it is proposed in the dialog with the values
/// that prompted it, and overridden with one click when it is wrong.
fn looks_like_identifier(values: &[String]) -> bool {
    values.iter().any(|v| {
        let v = v.trim();
        v.len() > 1 && v.starts_with('0') && v.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Written as a word, not as a number.
///
/// `parses_as` accepts `0` and `1` as booleans, because a user who picks
/// boolean for such a column means it. Inference must not: a column
/// holding only 0 and 1 is far more often a count, a flag to sum, or a
/// year offset than it is a truth value, and guessing boolean makes it
/// unjoinable against an integer key for reasons nothing on screen
/// explains.
fn spelled_boolean(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "true" | "false" | "t" | "f" | "yes" | "no"
    )
}

/// The type to propose for a column of text values.
fn suggest_type(values: &[String], fmt: NumberFormat) -> ColType {
    let nonempty: Vec<&String> = values.iter().filter(|v| !v.trim().is_empty()).collect();
    if nonempty.is_empty() || looks_like_identifier(values) {
        return ColType::Text;
    }
    if nonempty.iter().all(|v| spelled_boolean(v)) {
        return ColType::Boolean;
    }
    for ty in [ColType::Integer, ColType::Float] {
        if nonempty.iter().all(|v| parses_as(v, ty, fmt)) {
            return ty;
        }
    }
    ColType::Text
}

/// Guess the number format from the sample.
///
/// The tell is a group of exactly three digits after a separator: `3,739`
/// and `1 234` are grouped numbers, because no one writes three decimal
/// places on a count. `3,5` is left alone — it is genuinely ambiguous, and
/// a wrong guess there changes a value by a factor of a thousand rather
/// than failing visibly.
fn detect_number_format(cols: &[Vec<String>]) -> NumberFormat {
    let (mut comma, mut spaced) = (0usize, 0usize);
    for values in cols {
        for v in values.iter().take(SAMPLE_ROWS) {
            let t = v.trim();
            if t.is_empty() || t.len() > 32 {
                continue;
            }
            let bytes: Vec<char> = t.chars().collect();
            for i in 0..bytes.len() {
                let sep = bytes[i];
                let grouping = GROUPING.contains(&sep);
                if sep != ',' && !grouping {
                    continue;
                }
                // Three digits, then either the end or another separator.
                let after: Vec<char> = bytes[i + 1..].iter().copied().take(4).collect();
                let three = after.len() >= 3 && after[..3].iter().all(|c| c.is_ascii_digit());
                let ends = after.len() == 3
                    || after
                        .get(3)
                        .is_some_and(|c| !c.is_ascii_digit());
                let before_digit = i > 0 && bytes[i - 1].is_ascii_digit();
                if three && ends && before_digit {
                    if sep == ',' {
                        comma += 1;
                    } else {
                        spaced += 1;
                    }
                }
            }
        }
    }
    if comma == 0 && spaced == 0 {
        NumberFormat::Plain
    } else if comma >= spaced {
        NumberFormat::GroupComma
    } else {
        NumberFormat::DecimalComma
    }
}

/// Column names that conventionally hold a coordinate, most specific
/// first so `longitude` beats a bare `x` in a file carrying both.
const X_NAMES: [&str; 6] = ["longitude", "lon", "lng", "long", "x", "easting"];
const Y_NAMES: [&str; 5] = ["latitude", "lat", "y", "northing", "north"];

/// Propose a coordinate pair, and the CRS only when the values settle it.
///
/// Names find the columns; the values decide the CRS. Degrees fit inside
/// ±180 and ±90, and a projected grid does not — a national grid in
/// metres runs to the hundreds of thousands. Guessing WGS84 for those
/// would put the data a continent away from where it belongs and look
/// like a projection bug, so an EPSG of 0 means "the user must say".
fn suggest_points(
    plan: &[ColumnPlan],
    cols: &[Vec<String>],
    fmt: NumberFormat,
) -> Option<GeometryPlan> {
    let find = |wanted: &[&str]| -> Option<usize> {
        wanted.iter().find_map(|w| {
            plan.iter()
                .position(|c| c.name == *w || c.name.trim_start_matches('_') == *w)
        })
    };
    let (xi, yi) = (find(&X_NAMES)?, find(&Y_NAMES)?);
    if xi == yi {
        return None;
    }
    let numeric = |i: usize| {
        cols.get(i).is_some_and(|v| {
            let n: Vec<&String> = v.iter().filter(|s| !s.trim().is_empty()).collect();
            !n.is_empty() && n.iter().all(|s| parses_as(s, ColType::Float, fmt))
        })
    };
    if !numeric(xi) || !numeric(yi) {
        return None;
    }
    let within = |i: usize, limit: f64| {
        cols[i]
            .iter()
            .filter(|s| !s.trim().is_empty())
            .filter_map(|s| normalize_number(s, fmt).parse::<f64>().ok())
            .all(|v| v.abs() <= limit)
    };
    Some(GeometryPlan::Points {
        x: plan[xi].name.clone(),
        y: plan[yi].name.clone(),
        epsg: if within(xi, 180.0) && within(yi, 90.0) { 4326 } else { 0 },
    })
}

/// Read a sample and propose a plan.
pub fn inspect(source: &Source) -> Result<Preview, String> {
    let csv = is_csv(source);
    if !csv {
        reject_geoparquet(source)?;
    }
    let mut plan = ImportPlan {
        delimiter: if csv { detect_delimiter(source) } else { b',' },
        has_header: true,
        numbers: NumberFormat::Plain,
        geometry: GeometryPlan::None,
        columns: Vec::new(),
    };
    let (names, cols) = sample_text(source, &plan)?;
    if csv {
        plan.numbers = detect_number_format(&cols);
    }
    // Parquet declares its types; only CSV needs them guessed.
    let declared: Option<Vec<ColType>> = if csv {
        None
    } else {
        Some(
            read_parquet(source, 0, &[])?
                .0
                .fields()
                .iter()
                .map(|f| ColType::of(f.data_type()))
                .collect(),
        )
    };

    let mut sanitized: Vec<String> = names.iter().map(|n| sanitize(n)).collect();
    dedupe(&mut sanitized);

    let mut columns = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let values = cols.get(i).cloned().unwrap_or_default();
        let ty = match &declared {
            Some(d) => d.get(i).copied().unwrap_or(ColType::Text),
            None => suggest_type(&values, plan.numbers),
        };
        let (bad, bad_examples) = bad_values(&values, ty, plan.numbers);
        columns.push(ColumnPreview {
            inferred: ty,
            samples: distinct_samples(&values),
            bad,
            bad_examples,
        });
        plan.columns.push(ColumnPlan {
            source_name: name.clone(),
            name: sanitized[i].clone(),
            ty,
            include: true,
        });
    }
    if plan.columns.is_empty() {
        // A remote source that was never length-probed reads as empty
        // rather than failing, and an import dialog with no columns in it
        // says nothing about why.
        return Err(format!(
            "{}: no columns found. If it is remote, it may not have been \
             reachable; if it is local, it may not be tabular.",
            source.name(),
        ));
    }
    let sampled_rows = cols.first().map(Vec::len).unwrap_or(0);
    if let Some(g) = suggest_points(&plan.columns, &cols, plan.numbers) {
        plan.geometry = g;
    }
    Ok(Preview {
        plan,
        columns,
        sampled_rows,
        typed_source: !csv,
        sampled: true,
    })
}

/// What can be known about a source immediately.
///
/// For parquet that is the footer: every column name and its declared
/// type, with no values behind them. For CSV there is no such shortcut —
/// the types are only knowable by reading values — but a CSV is read
/// sequentially and the first rows arrive at once, so the full inspection
/// is already the fast one.
pub fn inspect_fast(source: &Source) -> Result<Preview, String> {
    if is_csv(source) {
        return inspect(source);
    }
    reject_geoparquet(source)?;
    let (schema, _) = read_parquet(source, 0, &[])?;
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    if names.is_empty() {
        return Err(format!("{}: no columns found.", source.name()));
    }
    let mut sanitized: Vec<String> = names.iter().map(|n| sanitize(n)).collect();
    dedupe(&mut sanitized);
    let columns = schema
        .fields()
        .iter()
        .map(|f| ColumnPreview {
            inferred: ColType::of(f.data_type()),
            samples: Vec::new(),
            bad: 0,
            bad_examples: Vec::new(),
        })
        .collect();
    let plan = ImportPlan {
        delimiter: b',',
        has_header: true,
        numbers: NumberFormat::Plain,
        geometry: GeometryPlan::None,
        columns: names
            .iter()
            .zip(&sanitized)
            .zip(schema.fields())
            .map(|((source_name, name), f)| ColumnPlan {
                source_name: source_name.clone(),
                name: name.clone(),
                ty: ColType::of(f.data_type()),
                include: true,
            })
            .collect(),
    };
    Ok(Preview {
        plan,
        columns,
        sampled_rows: 0,
        typed_source: true,
        sampled: false,
    })
}

/// The values behind the columns, and a coordinate guess if they suggest
/// one. The slow half of [`inspect_fast`], run once the dialog is up.
pub fn sample_columns(
    source: &Source,
    plan: &ImportPlan,
) -> Result<(Vec<ColumnPreview>, usize, Option<GeometryPlan>), String> {
    let (_, cols) = sample_text(source, plan)?;
    let columns: Vec<ColumnPreview> = plan
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let values = cols.get(i).cloned().unwrap_or_default();
            let (bad, bad_examples) = bad_values(&values, c.ty, plan.numbers);
            ColumnPreview {
                inferred: c.ty,
                samples: distinct_samples(&values),
                bad,
                bad_examples,
            }
        })
        .collect();
    let rows = cols.first().map(Vec::len).unwrap_or(0);
    Ok((columns, rows, suggest_points(&plan.columns, &cols, plan.numbers)))
}

/// Re-read the sample and recompute names, types and damage. Called when
/// the delimiter or the header setting changes, which redraws the columns
/// entirely.
pub fn reinspect(source: &Source, delimiter: u8, has_header: bool) -> Result<Preview, String> {
    let mut p = inspect(source)?;
    if p.typed_source || (p.plan.delimiter == delimiter && p.plan.has_header == has_header) {
        return Ok(p);
    }
    p.plan.delimiter = delimiter;
    p.plan.has_header = has_header;
    let (names, cols) = sample_text(source, &p.plan)?;
    p.plan.numbers = detect_number_format(&cols);
    let mut sanitized: Vec<String> = names.iter().map(|n| sanitize(n)).collect();
    dedupe(&mut sanitized);
    p.plan.columns.clear();
    p.columns.clear();
    for (i, name) in names.iter().enumerate() {
        let values = cols.get(i).cloned().unwrap_or_default();
        let ty = suggest_type(&values, p.plan.numbers);
        let (bad, bad_examples) = bad_values(&values, ty, p.plan.numbers);
        p.columns.push(ColumnPreview {
            inferred: ty,
            samples: distinct_samples(&values),
            bad,
            bad_examples,
        });
        p.plan.columns.push(ColumnPlan {
            source_name: name.clone(),
            name: sanitized[i].clone(),
            ty,
            include: true,
        });
    }
    p.sampled_rows = cols.first().map(Vec::len).unwrap_or(0);
    p.sampled = true;
    Ok(p)
}

/// Recompute what the current types would cost, after one was changed.
pub fn recheck(source: &Source, preview: &mut Preview) -> Result<(), String> {
    let (_, cols) = sample_text(source, &preview.plan)?;
    for (i, c) in preview.columns.iter_mut().enumerate() {
        let values = cols.get(i).cloned().unwrap_or_default();
        let (bad, examples) =
            bad_values(&values, preview.plan.columns[i].ty, preview.plan.numbers);
        c.bad = bad;
        c.bad_examples = examples;
    }
    Ok(())
}

fn distinct_samples(values: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for v in values {
        let v = v.trim();
        if v.is_empty() || out.iter().any(|s| s == v) {
            continue;
        }
        out.push(v.to_string());
        if out.len() == SAMPLE_VALUES {
            break;
        }
    }
    out
}

fn bad_values(values: &[String], ty: ColType, fmt: NumberFormat) -> (usize, Vec<String>) {
    let mut n = 0;
    let mut examples: Vec<String> = Vec::new();
    for v in values {
        if v.trim().is_empty() || parses_as(v, ty, fmt) {
            continue;
        }
        n += 1;
        let t = v.trim();
        if examples.len() < SAMPLE_VALUES && !examples.iter().any(|e| e == t) {
            examples.push(t.to_string());
        }
    }
    (n, examples)
}

// ---------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------

fn reject_geoparquet(source: &Source) -> Result<(), String> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(source.open()?)
        .map_err(|e| format!("{}: not a parquet file ({e})", source.name()))?;
    if builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .is_some_and(|kv| kv.iter().any(|e| e.key == "geo"))
    {
        return Err(format!(
            "{} is GeoParquet — open it as a layer (File → Open) and it is \
             queryable and joinable the same way, with a map presence too.",
            source.name(),
        ));
    }
    Ok(())
}

/// Read a parquet source. `max_rows` of 0 means schema only; `keep`
/// selects columns by index, or every column when empty.
///
/// The selection is pushed into the scan rather than applied after it.
/// Wide files are the reason: a census extract is ninety columns of ten
/// million rows, and reading all of them to keep three is the difference
/// between a table that opens and one that cannot.
fn read_parquet(
    source: &Source,
    max_rows: usize,
    keep: &[usize],
) -> Result<(SchemaRef, Vec<RecordBatch>), String> {
    use parquet::arrow::ProjectionMask;

    let mut builder = ParquetRecordBatchReaderBuilder::try_new(source.open()?)
        .map_err(|e| format!("{}: not a parquet file ({e})", source.name()))?;
    let meta = builder.metadata().clone();
    // Only what will actually be read counts against the cap.
    let declared: i64 = meta
        .row_groups()
        .iter()
        .map(|rg| {
            if keep.is_empty() {
                rg.total_byte_size()
            } else {
                keep.iter()
                    .filter_map(|&i| rg.columns().get(i))
                    .map(|c| c.uncompressed_size().max(0))
                    .sum()
            }
        })
        .sum();
    if max_rows == usize::MAX && declared > MAX_BYTES as i64 {
        return Err(over_cap(declared as usize));
    }
    let full = builder.schema().clone();
    if max_rows == 0 {
        return Ok((full, Vec::new()));
    }
    let schema: SchemaRef = if keep.is_empty() {
        full
    } else {
        let mask = ProjectionMask::roots(builder.parquet_schema(), keep.iter().copied());
        builder = builder.with_projection(mask);
        Arc::new(Schema::new(
            keep.iter()
                .filter_map(|&i| full.fields().get(i).map(|f| f.as_ref().clone()))
                .collect::<Vec<_>>(),
        ))
    };
    let reader = builder
        .with_batch_size(BATCH_SIZE)
        // Bounds the decode as well as the loop: a sample should not pay
        // for a whole row group of a wide file.
        .with_limit(max_rows.min(i32::MAX as usize))
        .build()
        .map_err(|e| format!("parquet read error: {e}"))?;
    let mut batches = Vec::new();
    let mut seen = 0usize;
    for b in reader {
        let b = b.map_err(|e| format!("parquet decode error: {e}"))?;
        seen += b.num_rows();
        batches.push(b);
        if seen >= max_rows {
            break;
        }
    }
    Ok((schema, batches))
}

fn over_cap(bytes: usize) -> String {
    format!(
        "{} of columns — over the {} an attribute table may hold. \
         Attribute tables are read whole (there is no geometry to page them \
         in by), so narrow it down first.",
        super::info::fmt_bytes(bytes as u64),
        super::info::fmt_bytes(MAX_BYTES as u64),
    )
}

/// Cast every column of `batch` to text, so one code path handles values
/// whatever the source declared them to be.
fn to_text(batch: &RecordBatch, schema: &SchemaRef) -> Result<RecordBatch, String> {
    let cols: Result<Vec<ArrayRef>, String> = batch
        .columns()
        .iter()
        .map(|c| {
            cast_with_options(c, &DataType::Utf8, &safe()).map_err(|e| format!("{e}"))
        })
        .collect();
    RecordBatch::try_new(Arc::clone(schema), cols?).map_err(|e| format!("reading table: {e}"))
}

/// A value that does not fit becomes NULL rather than failing the import.
/// The count is what the user is told afterwards.
fn safe() -> CastOptions<'static> {
    CastOptions {
        safe: true,
        ..Default::default()
    }
}

/// Rewrite a text column into the digits-and-a-dot form arrow's cast
/// understands. A no-op unless the column is headed for a number and the
/// file groups its digits.
fn numeric_text(col: &ArrayRef, ty: ColType, fmt: NumberFormat) -> ArrayRef {
    use arrow::array::StringBuilder;
    if fmt == NumberFormat::Plain || !matches!(ty, ColType::Integer | ColType::Float) {
        return Arc::clone(col);
    }
    let Some(a) = col.as_any().downcast_ref::<StringArray>() else {
        return Arc::clone(col);
    };
    let mut b = StringBuilder::with_capacity(a.len(), a.len() * 8);
    for i in 0..a.len() {
        if a.is_null(i) {
            b.append_null();
        } else {
            b.append_value(normalize_number(a.value(i), fmt).as_ref());
        }
    }
    Arc::new(b.finish())
}

/// Apply a plan: read as text, cast to the chosen types, keep the
/// included columns.
pub fn import(source: &Source, plan: &ImportPlan) -> Result<AttrData, String> {
    let keep: Vec<(usize, &ColumnPlan)> = plan
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.include)
        .collect();
    if keep.is_empty() {
        return Err("no columns selected — an empty table has nothing to join on".into());
    }
    let idx: Vec<usize> = keep.iter().map(|(i, _)| *i).collect();
    // Both readers hand back the kept columns, in source order, so the
    // k-th plan entry is the k-th column of every batch.
    let text_batches = if is_csv(source) {
        read_csv_text(source, plan)?
    } else {
        read_parquet_text(source, &idx)?
    };
    let out_schema: SchemaRef = Arc::new(Schema::new(
        keep.iter()
            .map(|(_, c)| Field::new(&c.name, c.ty.arrow(), true))
            .collect::<Vec<_>>(),
    ));

    let mut nulled = vec![0usize; keep.len()];
    let mut batches = Vec::with_capacity(text_batches.len());
    let mut bytes = 0usize;
    for batch in &text_batches {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(keep.len());
        for (k, (_, c)) in keep.iter().enumerate() {
            // CSV is read whole (there is no projection to push down), so
            // its batches still carry every column.
            let src = batch.column(if is_csv(source) { keep[k].0 } else { k });
            let before = src.null_count();
            // Grouping separators are removed before the cast, not by it:
            // arrow parses `3739`, never `3,739`.
            let src = numeric_text(src, c.ty, plan.numbers);
            let out = cast_with_options(&src, &c.ty.arrow(), &safe())
                .map_err(|e| format!("{}: {e}", c.name))?;
            nulled[k] += out.null_count().saturating_sub(before);
            cols.push(out);
        }
        let b = RecordBatch::try_new(Arc::clone(&out_schema), cols)
            .map_err(|e| format!("building table: {e}"))?;
        bytes += b.get_array_memory_size();
        if bytes > MAX_BYTES {
            return Err(over_cap(bytes));
        }
        batches.push(b);
    }
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    Ok(AttrData {
        schema: out_schema,
        batches,
        rows,
        bytes,
        nulled: keep
            .iter()
            .zip(&nulled)
            .filter(|(_, n)| **n > 0)
            .map(|((_, c), n)| (c.name.clone(), *n))
            .collect(),
    })
}

/// Write an imported table to parquet.
///
/// Plain parquet, no `geo` metadata: this has no geometry, and claiming
/// otherwise would make it refuse to open as a table next time. Its value
/// is that the work done in the import dialog — the types, the names, the
/// columns kept — is now in a file that declares them, so reopening it
/// needs no inference and no dialog decisions at all.
pub fn write_parquet(path: &std::path::Path, data: &AttrData) -> Result<(), String> {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::WriterProperties;

    let file = std::fs::File::create(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut w = ArrowWriter::try_new(file, Arc::clone(&data.schema), Some(props))
        .map_err(|e| format!("parquet writer: {e}"))?;
    for b in &data.batches {
        w.write(b).map_err(|e| format!("parquet write: {e}"))?;
    }
    w.close().map_err(|e| format!("parquet close: {e}"))?;
    Ok(())
}

/// Write an imported table as GeoParquet, one point per row.
///
/// The columns keep the types the dialog chose; a `geometry` column of
/// WKB points is added from `x` and `y`. A row missing either coordinate
/// gets a null geometry rather than a point at the origin — the Gulf of
/// Guinea is where bad coordinate handling goes to be noticed, and it is
/// better not to send anything there.
///
/// Returns the number of rows that got a geometry.
pub fn write_points(
    path: &std::path::Path,
    data: &AttrData,
    x: &str,
    y: &str,
    crs: &super::crs::Crs,
) -> Result<usize, String> {
    use arrow::array::{BinaryBuilder, Float64Array};

    let xi = data
        .schema
        .index_of(x)
        .map_err(|_| format!("no column named {x}"))?;
    let yi = data
        .schema
        .index_of(y)
        .map_err(|_| format!("no column named {y}"))?;

    let mut fields: Vec<Field> = data
        .schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let geom_col = fields.len();
    fields.push(Field::new("geometry", DataType::Binary, true));
    let out: SchemaRef = Arc::new(Schema::new(fields));

    let mut w = crate::sql::export::StreamWriter::new(path, &out, geom_col, crs)?;
    let mut placed = 0usize;
    let opts = wkb::writer::WriteOptions::default();
    for batch in &data.batches {
        let num = |i: usize| -> Result<Float64Array, String> {
            let c = cast_with_options(batch.column(i), &DataType::Float64, &safe())
                .map_err(|e| format!("{e}"))?;
            c.as_any()
                .downcast_ref::<Float64Array>()
                .cloned()
                .ok_or_else(|| "coordinate column is not numeric".to_string())
        };
        let (xs, ys) = (num(xi)?, num(yi)?);
        let mut g = BinaryBuilder::new();
        let mut buf: Vec<u8> = Vec::new();
        for r in 0..batch.num_rows() {
            if xs.is_null(r) || ys.is_null(r) {
                g.append_null();
                continue;
            }
            let (px, py) = (xs.value(r), ys.value(r));
            if !px.is_finite() || !py.is_finite() {
                g.append_null();
                continue;
            }
            buf.clear();
            let point = geo_types::Geometry::Point(geo_types::Point::new(px, py));
            wkb::writer::write_geometry(&mut buf, &point, &opts)
                .map_err(|e| format!("wkb encode: {e}"))?;
            g.append_value(&buf);
            placed += 1;
        }
        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        cols.push(Arc::new(g.finish()));
        let b = RecordBatch::try_new(Arc::clone(&out), cols)
            .map_err(|e| format!("building layer: {e}"))?;
        w.write(&b)?;
    }
    w.finish()?;
    Ok(placed)
}

/// Every row, as text, in the source's column order.
fn read_csv_text(source: &Source, plan: &ImportPlan) -> Result<Vec<RecordBatch>, String> {
    let names: Vec<String> = plan.columns.iter().map(|c| c.source_name.clone()).collect();
    let rdr = arrow::csv::ReaderBuilder::new(text_schema(&names))
        .with_format(csv_format(plan.delimiter, plan.has_header))
        .with_batch_size(BATCH_SIZE)
        .build(whole(source)?)
        .map_err(|e| format!("{}: cannot read as CSV ({e})", source.name()))?;
    let mut out = Vec::new();
    let mut bytes = 0usize;
    for b in rdr {
        let b = b.map_err(|e| format!("{}: CSV parse error ({e})", source.name()))?;
        bytes += b.get_array_memory_size();
        if bytes > MAX_BYTES {
            return Err(over_cap(bytes));
        }
        out.push(b);
    }
    Ok(out)
}

fn read_parquet_text(source: &Source, keep: &[usize]) -> Result<Vec<RecordBatch>, String> {
    let (schema, batches) = read_parquet(source, usize::MAX, keep)?;
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let text = text_schema(&names);
    batches.iter().map(|b| to_text(b, &text)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str, body: &str) -> Source {
        let dir = std::env::temp_dir().join(format!("geopq_attrs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        Source::Local(p)
    }

    #[test]
    fn names_become_writable_sql_identifiers() {
        // The case that started this: `2025` parses as a number, so
        // `n.2025` is a syntax error and no amount of quoting habit helps.
        assert_eq!(sanitize("2025"), "_2025");
        assert_eq!(sanitize("Code INSEE"), "code_insee");
        assert_eq!(sanitize("Nom officiel (2024)"), "nom_officiel_2024");
        assert_eq!(sanitize("  spaced  "), "spaced");
        assert_eq!(sanitize("--"), "column");
        let mut names = vec!["a".to_string(), "a".to_string(), "b".to_string()];
        dedupe(&mut names);
        assert_eq!(names, vec!["a", "a_2", "b"]);
    }

    #[test]
    fn semicolon_files_are_detected() {
        // French public data is overwhelmingly semicolon separated, and
        // guessing a comma loads the whole file as a single column.
        let src = temp("fr.csv", "code;nom;2025\n01001;L'Abergement;12\n");
        let p = inspect(&src).unwrap();
        assert_eq!(p.plan.delimiter, b';');
        assert_eq!(p.plan.columns.len(), 3);
        // A comma inside a quoted field must not outvote the real one.
        let src = temp("q.csv", "code;label\n01;\"Paris, France\"\n02;\"Lyon, France\"\n");
        assert_eq!(inspect(&src).unwrap().plan.delimiter, b';');
    }

    #[test]
    fn a_stray_marker_is_reported_rather_than_retyping_the_column() {
        // One "NA" in a column of counts is what silently made a numeric
        // column text. Now it is proposed as text *and* the cost of
        // forcing it back to integer is on screen.
        let src = temp("counts.csv", "code,2025\n01001,12\nNA,NA\n01002,7\n");
        let mut p = inspect(&src).unwrap();
        let i = p.plan.columns.iter().position(|c| c.name == "_2025").unwrap();
        assert_eq!(p.plan.columns[i].ty, ColType::Text, "inference sees NA");
        assert_eq!(p.columns[i].bad, 0, "as text, nothing is lost");

        p.plan.columns[i].ty = ColType::Integer;
        recheck(&src, &mut p).unwrap();
        assert_eq!(p.columns[i].bad, 1, "one value would be lost");
        assert_eq!(p.columns[i].bad_examples, vec!["NA".to_string()]);

        let data = import(&src, &p.plan).unwrap();
        assert_eq!(data.rows, 3);
        assert_eq!(data.nulled, vec![("_2025".to_string(), 1)], "counted, not fatal");
        assert_eq!(
            data.schema.field_with_name("_2025").unwrap().data_type(),
            &DataType::Int64,
            "the choice was honoured",
        );
    }

    #[test]
    fn leading_zero_codes_are_suggested_as_text_and_can_be_overridden() {
        let src = temp("insee.csv", "code,pop\n01001,120\n01002,340\n");
        let mut p = inspect(&src).unwrap();
        assert_eq!(p.plan.columns[0].ty, ColType::Text, "01001 is a code");
        assert_eq!(p.plan.columns[1].ty, ColType::Integer, "120 is a number");
        let data = import(&src, &p.plan).unwrap();
        let col = data.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(col.value(0), "01001", "the zero survived");

        // The suggestion is a suggestion.
        p.plan.columns[0].ty = ColType::Integer;
        let data = import(&src, &p.plan).unwrap();
        assert_eq!(
            data.schema.field(0).data_type(),
            &DataType::Int64,
            "overridden without argument",
        );
        assert!(data.nulled.is_empty(), "they do parse, they just should not");
    }

    #[test]
    fn columns_can_be_left_out_and_renamed() {
        let src = temp("wide.csv", "Code INSEE,Nom,junk\n01001,Abergement,x\n");
        let mut p = inspect(&src).unwrap();
        assert_eq!(p.plan.columns[0].name, "code_insee");
        p.plan.columns[1].name = "commune".into();
        p.plan.columns[2].include = false;
        let data = import(&src, &p.plan).unwrap();
        let names: Vec<&str> = data
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["code_insee", "commune"]);

        // Nothing selected is a mistake worth naming rather than an empty
        // table that fails later at the join.
        p.plan.columns.iter_mut().for_each(|c| c.include = false);
        assert!(import(&src, &p.plan).is_err());
    }

    /// A column of 0 and 1 is a count until proven otherwise.
    ///
    /// Guessing boolean here is not a cosmetic mistake: it makes the
    /// column unjoinable against an integer key, and the planner's
    /// complaint says nothing about where the boolean came from.
    #[test]
    fn zeroes_and_ones_are_numbers_not_booleans() {
        let src = temp("flags.csv", "band,flag,n
0,true,3
1,false,4
");
        let p = inspect(&src).unwrap();
        assert_eq!(p.plan.columns[0].ty, ColType::Integer, "0/1 stays numeric");
        assert_eq!(p.plan.columns[1].ty, ColType::Boolean, "spelled out");
        assert_eq!(p.plan.columns[2].ty, ColType::Integer);
        // Chosen deliberately, 0/1 is still accepted as boolean.
        let mut p = p;
        p.plan.columns[0].ty = ColType::Boolean;
        recheck(&src, &mut p).unwrap();
        assert_eq!(p.columns[0].bad, 0, "0 and 1 are valid booleans when asked for");
        let data = import(&src, &p.plan).unwrap();
        assert!(data.nulled.is_empty(), "and the cast agrees with the preview");
    }

    /// Grouped thousands: `3,739` is a number, and casting it must give
    /// 3739 rather than five values quietly turning into NULL.
    #[test]
    fn grouped_thousands_are_read_as_numbers() {
        let src = temp("naiss.csv", "code,2025\n01001,\"3,739\"\n01002,\"1,276\"\n01003,\"10,871\"\n");
        let p = inspect(&src).unwrap();
        assert_eq!(p.plan.numbers, NumberFormat::GroupComma, "detected");
        let i = p.plan.columns.iter().position(|c| c.name == "_2025").unwrap();
        assert_eq!(p.plan.columns[i].ty, ColType::Integer, "a number, not text");
        assert_eq!(p.columns[i].bad, 0, "nothing is lost");

        let data = import(&src, &p.plan).unwrap();
        assert!(data.nulled.is_empty(), "{:?}", data.nulled);
        let col = data.batches[0]
            .column(i)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("integer column");
        assert_eq!(
            (0..3).map(|r| col.value(r)).collect::<Vec<_>>(),
            vec![3739, 1276, 10871],
            "grouping removed, not the digits",
        );
    }

    /// The comma means the opposite thing in the two conventions, so the
    /// same text has to give different numbers under each.
    #[test]
    fn a_comma_is_read_as_the_file_says() {
        use NumberFormat::{DecimalComma, GroupComma, Plain};
        assert_eq!(normalize_number("3,739", GroupComma), "3739");
        assert_eq!(normalize_number("3,739", DecimalComma), "3.739");
        assert_eq!(normalize_number("3,739", Plain), "3,739");
        // French exports group with a non-breaking space and may group
        // with a dot.
        assert_eq!(normalize_number("1\u{00A0}234,56", DecimalComma), "1234.56");
        assert_eq!(normalize_number("1.234,56", DecimalComma), "1234.56");
        assert_eq!(normalize_number("1 234.56", GroupComma), "1234.56");
        // Untouched when there is nothing to do.
        assert_eq!(normalize_number("42", GroupComma), "42");
    }

    /// Three digits after a separator is grouping; two is a decimal, and
    /// guessing wrong there moves a value by a factor of a thousand.
    #[test]
    fn only_three_digit_groups_are_taken_as_grouping() {
        let src = temp("amb.csv", "a\n\"3,5\"\n\"4,25\"\n");
        let p = inspect(&src).unwrap();
        assert_eq!(
            p.plan.numbers,
            NumberFormat::Plain,
            "ambiguous, so left alone for the user to say",
        );
        let src = temp("spc.csv", "a\n1 234\n5 678\n");
        assert_eq!(inspect(&src).unwrap().plan.numbers, NumberFormat::DecimalComma);
    }

    #[test]
    fn sample_values_are_shown_so_a_type_can_be_judged() {
        let src = temp("s.csv", "a\nfoo\nfoo\nbar\nbaz\nqux\n");
        let p = inspect(&src).unwrap();
        assert_eq!(p.columns[0].samples, vec!["foo", "bar", "baz"], "distinct");
        assert_eq!(p.sampled_rows, 5);
    }

    /// The written copy is the import's whole point of persistence: what
    /// comes back out of it must be what the dialog decided, and it must
    /// reopen as a table rather than needing the dialog again.
    #[test]
    fn the_written_copy_reopens_with_the_chosen_shape() {
        let src = temp(
            "wr.csv",
            "Code INSEE;Nom;2025;junk\n01001;Abergement;\"3,739\";x\n01002;Ambérieu;NA;y\n",
        );
        let mut p = inspect(&src).unwrap();
        assert_eq!(p.plan.delimiter, b';');
        assert_eq!(p.plan.numbers, NumberFormat::GroupComma);
        // The choices a user would make: drop the junk, force the count.
        let i = p.plan.columns.iter().position(|c| c.name == "_2025").unwrap();
        p.plan.columns[i].ty = ColType::Integer;
        p.plan.columns[3].include = false;
        p.plan.columns[1].name = "commune".into();
        let data = import(&src, &p.plan).unwrap();
        assert_eq!(data.nulled, vec![("_2025".to_string(), 1)], "the NA");

        let out = std::env::temp_dir()
            .join(format!("geopq_attrs_{}", std::process::id()))
            .join("written.parquet");
        write_parquet(&out, &data).unwrap();

        // Reopened, it needs no inference: the types are declared.
        let back = inspect(&Source::Local(out.clone())).unwrap();
        assert!(back.typed_source, "parquet declares its types");
        let names: Vec<&str> = back
            .plan
            .columns
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(names, vec!["code_insee", "commune", "_2025"], "junk stayed out");
        assert_eq!(back.plan.columns[0].ty, ColType::Text, "codes stayed text");
        assert_eq!(back.plan.columns[2].ty, ColType::Integer);

        let data2 = import(&Source::Local(out), &back.plan).unwrap();
        assert_eq!(data2.rows, 2);
        let col = data2.batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("integer");
        assert_eq!(col.value(0), 3739, "grouping survived the round trip");
        assert!(col.is_null(1), "and so did the NULL the NA became");
        let code = data2.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text");
        assert_eq!(code.value(0), "01001", "leading zero survived too");
    }

    /// Coordinate columns are found by name, but the CRS comes from the
    /// values: degrees fit inside ±180/±90 and a projected grid does not.
    #[test]
    fn coordinate_columns_are_found_and_the_crs_is_only_guessed_when_certain() {
        let src = temp("pts.csv", "name,lon,lat\na,2.35,48.85\nb,-71.06,42.36\n");
        let p = inspect(&src).unwrap();
        assert_eq!(
            p.plan.geometry,
            GeometryPlan::Points { x: "lon".into(), y: "lat".into(), epsg: 4326 },
        );

        // Lambert-93 metres: the same names, but guessing degrees would
        // put Toulouse in the Gulf of Guinea.
        let src = temp("l93.csv", "name,x,y\na,573000,6280000\nb,650000,6860000\n");
        let p = inspect(&src).unwrap();
        assert_eq!(
            p.plan.geometry,
            GeometryPlan::Points { x: "x".into(), y: "y".into(), epsg: 0 },
            "out of range, so the user has to say",
        );

        // No coordinate-ish names, no geometry proposed.
        let src = temp("plain.csv", "code,pop\n01001,120\n");
        assert_eq!(inspect(&src).unwrap().plan.geometry, GeometryPlan::None);
        // A longitude with no latitude is not a pair.
        let src = temp("half.csv", "name,lon\na,2.35\n");
        assert_eq!(inspect(&src).unwrap().plan.geometry, GeometryPlan::None);
    }

    /// A CSV of coordinates becomes a real GeoParquet layer: geometry
    /// written, CRS recorded, and rows with no coordinate kept rather
    /// than dropped or sent to the origin.
    #[test]
    fn a_csv_of_coordinates_becomes_a_geoparquet_layer() {
        let src = temp(
            "cities.csv",
            "name,lon,lat\nParis,2.35,48.85\nBoston,-71.06,42.36\nNowhere,,\n",
        );
        let p = inspect(&src).unwrap();
        let GeometryPlan::Points { x, y, epsg } = p.plan.geometry.clone() else {
            panic!("expected points");
        };
        assert_eq!(epsg, 4326);
        let data = import(&src, &p.plan).unwrap();
        assert_eq!(data.rows, 3);

        let out = std::env::temp_dir()
            .join(format!("geopq_attrs_{}", std::process::id()))
            .join("cities.parquet");
        let crs = crate::data::crs::Crs::from_epsg(epsg).unwrap();
        let placed = write_points(&out, &data, &x, &y, &crs).unwrap();
        assert_eq!(placed, 2, "the row with no coordinates got no point");

        // It opens as a layer through the ordinary store path, which is
        // the whole claim: this is GeoParquet, not a table with numbers.
        let (store, back_crs, _, _) =
            crate::data::loader::open_store_for_test(&out).expect("opens as a layer");
        assert_eq!(store.total_rows(), 3, "every row is kept");
        assert_eq!(back_crs.epsg, Some(4326), "CRS recorded in the file");
        let geoms = store.fetch_geoms(&[0, 1, 2]).unwrap();
        let paris = geoms[0].1.clone().expect("Paris has a point");
        match paris {
            geo_types::Geometry::Point(pt) => {
                assert!((pt.x() - 2.35).abs() < 1e-9 && (pt.y() - 48.85).abs() < 1e-9);
            }
            other => panic!("expected a point, got {other:?}"),
        }
        assert!(geoms[2].1.is_none(), "the coordinate-less row has no geometry");
        // The attributes came along.
        assert!(
            store.schema.fields().iter().any(|f| f.name() == "name"),
            "columns survive alongside the geometry",
        );
    }

    /// Remote sources go through the same reader the layer path uses, so
    /// a CSV or parquet on HTTPS or S3 imports like a local one.
    ///
    /// The catch this pins: a remote source carries no length until it is
    /// probed, and an unprobed one reads as an empty file — no error, no
    /// columns, an import dialog with nothing in it. Resolving first is
    /// the fix; failing loudly is the backstop.
    #[test]
    fn remote_sources_import_once_resolved() {
        let dir = std::env::temp_dir().join(format!("geopq_remote_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("cities.csv"),
            "name,lon,lat\nParis,2.35,48.85\nBoston,-71.06,42.36\n",
        )
        .unwrap();
        std::fs::write(dir.join("fr.csv"), "code;pop\n01001;120\n02002;340\n").unwrap();
        let base = crate::data::source::testserver::spawn_dir(dir);

        let raw = Source::Remote {
            url: format!("{base}/cities.csv"),
            len: 0,
        };
        let err = inspect(&raw).expect_err("an unprobed remote has no readable length");
        assert!(err.contains("no columns found"), "{err}");

        let src = raw.resolve().expect("length probe");
        let p = inspect(&src).expect("inspects over HTTP");
        assert_eq!(
            p.plan.geometry,
            GeometryPlan::Points { x: "lon".into(), y: "lat".into(), epsg: 4326 },
            "coordinates are found over the network too",
        );
        assert_eq!(import(&src, &p.plan).unwrap().rows, 2);

        // Separator detection reads the first bytes over the network the
        // same way it reads them off a disk.
        let src = Source::Remote {
            url: format!("{base}/fr.csv"),
            len: 0,
        }
        .resolve()
        .unwrap();
        let p = inspect(&src).unwrap();
        assert_eq!(p.plan.delimiter, b';');
        assert_eq!(p.plan.columns[0].ty, ColType::Text, "leading zeros kept");
        assert_eq!(import(&src, &p.plan).unwrap().rows, 2);
    }

    /// Names and types come from the footer and cost one small read; the
    /// values behind them cost a page of every column. Splitting the two
    /// is what puts the dialog on screen in a second rather than half a
    /// minute on a wide remote file.
    #[test]
    fn a_parquet_gives_its_columns_before_its_values() {
        use arrow::array::{Int64Array, StringArray};
        use parquet::arrow::ArrowWriter;
        let dir = std::env::temp_dir().join(format!("geopq_attrs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two_phase.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("Code INSEE", DataType::Utf8, false),
            Field::new("Pop", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["01001", "01002"])),
                Arc::new(Int64Array::from(vec![120, 340])),
            ],
        )
        .unwrap();
        let mut w =
            ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let src = Source::Local(path);
        let fast = inspect_fast(&src).unwrap();
        assert!(!fast.sampled, "values have not been read yet");
        assert_eq!(fast.sampled_rows, 0);
        // Everything needed to choose columns is already right.
        let names: Vec<&str> = fast.plan.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["code_insee", "pop"], "renamed from the footer");
        assert_eq!(fast.plan.columns[0].ty, ColType::Text);
        assert_eq!(fast.plan.columns[1].ty, ColType::Integer);
        assert!(fast.columns.iter().all(|c| c.samples.is_empty()));

        // And the second pass fills in what it could not know.
        let (cols, rows, _) = sample_columns(&src, &fast.plan).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols[0].samples, vec!["01001", "01002"]);
        assert_eq!(cols.len(), fast.columns.len(), "same shape, so it can be swapped in");

        // Importing straight off the fast pass works: the plan is whole
        // without the samples, which is what makes showing it early safe.
        let data = import(&src, &fast.plan).unwrap();
        assert_eq!(data.rows, 2);
        assert!(data.nulled.is_empty());
    }

    #[test]
    fn geoparquet_is_refused_with_a_way_forward() {
        let fixture = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/polygons_5k_l93.parquet"
        ));
        if !fixture.exists() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let err = inspect(&Source::Local(fixture)).expect_err("GeoParquet is not a table");
        assert!(err.contains("GeoParquet"), "{err}");
        assert!(err.contains("File → Open"), "{err}");
    }

    #[test]
    fn plain_parquet_keeps_its_declared_types() {
        use arrow::array::{Float64Array, Int64Array};
        use parquet::arrow::ArrowWriter;
        let dir = std::env::temp_dir().join(format!("geopq_attrs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plain.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("Id", DataType::Int64, false),
            Field::new("Value", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(Float64Array::from(vec![1.5, 2.5])),
            ],
        )
        .unwrap();
        let mut w =
            ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let src = Source::Local(path);
        let p = inspect(&src).unwrap();
        assert!(p.typed_source, "parquet declares its types");
        // Renaming still applies: a parquet can have awkward names too.
        assert_eq!(p.plan.columns[0].name, "id");
        assert_eq!(p.plan.columns[0].ty, ColType::Integer);
        assert_eq!(p.plan.columns[1].ty, ColType::Float);
        let data = import(&src, &p.plan).unwrap();
        assert_eq!(data.rows, 2);
        assert!(data.nulled.is_empty());
    }
}

