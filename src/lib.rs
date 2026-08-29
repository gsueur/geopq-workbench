//! Shared library crate behind both binaries: the `geopq-workbench` GUI
//! (`src/main.rs`) and the `geopq-cli` batch converter (`src/bin/geopq-cli.rs`).
//! The CLI only touches `data`; the rest exists so the GUI binary keeps
//! working unchanged as a thin `main()` over this crate.
//!
//! Public is what a binary names: `data` for both, `app` and `map` for the
//! GUI's `main()`. Everything else the app builds on stays crate-private,
//! so the split adds a library boundary without also publishing an API
//! nothing outside this repo calls.

pub mod app;
pub(crate) mod context;
pub(crate) mod cookbook;
pub mod data;
pub mod map;
pub(crate) mod picking;
pub(crate) mod sql;
pub(crate) mod theme;
