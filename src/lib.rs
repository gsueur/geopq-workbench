//! Shared library crate behind both binaries: the `geopq-workbench` GUI
//! (`src/main.rs`) and the `geopq-cli` batch converter (`src/bin/geopq-cli.rs`).
//! The CLI only touches `data`; the rest exists so the GUI binary keeps
//! working unchanged as a thin `main()` over this crate.

pub mod app;
pub mod context;
pub mod cookbook;
pub mod data;
pub mod map;
pub mod picking;
pub mod sql;
pub mod theme;
