//! In-app SQL over loaded GeoParquet layers.
//!
//! DataFusion provides the parser/planner/executor; every loaded layer is
//! exposed as a table streaming straight out of its `FeatureStore` (local,
//! HTTP or S3 alike), and the spatial vocabulary is our own set of ST_*
//! functions built on geo-types with WKB in/out: scalar ones in `udf`,
//! aggregates in `agg`.

pub mod agg;
pub mod console;
pub mod engine;
pub mod export;
pub mod table;
pub mod udf;

