#!/usr/bin/env bash
# Regenerate the numbers in _WIKI/reference/h3-partition-benchmark.md.
#
# Everything runs through the app's own code (optimizer, STAC writer,
# STAC reader, viewport planner, DataFusion table provider) so the
# measurements are of the shipping paths, not of a re-implementation.
# The benchmark lives in src/data/bench_h3.rs, behind #[ignore].
#
#   bench/h3_partitions.sh write     # write the variants (~2.5 GB)
#   bench/h3_partitions.sh kdtree    # optional: geoparquet-io KD-tree variant
#   bench/h3_partitions.sh measure   # the per-viewport table
#   bench/h3_partitions.sh sql       # the pushdown table
#   bench/h3_partitions.sh all
#
# Overrides: GEOPQ_BENCH_INPUT (source parquet), GEOPQ_BENCH_DIR (output root).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# DATA/ is gitignored and lives in the primary checkout; a worktree has
# none of its own, so fall back to the sibling rather than inventing an
# empty one next to the branch.
DATA="$ROOT/DATA"
[ -d "$DATA" ] || DATA="$(dirname "$ROOT")/GEOPQ_WORKBENCH/DATA"

: "${GEOPQ_BENCH_INPUT:=$DATA/L3_TAXPAR_POLY_ASSESS_WEST_optimized.parquet}"
: "${GEOPQ_BENCH_DIR:=$DATA/bench}"
export GEOPQ_BENCH_INPUT GEOPQ_BENCH_DIR

run_test() {
  cargo test --release "$1" -- --ignored --nocapture
}

write_variants() { run_test bench_write_variants; }
measure()        { run_test bench_measure; }
sql()            { run_test bench_sql_pushdown; }

# KD-tree is the scheme the OGC distribution guidance leads with and the
# one this optimizer deliberately does not implement, so it comes from
# geoparquet-io. Its parts are then re-optimized with the same options as
# every other variant (zstd, Hilbert, covering, 65536-row groups) so the
# comparison is between partition schemes and not between writers.
# $1: partition count (power of 2), or "auto" for gpio's own choice.
# $2: variant directory name.
kdtree() {
  local spec="${1:-16}" name="${2:-kdtree}"
  local raw="$GEOPQ_BENCH_DIR/_kdtree_raw"
  local out="$GEOPQ_BENCH_DIR/$name"
  local arg
  if [ "$spec" = auto ]; then arg=(--auto 250000); else arg=(--partitions "$spec"); fi
  rm -rf "$raw" "$out"
  uvx --from geoparquet-io==1.2.0 gpio partition kdtree \
      "$GEOPQ_BENCH_INPUT" "$raw" "${arg[@]}"
  mkdir -p "$out"
  GEOPQ_BENCH_KDTREE_RAW="$raw" GEOPQ_BENCH_KDTREE_OUT="$out" \
      run_test bench_finish_kdtree
  rm -rf "$raw"
}

case "${1:-all}" in
  write)  write_variants ;;
  kdtree) shift; kdtree "$@" ;;
  measure) measure ;;
  sql)    sql ;;
  all)    write_variants
          kdtree 16 kdtree            || echo "kdtree skipped"
          kdtree auto kdtree_auto512  || echo "kdtree_auto skipped"
          measure; sql ;;
  *) echo "usage: $0 {write|kdtree|measure|sql|all}" >&2; exit 2 ;;
esac
