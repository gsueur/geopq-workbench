#!/usr/bin/env bash
# Regenerate the test fixtures (not committed: ~600 MB of generated parquet).
#
# Requirements: duckdb CLI (with the spatial extension available) and uv
# (used to run pyarrow for GeoParquet metadata patching).
#
# Usage:
#   ./regenerate.sh                       # synthetic fixtures only
#   ./regenerate.sh /path/to/parcels.parquet
#       also builds parcels_hilbert.parquet (Hilbert-sorted, GeoParquet 1.1
#       covering) from a MassGIS-style parcels file (needs geometry,
#       LOC_ID, CITY, TOTAL_VAL columns in EPSG:26986), and
#       parcels_cogp.parquet from it when the `cogp` CLI is installed.
set -euo pipefail
cd "$(dirname "$0")"

command -v duckdb >/dev/null || { echo "duckdb CLI required"; exit 1; }
command -v uv >/dev/null || { echo "uv required (for pyarrow)"; exit 1; }

echo "== points_1m_wgs84.parquet (1M random points, CRS84) =="
duckdb -c "
INSTALL spatial; LOAD spatial;
COPY (
  SELECT ST_Point(-5 + 15*random(), 41 + 10*random()) AS geometry,
         (random()*1000)::INT AS value,
         'pt_' || row_number() OVER () AS name
  FROM range(1000000)
) TO 'points_1m_wgs84.parquet' (FORMAT PARQUET);"

echo "== points_3m75.parquet (3.75M points benchmark) =="
duckdb -c "
LOAD spatial;
COPY (
  SELECT ST_Point(-5 + 15*random(), 41 + 10*random()) AS geometry,
         (random()*100000)::INT AS parcel_id, (random()*1000)::DOUBLE AS area,
         'zone_' || (random()*20)::INT AS zone
  FROM range(3750000)
) TO 'points_3m75.parquet' (FORMAT PARQUET);"

echo "== lines_20k_wgs84.parquet =="
duckdb -c "
LOAD spatial;
COPY (
  SELECT ST_MakeLine([ST_Point(2+random(), 48+random()),
                      ST_Point(2+random(), 48+random()),
                      ST_Point(2+random(), 48+random())]) AS geometry,
         row_number() OVER () AS seg_id
  FROM range(20000)
) TO 'lines_20k_wgs84.parquet' (FORMAT PARQUET);"

echo "== polygons_5k_l93.parquet (EPSG:2154, PROJJSON patched) =="
duckdb -c "
LOAD spatial;
COPY (
  SELECT ST_Buffer(ST_Point(400000 + 500000*random(), 6300000 + 500000*random()),
                   5000 + 15000*random(), 8) AS geometry,
         (random()*100)::INT AS score,
         'poly_' || row_number() OVER () AS name
  FROM range(5000)
) TO 'polygons_5k_l93_nocrs.parquet' (FORMAT PARQUET);"
uv run --with pyarrow python3 - <<'EOF'
import pyarrow.parquet as pq, json
t = pq.read_table('polygons_5k_l93_nocrs.parquet')
geo = {
    "version": "1.1.0",
    "primary_column": "geometry",
    "columns": {"geometry": {
        "encoding": "WKB",
        "geometry_types": ["Polygon"],
        "crs": {"type": "ProjectedCRS", "name": "RGF93 v1 / Lambert-93",
                "id": {"authority": "EPSG", "code": 2154}}
    }}
}
meta = dict(t.schema.metadata or {})
meta[b"geo"] = json.dumps(geo).encode()
pq.write_table(t.replace_schema_metadata(meta), 'polygons_5k_l93.parquet')
EOF
rm -f polygons_5k_l93_nocrs.parquet

if [[ $# -ge 1 && -f "$1" ]]; then
  SRC="$1"
  echo "== parcels_hilbert.parquet (from $SRC) =="
  duckdb -c "
LOAD spatial;
COPY (
  WITH ext AS (
    SELECT min(ST_XMin(geometry)) xa, min(ST_YMin(geometry)) ya,
           max(ST_XMax(geometry)) xb, max(ST_YMax(geometry)) yb
    FROM '$SRC'
  )
  SELECT p.geometry,
         {'xmin': ST_XMin(p.geometry), 'ymin': ST_YMin(p.geometry),
          'xmax': ST_XMax(p.geometry), 'ymax': ST_YMax(p.geometry)} AS bbox,
         p.LOC_ID, p.CITY, p.TOTAL_VAL
  FROM '$SRC' p, ext
  ORDER BY ST_Hilbert(p.geometry,
    {'min_x': ext.xa, 'min_y': ext.ya, 'max_x': ext.xb, 'max_y': ext.yb}::BOX_2D)
) TO 'parcels_hilbert_nocover.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE 131072);"
  uv run --with pyarrow python3 - <<'EOF'
import pyarrow.parquet as pq, json
t = pq.read_table('parcels_hilbert_nocover.parquet')
geo = {
    "version": "1.1.0",
    "primary_column": "geometry",
    "columns": {"geometry": {
        "encoding": "WKB",
        "geometry_types": ["Polygon", "MultiPolygon"],
        "crs": {"type": "ProjectedCRS", "name": "NAD83 / Massachusetts Mainland",
                "id": {"authority": "EPSG", "code": 26986}},
        "covering": {"bbox": {
            "xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
            "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"]}}
    }}
}
meta = dict(t.schema.metadata or {})
meta[b"geo"] = json.dumps(geo).encode()
pq.write_table(t.replace_schema_metadata(meta), 'parcels_hilbert.parquet',
               row_group_size=131072)
md = pq.read_metadata('parcels_hilbert.parquet')
print(f"parcels_hilbert.parquet: {md.num_rows} rows, {md.num_row_groups} row groups")
EOF
  rm -f parcels_hilbert_nocover.parquet

  # Cloud Optimized GeoParquet Profile: reorder the same features
  # coarse-to-fine across row groups and stamp the `cogp` block.
  #   cargo install --git https://github.com/Kanahiro/cloud-optimized-geoparquet cogp
  if command -v cogp >/dev/null; then
    echo "== parcels_cogp.parquet (COGP levels, default webmerc zooms) =="
    cogp convert parcels_hilbert.parquet parcels_cogp.parquet
    cogp validate parcels_cogp.parquet
  else
    echo "(cogp CLI not installed — skipping parcels_cogp.parquet;"
    echo " the COGP tests self-skip without it)"
  fi
else
  echo "(no parcels source given — skipping parcels_hilbert.parquet;"
  echo " pruning/covering tests will self-skip without it)"
fi

echo "done:"
ls -lh ./*.parquet
