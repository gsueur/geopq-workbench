#!/usr/bin/env python3
"""Convert a Natural Earth coastline GeoJSON to the app's NEC1 binary.

Format (little endian): "NEC1" u32 line_count { u32 n, n x (f32 lon, f32 lat) }

Usage:
    uv run python3 assets/make_coastline_bin.py 10m   # -> assets/ne_10m_coastline.bin
    uv run python3 assets/make_coastline_bin.py 50m   # -> assets/ne_50m_coastline.bin

Data: Natural Earth (public domain), via the official mirror repo.
"""

import json
import struct
import sys
import urllib.request
from pathlib import Path

SCALE = sys.argv[1] if len(sys.argv) > 1 else "10m"
URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    f"geojson/ne_{SCALE}_coastline.geojson"
)
OUT = Path(__file__).parent / f"ne_{SCALE}_coastline.bin"

print(f"fetching {URL}")
with urllib.request.urlopen(URL) as r:
    gj = json.load(r)

lines = []
for feat in gj["features"]:
    geom = feat["geometry"]
    if geom["type"] == "LineString":
        parts = [geom["coordinates"]]
    elif geom["type"] == "MultiLineString":
        parts = geom["coordinates"]
    else:
        continue
    for coords in parts:
        if len(coords) >= 2:
            lines.append([(float(x), float(y)) for x, y, *_ in coords])

buf = bytearray(b"NEC1")
buf += struct.pack("<I", len(lines))
pts = 0
for line in lines:
    buf += struct.pack("<I", len(line))
    for x, y in line:
        buf += struct.pack("<ff", x, y)
    pts += len(line)

OUT.write_bytes(buf)
print(f"{OUT}: {len(lines)} lines, {pts} points, {len(buf) / 1e6:.1f} MB")
