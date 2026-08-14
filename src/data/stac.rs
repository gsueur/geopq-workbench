//! STAC Collection metadata for published GeoParquet, per the upstream
//! [distributing-geoparquet best practices]: a `collection.json` beside
//! the data (single file) or at the dataset prefix (partitioned), with
//! `application/vnd.apache.parquet` assets and a WGS84 extent read from
//! the parquet footers. Deliberately collection-only: the document also
//! sketches per-part STAC Items for partitioned data, which can come
//! later without breaking anything written here.
//!
//! [distributing-geoparquet best practices]: https://github.com/opengeospatial/geoparquet/blob/main/format-specs/distributing-geoparquet.md

use std::path::{Path, PathBuf};

use parquet::file::reader::FileReader;
use serde_json::{json, Value};

use super::crs::Crs;

/// WGS84 lon/lat bbox of a data-CRS bbox: the border of the rect is
/// grid-sampled through the transform because a projected rect's edges
/// curve in lon/lat, so corners alone under-cover. None when nothing
/// transforms (broken CRS), which callers should surface as a world
/// extent rather than an invented one.
pub fn wgs84_bbox(b: [f64; 4], crs: &Crs) -> Option<[f64; 4]> {
    let wgs = super::crs::wgs84_cached();
    const N: usize = 8;
    let mut out = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut any = false;
    for i in 0..=N {
        for j in 0..=N {
            if i != 0 && i != N && j != 0 && j != N {
                continue; // border only
            }
            let x = b[0] + (b[2] - b[0]) * i as f64 / N as f64;
            let y = b[1] + (b[3] - b[1]) * j as f64 / N as f64;
            if let Ok((lon, lat)) = super::crs::transform_point(crs, wgs, x, y)
                && lon.is_finite()
                && lat.is_finite()
            {
                any = true;
                out[0] = out[0].min(lon);
                out[1] = out[1].min(lat);
                out[2] = out[2].max(lon);
                out[3] = out[3].max(lat);
            }
        }
    }
    any.then(|| {
        [
            out[0].clamp(-180.0, 180.0),
            out[1].clamp(-90.0, 90.0),
            out[2].clamp(-180.0, 180.0),
            out[3].clamp(-90.0, 90.0),
        ]
    })
}

/// Rows and data-CRS bbox of one parquet file, from the footer alone
/// (`geo` metadata bbox of the primary column).
fn file_facts(path: &Path) -> (u64, Option<[f64; 4]>) {
    let facts = || -> Option<(u64, Option<[f64; 4]>)> {
        let f = std::fs::File::open(path).ok()?;
        let r = parquet::file::reader::SerializedFileReader::new(f).ok()?;
        let md = r.metadata().file_metadata().clone();
        let rows = md.num_rows().max(0) as u64;
        let bbox = md
            .key_value_metadata()
            .and_then(|kv| kv.iter().find(|k| k.key == "geo"))
            .and_then(|k| k.value.clone())
            .and_then(|v| serde_json::from_str::<Value>(&v).ok())
            .and_then(|g| {
                let primary = g.get("primary_column")?.as_str()?.to_string();
                let arr = g.get("columns")?.get(&primary)?.get("bbox")?.as_array()?;
                let v: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
                (v.len() >= 4).then(|| [v[0], v[1], v[2], v[3]])
            });
        Some((rows, bbox))
    };
    facts().unwrap_or((0, None))
}

/// Every `.parquet` under `dir`, recursively (hive partitions nest),
/// sorted for a stable assets order.
fn parquet_parts(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "parquet") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// A collection id STAC tools accept: lowercase, `[a-z0-9._-]` only.
fn slug(name: &str) -> String {
    let s: String = name
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { "geoparquet".into() } else { s }
}

/// The STAC Collection document for a set of parquet parts.
/// `files`: (href relative to the collection.json, local path).
fn collection_doc(title: &str, files: &[(String, PathBuf)], crs: &Crs) -> Value {
    let mut rows = 0u64;
    let mut data_bbox: Option<[f64; 4]> = None;
    let mut assets = serde_json::Map::new();
    for (href, path) in files {
        let (r, b) = file_facts(path);
        rows += r;
        if let Some(b) = b {
            data_bbox = Some(match data_bbox {
                None => b,
                Some(u) => [u[0].min(b[0]), u[1].min(b[1]), u[2].max(b[2]), u[3].max(b[3])],
            });
        }
        let key = href.trim_start_matches("./").to_string();
        assets.insert(
            key,
            json!({
                "href": href,
                "type": "application/vnd.apache.parquet",
                "roles": ["data"],
            }),
        );
    }
    // Unknown extent is written as the world, never invented tighter.
    let bbox = data_bbox
        .and_then(|b| wgs84_bbox(b, crs))
        .unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    json!({
        "type": "Collection",
        "stac_version": "1.1.0",
        "id": slug(title),
        "title": title,
        "description": format!(
            "{title}: {rows} features in {} GeoParquet file{}.",
            files.len(),
            if files.len() == 1 { "" } else { "s" },
        ),
        "license": "other",
        "extent": {
            "spatial": { "bbox": [bbox] },
            "temporal": { "interval": [[null, null]] },
        },
        "links": [],
        "assets": Value::Object(assets),
    })
}

/// Write `collection.json` for an optimize output: inside `dst` when it
/// is a partitioned dataset directory (the tree upload then carries
/// it), beside it when it is a single file. Returns the written path.
pub fn write_for_output(dst: &Path, title: &str, crs: &Crs) -> Result<PathBuf, String> {
    let (files, out) = if dst.is_dir() {
        let parts = parquet_parts(dst);
        if parts.is_empty() {
            return Err("no parquet parts in the output".into());
        }
        let files = parts
            .into_iter()
            .map(|p| {
                let rel = p
                    .strip_prefix(dst)
                    .map(|r| format!("./{}", r.display()))
                    .unwrap_or_else(|_| p.display().to_string());
                (rel, p)
            })
            .collect::<Vec<_>>();
        (files, dst.join("collection.json"))
    } else {
        let name = dst
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or("output has no file name")?;
        (
            vec![(format!("./{name}"), dst.to_path_buf())],
            dst.with_file_name("collection.json"),
        )
    };
    write_doc(&out, title, &files, crs)?;
    Ok(out)
}

/// Same, for publishing an existing file as-is: the document describes
/// `file` but is written to `out` — never beside the user's source.
pub fn write_for_file_at(file: &Path, out: &Path, title: &str, crs: &Crs) -> Result<(), String> {
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or("file has no name")?;
    write_doc(out, title, &[(format!("./{name}"), file.to_path_buf())], crs)
}

fn write_doc(out: &Path, title: &str, files: &[(String, PathBuf)], crs: &Crs) -> Result<(), String> {
    let title = if title.is_empty() { "GeoParquet dataset" } else { title };
    let doc = collection_doc(title, files, crs);
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(out, text).map_err(|e| format!("write {}: {e}", out.display()))
}

/// The sibling `collection.json` object key of an `s3://…/file.parquet`
/// upload uri. None when the uri has no parent path to hang it on.
pub fn sibling_uri(uri: &str) -> Option<String> {
    let (head, tail) = uri.rsplit_once('/')?;
    if tail.is_empty() || head.ends_with('/') || head == "s3:/" {
        return None;
    }
    Some(format!("{head}/collection.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// A minimal parquet file whose footer carries `geo` metadata with a
    /// bbox; the columns don't matter, only the footer is read here.
    fn tiny_parquet(path: &Path, rows: usize, bbox: [f64; 4]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
        )
        .unwrap();
        let f = std::fs::File::create(path).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            "geo".to_string(),
            json!({
                "version": "1.1.0",
                "primary_column": "geometry",
                "columns": { "geometry": { "encoding": "WKB", "bbox": bbox } },
            })
            .to_string(),
        ));
        w.close().unwrap();
    }

    #[test]
    fn collection_covers_a_partitioned_tree_in_wgs84() {
        let dir = std::env::temp_dir().join("geopq_stac_tree");
        let _ = std::fs::remove_dir_all(&dir);
        // Lambert-93 bboxes (metric): two hive parts around France.
        tiny_parquet(&dir.join("dep=31/part-0.parquet"), 10, [500_000.0, 6_200_000.0, 600_000.0, 6_300_000.0]);
        tiny_parquet(&dir.join("dep=75/part-0.parquet"), 5, [640_000.0, 6_850_000.0, 660_000.0, 6_870_000.0]);

        let crs = Crs::from_epsg(2154).expect("Lambert-93");
        let out = write_for_output(&dir, "Parcelles Test", &crs).unwrap();
        assert_eq!(out, dir.join("collection.json"));
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();

        assert_eq!(doc["type"], "Collection");
        assert_eq!(doc["stac_version"], "1.1.0");
        assert_eq!(doc["id"], "parcelles-test");
        assert!(doc["description"].as_str().unwrap().contains("15 features"));
        assert_eq!(doc["license"], "other");
        let assets = doc["assets"].as_object().unwrap();
        assert_eq!(assets.len(), 2);
        let a = &assets["dep=31/part-0.parquet"];
        assert_eq!(a["href"], "./dep=31/part-0.parquet");
        assert_eq!(a["type"], "application/vnd.apache.parquet");
        // The union of the parts, in lon/lat: metric Lambert values would
        // be the symptom of a skipped transform.
        let b = doc["extent"]["spatial"]["bbox"][0].as_array().unwrap();
        let b: Vec<f64> = b.iter().map(|v| v.as_f64().unwrap()).collect();
        assert!(b[0] > -3.0 && b[2] < 6.0, "lon range around France: {b:?}");
        assert!(b[1] > 42.0 && b[3] < 50.0, "lat range around France: {b:?}");
        assert!(b[2] > b[0] && b[3] > b[1]);
        assert_eq!(doc["extent"]["temporal"]["interval"][0], json!([null, null]));
    }

    #[test]
    fn single_file_gets_a_sibling_collection() {
        let dir = std::env::temp_dir().join("geopq_stac_single");
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("roads.parquet");
        tiny_parquet(&file, 3, [-1.0, 46.0, 2.0, 48.0]);
        let out = write_for_output(&file, "Roads", &Crs::wgs84()).unwrap();
        assert_eq!(out, dir.join("collection.json"));
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(doc["assets"]["roads.parquet"]["href"], "./roads.parquet");
        // Already lon/lat: the extent is the bbox itself.
        let b = doc["extent"]["spatial"]["bbox"][0].as_array().unwrap();
        assert_eq!(b[0].as_f64().unwrap(), -1.0);
        assert_eq!(b[3].as_f64().unwrap(), 48.0);
    }

    #[test]
    fn sibling_uri_replaces_the_object_name() {
        assert_eq!(
            sibling_uri("s3://bucket/data/roads.parquet").as_deref(),
            Some("s3://bucket/data/collection.json")
        );
        assert_eq!(
            sibling_uri("s3://bucket/roads.parquet").as_deref(),
            Some("s3://bucket/collection.json")
        );
        assert_eq!(sibling_uri("s3://bucket/"), None);
    }
}
