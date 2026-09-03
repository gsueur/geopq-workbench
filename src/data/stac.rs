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

/// Rows, data-CRS bbox and derived flag of one parquet file, from the
/// footer alone (`geo` metadata bbox of the primary column, and the
/// `geopq:pyramid` entry an overview level carries).
struct FileFacts {
    rows: u64,
    bbox: Option<[f64; 4]>,
    /// True for an H3 pyramid overview part: the same features as the
    /// level below it, generalized. Counting its rows as data would
    /// report a dataset several times its own size.
    derived: bool,
}

fn file_facts(path: &Path) -> FileFacts {
    let facts = || -> Option<FileFacts> {
        let f = std::fs::File::open(path).ok()?;
        let r = parquet::file::reader::SerializedFileReader::new(f).ok()?;
        let md = r.metadata().file_metadata().clone();
        let rows = md.num_rows().max(0) as u64;
        let kv = |key: &str| -> Option<Value> {
            md.key_value_metadata()
                .and_then(|kv| kv.iter().find(|k| k.key == key))
                .and_then(|k| k.value.clone())
                .and_then(|v| serde_json::from_str::<Value>(&v).ok())
        };
        let bbox = kv("geo").and_then(|g| {
            let primary = g.get("primary_column")?.as_str()?.to_string();
            let arr = g.get("columns")?.get(&primary)?.get("bbox")?.as_array()?;
            let v: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
            (v.len() >= 4).then(|| [v[0], v[1], v[2], v[3]])
        });
        // The part says what it is: an overview file carries the key,
        // and deriving the answer from its `r<res>/` directory instead
        // would need the descriptor parsed and the path re-parsed.
        let derived = kv(super::pyramid::FILE_KEY)
            .and_then(|m| m.get("derived").and_then(Value::as_bool))
            .unwrap_or(false);
        Some(FileFacts { rows, bbox, derived })
    };
    facts().unwrap_or(FileFacts { rows: 0, bbox: None, derived: false })
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

/// The STAC Collection document for a set of parts.
/// `files`: (href relative to the collection.json, local path). A
/// non-parquet entry is a sidecar (the H3 pyramid descriptor) and is
/// described as metadata rather than data.
fn collection_doc(title: &str, files: &[(String, PathBuf)], crs: &Crs) -> Value {
    let mut rows = 0u64;
    let mut data_files = 0usize;
    let mut data_bbox: Option<[f64; 4]> = None;
    let mut assets = serde_json::Map::new();
    let mut descriptor_rows: Option<u64> = None;
    for (href, path) in files {
        let key = href.trim_start_matches("./").to_string();
        // The pyramid descriptor: a sidecar, and the one place that
        // knows how many source features the whole pyramid stands for.
        if path.extension().is_some_and(|x| x == "json") {
            if path.file_name().is_some_and(|n| n == super::pyramid::DESCRIPTOR) {
                descriptor_rows = std::fs::read_to_string(path)
                    .ok()
                    .and_then(|t| super::pyramid::Descriptor::parse(&t).ok())
                    .and_then(|d| d.rows);
            }
            assets.insert(
                key,
                json!({
                    "href": href,
                    "type": "application/json",
                    "roles": ["metadata"],
                }),
            );
            continue;
        }
        let f = file_facts(path);
        // Overview levels are the same features again, coarser: their
        // rows are not part of the dataset's feature count, and a reader
        // must be able to tell them from the leaf.
        if !f.derived {
            rows += f.rows;
            data_files += 1;
        }
        if let Some(b) = f.bbox {
            data_bbox = Some(match data_bbox {
                None => b,
                Some(u) => [u[0].min(b[0]), u[1].min(b[1]), u[2].max(b[2]), u[3].max(b[3])],
            });
        }
        let mut asset = json!({
            "href": href,
            "type": "application/vnd.apache.parquet",
            "roles": [if f.derived { "overview" } else { "data" }],
        });
        // Per-asset extent, same [w,s,e,n] convention as the collection's
        // (an extra asset field is valid STAC). Without it a reader can
        // only prune a partitioned dataset by the extent of the whole,
        // which is to say not at all: every part would claim every view.
        if let Some(wgs) = f.bbox.and_then(|b| wgs84_bbox(b, crs)) {
            asset["bbox"] = json!(wgs);
        }
        asset["rows"] = json!(f.rows);
        assets.insert(key, asset);
    }
    // The descriptor's own count wins where there is one: it is written
    // by the run that produced the leaf, and counting parquet rows
    // instead means counting every overview level again.
    let rows = descriptor_rows.unwrap_or(rows);
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
            "{title}: {rows} features in {data_files} GeoParquet file{}.",
            if data_files == 1 { "" } else { "s" },
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
        // An H3 pyramid's descriptor is what makes its parts readable as
        // a pyramid rather than as a pile of files, so it is published
        // with them as a metadata asset.
        let descriptor = dst.join(super::pyramid::DESCRIPTOR);
        let parts: Vec<PathBuf> = parts
            .into_iter()
            .chain(descriptor.is_file().then_some(descriptor))
            .collect();
        let files = parts
            .into_iter()
            .map(|p| {
                // Hrefs are URL paths whatever the platform: Windows
                // hands out backslashes here, and a collection published
                // with them is unreadable everywhere (same rule as
                // upload_tree's object keys).
                let rel = p
                    .strip_prefix(dst)
                    .map(|r| format!("./{}", r.display()))
                    .unwrap_or_else(|_| p.display().to_string())
                    .replace('\\', "/");
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

/// Merge our just-built collection into the one already published at
/// the destination: publishing a file into a prefix that has a
/// collection means adding a part to that dataset, not replacing it.
///
/// The existing document keeps its identity — id, title, description,
/// license, links — while our assets upsert by key and the spatial
/// extent becomes the union. Anything that is not a STAC Collection
/// with assets is refused: the sidecar found there is someone's file,
/// and overwriting a stranger's document is exactly what this exists
/// to stop.
pub fn merge_into(existing: &str, ours: &Value) -> Result<Value, String> {
    let mut base: Value = serde_json::from_str(existing).map_err(|_| {
        "the destination's collection.json is not JSON — not overwriting it".to_string()
    })?;
    if base.get("type").and_then(Value::as_str) != Some("Collection")
        || !base.get("assets").is_some_and(Value::is_object)
    {
        return Err(
            "the destination's collection.json is not a STAC Collection with assets — \
             not overwriting it"
                .into(),
        );
    }
    let bbox_of = |v: &Value| -> Option<[f64; 4]> {
        let b = v.pointer("/extent/spatial/bbox/0")?.as_array()?;
        let mut out = [0f64; 4];
        for (i, x) in b.iter().take(4).enumerate() {
            out[i] = x.as_f64()?;
        }
        (b.len() >= 4).then_some(out)
    };
    let union = match (bbox_of(&base), bbox_of(ours)) {
        (Some(a), Some(b)) => {
            Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])
        }
        (a, b) => a.or(b),
    };
    match (union, base.pointer_mut("/extent/spatial/bbox/0")) {
        (Some(u), Some(slot)) => *slot = json!(u),
        (Some(_), None) => {
            // The existing document has no extent to widen: take ours
            // wholesale rather than leaving the merged set undescribed.
            if let Some(e) = ours.get("extent") {
                base["extent"] = e.clone();
            }
        }
        _ => {}
    }
    if let (Some(dst), Some(src)) = (
        base.get_mut("assets").and_then(Value::as_object_mut),
        ours.get("assets").and_then(Value::as_object),
    ) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    Ok(base)
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
        assert_eq!(a["rows"], 10);
        // The union of the parts, in lon/lat: metric Lambert values would
        // be the symptom of a skipped transform.
        let b = doc["extent"]["spatial"]["bbox"][0].as_array().unwrap();
        let b: Vec<f64> = b.iter().map(|v| v.as_f64().unwrap()).collect();
        assert!(b[0] > -3.0 && b[2] < 6.0, "lon range around France: {b:?}");
        assert!(b[1] > 42.0 && b[3] < 50.0, "lat range around France: {b:?}");
        assert!(b[2] > b[0] && b[3] > b[1]);
        assert_eq!(doc["extent"]["temporal"]["interval"][0], json!([null, null]));

        // Each asset states its own extent, or a reader can only prune by
        // the whole: dep=31 is Toulouse, dep=75 is Paris, and neither may
        // come back as the union above.
        let bbox = |k: &str| -> Vec<f64> {
            assets[k]["bbox"]
                .as_array()
                .unwrap_or_else(|| panic!("{k} has no bbox"))
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        };
        let (t, p) = (bbox("dep=31/part-0.parquet"), bbox("dep=75/part-0.parquet"));
        assert!(t[3] < p[1], "Toulouse is south of Paris: {t:?} {p:?}");
        assert!(t[0] >= b[0] && p[2] <= b[2], "parts sit inside the extent");
        assert!(t[2] - t[0] < b[2] - b[0], "a part is narrower than the union");
    }

    /// The same, plus the `geopq:pyramid` entry an overview file
    /// carries: derived data, not source features.
    fn tiny_overview(path: &Path, rows: usize, res: u8, bbox: [f64; 4]) {
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
        w.append_key_value_metadata(parquet::file::metadata::KeyValue::new(
            super::super::pyramid::FILE_KEY.to_string(),
            json!({"res": res, "method": "simplify", "source_res": res + 1, "derived": true})
                .to_string(),
        ));
        w.close().unwrap();
    }

    /// An H3 pyramid is one dataset with generalized copies of itself
    /// beside the leaf. Walking every `.parquet` under the root counted
    /// each overview level's rows as more features and described them
    /// as data, so a 100k-feature pyramid published as a 300k-feature
    /// collection whose parts a reader could not tell apart.
    #[test]
    fn a_pyramid_counts_its_leaf_once_and_badges_its_overviews() {
        use super::super::pyramid;

        let dir = std::env::temp_dir().join("geopq_stac_pyramid");
        let _ = std::fs::remove_dir_all(&dir);
        tiny_parquet(&dir.join("r7/87283472bffffff.parquet"), 10, [-1.0, 46.0, 0.0, 47.0]);
        tiny_parquet(&dir.join("r7/87283472affffff.parquet"), 5, [0.0, 47.0, 1.0, 48.0]);
        tiny_overview(&dir.join("r6/86283472fffffff.parquet"), 4, 6, [-1.0, 46.0, 1.0, 48.0]);
        let desc = pyramid::Descriptor {
            version: pyramid::VERSION.to_string(),
            leaf: pyramid::Leaf { res: 7, adaptive_max_res: 7, target_rows: 100, null_part: false },
            levels: vec![
                pyramid::Level {
                    res: 6,
                    method: Some(pyramid::Method::Simplify),
                    cells: vec!["86283472fffffff".into()],
                    rows: Some(4),
                },
                pyramid::Level {
                    res: 7,
                    method: None,
                    cells: vec!["87283472bffffff".into(), "87283472affffff".into()],
                    rows: Some(15),
                },
            ],
            pixels_per_cell: pyramid::DEFAULT_PIXELS_PER_CELL,
            crs: Value::Null,
            bbox: Some([-1.0, 46.0, 1.0, 48.0]),
            rows: Some(15),
            methods: Value::Null,
        };
        std::fs::write(dir.join(pyramid::DESCRIPTOR), desc.to_json()).unwrap();

        let out = write_for_output(&dir, "Pyramid", &Crs::wgs84()).unwrap();
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();

        // The leaf's features, counted once, over the leaf's files.
        let d = doc["description"].as_str().unwrap();
        assert!(d.contains("15 features in 2 GeoParquet files"), "{d}");

        let assets = doc["assets"].as_object().unwrap();
        assert_eq!(assets["r7/87283472bffffff.parquet"]["roles"], json!(["data"]));
        assert_eq!(assets["r6/86283472fffffff.parquet"]["roles"], json!(["overview"]));
        // The descriptor is published with the parts and says what it is.
        let sidecar = &assets[pyramid::DESCRIPTOR];
        assert_eq!(sidecar["roles"], json!(["metadata"]));
        assert_eq!(sidecar["type"], "application/json");
        assert_eq!(sidecar["href"], format!("./{}", pyramid::DESCRIPTOR));
        let _ = std::fs::remove_dir_all(&dir);
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
        // The single asset covers the whole of it.
        assert_eq!(doc["assets"]["roads.parquet"]["bbox"], json!([-1.0, 46.0, 2.0, 48.0]));
        assert_eq!(doc["assets"]["roads.parquet"]["rows"], 3);
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

    #[test]
    /// Publishing a file into a prefix that already has a collection
    /// adds a part to that dataset: assets upsert by key, the extent
    /// widens to the union, and the existing document keeps its
    /// identity. Publishing the same href again replaces that asset.
    fn a_second_publish_merges_into_the_existing_collection() {
        let existing = json!({
            "type": "Collection",
            "stac_version": "1.1.0",
            "id": "parcels",
            "title": "Parcels",
            "license": "CC-BY-4.0",
            "extent": {
                "spatial": {"bbox": [[-2.0, 46.0, 0.0, 47.0]]},
                "temporal": {"interval": [[null, null]]},
            },
            "links": [],
            "assets": {
                "parcels.parquet": {"href": "./parcels.parquet", "rows": 10},
            },
        })
        .to_string();
        let ours = json!({
            "type": "Collection",
            "id": "roads",
            "title": "Roads",
            "extent": {"spatial": {"bbox": [[1.0, 45.0, 3.0, 46.5]]}},
            "assets": {
                "roads.parquet": {"href": "./roads.parquet", "rows": 3},
            },
        });
        let merged = merge_into(&existing, &ours).unwrap();
        assert_eq!(merged["title"], "Parcels", "their identity survives");
        assert_eq!(merged["license"], "CC-BY-4.0");
        assert_eq!(merged["assets"].as_object().unwrap().len(), 2);
        assert_eq!(merged["assets"]["roads.parquet"]["rows"], 3);
        assert_eq!(
            merged["extent"]["spatial"]["bbox"][0],
            json!([-2.0, 45.0, 3.0, 47.0]),
            "the union of both extents",
        );

        // Same href again: the asset is replaced, nothing duplicates.
        let again = json!({
            "type": "Collection",
            "extent": {"spatial": {"bbox": [[1.0, 45.0, 3.0, 46.5]]}},
            "assets": {
                "roads.parquet": {"href": "./roads.parquet", "rows": 5},
            },
        });
        let twice = merge_into(&serde_json::to_string(&merged).unwrap(), &again).unwrap();
        assert_eq!(twice["assets"].as_object().unwrap().len(), 2);
        assert_eq!(twice["assets"]["roads.parquet"]["rows"], 5);
    }

    #[test]
    /// A document that is not a STAC Collection is somebody's file, and
    /// the merge refuses to replace it rather than guessing.
    fn a_foreign_collection_json_is_not_overwritten() {
        let ours = json!({"type": "Collection", "assets": {}});
        for foreign in [
            "not json at all",
            r#"{"hello": "world"}"#,
            r#"{"type": "Feature", "assets": {}}"#,
            r#"{"type": "Collection", "assets": "not an object"}"#,
        ] {
            let err = merge_into(foreign, &ours).unwrap_err();
            assert!(err.contains("not overwriting"), "{foreign}: {err}");
        }
    }

    #[test]
    /// An existing collection with no spatial extent takes ours rather
    /// than leaving the merged dataset undescribed.
    fn a_merge_supplies_the_extent_the_existing_document_lacks() {
        let existing = json!({"type": "Collection", "assets": {}}).to_string();
        let ours = json!({
            "type": "Collection",
            "extent": {"spatial": {"bbox": [[1.0, 45.0, 3.0, 46.5]]}},
            "assets": {"a.parquet": {"href": "./a.parquet"}},
        });
        let merged = merge_into(&existing, &ours).unwrap();
        assert_eq!(merged["extent"]["spatial"]["bbox"][0], json!([1.0, 45.0, 3.0, 46.5]));
        assert_eq!(merged["assets"].as_object().unwrap().len(), 1);
    }
}
