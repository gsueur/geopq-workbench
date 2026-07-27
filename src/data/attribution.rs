//! Data attribution for a layer.
//!
//! Most open data carries a licence that asks for a credit somewhere the
//! user can see. Two places are checked, in order:
//!
//! - the parquet key-value metadata (`attribution`, `license`, …), which
//!   costs nothing to read and travels with the file;
//! - an `ATTRIBUTION.txt` sitting next to the data, the convention the
//!   parquetry repositories follow. Ancestors are walked from the file's
//!   own directory upwards and the nearest notice wins, so one file at a
//!   repository's root credits everything under it while a dataset that
//!   needs its own can still override it.
//!
//! Nothing here is required: a layer without attribution simply has
//! none, and no request is retried or reported.

use std::path::Path;

use super::source::{http_agent, Source};

/// Parquet key-value keys that carry a credit, most specific first.
const META_KEYS: [&str; 5] = [
    "attribution",
    "geopq:attribution",
    "copyright",
    "license",
    "source",
];

const FILE_NAME: &str = "ATTRIBUTION.txt";
/// How far up to look. A URL's ancestors stay inside one publisher's
/// namespace and stop at the host, so the walk can safely reach a
/// repository root. A local path's ancestors turn into the user's home
/// directory within a couple of steps, where an unrelated notice would
/// be someone else's data — hence the shorter leash.
const REMOTE_ANCESTORS: usize = 6;
const LOCAL_ANCESTORS: usize = 3;
/// Sanity cap: a credit file is prose, not a payload.
const MAX_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct Attribution {
    /// One-line credit for the map corner ("geoBoundaries").
    pub credit: String,
    /// Everything found, for the file info panel.
    pub text: String,
}

/// Pull the short credit out of an attribution document.
///
/// The convention these files follow is a `Key: value` block, so an
/// explicit `Attribution:` line wins; `Source:` is the usual runner-up.
/// Failing both, the first line that is neither blank nor a heading is
/// better than nothing.
pub fn parse(text: &str) -> Option<Attribution> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    // A `Key: value` line, plus any indented continuation lines that
    // follow it: these notices wrap, and a credit cut at the wrap reads
    // as "(c) European Union, Copernicus Land Monitoring Service," which
    // is worse than no credit at all.
    let labelled = |label: &str| -> Option<String> {
        let lines: Vec<&str> = text.lines().collect();
        let i = lines.iter().position(|l| {
            let l = l.trim().trim_start_matches(['-', '*', '#']).trim();
            l.split_once(':')
                .is_some_and(|(k, _)| k.trim().eq_ignore_ascii_case(label))
        })?;
        let first = lines[i]
            .trim()
            .trim_start_matches(['-', '*', '#'])
            .trim()
            .split_once(':')?
            .1
            .trim()
            .to_string();
        let mut out = first;
        for l in &lines[i + 1..] {
            // Indented, and not itself a labelled entry.
            if l.is_empty() || !l.starts_with(char::is_whitespace) {
                break;
            }
            let t = l.trim();
            if t.is_empty() || is_labelled_line(t) {
                break;
            }
            out.push(' ');
            out.push_str(t);
        }
        Some(out)
    };
    let credit = labelled("attribution")
        .or_else(|| labelled("source"))
        .or_else(|| {
            text.lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string)
        })
        .filter(|c| !c.is_empty())?;
    // The map corner is not the place for a paragraph.
    let credit = match credit.char_indices().nth(90) {
        Some((i, _)) => format!("{}…", credit[..i].trim_end()),
        None => credit,
    };
    Some(Attribution {
        credit,
        text: text.to_string(),
    })
}

/// A short `Key: value` opener, the shape these notices use for their
/// fields. A bare URL is not one: `https://…` has no space after its
/// colon, so a wrapped line carrying a link stays part of its value.
fn is_labelled_line(line: &str) -> bool {
    line.split_once(": ").is_some_and(|(k, _)| {
        !k.is_empty()
            && k.len() <= 24
            && k.chars().all(|c| c.is_alphanumeric() || c == ' ' || c == '_' || c == '-')
    })
}

/// Attribution for a layer, or None when the data carries none.
pub fn find(source: &Source, kv: &[parquet::file::metadata::KeyValue]) -> Option<Attribution> {
    for key in META_KEYS {
        let hit = kv
            .iter()
            .find(|e| e.key.eq_ignore_ascii_case(key))
            .and_then(|e| e.value.as_deref())
            .and_then(parse);
        if hit.is_some() {
            return hit;
        }
    }
    sidecar(source)
}

/// `ATTRIBUTION.txt` beside the data, then up the ancestors.
fn sidecar(source: &Source) -> Option<Attribution> {
    match source {
        Source::Local(p) => local_chain(p.parent()?),
        Source::Dir(p) => local_chain(p),
        Source::Remote { url, .. } => remote_chain(url),
        Source::S3 { url, uri, .. } => {
            // The signed URL is what can actually be fetched; the s3://
            // form is only a display name.
            let base = if url.is_empty() { uri } else { url };
            remote_chain(base)
        }
        Source::Stac { url, .. } => remote_chain(url),
        Source::Multi { urls, .. } => remote_chain(urls.first()?),
    }
}

/// Memoized per directory, negatives included: a hive dataset opens
/// dozens of parts out of the same folder, and each one would otherwise
/// repeat the same lookup — over the network, for a remote dataset.
fn cached(key: &str, lookup: impl FnOnce() -> Option<Attribution>) -> Option<Attribution> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, Option<Attribution>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .cloned()
    {
        return hit;
    }
    let found = lookup();
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.to_string(), found.clone());
    found
}

fn local(dir: &Path) -> Option<Attribution> {
    cached(&dir.to_string_lossy(), || local_uncached(dir))
}

fn local_uncached(dir: &Path) -> Option<Attribution> {
    let p = dir.join(FILE_NAME);
    let meta = std::fs::metadata(&p).ok()?;
    if meta.len() as usize > MAX_BYTES {
        return None;
    }
    parse(&std::fs::read_to_string(&p).ok()?)
}

fn local_chain(dir: &Path) -> Option<Attribution> {
    let mut cur = Some(dir);
    for _ in 0..LOCAL_ANCESTORS {
        let d = cur?;
        if let Some(a) = local(d) {
            return Some(a);
        }
        cur = d.parent();
    }
    None
}

fn remote_chain(url: &str) -> Option<Attribution> {
    let base = url.split(['?', '#']).next().unwrap_or(url);
    let mut dir = base.rsplit_once('/')?.0;
    for _ in 0..REMOTE_ANCESTORS {
        if let Some(a) = remote(dir) {
            return Some(a);
        }
        match parent_url(dir) {
            Some(p) => dir = p,
            None => break,
        }
    }
    None
}

/// The parent of a URL directory, or None once it is `scheme://host`.
fn parent_url(dir: &str) -> Option<&str> {
    let root_len = dir.find("://")? + 3;
    let (parent, _) = dir.rsplit_once('/')?;
    (parent.len() >= root_len).then_some(parent)
}

fn remote(dir: &str) -> Option<Attribution> {
    cached(dir, || remote_uncached(dir))
}

fn remote_uncached(dir: &str) -> Option<Attribution> {
    // A scheme-less or bucket-root path has nothing useful to ask for.
    if !dir.starts_with("http://") && !dir.starts_with("https://") {
        return None;
    }
    let res = http_agent()
        .get(format!("{dir}/{FILE_NAME}"))
        .call()
        .ok()?;
    if res.status().as_u16() != 200 {
        return None;
    }
    let body = res.into_body().read_to_string().ok()?;
    // Counted with the layer's own traffic: a sidecar lookup is part of
    // what opening a remote dataset costs, and it is one more round trip
    // against object storage.
    super::net::record(
        super::net::Channel::Data,
        &format!("{dir}/{FILE_NAME}"),
        body.len() as u64,
    );
    if body.len() > MAX_BYTES {
        return None;
    }
    parse(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEOBOUNDARIES: &str = "\
# Data attribution

This data is derived from geoBoundaries, produced by the geoBoundaries
project at the William & Mary geoLab.

  Source:       https://www.geoboundaries.org/
  Release:      geoBoundaries 6.0.0 (published 2023-09-14)
  License:      Creative Commons Attribution 4.0 International (CC BY 4.0)
  Attribution:  geoBoundaries (https://www.geoboundaries.org/)
";

    #[test]
    fn explicit_attribution_line_wins() {
        let a = parse(GEOBOUNDARIES).unwrap();
        assert_eq!(a.credit, "geoBoundaries (https://www.geoboundaries.org/)");
        // The full document is kept for the info panel.
        assert!(a.text.contains("CC BY 4.0"));
    }

    #[test]
    fn wrapped_values_are_joined() {
        // The CORINE notice wraps its credit across two lines; cutting
        // it at the wrap loses the attributed body.
        let a = parse(
            "  Source:       https://land.copernicus.eu/\n\
             \x20 Attribution:  (c) European Union, Copernicus Land Monitoring Service,\n\
             \x20               European Environment Agency (EEA)\n\
             \n\
             Downstream users MUST credit the service.\n",
        )
        .unwrap();
        assert!(a.credit.ends_with("European Environment Agency (EEA)"), "{}", a.credit);
        assert!(a.credit.starts_with("(c) European Union"), "{}", a.credit);
        // The next labelled entry ends the value.
        let a = parse("Attribution: One\n  License: CC BY 4.0\n").unwrap();
        assert_eq!(a.credit, "One");
        // An unindented line ends it too.
        let a = parse("Attribution: One\nNot a continuation\n").unwrap();
        assert_eq!(a.credit, "One");
    }

    #[test]
    fn falls_back_through_source_then_prose() {
        let a = parse("Source: Natural Earth\nLicense: public domain").unwrap();
        assert_eq!(a.credit, "Natural Earth");
        let a = parse("# Credits\n\n© Someone, 2026\n").unwrap();
        assert_eq!(a.credit, "© Someone, 2026");
        assert!(parse("   \n\n  ").is_none());
        assert!(parse("").is_none());
    }

    #[test]
    fn long_credits_are_cut_for_the_map_corner() {
        let long = format!("Attribution: {}", "x".repeat(200));
        let a = parse(&long).unwrap();
        assert!(a.credit.chars().count() <= 91, "{}", a.credit);
        assert!(a.credit.ends_with('…'));
        // The untruncated text stays available.
        assert!(a.text.contains(&"x".repeat(200)));
    }

    /// Distinct per test: `local` memoizes by directory.
    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("geopq_attr_{tag}_{}", std::process::id()))
    }

    #[test]
    fn metadata_keys_beat_the_sidecar_and_are_tried_in_order() {
        let kv = |k: &str, v: &str| parquet::file::metadata::KeyValue {
            key: k.to_string(),
            value: Some(v.to_string()),
        };
        let dir = temp_dir("meta");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("x.parquet");
        std::fs::write(&file, b"").unwrap();
        std::fs::write(dir.join(FILE_NAME), "Attribution: from the sidecar").unwrap();
        let src = Source::Local(file);

        assert_eq!(
            find(&src, &[kv("license", "CC BY 4.0")]).unwrap().credit,
            "CC BY 4.0"
        );
        // A more specific key wins over a less specific one.
        assert_eq!(
            find(&src, &[kv("license", "CC BY 4.0"), kv("attribution", "IGN")])
                .unwrap()
                .credit,
            "IGN"
        );
        // Nothing in the metadata: the sidecar answers.
        assert_eq!(find(&src, &[]).unwrap().credit, "from the sidecar");
        // An empty value is not an attribution.
        assert_eq!(
            find(&src, &[kv("attribution", "  ")]).unwrap().credit,
            "from the sidecar"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Live probe of the sidecar convention, opt-in:
    /// cargo test --release attribution_live -- --ignored --nocapture
    #[test]
    #[ignore]
    fn attribution_live() {
        let src = Source::Remote {
            url: "https://parquetry.geomermaids.com/geoboundaries/6.0.0/cgaz_adm0.parquet"
                .to_string(),
            len: 0,
        };
        let a = find(&src, &[]).expect("ATTRIBUTION.txt beside the data");
        eprintln!("geoboundaries: {}", a.credit);
        assert!(a.credit.contains("geoBoundaries"), "{}", a.credit);
        assert!(a.text.contains("CC BY 4.0"));

        // An OSM theme sits four directories below the repository's own
        // notice, and must still find it.
        let osm = Source::Remote {
            url: "https://parquetry.geomermaids.com/latest/country=US/state=US-AR/\
                  buildings.parquet"
                .replace(' ', "")
                .to_string(),
            len: 0,
        };
        let a = find(&osm, &[]).expect("repository-root ATTRIBUTION.txt");
        eprintln!("osm: {}", a.credit);
        assert!(a.credit.contains("OpenStreetMap"), "{}", a.credit);
        assert!(a.text.contains("ODbL"));

        // CORINE: the notice sits beside the data, like geoBoundaries.
        let clc = Source::Remote {
            url: "https://parquetry.geomermaids.com/clc/2018/clc_2018.parquet".to_string(),
            len: 0,
        };
        let a = find(&clc, &[]).expect("CLC ATTRIBUTION.txt");
        eprintln!("clc: {}", a.credit);
        assert!(a.credit.contains("Copernicus"), "{}", a.credit);
        // The wrapped credit must arrive whole, agency and all.
        assert!(a.credit.contains("European Environment Agency"), "{}", a.credit);
    }

    #[test]
    fn url_ancestors_stop_at_the_host() {
        assert_eq!(
            parent_url("https://h.example/a/b/c"),
            Some("https://h.example/a/b")
        );
        assert_eq!(parent_url("https://h.example/a"), Some("https://h.example"));
        // The host itself has no parent to ask.
        assert_eq!(parent_url("https://h.example"), None);
        assert_eq!(parent_url("no-scheme/a/b"), None);
    }

    #[test]
    fn sidecar_is_found_one_directory_up() {
        let root = temp_dir("parent");
        let parts = root.join("country=US");
        std::fs::create_dir_all(&parts).unwrap();
        std::fs::write(root.join(FILE_NAME), "Attribution: release-wide").unwrap();
        let f = parts.join("roads.parquet");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(
            find(&Source::Local(f), &[]).unwrap().credit,
            "release-wide"
        );
        // A dataset directory looks beside itself too.
        assert_eq!(
            find(&Source::Dir(parts.clone()), &[]).unwrap().credit,
            "release-wide"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_nearest_notice_wins() {
        // A repository-wide notice at the root, a dataset that overrides
        // it further down. Fresh directories: lookups are memoized, so a
        // tree must be built before it is probed.
        let root = temp_dir("nearest");
        let inner = root.join("dataset").join("v1");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(root.join(FILE_NAME), "Attribution: repository-wide").unwrap();
        std::fs::write(inner.join(FILE_NAME), "Attribution: this dataset").unwrap();

        let f = inner.join("x.parquet");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(find(&Source::Local(f), &[]).unwrap().credit, "this dataset");

        // A sibling dataset with no notice of its own inherits the root's.
        let bare = root.join("other").join("v1");
        std::fs::create_dir_all(&bare).unwrap();
        let g = bare.join("y.parquet");
        std::fs::write(&g, b"").unwrap();
        assert_eq!(
            find(&Source::Local(g), &[]).unwrap().credit,
            "repository-wide"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
