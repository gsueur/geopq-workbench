//! What the app is actually pulling over the network.
//!
//! Remote work is dominated by two very different streams: the byte
//! ranges a layer reads out of a parquet file, and the basemap's raster
//! tiles. Summing them hides the thing you want to know when a pan feels
//! slow, so they are counted apart.
//!
//! Counting happens once per completed request, after the body has been
//! read, so the numbers are bytes received rather than bytes asked for,
//! and the bookkeeping cannot affect throughput.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Rate is averaged over this trailing window: long enough that one
/// large range doesn't read as a spike, short enough that the number
/// falls back to zero promptly when transfers stop.
const WINDOW: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Channel {
    /// Parquet range requests: footers, row groups, covering scans.
    Data,
    /// Basemap raster tiles.
    Tiles,
}

#[derive(Default)]
struct Counter {
    bytes: AtomicU64,
    requests: AtomicU64,
}

/// Recent completed requests, for the rate. Kept per channel.
struct Recent {
    samples: Mutex<VecDeque<(Instant, u64)>>,
}

impl Recent {
    const fn new() -> Self {
        Self {
            samples: Mutex::new(VecDeque::new()),
        }
    }

    fn push(&self, bytes: u64) {
        let now = Instant::now();
        let mut s = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        s.push_back((now, bytes));
        while s.front().is_some_and(|(t, _)| now.duration_since(*t) > WINDOW) {
            s.pop_front();
        }
    }

    /// Bytes per second over the window, 0 when nothing recent arrived.
    fn rate(&self) -> f64 {
        let now = Instant::now();
        let mut s = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        while s.front().is_some_and(|(t, _)| now.duration_since(*t) > WINDOW) {
            s.pop_front();
        }
        let total: u64 = s.iter().map(|(_, b)| b).sum();
        if total == 0 {
            return 0.0;
        }
        // Measure over the window rather than over the span of the
        // samples: with one sample the span is zero, and a single 30 MB
        // range would otherwise report an infinite rate.
        total as f64 / WINDOW.as_secs_f64()
    }
}

static DATA: Counter = Counter {
    bytes: AtomicU64::new(0),
    requests: AtomicU64::new(0),
};
static TILES: Counter = Counter {
    bytes: AtomicU64::new(0),
    requests: AtomicU64::new(0),
};
static DATA_RECENT: Recent = Recent::new();
static TILES_RECENT: Recent = Recent::new();

/// Per-source totals, for the File info panel. Keyed by URL without its
/// query string: a presigned S3 URL carries a signature that changes
/// between sessions and even between refreshes, while the object it
/// names does not.
static BY_SOURCE: Mutex<Option<HashMap<String, (u64, u64)>>> = Mutex::new(None);

fn channel(c: Channel) -> (&'static Counter, &'static Recent) {
    match c {
        Channel::Data => (&DATA, &DATA_RECENT),
        Channel::Tiles => (&TILES, &TILES_RECENT),
    }
}

/// Key a URL by its path, dropping any query string.
pub fn source_key(url: &str) -> &str {
    url.split('?').next().unwrap_or(url)
}

/// One completed request of `bytes` from `url`.
pub fn record(c: Channel, url: &str, bytes: u64) {
    let (counter, recent) = channel(c);
    counter.bytes.fetch_add(bytes, Ordering::Relaxed);
    counter.requests.fetch_add(1, Ordering::Relaxed);
    recent.push(bytes);
    if c == Channel::Data {
        let mut m = BY_SOURCE.lock().unwrap_or_else(|e| e.into_inner());
        let map = m.get_or_insert_with(HashMap::new);
        let e = map.entry(source_key(url).to_string()).or_insert((0, 0));
        e.0 += bytes;
        e.1 += 1;
    }
}

/// Bytes and requests since the app started.
pub fn totals(c: Channel) -> (u64, u64) {
    let (counter, _) = channel(c);
    (
        counter.bytes.load(Ordering::Relaxed),
        counter.requests.load(Ordering::Relaxed),
    )
}

/// Current transfer rate in bytes per second.
pub fn rate(c: Channel) -> f64 {
    channel(c).1.rate()
}

/// Bytes and requests attributed to one source URL, if any were.
pub fn for_source(url: &str) -> Option<(u64, u64)> {
    let m = BY_SOURCE.lock().unwrap_or_else(|e| e.into_inner());
    m.as_ref()?.get(source_key(url)).copied()
}

/// Anything fetched at all this session?
pub fn any_traffic() -> bool {
    totals(Channel::Data).0 > 0 || totals(Channel::Tiles).0 > 0
}

/// Manual: fetch a remote parquet footer and report what it cost.
/// `GEOPQ_URL=https://... cargo test --release net_counts_a_real_fetch
/// -- --ignored --nocapture`
#[cfg(test)]
#[test]
#[ignore = "hits the network"]
fn net_counts_a_real_fetch() {
    let Ok(url) = std::env::var("GEOPQ_URL") else {
        eprintln!("set GEOPQ_URL");
        return;
    };
    let before = totals(Channel::Data);
    let src = crate::data::source::Source::Remote { url: url.clone(), len: 0 };
    match crate::data::loader::open_source_for_test(&src.resolve().unwrap()) {
        Ok((store, _crs, _info, _rg)) => {
            let (bytes, reqs) = totals(Channel::Data);
            eprintln!(
                "opened {} rows: {} B in {} requests (delta {} B, {} requests), rate {:.1} kB/s",
                store.total_rows(),
                bytes,
                reqs,
                bytes - before.0,
                reqs - before.1,
                rate(Channel::Data) / 1024.0,
            );
            assert!(bytes > before.0, "the open must have cost bytes");
            assert!(for_source(&url).is_some(), "attributed to its source");
        }
        Err(e) => eprintln!("open failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_presigned_url_counts_against_the_object_it_names() {
        // The signature changes per session; the object does not.
        let a = "https://s3.example.com/bucket/key.parquet?X-Amz-Signature=aaa";
        let b = "https://s3.example.com/bucket/key.parquet?X-Amz-Signature=bbb";
        assert_eq!(source_key(a), source_key(b));
        assert_eq!(source_key(a), "https://s3.example.com/bucket/key.parquet");
        // A plain URL is its own key.
        let plain = "https://example.com/data.parquet";
        assert_eq!(source_key(plain), plain);
    }

    #[test]
    fn rate_is_measured_over_the_window_not_the_samples() {
        let r = Recent::new();
        assert_eq!(r.rate(), 0.0, "idle reads zero");
        // A single large range must not read as an infinite rate just
        // because the samples span no time.
        r.push(30 << 20);
        let rate = r.rate();
        assert!(rate.is_finite() && rate > 0.0, "got {rate}");
        assert!(
            (rate - (30 << 20) as f64 / WINDOW.as_secs_f64()).abs() < 1.0,
            "one 30 MB range over a {:?} window, got {rate}",
            WINDOW
        );
    }

    #[test]
    fn channels_are_counted_apart() {
        let before_data = totals(Channel::Data);
        let before_tiles = totals(Channel::Tiles);
        record(Channel::Tiles, "https://tiles.example/1/2/3.png", 1000);
        assert_eq!(totals(Channel::Data), before_data, "data is untouched");
        assert_eq!(totals(Channel::Tiles).0, before_tiles.0 + 1000);
        assert_eq!(totals(Channel::Tiles).1, before_tiles.1 + 1);
        // Tiles are not attributed per source: they are not a layer.
        assert!(for_source("https://tiles.example/1/2/3.png").is_none());
    }
}
