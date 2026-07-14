//! Where a GeoParquet file lives: local disk or an HTTP(S) URL.
//!
//! Remote files are read through HTTP range requests via a [`ChunkReader`]
//! implementation, so the whole load pipeline (metadata pruning, covering
//! per-feature selection, lazy attributes, picking) works identically over
//! the network — only the byte ranges actually needed are downloaded.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::OnceLock;

use bytes::Bytes;
use parquet::errors::{ParquetError, Result as PqResult};
use parquet::file::reader::{ChunkReader, Length};

const USER_AGENT: &str = concat!("geopq-viewer/", env!("CARGO_PKG_VERSION"));

/// Shared agent: connection pooling across range requests.
fn http_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(ureq::Agent::new_with_defaults)
}

#[derive(Clone, Debug)]
pub enum Source {
    Local(PathBuf),
    /// HTTP(S) with range requests. `len` is resolved once at open.
    Remote { url: String, len: u64 },
}

impl Source {
    /// Resolve a URL into a source (fetches the content length; verifies
    /// the server answers). Network call — run off the UI thread.
    pub fn remote(url: &str) -> Result<Source, String> {
        let len = remote_len(url)?;
        Ok(Source::Remote {
            url: url.to_string(),
            len,
        })
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, Source::Remote { .. })
    }

    /// Full path or URL, for tooltips and error messages.
    pub fn label(&self) -> String {
        match self {
            Source::Local(p) => p.display().to_string(),
            Source::Remote { url, .. } => url.clone(),
        }
    }

    /// Short display name (file stem / last URL segment).
    pub fn name(&self) -> String {
        match self {
            Source::Local(p) => p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "layer".into()),
            Source::Remote { url, .. } => url
                .split('/')
                .next_back()
                .filter(|s| !s.is_empty())
                .map(|s| s.trim_end_matches(".parquet").to_string())
                .unwrap_or_else(|| "remote".into()),
        }
    }

    pub fn size(&self) -> u64 {
        match self {
            Source::Local(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            Source::Remote { len, .. } => *len,
        }
    }

    pub fn open(&self) -> Result<SourceReader, String> {
        match self {
            Source::Local(p) => {
                let f = File::open(p).map_err(|e| format!("cannot open file: {e}"))?;
                let len = f
                    .metadata()
                    .map_err(|e| format!("cannot stat file: {e}"))?
                    .len();
                Ok(SourceReader {
                    inner: Inner::Local(f),
                    len,
                })
            }
            Source::Remote { url, len } => Ok(SourceReader {
                inner: Inner::Remote { url: url.clone() },
                len: *len,
            }),
        }
    }
}

/// Content length via HEAD, falling back to a 1-byte range GET for servers
/// that answer HEAD without Content-Length.
fn remote_len(url: &str) -> Result<u64, String> {
    let head = http_agent()
        .head(url)
        .header("User-Agent", USER_AGENT)
        .call();
    if let Ok(res) = head {
        if let Some(len) = header(&res, "content-length").and_then(|v| v.parse::<u64>().ok()) {
            if len > 0 {
                return Ok(len);
            }
        }
    }
    let res = http_agent()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .header("Range", "bytes=0-0")
        .call()
        .map_err(|e| format!("cannot reach {url}: {e}"))?;
    if res.status() != 206 {
        return Err(format!(
            "{url}: server does not support range requests (status {})",
            res.status()
        ));
    }
    // Content-Range: bytes 0-0/12345
    header(&res, "content-range")
        .and_then(|v| v.rsplit('/').next()?.parse::<u64>().ok())
        .ok_or_else(|| format!("{url}: missing total length in Content-Range"))
}

fn header<B>(res: &ureq::http::Response<B>, name: &str) -> Option<String> {
    res.headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

pub struct SourceReader {
    inner: Inner,
    len: u64,
}

enum Inner {
    Local(File),
    Remote { url: String },
}

/// One bounded range request, fully read.
fn fetch_range(url: &str, start: u64, end_inclusive: u64) -> PqResult<Vec<u8>> {
    let expect = (end_inclusive - start + 1) as usize;
    let res = http_agent()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .header("Range", &format!("bytes={start}-{end_inclusive}"))
        .call()
        .map_err(|e| ParquetError::General(format!("range request failed: {e}")))?;
    match res.status().as_u16() {
        206 => {}
        // Whole-file answer is only usable from offset 0.
        200 if start == 0 => {}
        s => {
            return Err(ParquetError::General(format!(
                "server rejected range request ({s})"
            )))
        }
    }
    let mut buf = Vec::with_capacity(expect);
    let read = res
        .into_body()
        .into_reader()
        .take(expect as u64)
        .read_to_end(&mut buf)
        .map_err(|e| ParquetError::General(format!("range read: {e}")))?;
    if read != expect {
        return Err(ParquetError::EOF(format!(
            "expected {expect} bytes at offset {start}, got {read}"
        )));
    }
    Ok(buf)
}

/// Sequential remote reads without an end bound (parquet's `get_read`):
/// fetch in growing windows instead of one open-ended request, so the
/// server only ever sends bytes the reader actually consumes (plus at most
/// the current window).
struct WindowedRemote {
    url: String,
    pos: u64,
    end: u64,
    window: u64,
    chunk: std::io::Cursor<Vec<u8>>,
}

const WINDOW_START: u64 = 256 * 1024;
const WINDOW_MAX: u64 = 8 * 1024 * 1024;

impl Read for WindowedRemote {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let n = self.chunk.read(out)?;
            if n > 0 {
                return Ok(n);
            }
            if self.pos >= self.end {
                return Ok(0);
            }
            let take = self.window.min(self.end - self.pos);
            let data = fetch_range(&self.url, self.pos, self.pos + take - 1)
                .map_err(std::io::Error::other)?;
            self.pos += data.len() as u64;
            self.window = (self.window * 2).min(WINDOW_MAX);
            self.chunk = std::io::Cursor::new(data);
        }
    }
}

impl Inner {
    fn ranged(&self, start: u64, end_inclusive: Option<u64>, len: u64) -> PqResult<Box<dyn Read + Send>> {
        match self {
            Inner::Local(f) => {
                let mut r = f
                    .try_clone()
                    .map_err(|e| ParquetError::General(format!("clone: {e}")))?;
                r.seek(SeekFrom::Start(start))
                    .map_err(|e| ParquetError::General(format!("seek: {e}")))?;
                Ok(Box::new(BufReader::new(r)))
            }
            Inner::Remote { url } => match end_inclusive {
                Some(e) => Ok(Box::new(std::io::Cursor::new(fetch_range(url, start, e)?))),
                None => Ok(Box::new(WindowedRemote {
                    url: url.clone(),
                    pos: start,
                    end: len,
                    window: WINDOW_START,
                    chunk: std::io::Cursor::new(Vec::new()),
                })),
            },
        }
    }
}

impl Length for SourceReader {
    fn len(&self) -> u64 {
        self.len
    }
}

/// Minimal HTTP/1.1 range-request server over a local file, for tests:
/// supports HEAD (Content-Length) and GET with `Range: bytes=a-b` / `a-`,
/// one request per connection, and counts body bytes served.
#[cfg(test)]
pub(crate) mod testserver {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    pub struct RangeServer {
        pub url: String,
        pub bytes_served: Arc<AtomicU64>,
        pub requests: Arc<AtomicU64>,
    }

    pub fn spawn(file: PathBuf) -> RangeServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let len = std::fs::metadata(&file).unwrap().len();
        let bytes_served = Arc::new(AtomicU64::new(0));
        let requests = Arc::new(AtomicU64::new(0));
        let (b2, r2) = (bytes_served.clone(), requests.clone());
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                let (file, bytes, reqs) = (file.clone(), b2.clone(), r2.clone());
                std::thread::spawn(move || {
                    // Read request head.
                    let mut buf = Vec::new();
                    let mut b = [0u8; 1];
                    while !buf.ends_with(b"\r\n\r\n") && buf.len() < 8192 {
                        match conn.read(&mut b) {
                            Ok(1) => buf.push(b[0]),
                            _ => return,
                        }
                    }
                    reqs.fetch_add(1, Ordering::SeqCst);
                    let text = String::from_utf8_lossy(&buf);
                    let is_head = text.starts_with("HEAD");
                    let range = text
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                        .and_then(|l| l.split_once(':'))
                        .and_then(|(_, v)| v.trim().strip_prefix("bytes="))
                        .and_then(|v| v.split_once('-'))
                        .map(|(a, b)| {
                            let start: u64 = a.parse().unwrap_or(0);
                            let end: u64 = b.parse().unwrap_or(len - 1);
                            (start, end.min(len - 1))
                        });
                    if is_head {
                        let _ = write!(
                            conn,
                            "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
                        );
                        return;
                    }
                    let (start, end) = range.unwrap_or((0, len - 1));
                    let n = end - start + 1;
                    let (status, extra) = if range.is_some() {
                        (
                            "206 Partial Content",
                            format!("Content-Range: bytes {start}-{end}/{len}\r\n"),
                        )
                    } else {
                        ("200 OK", String::new())
                    };
                    let _ = write!(
                        conn,
                        "HTTP/1.1 {status}\r\nContent-Length: {n}\r\n{extra}Accept-Ranges: bytes\r\nConnection: close\r\n\r\n"
                    );
                    use std::os::unix::fs::FileExt;
                    let f = std::fs::File::open(&file).unwrap();
                    let mut pos = start;
                    let mut chunk = vec![0u8; 256 * 1024];
                    while pos <= end {
                        let take = ((end - pos + 1) as usize).min(chunk.len());
                        let read = f.read_at(&mut chunk[..take], pos).unwrap();
                        if read == 0 || conn.write_all(&chunk[..read]).is_err() {
                            return;
                        }
                        bytes.fetch_add(read as u64, Ordering::SeqCst);
                        pos += read as u64;
                    }
                });
            }
        });
        RangeServer {
            url: format!("http://127.0.0.1:{port}/data.parquet"),
            bytes_served,
            requests,
        }
    }
}

impl ChunkReader for SourceReader {
    type T = Box<dyn Read + Send>;

    fn get_read(&self, start: u64) -> PqResult<Self::T> {
        self.inner.ranged(start, None, self.len)
    }

    fn get_bytes(&self, start: u64, length: usize) -> PqResult<Bytes> {
        if length == 0 {
            return Ok(Bytes::new());
        }
        let end = start + length as u64 - 1;
        let mut buf = Vec::with_capacity(length);
        let read = self
            .inner
            .ranged(start, Some(end), self.len)?
            .take(length as u64)
            .read_to_end(&mut buf)
            .map_err(|e| ParquetError::General(format!("range read: {e}")))?;
        if read != length {
            return Err(ParquetError::EOF(format!(
                "expected {length} bytes at offset {start}, got {read}"
            )));
        }
        Ok(buf.into())
    }
}
