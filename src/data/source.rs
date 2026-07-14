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

/// Shared agent: connection pooling across range requests. HTTP error
/// statuses come back as responses (not `Err`), so callers can read
/// headers like `x-amz-bucket-region` from 301/403 answers.
fn http_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into()
    })
}

#[derive(Clone, Debug)]
pub enum Source {
    Local(PathBuf),
    /// HTTP(S) with range requests. `len` is resolved once at open.
    Remote { url: String, len: u64 },
    /// s3://bucket/key, read through a (pre)signed HTTPS URL resolved at
    /// open from the selected `~/.aws` profile / environment credentials
    /// (anonymous for public buckets). `endpoint` targets S3-compatible
    /// services (path-style); None = AWS. Empty `url` = unresolved.
    S3 {
        uri: String,
        profile: Option<String>,
        endpoint: Option<String>,
        url: String,
        len: u64,
    },
}

impl Source {
    /// Resolve a URL into a source (fetches the content length; verifies
    /// the server answers). Network call — run off the UI thread.
    #[cfg(test)]
    pub fn remote(url: &str) -> Result<Source, String> {
        Source::Remote {
            url: url.to_string(),
            len: 0,
        }
        .resolve()
    }

    /// Resolve credentials/length as needed (no-op when already resolved).
    /// Network + credential-file reads — run off the UI thread.
    pub fn resolve(self) -> Result<Source, String> {
        match self {
            Source::Remote { url, len: 0 } => {
                let len = remote_len(&url)?;
                Ok(Source::Remote { url, len })
            }
            Source::S3 {
                uri,
                profile,
                endpoint,
                url,
                ..
            } if url.is_empty() => {
                let url = aws::presign(&uri, profile.as_deref(), endpoint.as_deref())?;
                let len = remote_len(&url)
                    .map_err(|e| format!("{uri}: {}", redact_presign(&e)))?;
                Ok(Source::S3 {
                    uri,
                    profile,
                    endpoint,
                    url,
                    len,
                })
            }
            other => Ok(other),
        }
    }

    pub fn is_remote(&self) -> bool {
        !matches!(self, Source::Local(_))
    }

    /// Full path / URL / S3 URI, for tooltips and error messages (never
    /// the presigned URL — it embeds a signed access grant).
    pub fn label(&self) -> String {
        match self {
            Source::Local(p) => p.display().to_string(),
            Source::Remote { url, .. } => url.clone(),
            Source::S3 { uri, .. } => uri.clone(),
        }
    }

    /// Short display name (file stem / last URL segment).
    pub fn name(&self) -> String {
        match self {
            Source::Local(p) => p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "layer".into()),
            Source::Remote { url, .. } | Source::S3 { uri: url, .. } => url
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
            Source::Remote { len, .. } | Source::S3 { len, .. } => *len,
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
            Source::Remote { url, len } | Source::S3 { url, len, .. } => Ok(SourceReader {
                inner: Inner::Remote { url: url.clone() },
                len: *len,
            }),
        }
    }
}

/// Keep signed query parameters out of error strings.
fn redact_presign(msg: &str) -> String {
    match msg.find("?X-Amz") {
        Some(i) => format!("{}?<presigned>", &msg[..i]),
        None => msg.to_string(),
    }
}

/// AWS credential/profile handling and S3 presigning. Kept deliberately
/// small: static keys (+ optional session token) from `~/.aws` files or
/// the environment, region from profile/env/bucket probe, anonymous
/// fallback for public buckets. No SSO/IMDS credential providers.
pub mod aws {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn aws_dir() -> PathBuf {
        std::env::var_os("AWS_SHARED_CREDENTIALS_FILE")
            .map(|f| PathBuf::from(f).parent().map(Path::to_path_buf).unwrap_or_default())
            .or_else(|| dirs_home().map(|h| h.join(".aws")))
            .unwrap_or_default()
    }

    fn dirs_home() -> Option<PathBuf> {
        std::env::var_os("HOME").map(PathBuf::from)
    }

    /// Minimal INI parser: `[section]` + `key = value` lines.
    fn ini(path: &Path) -> HashMap<String, HashMap<String, String>> {
        let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
        let Ok(text) = std::fs::read_to_string(path) else {
            return out;
        };
        let mut section = String::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                // config file uses "[profile x]", credentials file "[x]".
                section = name.trim().trim_start_matches("profile ").trim().to_string();
            } else if let Some((k, v)) = line.split_once('=') {
                out.entry(section.clone())
                    .or_default()
                    .insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }
        out
    }

    /// Profile names available in `~/.aws/{credentials,config}` (for the UI).
    pub fn profiles() -> Vec<String> {
        let dir = aws_dir();
        let mut names: Vec<String> = ini(&dir.join("credentials"))
            .into_keys()
            .chain(ini(&dir.join("config")).into_keys())
            .filter(|n| !n.is_empty())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    struct Creds {
        key: String,
        secret: String,
        token: Option<String>,
    }

    /// Static credentials for a profile (or the environment / default
    /// profile when none is selected). None = anonymous.
    fn credentials(profile: Option<&str>) -> Option<Creds> {
        let from_files = |name: &str| -> Option<Creds> {
            let dir = aws_dir();
            let mut merged = ini(&dir.join("config"));
            for (sec, kv) in ini(&dir.join("credentials")) {
                merged.entry(sec).or_default().extend(kv);
            }
            let s = merged.get(name)?;
            Some(Creds {
                key: s.get("aws_access_key_id")?.clone(),
                secret: s.get("aws_secret_access_key")?.clone(),
                token: s.get("aws_session_token").cloned(),
            })
        };
        if let Some(p) = profile {
            return from_files(p);
        }
        if let (Ok(key), Ok(secret)) = (
            std::env::var("AWS_ACCESS_KEY_ID"),
            std::env::var("AWS_SECRET_ACCESS_KEY"),
        ) {
            return Some(Creds {
                key,
                secret,
                token: std::env::var("AWS_SESSION_TOKEN").ok(),
            });
        }
        let name = std::env::var("AWS_PROFILE").unwrap_or_else(|_| "default".into());
        from_files(&name)
    }

    /// Region for a profile, falling back to env, then a bucket-location
    /// probe (`x-amz-bucket-region` is present even on 403/301 answers),
    /// then us-east-1.
    fn region(profile: Option<&str>, bucket: &str, custom_endpoint: bool) -> String {
        let dir = aws_dir();
        let files_region = |name: &str| -> Option<String> {
            let config = ini(&dir.join("config"));
            let creds = ini(&dir.join("credentials"));
            config
                .get(name)
                .and_then(|s| s.get("region").cloned())
                .or_else(|| creds.get(name).and_then(|s| s.get("region").cloned()))
        };
        if let Some(r) = profile.and_then(files_region) {
            return r;
        }
        if let Ok(r) = std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION")) {
            return r;
        }
        if profile.is_none() {
            let name = std::env::var("AWS_PROFILE").unwrap_or_else(|_| "default".into());
            if let Some(r) = files_region(&name) {
                return r;
            }
        }
        // Probe (AWS only): any response carries the bucket's region header.
        if custom_endpoint {
            return "us-east-1".into();
        }
        if let Ok(res) = super::http_agent()
            .head(&format!("https://{bucket}.s3.amazonaws.com/"))
            .header("User-Agent", super::USER_AGENT)
            .call()
        {
            if let Some(r) = super::header(&res, "x-amz-bucket-region") {
                return r;
            }
        }
        "us-east-1".into()
    }

    /// Turn `s3://bucket/key` into a GET URL: presigned when credentials
    /// exist, plain URL for anonymous/public access. Endpoint priority:
    /// explicit `endpoint` (UI field) > `AWS_ENDPOINT_URL` > the profile's
    /// `endpoint_url`; custom endpoints use path-style requests.
    pub fn presign(
        uri: &str,
        profile: Option<&str>,
        endpoint: Option<&str>,
    ) -> Result<String, String> {
        let rest = uri
            .strip_prefix("s3://")
            .ok_or_else(|| format!("not an s3:// URI: {uri}"))?;
        let (bucket, key) = rest
            .split_once('/')
            .filter(|(b, k)| !b.is_empty() && !k.is_empty())
            .ok_or_else(|| format!("expected s3://bucket/key, got {uri}"))?;

        let creds = credentials(profile);
        if profile.is_some() && creds.is_none() {
            return Err(format!(
                "profile '{}' has no static credentials in ~/.aws",
                profile.unwrap()
            ));
        }
        // Custom endpoints (MinIO, Wasabi, ...): explicit field first,
        // then env, then the profile's endpoint_url (AWS CLI v2 convention).
        let normalize = |e: &str| -> String {
            let e = e.trim().trim_end_matches('/');
            if e.starts_with("http://") || e.starts_with("https://") {
                e.to_string()
            } else {
                format!("https://{e}")
            }
        };
        let endpoint_env = endpoint
            .filter(|e| !e.trim().is_empty())
            .map(normalize)
            .or_else(|| std::env::var("AWS_ENDPOINT_URL").ok().map(|e| normalize(&e)))
            .or_else(|| {
            let name = profile
                .map(str::to_string)
                .or_else(|| std::env::var("AWS_PROFILE").ok())
                .unwrap_or_else(|| "default".into());
            let dir = aws_dir();
            let config = ini(&dir.join("config"));
            let creds_file = ini(&dir.join("credentials"));
                config
                    .get(&name)
                    .and_then(|s| s.get("endpoint_url").cloned())
                    .or_else(|| {
                        creds_file.get(&name).and_then(|s| s.get("endpoint_url").cloned())
                    })
                    .map(|e| normalize(&e))
            });
        let region = region(profile, bucket, endpoint_env.is_some());
        let (endpoint, style) = match &endpoint_env {
            Some(e) => (e.clone(), rusty_s3::UrlStyle::Path),
            None => (
                format!("https://s3.{region}.amazonaws.com"),
                rusty_s3::UrlStyle::VirtualHost,
            ),
        };

        match creds {
            None => {
                // Anonymous: plain object URL.
                Ok(match &endpoint_env {
                    Some(e) => format!("{}/{bucket}/{key}", e.trim_end_matches('/')),
                    None => format!("https://{bucket}.s3.{region}.amazonaws.com/{key}"),
                })
            }
            Some(c) => {
                use rusty_s3::S3Action;
                let endpoint = endpoint
                    .parse()
                    .map_err(|e| format!("bad S3 endpoint: {e}"))?;
                let b = rusty_s3::Bucket::new(endpoint, style, bucket.to_string(), region)
                    .map_err(|e| format!("bad S3 bucket: {e}"))?;
                let rc = match &c.token {
                    Some(t) => rusty_s3::Credentials::new_with_token(&c.key, &c.secret, t),
                    None => rusty_s3::Credentials::new(&c.key, &c.secret),
                };
                // Temporary credentials expire; long-lived keys allow up
                // to 7 days of presign validity.
                let expiry = if c.token.is_some() {
                    Duration::from_secs(6 * 3600)
                } else {
                    Duration::from_secs(6 * 24 * 3600)
                };
                Ok(b.get_object(Some(&rc), key).sign(expiry).to_string())
            }
        }
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn ini_parses_profiles_and_config_prefix() {
            let dir = std::env::temp_dir().join("geopq_awstest");
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("ini");
            std::fs::write(
                &p,
                "[default]\naws_access_key_id = AKIA1\n\n[profile geo]\nregion = eu-west-3\n; comment\n",
            )
            .unwrap();
            let m = super::ini(&p);
            assert_eq!(m["default"]["aws_access_key_id"], "AKIA1");
            assert_eq!(m["geo"]["region"], "eu-west-3");
        }

        #[test]
        fn presign_shape() {
            // Uses explicit env-independent parts: build via rusty_s3 directly
            // mirrors presign()'s signed branch.
            use rusty_s3::S3Action;
            let b = rusty_s3::Bucket::new(
                "https://s3.eu-west-3.amazonaws.com".parse().unwrap(),
                rusty_s3::UrlStyle::VirtualHost,
                "my-bucket",
                "eu-west-3",
            )
            .unwrap();
            let c = rusty_s3::Credentials::new("AKIAEXAMPLE", "secret");
            let url = b
                .get_object(Some(&c), "path/to/data.parquet")
                .sign(std::time::Duration::from_secs(3600))
                .to_string();
            assert!(url.starts_with("https://my-bucket.s3.eu-west-3.amazonaws.com/path/to/data.parquet?"));
            assert!(url.contains("X-Amz-Signature="));
            assert!(url.contains("X-Amz-Expires=3600"));
        }

        #[test]
        fn s3_uri_validation() {
            assert!(super::presign("s3://bucket-only", None, None).is_err());
            assert!(super::presign("http://x/y", None, None).is_err());
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
        // Error answers also carry a Content-Length (their body's); only a
        // 200 tells us the object size.
        if res.status() == 200 {
            if let Some(len) = header(&res, "content-length").and_then(|v| v.parse::<u64>().ok()) {
                if len > 0 {
                    return Ok(len);
                }
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
    log::trace!("range GET {start}..={end_inclusive} ({expect} B)");
    let res = http_agent()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .header("Range", &format!("bytes={start}-{end_inclusive}"))
        .call()
        .map_err(|e| ParquetError::General(format!("range request failed: {e}")))?;
    let ranged = match res.status().as_u16() {
        206 => true,
        // Whole-file answer is only usable from offset 0.
        200 if start == 0 => false,
        s => {
            return Err(ParquetError::General(format!(
                "server rejected range request ({s})"
            )))
        }
    };
    // Read one byte beyond the expected length: observing the body's EOF
    // is what lets the agent return the connection to its pool — capping
    // exactly at `expect` closes the connection and forces a new TLS
    // handshake per request (~1 s each against S3).
    let cap = expect as u64 + u64::from(ranged);
    let mut buf = Vec::with_capacity(expect);
    let read = res
        .into_body()
        .into_reader()
        .take(cap)
        .read_to_end(&mut buf)
        .map_err(|e| ParquetError::General(format!("range read: {e}")))?;
    if read < expect || (ranged && read != expect) {
        return Err(ParquetError::EOF(format!(
            "expected {expect} bytes at offset {start}, got {read}"
        )));
    }
    buf.truncate(expect);
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
