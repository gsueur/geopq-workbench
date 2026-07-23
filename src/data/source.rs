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

const USER_AGENT: &str =
    concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

/// Shared agent: connection pooling across range requests. HTTP error
/// statuses come back as responses (not `Err`), so callers can read
/// headers like `x-amz-bucket-region` from 301/403 answers.
pub(crate) fn http_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            // A stalled connection must never hang a load forever:
            // cancellation flags are only checked between reads, so a
            // blocked read would be uninterruptible. Individual range
            // requests are bounded (windows ≤ 8 MB, prefetch segments
            // split to ≤ 32 MB), so these are ample even on slow links.
            .timeout_resolve(Some(std::time::Duration::from_secs(10)))
            .timeout_connect(Some(std::time::Duration::from_secs(10)))
            .timeout_recv_response(Some(std::time::Duration::from_secs(30)))
            .timeout_recv_body(Some(std::time::Duration::from_secs(120)))
            .build()
            .into()
    })
}

#[derive(Clone, Debug)]
pub enum Source {
    Local(PathBuf),
    /// A local directory holding a (possibly hive-partitioned) multi-file
    /// GeoParquet dataset. Opened as one layer; the store reads through
    /// per-file `Local` sources, so `open()` is never valid on this.
    Dir(PathBuf),
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
    /// A STAC type collection (collection.json URL): a multi-file remote
    /// dataset resolved at open into the part files intersecting the
    /// viewport; the store reads through per-part `Remote` sources, so
    /// `open()` is never valid on this (like `Dir`).
    Stac { url: String, name: String },
    /// A fixed set of same-schema remote parquet parts opened as one
    /// layer (repository "all states" loads). Hive `key=value` URL path
    /// segments become partition columns; parts resolve at open, so
    /// `open()` is never valid on this (like `Dir` and `Stac`).
    Multi { name: String, urls: Vec<String> },
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

    /// An `s3://` URI naming a dataset rather than one object: a
    /// trailing slash (prefix), a bucket with no key, or a `*` glob
    /// pattern (`s3://bucket/d/state=*/roads.parquet`). Opened by
    /// listing.
    pub fn is_s3_prefix(&self) -> bool {
        match self {
            Source::S3 { uri, .. } => {
                let rest = uri.strip_prefix("s3://").unwrap_or(uri);
                uri.ends_with('/') || !rest.contains('/') || rest.contains('*')
            }
            _ => false,
        }
    }

    /// Resolve credentials/length as needed (no-op when already resolved).
    /// Network + credential-file reads — run off the UI thread.
    pub fn resolve(self) -> Result<Source, String> {
        if self.is_s3_prefix() {
            // Prefix datasets resolve per part at open, not here.
            return Ok(self);
        }
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
        !matches!(self, Source::Local(_) | Source::Dir(_))
    }

    /// Full path / URL / S3 URI, for tooltips and error messages (never
    /// the presigned URL — it embeds a signed access grant).
    pub fn label(&self) -> String {
        match self {
            Source::Local(p) | Source::Dir(p) => p.display().to_string(),
            Source::Remote { url, .. } => url.clone(),
            Source::S3 { uri, .. } => uri.clone(),
            Source::Stac { url, .. } => url.clone(),
            Source::Multi { name, urls } => format!("{name} ({} parts)", urls.len()),
        }
    }

    /// Short display name (file stem / directory name / last URL segment).
    pub fn name(&self) -> String {
        match self {
            Source::Local(p) => p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "layer".into()),
            Source::Dir(p) => p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "dataset".into()),
            Source::Remote { url, .. } | Source::S3 { uri: url, .. } => url
                .trim_end_matches('/')
                .split('/')
                .next_back()
                .filter(|s| !s.is_empty())
                .map(|s| s.trim_end_matches(".parquet").to_string())
                .unwrap_or_else(|| "remote".into()),
            Source::Stac { name, .. } | Source::Multi { name, .. } => name.clone(),
        }
    }

    pub fn size(&self) -> u64 {
        match self {
            Source::Local(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            // Aggregated per fragment.
            Source::Dir(_) | Source::Stac { .. } | Source::Multi { .. } => 0,
            Source::Remote { len, .. } | Source::S3 { len, .. } => *len,
        }
    }

    pub fn open(&self) -> Result<SourceReader, String> {
        if self.is_s3_prefix() {
            return Err(format!("{} is an S3 prefix, not a file", self.label()));
        }
        match self {
            Source::Dir(p) => Err(format!(
                "{} is a dataset directory, not a file",
                p.display()
            )),
            Source::Stac { url, .. } => {
                Err(format!("{url} is a STAC collection, not a file"))
            }
            Source::Multi { name, .. } => {
                Err(format!("{name} is a multi-part dataset, not a file"))
            }
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

/// Keep signed query parameters out of error strings: replace each
/// presigned query string with `?<presigned>` while preserving the rest of
/// the message — errors are formatted "{url}: {cause}", and the cause
/// (e.g. "(status 403)") is what distinguishes expired credentials from an
/// unreachable host.
fn redact_presign(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let mut rest = msg;
    while let Some(i) = rest.find("?X-Amz") {
        out.push_str(&rest[..i]);
        out.push_str("?<presigned>");
        let tail = &rest[i..];
        // The query string ends at the first character that cannot appear
        // in it: whitespace, or the ": " separating the URL from the error
        // cause (AWS presign params are percent-encoded, so neither occurs
        // inside them).
        let end = tail
            .char_indices()
            .find(|&(j, c)| c.is_whitespace() || (c == ':' && tail[j + 1..].starts_with(' ')))
            .map(|(j, _)| j)
            .unwrap_or(tail.len());
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

/// AWS credential/profile handling and S3 presigning. Kept deliberately
/// small: static keys (+ optional session token) from `~/.aws` files or
/// the environment, region from profile/env/bucket probe, anonymous
/// fallback for public buckets. No SSO/IMDS credential providers.
pub mod aws {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn dirs_home() -> Option<PathBuf> {
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
    }

    /// `~/.aws/credentials`, or `AWS_SHARED_CREDENTIALS_FILE`.
    fn credentials_file() -> PathBuf {
        std::env::var_os("AWS_SHARED_CREDENTIALS_FILE")
            .map(PathBuf::from)
            .or_else(|| dirs_home().map(|h| h.join(".aws").join("credentials")))
            .unwrap_or_default()
    }

    /// `~/.aws/config`, or `AWS_CONFIG_FILE` (independent of the
    /// credentials file — the two need not share a directory).
    fn config_file() -> PathBuf {
        std::env::var_os("AWS_CONFIG_FILE")
            .map(PathBuf::from)
            .or_else(|| dirs_home().map(|h| h.join(".aws").join("config")))
            .unwrap_or_default()
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
        let mut names: Vec<String> = ini(&credentials_file())
            .into_keys()
            .chain(ini(&config_file()).into_keys())
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
            let mut merged = ini(&config_file());
            for (sec, kv) in ini(&credentials_file()) {
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
        let files_region = |name: &str| -> Option<String> {
            let config = ini(&config_file());
            let creds = ini(&credentials_file());
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

    /// RFC 3986 path encoding of an object key: percent-encode every byte
    /// outside the unreserved set, preserving `/` separators. Produces the
    /// same encoding rusty_s3 applies on the signed branch, so anonymous
    /// and presigned URLs address the same object (a raw '#' would
    /// truncate the key to a fragment, a raw space breaks URI parsing).
    fn encode_key_path(key: &str) -> String {
        let mut out = String::with_capacity(key.len());
        for &b in key.as_bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                    out.push(b as char)
                }
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }

    /// Custom endpoints (MinIO, Wasabi, ...): explicit field first, then
    /// env, then the profile's `endpoint_url` (AWS CLI v2 convention).
    fn resolve_endpoint(profile: Option<&str>, endpoint: Option<&str>) -> Option<String> {
        let normalize = |e: &str| -> String {
            let e = e.trim().trim_end_matches('/');
            if e.starts_with("http://") || e.starts_with("https://") {
                e.to_string()
            } else {
                format!("https://{e}")
            }
        };
        endpoint
            .filter(|e| !e.trim().is_empty())
            .map(normalize)
            .or_else(|| std::env::var("AWS_ENDPOINT_URL").ok().map(|e| normalize(&e)))
            .or_else(|| {
                let name = profile
                    .map(str::to_string)
                    .or_else(|| std::env::var("AWS_PROFILE").ok())
                    .unwrap_or_else(|| "default".into());
                let config = ini(&config_file());
                let creds_file = ini(&credentials_file());
                config
                    .get(&name)
                    .and_then(|s| s.get("endpoint_url").cloned())
                    .or_else(|| {
                        creds_file.get(&name).and_then(|s| s.get("endpoint_url").cloned())
                    })
                    .map(|e| normalize(&e))
            })
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
        let endpoint_env = resolve_endpoint(profile, endpoint);
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
                // Anonymous: plain object URL, key encoded like the
                // signed branch.
                let key = encode_key_path(key);
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

    /// Percent-decode a ListObjectsV2 key (`encoding-type=url` responses).
    fn percent_decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%'
                && i + 2 < bytes.len()
                && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16)
            {
                out.push(v);
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// List every object under an `s3://bucket/prefix` (paginated
    /// ListObjectsV2; anonymous or signed like `presign`). Returns
    /// `(key, size)` pairs in listing order.
    pub fn list_prefix(
        uri: &str,
        profile: Option<&str>,
        endpoint: Option<&str>,
    ) -> Result<Vec<(String, u64)>, String> {
        let rest = uri
            .strip_prefix("s3://")
            .ok_or_else(|| format!("not an s3:// URI: {uri}"))?;
        let (bucket_name, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p),
            None => (rest, ""),
        };
        if bucket_name.is_empty() {
            return Err(format!("expected s3://bucket/prefix/, got {uri}"));
        }

        let creds = credentials(profile);
        if let Some(p) = profile
            && creds.is_none()
        {
            return Err(format!("profile '{p}' has no static credentials in ~/.aws"));
        }
        let endpoint_env = resolve_endpoint(profile, endpoint);
        let region = region(profile, bucket_name, endpoint_env.is_some());
        let (endpoint_url, style) = match &endpoint_env {
            Some(e) => (e.clone(), rusty_s3::UrlStyle::Path),
            None => (
                format!("https://s3.{region}.amazonaws.com"),
                rusty_s3::UrlStyle::VirtualHost,
            ),
        };
        let b = rusty_s3::Bucket::new(
            endpoint_url.parse().map_err(|e| format!("bad S3 endpoint: {e}"))?,
            style,
            bucket_name.to_string(),
            region,
        )
        .map_err(|e| format!("bad S3 bucket: {e}"))?;
        let rc = creds.map(|c| match &c.token {
            Some(t) => rusty_s3::Credentials::new_with_token(&c.key, &c.secret, t),
            None => rusty_s3::Credentials::new(&c.key, &c.secret),
        });

        let mut out = Vec::new();
        let mut token: Option<String> = None;
        loop {
            use rusty_s3::S3Action;
            let mut action = b.list_objects_v2(rc.as_ref());
            if !prefix.is_empty() {
                action.with_prefix(prefix);
            }
            if let Some(t) = &token {
                action.with_continuation_token(t.as_str());
            }
            let url = action.sign(Duration::from_secs(300)).to_string();
            let res = super::http_agent()
                .get(&url)
                .header("User-Agent", super::USER_AGENT)
                .call()
                .map_err(|e| format!("S3 listing failed: {}", super::redact_presign(&e.to_string())))?;
            if res.status() != 200 {
                return Err(format!("{uri}: S3 listing returned HTTP {}", res.status()));
            }
            let body = res
                .into_body()
                .read_to_string()
                .map_err(|e| format!("S3 listing read: {e}"))?;
            let parsed = rusty_s3::actions::ListObjectsV2::parse_response(&body)
                .map_err(|e| format!("S3 listing parse: {e}"))?;
            for c in parsed.contents {
                out.push((percent_decode(&c.key), c.size));
            }
            match parsed.next_continuation_token {
                Some(t) if !t.is_empty() => token = Some(t),
                _ => break,
            }
            if out.len() > 100_000 {
                return Err(format!("{uri}: prefix lists over 100k objects"));
            }
        }
        Ok(out)
    }

    /// Multipart part size, and the single-PUT threshold. S3's minimum
    /// part is 5 MB; 64 MB keeps a 5 GB file under 80 parts.
    const UPLOAD_PART_SIZE: usize = 64 * 1024 * 1024;

    /// Upload one local file to `s3://bucket/key`. Requires credentials
    /// (there is no anonymous write); endpoint resolution matches the
    /// read path, so R2/MinIO work the same way. `progress(sent, total)`
    /// runs after every uploaded chunk.
    pub fn upload_file(
        local: &Path,
        uri: &str,
        profile: Option<&str>,
        endpoint: Option<&str>,
        progress: &dyn Fn(u64, u64),
    ) -> Result<(), String> {
        upload_file_with(local, uri, profile, endpoint, UPLOAD_PART_SIZE, progress)
    }

    fn upload_file_with(
        local: &Path,
        uri: &str,
        profile: Option<&str>,
        endpoint: Option<&str>,
        part_size: usize,
        progress: &dyn Fn(u64, u64),
    ) -> Result<(), String> {
        let rest = uri
            .strip_prefix("s3://")
            .ok_or_else(|| format!("not an s3:// URI: {uri}"))?;
        let (bucket_name, key) = rest
            .split_once('/')
            .filter(|(b, k)| !b.is_empty() && !k.is_empty() && !k.ends_with('/'))
            .ok_or_else(|| format!("expected s3://bucket/key, got {uri}"))?;

        let c = credentials(profile).ok_or_else(|| {
            "uploading requires credentials: add a profile in ~/.aws \
             (for R2, an API token with write scope)"
                .to_string()
        })?;
        let rc = match &c.token {
            Some(t) => rusty_s3::Credentials::new_with_token(&c.key, &c.secret, t),
            None => rusty_s3::Credentials::new(&c.key, &c.secret),
        };
        let endpoint_env = resolve_endpoint(profile, endpoint);
        let region = region(profile, bucket_name, endpoint_env.is_some());
        let (endpoint_url, style) = match &endpoint_env {
            Some(e) => (e.clone(), rusty_s3::UrlStyle::Path),
            None => (
                format!("https://s3.{region}.amazonaws.com"),
                rusty_s3::UrlStyle::VirtualHost,
            ),
        };
        let b = rusty_s3::Bucket::new(
            endpoint_url.parse().map_err(|e| format!("bad S3 endpoint: {e}"))?,
            style,
            bucket_name.to_string(),
            region,
        )
        .map_err(|e| format!("bad S3 bucket: {e}"))?;
        upload_to_bucket(&b, &rc, key, uri, local, part_size, progress)
    }

    /// The transport half of the upload, bucket and credentials already
    /// resolved (separated so tests can target a local fake endpoint).
    fn upload_to_bucket(
        b: &rusty_s3::Bucket,
        rc: &rusty_s3::Credentials,
        key: &str,
        uri: &str,
        local: &Path,
        part_size: usize,
        progress: &dyn Fn(u64, u64),
    ) -> Result<(), String> {
        use std::io::Read as _;

        use rusty_s3::S3Action;

        let total = std::fs::metadata(local)
            .map_err(|e| format!("{}: {e}", local.display()))?
            .len();
        let mut f = std::fs::File::open(local)
            .map_err(|e| format!("{}: {e}", local.display()))?;
        let expiry = Duration::from_secs(3600);
        let put = |url: String, body: &[u8]| {
            super::http_agent()
                .put(&url)
                .header("User-Agent", super::USER_AGENT)
                .send(body)
                .map_err(|e| {
                    format!("upload failed: {}", super::redact_presign(&e.to_string()))
                })
        };

        if total as usize <= part_size {
            let mut buf = Vec::with_capacity(total as usize);
            f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            let url = b.put_object(Some(rc), key).sign(expiry).to_string();
            let res = put(url, &buf)?;
            if res.status() != 200 {
                return Err(format!("{uri}: upload returned HTTP {}", res.status()));
            }
            progress(total, total);
            return Ok(());
        }

        // Multipart: create, upload parts, complete; abort on any error
        // so half-uploads don't linger (and bill) in the bucket.
        let url = b
            .create_multipart_upload(Some(rc), key)
            .sign(expiry)
            .to_string();
        let res = super::http_agent()
            .post(&url)
            .header("User-Agent", super::USER_AGENT)
            .send(&[][..])
            .map_err(|e| {
                format!("multipart create: {}", super::redact_presign(&e.to_string()))
            })?;
        if res.status() != 200 {
            return Err(format!(
                "{uri}: multipart create returned HTTP {}",
                res.status()
            ));
        }
        let body = res.into_body().read_to_string().map_err(|e| e.to_string())?;
        let created = rusty_s3::actions::CreateMultipartUpload::parse_response(&body)
            .map_err(|e| format!("multipart create parse: {e}"))?;
        let upload_id = created.upload_id().to_string();

        let abort = |reason: String| -> String {
            let url = b
                .abort_multipart_upload(Some(rc), key, &upload_id)
                .sign(expiry)
                .to_string();
            let _ = super::http_agent()
                .delete(&url)
                .header("User-Agent", super::USER_AGENT)
                .call();
            reason
        };

        let mut etags: Vec<String> = Vec::new();
        let mut sent: u64 = 0;
        let mut buf = vec![0u8; part_size];
        for part_no in 1u16..=10_000 {
            let mut filled = 0usize;
            while filled < buf.len() {
                match f.read(&mut buf[filled..]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    Err(e) => return Err(abort(e.to_string())),
                }
            }
            if filled == 0 {
                break;
            }
            let url = b
                .upload_part(Some(rc), key, part_no, &upload_id)
                .sign(expiry)
                .to_string();
            let res = match put(url, &buf[..filled]) {
                Ok(r) => r,
                Err(e) => return Err(abort(e)),
            };
            if res.status() != 200 {
                return Err(abort(format!("part {part_no}: HTTP {}", res.status())));
            }
            let Some(etag) = super::header(&res, "etag").filter(|t| !t.is_empty()) else {
                return Err(abort(format!("part {part_no}: no ETag in response")));
            };
            etags.push(etag);
            sent += filled as u64;
            progress(sent.min(total), total);
            if filled < buf.len() {
                break;
            }
        }
        let action = b.complete_multipart_upload(
            Some(rc),
            key,
            &upload_id,
            etags.iter().map(|s| s.as_str()),
        );
        let url = action.sign(expiry).to_string();
        let xml = action.body();
        let res = super::http_agent()
            .post(&url)
            .header("User-Agent", super::USER_AGENT)
            .send(xml.as_str())
            .map_err(|e| {
                abort(format!(
                    "complete: {}",
                    super::redact_presign(&e.to_string())
                ))
            })?;
        // S3 can answer Complete with 200 + an <Error> body.
        let status = res.status();
        let text = res.into_body().read_to_string().unwrap_or_default();
        if status != 200 || text.contains("<Error>") {
            return Err(abort(format!("{uri}: complete failed (HTTP {status}): {text}")));
        }
        Ok(())
    }

    /// Upload every file under `root` to `s3://bucket/prefix/…`,
    /// preserving relative paths (partitioned optimize output).
    /// `progress(sent, total, current_file)` spans the whole tree.
    /// Returns the number of files uploaded.
    pub fn upload_tree(
        root: &Path,
        uri: &str,
        profile: Option<&str>,
        endpoint: Option<&str>,
        progress: &dyn Fn(u64, u64, &str),
    ) -> Result<usize, String> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
            let mut es: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .collect();
            es.sort();
            for p in es {
                if p.is_dir() {
                    walk(&p, out)?;
                } else {
                    out.push(p);
                }
            }
            Ok(())
        }
        let prefix = uri.trim_end_matches('/');
        let mut files = Vec::new();
        walk(root, &mut files)?;
        if files.is_empty() {
            return Err(format!("nothing to upload under {}", root.display()));
        }
        let total: u64 = files
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();
        let mut done: u64 = 0;
        for p in &files {
            let rel = p
                .strip_prefix(root)
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/");
            let dst = format!("{prefix}/{rel}");
            upload_file(p, &dst, profile, endpoint, &|sent, _| {
                progress(done + sent, total, &rel)
            })?;
            done += std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        }
        Ok(files.len())
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

        #[test]
        fn percent_decode_listing_keys() {
            assert_eq!(super::percent_decode("a/b%20c%2Bd.parquet"), "a/b c+d.parquet");
            assert_eq!(super::percent_decode("plain/key.parquet"), "plain/key.parquet");
            // Malformed escapes pass through untouched.
            assert_eq!(super::percent_decode("bad%2"), "bad%2");
        }

        #[test]
        fn listing_xml_parses() {
            let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>b</Name><Prefix>d/</Prefix><KeyCount>2</KeyCount><MaxKeys>1000</MaxKeys>
  <IsTruncated>false</IsTruncated>
  <Contents><Key>d/state%3DMA/part-0.parquet</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>&quot;x&quot;</ETag><Size>123</Size></Contents>
  <Contents><Key>d/state%3DNH/part-0.parquet</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>&quot;y&quot;</ETag><Size>456</Size></Contents>
</ListBucketResult>"#;
            let r = rusty_s3::actions::ListObjectsV2::parse_response(xml).unwrap();
            assert_eq!(r.contents.len(), 2);
            assert_eq!(super::percent_decode(&r.contents[0].key), "d/state=MA/part-0.parquet");
            assert_eq!(r.contents[1].size, 456);
            assert!(r.next_continuation_token.is_none());
        }

        /// Fake S3 endpoint accepting PUT / multipart flows; uploads land
        /// in a temp dir keyed by object path.
        fn spawn_upload_server(store: std::path::PathBuf) -> String {
            use std::io::{Read as _, Write as _};
            use std::net::TcpListener;

            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            std::thread::spawn(move || {
                for conn in listener.incoming() {
                    let Ok(mut conn) = conn else { continue };
                    let store = store.clone();
                    std::thread::spawn(move || {
                        let mut head = Vec::new();
                        let mut b = [0u8; 1];
                        while !head.ends_with(b"\r\n\r\n") && head.len() < 16384 {
                            match conn.read(&mut b) {
                                Ok(1) => head.push(b[0]),
                                _ => return,
                            }
                        }
                        let text = String::from_utf8_lossy(&head).into_owned();
                        let line1 = text.lines().next().unwrap_or_default().to_string();
                        let method = line1.split_whitespace().next().unwrap_or_default();
                        let target =
                            line1.split_whitespace().nth(1).unwrap_or_default().to_string();
                        let (path, query) = target.split_once('?').unwrap_or((&*target, ""));
                        let clen: usize = text
                            .lines()
                            .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                            .and_then(|l| l.split(':').nth(1))
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        let mut body = vec![0u8; clen];
                        if clen > 0 && conn.read_exact(&mut body).is_err() {
                            return;
                        }
                        let rel = super::percent_decode(path.trim_start_matches('/'));
                        let obj = store.join(&rel);
                        let reply = |conn: &mut std::net::TcpStream,
                                     status: &str,
                                     extra: &str,
                                     body: &str| {
                            let _ = write!(
                                conn,
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n{body}",
                                body.len()
                            );
                        };
                        match method {
                            "POST" if query.contains("uploads") => {
                                // Create multipart.
                                let xml = format!(
                                    "<InitiateMultipartUploadResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Bucket>b</Bucket><Key>{rel}</Key><UploadId>testupload</UploadId></InitiateMultipartUploadResult>"
                                );
                                reply(&mut conn, "200 OK", "", &xml);
                            }
                            "PUT" if query.contains("partNumber=") => {
                                let n: u32 = query
                                    .split('&')
                                    .find_map(|kv| kv.strip_prefix("partNumber="))
                                    .and_then(|v| v.parse().ok())
                                    .unwrap_or(0);
                                let part = obj.with_extension(format!("part{n}"));
                                std::fs::create_dir_all(part.parent().unwrap()).unwrap();
                                std::fs::write(&part, &body).unwrap();
                                reply(&mut conn, "200 OK", &format!("ETag: \"p{n}\"\r\n"), "");
                            }
                            "POST" if query.contains("uploadId=") => {
                                // Complete: stitch parts in order.
                                let mut out = Vec::new();
                                for n in 1u32.. {
                                    let part = obj.with_extension(format!("part{n}"));
                                    match std::fs::read(&part) {
                                        Ok(mut d) => out.append(&mut d),
                                        Err(_) => break,
                                    }
                                }
                                std::fs::create_dir_all(obj.parent().unwrap()).unwrap();
                                std::fs::write(&obj, &out).unwrap();
                                reply(&mut conn, "200 OK", "", "<CompleteMultipartUploadResult/>");
                            }
                            "PUT" => {
                                std::fs::create_dir_all(obj.parent().unwrap()).unwrap();
                                std::fs::write(&obj, &body).unwrap();
                                reply(&mut conn, "200 OK", "ETag: \"whole\"\r\n", "");
                            }
                            "DELETE" => reply(&mut conn, "204 No Content", "", ""),
                            _ => reply(&mut conn, "404 Not Found", "", ""),
                        }
                    });
                }
            });
            format!("http://127.0.0.1:{port}")
        }

        #[test]
        fn upload_single_put_and_multipart() {
            let store = std::env::temp_dir().join("geopq_upload_srv");
            let _ = std::fs::remove_dir_all(&store);
            std::fs::create_dir_all(&store).unwrap();
            let endpoint = spawn_upload_server(store.clone());
            let bucket = rusty_s3::Bucket::new(
                endpoint.parse().unwrap(),
                rusty_s3::UrlStyle::Path,
                "bucket".to_string(),
                "us-east-1".to_string(),
            )
            .unwrap();
            let rc = rusty_s3::Credentials::new("test-key", "test-secret");

            // Single PUT (fits in one part).
            let src = store.join("src_small.bin");
            std::fs::write(&src, vec![7u8; 10_000]).unwrap();
            let seen = std::sync::Mutex::new((0u64, 0u64));
            super::upload_to_bucket(&bucket, &rc, "d/state=MA/f.parquet", "s3://bucket/d/state=MA/f.parquet", &src, 1 << 20, &|s, t| {
                *seen.lock().unwrap() = (s, t);
            })
            .unwrap();
            assert_eq!(*seen.lock().unwrap(), (10_000, 10_000));
            assert_eq!(
                std::fs::read(store.join("bucket/d/state=MA/f.parquet")).unwrap(),
                vec![7u8; 10_000]
            );

            // Multipart: 25 kB with 10 kB parts = 3 parts, byte-identical.
            let src = store.join("src_big.bin");
            let payload: Vec<u8> = (0..25_000u32).map(|i| (i % 251) as u8).collect();
            std::fs::write(&src, &payload).unwrap();
            super::upload_to_bucket(&bucket, &rc, "d/big.parquet", "s3://bucket/d/big.parquet", &src, 10_000, &|_, _| {})
                .unwrap();
            assert_eq!(std::fs::read(store.join("bucket/d/big.parquet")).unwrap(), payload);
        }

        /// Live anonymous listing against the public Overture bucket.
        /// Network — run explicitly with `cargo test -- --ignored`. The
        /// bucket only retains recent releases, so the date needs an
        /// occasional bump when this starts returning nothing.
        #[test]
        #[ignore = "network"]
        fn live_anonymous_prefix_listing() {
            let keys = super::list_prefix(
                "s3://overturemaps-us-west-2/release/2026-06-17.0/theme=divisions/type=division_area/",
                None,
                None,
            )
            .unwrap();
            assert!(!keys.is_empty());
            assert!(keys.iter().all(|(k, _)| k.contains("type=division_area")));
        }

        #[test]
        fn key_path_encoding() {
            // Unreserved characters and '/' pass through untouched.
            assert_eq!(
                super::encode_key_path("path/to/data-1.0_x~.parquet"),
                "path/to/data-1.0_x~.parquet"
            );
            // '#' would truncate to a fragment, '?' starts a query, a raw
            // space breaks parsing, '%' must not be double-decodable.
            assert_eq!(
                super::encode_key_path("a b#c?100%.parquet"),
                "a%20b%23c%3F100%25.parquet"
            );
            // Non-ASCII is encoded byte-wise as UTF-8.
            assert_eq!(super::encode_key_path("Zürich.parquet"), "Z%C3%BCrich.parquet");
        }

        #[test]
        fn anonymous_key_encoding_matches_signed_branch() {
            // The signed and anonymous URL forms must address the same
            // object: compare our encoder against rusty_s3's path.
            use rusty_s3::S3Action;
            let key = "path/to/héllo #1 100%.parquet";
            let b = rusty_s3::Bucket::new(
                "https://s3.eu-west-3.amazonaws.com".parse().unwrap(),
                rusty_s3::UrlStyle::VirtualHost,
                "my-bucket",
                "eu-west-3",
            )
            .unwrap();
            let c = rusty_s3::Credentials::new("AKIAEXAMPLE", "secret");
            let url = b
                .get_object(Some(&c), key)
                .sign(std::time::Duration::from_secs(60))
                .to_string();
            let path = url.split('?').next().unwrap();
            assert_eq!(
                path,
                format!(
                    "https://my-bucket.s3.eu-west-3.amazonaws.com/{}",
                    super::encode_key_path(key)
                )
            );
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
    use std::io::Write;
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
                    use std::io::{Read as _, Seek, SeekFrom};
                    let mut f = std::fs::File::open(&file).unwrap();
                    f.seek(SeekFrom::Start(start)).unwrap();
                    let mut pos = start;
                    let mut chunk = vec![0u8; 256 * 1024];
                    while pos <= end {
                        let take = ((end - pos + 1) as usize).min(chunk.len());
                        let read = f.read(&mut chunk[..take]).unwrap();
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

    /// Serve a directory tree (files resolved from the request path, 404
    /// on miss). Full GETs only — enough for STAC JSON documents plus
    /// small parquet fixtures. Returns the base URL (no trailing slash).
    pub fn spawn_dir(root: PathBuf) -> String {
        use std::io::Read as _;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                let root = root.clone();
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut b = [0u8; 1];
                    while !buf.ends_with(b"\r\n\r\n") && buf.len() < 8192 {
                        match conn.read(&mut b) {
                            Ok(1) => buf.push(b[0]),
                            _ => return,
                        }
                    }
                    let text = String::from_utf8_lossy(&buf);
                    let is_head = text.starts_with("HEAD");
                    let path = text
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("/")
                        .trim_start_matches('/')
                        .to_string();
                    // Fixture paths are test-authored; no traversal guard.
                    let body = std::fs::read(root.join(&path)).ok();
                    match body {
                        None => {
                            let _ = write!(
                                conn,
                                "HTTP/1.1 404 NF\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            );
                        }
                        Some(data) => {
                            let range = text
                                .lines()
                                .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                                .and_then(|l| l.split_once(':'))
                                .and_then(|(_, v)| v.trim().strip_prefix("bytes="))
                                .and_then(|v| v.split_once('-'))
                                .map(|(a, b)| {
                                    let start: usize = a.parse().unwrap_or(0);
                                    let end: usize =
                                        b.parse().unwrap_or(data.len() - 1);
                                    (start, end.min(data.len() - 1))
                                });
                            if is_head {
                                let _ = write!(
                                    conn,
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                    data.len()
                                );
                                return;
                            }
                            let (status, extra, slice) = match range {
                                Some((s, e)) => (
                                    "206 Partial Content",
                                    format!(
                                        "Content-Range: bytes {s}-{e}/{}\r\n",
                                        data.len()
                                    ),
                                    &data[s..=e],
                                ),
                                None => ("200 OK", String::new(), &data[..]),
                            };
                            let _ = write!(
                                conn,
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                slice.len()
                            );
                            let _ = conn.write_all(slice);
                        }
                    }
                });
            }
        });
        format!("http://127.0.0.1:{port}")
    }
}

#[cfg(test)]
mod redact_tests {
    use super::redact_presign;

    #[test]
    fn keeps_error_cause_after_query() {
        let msg = "https://b.s3.eu-west-3.amazonaws.com/k.parquet?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA%2F20260716%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Signature=abc123: server does not support range requests (status 403)";
        let red = redact_presign(msg);
        assert_eq!(
            red,
            "https://b.s3.eu-west-3.amazonaws.com/k.parquet?<presigned>: server does not support range requests (status 403)"
        );
        assert!(red.contains("(status 403)"));
        assert!(!red.contains("X-Amz"));
    }

    #[test]
    fn message_ending_at_query() {
        assert_eq!(
            redact_presign("cannot reach https://x/y.parquet?X-Amz-Signature=abc"),
            "cannot reach https://x/y.parquet?<presigned>"
        );
    }

    #[test]
    fn query_followed_by_whitespace() {
        assert_eq!(
            redact_presign("https://x/y?X-Amz-Signature=abc timed out"),
            "https://x/y?<presigned> timed out"
        );
    }

    #[test]
    fn multiple_urls_redacted() {
        assert_eq!(
            redact_presign(
                "https://a/1?X-Amz-Signature=s1: moved to https://b/2?X-Amz-Signature=s2: gone"
            ),
            "https://a/1?<presigned>: moved to https://b/2?<presigned>: gone"
        );
    }

    #[test]
    fn no_presign_untouched() {
        assert_eq!(redact_presign("plain error: no url here"), "plain error: no url here");
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
