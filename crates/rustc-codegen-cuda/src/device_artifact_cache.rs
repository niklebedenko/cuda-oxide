//! Immutable content-addressed cache for final device-owner host objects and
//! their exact PTX audit inputs.
//!
//! The backend consults this cache only after rustc has collected the exact
//! reachable device graph and cargo-oxide has supplied its scoped codegen and
//! CUDA-tool provenance. Cached bytes are still parsed and checked against the
//! current owner entries before they can be attached to host codegen.

use fs2::FileExt as _;
use sha2::{Digest as _, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const CACHE_PROTOCOL: &str = "device-owner-object-v3";
const ARTIFACT_FILE: &str = "artifact.o";
const AUDIT_PTX_FILE: &str = "audit.ptx";
const AUDIT_PTX_BUNDLE_FILE: &str = "audit.ptx.bundle";
const MANIFEST_FILE: &str = "manifest-v1";
const MAX_MANIFEST_BYTES: u64 = 4096;
const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CACHE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
static PUBLICATION_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(crate) enum CacheRead {
    Miss,
    Hit(CachedArtifactObject),
    Corrupt(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuditArtifactKind {
    Ptx,
    PtxBundle,
}

impl AuditArtifactKind {
    fn manifest_value(self) -> &'static str {
        match self {
            Self::Ptx => "ptx",
            Self::PtxBundle => "ptx-bundle",
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Self::Ptx => AUDIT_PTX_FILE,
            Self::PtxBundle => AUDIT_PTX_BUNDLE_FILE,
        }
    }

    fn parse(value: &str) -> Result<Option<Self>, String> {
        match value {
            "none" => Ok(None),
            "ptx" => Ok(Some(Self::Ptx)),
            "ptx-bundle" => Ok(Some(Self::PtxBundle)),
            _ => Err(format!("invalid cache audit artifact kind {value:?}")),
        }
    }
}

#[derive(Debug)]
pub(crate) struct CachedAuditArtifact {
    pub(crate) kind: AuditArtifactKind,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct CachedArtifactObject {
    pub(crate) object: Vec<u8>,
    pub(crate) audit: Option<CachedAuditArtifact>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpectedArtifactEntry {
    pub(crate) symbol: String,
    pub(crate) kind: oxide_artifacts::ArtifactEntryKind,
    pub(crate) root_descriptor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpectedArtifactObject {
    pub(crate) bundle_name: String,
    pub(crate) target: String,
    pub(crate) compile_options: oxide_artifacts::ArtifactCompileOptions,
    pub(crate) entries: Vec<ExpectedArtifactEntry>,
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceArtifactObjectCache {
    root: PathBuf,
}

impl DeviceArtifactObjectCache {
    pub(crate) fn from_env() -> Result<Option<Self>, String> {
        let Some(root) = std::env::var_os(reserved_oxide_symbols::DEVICE_ARTIFACT_CACHE_DIR_ENV)
        else {
            return Ok(None);
        };
        if root.is_empty() {
            return Err(format!(
                "{} must be a non-empty absolute path",
                reserved_oxide_symbols::DEVICE_ARTIFACT_CACHE_DIR_ENV,
            ));
        }
        let root = PathBuf::from(root);
        if !root.is_absolute() || root.parent().is_none() {
            return Err(format!(
                "{} must be an absolute non-root path, got {}",
                reserved_oxide_symbols::DEVICE_ARTIFACT_CACHE_DIR_ENV,
                root.display(),
            ));
        }
        Ok(Some(Self { root }))
    }

    #[cfg(test)]
    fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub(crate) fn key<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
        let mut hash = Sha256::new();
        hash.update(CACHE_PROTOCOL.len().to_le_bytes());
        hash.update(CACHE_PROTOCOL.as_bytes());
        for part in parts {
            hash.update(part.len().to_le_bytes());
            hash.update(part);
        }
        digest_hex(hash.finalize().into())
    }

    pub(crate) fn load(&self, key: &str) -> CacheRead {
        if !is_digest(key) {
            return CacheRead::Corrupt(format!("invalid cache key {key:?}"));
        }
        let entry = self.entry_path(key);
        match fs::symlink_metadata(&entry) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => {
                return CacheRead::Corrupt(format!(
                    "cache entry is not a directory: {}",
                    entry.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return CacheRead::Miss;
            }
            Err(error) => {
                return CacheRead::Corrupt(format!(
                    "cannot inspect cache entry {}: {error}",
                    entry.display()
                ));
            }
        }

        match self.read_entry(&entry, key) {
            Ok(bytes) => CacheRead::Hit(bytes),
            Err(error) => CacheRead::Corrupt(error),
        }
    }

    pub(crate) fn publish_validated_artifact(
        &self,
        key: &str,
        bytes: &[u8],
        audit: Option<(AuditArtifactKind, &[u8])>,
        expected: &ExpectedArtifactObject,
    ) -> Result<(), String> {
        validate_artifact_object(bytes, expected)
            .map_err(|error| format!("refusing invalid device artifact publication: {error}"))?;
        self.publish_impl(key, bytes, audit, Some(expected))
    }

    #[cfg(test)]
    fn publish(
        &self,
        key: &str,
        bytes: &[u8],
        audit: Option<(AuditArtifactKind, &[u8])>,
    ) -> Result<(), String> {
        self.publish_impl(key, bytes, audit, None)
    }

    fn publish_impl(
        &self,
        key: &str,
        bytes: &[u8],
        audit: Option<(AuditArtifactKind, &[u8])>,
        expected: Option<&ExpectedArtifactObject>,
    ) -> Result<(), String> {
        if !is_digest(key) {
            return Err(format!("refusing invalid cache key {key:?}"));
        }
        if bytes.is_empty() || bytes.len() as u64 > MAX_ARTIFACT_BYTES {
            return Err(format!(
                "refusing device artifact cache object with {} bytes",
                bytes.len()
            ));
        }

        let family = self.root.join(CACHE_PROTOCOL);
        fs::create_dir_all(&family).map_err(|error| {
            format!(
                "cannot create device artifact cache {}: {error}",
                family.display()
            )
        })?;
        let _publication_lock = PublicationLock::acquire(&family, key)?;
        let nonce = PUBLICATION_ID.fetch_add(1, Ordering::Relaxed);
        let staging = family.join(format!(".{key}.{}.{}.staging", std::process::id(), nonce));
        fs::create_dir(&staging).map_err(|error| {
            format!(
                "cannot create device artifact staging directory {}: {error}",
                staging.display()
            )
        })?;
        let mut cleanup = StagingCleanup::new(staging.clone());

        let artifact_digest = digest_hex(Sha256::digest(bytes).into());
        write_synced(&staging.join(ARTIFACT_FILE), bytes)?;
        let (audit_kind, audit_bytes, audit_digest) = match audit {
            Some((kind, audit_bytes)) => {
                if audit_bytes.is_empty() || audit_bytes.len() as u64 > MAX_ARTIFACT_BYTES {
                    return Err(format!(
                        "refusing device artifact cache audit input with {} bytes",
                        audit_bytes.len()
                    ));
                }
                write_synced(&staging.join(kind.file_name()), audit_bytes)?;
                (
                    kind.manifest_value(),
                    audit_bytes.len(),
                    digest_hex(Sha256::digest(audit_bytes).into()),
                )
            }
            None => ("none", 0, "none".to_string()),
        };
        let manifest = format!(
            "{CACHE_PROTOCOL}\nkey={key}\nbytes={}\nsha256={artifact_digest}\naudit_kind={audit_kind}\naudit_bytes={audit_bytes}\naudit_sha256={audit_digest}\n",
            bytes.len()
        );
        write_synced(&staging.join(MANIFEST_FILE), manifest.as_bytes())?;
        sync_directory(&staging)?;

        let entry = self.entry_path(key);
        if entry.exists() {
            let replace = match self.load(key) {
                CacheRead::Hit(existing)
                    if existing.object == bytes
                        && audit_matches(existing.audit.as_ref(), audit) =>
                {
                    return Ok(());
                }
                CacheRead::Hit(existing) => {
                    if expected.is_some_and(|expected| {
                        validate_artifact_object(&existing.object, expected).is_err()
                    }) {
                        true
                    } else {
                        return Err(format!(
                            "refusing different device artifact bytes for existing semantic cache key {key}; this indicates nondeterministic codegen or an incomplete cache identity"
                        ));
                    }
                }
                CacheRead::Corrupt(_) => true,
                CacheRead::Miss => false,
            };
            if replace {
                return replace_entry(
                    self,
                    &family,
                    &entry,
                    &staging,
                    &mut cleanup,
                    key,
                    nonce,
                    bytes,
                    audit,
                    expected,
                );
            }
        }

        match fs::rename(&staging, &entry) {
            Ok(()) => {
                cleanup.disarm();
                sync_directory(&family)?;
                Ok(())
            }
            Err(error) if entry.exists() => match self.load(key) {
                CacheRead::Hit(existing)
                    if existing.object == bytes
                        && audit_matches(existing.audit.as_ref(), audit) =>
                {
                    Ok(())
                }
                CacheRead::Hit(existing)
                    if expected.is_some_and(|expected| {
                        validate_artifact_object(&existing.object, expected).is_err()
                    }) =>
                {
                    replace_entry(
                        self,
                        &family,
                        &entry,
                        &staging,
                        &mut cleanup,
                        key,
                        nonce,
                        bytes,
                        audit,
                        expected,
                    )
                }
                CacheRead::Hit(_) => Err(format!(
                    "concurrent publication produced different device artifact bytes for semantic cache key {key}; preserving {} ({error})",
                    entry.display()
                )),
                CacheRead::Corrupt(reason) => Err(format!(
                    "concurrent device artifact publication left corrupt entry {} after {error}: {reason}",
                    entry.display()
                )),
                CacheRead::Miss => Err(format!(
                    "concurrent device artifact publication left no entry {} after {error}",
                    entry.display()
                )),
            },
            Err(error) => Err(format!(
                "cannot publish device artifact cache entry {}: {error}",
                entry.display()
            )),
        }
    }

    pub(crate) fn reclaim(&self, preserve_key: &str) -> Result<(), String> {
        self.reclaim_to_budget(preserve_key, MAX_CACHE_BYTES)
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        self.root.join(CACHE_PROTOCOL).join(key)
    }

    fn read_entry(&self, entry: &Path, key: &str) -> Result<CachedArtifactObject, String> {
        let manifest_path = entry.join(MANIFEST_FILE);
        require_regular_file(&manifest_path, MAX_MANIFEST_BYTES)?;
        let manifest = fs::read_to_string(&manifest_path).map_err(|error| {
            format!(
                "cannot read cache manifest {}: {error}",
                manifest_path.display()
            )
        })?;
        let parsed = ParsedManifest::parse(&manifest)?;
        if parsed.key != key {
            return Err(format!(
                "cache manifest key {} does not match directory key {key}",
                parsed.key
            ));
        }

        let artifact_path = entry.join(ARTIFACT_FILE);
        let metadata = require_regular_file(&artifact_path, MAX_ARTIFACT_BYTES)?;
        if metadata.len() != parsed.bytes {
            return Err(format!(
                "cache object {} has {} bytes, manifest declares {}",
                artifact_path.display(),
                metadata.len(),
                parsed.bytes
            ));
        }
        let bytes = fs::read(&artifact_path).map_err(|error| {
            format!(
                "cannot read cache object {}: {error}",
                artifact_path.display()
            )
        })?;
        let actual = digest_hex(Sha256::digest(&bytes).into());
        if actual != parsed.sha256 {
            return Err(format!(
                "cache object {} digest mismatch: manifest={}, actual={actual}",
                artifact_path.display(),
                parsed.sha256
            ));
        }
        let audit = match parsed.audit_kind {
            Some(kind) => {
                let audit_path = entry.join(kind.file_name());
                let metadata = require_regular_file(&audit_path, MAX_ARTIFACT_BYTES)?;
                if metadata.len() != parsed.audit_bytes {
                    return Err(format!(
                        "cache audit input {} has {} bytes, manifest declares {}",
                        audit_path.display(),
                        metadata.len(),
                        parsed.audit_bytes
                    ));
                }
                let audit_bytes = fs::read(&audit_path).map_err(|error| {
                    format!(
                        "cannot read cache audit input {}: {error}",
                        audit_path.display()
                    )
                })?;
                let actual = digest_hex(Sha256::digest(&audit_bytes).into());
                if actual != parsed.audit_sha256 {
                    return Err(format!(
                        "cache audit input {} digest mismatch: manifest={}, actual={actual}",
                        audit_path.display(),
                        parsed.audit_sha256
                    ));
                }
                Some(CachedAuditArtifact {
                    kind,
                    bytes: audit_bytes,
                })
            }
            None => None,
        };
        Ok(CachedArtifactObject {
            object: bytes,
            audit,
        })
    }

    fn reclaim_to_budget(&self, preserve_key: &str, budget: u64) -> Result<(), String> {
        let family = self.root.join(CACHE_PROTOCOL);
        let entries = match fs::read_dir(&family) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "cannot inspect device artifact cache {}: {error}",
                    family.display()
                ));
            }
        };
        let mut usage = Vec::new();
        let mut total = 0_u64;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "cannot enumerate device artifact cache {}: {error}",
                    family.display()
                )
            })?;
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !is_digest(&name) {
                continue;
            }
            let metadata = fs::symlink_metadata(entry.path()).map_err(|error| {
                format!(
                    "cannot inspect cache entry {}: {error}",
                    entry.path().display()
                )
            })?;
            if !metadata.file_type().is_dir() {
                continue;
            }
            let bytes = [
                ARTIFACT_FILE,
                MANIFEST_FILE,
                AUDIT_PTX_FILE,
                AUDIT_PTX_BUNDLE_FILE,
            ]
            .into_iter()
            .filter_map(|file| fs::symlink_metadata(entry.path().join(file)).ok())
            .filter(|metadata| metadata.file_type().is_file())
            .map(|metadata| metadata.len())
            .sum::<u64>();
            total = total.saturating_add(bytes);
            usage.push((metadata.modified().ok(), name, entry.path(), bytes));
        }
        if total <= budget {
            return Ok(());
        }
        usage.sort_by_key(|(modified, name, _, _)| (*modified, name.clone()));
        for (_, name, path, bytes) in usage {
            if total <= budget {
                break;
            }
            if name == preserve_key {
                continue;
            }
            fs::remove_dir_all(&path).map_err(|error| {
                format!("cannot reclaim cache entry {}: {error}", path.display())
            })?;
            total = total.saturating_sub(bytes);
        }
        sync_directory(&family)
    }
}

pub(crate) fn validate_artifact_object(
    bytes: &[u8],
    expected: &ExpectedArtifactObject,
) -> Result<(), String> {
    let mut bundles = oxide_artifacts::read_artifact_bundles_from_object_bytes(bytes)
        .map_err(|error| format!("cannot parse cached device artifact object: {error}"))?;
    if bundles.len() != 1 {
        return Err(format!(
            "cached device artifact object contains {} bundles, expected exactly one",
            bundles.len()
        ));
    }
    let bundle = bundles.pop().expect("one bundle checked above");
    if bundle.name != expected.bundle_name {
        return Err(format!(
            "cached bundle name {:?} does not match {:?}",
            bundle.name, expected.bundle_name
        ));
    }
    if bundle.target != expected.target {
        return Err(format!(
            "cached bundle target {:?} does not match {:?}",
            bundle.target, expected.target
        ));
    }
    if bundle.compile_options != expected.compile_options {
        return Err("cached bundle compile options do not match this invocation".to_string());
    }
    if bundle.payloads.len() != 1
        || bundle.payloads[0].kind != oxide_artifacts::ArtifactPayloadKind::Cubin
        || bundle.payloads[0].bytes.is_empty()
    {
        return Err(
            "cached materialized bundle must contain exactly one non-empty cubin payload"
                .to_string(),
        );
    }
    let actual_entries = bundle
        .entries
        .into_iter()
        .map(|entry| ExpectedArtifactEntry {
            symbol: entry.symbol,
            kind: entry.kind,
            root_descriptor: entry.root_descriptor,
        })
        .collect::<Vec<_>>();
    if actual_entries != expected.entries {
        return Err("cached bundle entries do not match the collected device graph".to_string());
    }
    Ok(())
}

pub(crate) fn trace(event: &str, key: &str) {
    if std::env::var_os(reserved_oxide_symbols::DEVICE_ARTIFACT_CACHE_TRACE_ENV).is_some() {
        eprintln!("[rustc_codegen_cuda] device artifact cache {event}: {key}");
    }
}

pub(crate) fn file_sha256(path: &Path) -> Result<String, String> {
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", path.display()))?;
    let mut file = File::open(&canonical)
        .map_err(|error| format!("cannot open {}: {error}", canonical.display()))?;
    let mut hash = Sha256::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|error| format!("cannot read {}: {error}", canonical.display()))?;
        if read == 0 {
            break;
        }
        hash.update(&chunk[..read]);
    }
    Ok(digest_hex(hash.finalize().into()))
}

struct ParsedManifest<'a> {
    key: &'a str,
    bytes: u64,
    sha256: &'a str,
    audit_kind: Option<AuditArtifactKind>,
    audit_bytes: u64,
    audit_sha256: &'a str,
}

impl<'a> ParsedManifest<'a> {
    fn parse(value: &'a str) -> Result<Self, String> {
        let mut lines = value.lines();
        if lines.next() != Some(CACHE_PROTOCOL) {
            return Err("device artifact cache manifest protocol mismatch".to_string());
        }
        let key = parse_field(lines.next(), "key")?;
        let bytes = parse_field(lines.next(), "bytes")?
            .parse::<u64>()
            .map_err(|error| format!("invalid cache manifest byte count: {error}"))?;
        let sha256 = parse_field(lines.next(), "sha256")?;
        let audit_kind = AuditArtifactKind::parse(parse_field(lines.next(), "audit_kind")?)?;
        let audit_bytes = parse_field(lines.next(), "audit_bytes")?
            .parse::<u64>()
            .map_err(|error| format!("invalid cache audit byte count: {error}"))?;
        let audit_sha256 = parse_field(lines.next(), "audit_sha256")?;
        let audit_is_canonical = match audit_kind {
            Some(_) => audit_bytes > 0 && is_digest(audit_sha256),
            None => audit_bytes == 0 && audit_sha256 == "none",
        };
        if lines.next().is_some() || !is_digest(key) || !is_digest(sha256) || !audit_is_canonical {
            return Err("device artifact cache manifest is not canonical".to_string());
        }
        Ok(Self {
            key,
            bytes,
            sha256,
            audit_kind,
            audit_bytes,
            audit_sha256,
        })
    }
}

fn parse_field<'a>(line: Option<&'a str>, name: &str) -> Result<&'a str, String> {
    line.and_then(|line| {
        line.strip_prefix(name)
            .and_then(|value| value.strip_prefix('='))
    })
    .ok_or_else(|| format!("device artifact cache manifest is missing `{name}`"))
}

fn audit_matches(
    cached: Option<&CachedAuditArtifact>,
    proposed: Option<(AuditArtifactKind, &[u8])>,
) -> bool {
    match (cached, proposed) {
        (None, None) => true,
        (Some(cached), Some((kind, bytes))) => cached.kind == kind && cached.bytes == bytes,
        (None, Some(_)) | (Some(_), None) => false,
    }
}

fn replace_entry(
    cache: &DeviceArtifactObjectCache,
    family: &Path,
    entry: &Path,
    staging: &Path,
    cleanup: &mut StagingCleanup,
    key: &str,
    nonce: u64,
    proposed_bytes: &[u8],
    proposed_audit: Option<(AuditArtifactKind, &[u8])>,
    expected: Option<&ExpectedArtifactObject>,
) -> Result<(), String> {
    let quarantine = family.join(format!(".{key}.{}.{}.replaced", std::process::id(), nonce));
    fs::rename(entry, &quarantine).map_err(|error| {
        format!(
            "cannot quarantine invalid device artifact cache entry {}: {error}",
            entry.display()
        )
    })?;

    if let Ok(existing) = cache.read_entry(&quarantine, key) {
        let identical = existing.object == proposed_bytes
            && audit_matches(existing.audit.as_ref(), proposed_audit);
        let valid = expected
            .is_some_and(|expected| validate_artifact_object(&existing.object, expected).is_ok());
        if identical || valid {
            fs::rename(&quarantine, entry).map_err(|error| {
                format!(
                    "cannot restore validated device artifact cache entry {}: {error}",
                    entry.display()
                )
            })?;
            if identical {
                return Ok(());
            }
            return Err(format!(
                "refusing different device artifact bytes for existing semantic cache key {key}; preserving the first validated publication"
            ));
        }
    }

    fs::rename(staging, entry).map_err(|error| {
        let _ = fs::rename(&quarantine, entry);
        format!(
            "cannot publish device artifact cache entry {}: {error}",
            entry.display()
        )
    })?;
    cleanup.disarm();
    sync_directory(family)?;
    let _ = fs::remove_dir_all(quarantine);
    Ok(())
}

struct PublicationLock {
    _file: File,
}

impl PublicationLock {
    fn acquire(family: &Path, key: &str) -> Result<Self, String> {
        let path = family.join(format!(".{key}.lock"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                format!(
                    "cannot open device artifact publication lock {}: {error}",
                    path.display()
                )
            })?;
        file.lock_exclusive().map_err(|error| {
            format!(
                "cannot acquire device artifact publication lock {}: {error}",
                path.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

fn require_regular_file(path: &Path, max_bytes: u64) -> Result<fs::Metadata, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect cache file {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.len() == 0 || metadata.len() > max_bytes {
        return Err(format!(
            "cache file {} is not a regular bounded non-empty file ({} bytes)",
            path.display(),
            metadata.len()
        ));
    }
    Ok(metadata)
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create cache file {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write cache file {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync cache file {}: {error}", path.display()))
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("cannot sync cache directory {}: {error}", path.display()))
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

pub(crate) fn is_sha256_digest(value: &str) -> bool {
    is_digest(value)
}

fn digest_hex(digest: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut value = String::with_capacity(64);
    for byte in digest {
        write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

struct StagingCleanup {
    path: Option<PathBuf>,
}

impl StagingCleanup {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for StagingCleanup {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache(label: &str) -> (PathBuf, DeviceArtifactObjectCache) {
        let nonce = PUBLICATION_ID.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("cuda-oxide-{label}-{}-{nonce}", std::process::id()));
        (root.clone(), DeviceArtifactObjectCache::at(root))
    }

    fn materialized_object(target: &str, cubin: &[u8]) -> (Vec<u8>, ExpectedArtifactObject) {
        let compile_options = oxide_artifacts::ArtifactCompileOptions::new()
            .with_fma_contraction(false)
            .with_debug_policy(oxide_artifacts::ArtifactDebugPolicy::LineTables);
        let root_descriptor = "kernels\nrust-instance-v1:kernels::advance";
        let blob = oxide_artifacts::build_artifact_blob(
            &oxide_artifacts::ArtifactBundleSpec::new("kernels", target)
                .with_compile_options(compile_options)
                .with_payload(oxide_artifacts::ArtifactPayloadSpec::new(
                    oxide_artifacts::ArtifactPayloadKind::Cubin,
                    "kernels.cubin",
                    cubin,
                ))
                .with_entry(
                    oxide_artifacts::ArtifactEntrySpec::new(
                        "advance",
                        oxide_artifacts::ArtifactEntryKind::Kernel,
                    )
                    .with_root_descriptor(root_descriptor),
                ),
        )
        .unwrap();
        let object = oxide_artifacts::build_host_object_for_target(
            &blob,
            "x86_64-unknown-linux-gnu",
            Some("anchor"),
        )
        .unwrap();
        let expected = ExpectedArtifactObject {
            bundle_name: "kernels".to_string(),
            target: target.to_string(),
            compile_options,
            entries: vec![ExpectedArtifactEntry {
                symbol: "advance".to_string(),
                kind: oxide_artifacts::ArtifactEntryKind::Kernel,
                root_descriptor: Some(root_descriptor.to_string()),
            }],
        };
        (object, expected)
    }

    #[test]
    fn cold_miss_becomes_digest_checked_warm_hit() {
        let (root, cache) = temp_cache("owner-cache-hit");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice(), b"graph".as_slice()]);
        assert!(matches!(cache.load(&key), CacheRead::Miss));
        cache.publish(&key, b"host object", None).unwrap();
        assert!(
            matches!(cache.load(&key), CacheRead::Hit(value) if value.object == b"host object" && value.audit.is_none())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repeated_identical_publication_is_idempotent() {
        let (root, cache) = temp_cache("owner-cache-idempotent");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        let audit = Some((AuditArtifactKind::Ptx, b"same ptx".as_slice()));
        cache.publish(&key, b"same object", audit).unwrap();
        cache.publish(&key, b"same object", audit).unwrap();
        assert!(
            matches!(cache.load(&key), CacheRead::Hit(value) if value.object == b"same object" && value.audit.is_some())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn different_valid_publication_preserves_the_first_immutable_entry() {
        let (root, cache) = temp_cache("owner-cache-collision");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        cache.publish(&key, b"first object", None).unwrap();

        let error = cache.publish(&key, b"other object", None).unwrap_err();
        assert!(error.contains("nondeterministic codegen or an incomplete cache identity"));
        assert!(
            matches!(cache.load(&key), CacheRead::Hit(value) if value.object == b"first object")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_identical_publishers_converge_on_one_valid_entry() {
        use std::sync::{Arc, Barrier};

        let (root, cache) = temp_cache("owner-cache-concurrent");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        let barrier = Arc::new(Barrier::new(8));
        let handles = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let key = key.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    cache.publish(
                        &key,
                        b"same object",
                        Some((AuditArtifactKind::Ptx, b"same ptx")),
                    )
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
        assert!(
            matches!(cache.load(&key), CacheRead::Hit(value) if value.object == b"same object" && value.audit.is_some())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_different_publishers_never_replace_the_winner() {
        use std::sync::{Arc, Barrier};

        let (root, cache) = temp_cache("owner-cache-concurrent-collision");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        let barrier = Arc::new(Barrier::new(2));
        let handles = [b"first".as_slice(), b"other".as_slice()]
            .into_iter()
            .map(|bytes| {
                let cache = cache.clone();
                let key = key.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    (bytes, cache.publish(&key, bytes, None))
                })
            })
            .collect::<Vec<_>>();
        let outcomes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes.iter().filter(|(_, result)| result.is_ok()).count(),
            1
        );
        let winner = outcomes
            .iter()
            .find_map(|(bytes, result)| result.is_ok().then_some(*bytes))
            .unwrap();
        assert!(matches!(cache.load(&key), CacheRead::Hit(value) if value.object == winner));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_identity_component_changes_the_key() {
        let base = [
            b"owner".as_slice(),
            b"graph".as_slice(),
            b"sm_86".as_slice(),
        ];
        let key = DeviceArtifactObjectCache::key(base);
        for changed in [
            [
                b"other".as_slice(),
                b"graph".as_slice(),
                b"sm_86".as_slice(),
            ],
            [
                b"owner".as_slice(),
                b"body2".as_slice(),
                b"sm_86".as_slice(),
            ],
            [
                b"owner".as_slice(),
                b"graph".as_slice(),
                b"sm_90".as_slice(),
            ],
        ] {
            assert_ne!(key, DeviceArtifactObjectCache::key(changed));
        }
    }

    #[test]
    fn corrupt_object_is_rejected_and_replaced_atomically() {
        let (root, cache) = temp_cache("owner-cache-corrupt");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        cache.publish(&key, b"first", None).unwrap();
        fs::write(cache.entry_path(&key).join(ARTIFACT_FILE), b"corrupt").unwrap();
        assert!(matches!(cache.load(&key), CacheRead::Corrupt(_)));
        cache.publish(&key, b"recovered", None).unwrap();
        assert!(matches!(cache.load(&key), CacheRead::Hit(value) if value.object == b"recovered"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn identity_invalid_object_is_rejected_then_replaced_by_a_valid_rebuild() {
        let (root, cache) = temp_cache("owner-cache-identity-recovery");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        let (wrong_object, _) = materialized_object("sm_90", b"wrong cubin");
        let (expected_object, expected) = materialized_object("sm_86", b"expected cubin");
        cache.publish(&key, &wrong_object, None).unwrap();

        let CacheRead::Hit(cached) = cache.load(&key) else {
            panic!("digest-valid test object did not load");
        };
        assert!(validate_artifact_object(&cached.object, &expected).is_err());

        cache
            .publish_validated_artifact(&key, &expected_object, None, &expected)
            .unwrap();
        assert!(
            matches!(cache.load(&key), CacheRead::Hit(value) if value.object == expected_object)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_identity_recovery_preserves_the_first_valid_rebuild() {
        use std::sync::{Arc, Barrier};

        let (root, cache) = temp_cache("owner-cache-concurrent-identity-recovery");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        let (wrong_object, _) = materialized_object("sm_90", b"wrong cubin");
        cache.publish(&key, &wrong_object, None).unwrap();

        let (first, expected) = materialized_object("sm_86", b"first valid cubin");
        let (second, second_expected) = materialized_object("sm_86", b"second valid cubin");
        assert_eq!(expected, second_expected);
        let barrier = Arc::new(Barrier::new(2));
        let handles = [first, second]
            .into_iter()
            .map(|object| {
                let cache = cache.clone();
                let key = key.clone();
                let expected = expected.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let result = cache.publish_validated_artifact(&key, &object, None, &expected);
                    (object, result)
                })
            })
            .collect::<Vec<_>>();
        let outcomes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes.iter().filter(|(_, result)| result.is_ok()).count(),
            1
        );
        let winner = outcomes
            .iter()
            .find_map(|(object, result)| result.is_ok().then_some(object))
            .unwrap();
        assert!(matches!(cache.load(&key), CacheRead::Hit(value) if value.object == *winner));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_staging_directory_is_not_observable() {
        let (root, cache) = temp_cache("owner-cache-interrupted");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        let family = root.join(CACHE_PROTOCOL);
        fs::create_dir_all(&family).unwrap();
        fs::create_dir(family.join(format!(".{key}.staging"))).unwrap();
        assert!(matches!(cache.load(&key), CacheRead::Miss));
        cache.publish(&key, b"complete", None).unwrap();
        assert!(matches!(cache.load(&key), CacheRead::Hit(value) if value.object == b"complete"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn audit_sidecar_round_trips_with_the_materialized_object() {
        let (root, cache) = temp_cache("owner-cache-audit");
        let key = DeviceArtifactObjectCache::key([b"owner".as_slice()]);
        cache
            .publish(
                &key,
                b"host object",
                Some((AuditArtifactKind::PtxBundle, b"ptx bundle")),
            )
            .unwrap();
        let CacheRead::Hit(value) = cache.load(&key) else {
            panic!("published cache entry did not load");
        };
        assert_eq!(value.object, b"host object");
        let audit = value.audit.expect("cached audit sidecar");
        assert_eq!(audit.kind, AuditArtifactKind::PtxBundle);
        assert_eq!(audit.bytes, b"ptx bundle");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cached_object_must_match_the_exact_materialized_bundle_contract() {
        let (object, mut expected) = materialized_object("sm_86", b"cubin");
        validate_artifact_object(&object, &expected).unwrap();

        expected.target = "sm_90".to_string();
        assert!(validate_artifact_object(&object, &expected).is_err());
    }

    #[test]
    fn reclaim_preserves_the_just_published_entry() {
        let (root, cache) = temp_cache("owner-cache-reclaim");
        let keys = ["first", "second", "current"]
            .map(|part| DeviceArtifactObjectCache::key([part.as_bytes()]));
        for (index, key) in keys.iter().enumerate() {
            cache
                .publish(key, format!("artifact-{index}").as_bytes(), None)
                .unwrap();
        }
        cache.reclaim_to_budget(&keys[2], 1).unwrap();
        assert!(matches!(cache.load(&keys[0]), CacheRead::Miss));
        assert!(matches!(cache.load(&keys[1]), CacheRead::Miss));
        assert!(matches!(cache.load(&keys[2]), CacheRead::Hit(_)));
        fs::remove_dir_all(root).unwrap();
    }
}
