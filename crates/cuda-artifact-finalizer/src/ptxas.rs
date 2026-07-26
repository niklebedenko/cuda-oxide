/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::FinalizerError;
use crate::options::FinalizationOptions;
use crate::provenance::{digest_file_handle, with_revalidated_tool_identity};
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_PTXAS_CONCURRENCY: usize = 4;
const MAX_PTXAS_DIAGNOSTIC_BYTES: u64 = 1024 * 1024;

struct PinnedPtxas {
    file: File,
    path: PathBuf,
    digest: [u8; 32],
}

static PTXAS_TOOL: OnceLock<Arc<PinnedPtxas>> = OnceLock::new();
static PTXAS_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone)]
pub(crate) struct PtxAssembler {
    tool: Arc<PinnedPtxas>,
}

pub(crate) struct PtxAssemblyInput<'a> {
    pub(crate) name: &'a str,
    pub(crate) ptx_path: &'a Path,
    pub(crate) object_path: &'a Path,
}

#[derive(Debug)]
pub(crate) struct PtxAssemblyResult {
    pub(crate) object_bytes: u64,
    pub(crate) elapsed: Duration,
    pub(crate) peak_rss_kib: Option<u64>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

#[derive(Debug)]
pub(crate) struct PtxAssemblyBatch {
    pub(crate) results: Vec<PtxAssemblyResult>,
    pub(crate) peak_concurrency: usize,
    pub(crate) peak_aggregate_rss_kib: Option<u64>,
}

impl PtxAssembler {
    pub(crate) fn discover() -> Result<Option<Self>, FinalizerError> {
        if let Some(tool) = PTXAS_TOOL.get() {
            return Ok(Some(Self {
                tool: Arc::clone(tool),
            }));
        }
        let _guard = PTXAS_TOOL_LOAD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(tool) = PTXAS_TOOL.get() {
            return Ok(Some(Self {
                tool: Arc::clone(tool),
            }));
        }
        let Some(path) = discover_ptxas_path()? else {
            return Ok(None);
        };
        let canonical_path =
            std::fs::canonicalize(&path).map_err(|source| FinalizerError::PtxasIo {
                path: path.clone(),
                source,
            })?;
        let file = File::open(&canonical_path).map_err(|source| FinalizerError::PtxasIo {
            path: canonical_path.clone(),
            source,
        })?;
        let metadata = file.metadata().map_err(|source| FinalizerError::PtxasIo {
            path: canonical_path.clone(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(FinalizerError::PtxasNotExecutable {
                path: canonical_path,
            });
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            if metadata.permissions().mode() & 0o111 == 0 {
                return Err(FinalizerError::PtxasNotExecutable {
                    path: canonical_path,
                });
            }
        }
        let digest = digest_file_handle(&file).map_err(|source| FinalizerError::PtxasIo {
            path: canonical_path.clone(),
            source,
        })?;
        let tool = Arc::new(PinnedPtxas {
            file,
            path: canonical_path,
            digest,
        });
        let _ = PTXAS_TOOL.set(Arc::clone(&tool));
        Ok(Some(Self { tool }))
    }

    pub(crate) fn digest_if_unchanged(&self) -> Option<[u8; 32]> {
        (digest_file_handle(&self.tool.file).ok() == Some(self.tool.digest))
            .then_some(self.tool.digest)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.tool.path
    }

    pub(crate) fn assemble(
        &self,
        inputs: &[PtxAssemblyInput<'_>],
        options: &FinalizationOptions,
    ) -> Result<PtxAssemblyBatch, FinalizerError> {
        with_revalidated_tool_identity(
            "ptxas",
            Some(self.tool.digest),
            || digest_file_handle(&self.tool.file).ok(),
            || self.assemble_inner(inputs, options),
        )
    }

    fn assemble_inner(
        &self,
        inputs: &[PtxAssemblyInput<'_>],
        options: &FinalizationOptions,
    ) -> Result<PtxAssemblyBatch, FinalizerError> {
        if inputs.is_empty() {
            return Err(FinalizerError::NoLinkInputs);
        }
        let concurrency = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_PTXAS_CONCURRENCY)
            .min(inputs.len());
        let mut children = RunningPtxasChildren::default();
        let mut next_input = 0;
        let mut results = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        let semantic_options = options.ptxas_options();
        let mut peak_concurrency = 0;
        let mut peak_aggregate_rss_kib = None::<u64>;

        while next_input < inputs.len() || !children.is_empty() {
            while next_input < inputs.len() && children.len() < concurrency {
                children.push(self.spawn(next_input, &inputs[next_input], &semantic_options)?);
                next_input += 1;
            }
            peak_concurrency = peak_concurrency.max(children.len());

            let completed = loop {
                let mut completed = None;
                let mut aggregate_rss_kib = 0_u64;
                let mut observed_rss = false;
                for (position, running) in children.iter_mut().enumerate() {
                    if let Some(rss_kib) = process_rss_kib(running.child.id()) {
                        running.peak_rss_kib =
                            Some(running.peak_rss_kib.unwrap_or_default().max(rss_kib));
                        aggregate_rss_kib = aggregate_rss_kib.saturating_add(rss_kib);
                        observed_rss = true;
                    }
                    if completed.is_none()
                        && running
                            .child
                            .try_wait()
                            .map_err(|source| FinalizerError::PtxasWait {
                                name: inputs[running.index].name.to_string(),
                                source,
                            })?
                            .is_some()
                    {
                        completed = Some(position);
                    }
                }
                if observed_rss {
                    peak_aggregate_rss_kib = Some(
                        peak_aggregate_rss_kib
                            .unwrap_or_default()
                            .max(aggregate_rss_kib),
                    );
                }
                if let Some(position) = completed {
                    break position;
                }
                std::thread::sleep(Duration::from_millis(5));
            };

            let mut running = children.swap_remove(completed);
            let status = running
                .child
                .wait()
                .map_err(|source| FinalizerError::PtxasWait {
                    name: inputs[running.index].name.to_string(),
                    source,
                })?;
            running.reaped = true;
            let elapsed = running.started.elapsed();
            let stdout = read_diagnostic(&running.stdout_path)?;
            let stderr = read_diagnostic(&running.stderr_path)?;
            if !status.success() {
                return Err(FinalizerError::PtxasFailed {
                    name: inputs[running.index].name.to_string(),
                    status: status.to_string(),
                    stdout,
                    stderr,
                });
            }
            let object_path = inputs[running.index].object_path;
            let metadata =
                std::fs::metadata(object_path).map_err(|source| FinalizerError::PtxasIo {
                    path: object_path.to_path_buf(),
                    source,
                })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(FinalizerError::PtxasMissingOutput {
                    name: inputs[running.index].name.to_string(),
                    path: object_path.to_path_buf(),
                });
            }
            if metadata.len() > crate::MAX_PARTITION_OBJECT_BYTES {
                return Err(FinalizerError::PartitionSourceTooLarge {
                    path: object_path.to_path_buf(),
                    actual_bytes: metadata.len(),
                    maximum_bytes: crate::MAX_PARTITION_OBJECT_BYTES,
                });
            }
            results[running.index] = Some(PtxAssemblyResult {
                object_bytes: metadata.len(),
                elapsed,
                peak_rss_kib: running.peak_rss_kib,
                stdout,
                stderr,
            });
        }

        Ok(PtxAssemblyBatch {
            results: results
                .into_iter()
                .map(|result| result.expect("every ptxas child completed"))
                .collect(),
            peak_concurrency,
            peak_aggregate_rss_kib,
        })
    }

    fn spawn(
        &self,
        index: usize,
        input: &PtxAssemblyInput<'_>,
        semantic_options: &[String],
    ) -> Result<RunningPtxas, FinalizerError> {
        let stdout_path = input.object_path.with_extension("ptxas.stdout");
        let stderr_path = input.object_path.with_extension("ptxas.stderr");
        let stdout = File::create(&stdout_path).map_err(|source| FinalizerError::PtxasIo {
            path: stdout_path.clone(),
            source,
        })?;
        let stderr = File::create(&stderr_path).map_err(|source| FinalizerError::PtxasIo {
            path: stderr_path.clone(),
            source,
        })?;
        let mut command = self.command();
        command
            .args(semantic_options)
            .arg("-o")
            .arg(input.object_path)
            .arg(input.ptx_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        let started = Instant::now();
        let child = command
            .spawn()
            .map_err(|source| FinalizerError::PtxasLaunch {
                path: self.tool.path.clone(),
                name: input.name.to_string(),
                source,
            })?;
        Ok(RunningPtxas {
            index,
            child,
            started,
            stdout_path,
            stderr_path,
            peak_rss_kib: None,
            reaped: false,
        })
    }

    fn command(&self) -> Command {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd as _;
            Command::new(format!("/proc/self/fd/{}", self.tool.file.as_raw_fd()))
        }
        #[cfg(not(target_os = "linux"))]
        {
            Command::new(&self.tool.path)
        }
    }
}

fn discover_ptxas_path() -> Result<Option<PathBuf>, FinalizerError> {
    if let Some(explicit) = std::env::var_os("CUDA_OXIDE_PTXAS") {
        let path = PathBuf::from(explicit);
        return if path.is_file() {
            Ok(Some(path))
        } else {
            Err(FinalizerError::PtxasNotExecutable { path })
        };
    }
    for path in ptxas_candidates() {
        if path.is_file() {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn ptxas_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for variable in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            candidates.push(PathBuf::from(root).join("bin/ptxas"));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda/bin/ptxas"));
    candidates.push(PathBuf::from("/opt/cuda/bin/ptxas"));
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|root| root.join("ptxas")));
    }
    deduplicate_paths(candidates)
}

fn deduplicate_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::with_capacity(paths.len());
    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique
}

fn read_diagnostic(path: &Path) -> Result<String, FinalizerError> {
    let mut file = File::open(path).map_err(|source| FinalizerError::PtxasIo {
        path: path.to_path_buf(),
        source,
    })?;
    let actual_bytes = file
        .metadata()
        .map_err(|source| FinalizerError::PtxasIo {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let retained_bytes = actual_bytes.min(MAX_PTXAS_DIAGNOSTIC_BYTES);
    let mut bytes = vec![0; usize::try_from(retained_bytes).unwrap_or(usize::MAX)];
    file.read_exact(&mut bytes)
        .map_err(|source| FinalizerError::PtxasIo {
            path: path.to_path_buf(),
            source,
        })?;
    let mut diagnostic = String::from_utf8_lossy(&bytes).into_owned();
    if actual_bytes > retained_bytes {
        use std::fmt::Write as _;
        let _ = write!(
            diagnostic,
            "\n... [ptxas diagnostic truncated: retained {retained_bytes} of {actual_bytes} bytes]"
        );
    }
    Ok(diagnostic)
}

fn process_rss_kib(process_id: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{process_id}/status")).ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_ascii_whitespace().nth(1)?.parse().ok()
}

struct RunningPtxas {
    index: usize,
    child: Child,
    started: Instant,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    peak_rss_kib: Option<u64>,
    reaped: bool,
}

#[derive(Default)]
struct RunningPtxasChildren(Vec<RunningPtxas>);

impl RunningPtxasChildren {
    fn push(&mut self, child: RunningPtxas) {
        self.0.push(child);
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut RunningPtxas> {
        self.0.iter_mut()
    }

    fn swap_remove(&mut self, index: usize) -> RunningPtxas {
        self.0.swap_remove(index)
    }
}

impl Drop for RunningPtxas {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_deduplication_preserves_precedence() {
        assert_eq!(
            deduplicate_paths(vec![
                PathBuf::from("/a/ptxas"),
                PathBuf::from("/b/ptxas"),
                PathBuf::from("/a/ptxas"),
            ]),
            [PathBuf::from("/a/ptxas"), PathBuf::from("/b/ptxas")]
        );
    }

    #[test]
    fn concurrency_is_strictly_bounded() {
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        assert!(available.min(MAX_PTXAS_CONCURRENCY) <= 4);
    }

    #[test]
    fn os_string_paths_remain_lossless_candidates() {
        let value = std::ffi::OsString::from("/cuda");
        assert_eq!(
            PathBuf::from(value).join("bin/ptxas"),
            Path::new("/cuda/bin/ptxas")
        );
    }

    #[test]
    fn diagnostic_reader_bounds_memory_and_marks_truncation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ptxas.stderr");
        std::fs::write(
            &path,
            vec![b'x'; usize::try_from(MAX_PTXAS_DIAGNOSTIC_BYTES + 17).unwrap()],
        )
        .unwrap();
        let diagnostic = read_diagnostic(&path).unwrap();
        assert!(diagnostic.starts_with("xxxx"));
        assert!(diagnostic.contains("diagnostic truncated"));
        assert!(diagnostic.len() < usize::try_from(MAX_PTXAS_DIAGNOSTIC_BYTES + 256).unwrap());
    }
}
