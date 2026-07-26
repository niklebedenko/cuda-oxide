/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod diagnostics;

use crate::FinalizerError;
use crate::options::FinalizationOptions;
use crate::provenance::{digest_file_handle, with_revalidated_tool_identity};
use diagnostics::{DiagnosticBudget, DiagnosticReaders};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_PTXAS_CONCURRENCY: usize = 4;

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
        let concurrency = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_PTXAS_CONCURRENCY)
            .min(inputs.len());
        self.assemble_inner_with_concurrency(inputs, options, concurrency)
    }

    fn assemble_inner_with_concurrency(
        &self,
        inputs: &[PtxAssemblyInput<'_>],
        options: &FinalizationOptions,
        concurrency: usize,
    ) -> Result<PtxAssemblyBatch, FinalizerError> {
        if inputs.is_empty() {
            return Err(FinalizerError::NoLinkInputs);
        }
        let concurrency = concurrency
            .clamp(1, MAX_PTXAS_CONCURRENCY)
            .min(inputs.len());
        let diagnostic_budget = DiagnosticBudget::for_batch(inputs.len());
        let mut children = RunningPtxasChildren::default();
        let mut next_input = 0;
        let mut results = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        let semantic_options = options.ptxas_options();
        let mut peak_concurrency = 0;
        let mut peak_aggregate_rss_kib = None::<u64>;

        while next_input < inputs.len() || !children.is_empty() {
            while next_input < inputs.len() && children.len() < concurrency {
                children.push(self.spawn(
                    next_input,
                    &inputs[next_input],
                    &semantic_options,
                    diagnostic_budget.bytes_per_stream(),
                )?);
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
            let (stdout, stderr) = running.finish_diagnostics(inputs[running.index].name)?;
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
        diagnostic_bytes_per_stream: usize,
    ) -> Result<RunningPtxas, FinalizerError> {
        let mut command = self.command();
        command
            .args(semantic_options)
            .arg("-o")
            .arg(input.object_path)
            .arg(input.ptx_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started = Instant::now();
        let child = command
            .spawn()
            .map_err(|source| FinalizerError::PtxasLaunch {
                path: self.tool.path.clone(),
                name: input.name.to_string(),
                source,
            })?;
        let mut running = RunningPtxas {
            index,
            child,
            started,
            diagnostics: DiagnosticReaders::default(),
            peak_rss_kib: None,
            reaped: false,
        };
        running
            .diagnostics
            .attach(&mut running.child, input.name, diagnostic_bytes_per_stream)?;
        Ok(running)
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

fn process_rss_kib(process_id: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{process_id}/status")).ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_ascii_whitespace().nth(1)?.parse().ok()
}

struct RunningPtxas {
    index: usize,
    child: Child,
    started: Instant,
    diagnostics: DiagnosticReaders,
    peak_rss_kib: Option<u64>,
    reaped: bool,
}

impl RunningPtxas {
    fn finish_diagnostics(&mut self, name: &str) -> Result<(String, String), FinalizerError> {
        self.diagnostics.finish(name)
    }
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
        self.diagnostics.discard();
    }
}

#[cfg(test)]
mod tests;
