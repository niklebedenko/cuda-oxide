/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::nvvm::{NvvmCompileSession, NvvmCompiler};
use crate::options::FinalizationOptions;
use crate::{
    FinalizerError, MAX_PARTITION_NVVM_IR_BYTES, MAX_PARTITION_PTX_BYTES, PartitionFileInput,
    PartitionFileInputKind, read_partition_source_capped, validate_name,
};
use std::ffi::OsStr;
use std::io::Write as _;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const MAX_NVVM_CONCURRENCY: usize = 4;
const NVVM_WORKERS_ENVIRONMENT: &str = "CUDA_OXIDE_NVVM_WORKERS";

pub(crate) struct PreparedPartitionPtx {
    pub(crate) source_bytes: usize,
    pub(crate) ptx_bytes: usize,
    pub(crate) nvvm_compile_elapsed: Duration,
}

pub(crate) struct PreparedPartitionBatch {
    pub(crate) partitions: Vec<PreparedPartitionPtx>,
    pub(crate) nvvm_wall_elapsed: Duration,
    pub(crate) nvvm_peak_concurrency: usize,
}

pub(crate) fn configured_nvvm_worker_limit() -> Result<usize, FinalizerError> {
    parse_nvvm_worker_limit(std::env::var_os(NVVM_WORKERS_ENVIRONMENT).as_deref())
        .map(|configured| configured.unwrap_or(MAX_NVVM_CONCURRENCY))
}

fn parse_nvvm_worker_limit(value: Option<&OsStr>) -> Result<Option<usize>, FinalizerError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let displayed = value.to_string_lossy().into_owned();
    let parsed = displayed
        .parse::<usize>()
        .ok()
        .filter(|workers| (1..=MAX_NVVM_CONCURRENCY).contains(workers))
        .ok_or_else(|| FinalizerError::InvalidNvvmWorkerCount {
            value: displayed.clone(),
        })?;
    Ok(Some(parsed))
}

pub(crate) fn prepare_partition_ptx_files(
    compiler: &NvvmCompiler,
    inputs: &[PartitionFileInput<'_>],
    temporary_directory: &Path,
    options: &FinalizationOptions,
    worker_limit: usize,
    report_partition_stages: bool,
) -> Result<PreparedPartitionBatch, FinalizerError> {
    for input in inputs {
        validate_name(input.name)?;
    }
    let worker_count = effective_nvvm_worker_count(inputs.len(), worker_limit)?;
    let activity = NvvmActivity::default();
    let has_nvvm_input = inputs
        .iter()
        .any(|input| input.kind == PartitionFileInputKind::NvvmIr);
    let partitions = if has_nvvm_input {
        compiler.with_revalidated_session(options, |session| {
            run_partition_workers(
                Some(session),
                inputs,
                temporary_directory,
                worker_count,
                &activity,
                report_partition_stages,
            )
        })?
    } else {
        run_partition_workers(
            None,
            inputs,
            temporary_directory,
            worker_count,
            &activity,
            report_partition_stages,
        )?
    };

    Ok(PreparedPartitionBatch {
        partitions,
        nvvm_wall_elapsed: activity.wall_elapsed(),
        nvvm_peak_concurrency: activity.peak.load(Ordering::Relaxed),
    })
}

fn run_partition_workers(
    session: Option<NvvmCompileSession<'_>>,
    inputs: &[PartitionFileInput<'_>],
    temporary_directory: &Path,
    worker_count: usize,
    activity: &NvvmActivity,
    report_partition_stages: bool,
) -> Result<Vec<PreparedPartitionPtx>, FinalizerError> {
    let result = run_bounded_indexed_jobs(inputs.len(), worker_count, |index| {
        prepare_one_partition(
            session,
            index,
            &inputs[index],
            temporary_directory,
            activity,
            report_partition_stages,
        )
    });
    match result {
        Ok(batch) => {
            debug_assert!(
                activity.peak.load(Ordering::Relaxed) <= batch.peak_concurrency,
                "active libNVVM programs cannot exceed active partition workers",
            );
            Ok(batch.results)
        }
        Err(IndexedJobFailure::Job { index, error }) => {
            if report_partition_stages {
                eprintln!(
                    "[cuda_artifact_finalizer] partition preparation: \
                     lowest_failed_index={index}"
                );
            }
            Err(error)
        }
        Err(IndexedJobFailure::Panicked { index }) => {
            let name = index
                .and_then(|index| inputs.get(index))
                .map_or("<scheduler>", |input| input.name)
                .to_string();
            Err(FinalizerError::NvvmWorkerPanicked { index, name })
        }
        Err(IndexedJobFailure::Spawn { source }) => Err(FinalizerError::NvvmWorkerSpawn { source }),
    }
}

fn effective_nvvm_worker_count(
    input_count: usize,
    worker_limit: usize,
) -> Result<usize, FinalizerError> {
    if !(1..=MAX_NVVM_CONCURRENCY).contains(&worker_limit) {
        return Err(FinalizerError::InvalidNvvmWorkerCount {
            value: worker_limit.to_string(),
        });
    }
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    Ok(available
        .min(MAX_NVVM_CONCURRENCY)
        .min(worker_limit)
        .min(input_count))
}

fn prepare_one_partition(
    session: Option<NvvmCompileSession<'_>>,
    index: usize,
    input: &PartitionFileInput<'_>,
    temporary_directory: &Path,
    activity: &NvvmActivity,
    report_partition_stages: bool,
) -> Result<PreparedPartitionPtx, FinalizerError> {
    if report_partition_stages {
        eprintln!(
            "[cuda_artifact_finalizer] partition stage: index={index} name={} stage=read begin",
            input.name,
        );
    }
    let source_limit = match input.kind {
        PartitionFileInputKind::NvvmIr => MAX_PARTITION_NVVM_IR_BYTES,
        PartitionFileInputKind::Ptx => MAX_PARTITION_PTX_BYTES,
    };
    let source = read_partition_source_capped(input.path, source_limit)?;
    if source.is_empty() {
        return Err(FinalizerError::EmptyInput {
            name: input.name.to_string(),
        });
    }
    let source_bytes = source.len();
    if report_partition_stages {
        eprintln!(
            "[cuda_artifact_finalizer] partition stage: index={index} name={} \
             stage=compile begin source_kind={:?} source_bytes={source_bytes}",
            input.name, input.kind,
        );
    }

    let (ptx, nvvm_compile_elapsed) = match input.kind {
        PartitionFileInputKind::NvvmIr => {
            let compile_started = Instant::now();
            let _active = activity.enter();
            let ptx = session
                .expect("an NVVM input is always prepared inside a guarded compiler session")
                .compile_nvvm_ir_to_ptx_capped(input.name, &source, MAX_PARTITION_PTX_BYTES)?;
            (ptx, compile_started.elapsed())
        }
        PartitionFileInputKind::Ptx => (source, Duration::ZERO),
    };
    if report_partition_stages {
        eprintln!(
            "[cuda_artifact_finalizer] partition stage: index={index} name={} \
             stage=compile complete ptx_bytes={} elapsed={nvvm_compile_elapsed:?}",
            input.name,
            ptx.len(),
        );
    }

    let ptx_bytes = ptx.len();
    let ptx_path = temporary_directory.join(format!("{index:04}.ptx"));
    let mut ptx_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&ptx_path)
        .map_err(|source| FinalizerError::Io {
            path: ptx_path.clone(),
            source,
        })?;
    ptx_file
        .write_all(&ptx)
        .map_err(|source| FinalizerError::Io {
            path: ptx_path,
            source,
        })?;

    Ok(PreparedPartitionPtx {
        source_bytes,
        ptx_bytes,
        nvvm_compile_elapsed,
    })
}

#[derive(Default)]
struct NvvmActivity {
    active: AtomicUsize,
    peak: AtomicUsize,
    window: Mutex<NvvmActivityWindow>,
}

#[derive(Default)]
struct NvvmActivityWindow {
    first_started: Option<Instant>,
    last_finished: Option<Instant>,
}

impl NvvmActivity {
    fn enter(&self) -> NvvmActivityGuard<'_> {
        let now = Instant::now();
        let mut window = self
            .window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        window.first_started.get_or_insert(now);
        drop(window);

        let active = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(active, Ordering::Relaxed);
        NvvmActivityGuard { activity: self }
    }

    fn wall_elapsed(&self) -> Duration {
        let window = self
            .window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match (window.first_started, window.last_finished) {
            (Some(started), Some(finished)) => finished.duration_since(started),
            _ => Duration::ZERO,
        }
    }
}

struct NvvmActivityGuard<'a> {
    activity: &'a NvvmActivity,
}

impl Drop for NvvmActivityGuard<'_> {
    fn drop(&mut self) {
        self.activity.active.fetch_sub(1, Ordering::Relaxed);
        self.activity
            .window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last_finished = Some(Instant::now());
    }
}

#[derive(Debug)]
struct IndexedJobBatch<T> {
    results: Vec<T>,
    peak_concurrency: usize,
}

#[derive(Debug)]
enum IndexedJobFailure<E> {
    Job { index: usize, error: E },
    Panicked { index: Option<usize> },
    Spawn { source: std::io::Error },
}

enum IndexedJobOutcome<T, E> {
    Success(T),
    Failure(E),
    Panicked,
}

#[derive(Default)]
struct IndexedJobState {
    next_index: usize,
    cancelled: bool,
}

fn run_bounded_indexed_jobs<T, E>(
    job_count: usize,
    worker_limit: usize,
    job: impl Fn(usize) -> Result<T, E> + Sync,
) -> Result<IndexedJobBatch<T>, IndexedJobFailure<E>>
where
    T: Send,
    E: Send,
{
    if job_count == 0 {
        return Ok(IndexedJobBatch {
            results: Vec::new(),
            peak_concurrency: 0,
        });
    }
    let worker_count = worker_limit.clamp(1, MAX_NVVM_CONCURRENCY).min(job_count);
    let state = Mutex::new(IndexedJobState::default());
    let outcomes = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(job_count)
            .collect::<Vec<Option<IndexedJobOutcome<T, E>>>>(),
    );
    let unexpected_panics = Mutex::new(Vec::<Option<usize>>::new());
    let active = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);

    let spawn_error = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut spawn_error = None;
        for worker_index in 0..worker_count {
            let state = &state;
            let outcomes = &outcomes;
            let unexpected_panics = &unexpected_panics;
            let active = &active;
            let peak = &peak;
            let job = &job;
            let worker = std::thread::Builder::new()
                .name(format!("cuda-nvvm-{worker_index}"))
                .spawn_scoped(scope, move || {
                    let mut current_index = None;
                    let worker_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            loop {
                                let index = {
                                    let mut state = state
                                        .lock()
                                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                                    if state.cancelled || state.next_index == job_count {
                                        break;
                                    }
                                    let index = state.next_index;
                                    state.next_index += 1;
                                    index
                                };
                                current_index = Some(index);

                                let current_active = active.fetch_add(1, Ordering::Relaxed) + 1;
                                peak.fetch_max(current_active, Ordering::Relaxed);
                                let outcome = match std::panic::catch_unwind(
                                    std::panic::AssertUnwindSafe(|| job(index)),
                                ) {
                                    Ok(Ok(result)) => IndexedJobOutcome::Success(result),
                                    Ok(Err(error)) => IndexedJobOutcome::Failure(error),
                                    Err(_) => IndexedJobOutcome::Panicked,
                                };
                                active.fetch_sub(1, Ordering::Relaxed);

                                let failed = !matches!(outcome, IndexedJobOutcome::Success(_));
                                if failed {
                                    state
                                        .lock()
                                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                                        .cancelled = true;
                                }
                                outcomes
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner())[index] =
                                    Some(outcome);
                                current_index = None;
                                if failed {
                                    break;
                                }
                            }
                        }));
                    if worker_result.is_err() {
                        state
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .cancelled = true;
                        unexpected_panics
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(current_index);
                    }
                });
            match worker {
                Ok(handle) => handles.push(handle),
                Err(source) => {
                    state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .cancelled = true;
                    spawn_error = Some(source);
                    break;
                }
            }
        }
        for handle in handles {
            if handle.join().is_err() {
                unexpected_panics
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(None);
            }
        }
        spawn_error
    });

    if let Some(source) = spawn_error {
        return Err(IndexedJobFailure::Spawn { source });
    }

    let unexpected_panics = unexpected_panics
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let unexpected_panic_index = unexpected_panics.iter().flatten().copied().min();
    let has_unindexed_panic = unexpected_panics.iter().any(Option::is_none);
    let outcomes = outcomes
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut ordered = Vec::with_capacity(job_count);
    for (index, outcome) in outcomes.into_iter().enumerate() {
        if unexpected_panic_index.is_some_and(|panic_index| panic_index <= index) {
            return Err(IndexedJobFailure::Panicked {
                index: unexpected_panic_index,
            });
        }
        match outcome {
            Some(IndexedJobOutcome::Success(result)) => ordered.push(result),
            Some(IndexedJobOutcome::Failure(error)) => {
                return Err(IndexedJobFailure::Job { index, error });
            }
            Some(IndexedJobOutcome::Panicked) => {
                return Err(IndexedJobFailure::Panicked { index: Some(index) });
            }
            None => {
                return Err(IndexedJobFailure::Panicked {
                    index: unexpected_panic_index,
                });
            }
        }
    }
    if has_unindexed_panic {
        return Err(IndexedJobFailure::Panicked { index: None });
    }

    Ok(IndexedJobBatch {
        results: ordered,
        peak_concurrency: peak.load(Ordering::Relaxed),
    })
}

#[cfg(test)]
mod tests;
