/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::{Barrier, Mutex};
use std::thread;

#[test]
fn worker_limit_parser_accepts_only_one_through_four() {
    assert_eq!(parse_nvvm_worker_limit(None).unwrap(), None);
    for workers in 1..=4 {
        assert_eq!(
            parse_nvvm_worker_limit(Some(OsStr::new(&workers.to_string()))).unwrap(),
            Some(workers),
        );
    }
    for invalid in ["", "0", "5", "-1", "many"] {
        assert!(matches!(
            parse_nvvm_worker_limit(Some(OsStr::new(invalid))).unwrap_err(),
            FinalizerError::InvalidNvvmWorkerCount { value } if value == invalid
        ));
    }
}

#[test]
fn direct_ptx_jobs_stage_in_index_order_without_nvvm_activity() {
    let directory = tempfile::tempdir().unwrap();
    let stored = (0..6)
        .map(|index| {
            let name = format!("direct-{index}.ptx");
            let path = directory.path().join(&name);
            let ptx = format!(".visible .entry direct_{index}() {{ ret; }}\n");
            std::fs::write(&path, ptx.as_bytes()).unwrap();
            (name, path, ptx.into_bytes())
        })
        .collect::<Vec<_>>();
    let inputs = stored
        .iter()
        .map(|(name, path, _)| PartitionFileInput::new(name, path, PartitionFileInputKind::Ptx))
        .collect::<Vec<_>>();
    let activity = NvvmActivity::default();

    let prepared =
        run_partition_workers(None, &inputs, directory.path(), 4, &activity, false).unwrap();

    assert_eq!(prepared.len(), inputs.len());
    assert_eq!(activity.peak.load(Ordering::Relaxed), 0);
    assert_eq!(activity.wall_elapsed(), Duration::ZERO);
    for (index, ((_, _, expected), metadata)) in stored.iter().zip(prepared).enumerate() {
        assert_eq!(
            std::fs::read(directory.path().join(format!("{index:04}.ptx"))).unwrap(),
            *expected,
        );
        assert_eq!(metadata.source_bytes, expected.len());
        assert_eq!(metadata.ptx_bytes, expected.len());
        assert_eq!(metadata.nvvm_compile_elapsed, Duration::ZERO);
    }
}

#[test]
fn bounded_scheduler_preserves_order_after_reverse_completion() {
    let batch = run_bounded_indexed_jobs(12, 4, |index| {
        thread::sleep(Duration::from_millis((12 - index) as u64));
        Ok::<_, ()>(index * 2)
    })
    .unwrap();

    assert_eq!(
        batch.results,
        (0..12).map(|index| index * 2).collect::<Vec<_>>()
    );
    assert!((2..=4).contains(&batch.peak_concurrency));
}

#[test]
fn bounded_scheduler_enforces_one_and_four_worker_paths() {
    let serial = run_bounded_indexed_jobs(8, 1, Ok::<_, ()>).unwrap();
    assert_eq!(serial.peak_concurrency, 1);

    let barrier = Arc::new(Barrier::new(4));
    let parallel = run_bounded_indexed_jobs(8, usize::MAX, {
        let barrier = Arc::clone(&barrier);
        move |index| {
            if index < 4 {
                barrier.wait();
            }
            Ok::<_, ()>(index)
        }
    })
    .unwrap();
    assert_eq!(parallel.peak_concurrency, 4);
    assert_eq!(parallel.results, (0..8).collect::<Vec<_>>());

    let empty = run_bounded_indexed_jobs(0, 4, |_| Ok::<_, ()>(())).unwrap();
    assert_eq!(empty.peak_concurrency, 0);
    assert!(empty.results.is_empty());
}

#[test]
fn lowest_index_failure_wins_independent_of_completion_order() {
    let barrier = Arc::new(Barrier::new(4));
    let error = run_bounded_indexed_jobs(8, 4, {
        let barrier = Arc::clone(&barrier);
        move |index| {
            if index < 4 {
                barrier.wait();
            }
            match index {
                1 => {
                    thread::sleep(Duration::from_millis(20));
                    Err("one")
                }
                3 => Err("three"),
                _ => {
                    thread::sleep(Duration::from_millis(40));
                    Ok(index)
                }
            }
        }
    })
    .unwrap_err();

    assert!(matches!(
        error,
        IndexedJobFailure::Job {
            index: 1,
            error: "one"
        }
    ));
}

#[test]
fn failure_cancels_new_claims_and_joins_running_workers() {
    let barrier = Arc::new(Barrier::new(4));
    let started = Arc::new(Mutex::new(BTreeSet::new()));
    let completed = Arc::new(AtomicUsize::new(0));
    let result = run_bounded_indexed_jobs(100, 4, {
        let barrier = Arc::clone(&barrier);
        let started = Arc::clone(&started);
        let completed = Arc::clone(&completed);
        move |index| {
            started
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(index);
            if index < 4 {
                barrier.wait();
            }
            if index == 0 {
                return Err(index);
            }
            thread::sleep(Duration::from_millis(20));
            completed.fetch_add(1, Ordering::Relaxed);
            Ok(index)
        }
    });

    assert!(matches!(
        result,
        Err(IndexedJobFailure::Job { index: 0, error: 0 })
    ));
    assert_eq!(
        *started
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        BTreeSet::from([0, 1, 2, 3]),
    );
    assert_eq!(completed.load(Ordering::Relaxed), 3);
}

#[test]
fn worker_panic_is_converted_after_siblings_join() {
    let barrier = Arc::new(Barrier::new(4));
    let completed = Arc::new(AtomicUsize::new(0));
    let result = run_bounded_indexed_jobs(8, 4, {
        let barrier = Arc::clone(&barrier);
        let completed = Arc::clone(&completed);
        move |index| {
            if index < 4 {
                barrier.wait();
            }
            if index == 2 {
                panic!("injected indexed worker panic");
            }
            thread::sleep(Duration::from_millis(10));
            completed.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(index)
        }
    });

    assert!(matches!(
        result,
        Err(IndexedJobFailure::Panicked { index: Some(2) })
    ));
    assert_eq!(completed.load(Ordering::Relaxed), 3);
}
