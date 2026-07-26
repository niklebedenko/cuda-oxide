/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::diagnostics::{MAX_PTXAS_BATCH_DIAGNOSTIC_BYTES, MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES};
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

#[cfg(target_os = "linux")]
#[test]
fn multi_input_diagnostics_are_bounded_and_schedule_independent() {
    let directory = tempfile::tempdir().unwrap();
    let tool_path = compile_fake_ptxas(directory.path());
    let assembler = test_assembler(&tool_path);
    let stored_inputs = create_inputs(directory.path(), &["success-a", "success-b", "success-c"]);
    let inputs = borrow_inputs(&stored_inputs);
    let options = FinalizationOptions::new("sm_86".parse().unwrap());

    let parallel = assembler
        .assemble_inner_with_concurrency(&inputs, &options, 3)
        .unwrap();
    let serial = assembler
        .assemble_inner_with_concurrency(&inputs, &options, 1)
        .unwrap();

    assert_eq!(parallel.peak_concurrency, 3);
    assert_eq!(serial.peak_concurrency, 1);
    let total_diagnostic_bytes = parallel
        .results
        .iter()
        .map(|result| result.stdout.len().saturating_add(result.stderr.len()))
        .sum::<usize>();
    assert!(total_diagnostic_bytes <= MAX_PTXAS_BATCH_DIAGNOSTIC_BYTES);
    for ((stored, parallel), serial) in stored_inputs
        .iter()
        .zip(&parallel.results)
        .zip(&serial.results)
    {
        let tag = stored.object_path.file_name().unwrap().to_str().unwrap();
        assert!(parallel.stdout.starts_with(tag));
        assert!(parallel.stderr.starts_with(tag));
        assert!(parallel.stdout.contains("diagnostic truncated"));
        assert!(parallel.stderr.contains("diagnostic truncated"));
        assert!(parallel.stdout.len() <= MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES);
        assert!(parallel.stderr.len() <= MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES);
        assert_eq!(parallel.stdout, serial.stdout);
        assert_eq!(parallel.stderr, serial.stderr);
        assert!(!stored.object_path.with_extension("ptxas.stdout").exists());
        assert!(!stored.object_path.with_extension("ptxas.stderr").exists());
    }
}

#[cfg(target_os = "linux")]
#[test]
fn failed_batch_reaps_sibling_ptxas_processes() {
    let directory = tempfile::tempdir().unwrap();
    let tool_path = compile_fake_ptxas(directory.path());
    let assembler = test_assembler(&tool_path);
    let stored_inputs = create_inputs(directory.path(), &["slow-a", "fail", "slow-b"]);
    let inputs = borrow_inputs(&stored_inputs);
    let options = FinalizationOptions::new("sm_86".parse().unwrap());

    let error = assembler
        .assemble_inner_with_concurrency(&inputs, &options, 3)
        .unwrap_err();
    match error {
        FinalizerError::PtxasFailed {
            name,
            stdout,
            stderr,
            ..
        } => {
            assert_eq!(name, "fail");
            assert!(stdout.len() <= MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES);
            assert!(stderr.len() <= MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES);
            assert!(stdout.contains("diagnostic truncated"));
            assert!(stderr.contains("diagnostic truncated"));
        }
        other => panic!("unexpected ptxas result: {other}"),
    }

    for stored in [&stored_inputs[0], &stored_inputs[2]] {
        let pid_path = stored.object_path.with_extension("o.pid");
        let pid = std::fs::read_to_string(&pid_path)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        assert!(
            !Path::new(&format!("/proc/{pid}")).exists(),
            "ptxas sibling {pid} was not reaped"
        );
    }
}

#[cfg(target_os = "linux")]
struct StoredInput {
    name: String,
    ptx_path: PathBuf,
    object_path: PathBuf,
}

#[cfg(target_os = "linux")]
fn create_inputs(directory: &Path, names: &[&str]) -> Vec<StoredInput> {
    names
        .iter()
        .map(|name| {
            let ptx_path = directory.join(format!("{name}.ptx"));
            let object_path = directory.join(format!("{name}.o"));
            std::fs::write(&ptx_path, b".version 8.0\n").unwrap();
            StoredInput {
                name: (*name).to_string(),
                ptx_path,
                object_path,
            }
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn borrow_inputs(stored: &[StoredInput]) -> Vec<PtxAssemblyInput<'_>> {
    stored
        .iter()
        .map(|input| PtxAssemblyInput {
            name: &input.name,
            ptx_path: &input.ptx_path,
            object_path: &input.object_path,
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn test_assembler(path: &Path) -> PtxAssembler {
    let path = std::fs::canonicalize(path).unwrap();
    let file = File::open(&path).unwrap();
    let digest = digest_file_handle(&file).unwrap();
    PtxAssembler {
        tool: Arc::new(PinnedPtxas { file, path, digest }),
    }
}

#[cfg(target_os = "linux")]
fn compile_fake_ptxas(directory: &Path) -> PathBuf {
    let source_path = directory.join("fake-ptxas.c");
    let tool_path = directory.join("fake-ptxas");
    std::fs::write(
        &source_path,
        r#"
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void emit_diagnostic(FILE *stream, const char *tag, char fill) {
    fputs(tag, stream);
    fputc('\n', stream);
    char buffer[8192];
    memset(buffer, fill, sizeof(buffer));
    size_t remaining = 1200000;
    while (remaining > 0) {
        size_t chunk = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        fwrite(buffer, 1, chunk, stream);
        remaining -= chunk;
    }
    fflush(stream);
}

int main(int argc, char **argv) {
    const char *output = NULL;
    for (int index = 1; index + 1 < argc; ++index) {
        if (strcmp(argv[index], "-o") == 0) {
            output = argv[index + 1];
            break;
        }
    }
    if (output == NULL) {
        return 90;
    }

    const char *tag = strrchr(output, '/');
    tag = tag == NULL ? output : tag + 1;
    if (strstr(tag, "slow") != NULL) {
        char pid_path[4096];
        snprintf(pid_path, sizeof(pid_path), "%s.pid", output);
        FILE *pid_file = fopen(pid_path, "w");
        if (pid_file == NULL) {
            return 91;
        }
        fprintf(pid_file, "%ld\n", (long)getpid());
        fclose(pid_file);
        for (;;) {
            pause();
        }
    }

    if (strstr(tag, "fail") != NULL) {
        usleep(500000);
        emit_diagnostic(stdout, tag, 'F');
        emit_diagnostic(stderr, tag, 'E');
        return 7;
    }

    emit_diagnostic(stdout, tag, 'O');
    emit_diagnostic(stderr, tag, 'E');
    FILE *object = fopen(output, "wb");
    if (object == NULL) {
        return 92;
    }
    fputc('x', object);
    fclose(object);
    return 0;
}
"#,
    )
    .unwrap();
    let status = Command::new("cc")
        .args(["-O2", "-std=c11"])
        .arg(&source_path)
        .arg("-o")
        .arg(&tool_path)
        .status()
        .expect("run C compiler for fake ptxas");
    assert!(status.success(), "C compiler failed with {status}");
    tool_path
}
