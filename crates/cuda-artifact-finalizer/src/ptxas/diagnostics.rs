/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::FinalizerError;
use std::io::{self, Read};
use std::process::Child;
use std::thread::{self, JoinHandle};

pub(super) const MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES: usize = 1024 * 1024;
pub(super) const MAX_PTXAS_BATCH_DIAGNOSTIC_BYTES: usize = 4 * 1024 * 1024;

const STREAMS_PER_INPUT: usize = 2;

pub(super) struct DiagnosticBudget {
    bytes_per_stream: usize,
}

impl DiagnosticBudget {
    pub(super) fn for_batch(input_count: usize) -> Self {
        let stream_count = input_count.saturating_mul(STREAMS_PER_INPUT);
        let batch_share = MAX_PTXAS_BATCH_DIAGNOSTIC_BYTES
            .checked_div(stream_count)
            .unwrap_or_default();
        Self {
            bytes_per_stream: batch_share.min(MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES),
        }
    }

    pub(super) fn bytes_per_stream(&self) -> usize {
        self.bytes_per_stream
    }
}

#[derive(Default)]
pub(super) struct DiagnosticReaders {
    stdout: Option<DiagnosticReader>,
    stderr: Option<DiagnosticReader>,
    bytes_per_stream: usize,
}

impl DiagnosticReaders {
    pub(super) fn attach(
        &mut self,
        child: &mut Child,
        name: &str,
        bytes_per_stream: usize,
    ) -> Result<(), FinalizerError> {
        let stdout =
            child
                .stdout
                .take()
                .ok_or_else(|| FinalizerError::PtxasDiagnosticUnavailable {
                    name: name.to_string(),
                    stream: "stdout",
                })?;
        let stderr =
            child
                .stderr
                .take()
                .ok_or_else(|| FinalizerError::PtxasDiagnosticUnavailable {
                    name: name.to_string(),
                    stream: "stderr",
                })?;

        self.bytes_per_stream = bytes_per_stream;
        self.stdout = Some(
            spawn_reader(stdout, bytes_per_stream, "stdout").map_err(|source| {
                FinalizerError::PtxasDiagnosticIo {
                    name: name.to_string(),
                    stream: "stdout",
                    source,
                }
            })?,
        );
        self.stderr = Some(
            spawn_reader(stderr, bytes_per_stream, "stderr").map_err(|source| {
                FinalizerError::PtxasDiagnosticIo {
                    name: name.to_string(),
                    stream: "stderr",
                    source,
                }
            })?,
        );
        Ok(())
    }

    pub(super) fn finish(&mut self, name: &str) -> Result<(String, String), FinalizerError> {
        let stdout = join_reader(self.stdout.take(), name, "stdout", self.bytes_per_stream);
        let stderr = join_reader(self.stderr.take(), name, "stderr", self.bytes_per_stream);
        Ok((stdout?, stderr?))
    }

    pub(super) fn discard(&mut self) {
        for reader in [self.stdout.take(), self.stderr.take()]
            .into_iter()
            .flatten()
        {
            let _ = reader.join();
        }
    }
}

impl Drop for DiagnosticReaders {
    fn drop(&mut self) {
        self.discard();
    }
}

type DiagnosticReader = JoinHandle<io::Result<CapturedDiagnostic>>;

fn spawn_reader(
    reader: impl Read + Send + 'static,
    retained_bytes: usize,
    stream: &'static str,
) -> io::Result<DiagnosticReader> {
    thread::Builder::new()
        .name(format!("cuda-oxide-ptxas-{stream}"))
        .spawn(move || capture_diagnostic(reader, retained_bytes))
}

fn join_reader(
    reader: Option<DiagnosticReader>,
    name: &str,
    stream: &'static str,
    output_limit: usize,
) -> Result<String, FinalizerError> {
    let reader = reader.ok_or_else(|| FinalizerError::PtxasDiagnosticUnavailable {
        name: name.to_string(),
        stream,
    })?;
    match reader.join() {
        Ok(Ok(diagnostic)) => Ok(diagnostic.into_bounded_string(output_limit)),
        Ok(Err(source)) => Err(FinalizerError::PtxasDiagnosticIo {
            name: name.to_string(),
            stream,
            source,
        }),
        Err(_) => Err(FinalizerError::PtxasDiagnosticPanicked {
            name: name.to_string(),
            stream,
        }),
    }
}

struct CapturedDiagnostic {
    retained: Vec<u8>,
    actual_bytes: u64,
}

impl CapturedDiagnostic {
    fn into_bounded_string(self, output_limit: usize) -> String {
        let retained_bytes = u64::try_from(self.retained.len()).unwrap_or(u64::MAX);
        let (diagnostic, complete) = lossy_prefix(&self.retained, output_limit);
        if self.actual_bytes == retained_bytes && complete {
            return diagnostic;
        }

        let marker = if self.actual_bytes > retained_bytes {
            format!(
                "\n... [ptxas diagnostic truncated: captured {retained_bytes} of {} bytes]",
                self.actual_bytes
            )
        } else {
            format!(
                "\n... [ptxas diagnostic truncated while rendering {} input bytes]",
                self.actual_bytes
            )
        };
        if marker.len() >= output_limit {
            return utf8_prefix(&marker, output_limit).to_string();
        }
        let payload_limit = output_limit - marker.len();
        let (mut diagnostic, _) = lossy_prefix(&self.retained, payload_limit);
        diagnostic.push_str(&marker);
        diagnostic
    }
}

fn capture_diagnostic(
    mut reader: impl Read,
    retained_limit: usize,
) -> io::Result<CapturedDiagnostic> {
    let mut retained = Vec::with_capacity(retained_limit);
    let mut actual_bytes = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        actual_bytes = actual_bytes.saturating_add(u64::try_from(bytes_read).unwrap_or(u64::MAX));
        let remaining = retained_limit.saturating_sub(retained.len());
        retained.extend_from_slice(&buffer[..bytes_read.min(remaining)]);
    }
    Ok(CapturedDiagnostic {
        retained,
        actual_bytes,
    })
}

fn lossy_prefix(bytes: &[u8], maximum_bytes: usize) -> (String, bool) {
    let mut remaining = bytes;
    let mut output = String::with_capacity(maximum_bytes.min(bytes.len()));
    while !remaining.is_empty() {
        match std::str::from_utf8(remaining) {
            Ok(valid) => {
                let prefix = utf8_prefix(valid, maximum_bytes.saturating_sub(output.len()));
                output.push_str(prefix);
                return (output, prefix.len() == valid.len());
            }
            Err(error) => {
                let valid = std::str::from_utf8(&remaining[..error.valid_up_to()])
                    .expect("Utf8Error valid prefix is UTF-8");
                let prefix = utf8_prefix(valid, maximum_bytes.saturating_sub(output.len()));
                output.push_str(prefix);
                if prefix.len() != valid.len() {
                    return (output, false);
                }

                let invalid_bytes = error
                    .error_len()
                    .unwrap_or_else(|| remaining.len() - error.valid_up_to());
                if output
                    .len()
                    .saturating_add(char::REPLACEMENT_CHARACTER.len_utf8())
                    > maximum_bytes
                {
                    return (output, false);
                }
                output.push(char::REPLACEMENT_CHARACTER);
                remaining = &remaining[error.valid_up_to() + invalid_bytes..];
            }
        }
    }
    (output, true)
}

fn utf8_prefix(value: &str, maximum_bytes: usize) -> &str {
    let mut end = value.len().min(maximum_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_capture_drains_input_and_bounds_rendered_text() {
        let input = vec![b'x'; MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES + 17];
        let captured = capture_diagnostic(input.as_slice(), 128).unwrap();
        let diagnostic = captured.into_bounded_string(128);

        assert_eq!(diagnostic.len(), 128);
        assert!(diagnostic.contains("diagnostic truncated"));
    }

    #[test]
    fn batch_budget_is_split_deterministically_across_streams() {
        let budget = DiagnosticBudget::for_batch(7);
        assert_eq!(
            budget.bytes_per_stream(),
            MAX_PTXAS_BATCH_DIAGNOSTIC_BYTES / 14
        );
        assert!(budget.bytes_per_stream().saturating_mul(14) <= MAX_PTXAS_BATCH_DIAGNOSTIC_BYTES);
        assert!(
            DiagnosticBudget::for_batch(1).bytes_per_stream() <= MAX_PTXAS_STREAM_DIAGNOSTIC_BYTES
        );
    }

    #[test]
    fn invalid_utf8_cannot_expand_past_the_stream_budget() {
        let captured = capture_diagnostic([0xff; 64].as_slice(), 64).unwrap();
        let diagnostic = captured.into_bounded_string(64);

        assert!(diagnostic.len() <= 64);
        assert!(diagnostic.is_char_boundary(diagnostic.len()));
    }
}
