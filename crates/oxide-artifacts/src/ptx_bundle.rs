/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Versioned, streaming container for the ordered PTX inputs of one owner.

use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::io::{Read, Write};

/// Fixed V1 bundle magic, including its trailing NUL.
pub const PTX_BUNDLE_MAGIC: [u8; 24] = *b"CUDAOXIDE_PTX_BUNDLE_V1\0";
/// Current PTX bundle version.
pub const PTX_BUNDLE_VERSION: u32 = 1;
/// Suffix appended to an owner name (`<owner>.ptx.bundle`).
pub const PTX_BUNDLE_SUFFIX: &str = "ptx.bundle";

/// Resource limits enforced before a parser allocates or visits record data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PtxBundleLimits {
    pub max_records: u32,
    pub max_name_bytes: u32,
    pub max_record_ptx_bytes: u64,
    pub max_total_ptx_bytes: u64,
}

impl Default for PtxBundleLimits {
    fn default() -> Self {
        Self {
            max_records: 4096,
            max_name_bytes: 1024,
            max_record_ptx_bytes: 1024 * 1024 * 1024,
            max_total_ptx_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

/// Header supplied to a bounded record visitor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PtxBundleRecordHeader {
    pub index: u32,
    pub name: String,
    pub ptx_bytes: u64,
    pub sha256: [u8; 32],
}

/// PTX bundle encode/decode failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PtxBundleError {
    Io(String),
    BadMagic,
    UnsupportedVersion(u32),
    EmptyBundle,
    TooManyRecords {
        actual: u32,
        maximum: u32,
    },
    EmptyName,
    NameTooLarge {
        actual: usize,
        maximum: u32,
    },
    InvalidName(String),
    DuplicateName(String),
    EmptyPtx(String),
    RecordTooLarge {
        name: String,
        actual: u64,
        maximum: u64,
    },
    TotalTooLarge {
        actual: u64,
        maximum: u64,
    },
    Truncated(&'static str),
    InvalidUtf8Name,
    DigestMismatch(String),
    TrailingBytes,
    RecordCountMismatch {
        expected: u32,
        actual: u32,
    },
    Visitor(String),
}

impl fmt::Display for PtxBundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "PTX bundle I/O: {error}"),
            Self::BadMagic => write!(formatter, "invalid PTX bundle magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported PTX bundle version {version}")
            }
            Self::EmptyBundle => write!(formatter, "PTX bundle has no records"),
            Self::TooManyRecords { actual, maximum } => {
                write!(
                    formatter,
                    "PTX bundle has {actual} records; maximum is {maximum}"
                )
            }
            Self::EmptyName => write!(formatter, "PTX bundle record name is empty"),
            Self::NameTooLarge { actual, maximum } => write!(
                formatter,
                "PTX bundle record name has {actual} bytes; maximum is {maximum}"
            ),
            Self::InvalidName(name) => write!(formatter, "invalid PTX bundle record name {name:?}"),
            Self::DuplicateName(name) => {
                write!(formatter, "duplicate PTX bundle record name {name:?}")
            }
            Self::EmptyPtx(name) => write!(formatter, "PTX bundle record {name:?} is empty"),
            Self::RecordTooLarge {
                name,
                actual,
                maximum,
            } => write!(
                formatter,
                "PTX bundle record {name:?} has {actual} bytes; maximum is {maximum}"
            ),
            Self::TotalTooLarge { actual, maximum } => write!(
                formatter,
                "PTX bundle records total {actual} bytes; maximum is {maximum}"
            ),
            Self::Truncated(field) => write!(formatter, "truncated PTX bundle {field}"),
            Self::InvalidUtf8Name => write!(formatter, "PTX bundle record name is not UTF-8"),
            Self::DigestMismatch(name) => {
                write!(
                    formatter,
                    "PTX bundle record {name:?} failed SHA-256 verification"
                )
            }
            Self::TrailingBytes => write!(formatter, "PTX bundle has trailing bytes"),
            Self::RecordCountMismatch { expected, actual } => write!(
                formatter,
                "PTX bundle writer expected {expected} records but received {actual}"
            ),
            Self::Visitor(error) => write!(formatter, "PTX bundle visitor: {error}"),
        }
    }
}

impl std::error::Error for PtxBundleError {}

/// Streaming writer that retains only record names, not prior PTX payloads.
pub struct PtxBundleWriter<W: Write> {
    output: W,
    limits: PtxBundleLimits,
    expected_records: u32,
    written_records: u32,
    total_ptx_bytes: u64,
    names: HashSet<String>,
}

impl<W: Write> PtxBundleWriter<W> {
    pub fn new(
        mut output: W,
        expected_records: u32,
        limits: PtxBundleLimits,
    ) -> Result<Self, PtxBundleError> {
        validate_record_count(expected_records, limits)?;
        output
            .write_all(&PTX_BUNDLE_MAGIC)
            .and_then(|_| output.write_all(&PTX_BUNDLE_VERSION.to_le_bytes()))
            .and_then(|_| output.write_all(&expected_records.to_le_bytes()))
            .map_err(io_error)?;
        Ok(Self {
            output,
            limits,
            expected_records,
            written_records: 0,
            total_ptx_bytes: 0,
            names: HashSet::new(),
        })
    }

    pub fn add_record(&mut self, name: &str, ptx: &[u8]) -> Result<[u8; 32], PtxBundleError> {
        validate_name(name, self.limits)?;
        if !self.names.insert(name.to_string()) {
            return Err(PtxBundleError::DuplicateName(name.to_string()));
        }
        validate_ptx_size(name, ptx.len() as u64, self.limits)?;
        self.total_ptx_bytes = self.total_ptx_bytes.checked_add(ptx.len() as u64).ok_or(
            PtxBundleError::TotalTooLarge {
                actual: u64::MAX,
                maximum: self.limits.max_total_ptx_bytes,
            },
        )?;
        if self.total_ptx_bytes > self.limits.max_total_ptx_bytes {
            return Err(PtxBundleError::TotalTooLarge {
                actual: self.total_ptx_bytes,
                maximum: self.limits.max_total_ptx_bytes,
            });
        }
        if self.written_records == self.expected_records {
            return Err(PtxBundleError::RecordCountMismatch {
                expected: self.expected_records,
                actual: self.written_records + 1,
            });
        }
        let digest: [u8; 32] = Sha256::digest(ptx).into();
        self.output
            .write_all(&(name.len() as u32).to_le_bytes())
            .and_then(|_| self.output.write_all(&(ptx.len() as u64).to_le_bytes()))
            .and_then(|_| self.output.write_all(&digest))
            .and_then(|_| self.output.write_all(name.as_bytes()))
            .and_then(|_| self.output.write_all(ptx))
            .map_err(io_error)?;
        self.written_records += 1;
        Ok(digest)
    }

    pub fn finish(mut self) -> Result<W, PtxBundleError> {
        if self.written_records != self.expected_records {
            return Err(PtxBundleError::RecordCountMismatch {
                expected: self.expected_records,
                actual: self.written_records,
            });
        }
        self.output.flush().map_err(io_error)?;
        Ok(self.output)
    }
}

/// Visit each record without allocating any PTX-sized aggregate.
///
/// The visitor receives a reader capped at the current record. Unconsumed
/// bytes are drained and hashed before the next record. A visitor can therefore
/// stream a record directly to a transaction-owned file.
pub fn visit_ptx_bundle<R, F>(
    mut input: R,
    limits: PtxBundleLimits,
    mut visitor: F,
) -> Result<(), PtxBundleError>
where
    R: Read,
    F: FnMut(&PtxBundleRecordHeader, &mut dyn Read) -> std::io::Result<()>,
{
    let mut magic = [0_u8; PTX_BUNDLE_MAGIC.len()];
    read_exact(&mut input, &mut magic, "magic")?;
    if magic != PTX_BUNDLE_MAGIC {
        return Err(PtxBundleError::BadMagic);
    }
    let version = read_u32(&mut input, "version")?;
    if version != PTX_BUNDLE_VERSION {
        return Err(PtxBundleError::UnsupportedVersion(version));
    }
    let count = read_u32(&mut input, "record count")?;
    validate_record_count(count, limits)?;

    let mut names = HashSet::new();
    let mut total_ptx_bytes = 0_u64;
    for index in 0..count {
        let name_len = read_u32(&mut input, "record name length")? as usize;
        let ptx_bytes = read_u64(&mut input, "record PTX length")?;
        let mut sha256 = [0_u8; 32];
        read_exact(&mut input, &mut sha256, "record SHA-256")?;
        if name_len > limits.max_name_bytes as usize {
            return Err(PtxBundleError::NameTooLarge {
                actual: name_len,
                maximum: limits.max_name_bytes,
            });
        }
        let mut name = vec![0_u8; name_len];
        read_exact(&mut input, &mut name, "record name")?;
        let name = String::from_utf8(name).map_err(|_| PtxBundleError::InvalidUtf8Name)?;
        validate_name(&name, limits)?;
        if !names.insert(name.clone()) {
            return Err(PtxBundleError::DuplicateName(name));
        }
        validate_ptx_size(&name, ptx_bytes, limits)?;
        total_ptx_bytes =
            total_ptx_bytes
                .checked_add(ptx_bytes)
                .ok_or(PtxBundleError::TotalTooLarge {
                    actual: u64::MAX,
                    maximum: limits.max_total_ptx_bytes,
                })?;
        if total_ptx_bytes > limits.max_total_ptx_bytes {
            return Err(PtxBundleError::TotalTooLarge {
                actual: total_ptx_bytes,
                maximum: limits.max_total_ptx_bytes,
            });
        }

        let header = PtxBundleRecordHeader {
            index,
            name: name.clone(),
            ptx_bytes,
            sha256,
        };
        let mut record = DigestingRecordReader::new(&mut input, ptx_bytes);
        visitor(&header, &mut record)
            .map_err(|error| PtxBundleError::Visitor(error.to_string()))?;
        std::io::copy(&mut record, &mut std::io::sink())
            .map_err(|error| PtxBundleError::Io(error.to_string()))?;
        if record.remaining != 0 {
            return Err(PtxBundleError::Truncated("record PTX"));
        }
        let actual_digest = record.finish();
        if actual_digest != sha256 {
            return Err(PtxBundleError::DigestMismatch(name));
        }
    }

    let mut trailing = [0_u8; 1];
    if input.read(&mut trailing).map_err(io_error)? != 0 {
        return Err(PtxBundleError::TrailingBytes);
    }
    Ok(())
}

struct DigestingRecordReader<'a, R> {
    input: &'a mut R,
    remaining: u64,
    digest: Sha256,
}

impl<'a, R: Read> DigestingRecordReader<'a, R> {
    fn new(input: &'a mut R, remaining: u64) -> Self {
        Self {
            input,
            remaining,
            digest: Sha256::new(),
        }
    }

    fn finish(self) -> [u8; 32] {
        self.digest.finalize().into()
    }
}

impl<R: Read> Read for DigestingRecordReader<'_, R> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || output.is_empty() {
            return Ok(0);
        }
        let limit = usize::try_from(self.remaining)
            .unwrap_or(usize::MAX)
            .min(output.len());
        let read = self.input.read(&mut output[..limit])?;
        if read != 0 {
            self.remaining -= read as u64;
            self.digest.update(&output[..read]);
        }
        Ok(read)
    }
}

fn validate_record_count(count: u32, limits: PtxBundleLimits) -> Result<(), PtxBundleError> {
    if count == 0 {
        return Err(PtxBundleError::EmptyBundle);
    }
    if count > limits.max_records {
        return Err(PtxBundleError::TooManyRecords {
            actual: count,
            maximum: limits.max_records,
        });
    }
    Ok(())
}

fn validate_name(name: &str, limits: PtxBundleLimits) -> Result<(), PtxBundleError> {
    if name.is_empty() {
        return Err(PtxBundleError::EmptyName);
    }
    if name.len() > limits.max_name_bytes as usize {
        return Err(PtxBundleError::NameTooLarge {
            actual: name.len(),
            maximum: limits.max_name_bytes,
        });
    }
    if matches!(name, "." | "..")
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PtxBundleError::InvalidName(name.to_string()));
    }
    Ok(())
}

fn validate_ptx_size(
    name: &str,
    ptx_bytes: u64,
    limits: PtxBundleLimits,
) -> Result<(), PtxBundleError> {
    if ptx_bytes == 0 {
        return Err(PtxBundleError::EmptyPtx(name.to_string()));
    }
    if ptx_bytes > limits.max_record_ptx_bytes {
        return Err(PtxBundleError::RecordTooLarge {
            name: name.to_string(),
            actual: ptx_bytes,
            maximum: limits.max_record_ptx_bytes,
        });
    }
    Ok(())
}

fn read_u32(input: &mut impl Read, field: &'static str) -> Result<u32, PtxBundleError> {
    let mut bytes = [0_u8; 4];
    read_exact(input, &mut bytes, field)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &mut impl Read, field: &'static str) -> Result<u64, PtxBundleError> {
    let mut bytes = [0_u8; 8];
    read_exact(input, &mut bytes, field)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact(
    input: &mut impl Read,
    output: &mut [u8],
    field: &'static str,
) -> Result<(), PtxBundleError> {
    input
        .read_exact(output)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => PtxBundleError::Truncated(field),
            _ => io_error(error),
        })
}

fn io_error(error: std::io::Error) -> PtxBundleError {
    PtxBundleError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn bundle(records: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer =
            PtxBundleWriter::new(Vec::new(), records.len() as u32, PtxBundleLimits::default())
                .unwrap();
        for (name, ptx) in records {
            writer.add_record(name, ptx).unwrap();
        }
        writer.finish().unwrap()
    }

    #[test]
    fn round_trip_is_ordered_and_bounded() {
        let bytes = bundle(&[
            ("part-b.ptx", b".entry b() {}\n"),
            ("part-a.ptx", b".entry a() {}\n"),
        ]);
        let mut records = Vec::new();
        visit_ptx_bundle(
            Cursor::new(bytes),
            PtxBundleLimits::default(),
            |header, ptx| {
                let mut bytes = Vec::new();
                ptx.read_to_end(&mut bytes)?;
                records.push((header.clone(), bytes));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(records[0].0.name, "part-b.ptx");
        assert_eq!(records[0].1, b".entry b() {}\n");
        assert_eq!(records[1].0.name, "part-a.ptx");
        assert_eq!(records[1].1, b".entry a() {}\n");
    }

    #[test]
    fn malformed_and_truncated_bundles_fail_closed() {
        let mut bad_magic = bundle(&[("a.ptx", b"a")]);
        bad_magic[0] ^= 1;
        assert_eq!(
            visit_ptx_bundle(
                Cursor::new(bad_magic),
                PtxBundleLimits::default(),
                |_, _| Ok(())
            ),
            Err(PtxBundleError::BadMagic)
        );

        let mut truncated = bundle(&[("a.ptx", b"abcdef")]);
        truncated.pop();
        assert_eq!(
            visit_ptx_bundle(
                Cursor::new(truncated),
                PtxBundleLimits::default(),
                |_, _| Ok(())
            ),
            Err(PtxBundleError::Truncated("record PTX"))
        );
    }

    #[test]
    fn duplicate_names_are_rejected_by_writer_and_parser() {
        let mut writer = PtxBundleWriter::new(Vec::new(), 2, PtxBundleLimits::default()).unwrap();
        writer.add_record("same.ptx", b"a").unwrap();
        assert_eq!(
            writer.add_record("same.ptx", b"b"),
            Err(PtxBundleError::DuplicateName("same.ptx".to_string()))
        );

        let mut raw = Vec::new();
        raw.extend_from_slice(&PTX_BUNDLE_MAGIC);
        raw.extend_from_slice(&PTX_BUNDLE_VERSION.to_le_bytes());
        raw.extend_from_slice(&2_u32.to_le_bytes());
        for ptx in [b"a".as_slice(), b"b".as_slice()] {
            raw.extend_from_slice(&8_u32.to_le_bytes());
            raw.extend_from_slice(&1_u64.to_le_bytes());
            raw.extend_from_slice(&<[u8; 32]>::from(Sha256::digest(ptx)));
            raw.extend_from_slice(b"same.ptx");
            raw.extend_from_slice(ptx);
        }
        assert_eq!(
            visit_ptx_bundle(Cursor::new(raw), PtxBundleLimits::default(), |_, _| Ok(())),
            Err(PtxBundleError::DuplicateName("same.ptx".to_string()))
        );
    }

    #[test]
    fn record_count_and_size_limits_are_enforced_before_payload_visit() {
        let bytes = bundle(&[("a.ptx", b"abcd"), ("b.ptx", b"efgh")]);
        let mut count_limits = PtxBundleLimits::default();
        count_limits.max_records = 1;
        assert_eq!(
            visit_ptx_bundle(Cursor::new(&bytes), count_limits, |_, _| Ok(())),
            Err(PtxBundleError::TooManyRecords {
                actual: 2,
                maximum: 1
            })
        );

        let mut size_limits = PtxBundleLimits::default();
        size_limits.max_record_ptx_bytes = 3;
        assert_eq!(
            visit_ptx_bundle(Cursor::new(bytes), size_limits, |_, _| Ok(())),
            Err(PtxBundleError::RecordTooLarge {
                name: "a.ptx".to_string(),
                actual: 4,
                maximum: 3
            })
        );
    }

    #[test]
    fn writer_count_digest_and_trailing_bytes_fail_closed() {
        let writer = PtxBundleWriter::new(Vec::new(), 2, PtxBundleLimits::default()).unwrap();
        assert!(matches!(
            writer.finish(),
            Err(PtxBundleError::RecordCountMismatch {
                expected: 2,
                actual: 0
            })
        ));

        let mut digest = bundle(&[("a.ptx", b"abc")]);
        *digest.last_mut().unwrap() ^= 1;
        assert_eq!(
            visit_ptx_bundle(Cursor::new(digest), PtxBundleLimits::default(), |_, _| Ok(
                ()
            )),
            Err(PtxBundleError::DigestMismatch("a.ptx".to_string()))
        );

        let mut trailing = bundle(&[("a.ptx", b"abc")]);
        trailing.push(0);
        assert_eq!(
            visit_ptx_bundle(
                Cursor::new(trailing),
                PtxBundleLimits::default(),
                |_, _| Ok(())
            ),
            Err(PtxBundleError::TrailingBytes)
        );
    }

    #[test]
    fn record_names_are_conservative_path_components() {
        for name in [
            ".",
            "..",
            "../part.ptx",
            "nested/part.ptx",
            "nested\\part.ptx",
            "part name.ptx",
            "part\nname.ptx",
            "π.ptx",
        ] {
            let mut writer =
                PtxBundleWriter::new(Vec::new(), 1, PtxBundleLimits::default()).unwrap();
            assert_eq!(
                writer.add_record(name, b"ptx"),
                Err(PtxBundleError::InvalidName(name.to_string())),
                "{name:?}"
            );
        }
        let mut writer = PtxBundleWriter::new(Vec::new(), 1, PtxBundleLimits::default()).unwrap();
        writer.add_record("part-a_b.01.ptx", b"ptx").unwrap();
        writer.finish().unwrap();
    }
}
