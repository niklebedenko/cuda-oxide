/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-independent CUDA artifact finalization.
//!
//! This crate is the single owner of cuda-oxide's libNVVM, ptxas, and nvJitLink
//! compilation policy. It deliberately does not link the CUDA Driver. Both
//! build-time materialization and runtime fallback use the same typed target,
//! FMA, debug, input-order, validation, and provenance rules.

mod link;
mod nvvm;
mod options;
mod partition;
mod provenance;
mod ptxas;
mod validation;

pub use libnvvm_sys::{CudaArch, CudaArchParseError, LibdeviceNotFound, NvvmError, find_libdevice};
pub use link::LtoLinker;
pub use nvjitlink_sys::NvJitLinkError;
pub use nvvm::NvvmCompiler;
pub use options::{DebugPolicy, FinalizationOptions, FinalizerOutput, NamedInput};
pub use provenance::{ToolProvenance, recipe_digest};
pub use validation::is_valid_cubin;

use provenance::common_provenance_digest;
use ptxas::{PtxAssembler, PtxAssemblyInput};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use thiserror::Error;

/// Maximum bytes accepted from one on-disk NVVM IR owner partition.
///
/// The partition planner targets substantially smaller artifacts. This is a
/// fail-closed allocation ceiling for corrupt, stale, or unexpectedly large
/// materialization inputs.
pub const MAX_PARTITION_NVVM_IR_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum bytes accepted by ptxas from one compiled PTX partition.
///
/// This separately bounds NVVM IR that expands pathologically during
/// compilation. The 32 MiB ceiling retains 57% headroom over the largest
/// measured production partition while still rejecting runaway expansion.
pub const MAX_PARTITION_PTX_BYTES: u64 = 32 * 1024 * 1024;

/// Maximum direct PTX input accepted by the slower nvJitLink fallback.
pub const MAX_PARTITION_NVJITLINK_PTX_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum relocatable object bytes read back from one ptxas child.
pub const MAX_PARTITION_OBJECT_BYTES: u64 = 64 * 1024 * 1024;

/// Failures while compiling NVVM IR or linking CUDA device artifacts.
#[derive(Debug, Error)]
pub enum FinalizerError {
    /// libNVVM failed to load, validate, or compile.
    #[error("libnvvm: {0}")]
    Nvvm(#[from] libnvvm_sys::NvvmError),

    /// nvJitLink failed to load or link.
    #[error("nvJitLink: {0}")]
    NvJitLink(#[from] nvjitlink_sys::NvJitLinkError),

    /// `libdevice.10.bc` could not be found.
    #[error(
        "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibdeviceNotFound {
        /// Newline-separated discovery paths.
        tried: String,
    },

    /// A finalizer input could not be read.
    #[error("Failed reading {path}: {source}")]
    Io {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },

    /// One owner partition exceeded the finalizer's hard allocation ceiling.
    #[error(
        "CUDA owner partition {path} is {actual_bytes} bytes, exceeding the {maximum_bytes}-byte source limit"
    )]
    PartitionSourceTooLarge {
        path: PathBuf,
        actual_bytes: u64,
        maximum_bytes: u64,
    },

    /// An owner partition path did not identify a regular file.
    #[error("CUDA owner partition is not a regular file: {path}")]
    PartitionSourceNotRegular { path: PathBuf },

    /// An owner partition changed after its size was checked.
    #[error(
        "CUDA owner partition {path} changed while it was being read (initial size {initial_bytes} bytes)"
    )]
    PartitionSourceChanged { path: PathBuf, initial_bytes: u64 },

    /// Compiled PTX exceeded the selected per-record consumer ceiling.
    #[error(
        "CUDA owner partition {name} compiled to {actual_bytes} PTX bytes, exceeding the {maximum_bytes}-byte per-partition limit"
    )]
    CompiledPtxTooLarge {
        name: String,
        actual_bytes: u64,
        maximum_bytes: u64,
    },

    /// The bounded libNVVM worker override was not an integer from one to four.
    #[error(
        "invalid CUDA_OXIDE_NVVM_WORKERS value {value:?}; expected an integer from 1 through 4"
    )]
    InvalidNvvmWorkerCount { value: String },

    /// An operating-system thread could not be created for the NVVM batch.
    #[error("could not create a bounded libNVVM worker: {source}")]
    NvvmWorkerSpawn {
        #[source]
        source: std::io::Error,
    },

    /// A bounded NVVM worker panicked instead of returning a typed error.
    #[error("libNVVM partition worker panicked at index {index:?} for {name}")]
    NvvmWorkerPanicked { index: Option<usize>, name: String },

    /// An explicitly selected ptxas path is not an executable regular file.
    #[error("ptxas path is not an executable regular file: {path}")]
    PtxasNotExecutable { path: PathBuf },

    /// ptxas discovery, diagnostics, or output I/O failed.
    #[error("ptxas I/O failed for {path}: {source}")]
    PtxasIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A pinned ptxas process could not be launched.
    #[error("could not launch pinned ptxas {path} for {name}: {source}")]
    PtxasLaunch {
        path: PathBuf,
        name: String,
        #[source]
        source: std::io::Error,
    },

    /// A launched ptxas process could not be observed or reaped.
    #[error("could not wait for ptxas compiling {name}: {source}")]
    PtxasWait {
        name: String,
        #[source]
        source: std::io::Error,
    },

    /// A ptxas diagnostic pipe or capture worker could not perform I/O.
    #[error("could not capture ptxas {stream} for {name}: {source}")]
    PtxasDiagnosticIo {
        name: String,
        stream: &'static str,
        #[source]
        source: std::io::Error,
    },

    /// A launched ptxas child did not expose the configured diagnostic pipe.
    #[error("ptxas {stream} capture pipe is unavailable for {name}")]
    PtxasDiagnosticUnavailable { name: String, stream: &'static str },

    /// A ptxas diagnostic capture worker panicked.
    #[error("ptxas {stream} capture worker panicked for {name}")]
    PtxasDiagnosticPanicked { name: String, stream: &'static str },

    /// ptxas rejected one partition.
    #[error(
        "ptxas failed compiling {name} ({status})\n--- ptxas stdout ---\n{stdout}\n--- ptxas stderr ---\n{stderr}"
    )]
    PtxasFailed {
        name: String,
        status: String,
        stdout: String,
        stderr: String,
    },

    /// ptxas reported success without a usable relocatable.
    #[error("ptxas reported success for {name} but did not create a nonempty object at {path}")]
    PtxasMissingOutput { name: String, path: PathBuf },

    /// Owner-level PTX bundle encoding failed.
    #[error("PTX bundle: {0}")]
    PtxBundle(#[from] oxide_artifacts::ptx_bundle::PtxBundleError),

    /// The installed toolkit does not accept cuda-oxide's NVVM IR version.
    #[error("installed libNVVM accepts NVVM IR {major}.{minor}, but cuda-oxide emits NVVM IR 2.0")]
    UnsupportedNvvmIrVersion { major: i32, minor: i32 },

    /// Runtime toolkit dialect discovery disagreed with the target policy.
    #[error(
        "libNVVM reports LLVM {llvm_major} for {target}, which disagrees with cuda-oxide's expected {expected} dialect"
    )]
    DialectMismatch {
        target: String,
        llvm_major: i32,
        expected: &'static str,
    },

    /// A diagnostic input name cannot be represented by the CUDA C APIs.
    #[error("CUDA artifact input name contains an interior NUL byte: {name:?}")]
    InvalidInputName { name: String },

    /// A supplied compiler or linker input contained no bytes.
    #[error("CUDA artifact input is empty: {name}")]
    EmptyInput { name: String },

    /// nvJitLink may parse PTX as a C string and ignore bytes after a NUL.
    #[error("CUDA PTX input {name:?} contains an interior NUL byte at offset {offset}")]
    InteriorNulPtx {
        /// Diagnostic input name supplied by the caller.
        name: String,
        /// Byte offset of the first non-trailing NUL.
        offset: usize,
    },

    /// PTX is textual input and must be valid UTF-8 for deterministic
    /// cross-partition definition validation.
    #[error("CUDA PTX input {name:?} is not valid UTF-8")]
    InvalidPtxText { name: String },

    /// A generated weak storage declaration was truncated or malformed.
    #[error("CUDA PTX input {name:?} has malformed weak storage declaration: {declaration:?}")]
    MalformedWeakStorageDefinition { name: String, declaration: String },

    /// ODR-coalesced statics must have identical type, address space,
    /// alignment, and initializer in every partition.
    #[error(
        "CUDA PTX inputs {first_input:?} and {second_input:?} define incompatible weak storage symbol {symbol:?}"
    )]
    IncompatibleWeakStorageDefinition {
        symbol: String,
        first_input: String,
        second_input: String,
    },

    /// nvJitLink was invoked without an input module.
    #[error("at least one ordered linker input is required")]
    NoLinkInputs,

    /// nvJitLink returned bytes that are not a complete CUDA ELF image.
    #[error("nvJitLink returned an invalid or truncated cubin")]
    InvalidCubin,

    /// The cubin symbol table could not establish its public kernel inventory.
    #[error("could not validate CUDA cubin kernel inventory: {detail}")]
    InvalidCubinKernelInventory { detail: String },

    /// The linked cubin did not preserve exactly the public PTX entry points.
    #[error(
        "linked CUDA cubin kernel inventory differs from PTX (expected {expected_count}, found {actual_count}; missing={missing:?}, unexpected={unexpected:?})"
    )]
    CubinKernelInventoryMismatch {
        expected_count: usize,
        actual_count: usize,
        missing: Vec<String>,
        unexpected: Vec<String>,
    },

    /// nvJitLink returned no PTX bytes.
    #[error("nvJitLink returned an empty PTX artifact")]
    EmptyPtx,

    /// A pinned CUDA compiler DSO changed around an operation. Its output can
    /// no longer be attributed to the provenance used by Cargo or a cache key.
    #[error(
        "the pinned {tool} file changed before or during CUDA artifact finalization; refusing the unverified output"
    )]
    ToolIdentityChanged { tool: &'static str },
}

/// Outputs from one NVVM IR materialization plan.
///
/// `ptx_input` is the exact whole-module PTX source produced by libNVVM.
/// nvJitLink may receive separate trailing-NUL FFI backing without changing
/// these source bytes. Keeping the source and cubin from the same invocation
/// lets callers publish an audit sidecar without recompiling the NVVM IR or
/// guessing which PTX policy produced the embedded image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedNvvmIr {
    /// Exact whole-module PTX source produced by libNVVM.
    pub ptx_input: Vec<u8>,
    /// Validated target-specific cubin produced from `ptx_input`.
    pub cubin: Vec<u8>,
}

/// On-disk input kind for bounded owner materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartitionFileInputKind {
    /// NVVM IR compiled to PTX by libNVVM before it is added to nvJitLink.
    NvvmIr,
    /// Existing PTX added directly to nvJitLink.
    Ptx,
}

/// One deterministically named owner partition stored on disk.
#[derive(Clone, Copy, Debug)]
pub struct PartitionFileInput<'a> {
    /// Diagnostic/link input name. Ordered names are part of the build plan.
    pub name: &'a str,
    /// NVVM IR or PTX source file.
    pub path: &'a Path,
    /// Source representation.
    pub kind: PartitionFileInputKind,
}

impl<'a> PartitionFileInput<'a> {
    pub fn new(name: &'a str, path: &'a Path, kind: PartitionFileInputKind) -> Self {
        Self { name, path, kind }
    }
}

/// Measurements and sidecar identity for one sequentially consumed partition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedPartition {
    pub name: String,
    pub source_bytes: usize,
    pub ptx_bytes: usize,
    pub ptx_sha256: [u8; 32],
    /// Time spent in libNVVM. Direct PTX inputs report zero.
    pub nvvm_compile_elapsed: Duration,
    /// Time spent compiling this PTX partition to a relocatable object.
    /// Direct nvJitLink PTX fallback reports zero.
    pub ptxas_elapsed: Duration,
    /// Relocatable object size. Direct nvJitLink PTX fallback reports zero.
    pub object_bytes: usize,
    /// Peak resident set sampled from this ptxas child.
    pub ptxas_peak_rss_kib: Option<u64>,
    /// Time spent adding this PTX or relocatable module to nvJitLink.
    pub jit_link_add_elapsed: Duration,
    /// Process high-water resident set after adding this input.
    pub peak_rss_kib: Option<u64>,
}

/// One cubin linked from bounded, sequential on-disk partitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedPartitionedOwner {
    pub partitions: Vec<MaterializedPartition>,
    /// Wall time from the first libNVVM compile starting until the last one
    /// finishes. Direct-PTX-only plans report zero.
    pub nvvm_wall_elapsed: Duration,
    /// Maximum number of simultaneously active libNVVM program compilations.
    pub nvvm_peak_concurrency: usize,
    /// Maximum number of simultaneously live ptxas children.
    pub ptxas_peak_concurrency: usize,
    /// Maximum sampled sum of resident ptxas child memory.
    pub ptxas_peak_aggregate_rss_kib: Option<u64>,
    pub ptx_bundle_path: PathBuf,
    /// Present when the complete bundle was atomically published but syncing
    /// its parent directory failed. The live file remains safe to consume;
    /// only its survival across an immediate system crash is uncertain.
    pub ptx_bundle_durability_warning: Option<String>,
    pub cubin: Vec<u8>,
    /// Time spent completing the already-populated nvJitLink state.
    pub link_elapsed: Duration,
    pub peak_rss_kib: Option<u64>,
}

/// Complete NVVM IR to cubin/PTX finalizer.
#[derive(Clone)]
pub struct Finalizer {
    compiler: NvvmCompiler,
    linker: LtoLinker,
    assembler: Option<PtxAssembler>,
}

impl Finalizer {
    /// Discover libNVVM, libdevice, and nvJitLink without loading the Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            compiler: NvvmCompiler::discover()?,
            linker: LtoLinker::discover()?,
            assembler: PtxAssembler::discover()?,
        })
    }

    /// Compile one NVVM IR module and return a validated target-specific cubin.
    pub fn materialize_nvvm_ir(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        Ok(self
            .materialize_nvvm_ir_with_ptx(module_name, nvvm_ir, options)?
            .cubin)
    }

    /// Compile one NVVM IR module and return both the exact whole-module PTX
    /// source and the validated target-specific cubin produced from it.
    pub fn materialize_nvvm_ir_with_ptx(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<MaterializedNvvmIr, FinalizerError> {
        let ptx_input = self
            .compiler
            .compile_nvvm_ir_to_ptx(module_name, nvvm_ir, options)?;
        let ptx_name = format!("{module_name}.ptx");
        let cubin = self
            .linker
            .link_ptx(&[NamedInput::new(&ptx_name, &ptx_input)], options)?;
        Ok(MaterializedNvvmIr { ptx_input, cubin })
    }

    /// Consume ordered owner partitions with bounded compilation and link one cubin.
    ///
    /// Up to four distinct libNVVM programs compile in scoped workers. During
    /// deferred-inline selection, one worker may temporarily retain its bounded
    /// source, baseline PTX, promoted source, and promoted PTX. Each worker then
    /// writes one private indexed PTX file and returns only bounded metadata.
    /// Validation, bundle construction, ptxas assembly, and nvJitLink consumption
    /// remain deterministic and phase-separated. `CUDA_OXIDE_NVVM_WORKERS=1..4`
    /// can lower the worker ceiling for controlled validation.
    pub fn materialize_partition_files(
        &self,
        inputs: &[PartitionFileInput<'_>],
        ptx_bundle_path: &Path,
        options: &FinalizationOptions,
    ) -> Result<MaterializedPartitionedOwner, FinalizerError> {
        let worker_limit = partition::configured_nvvm_worker_limit()?;
        self.materialize_partition_files_with_nvvm_workers(
            inputs,
            ptx_bundle_path,
            options,
            worker_limit,
        )
    }

    fn materialize_partition_files_with_nvvm_workers(
        &self,
        inputs: &[PartitionFileInput<'_>],
        ptx_bundle_path: &Path,
        options: &FinalizationOptions,
        worker_limit: usize,
    ) -> Result<MaterializedPartitionedOwner, FinalizerError> {
        if inputs.is_empty() {
            return Err(FinalizerError::NoLinkInputs);
        }
        let (pending_bundle, bundle_file) = PendingPtxBundle::create(ptx_bundle_path)?;
        let temporary_directory = tempfile::Builder::new()
            .prefix(".cuda-oxide-finalizer-")
            .tempdir_in(nonempty_parent(ptx_bundle_path))
            .map_err(|source| FinalizerError::Io {
                path: nonempty_parent(ptx_bundle_path).to_path_buf(),
                source,
            })?;
        let mut bundle = oxide_artifacts::ptx_bundle::PtxBundleWriter::new(
            bundle_file,
            u32::try_from(inputs.len()).map_err(|_| {
                oxide_artifacts::ptx_bundle::PtxBundleError::TooManyRecords {
                    actual: u32::MAX,
                    maximum: oxide_artifacts::ptx_bundle::PtxBundleLimits::default().max_records,
                }
            })?,
            oxide_artifacts::ptx_bundle::PtxBundleLimits::default(),
        )?;
        let mut partitions = Vec::with_capacity(inputs.len());
        let mut prepared = Vec::with_capacity(inputs.len());
        let mut weak_storage_definitions = WeakStorageDefinitions::default();
        let mut expected_kernel_entries = BTreeSet::new();
        let report_partition_stages = std::env::var_os("CUDA_OXIDE_PARTITION_STATS").is_some();
        if report_partition_stages {
            eprintln!(
                "[cuda_artifact_finalizer] partition preparation: partitions={} \
                 worker_limit={} begin",
                inputs.len(),
                worker_limit,
            );
        }
        let prepared_ptx = partition::prepare_partition_ptx_files(
            &self.compiler,
            inputs,
            temporary_directory.path(),
            options,
            worker_limit,
            report_partition_stages,
        )?;
        if report_partition_stages {
            eprintln!(
                "[cuda_artifact_finalizer] partition preparation: nvvm_wall_elapsed={:?} \
                 nvvm_peak_concurrency={} complete",
                prepared_ptx.nvvm_wall_elapsed, prepared_ptx.nvvm_peak_concurrency,
            );
        }
        let nvvm_wall_elapsed = prepared_ptx.nvvm_wall_elapsed;
        let nvvm_peak_concurrency = prepared_ptx.nvvm_peak_concurrency;
        for (index, (input, metadata)) in inputs.iter().zip(prepared_ptx.partitions).enumerate() {
            let ptx_path = temporary_directory.path().join(format!("{index:04}.ptx"));
            let ptx = read_partition_source_capped(&ptx_path, MAX_PARTITION_PTX_BYTES)?;
            if ptx.len() != metadata.ptx_bytes {
                return Err(FinalizerError::PartitionSourceChanged {
                    path: ptx_path,
                    initial_bytes: metadata.ptx_bytes as u64,
                });
            }
            validate_compiled_ptx_size(input.name, &ptx, MAX_PARTITION_PTX_BYTES)?;
            weak_storage_definitions.observe(input.name, &ptx)?;
            let ptx_text = std::str::from_utf8(&ptx)
                .expect("weak-storage validation already established UTF-8 PTX");
            expected_kernel_entries.extend(validation::ptx_kernel_entries(ptx_text));
            let ptx_name = Path::new(input.name)
                .with_extension("ptx")
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(input.name)
                .to_string();
            let ptx_sha256 = bundle.add_record(&ptx_name, &ptx)?;
            let object_path = temporary_directory.path().join(format!("{index:04}.o"));
            prepared.push(PreparedPartition {
                ptx_name,
                ptx_path,
                object_path,
            });
            partitions.push(MaterializedPartition {
                name: input.name.to_string(),
                source_bytes: metadata.source_bytes,
                ptx_bytes: ptx.len(),
                ptx_sha256,
                nvvm_compile_elapsed: metadata.nvvm_compile_elapsed,
                ptxas_elapsed: Duration::ZERO,
                object_bytes: 0,
                ptxas_peak_rss_kib: None,
                jit_link_add_elapsed: Duration::ZERO,
                peak_rss_kib: process_peak_rss_kib(),
            });
        }

        let mut ptxas_peak_concurrency = 0;
        let mut ptxas_peak_aggregate_rss_kib = None;
        let (cubin, link_elapsed) = if let Some(assembler) = &self.assembler {
            if report_partition_stages {
                eprintln!(
                    "[cuda_artifact_finalizer] ptxas batch: partitions={} tool={} begin",
                    prepared.len(),
                    assembler.path().display(),
                );
            }
            let assembly_inputs = prepared
                .iter()
                .zip(inputs)
                .map(|(prepared, input)| PtxAssemblyInput {
                    name: input.name,
                    ptx_path: &prepared.ptx_path,
                    object_path: &prepared.object_path,
                })
                .collect::<Vec<_>>();
            let assembly_batch = assembler.assemble(&assembly_inputs, options)?;
            ptxas_peak_concurrency = assembly_batch.peak_concurrency;
            ptxas_peak_aggregate_rss_kib = assembly_batch.peak_aggregate_rss_kib;
            for (index, result) in assembly_batch.results.iter().enumerate() {
                partitions[index].ptxas_elapsed = result.elapsed;
                partitions[index].object_bytes =
                    usize::try_from(result.object_bytes).unwrap_or(usize::MAX);
                partitions[index].ptxas_peak_rss_kib = result.peak_rss_kib;
                if !result.stdout.trim().is_empty() {
                    eprintln!(
                        "[cuda_artifact_finalizer] ptxas stdout: index={index} name={}\n{}",
                        partitions[index].name,
                        result.stdout.trim_end(),
                    );
                }
                if !result.stderr.trim().is_empty() {
                    eprintln!(
                        "[cuda_artifact_finalizer] ptxas stderr: index={index} name={}\n{}",
                        partitions[index].name,
                        result.stderr.trim_end(),
                    );
                }
                if report_partition_stages {
                    eprintln!(
                        "[cuda_artifact_finalizer] partition stage: index={index} name={} \
                         stage=ptxas complete object_bytes={} elapsed={:?}",
                        partitions[index].name, result.object_bytes, result.elapsed,
                    );
                }
            }
            if report_partition_stages {
                eprintln!(
                    "[cuda_artifact_finalizer] ptxas batch: peak_concurrency={} \
                     peak_aggregate_rss_kib={:?} complete",
                    ptxas_peak_concurrency, ptxas_peak_aggregate_rss_kib,
                );
            }
            self.linker.link_cubin_streaming(options, |linker| {
                for (index, prepared) in prepared.iter().enumerate() {
                    let object = read_partition_source_capped(
                        &prepared.object_path,
                        MAX_PARTITION_OBJECT_BYTES,
                    )?;
                    if object.len() != partitions[index].object_bytes {
                        return Err(FinalizerError::PartitionSourceChanged {
                            path: prepared.object_path.clone(),
                            initial_bytes: partitions[index].object_bytes as u64,
                        });
                    }
                    let object_name = Path::new(&prepared.ptx_name)
                        .with_extension("o")
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(&prepared.ptx_name)
                        .to_string();
                    let link_add_started = std::time::Instant::now();
                    linker.add(&object_name, &object)?;
                    partitions[index].jit_link_add_elapsed = link_add_started.elapsed();
                    partitions[index].peak_rss_kib = process_peak_rss_kib();
                }
                Ok(())
            })?
        } else {
            if report_partition_stages {
                eprintln!(
                    "[cuda_artifact_finalizer] ptxas unavailable; using bounded nvJitLink PTX fallback"
                );
            }
            self.linker.link_ptx_streaming(options, |linker| {
                for (index, prepared) in prepared.iter().enumerate() {
                    let ptx = read_partition_source_capped(
                        &prepared.ptx_path,
                        MAX_PARTITION_NVJITLINK_PTX_BYTES,
                    )?;
                    validate_compiled_ptx_size(
                        &prepared.ptx_name,
                        &ptx,
                        MAX_PARTITION_NVJITLINK_PTX_BYTES,
                    )?;
                    let link_add_started = std::time::Instant::now();
                    linker.add(&prepared.ptx_name, &ptx)?;
                    partitions[index].jit_link_add_elapsed = link_add_started.elapsed();
                    partitions[index].peak_rss_kib = process_peak_rss_kib();
                }
                Ok(())
            })?
        };
        validate_cubin_kernel_inventory(&cubin, &expected_kernel_entries)?;
        let bundle_file = bundle.finish()?;
        let ptx_bundle_durability_warning = pending_bundle.publish(bundle_file)?;
        Ok(MaterializedPartitionedOwner {
            partitions,
            nvvm_wall_elapsed,
            nvvm_peak_concurrency,
            ptxas_peak_concurrency,
            ptxas_peak_aggregate_rss_kib,
            ptx_bundle_path: ptx_bundle_path.to_path_buf(),
            ptx_bundle_durability_warning,
            cubin,
            link_elapsed,
            peak_rss_kib: process_peak_rss_kib(),
        })
    }

    /// Link ordered LTOIR modules to cubin or PTX.
    pub fn link_ltoir(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.linker.link_ltoir(inputs, options, output)
    }

    /// Link ordered PTX modules to a cubin.
    pub fn link_ptx(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.linker.link_ptx(inputs, options)
    }

    /// Compiler component, including exact libdevice bytes and provenance.
    pub fn compiler(&self) -> &NvvmCompiler {
        &self.compiler
    }

    /// Ordered LTOIR/PTX linker component.
    pub fn linker(&self) -> &LtoLinker {
        &self.linker
    }

    /// Exact discovered tool and libdevice digests.
    pub fn provenance(&self) -> ToolProvenance {
        ToolProvenance {
            libnvvm_sha256: self.compiler.libnvvm_digest(),
            nvjitlink_sha256: self.linker.nvjitlink_digest(),
            ptxas_sha256: self
                .assembler
                .as_ref()
                .and_then(PtxAssembler::digest_if_unchanged),
            libdevice_sha256: self.compiler.libdevice_digest(),
        }
    }

    /// Exact full-pipeline provenance, or `None` if a loaded DSO is unknown.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        let provenance = self.provenance();
        let ptxas_sha256 = match &self.assembler {
            Some(assembler) => Some(assembler.digest_if_unchanged()?),
            None => None,
        };
        Some(common_provenance_digest(
            &provenance.libnvvm_sha256?,
            &provenance.nvjitlink_sha256?,
            ptxas_sha256.as_ref(),
            &provenance.libdevice_sha256,
        ))
    }

    /// Digest the full NVVM IR to cubin recipe, including ordered options.
    pub fn nvvm_ir_artifact_digest(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Option<[u8; 32]> {
        let ptx_module_name = format!("{module_name}.ptx");
        nvvm_ir_artifact_digest_with_provenance(
            module_name,
            &ptx_module_name,
            nvvm_ir,
            options,
            self.provenance(),
        )
    }
}

struct PreparedPartition {
    ptx_name: String,
    ptx_path: PathBuf,
    object_path: PathBuf,
}

fn validate_cubin_kernel_inventory(
    cubin: &[u8],
    expected: &BTreeSet<String>,
) -> Result<(), FinalizerError> {
    if expected.is_empty() {
        return Err(FinalizerError::InvalidCubinKernelInventory {
            detail: "partitioned owner PTX exposes no public kernel entries".to_string(),
        });
    }
    let actual = validation::cubin_kernel_entries(cubin)
        .map_err(|detail| FinalizerError::InvalidCubinKernelInventory { detail })?;
    if &actual == expected {
        return Ok(());
    }
    const DIAGNOSTIC_SYMBOL_LIMIT: usize = 8;
    let missing = expected
        .difference(&actual)
        .take(DIAGNOSTIC_SYMBOL_LIMIT)
        .cloned()
        .collect();
    let unexpected = actual
        .difference(expected)
        .take(DIAGNOSTIC_SYMBOL_LIMIT)
        .cloned()
        .collect();
    Err(FinalizerError::CubinKernelInventoryMismatch {
        expected_count: expected.len(),
        actual_count: actual.len(),
        missing,
        unexpected,
    })
}

fn validate_compiled_ptx_size(
    name: &str,
    ptx: &[u8],
    maximum_bytes: u64,
) -> Result<(), FinalizerError> {
    let actual_bytes = u64::try_from(ptx.len()).unwrap_or(u64::MAX);
    if actual_bytes > maximum_bytes {
        Err(FinalizerError::CompiledPtxTooLarge {
            name: name.to_string(),
            actual_bytes,
            maximum_bytes,
        })
    } else {
        Ok(())
    }
}

fn read_partition_source_capped(
    path: &Path,
    maximum_bytes: u64,
) -> Result<Vec<u8>, FinalizerError> {
    let path_metadata = std::fs::symlink_metadata(path).map_err(|source| FinalizerError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !path_metadata.file_type().is_file() {
        return Err(FinalizerError::PartitionSourceNotRegular {
            path: path.to_path_buf(),
        });
    }

    let mut open_options = std::fs::OpenOptions::new();
    open_options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        open_options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = open_options
        .open(path)
        .map_err(|source| FinalizerError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let opened_metadata = file.metadata().map_err(|source| FinalizerError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !opened_metadata.file_type().is_file() {
        return Err(FinalizerError::PartitionSourceNotRegular {
            path: path.to_path_buf(),
        });
    }
    let actual_bytes = opened_metadata.len();
    if actual_bytes > maximum_bytes {
        return Err(FinalizerError::PartitionSourceTooLarge {
            path: path.to_path_buf(),
            actual_bytes,
            maximum_bytes,
        });
    }
    let source_len =
        usize::try_from(actual_bytes).map_err(|_| FinalizerError::PartitionSourceTooLarge {
            path: path.to_path_buf(),
            actual_bytes,
            maximum_bytes,
        })?;
    let mut source = vec![0; source_len];
    file.read_exact(&mut source)
        .map_err(|source| FinalizerError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    // Partition artifacts are immutable regular files. A byte beyond the
    // opened file's metadata means the input changed while it was consumed.
    // Reject it without growing the allocation.
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|source| FinalizerError::Io {
            path: path.to_path_buf(),
            source,
        })?
        != 0
    {
        return Err(FinalizerError::PartitionSourceChanged {
            path: path.to_path_buf(),
            initial_bytes: actual_bytes,
        });
    }
    Ok(source)
}

#[derive(Default)]
struct WeakStorageDefinitions {
    by_symbol: BTreeMap<String, ([u8; 32], String)>,
}

impl WeakStorageDefinitions {
    fn observe(&mut self, input_name: &str, ptx: &[u8]) -> Result<(), FinalizerError> {
        let ptx = std::str::from_utf8(ptx).map_err(|_| FinalizerError::InvalidPtxText {
            name: input_name.to_string(),
        })?;
        let mut pending = None::<String>;
        for raw_line in ptx.lines() {
            let line = raw_line
                .split_once("//")
                .map_or(raw_line, |(code, _)| code)
                .trim();
            if line.is_empty() {
                continue;
            }
            if let Some(declaration) = pending.as_mut() {
                declaration.push(' ');
                declaration.push_str(line);
                if declaration.contains(';') {
                    let complete = pending.take().expect("pending declaration exists");
                    self.observe_declaration(input_name, &complete)?;
                }
                continue;
            }
            if line.contains(".weak") && (line.contains(".global") || line.contains(".const")) {
                if line.contains(';') {
                    self.observe_declaration(input_name, line)?;
                } else {
                    pending = Some(line.to_string());
                }
            }
        }
        if let Some(declaration) = pending {
            return Err(FinalizerError::MalformedWeakStorageDefinition {
                name: input_name.to_string(),
                declaration,
            });
        }
        Ok(())
    }

    fn observe_declaration(
        &mut self,
        input_name: &str,
        declaration: &str,
    ) -> Result<(), FinalizerError> {
        let semicolon = declaration.find(';').ok_or_else(|| {
            FinalizerError::MalformedWeakStorageDefinition {
                name: input_name.to_string(),
                declaration: declaration.to_string(),
            }
        })?;
        let declaration = &declaration[..=semicolon];
        let left = declaration
            .split_once('=')
            .map_or(declaration.trim_end_matches(';'), |(left, _)| left);
        let symbol_token = left.split_ascii_whitespace().last().ok_or_else(|| {
            FinalizerError::MalformedWeakStorageDefinition {
                name: input_name.to_string(),
                declaration: declaration.to_string(),
            }
        })?;
        let symbol = symbol_token
            .split_once('[')
            .map_or(symbol_token, |(symbol, _)| symbol);
        if symbol.is_empty() || symbol.starts_with('.') {
            return Err(FinalizerError::MalformedWeakStorageDefinition {
                name: input_name.to_string(),
                declaration: declaration.to_string(),
            });
        }
        let mut definition_hasher = Sha256::new();
        for token in declaration.split_ascii_whitespace() {
            definition_hasher.update((token.len() as u64).to_le_bytes());
            definition_hasher.update(token.as_bytes());
        }
        let definition_digest = definition_hasher.finalize().into();
        if let Some((first_digest, first_input)) = self.by_symbol.get(symbol) {
            if first_digest != &definition_digest {
                return Err(FinalizerError::IncompatibleWeakStorageDefinition {
                    symbol: symbol.to_string(),
                    first_input: first_input.clone(),
                    second_input: input_name.to_string(),
                });
            }
        } else {
            self.by_symbol.insert(
                symbol.to_string(),
                (definition_digest, input_name.to_string()),
            );
        }
        Ok(())
    }
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_ascii_whitespace().nth(1)?.parse().ok()
}

static PTX_BUNDLE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct PendingPtxBundle {
    target: PathBuf,
    temporary: PathBuf,
    published: bool,
}

impl PendingPtxBundle {
    fn create(target: &Path) -> Result<(Self, std::fs::File), FinalizerError> {
        let parent = nonempty_parent(target);
        let file_name = target.file_name().ok_or_else(|| FinalizerError::Io {
            path: target.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "PTX bundle path has no file name",
            ),
        })?;
        for _ in 0..128 {
            let sequence = PTX_BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut temporary_name = file_name.to_os_string();
            temporary_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
            let temporary = parent.join(temporary_name);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
            {
                Ok(file) => {
                    return Ok((
                        Self {
                            target: target.to_path_buf(),
                            temporary,
                            published: false,
                        },
                        file,
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(FinalizerError::Io {
                        path: temporary,
                        source,
                    });
                }
            }
        }
        Err(FinalizerError::Io {
            path: target.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate a unique PTX bundle temporary file",
            ),
        })
    }

    fn publish(mut self, file: std::fs::File) -> Result<Option<String>, FinalizerError> {
        self.publish_with_directory_sync(file, |directory| directory.sync_all())
    }

    fn publish_with_directory_sync(
        &mut self,
        file: std::fs::File,
        sync_directory: impl FnOnce(&std::fs::File) -> std::io::Result<()>,
    ) -> Result<Option<String>, FinalizerError> {
        file.sync_all().map_err(|source| FinalizerError::Io {
            path: self.temporary.clone(),
            source,
        })?;
        drop(file);
        let parent = nonempty_parent(&self.target);
        let directory = std::fs::File::open(parent).map_err(|source| FinalizerError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        std::fs::rename(&self.temporary, &self.target).map_err(|source| FinalizerError::Io {
            path: self.target.clone(),
            source,
        })?;
        self.published = true;
        Ok(sync_directory(&directory).err().map(|error| {
            format!(
                "PTX bundle {} is live, but its parent directory {} could not be synced: {error}",
                self.target.display(),
                parent.display(),
            )
        }))
    }
}

fn nonempty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

impl Drop for PendingPtxBundle {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

/// Digest a complete finalization plan from already-established provenance.
///
/// This is useful to fingerprint Cargo work before executing the plan. It
/// returns `None` unless both loaded tool identities are exact.
pub fn nvvm_ir_artifact_digest_with_provenance(
    module_name: &str,
    ptx_module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    provenance: ToolProvenance,
) -> Option<[u8; 32]> {
    let compiler_digest = nvvm::nvvm_ir_ptx_artifact_digest_parts(
        module_name,
        nvvm_ir,
        options,
        &provenance.libdevice_sha256,
        &provenance.libnvvm_sha256?,
    );
    let linker_digest = link::ptx_artifact_digest_parts(
        &[NamedInput::new(ptx_module_name, &compiler_digest)],
        options,
        &provenance.nvjitlink_sha256?,
    );
    Some(
        provenance::StableDigest::new()
            .field("recipe", recipe_digest())
            .field("route", b"nvvm-ir-via-ptx-to-final-output")
            .field("compiler-plan", compiler_digest)
            .field("linker-plan", linker_digest)
            .finish(),
    )
}

/// Digest an ordered LTOIR link from an established exact linker identity.
pub fn ltoir_artifact_digest_with_provenance(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_sha256: &[u8; 32],
) -> [u8; 32] {
    link::ltoir_artifact_digest_parts(inputs, options, output, nvjitlink_sha256)
}

fn validate_name(name: &str) -> Result<(), FinalizerError> {
    if name.as_bytes().contains(&0) {
        Err(FinalizerError::InvalidInputName {
            name: name.to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    use std::io::Write as _;

    const LEGACY_NVVM_IR: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void ()* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

    fn exact_provenance() -> ToolProvenance {
        ToolProvenance {
            libnvvm_sha256: Some([1; 32]),
            nvjitlink_sha256: Some([2; 32]),
            ptxas_sha256: Some([4; 32]),
            libdevice_sha256: [3; 32],
        }
    }

    #[test]
    fn abandoned_bundle_publication_preserves_target_and_removes_temporary() {
        let sequence = PTX_BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-oxide-pending-bundle-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("owner.ptx.bundle");
        std::fs::write(&target, b"previous complete bundle").unwrap();

        let (pending, mut temporary_file) = PendingPtxBundle::create(&target).unwrap();
        let temporary_path = pending.temporary.clone();
        temporary_file.write_all(b"partial replacement").unwrap();
        drop(temporary_file);
        drop(pending);

        assert_eq!(std::fs::read(&target).unwrap(), b"previous complete bundle");
        assert!(!temporary_path.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn post_rename_directory_sync_failure_reports_live_nondurable_bundle() {
        let sequence = PTX_BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-oxide-bundle-durability-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("owner.ptx.bundle");
        std::fs::write(&target, b"previous complete bundle").unwrap();

        let (mut pending, mut temporary_file) = PendingPtxBundle::create(&target).unwrap();
        let temporary_path = pending.temporary.clone();
        temporary_file.write_all(b"new complete bundle").unwrap();
        let warning = pending
            .publish_with_directory_sync(temporary_file, |_| {
                Err(std::io::Error::other("injected directory sync failure"))
            })
            .unwrap()
            .expect("post-publication durability failure must remain observable");

        assert!(pending.published);
        assert_eq!(std::fs::read(&target).unwrap(), b"new complete bundle");
        assert!(!temporary_path.exists());
        assert!(warning.contains("is live") && warning.contains("injected directory sync failure"));

        drop(pending);
        assert_eq!(std::fs::read(&target).unwrap(), b"new complete bundle");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn weak_storage_odr_validation_accepts_identical_and_rejects_mismatch() {
        let mut definitions = WeakStorageDefinitions::default();
        definitions
            .observe(
                "part-a.ptx",
                b".weak .global .align 4 .u32 shared_mut = 11;\n\
                  .weak .const .align 8 .b8 LOOKUP[2] = {1, 2};\n",
            )
            .unwrap();
        definitions
            .observe(
                "part-b.ptx",
                b"// .weak helper comment\n\
                  .weak   .global .align 4 .u32 shared_mut = 11;\n\
                  .weak .const .align 8 .b8 LOOKUP[2]\n = {1, 2};\n",
            )
            .unwrap();

        let error = definitions
            .observe(
                "part-c.ptx",
                b".weak .global .align 4 .u32 shared_mut = 12;\n",
            )
            .unwrap_err();
        assert!(matches!(
            error,
            FinalizerError::IncompatibleWeakStorageDefinition {
                symbol,
                first_input,
                second_input,
            } if symbol == "shared_mut"
                && first_input == "part-a.ptx"
                && second_input == "part-c.ptx"
        ));
    }

    #[test]
    fn weak_storage_odr_validation_fails_closed_on_truncation_and_non_text() {
        let mut definitions = WeakStorageDefinitions::default();
        assert!(matches!(
            definitions
                .observe("truncated.ptx", b".weak .global .align 4 .u32 shared = 1")
                .unwrap_err(),
            FinalizerError::MalformedWeakStorageDefinition { .. }
        ));
        assert!(matches!(
            definitions
                .observe("binary.ptx", &[0xff, 0xfe])
                .unwrap_err(),
            FinalizerError::InvalidPtxText { .. }
        ));
    }

    #[test]
    fn partition_source_reader_accepts_exact_cap_and_rejects_cap_plus_one() {
        let sequence = PTX_BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-oxide-partition-source-cap-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let exact_cap = directory.join("exact.ll");
        let cap_plus_one = directory.join("oversize.ll");
        std::fs::write(&exact_cap, b"12345678").unwrap();
        std::fs::write(&cap_plus_one, b"123456789").unwrap();

        assert_eq!(
            read_partition_source_capped(&exact_cap, 8).unwrap(),
            b"12345678"
        );
        assert!(matches!(
            read_partition_source_capped(&cap_plus_one, 8).unwrap_err(),
            FinalizerError::PartitionSourceTooLarge {
                actual_bytes: 9,
                maximum_bytes: 8,
                ..
            }
        ));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn partition_source_reader_rejects_symlinks_and_fifos() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt as _;
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let regular = directory.path().join("regular.ll");
        let symlink_path = directory.path().join("symlink.ll");
        let fifo_path = directory.path().join("partition.fifo");
        let directory_path = directory.path().join("partition-directory");
        std::fs::write(&regular, b"source").unwrap();
        symlink(&regular, &symlink_path).unwrap();
        std::fs::create_dir(&directory_path).unwrap();
        let fifo_path_c = CString::new(fifo_path.as_os_str().as_bytes()).unwrap();
        // SAFETY: `fifo_path_c` is a live, NUL-terminated pathname.
        let result = unsafe { libc::mkfifo(fifo_path_c.as_ptr(), 0o600) };
        assert_eq!(
            result,
            0,
            "mkfifo failed: {}",
            std::io::Error::last_os_error()
        );

        for path in [&symlink_path, &fifo_path, &directory_path] {
            assert!(matches!(
                read_partition_source_capped(path, 64).unwrap_err(),
                FinalizerError::PartitionSourceNotRegular { path: rejected }
                    if rejected == *path
            ));
        }
    }

    #[test]
    fn compiled_ptx_ceiling_accepts_exact_cap_and_rejects_cap_plus_one() {
        validate_compiled_ptx_size("exact.ptx", b"12345678", 8).unwrap();
        assert!(matches!(
            validate_compiled_ptx_size("oversize.ptx", b"123456789", 8).unwrap_err(),
            FinalizerError::CompiledPtxTooLarge {
                name,
                actual_bytes: 9,
                maximum_bytes: 8,
            } if name == "oversize.ptx"
        ));
    }

    #[test]
    fn nvjitlink_object_style_empty_elf_cannot_satisfy_a_ptx_kernel_plan() {
        // nvJitLink's host `Object` input accepts a CUDA relocatable but can
        // return a structurally valid ELF with no device entry points.
        let cubin = validation::empty_inventory_cubin_fixture();
        assert!(is_valid_cubin(&cubin));
        let expected = BTreeSet::from(["expected_kernel".to_string()]);
        assert!(matches!(
            validate_cubin_kernel_inventory(&cubin, &expected).unwrap_err(),
            FinalizerError::CubinKernelInventoryMismatch {
                expected_count: 1,
                actual_count: 0,
                missing,
                unexpected,
            } if missing == ["expected_kernel"] && unexpected.is_empty()
        ));
    }

    #[test]
    fn empty_ptx_kernel_plan_is_rejected_before_cubin_inventory_can_match() {
        let cubin = validation::empty_inventory_cubin_fixture();
        let error = validate_cubin_kernel_inventory(&cubin, &BTreeSet::new()).unwrap_err();
        assert!(matches!(
            error,
            FinalizerError::InvalidCubinKernelInventory { detail }
                if detail.contains("no public kernel entries")
        ));
    }

    #[test]
    fn materialization_digest_tracks_the_whole_ptx_plan() {
        let options = FinalizationOptions::new("sm_86".parse().unwrap());
        let baseline = nvvm_ir_artifact_digest_with_provenance(
            "kernel.ll",
            "kernel.ll.ptx",
            b"nvvm ir",
            &options,
            exact_provenance(),
        )
        .unwrap();

        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_with_provenance(
                "kernel.ll",
                "renamed.ptx",
                b"nvvm ir",
                &options,
                exact_provenance(),
            )
            .unwrap()
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_with_provenance(
                "kernel.ll",
                "kernel.ll.ptx",
                b"nvvm ir",
                &options.clone().with_fma_contraction(false),
                exact_provenance(),
            )
            .unwrap()
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_with_provenance(
                "kernel.ll",
                "kernel.ll.ptx",
                b"nvvm ir",
                &options.clone().with_debug_policy(DebugPolicy::LineTables),
                exact_provenance(),
            )
            .unwrap()
        );
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM, ptxas, nvJitLink, and libdevice"]
    fn sequential_and_parallel_partition_materialization_are_byte_identical() {
        let directory = tempfile::tempdir().unwrap();
        let stored_inputs = (0..8)
            .map(|index| {
                let name = format!("equivalence-{index:04}.ll");
                let path = directory.path().join(&name);
                let kernel = format!("equivalence_kernel_{index}");
                std::fs::write(&path, named_legacy_nvvm_module(&kernel)).unwrap();
                (name, path)
            })
            .collect::<Vec<_>>();
        let inputs = stored_inputs
            .iter()
            .map(|(name, path)| PartitionFileInput::new(name, path, PartitionFileInputKind::NvvmIr))
            .collect::<Vec<_>>();
        let options = FinalizationOptions::new("sm_86".parse().unwrap());
        let finalizer = Finalizer::discover().unwrap();
        let sequential_bundle = directory.path().join("sequential.ptx.bundle");
        let parallel_bundle = directory.path().join("parallel.ptx.bundle");

        let sequential = finalizer
            .materialize_partition_files_with_nvvm_workers(&inputs, &sequential_bundle, &options, 1)
            .unwrap();
        let parallel = finalizer
            .materialize_partition_files_with_nvvm_workers(&inputs, &parallel_bundle, &options, 4)
            .unwrap();

        assert_eq!(sequential.nvvm_peak_concurrency, 1);
        let expected_parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(4)
            .min(inputs.len());
        assert!((1..=expected_parallelism).contains(&parallel.nvvm_peak_concurrency));
        assert!(sequential.nvvm_wall_elapsed > Duration::ZERO);
        assert!(parallel.nvvm_wall_elapsed > Duration::ZERO);
        assert_eq!(
            partition_identities(&parallel),
            partition_identities(&sequential),
        );
        assert_eq!(
            std::fs::read(&parallel_bundle).unwrap(),
            std::fs::read(&sequential_bundle).unwrap(),
        );
        assert_eq!(parallel.cubin, sequential.cubin);
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM, ptxas, nvJitLink, and libdevice"]
    fn failed_parallel_partition_batch_preserves_bundle_and_cleans_private_files() {
        let directory = tempfile::tempdir().unwrap();
        let stored_inputs = (0..4)
            .map(|index| {
                let name = format!("failure-{index:04}.ll");
                let path = directory.path().join(&name);
                let source = if index == 1 || index == 3 {
                    b"not valid NVVM IR".to_vec()
                } else {
                    named_legacy_nvvm_module(&format!("failure_kernel_{index}"))
                };
                std::fs::write(&path, source).unwrap();
                (name, path)
            })
            .collect::<Vec<_>>();
        let inputs = stored_inputs
            .iter()
            .map(|(name, path)| PartitionFileInput::new(name, path, PartitionFileInputKind::NvvmIr))
            .collect::<Vec<_>>();
        let bundle = directory.path().join("owner.ptx.bundle");
        std::fs::write(&bundle, b"previous complete bundle").unwrap();
        let options = FinalizationOptions::new("sm_86".parse().unwrap());
        let finalizer = Finalizer::discover().unwrap();

        let error = finalizer
            .materialize_partition_files_with_nvvm_workers(&inputs, &bundle, &options, 4)
            .expect_err("invalid NVVM IR must fail the parallel batch");

        assert!(matches!(error, FinalizerError::Nvvm(_)));
        assert_eq!(std::fs::read(&bundle).unwrap(), b"previous complete bundle",);
        for entry in std::fs::read_dir(directory.path()).unwrap() {
            let name = entry.unwrap().file_name();
            let name = name.to_string_lossy();
            assert!(!name.starts_with(".cuda-oxide-finalizer-"));
            assert!(!name.starts_with("owner.ptx.bundle.tmp-"));
        }
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM, nvJitLink, and libdevice"]
    fn live_pipeline_accepts_toolkit_cubins_and_emits_ptx_for_both_fma_policies() {
        let finalizer = Finalizer::discover().unwrap();
        assert!(finalizer.provenance_digest().is_some());
        let target: CudaArch = "sm_86".parse().unwrap();

        for (allow_fma, debug) in [
            (false, DebugPolicy::None),
            (false, DebugPolicy::LineTables),
            (true, DebugPolicy::Full),
        ] {
            let options = FinalizationOptions::new(target.clone())
                .with_fma_contraction(allow_fma)
                .with_debug_policy(debug);
            let ltoir = finalizer
                .compiler()
                .compile_nvvm_ir_to_ltoir("kernel.ll", LEGACY_NVVM_IR, &options)
                .unwrap();
            assert!(!ltoir.is_empty());
            let linkable_ptx = finalizer
                .compiler()
                .compile_nvvm_ir_to_ptx("kernel.ll", LEGACY_NVVM_IR, &options)
                .unwrap();
            assert!(
                linkable_ptx
                    .windows(b".version".len())
                    .any(|part| part == b".version")
            );
            let ptx_input = [NamedInput::new("kernel.ll.ptx", &linkable_ptx)];
            let whole_ptx_cubin = finalizer.link_ptx(&ptx_input, &options).unwrap();
            assert!(is_valid_cubin(&whole_ptx_cubin));
            let materialized = finalizer
                .materialize_nvvm_ir_with_ptx("kernel.ll", LEGACY_NVVM_IR, &options)
                .unwrap();
            assert_eq!(materialized.ptx_input, linkable_ptx);
            assert_eq!(materialized.cubin, whole_ptx_cubin);
            assert_eq!(
                finalizer
                    .materialize_nvvm_ir("kernel.ll", LEGACY_NVVM_IR, &options)
                    .unwrap(),
                materialized.cubin
            );
            let input = [NamedInput::new("kernel.ltoir", &ltoir)];
            let cubin = finalizer
                .link_ltoir(&input, &options, FinalizerOutput::Cubin)
                .unwrap();
            assert!(is_valid_cubin(&cubin));
            let ptx = finalizer
                .link_ltoir(&input, &options, FinalizerOutput::Ptx)
                .unwrap();
            assert!(
                ptx.windows(b".version".len())
                    .any(|part| part == b".version")
            );
        }
    }

    fn named_legacy_nvvm_module(kernel: &str) -> Vec<u8> {
        std::str::from_utf8(LEGACY_NVVM_IR)
            .unwrap()
            .replace("@kernel", &format!("@{kernel}"))
            .into_bytes()
    }

    fn partition_identities(
        materialized: &MaterializedPartitionedOwner,
    ) -> Vec<(&str, usize, usize, [u8; 32])> {
        materialized
            .partitions
            .iter()
            .map(|partition| {
                (
                    partition.name.as_str(),
                    partition.source_bytes,
                    partition.ptx_bytes,
                    partition.ptx_sha256,
                )
            })
            .collect()
    }
}
