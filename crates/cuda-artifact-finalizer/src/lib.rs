/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-independent CUDA artifact finalization.
//!
//! This crate is the single owner of cuda-oxide's libNVVM and nvJitLink
//! compilation policy. It deliberately does not link the CUDA Driver. Both
//! build-time materialization and runtime fallback use the same typed target,
//! FMA, debug, input-order, validation, and provenance rules.

mod link;
mod nvvm;
mod options;
mod provenance;
mod validation;

pub use libnvvm_sys::{CudaArch, CudaArchParseError, LibdeviceNotFound, NvvmError, find_libdevice};
pub use link::LtoLinker;
pub use nvjitlink_sys::NvJitLinkError;
pub use nvvm::NvvmCompiler;
pub use options::{DebugPolicy, FinalizationOptions, FinalizerOutput, NamedInput};
pub use provenance::{ToolProvenance, recipe_digest};
pub use validation::is_valid_cubin;

use provenance::common_provenance_digest;
use std::path::PathBuf;
use thiserror::Error;

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

    /// nvJitLink was invoked without an input module.
    #[error("at least one ordered linker input is required")]
    NoLinkInputs,

    /// nvJitLink returned bytes that are not a complete CUDA ELF image.
    #[error("nvJitLink returned an invalid or truncated cubin")]
    InvalidCubin,

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

/// Complete NVVM IR to cubin/PTX finalizer.
#[derive(Clone)]
pub struct Finalizer {
    compiler: NvvmCompiler,
    linker: LtoLinker,
}

impl Finalizer {
    /// Discover libNVVM, libdevice, and nvJitLink without loading the Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            compiler: NvvmCompiler::discover()?,
            linker: LtoLinker::discover()?,
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
            libdevice_sha256: self.compiler.libdevice_digest(),
        }
    }

    /// Exact full-pipeline provenance, or `None` if a loaded DSO is unknown.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        let provenance = self.provenance();
        Some(common_provenance_digest(
            &provenance.libnvvm_sha256?,
            &provenance.nvjitlink_sha256?,
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
            libdevice_sha256: [3; 32],
        }
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
}
