/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # rustc_codegen_cuda: Unified Host/Device Compilation Backend
//!
//! A custom rustc codegen backend that enables single-source CUDA compilation for Rust,
//! similar to NVIDIA's nvc++ compiler for C++. This backend intercepts rustc's code
//! generation phase to extract device code and compile it to PTX while delegating
//! host code compilation to the standard LLVM backend.
//!
//! ## Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                              RUSTC COMPILATION                                  │
//! │                                                                                 │
//! │   Source Code (.rs)                                                             │
//! │         │                                                                       │
//! │         ▼                                                                       │
//! │   ┌───────────────────────────────────────────────────────────────────────┐     │
//! │   │                         RUSTC FRONTEND                                │     │
//! │   │                                                                       │     │
//! │   │   Parsing ──▶ HIR ──▶ Type Check ──▶ MIR Generation ──▶ MIR Passes    │     │
//! │   │                                                                       │     │
//! │   │   Outputs: Fully monomorphized, OPTIMIZED MIR                         │     │
//! │   │            (affected by -C opt-level, -Z mir-enable-passes)           │     │
//! │   └───────────────────────────────────────────────────────────────────────┘     │
//! │         │                                                                       │
//! │         │  MIR passes have ALREADY run by this point                            │
//! │         │  (including JumpThreading unless disabled)                            │
//! │         ▼                                                                       │
//! │   ┌───────────────────────────────────────────────────────────────────────┐     │
//! │   │                    rustc_codegen_cuda (THIS BACKEND)                  │     │
//! │   │                                                                       │     │
//! │   │   Entry: codegen_crate(TyCtxt) called by rustc                        │     │
//! │   │                                                                       │     │
//! │   │   ┌─────────────────────────────────────────────────────────────┐     │     │
//! │   │   │  1. KERNEL DETECTION                                        │     │     │
//! │   │   │     - Scan CGUs for functions in the reserved namespace     │     │     │
//! │   │   │       `cuda_oxide_kernel_<hash>_*` (set by #[kernel] macro) │     │     │
//! │   │   └─────────────────────────────────────────────────────────────┘     │     │
//! │   │                          │                                            │     │
//! │   │                          ▼                                            │     │
//! │   │   ┌─────────────────────────────────────────────────────────────┐     │     │
//! │   │   │  2. DEVICE FUNCTION COLLECTION (collector.rs)               │     │     │
//! │   │   │     - Start from kernel entry points                        │     │     │
//! │   │   │     - Walk MIR call graph transitively                      │     │     │
//! │   │   │     - Collect all reachable functions from:                 │     │     │
//! │   │   │       • Local crate                                         │     │     │
//! │   │   │       • cuda_device (intrinsics)                            │     │     │
//! │   │   │       • core (iterators, Option, etc.)                      │     │     │
//! │   │   │     - Filter out: fmt::*, panicking::*, intrinsic stubs     │     │     │
//! │   │   └─────────────────────────────────────────────────────────────┘     │     │
//! │   │                          │                                            │     │
//! │   │          ┌───────────────┴───────────────┐                            │     │
//! │   │          ▼                               ▼                            │     │
//! │   │   ┌─────────────────┐           ┌─────────────────────────┐           │     │
//! │   │   │  DEVICE PATH    │           │      HOST PATH          │           │     │
//! │   │   │                 │           │                         │           │     │
//! │   │   │  3. Bridge to   │           │  4. Delegate to         │           │     │
//! │   │   │     stable_mir  │           │     rustc_codegen_llvm  │           │     │
//! │   │   │                 │           │                         │           │     │
//! │   │   │  device_codegen │           │  Standard LLVM backend  │           │     │
//! │   │   │  .rs handles    │           │  handles all host code  │           │     │
//! │   │   └────────┬────────┘           └────────────┬────────────┘           │     │
//! │   │            │                                 │                        │     │
//! │   │            ▼                                 ▼                        │     │
//! │   │   ┌─────────────────┐           ┌─────────────────────────┐           │     │
//! │   │   │ cuda-oxide      │           │  Host Object Files      │           │     │
//! │   │   │ Pipeline:       │           │  (.o / .rlib)           │           │     │
//! │   │   │                 │           │                         │           │     │
//! │   │   │ dialect-mir     │           │  Standard x86_64 code   │           │     │
//! │   │   │     ▼ (mem2reg) │           │                         │           │     │
//! │   │   │ LLVM dialect    │           │                         │           │     │
//! │   │   │     ▼           │           │                         │           │     │
//! │   │   │ LLVM IR (.ll)   │           │                         │           │     │
//! │   │   │     ▼ (llc)     │           │                         │           │     │
//! │   │   │ PTX (.ptx)      │           │                         │           │     │
//! │   │   └─────────────────┘           └─────────────────────────┘           │     │
//! │   │                                                                       │     │
//! │   └───────────────────────────────────────────────────────────────────────┘     │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## How MIR is Obtained
//!
//! When `codegen_crate()` is called, rustc has ALREADY:
//!
//! 1. **Parsed** the source code
//! 2. **Type checked** everything
//! 3. **Generated MIR** for all functions
//! 4. **Run MIR optimization passes** based on `-C opt-level` and `-Z mir-enable-passes`
//!
//! We receive a `TyCtxt` containing **optimized MIR**. The MIR we get depends entirely
//! on what flags were passed to rustc:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                           MIR OPTIMIZATION PASSES                               │
//! │                                                                                 │
//! │   User runs:  rustc -C opt-level=3 -Z mir-enable-passes=-JumpThreading ...      │
//! │                         │                        │                              │
//! │                         ▼                        ▼                              │
//! │              ┌──────────────────┐    ┌──────────────────────────┐               │
//! │              │ Enable passes:   │    │ Disable passes:          │               │
//! │              │ - Inlining       │    │ - JumpThreading (MUST!)  │               │
//! │              │ - ConstProp      │    │                          │               │
//! │              │ - GVN            │    │                          │               │
//! │              │ - DeadCode       │    │                          │               │
//! │              │ - etc.           │    │                          │               │
//! │              └──────────────────┘    └──────────────────────────┘               │
//! │                                                                                 │
//! │   Result: We get MIR that has been through these passes                         │
//! │           This affects BOTH host and device code (same MIR for both)            │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Required Compiler Flags
//!
//! The following flags MUST be passed to rustc when using this backend:
//!
//! | Flag                                  | Purpose                | Why Required                                                                                               |
//! |---------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------|
//! | `-Z mir-enable-passes=-JumpThreading` | Disable JumpThreading  | **CRITICAL**: JumpThreading duplicates barrier calls into branches, breaking GPU synchronization semantics |
//! | `-Z always-encode-mir`                | Encode cross-crate MIR | Device codegen is whole-program: without it, a dependency function that is neither `#[inline]` nor generic (canonically a recursive one) is *called* by the emitted IR but never *defined*, failing with "Symbol not found" |
//!
//! Recommended for production:
//!
//! | Flag                       | Purpose                  | Why Recommended                                              |
//! |----------------------------|--------------------------|--------------------------------------------------------------|
//! | `-C opt-level=3`           | Maximum MIR optimization | Better inlining, smaller device code                         |
//! | `-C debug-assertions=off`  | Remove debug checks      | `debug_assert!` pulls in fmt code that can't compile for GPU |
//!
//! **Note:** `panic=abort` is **NOT required**. The codegen backend treats all unwind
//! paths as unreachable since the CUDA toolchain does not support unwinding today. This means standard library code
//! compiled without `panic=abort` works fine -- unwind edges are simply ignored.
//!
//! ### Why JumpThreading Must Be Disabled
//!
//! JumpThreading is a MIR optimization that duplicates code to eliminate jumps.
//! This is problematic for GPU code because it can duplicate barrier calls:
//!
//! ```text
//! BEFORE JumpThreading:              AFTER JumpThreading (BROKEN!):
//! ┌─────────────────────────┐        ┌─────────────────────────────────────┐
//! │ bb0:                    │        │ bb0:                                │
//! │   if cond -> bb1, bb2   │        │   if cond -> bb1, bb2               │
//! │                         │        │                                     │
//! │ bb1:                    │        │ bb1:                                │
//! │   a()                   │        │   a()                               │
//! │   goto bb3              │        │   __syncthreads()  ◄─── Thread 0-15 │
//! │                         │        │   c()                               │
//! │ bb2:                    │        │   return                            │
//! │   goto bb3              │        │                                     │
//! │                         │        │ bb2:                                │
//! │ bb3:                    │        │   __syncthreads()  ◄─── Thread 16-31│
//! │   __syncthreads()       │        │   c()                               │
//! │   c()                   │        │   return                            │
//! │   return                │        │                                     │
//! └─────────────────────────┘        └─────────────────────────────────────┘
//!
//! Different threads execute DIFFERENT barrier instances = DEADLOCK!
//! ```
//!
//! ## `no_std` Requirement
//!
//! Kernel crates MUST use `#![no_std]`. The collector enforces this with a
//! single hard rule: **the `std` crate itself is forbidden, every other
//! crate is allowed** (provided it's reachable from a kernel and itself
//! avoids `std`). The check is on the *originating crate*
//! (`tcx.crate_name(def_id.krate)`), not on display paths -- which matters,
//! because rustc's MIR pretty-printer routinely emits `std::*` for items
//! that are merely re-exported from `core`.
//!
//! See [`collector::should_collect_from_crate`] for the exact policy.
//!
//! ### Why `std::*` shows up in MIR dumps (and isn't a problem)
//!
//! Run `cargo oxide pipeline vecadd` (or `atomics`, or most other examples)
//! and the rustc MIR section will be peppered with paths like:
//!
//! ```text
//! _4 = std::option::Option::<&mut f32>::Some(copy _21)
//! _4 = const std::option::Option::<&mut f32>::None
//! _3 = copy _14 as *const std::sync::atomic::Atomic<u32> (PtrToPtr)
//! _4 = std::intrinsics::atomic_xadd::<u32, u32, ...>(move _15, ...) -> ...
//! ```
//!
//! These are **`core` items shown under their `std::` re-export path**.
//! `def_path_str` chooses the most user-visible path, which is usually the
//! `std::*` form. The actual `DefId` lives in `core` (or `core::sync::atomic`,
//! `core::intrinsics`, ...), so the collector's
//! `crate_name(def_id.krate) == "std"` check is `false` and they're collected
//! normally. Treat `std::*` in MIR output as cosmetic; only a hard collector
//! error means actual `std` was reached.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                        CRATE FILTERING FOR DEVICE CODE                          │
//! │                                                                                 │
//! │   Allowed (originating crate, i.e. DefId.krate):                                │
//! │   ┌──────────────────────────────────────────────────────────────────────┐      │
//! │   │ local crate (your kernel code)                                       │      │
//! │   │ cuda_device  (GPU intrinsics)                                        │      │
//! │   │ core         (Option, Result, UnsafeCell, sync::atomic, intrinsics)  │      │
//! │   │ alloc        (Vec / Box, only if you wired up a GPU allocator)       │      │
//! │   │ any other no_std crate, if transitively reachable from a kernel      │      │
//! │   │   (libm, num-traits, your own helper crates, ...)                    │      │
//! │   └──────────────────────────────────────────────────────────────────────┘      │
//! │                                                                                 │
//! │   Forbidden (hard error at collection time):                                    │
//! │   ┌──────────────────────────────────────────────────────────────────────┐      │
//! │   │ std -- only when the *originating* crate is std (not just a display  │      │
//! │   │        re-export). Example: an actual call into std::thread,         │      │
//! │   │        std::fs, std::io, std::sync::Mutex, etc.                      │      │
//! │   └──────────────────────────────────────────────────────────────────────┘      │
//! │                                                                                 │
//! │   When that genuine std call is reached, the collector emits a                  │
//! │   CollectDecision::Forbidden, and process_call_operand aborts compilation       │
//! │   with a formatted error box naming the function -- no silent skip, no          │
//! │   cryptic PTX "undefined symbol" later in the pipeline.                         │
//! │                                                                                 │
//! │   Intentionally skipped (no error, just dropped): `core::fmt::*`,               │
//! │   `core::panicking::*`, and `*::precondition_check`. These are reached          │
//! │   by panic/UB-check paths that can't actually fire at runtime under             │
//! │   panic=abort + `-C debug-assertions=off`.                                      │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Compilation Model
//!
//! **Unified single-source** compilation is fully supported. Device code is marked
//! with `#[kernel]` and the backend automatically splits based on kernel reachability
//! -- no `#[cfg(cuda_device)]` needed.
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, DisjointSlice};
//! use cuda_host::cuda_launch;
//!
//! #[kernel]
//! pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
//!     let idx = thread::index_1d();
//!     if let Some(c_elem) = c.get_mut(idx) {
//!         *c_elem = a[idx.get()] + b[idx.get()];
//!     }
//! }
//!
//! fn main() {
//!     // Host code -- compiled to native x86_64 by LLVM
//!     // Kernel is compiled to PTX by cuda-oxide
//! }
//! ```
//!
//! See `examples/` for working examples.
//!
//! ## Example Usage
//!
//! ```bash
//! # Build the backend
//! cd crates/rustc-codegen-cuda
//! cargo build
//!
//! # Compile a kernel crate with the backend
//! CUDA_OXIDE_VERBOSE=1 rustc \
//!     --edition 2021 \
//!     -C opt-level=3 \
//!     -C debug-assertions=off \
//!     -Z mir-enable-passes=-JumpThreading \
//!     -Z always-encode-mir \
//!     -Z codegen-backend=./target/debug/librustc_codegen_cuda.so \
//!     my_kernel.rs
//! ```
//!
//! ## Environment Variables
//!
//! | Variable                          | Effect                               |
//! |-----------------------------------|--------------------------------------|
//! | `CUDA_OXIDE_VERBOSE`              | Print detailed compilation progress  |
//! | `CUDA_OXIDE_DUMP_MIR`             | Dump the `dialect-mir` module        |
//! | `CUDA_OXIDE_DUMP_LLVM`            | Dump the LLVM dialect module         |
//! | `CUDA_OXIDE_PTX_DIR`              | Override PTX output directory        |
//! | `CUDA_OXIDE_TARGET`               | Override GPU target (e.g., `sm_90a`) |
//! | `CUDA_OXIDE_DEVICE_CODEGEN_CRATE` | Filter device owner crate names      |
//!
//! ## Module Structure
//!
//! - [`collector`]: Device function collection via MIR call graph traversal
//! - [`device_codegen`]: Bridge to the cuda-oxide pipeline (MIR → PTX)
//! - [`layout`]: Unified type layouts for host/device ABI compatibility

#![feature(rustc_private)]
#![allow(unused_imports)]
#![allow(dead_code)]

// Import rustc internal crates
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

// rustc_public (stable MIR) and its bridge - for calling mir-importer
extern crate rustc_public;
extern crate rustc_public_bridge;

// The standard LLVM backend - we delegate host codegen to this
extern crate rustc_codegen_llvm;

mod collector;
mod device_codegen;
mod generated_intrinsics;
mod materialize;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleKind};
use rustc_data_structures::fx::FxIndexMap;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::Session;
use rustc_session::config::OutputFilenames;
use std::any::Any;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The CUDA codegen backend.
///
/// This backend wraps `rustc_codegen_llvm` for host code while adding
/// device code compilation via cuda-oxide. It implements the [`CodegenBackend`]
/// trait which rustc uses to delegate code generation.
///
/// ## Delegation Strategy
///
/// Rather than reimplementing all of LLVM codegen, we:
/// 1. Intercept `codegen_crate()` to extract and compile device code
/// 2. Delegate ALL other methods to `rustc_codegen_llvm`
///
/// This means host code gets the full, battle-tested LLVM backend while
/// device code goes through our specialized cuda-oxide pipeline.
pub struct CudaCodegenBackend {
    config: CudaCodegenConfig,
    /// The underlying LLVM backend for host code generation
    llvm_backend: Box<dyn CodegenBackend>,
}

struct CudaOngoingCodegen {
    host: Box<dyn Any>,
    artifact_objects: Vec<ArtifactObject>,
}

struct ArtifactObject {
    path: PathBuf,
    retain_in_host_cgus: bool,
}

static ARTIFACT_OBJECT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configuration for the CUDA codegen backend.
///
/// All configuration is read from environment variables at backend load time.
/// This avoids the need to thread configuration through rustc's argument parsing.
#[derive(Clone, Default)]
pub struct CudaCodegenConfig {
    /// Print detailed compilation progress to stderr.
    pub verbose: bool,
    /// Dump raw rustc MIR before translation (requires --verbose flag).
    pub dump_rustc_mir: bool,
    /// Dump the `dialect-mir` module during device compilation.
    pub dump_mir_dialect: bool,
    /// Dump the LLVM dialect module during device compilation.
    pub dump_llvm_dialect: bool,
    /// Override PTX output directory (defaults to current directory).
    pub ptx_output_dir: Option<std::path::PathBuf>,
    /// When set, emit device code only for these normalized local crate names.
    /// Host code still goes through the wrapped LLVM backend for every crate.
    pub device_codegen_crates: Option<BTreeSet<String>>,
}

impl CudaCodegenConfig {
    /// Load configuration from environment variables.
    ///
    /// | Variable                             | Config Field             |
    /// |--------------------------------------|--------------------------|
    /// | `CUDA_OXIDE_VERBOSE`                 | `verbose`                |
    /// | `CUDA_OXIDE_SHOW_RUSTC_MIR`          | `dump_rustc_mir`         |
    /// | `CUDA_OXIDE_DUMP_MIR`                | `dump_mir_dialect`       |
    /// | `CUDA_OXIDE_DUMP_LLVM`               | `dump_llvm_dialect`      |
    /// | `CUDA_OXIDE_PTX_DIR`                 | `ptx_output_dir`         |
    /// | `CUDA_OXIDE_DEVICE_CODEGEN_CRATE`  | `device_codegen_crates`  |
    pub fn from_env() -> Self {
        Self {
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            dump_rustc_mir: std::env::var("CUDA_OXIDE_SHOW_RUSTC_MIR").is_ok(),
            dump_mir_dialect: std::env::var("CUDA_OXIDE_DUMP_MIR").is_ok(),
            dump_llvm_dialect: std::env::var("CUDA_OXIDE_DUMP_LLVM").is_ok(),
            ptx_output_dir: std::env::var("CUDA_OXIDE_PTX_DIR")
                .ok()
                .map(std::path::PathBuf::from),
            device_codegen_crates: parse_device_codegen_crates(
                std::env::var(reserved_oxide_symbols::DEVICE_CODEGEN_CRATE_ENV)
                    .ok()
                    .as_deref(),
            ),
        }
    }

    fn allows_device_codegen_for(&self, crate_name: &str) -> bool {
        self.device_codegen_crates
            .as_ref()
            .is_none_or(|owners| owners.contains(&normalize_device_crate_name(crate_name)))
    }
}

fn normalize_device_crate_name(name: &str) -> String {
    name.trim().replace('-', "_")
}

fn parse_device_codegen_crates(raw: Option<&str>) -> Option<BTreeSet<String>> {
    raw.and_then(|raw| {
        let owners: BTreeSet<_> = raw
            .split(',')
            .map(normalize_device_crate_name)
            .filter(|name| !name.is_empty())
            .collect();
        (!owners.is_empty()).then_some(owners)
    })
}

fn should_codegen_device_crate(
    config: &CudaCodegenConfig,
    crate_name: &str,
    contains_device_code: bool,
) -> bool {
    contains_device_code && config.allows_device_codegen_for(crate_name)
}

fn reject_unsupported_codegen_protocol(
    scoped_fingerprint_present: bool,
    unsupported_root_present: bool,
) -> bool {
    scoped_fingerprint_present && unsupported_root_present
}

impl CodegenBackend for CudaCodegenBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn init(&self, sess: &Session) {
        // Note: Don't log here - init() is called for ALL crates including dependencies.
        // We log in codegen_crate() only when there are kernels to compile.

        // Initialize the underlying LLVM backend
        self.llvm_backend.init(sess);
    }

    fn print_version(&self) {
        println!(
            "rustc_codegen_cuda version {} (wrapping rustc_codegen_llvm)",
            env!("CARGO_PKG_VERSION")
        );
        self.llvm_backend.print_version();
    }

    fn target_cpu(&self, sess: &Session) -> String {
        self.llvm_backend.target_cpu(sess)
    }

    fn target_config(&self, sess: &Session) -> rustc_codegen_ssa::TargetConfig {
        self.llvm_backend.target_config(sess)
    }

    fn provide(&self, providers: &mut rustc_middle::util::Providers) {
        // Delegate to LLVM backend
        self.llvm_backend.provide(providers);
    }

    /// Main codegen entry point - this is where device/host splitting happens.
    ///
    /// ## Execution Flow
    ///
    /// ```text
    /// codegen_crate(TyCtxt)
    ///       │
    ///       ├──▶ 1. Get monomorphized items from rustc
    ///       │       tcx.collect_and_partition_mono_items()
    ///       │
    ///       ├──▶ 2. Count kernels (functions in the reserved cuda_oxide_kernel_ namespace)
    ///       │
    ///       ├──▶ 3. If kernels found:
    ///       │       │
    ///       │       ├──▶ collector::collect_device_functions()
    ///       │       │       Walk call graph from kernels
    ///       │       │       Return Vec<CollectedFunction>
    ///       │       │
    ///       │       └──▶ device_codegen::generate_device_code()
    ///       │               Enter stable_mir context
    ///       │               Convert instances
    ///       │               Call mir_importer::run_pipeline()
    ///       │               Output: .ll and .ptx files
    ///       │
    ///       └──▶ 4. llvm_backend.codegen_crate(tcx)
    ///               Let LLVM handle ALL host code
    /// ```
    fn codegen_crate(&self, tcx: TyCtxt<'_>, crate_info: &CrateInfo) -> Box<dyn Any> {
        // Wrap entire function in with_no_trimmed_paths! to prevent diagnostic state issues.
        // This is necessary because we use tcx.def_path_str() and other functions that
        // trigger trimmed_def_paths. rust-gpu uses the same pattern.
        with_no_trimmed_paths!({
            // Step 1: Analyze for device code
            let mono_partitions = tcx.collect_and_partition_mono_items(());
            let kernel_count = collector::count_kernels_in_cgus(tcx, mono_partitions.codegen_units);
            let device_fn_count =
                collector::count_device_fns_in_cgus(tcx, mono_partitions.codegen_units);
            let unsupported_protocol_root = collector::unsupported_codegen_protocol_root_in_cgus(
                tcx,
                mono_partitions.codegen_units,
            );
            if reject_unsupported_codegen_protocol(
                std::env::var_os(reserved_oxide_symbols::CODEGEN_FINGERPRINT_ENV).is_some(),
                unsupported_protocol_root.is_some(),
            ) {
                let root = unsupported_protocol_root.expect("presence checked above");
                tcx.dcx().fatal(format!(
                    "[rustc_codegen_cuda] Device-code root `{root}` was emitted by a cuda-macros version that predates the scoped Cargo cache protocol. Rebuild the source with cuda-macros from the same cuda-oxide revision; older macro expansions, including pre-expanded output from before this protocol, cannot be cached safely across output and architecture changes."
                ));
            }
            let crate_name = tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE);
            let owner_selected = self.config.allows_device_codegen_for(crate_name.as_str());
            let contains_device_code = kernel_count > 0 || device_fn_count > 0;
            let has_device_code = should_codegen_device_crate(
                &self.config,
                crate_name.as_str(),
                contains_device_code,
            );
            let mut artifact_objects = Vec::new();

            if self.config.verbose && contains_device_code && !owner_selected {
                eprintln!(
                    "[rustc_codegen_cuda] Skipping device code for crate '{}' because it is not in CUDA_OXIDE_DEVICE_CODEGEN_CRATE",
                    crate_name
                );
            }

            // An explicit owner filter activates a target-specific anchor
            // handshake between `#[cuda_module]` and the backend. A selected
            // target with no monomorphized device code still defines that
            // anchor in a placeholder, so generic-only modules can safely
            // reference it. An unselected crate with kernels defines only the
            // weak legacy placeholder needed by older macro expansions.
            let anchor_placeholder = artifact_anchor_placeholder_kind(
                self.config.device_codegen_crates.is_some(),
                owner_selected,
                contains_device_code,
                kernel_count,
            );
            if let Some(anchor_placeholder) = anchor_placeholder {
                let output_dir = self
                    .config
                    .ptx_output_dir
                    .clone()
                    .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));
                match write_artifact_anchor_placeholder(
                    &output_dir,
                    crate_name.as_str(),
                    tcx.sess.target.llvm_target.as_ref(),
                    anchor_placeholder,
                ) {
                    Ok(path) => artifact_objects.push(ArtifactObject {
                        path,
                        retain_in_host_cgus: false,
                    }),
                    Err(error) => tcx.dcx().fatal(format!(
                        "[rustc_codegen_cuda] Failed to write filtered artifact anchor: {error}"
                    )),
                }
            }

            // Only log for crates that have device code (reduces noise from dependency crates)
            if self.config.verbose && has_device_code {
                eprintln!(
                    "[rustc_codegen_cuda] Compiling crate '{}': {} CGUs, {} kernel(s), {} device fn(s)",
                    crate_name,
                    mono_partitions.codegen_units.len(),
                    kernel_count,
                    device_fn_count
                );
            }

            // Step 2: If device code exists, compile via cuda-oxide
            let _device_result = if has_device_code {
                let materialization_request =
                    materialize::request_from_env().unwrap_or_else(|error| {
                        tcx.dcx().fatal(format!(
                            "[rustc_codegen_cuda] Invalid cubin materialization request: {error}"
                        ))
                    });
                if self.config.verbose {
                    eprintln!("[rustc_codegen_cuda] Compiling device code via cuda-oxide...");
                }

                // Collect all device-reachable functions (kernels + their callees)
                let collection_result = collector::collect_device_functions(
                    tcx,
                    mono_partitions.codegen_units,
                    self.config.verbose,
                );

                materialize::validate_collection(
                    materialization_request,
                    !collection_result.device_externs.is_empty(),
                )
                .unwrap_or_else(|error| {
                    tcx.dcx().fatal(format!(
                        "[rustc_codegen_cuda] Cannot materialize this device artifact: {error}"
                    ))
                });

                if self.config.verbose {
                    eprintln!(
                        "[rustc_codegen_cuda] Collected {} device functions, {} device externs for PTX compilation",
                        collection_result.functions.len(),
                        collection_result.device_externs.len()
                    );

                    // Dump MIR info for verification
                    collector::dump_device_mir_info(tcx, &collection_result.functions);
                }

                // Extract references for the pipeline
                let device_functions = &collection_result.functions;

                // Create device codegen config from our config
                let device_config =
                    device_codegen::DeviceCodegenConfig {
                        output_dir: self.config.ptx_output_dir.clone().unwrap_or_else(|| {
                            std::env::current_dir().unwrap_or_else(|_| ".".into())
                        }),
                        output_name: tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).to_string(),
                        verbose: self.config.verbose,
                        dump_rustc_mir: self.config.dump_rustc_mir,
                        dump_mir_dialect: self.config.dump_mir_dialect,
                        dump_llvm_dialect: self.config.dump_llvm_dialect,
                        partition_large_owner: materialization_request.is_some(),
                    };

                // Run the cuda-oxide pipeline, catching backend panics and
                // re-emitting them as a cuda-oxide diagnostic. A panic
                // inside the pipeline (typically pliron's IR invariant
                // checks) would otherwise escape to rustc's panic hook and
                // get dressed up as "the compiler unexpectedly panicked,
                // please file a rustc bug". The bug is in cuda-oxide, so
                // we want users pointed at our tracker, not rustc's.
                //
                // We also briefly swap rustc's ICE hook for our own, because
                // panic hooks fire *before* catch_unwind catches the unwind.
                // Without the swap, the rustc-flavoured banner would still
                // print to stderr ahead of our diagnostic. The replacement
                // hook also captures a backtrace, since by the time we
                // catch the unwind the stack we want is gone. Capture
                // honours `RUST_BACKTRACE` so an unset env var still costs
                // nothing. Hooks are global; rustc's codegen at this entry
                // point is effectively single-threaded, so the brief
                // window where the hook is swapped is safe.
                let (panic_outcome, panic_backtrace) = {
                    use std::backtrace::Backtrace;
                    use std::panic::{AssertUnwindSafe, catch_unwind};
                    use std::sync::{Arc, Mutex};
                    let bt_slot: Arc<Mutex<Option<Backtrace>>> = Arc::new(Mutex::new(None));
                    let bt_setter = Arc::clone(&bt_slot);
                    let prev_hook = std::panic::take_hook();
                    std::panic::set_hook(Box::new(move |_info| {
                        if let Ok(mut g) = bt_setter.lock() {
                            *g = Some(Backtrace::capture());
                        }
                    }));
                    let r = catch_unwind(AssertUnwindSafe(|| {
                        device_codegen::generate_device_code(
                            tcx,
                            device_functions,
                            &collection_result.function_families,
                            &collection_result.device_externs,
                            &device_config,
                        )
                    }));
                    std::panic::set_hook(prev_hook);
                    (r, bt_slot)
                };

                match panic_outcome {
                    Err(payload) => {
                        let msg = payload
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "<opaque panic payload>".into());
                        match panic_backtrace.lock().ok().and_then(|mut g| g.take()) {
                            Some(bt)
                                if bt.status() == std::backtrace::BacktraceStatus::Captured =>
                            {
                                eprintln!("[rustc_codegen_cuda] backtrace:\n{bt}");
                            }
                            _ => {
                                eprintln!(
                                    "[rustc_codegen_cuda] note: run with `RUST_BACKTRACE=1` to display a backtrace"
                                );
                            }
                        }
                        tcx.dcx().fatal(format!(
                            "[rustc_codegen_cuda] Internal compiler error in \
                             device codegen: {msg}. This is a bug in cuda-oxide. \
                             Please file at https://github.com/NVlabs/cuda-oxide/issues"
                        ));
                    }
                    Ok(Ok(result)) => {
                        if self.config.verbose
                            && let Some(artifact) = result.artifacts.first()
                        {
                            eprintln!(
                                "[rustc_codegen_cuda] Device codegen complete: {} partition(s), first={} ({:?}, target: {})",
                                result.artifacts.len(),
                                artifact.name,
                                artifact.kind,
                                result.target
                            );
                        }
                        if !result.artifacts.is_empty() {
                            match write_device_artifact_object(
                                &device_config.output_dir,
                                &device_config.output_name,
                                tcx.sess.target.llvm_target.as_ref(),
                                &result,
                                device_functions,
                                self.config.device_codegen_crates.is_some(),
                                materialization_request,
                            ) {
                                Ok(path) => {
                                    if self.config.verbose {
                                        eprintln!(
                                            "[rustc_codegen_cuda] Embedded artifact object complete: {}",
                                            path.path.display()
                                        );
                                    }
                                    artifact_objects.push(path);
                                }
                                Err(e) => {
                                    tcx.dcx().fatal(format!(
                                        "[rustc_codegen_cuda] Failed to embed device artifact: {e}"
                                    ));
                                }
                            }
                        } else {
                            tcx.dcx().fatal(
                                "[rustc_codegen_cuda] Device codegen did not produce an embeddable artifact",
                            );
                        }
                        Some(result)
                    }
                    Ok(Err(e)) => {
                        // Hard-fail: a swallowed device codegen error produces
                        // a host binary with stale or missing PTX, which then
                        // silently mis-runs on the GPU. The wrapper script
                        // (cargo-oxide) reports "✓ Build succeeded" in that
                        // case because the host LLVM backend below succeeds.
                        // Surface the failure as a rustc fatal so cargo exits
                        // non-zero and the wrapper's success print never fires.
                        // See `.cursor/rules/compiler-gaps-are-bugs.mdc`.
                        tcx.dcx()
                            .fatal(format!("[rustc_codegen_cuda] Device codegen failed: {}", e));
                    }
                }
            } else {
                None
            };

            // Step 3: Delegate ALL host codegen to LLVM backend
            // (No logging here - it fires for every crate including dependencies)
            let host_result = self.llvm_backend.codegen_crate(tcx, crate_info);

            // Return the LLVM backend's result
            Box::new(CudaOngoingCodegen {
                host: host_result,
                artifact_objects,
            })
        })
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let ongoing = *ongoing_codegen
            .downcast::<CudaOngoingCodegen>()
            .expect("rustc_codegen_cuda received unexpected ongoing codegen state");
        let (mut compiled_modules, work_products) =
            self.llvm_backend.join_codegen(ongoing.host, sess, outputs);
        let (retained, standalone): (Vec<_>, Vec<_>) = ongoing
            .artifact_objects
            .into_iter()
            .partition(|artifact| artifact.retain_in_host_cgus);
        if !retained.is_empty()
            && let Err(error) =
                merge_artifacts_into_host_cgus(sess, &mut compiled_modules, &retained)
        {
            sess.dcx().fatal(format!(
                "[rustc_codegen_cuda] Failed to retain generic device artifact: {error}"
            ));
        }
        for (index, artifact) in standalone.into_iter().enumerate() {
            compiled_modules.modules.push(CompiledModule {
                name: format!("oxide_artifact_embed_{index}"),
                kind: ModuleKind::Regular,
                object: Some(artifact.path),
                dwarf_object: None,
                bytecode: None,
                assembly: None,
                llvm_ir: None,
                links_from_incr_cache: Vec::new(),
            });
        }
        (compiled_modules, work_products)
    }

    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        self.llvm_backend
            .link(sess, compiled_modules, crate_info, metadata, outputs);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_device_artifact_object(
    output_dir: &Path,
    output_name: &str,
    host_target: &str,
    result: &device_codegen::DeviceCodegenResult,
    functions: &[collector::CollectedFunction<'_>],
    use_target_specific_anchor: bool,
    materialization_request: Option<materialize::MaterializationRequest>,
) -> Result<ArtifactObject, Box<dyn std::error::Error>> {
    let bundle_name = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| output_name.to_string());
    let materialized_artifact;
    let materialized_ptx_audit_path;
    let source_artifact = result
        .artifacts
        .first()
        .ok_or("device codegen produced no embeddable artifact")?;
    let (artifact, was_materialized) =
        match materialize_artifact_for_embedding(materialization_request, &bundle_name, result)? {
            Some(materialized) => {
                materialized_artifact = materialized.artifact;
                materialized_ptx_audit_path = materialized.ptx_audit_path;
                (&materialized_artifact, true)
            }
            None => {
                if result.ptx_bundle_path.is_some() {
                    return Err("partitioned device artifacts require cubin materialization".into());
                }
                if result.artifacts.len() != 1 {
                    return Err("unpartitioned device codegen must produce one artifact".into());
                }
                materialized_ptx_audit_path = None;
                (source_artifact, false)
            }
        };
    if let Some(audit_path) = &materialized_ptx_audit_path
        && !audit_path.is_file()
    {
        return Err(format!(
            "materialized PTX audit artifact was not published: {}",
            audit_path.display()
        )
        .into());
    }
    let payload_kind = match artifact.kind {
        device_codegen::DeviceCodegenArtifactKind::Ptx => oxide_artifacts::ArtifactPayloadKind::Ptx,
        device_codegen::DeviceCodegenArtifactKind::NvvmIr => {
            oxide_artifacts::ArtifactPayloadKind::NvvmIr
        }
        device_codegen::DeviceCodegenArtifactKind::Ltoir => {
            oxide_artifacts::ArtifactPayloadKind::Ltoir
        }
        device_codegen::DeviceCodegenArtifactKind::Cubin => {
            oxide_artifacts::ArtifactPayloadKind::Cubin
        }
    };
    // Preserve the actual policy even after IR has become a cubin. Consumers
    // and diagnostics must not mistake `--no-fmad` materialization for a
    // default-policy artifact merely because no compilation remains to do.
    let carries_compile_policy = was_materialized
        || matches!(
            artifact.kind,
            device_codegen::DeviceCodegenArtifactKind::NvvmIr
                | device_codegen::DeviceCodegenArtifactKind::Ltoir
        );
    let compile_options = embedded_compile_options(
        result.allow_fma_contraction,
        result.debug_kind,
        carries_compile_policy,
    );
    let mut spec = oxide_artifacts::ArtifactBundleSpec::new(&bundle_name, &result.target)
        .with_compile_options(compile_options)
        .with_payload(oxide_artifacts::ArtifactPayloadSpec::new(
            payload_kind,
            &artifact.name,
            artifact
                .bytes
                .as_deref()
                .ok_or("embedded artifact payload was not retained in memory")?,
        ));
    for function in functions {
        let kind = if function.is_kernel {
            oxide_artifacts::ArtifactEntryKind::Kernel
        } else {
            oxide_artifacts::ArtifactEntryKind::DeviceFunction
        };
        spec = spec.with_entry(oxide_artifacts::ArtifactEntrySpec::new(
            &function.export_name,
            kind,
        ));
    }

    let blob = oxide_artifacts::build_artifact_blob(&spec)?;
    // Define a link-anchor symbol at the start of the `.oxart` data. When
    // this crate is a library, the artifact object becomes an rlib archive
    // member, and the linker only extracts it if some other object holds an
    // undefined reference to a symbol defined here. The `#[cuda_module]`
    // macro emits that reference from the generated `load_named()`. Normal
    // builds preserve the legacy package-level symbol. Owner-filtered builds
    // use a target-specific v2 symbol plus a weak legacy alias, so an older
    // macro still links without letting a filtered target hide a selected
    // target's artifact. Without an anchor, library-crate bundles were
    // dead-stripped and `load()` failed at runtime with ModuleNotFound
    // (issue #72).
    let package_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let legacy_anchor =
        reserved_oxide_symbols::artifact_anchor_symbol(&bundle_name, &package_version);
    let generic_only = artifact_contains_only_generic_kernels(functions);
    // A generic specialization can be emitted in a downstream crate that has
    // no matching `#[cuda_module]` anchor reference. Put generic-only bundles
    // in a COMDAT group and attach that object to every host CGU below. Any
    // archive member selected by a consumer then carries the bundle, and the
    // final linker keeps one copy when it selects several members.
    let object = if generic_only {
        let binary_name = std::env::var("CARGO_BIN_NAME").ok();
        let target_anchor = reserved_oxide_symbols::artifact_anchor_symbol_v2(
            &bundle_name,
            &package_version,
            output_name,
            binary_name.as_deref(),
        );
        let comdat_symbol = generic_artifact_comdat_symbol(&blob);
        oxide_artifacts::build_host_object_for_target_with_legacy_anchor_and_comdat(
            &blob,
            host_target,
            &target_anchor,
            &legacy_anchor,
            &comdat_symbol,
        )?
    } else if use_target_specific_anchor {
        let binary_name = std::env::var("CARGO_BIN_NAME").ok();
        let target_anchor = reserved_oxide_symbols::artifact_anchor_symbol_v2(
            &bundle_name,
            &package_version,
            output_name,
            binary_name.as_deref(),
        );
        oxide_artifacts::build_host_object_for_target_with_legacy_anchor(
            &blob,
            host_target,
            &target_anchor,
            &legacy_anchor,
        )?
    } else {
        oxide_artifacts::build_host_object_for_target(&blob, host_target, Some(&legacy_anchor))?
    };
    Ok(ArtifactObject {
        path: write_artifact_object(output_dir, output_name, host_target, &object, "embed")?,
        retain_in_host_cgus: generic_only,
    })
}

fn embedded_compile_options(
    allow_fma_contraction: bool,
    debug_kind: llvm_export::export::DebugKind,
    carries_compile_policy: bool,
) -> oxide_artifacts::ArtifactCompileOptions {
    if carries_compile_policy {
        let debug_policy = match debug_kind {
            llvm_export::export::DebugKind::Off => oxide_artifacts::ArtifactDebugPolicy::None,
            llvm_export::export::DebugKind::LineTables => {
                oxide_artifacts::ArtifactDebugPolicy::LineTables
            }
            llvm_export::export::DebugKind::Full => oxide_artifacts::ArtifactDebugPolicy::Full,
        };
        oxide_artifacts::ArtifactCompileOptions::new()
            .with_fma_contraction(allow_fma_contraction)
            .with_debug_policy(debug_policy)
    } else {
        // Preserve main's byte-level PTX/already-cubin default. These payloads
        // never carried later-stage compile policy before materialization was
        // added, including when the backend itself ran with --no-fmad.
        oxide_artifacts::ArtifactCompileOptions::new()
    }
}

/// Opt-in (`CUDA_OXIDE_MATERIALIZE_CUBIN`): compile PTX / NVVM IR / LTOIR
/// artifacts down to a final cubin before embedding, so the consuming
/// binary loads device code directly through the CUDA driver — no libNVVM
/// or nvJitLink on the deployment host and no first-load compile hit. The
/// cubin is pinned to the emitted architecture; the default (embed the IR,
/// compile at load) keeps cuda-host's execution routing, including the PTX
/// bridge to newer GPUs. The ordinary optimized PTX route is preferred; NVVM
/// IR and LTOIR cover modules that require deferred compilation. Every route
/// uses the wrapper's provenance-checked finalizer. See `materialize` for the
/// trade-offs.
struct MaterializedEmbeddingArtifact {
    artifact: device_codegen::DeviceCodegenArtifact,
    /// Single owner-level audit artifact: ordinary `.ptx` below the threshold
    /// or versioned `.ptx.bundle` for a partitioned owner.
    ptx_audit_path: Option<PathBuf>,
}

/// Partition IR/PTX files are private compiler intermediates. Remove the exact
/// hidden owner directory after bounded materialization, including error
/// paths, so changed partition membership cannot accumulate stale modules.
struct PartitionIntermediatesCleanup {
    owner_dir: PathBuf,
}

impl PartitionIntermediatesCleanup {
    fn for_artifacts(
        artifacts: &[device_codegen::DeviceCodegenArtifact],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let owner_dir = artifacts
            .first()
            .and_then(|artifact| artifact.path.parent())
            .ok_or("partitioned artifact has no owner directory")?
            .to_path_buf();
        if owner_dir
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some(".cuda-oxide-partitions")
        {
            return Err(format!(
                "partitioned artifact escaped the hidden compiler directory: {}",
                owner_dir.display()
            )
            .into());
        }
        if artifacts
            .iter()
            .any(|artifact| artifact.path.parent() != Some(owner_dir.as_path()))
        {
            return Err("partitioned artifacts do not share one owner directory".into());
        }
        Ok(Self { owner_dir })
    }
}

impl Drop for PartitionIntermediatesCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.owner_dir);
    }
}

fn materialize_artifact_for_embedding(
    request: Option<materialize::MaterializationRequest>,
    bundle_name: &str,
    result: &device_codegen::DeviceCodegenResult,
) -> Result<Option<MaterializedEmbeddingArtifact>, Box<dyn std::error::Error>> {
    let Some(request) = request else {
        return Ok(None);
    };
    let debug_policy = match result.debug_kind {
        llvm_export::export::DebugKind::Off => cuda_artifact_finalizer::DebugPolicy::None,
        llvm_export::export::DebugKind::LineTables => {
            cuda_artifact_finalizer::DebugPolicy::LineTables
        }
        llvm_export::export::DebugKind::Full => cuda_artifact_finalizer::DebugPolicy::Full,
    };
    if let Some(bundle_path) = result.ptx_bundle_path.as_deref() {
        if result.artifacts.len() < 2 {
            return Err(format!(
                "partitioned owner expected at least two artifacts, found {}",
                result.artifacts.len()
            )
            .into());
        }
        let _partition_intermediates =
            PartitionIntermediatesCleanup::for_artifacts(&result.artifacts)?;
        let inputs = result
            .artifacts
            .iter()
            .map(|artifact| {
                let kind = match artifact.kind {
                    device_codegen::DeviceCodegenArtifactKind::NvvmIr => {
                        cuda_artifact_finalizer::PartitionFileInputKind::NvvmIr
                    }
                    device_codegen::DeviceCodegenArtifactKind::Ptx => {
                        cuda_artifact_finalizer::PartitionFileInputKind::Ptx
                    }
                    device_codegen::DeviceCodegenArtifactKind::Ltoir
                    | device_codegen::DeviceCodegenArtifactKind::Cubin => {
                        return Err(format!(
                            "partitioned owner input {} has unsupported kind {:?}",
                            artifact.name, artifact.kind
                        ));
                    }
                };
                Ok(cuda_artifact_finalizer::PartitionFileInput::new(
                    &artifact.name,
                    &artifact.path,
                    kind,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let materialized = materialize::partition_files_to_cubin(
            request,
            &inputs,
            bundle_path,
            &result.target,
            result.allow_fma_contraction,
            debug_policy,
        )?;
        if let Some(warning) = &materialized.ptx_bundle_durability_warning {
            eprintln!("[rustc_codegen_cuda] warning: {warning}");
        }
        if std::env::var_os("CUDA_OXIDE_PARTITION_STATS").is_some() {
            for (index, partition) in materialized.partitions.iter().enumerate() {
                eprintln!(
                    "[rustc_codegen_cuda] owner partition finalization: index={index} \
                     name={} source_bytes={} ptx_bytes={} nvvm_compile_elapsed={:?} \
                     jit_link_add_elapsed={:?} peak_rss_kib={:?}",
                    partition.name,
                    partition.source_bytes,
                    partition.ptx_bytes,
                    partition.nvvm_compile_elapsed,
                    partition.jit_link_add_elapsed,
                    partition.peak_rss_kib,
                );
            }
            eprintln!(
                "[rustc_codegen_cuda] owner partition link: partitions={} cubin_bytes={} \
                 link_elapsed={:?} peak_rss_kib={:?} ptx_bundle={}",
                materialized.partitions.len(),
                materialized.cubin.len(),
                materialized.link_elapsed,
                materialized.peak_rss_kib,
                materialized.ptx_bundle_path.display(),
            );
        }
        return Ok(Some(materialized_cubin_embedding(
            bundle_name,
            materialized.cubin,
            Some(materialized.ptx_bundle_path),
        )));
    }
    if result.artifacts.len() != 1 {
        return Err(format!(
            "unpartitioned owner expected one artifact, found {}",
            result.artifacts.len()
        )
        .into());
    }

    let artifact = result
        .artifacts
        .first()
        .ok_or("device codegen produced no materialization input")?;
    let bytes = artifact
        .bytes
        .as_deref()
        .ok_or("single-module materialization input was not retained")?;
    let materialized = match artifact.kind {
        device_codegen::DeviceCodegenArtifactKind::NvvmIr => {
            let artifacts = materialize::nvvm_ir_to_artifacts(
                request,
                bytes,
                bundle_name,
                &result.target,
                result.allow_fma_contraction,
                debug_policy,
            )?;
            write_materialized_ptx_sidecar(&result.ptx_path, &artifacts.ptx_input)?;
            materialized_cubin_embedding(
                bundle_name,
                artifacts.cubin,
                Some(result.ptx_path.clone()),
            )
        }
        device_codegen::DeviceCodegenArtifactKind::Ltoir => {
            let cubin = materialize::ltoir_to_cubin(
                request,
                bytes,
                &artifact.name,
                &result.target,
                result.allow_fma_contraction,
                debug_policy,
            )?;
            materialized_cubin_embedding(bundle_name, cubin, None)
        }
        device_codegen::DeviceCodegenArtifactKind::Ptx => {
            let cubin = materialize::ptx_to_cubin(
                request,
                bytes,
                &artifact.name,
                &result.target,
                result.allow_fma_contraction,
                debug_policy,
            )?;
            materialized_cubin_embedding(bundle_name, cubin, Some(result.ptx_path.clone()))
        }
        device_codegen::DeviceCodegenArtifactKind::Cubin => {
            return Err(Box::new(materialize::MaterializeError::CubinInput));
        }
    };
    Ok(Some(materialized))
}

fn materialized_cubin_embedding(
    bundle_name: &str,
    cubin: Vec<u8>,
    ptx_audit_path: Option<PathBuf>,
) -> MaterializedEmbeddingArtifact {
    MaterializedEmbeddingArtifact {
        artifact: device_codegen::DeviceCodegenArtifact {
            kind: device_codegen::DeviceCodegenArtifactKind::Cubin,
            name: format!("{bundle_name}.cubin"),
            path: PathBuf::from(format!("{bundle_name}.cubin")),
            ptx_sidecar_path: ptx_audit_path.clone().unwrap_or_default(),
            bytes: Some(cubin),
        },
        ptx_audit_path,
    }
}

fn write_materialized_ptx_sidecar(path: &Path, ptx: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, ptx)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactAnchorPlaceholderKind {
    SelectedTarget,
    UnselectedLegacy,
}

fn artifact_contains_only_generic_kernels(functions: &[collector::CollectedFunction<'_>]) -> bool {
    generic_kernel_exports_require_cgu_retention(
        functions
            .iter()
            .filter(|function| function.is_kernel)
            .map(|function| function.export_name.as_str()),
    )
}

fn generic_kernel_exports_require_cgu_retention<'a>(
    export_names: impl IntoIterator<Item = &'a str>,
) -> bool {
    let mut export_names = export_names.into_iter().peekable();
    export_names.peek().is_some() && export_names.all(|name| name.contains("_TID_"))
}

fn generic_artifact_comdat_symbol(blob: &[u8]) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    blob.hash(&mut hasher);
    format!(
        "cuda_oxide_artifact_comdat_246e25db_{:016x}",
        hasher.finish()
    )
}

fn merge_artifacts_into_host_cgus(
    sess: &Session,
    compiled_modules: &mut CompiledModules,
    artifacts: &[ArtifactObject],
) -> Result<(), String> {
    let rust_lld = sess
        .get_tools_search_paths(false)
        .into_iter()
        .map(|directory| directory.join("rust-lld"))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| "rust-lld was not found in the compiler tool paths".to_string())?;
    let mut host_object_count = 0;
    for module in &mut compiled_modules.modules {
        if !matches!(module.kind, ModuleKind::Regular) {
            continue;
        }
        let Some(host_object) = module.object.as_ref() else {
            continue;
        };
        host_object_count += 1;
        let merge_id = ARTIFACT_OBJECT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let merged_object = host_object.with_extension(format!("oxide-merge-{merge_id}.o"));
        let output = std::process::Command::new(&rust_lld)
            .args(["-flavor", "gnu", "-r"])
            .arg(&host_object)
            .args(artifacts.iter().map(|artifact| artifact.path.as_path()))
            .arg("-o")
            .arg(&merged_object)
            .output()
            .map_err(|error| format!("could not run {}: {error}", rust_lld.display()))?;
        if !output.status.success() {
            let _ = std::fs::remove_file(&merged_object);
            return Err(format!(
                "{} failed while merging {}: {}",
                rust_lld.display(),
                host_object.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        module.object = Some(merged_object);
    }
    if host_object_count == 0 {
        return Err("host LLVM codegen produced no regular object files".to_string());
    }
    Ok(())
}

fn artifact_anchor_placeholder_kind(
    owner_filter_active: bool,
    owner_selected: bool,
    contains_device_code: bool,
    kernel_count: usize,
) -> Option<ArtifactAnchorPlaceholderKind> {
    if !owner_filter_active {
        None
    } else if owner_selected && !contains_device_code {
        Some(ArtifactAnchorPlaceholderKind::SelectedTarget)
    } else if !owner_selected && kernel_count > 0 {
        Some(ArtifactAnchorPlaceholderKind::UnselectedLegacy)
    } else {
        None
    }
}

fn write_artifact_anchor_placeholder(
    output_dir: &Path,
    output_name: &str,
    host_target: &str,
    kind: ArtifactAnchorPlaceholderKind,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let bundle_name = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| output_name.to_string());
    let package_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let legacy_anchor =
        reserved_oxide_symbols::artifact_anchor_symbol(&bundle_name, &package_version);
    let object = match kind {
        ArtifactAnchorPlaceholderKind::SelectedTarget => {
            let binary_name = std::env::var("CARGO_BIN_NAME").ok();
            let target_anchor = reserved_oxide_symbols::artifact_anchor_symbol_v2(
                &bundle_name,
                &package_version,
                output_name,
                binary_name.as_deref(),
            );
            oxide_artifacts::build_host_anchor_object_for_target_with_legacy_anchor(
                host_target,
                &target_anchor,
                &legacy_anchor,
            )?
        }
        ArtifactAnchorPlaceholderKind::UnselectedLegacy => {
            oxide_artifacts::build_host_anchor_object_for_target(host_target, &legacy_anchor)?
        }
    };
    write_artifact_object(output_dir, output_name, host_target, &object, "anchor")
}

fn write_artifact_object(
    output_dir: &Path,
    output_name: &str,
    host_target: &str,
    object: &[u8],
    kind: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let safe_output_name = sanitize_path_component(output_name);
    let artifact_id = ARTIFACT_OBJECT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let object_dir = output_dir
        .join(".oxide-artifacts")
        .join(&safe_output_name)
        .join(sanitize_path_component(host_target));
    std::fs::create_dir_all(&object_dir)?;
    let object_path = object_dir.join(format!(
        "{safe_output_name}.{}.{artifact_id}.{kind}.o",
        std::process::id(),
    ));
    std::fs::write(&object_path, object)?;
    Ok(object_path)
}

fn sanitize_path_component(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => ch,
            _ => '_',
        })
        .collect()
}

/// Entry point called by rustc to instantiate the backend.
///
/// This function is discovered by rustc via the `#[no_mangle]` attribute and the
/// specific name `__rustc_codegen_backend`. When a user specifies
/// `-Z codegen-backend=path/to/librustc_codegen_cuda.so`, rustc loads the shared
/// library and calls this function to get a `Box<dyn CodegenBackend>`.
///
/// ## Initialization Sequence
///
/// ```text
/// rustc -Z codegen-backend=librustc_codegen_cuda.so ...
///       │
///       ├──▶ dlopen("librustc_codegen_cuda.so")
///       │
///       ├──▶ dlsym("__rustc_codegen_backend")
///       │
///       └──▶ __rustc_codegen_backend()
///               │
///               ├──▶ CudaCodegenConfig::from_env()
///               │       Read CUDA_OXIDE_* env vars
///               │
///               ├──▶ rustc_codegen_llvm::LlvmCodegenBackend::new()
///               │       Create the wrapped LLVM backend
///               │
///               └──▶ Return Box<CudaCodegenBackend>
/// ```
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    let config = CudaCodegenConfig::from_env();

    // Note: Don't log here - this function is called for EVERY crate in the dependency tree.
    // We log in codegen_crate() only when there are kernels to compile.

    // Get the LLVM backend - this is the same function rustc calls normally
    let llvm_backend = rustc_codegen_llvm::LlvmCodegenBackend::new();

    Box::new(CudaCodegenBackend {
        config,
        llvm_backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_codegen_owner_filter_normalizes_and_matches_crate_names() {
        let owners = parse_device_codegen_crates(Some(" gpu-kernels,math_gpu , gpu-kernels "))
            .expect("an explicitly configured filter");
        assert_eq!(
            owners,
            BTreeSet::from(["gpu_kernels".to_string(), "math_gpu".to_string()])
        );

        let config = CudaCodegenConfig {
            device_codegen_crates: Some(owners),
            ..CudaCodegenConfig::default()
        };
        assert!(config.allows_device_codegen_for("gpu_kernels"));
        assert!(config.allows_device_codegen_for("gpu-kernels"));
        assert!(config.allows_device_codegen_for("math_gpu"));
        assert!(!config.allows_device_codegen_for("host_app"));
        assert!(should_codegen_device_crate(&config, "gpu_kernels", true));
        assert!(!should_codegen_device_crate(&config, "host_app", true));
        assert!(!should_codegen_device_crate(&config, "gpu_kernels", false));
    }

    #[test]
    fn absent_owner_filter_allows_device_codegen_for_every_crate() {
        let config = CudaCodegenConfig::default();
        assert!(config.allows_device_codegen_for("gpu_kernels"));
        assert!(config.allows_device_codegen_for("host_app"));
    }

    #[test]
    fn empty_owner_filter_is_treated_as_unset() {
        let config = CudaCodegenConfig {
            device_codegen_crates: parse_device_codegen_crates(Some(" , ")),
            ..CudaCodegenConfig::default()
        };
        assert!(config.allows_device_codegen_for("gpu_kernels"));
    }

    #[test]
    fn scoped_cache_protocol_rejects_old_roots_even_when_codegen_would_be_filtered() {
        assert!(reject_unsupported_codegen_protocol(true, true));
        assert!(!reject_unsupported_codegen_protocol(false, true));
        assert!(!reject_unsupported_codegen_protocol(true, false));

        let config = CudaCodegenConfig {
            device_codegen_crates: parse_device_codegen_crates(Some("some_other_crate")),
            ..CudaCodegenConfig::default()
        };
        assert!(!config.allows_device_codegen_for("legacy_kernel_crate"));
        // Protocol validation is intentionally independent of owner selection:
        // filtered legacy output must not become fresh forever when the filter
        // later selects the crate.
        assert!(reject_unsupported_codegen_protocol(true, true));
    }

    #[test]
    fn materialized_cubin_keeps_target_entries_and_no_fmad_policy() {
        use oxide_artifacts::{
            ArtifactBundleSpec, ArtifactEntryKind, ArtifactEntrySpec, ArtifactPayloadKind,
            ArtifactPayloadSpec, build_artifact_blob, parse_artifact_blob,
        };

        let materialized = materialized_cubin_embedding(
            "demo",
            b"final cubin".to_vec(),
            Some(PathBuf::from("demo.ptx")),
        );
        assert_eq!(
            materialized.ptx_audit_path.as_deref(),
            Some(Path::new("demo.ptx"))
        );
        assert_eq!(
            materialized.artifact.kind,
            device_codegen::DeviceCodegenArtifactKind::Cubin
        );

        let blob = build_artifact_blob(
            &ArtifactBundleSpec::new("demo", "sm_90a")
                .with_compile_options(embedded_compile_options(
                    false,
                    llvm_export::export::DebugKind::LineTables,
                    true,
                ))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Cubin,
                    &materialized.artifact.name,
                    materialized.artifact.bytes.as_deref().unwrap(),
                ))
                .with_entry(ArtifactEntrySpec::new(
                    "kernel_a",
                    ArtifactEntryKind::Kernel,
                ))
                .with_entry(ArtifactEntrySpec::new(
                    "helper",
                    ArtifactEntryKind::DeviceFunction,
                )),
        )
        .unwrap();
        let bundle = parse_artifact_blob(&blob).unwrap();

        assert_eq!(bundle.name, "demo");
        assert_eq!(bundle.target, "sm_90a");
        assert_eq!(
            bundle.payload(ArtifactPayloadKind::Cubin),
            Some(&b"final cubin"[..])
        );
        assert_eq!(bundle.payload(ArtifactPayloadKind::Ptx), None);
        assert!(!bundle.compile_options.fma_contraction_enabled());
        assert_eq!(
            bundle.compile_options.debug_policy(),
            oxide_artifacts::ArtifactDebugPolicy::LineTables
        );
        assert_eq!(bundle.entries.len(), 2);
        assert_eq!(
            bundle.entry("kernel_a").map(|entry| entry.kind),
            Some(ArtifactEntryKind::Kernel)
        );
        assert_eq!(
            bundle.entry("helper").map(|entry| entry.kind),
            Some(ArtifactEntryKind::DeviceFunction)
        );
    }

    #[test]
    fn direct_ptx_materialization_keeps_audit_path_and_cubin() {
        let cubin = b"final cubin".to_vec();

        let materialized =
            materialized_cubin_embedding("demo", cubin.clone(), Some(PathBuf::from("demo.ptx")));

        assert_eq!(
            materialized.ptx_audit_path.as_deref(),
            Some(Path::new("demo.ptx"))
        );
        assert_eq!(
            materialized.artifact.kind,
            device_codegen::DeviceCodegenArtifactKind::Cubin
        );
        assert_eq!(materialized.artifact.name, "demo.cubin");
        assert_eq!(materialized.artifact.bytes, Some(cubin));
    }

    #[test]
    fn non_nul_ptx_sidecar_preserves_exact_source_bytes() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "cuda-codegen-materialized-ptx-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ptx_path = temp_dir.join("demo.ptx");
        let ptx_source = b".version 8.7\n.visible .entry demo() { ret; }\n";

        write_materialized_ptx_sidecar(&ptx_path, ptx_source).unwrap();

        let sidecar = std::fs::read(&ptx_path).unwrap();
        assert_eq!(sidecar, ptx_source);
        assert_ne!(sidecar.last(), Some(&0));
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn ordinary_ptx_keeps_legacy_default_bundle_header() {
        let blob = oxide_artifacts::build_artifact_blob(
            &oxide_artifacts::ArtifactBundleSpec::new("demo", "sm_90")
                .with_compile_options(embedded_compile_options(
                    false,
                    llvm_export::export::DebugKind::Full,
                    false,
                ))
                .with_payload(oxide_artifacts::ArtifactPayloadSpec::new(
                    oxide_artifacts::ArtifactPayloadKind::Ptx,
                    "demo.ptx",
                    b"ptx",
                )),
        )
        .unwrap();

        assert_eq!(&blob[..8], &oxide_artifacts::ARTIFACT_MAGIC);
        assert_eq!(u16::from_le_bytes([blob[8], blob[9]]), 1);
        let parsed = oxide_artifacts::parse_artifact_blob(&blob).unwrap();
        assert!(parsed.compile_options.fma_contraction_enabled());
        assert_eq!(
            parsed.compile_options.debug_policy(),
            oxide_artifacts::ArtifactDebugPolicy::None
        );
        assert_eq!(
            parsed.payload(oxide_artifacts::ArtifactPayloadKind::Ptx),
            Some(&b"ptx"[..])
        );
    }

    #[test]
    fn owner_anchor_placeholder_policy_matches_artifact_production() {
        use ArtifactAnchorPlaceholderKind::{SelectedTarget, UnselectedLegacy};

        assert_eq!(
            artifact_anchor_placeholder_kind(true, true, false, 0),
            Some(SelectedTarget)
        );
        assert_eq!(
            artifact_anchor_placeholder_kind(true, false, true, 1),
            Some(UnselectedLegacy)
        );
        assert_eq!(artifact_anchor_placeholder_kind(true, true, true, 1), None);
        assert_eq!(
            artifact_anchor_placeholder_kind(false, true, false, 0),
            None
        );
    }

    #[test]
    fn only_generic_kernel_exports_require_cgu_retention() {
        assert!(generic_kernel_exports_require_cgu_retention([
            "scale_TID_0123456789abcdef0123456789abcdef",
            "map_TID_fedcba9876543210fedcba9876543210",
        ]));
        assert!(!generic_kernel_exports_require_cgu_retention([
            "scale_TID_0123456789abcdef0123456789abcdef",
            "concrete",
        ]));
        assert!(!generic_kernel_exports_require_cgu_retention([]));
    }
}
