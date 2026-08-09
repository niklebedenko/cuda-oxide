/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # Device Code Generation via cuda-oxide Pipeline
//!
//! This module bridges rustc's internal MIR representation to cuda-oxide's
//! existing MIR→PTX pipeline using `rustc_public::rustc_internal` to convert
//! between internal and stable_mir types.
//!
//! ## The Bridge Problem
//!
//! We have two different MIR representations:
//!
//! | API                         | Used By                       | Type                                |
//! |-----------------------------|-------------------------------|-------------------------------------|
//! | `rustc_middle` (internal)   | rustc internals, this backend | `rustc_middle::ty::Instance<'tcx>`  |
//! | `rustc_public` (stable MIR) | mir-importer pipeline         | `rustc_public::mir::mono::Instance` |
//!
//! The cuda-oxide pipeline (mir-importer) was built using `rustc_public` APIs because
//! they're more stable. But as a codegen backend, we receive `rustc_middle` types from
//! rustc. This module bridges between them.
//!
//! ## Bridge Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                         DEVICE CODE GENERATION                                  │
//! │                                                                                 │
//! │   Input: Vec<CollectedFunction<'tcx>>                                           │
//! │          (using rustc_middle::ty::Instance)                                     │
//! │                                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 1: Enter stable_mir Context                                       │   │
//! │   │                                                                         │   │
//! │   │  rustc_internal::run(tcx, || { ... })                                   │   │
//! │   │                                                                         │   │
//! │   │  This sets up the Tables and CompilerCtxt that enable type conversion   │   │
//! │   │  between rustc_middle and rustc_public types.                           │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 2: Convert Instances                                              │   │
//! │   │                                                                         │   │
//! │   │  for each CollectedFunction<'tcx>:                                      │   │
//! │   │      stable_instance = rustc_internal::stable(func.instance)            │   │
//! │   │                                                                         │   │
//! │   │  This converts:                                                         │   │
//! │   │    rustc_middle::ty::Instance<'tcx>                                     │   │
//! │   │         ▼                                                               │   │
//! │   │    rustc_public::mir::mono::Instance                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 3: Run cuda-oxide Pipeline                                        │   │
//! │   │                                                                         │   │
//! │   │  mir_importer::run_pipeline(&stable_functions, &config)                 │   │
//! │   │                                                                         │   │
//! │   │  Pipeline stages:                                                       │   │
//! │   │    1. Rust MIR → `dialect-mir` (alloca form)                            │   │
//! │   │    2. `dialect-mir` → `dialect-mir` (mem2reg → SSA)                     │   │
//! │   │    3. Apply annotated loop unrolling                                    │   │
//! │   │    4. `dialect-mir` → LLVM dialect (via `mir-lower`)                    │   │
//! │   │    5. LLVM dialect → textual LLVM IR (.ll)                              │   │
//! │   │    6. LLVM IR → PTX via `llc` (.ptx)                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  Output: DeviceCodegenResult                                            │   │
//! │   │                                                                         │   │
//! │   │    - ptx_path: Path to generated .ptx file                              │   │
//! │   │    - ll_path: Path to generated .ll file                                │   │
//! │   │    - target: GPU target (e.g., "sm_80", "sm_90a")                       │   │
//! │   │    - ptx_content: PTX as string, when PTX was generated                 │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why This Design?
//!
//! We chose to bridge to stable_mir rather than rewrite mir-importer because:
//!
//! 1. **Code reuse**: mir-importer already works and is well-tested
//! 2. **Stability**: rustc_public APIs change less than rustc internals
//! 3. **Simplicity**: ~100 lines of bridge code vs rewriting the pipeline
//! 4. **Maintainability**: Changes to mir-importer automatically work here
//!
//! The cost is one extra type conversion step, but this happens once per function
//! and is negligible compared to actual compilation time.

use crate::collector::{CollectedFunction, DeviceExternDecl, DeviceFunctionFamily};
use llvm_export::ops::{
    DebugInlinedScope, DebugSourcePosition, DebugSourceScope, DebugSourceScopeLocation,
    DebugSourceScopeMap,
};
use rustc_middle::ty::{EarlyBinder, Instance, InstanceKind, TypingEnv};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_session::config::DebugInfo;
use rustc_span::{Span, hygiene};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::hash::Hash;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceExternTypePosition {
    Parameter,
    Result,
    Pointee,
}

fn inline_attr_for_device_function(
    tcx: TyCtxt<'_>,
    instance: Instance<'_>,
) -> mir_importer::InlineAttr {
    let def_id = instance.def_id();
    match tcx.codegen_fn_attrs(def_id).inline {
        rustc_hir::attrs::InlineAttr::Hint
            if is_primitive_float_operator(tcx, def_id) || is_tiny_inline_helper(tcx, def_id) =>
        {
            mir_importer::InlineAttr::DeviceAlways
        }
        rustc_hir::attrs::InlineAttr::Hint => mir_importer::InlineAttr::Hint,
        rustc_hir::attrs::InlineAttr::Always | rustc_hir::attrs::InlineAttr::Force { .. }
            if is_tiny_inline_helper(tcx, def_id) =>
        {
            mir_importer::InlineAttr::DeviceAlways
        }
        rustc_hir::attrs::InlineAttr::Always | rustc_hir::attrs::InlineAttr::Force { .. } => {
            mir_importer::InlineAttr::Always
        }
        rustc_hir::attrs::InlineAttr::None | rustc_hir::attrs::InlineAttr::Never => {
            mir_importer::InlineAttr::None
        }
    }
}

#[derive(Default)]
struct DeviceInlinePlan<'tcx> {
    array_builders: HashSet<Instance<'tcx>>,
    array_closures: HashSet<Instance<'tcx>>,
    borrowed_kernel_closures: HashSet<Instance<'tcx>>,
    borrowed_always_inline_helper_closures: HashSet<Instance<'tcx>>,
    deferred_full_unroll_helpers: HashSet<Instance<'tcx>>,
    considered_array_builders: usize,
    rejected_array_builders: usize,
    conflicting_array_closures: usize,
    conflicting_deferred_unroll_closures: usize,
}

/// Exact per-function policy derived before stable-MIR import.
///
/// Device artifact cache identities consume this same policy so a query used
/// only by the planner (for example an enclosing closure parent's inline
/// attribute) cannot change emitted code behind an otherwise unchanged cache
/// key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceFunctionCodegenPolicy {
    pub(crate) inline_attr: mir_importer::InlineAttr,
    pub(crate) device_link_always: bool,
    pub(crate) device_link_inline_candidate: bool,
    pub(crate) deferred_full_unroll: bool,
}

pub(crate) fn device_function_codegen_policies<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
) -> Vec<DeviceFunctionCodegenPolicy> {
    let plan = build_device_inline_plan(tcx, functions);
    device_function_codegen_policies_from_plan(tcx, functions, &plan)
}

fn device_function_codegen_policies_from_plan<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
    plan: &DeviceInlinePlan<'tcx>,
) -> Vec<DeviceFunctionCodegenPolicy> {
    functions
        .iter()
        .map(|function| {
            let instance = function.instance;
            let borrowed_closure = plan.borrowed_kernel_closures.contains(&instance)
                || plan
                    .borrowed_always_inline_helper_closures
                    .contains(&instance);
            DeviceFunctionCodegenPolicy {
                inline_attr: if borrowed_closure {
                    mir_importer::InlineAttr::DeviceAlways
                } else {
                    inline_attr_for_device_function(tcx, instance)
                },
                device_link_always: plan.array_closures.contains(&instance)
                    || plan.deferred_full_unroll_helpers.contains(&instance),
                device_link_inline_candidate: plan.array_builders.contains(&instance),
                deferred_full_unroll: plan.deferred_full_unroll_helpers.contains(&instance),
            }
        })
        .collect()
}

// These are compile-resource guards rather than semantic limits. Mandatory
// device-link inlining is restricted to concrete array callbacks and the
// erased helpers selected for bounded deferred full unrolling.
const MAX_DEVICE_LINK_INLINE_BLOCKS: usize = 24;
const MAX_DEVICE_LINK_INLINE_STATEMENTS: usize = 128;
// Three-axis tensor operators can have a large optimized callback even though
// the array builder invokes it only three times. Bound that special case by
// both its individual body and the total code duplicated by full unrolling.
const MAX_DEVICE_LINK_WIDE_CALLBACK_ARRAY_EXTENT: u64 = 3;
const MAX_DEVICE_LINK_WIDE_CALLBACK_BLOCKS: usize = 96;
const MAX_DEVICE_LINK_WIDE_CALLBACK_STATEMENTS: usize = 1024;
const MAX_DEVICE_LINK_WIDE_CALLBACK_TOTAL_BLOCKS: usize =
    MAX_DEVICE_LINK_WIDE_CALLBACK_BLOCKS * MAX_DEVICE_LINK_WIDE_CALLBACK_ARRAY_EXTENT as usize;
const MAX_DEVICE_LINK_WIDE_CALLBACK_TOTAL_STATEMENTS: usize =
    MAX_DEVICE_LINK_WIDE_CALLBACK_STATEMENTS * MAX_DEVICE_LINK_WIDE_CALLBACK_ARRAY_EXTENT as usize;
const MAX_DEVICE_LINK_CLOSURE_CAPTURE_BYTES: u64 = 256;
const MAX_DEVICE_LINK_ARRAY_EXTENT: usize = 128;
const MAX_DEVICE_LINK_ARRAY_OUTPUT_BYTES: u64 = 32 * 1024;
// Full unrolling may duplicate the transitive always-inline callback body.
// Eight copies is the accepted bounded code-growth policy for this targeted
// array-builder optimization.
const MAX_DEFERRED_FULL_UNROLL_ARRAY_EXTENT: usize = 8;
// A fixed array borrowed by or created inside an always-inline helper cannot
// stay in SSA while a lowered iterator loop indexes it dynamically. Preserve
// full-unroll intent until the helper is inlined and LLVM can see the concrete
// trip count.
const MAX_DEFERRED_FIXED_ARRAY_EXTENT: u64 = 16;

fn build_device_inline_plan<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
) -> DeviceInlinePlan<'tcx> {
    let trace_plan = std::env::var_os("CUDA_OXIDE_INLINE_TRACE").is_some();
    let mut plan = DeviceInlinePlan::default();
    let mut accepted_builders = Vec::new();
    let mut accepted_closures = HashSet::new();
    let mut rejected_closures = HashSet::new();
    let mut accepted_deferred_unroll_closures = HashSet::new();
    let mut rejected_deferred_unroll_closures = HashSet::new();
    for function in functions {
        let instance = function.instance;
        let def_id = instance.def_id();
        let path = tcx.def_path_str(def_id);
        if !tcx.is_mir_available(def_id)
            || !is_core_def(tcx, def_id)
            || !is_concrete_array_builder_path(&path)
        {
            continue;
        }
        plan.considered_array_builders += 1;
        let closures = closure_instances_in_args(tcx, instance);
        if closures.len() != 1 {
            rejected_closures.extend(closures.iter().copied());
            rejected_deferred_unroll_closures.extend(closures.iter().copied());
            plan.rejected_array_builders += 1;
            if trace_plan {
                trace_array_inline_seed(tcx, instance, &path, &closures, false);
            }
            continue;
        }
        let closure = *closures.iter().next().expect("checked exact closure count");
        let array_extent = concrete_array_extent(tcx, instance);
        let accepted = is_bounded_inline_body(tcx, def_id)
            && array_extent.is_some_and(array_extent_within_budget)
            && array_output_layout_within_budget(tcx, instance)
            && array_extent.is_some_and(|extent| {
                is_bounded_array_callback_body(tcx, closure.def_id(), extent)
            })
            && closure_capture_layout_bytes(tcx, closure)
                .is_some_and(closure_capture_size_within_budget);
        if trace_plan {
            trace_array_inline_seed(tcx, instance, &path, &closures, accepted);
        }
        if accepted {
            accepted_builders.push((instance, closure));
            accepted_closures.insert(closure);
        } else {
            rejected_closures.insert(closure);
            plan.rejected_array_builders += 1;
        }
        if accepted
            && array_extent.is_some_and(|extent| {
                array_extent_within_limit(extent, MAX_DEFERRED_FULL_UNROLL_ARRAY_EXTENT)
            })
        {
            accepted_deferred_unroll_closures.insert(closure);
        } else {
            rejected_deferred_unroll_closures.insert(closure);
        }
    }
    plan.conflicting_array_closures =
        retain_uncontested_closures(&mut accepted_closures, &rejected_closures);
    plan.conflicting_deferred_unroll_closures = retain_uncontested_closures(
        &mut accepted_deferred_unroll_closures,
        &rejected_deferred_unroll_closures,
    );
    plan.array_builders = accepted_builders
        .into_iter()
        .filter_map(|(builder, closure)| accepted_closures.contains(&closure).then_some(builder))
        .collect();
    plan.array_closures = accepted_closures;
    plan.deferred_full_unroll_helpers = functions
        .iter()
        .map(|function| function.instance)
        .filter(|instance| {
            let def_id = instance.def_id();
            if !is_core_def(tcx, def_id)
                || !is_erased_array_builder_helper_path(&tcx.def_path_str(def_id))
            {
                return false;
            }
            let closures = closure_instances_anywhere_in_args(tcx, *instance);
            closures.len() == 1
                && closures
                    .iter()
                    .all(|closure| accepted_deferred_unroll_closures.contains(closure))
        })
        .collect();
    plan.deferred_full_unroll_helpers.extend(
        functions
            .iter()
            .filter(|function| !function.is_kernel)
            .map(|function| function.instance)
            .filter(|instance| {
                let def_id = instance.def_id();
                matches!(
                    tcx.codegen_fn_attrs(def_id).inline,
                    rustc_hir::attrs::InlineAttr::Always
                        | rustc_hir::attrs::InlineAttr::Force { .. }
                ) && is_bounded_inline_body(tcx, def_id)
                    && (has_bounded_fixed_array_reference_argument(tcx, *instance)
                        || has_bounded_fixed_array_local(tcx, def_id))
            }),
    );
    plan.borrowed_kernel_closures = functions
        .iter()
        .map(|function| function.instance)
        .filter(|instance| {
            is_direct_kernel_closure(tcx, *instance)
                && closure_has_borrowed_capture(*instance)
                && !rejected_closures.contains(instance)
        })
        .collect();
    plan.borrowed_always_inline_helper_closures = functions
        .iter()
        .map(|function| function.instance)
        .filter(|instance| is_bounded_borrowed_closure_in_always_inline_helper(tcx, *instance))
        .collect();
    plan
}

/// Return whether this is a closure written directly inside a kernel.
///
/// A directly nested closure is private to that entry and cannot gain code
/// reuse across kernels by remaining out of line. Its MIR can be tiny even
/// when a transitively inlined callee makes the LLVM body exceed heuristic
/// inline budgets, so source-body size is not a reliable guard here. An
/// explicit `#[inline(never)]` remains authoritative.
fn is_direct_kernel_closure(tcx: TyCtxt<'_>, instance: Instance<'_>) -> bool {
    let def_id = instance.def_id();
    tcx.def_kind(def_id) == rustc_hir::def::DefKind::Closure
        && !tcx.is_coroutine(def_id)
        && tcx
            .opt_parent(def_id)
            .is_some_and(|parent| crate::collector::is_kernel_function(tcx, parent))
        && !matches!(
            tcx.codegen_fn_attrs(def_id).inline,
            rustc_hir::attrs::InlineAttr::Never
        )
}

/// Return whether a closure carries any capture by Rust reference.
///
/// Calling such a closure out of line makes its caller materialize borrowed
/// SSA values in an addressable frame. This is especially costly at a kernel
/// boundary, where a large by-value launch aggregate otherwise stays in
/// parameter space and registers.
fn closure_has_borrowed_capture(closure: Instance<'_>) -> bool {
    let captures = closure.args.as_closure().tupled_upvars_ty();
    let TyKind::Tuple(captures) = captures.kind() else {
        return false;
    };
    captures
        .iter()
        .any(|capture| matches!(capture.kind(), TyKind::Ref(..)))
}

/// Return whether a bounded borrowed closure is lexically nested in a helper
/// whose source contract already requires inlining.
///
/// Trait default methods can invoke a caller-provided closure several times
/// after the enclosing always-inline helper has been expanded. If that closure
/// keeps ordinary heuristic inline intent, aggregate captures must be
/// materialized solely to cross the remaining helper boundary. Promoting only
/// bounded borrowed closures under an explicit always-inline parent preserves
/// the parent's source intent without applying a module-wide closure policy.
fn is_bounded_borrowed_closure_in_always_inline_helper<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    let def_id = instance.def_id();
    if tcx.def_kind(def_id) != rustc_hir::def::DefKind::Closure
        || tcx.is_coroutine(def_id)
        || matches!(
            tcx.codegen_fn_attrs(def_id).inline,
            rustc_hir::attrs::InlineAttr::Never
        )
        || !closure_has_borrowed_capture(instance)
        || !is_bounded_inline_body(tcx, def_id)
        || !closure_capture_layout_bytes(tcx, instance)
            .is_some_and(closure_capture_size_within_budget)
    {
        return false;
    }

    tcx.opt_parent(def_id).is_some_and(|parent| {
        !crate::collector::is_kernel_function(tcx, parent)
            && matches!(
                tcx.codegen_fn_attrs(parent).inline,
                rustc_hir::attrs::InlineAttr::Always | rustc_hir::attrs::InlineAttr::Force { .. }
            )
    })
}

fn retain_uncontested_closures<T: Eq + Hash>(
    accepted: &mut HashSet<T>,
    rejected: &HashSet<T>,
) -> usize {
    let conflicts = accepted.intersection(rejected).count();
    accepted.retain(|closure| !rejected.contains(closure));
    conflicts
}

fn trace_array_inline_seed<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    path: &str,
    closures: &HashSet<Instance<'tcx>>,
    accepted: bool,
) {
    let (closure_path, capture_bytes) = closures.iter().next().copied().map_or_else(
        || ("<none>".to_string(), None),
        |closure| {
            (
                tcx.def_path_str(closure.def_id()),
                closure_capture_layout_bytes(tcx, closure),
            )
        },
    );
    let builder_body = inline_body_size(tcx, instance.def_id());
    let closure_body = closures
        .iter()
        .next()
        .map(|closure| inline_body_size(tcx, closure.def_id()));
    eprintln!(
        "[rustc_codegen_cuda] array callback inline candidate: builder={path} \
         builder_symbol={} accepted={accepted} \
         closures={} closure={closure_path} \
         array_extent={:?} \
         builder_body={builder_body:?} closure_body={closure_body:?} \
         capture_bytes={capture_bytes:?} output_bytes={:?}",
        tcx.symbol_name(instance).name,
        closures.len(),
        concrete_array_extent(tcx, instance),
        array_output_layout_bytes(tcx, instance)
    );
}

fn closure_capture_layout_bytes<'tcx>(tcx: TyCtxt<'tcx>, closure: Instance<'tcx>) -> Option<u64> {
    let captures = closure.args.as_closure().tupled_upvars_ty();
    tcx.layout_of(TypingEnv::fully_monomorphized().as_query_input(captures))
        .ok()
        .map(|layout| layout.size.bytes())
}

fn closure_capture_size_within_budget(bytes: u64) -> bool {
    bytes <= MAX_DEVICE_LINK_CLOSURE_CAPTURE_BYTES
}

fn closure_instances_in_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> HashSet<Instance<'tcx>> {
    let path = tcx.def_path_str(instance.def_id());
    let generic_types: Vec<_> = instance
        .args
        .iter()
        .filter_map(|argument| argument.as_type())
        .collect();
    let Some(root) = array_callable_type(&path, &generic_types) else {
        return HashSet::new();
    };
    outermost_closure_instances(tcx, closure_instances_in_type(tcx, *root))
}

fn array_callable_type<'a, T>(path: &str, generic_types: &'a [T]) -> Option<&'a T> {
    // Both concrete builders carry their callable F as the final type
    // parameter. The recursive type walk below finds callbacks nested in
    // core's `Wrapped` and array-drain adapters.
    is_concrete_array_builder_path(path)
        .then(|| generic_types.last())
        .flatten()
}

fn closure_instances_in_type<'tcx>(tcx: TyCtxt<'tcx>, root: Ty<'tcx>) -> HashSet<Instance<'tcx>> {
    let mut closures = HashSet::new();
    collect_closure_instances_in_type(tcx, root, &mut closures);
    closures
}

fn collect_closure_instances_in_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    callable: Ty<'tcx>,
    closures: &mut HashSet<Instance<'tcx>>,
) {
    match callable.kind() {
        TyKind::Closure(closure_def_id, closure_args) => {
            if let Some(closure) = Instance::try_resolve(
                tcx,
                TypingEnv::fully_monomorphized(),
                *closure_def_id,
                closure_args,
            )
            .ok()
            .flatten()
            .filter(|closure| tcx.is_mir_available(closure.def_id()))
            {
                closures.insert(closure);
            }
        }
        TyKind::RawPtr(pointee, _) | TyKind::Ref(_, pointee, _) => {
            collect_closure_instances_in_type(tcx, *pointee, closures);
        }
        TyKind::Array(element, _) | TyKind::Slice(element) => {
            collect_closure_instances_in_type(tcx, *element, closures);
        }
        TyKind::Tuple(elements) => {
            for element in elements.iter() {
                collect_closure_instances_in_type(tcx, element, closures);
            }
        }
        TyKind::Adt(_, arguments) => {
            for nested in arguments.iter().filter_map(|argument| argument.as_type()) {
                collect_closure_instances_in_type(tcx, nested, closures);
            }
        }
        _ => {}
    }
}

fn closure_instances_anywhere_in_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> HashSet<Instance<'tcx>> {
    let mut closures = HashSet::new();
    for ty in instance
        .args
        .iter()
        .filter_map(|argument| argument.as_type())
    {
        collect_closure_instances_in_type(tcx, ty, &mut closures);
    }
    outermost_closure_instances(tcx, closures)
}

fn outermost_closure_instances<'tcx>(
    tcx: TyCtxt<'tcx>,
    closures: HashSet<Instance<'tcx>>,
) -> HashSet<Instance<'tcx>> {
    let all = closures.iter().copied().collect::<Vec<_>>();
    closures
        .into_iter()
        .filter(|candidate| {
            !all.iter().any(|ancestor| {
                ancestor != candidate && closure_is_nested_under(tcx, *candidate, ancestor.def_id())
            })
        })
        .collect()
}

fn closure_is_nested_under(
    tcx: TyCtxt<'_>,
    closure: Instance<'_>,
    ancestor: rustc_hir::def_id::DefId,
) -> bool {
    let mut parent = tcx.opt_parent(closure.def_id());
    while let Some(def_id) = parent {
        if def_id == ancestor {
            return true;
        }
        parent = tcx.opt_parent(def_id);
    }
    false
}

fn is_core_def(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    tcx.lang_items()
        .add_trait()
        .is_some_and(|core_item| core_item.krate == def_id.krate)
}

fn concrete_array_extent<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<u64> {
    let mut extent = None;
    for arg in instance.args.iter() {
        let Some(value) = arg.as_const() else {
            continue;
        };
        let value = value.try_to_target_usize(tcx)?;
        if extent.replace(value).is_some() {
            return None;
        };
    }
    extent
}

fn array_output_layout_within_budget<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    array_output_layout_bytes(tcx, instance).is_some_and(array_output_size_within_budget)
}

fn has_bounded_fixed_array_reference_argument<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    let signature = tcx
        .fn_sig(instance.def_id())
        .instantiate(tcx, instance.args)
        .skip_binder();
    signature.inputs().iter().any(|input| {
        let TyKind::Ref(_, pointee, _) = input.kind() else {
            return false;
        };
        let TyKind::Array(_, extent) = pointee.kind() else {
            return false;
        };
        extent
            .try_to_target_usize(tcx)
            .is_some_and(|extent| extent <= MAX_DEFERRED_FIXED_ARRAY_EXTENT)
    })
}

fn has_bounded_fixed_array_local(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    let body = tcx.optimized_mir(def_id);
    body.local_decls
        .iter()
        .skip(body.arg_count + 1)
        .any(|local| {
            let TyKind::Array(_, extent) = local.ty.kind() else {
                return false;
            };
            extent
                .try_to_target_usize(tcx)
                .is_some_and(|extent| extent <= MAX_DEFERRED_FIXED_ARRAY_EXTENT)
        })
}

fn array_output_layout_bytes<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<u64> {
    let signature = tcx
        .fn_sig(instance.def_id())
        .instantiate(tcx, instance.args);
    let output = signature.skip_binder().output();
    tcx.layout_of(TypingEnv::fully_monomorphized().as_query_input(output))
        .ok()
        .map(|layout| layout.size.bytes())
}

fn array_output_size_within_budget(bytes: u64) -> bool {
    bytes <= MAX_DEVICE_LINK_ARRAY_OUTPUT_BYTES
}

fn array_extent_within_budget(extent: u64) -> bool {
    extent <= MAX_DEVICE_LINK_ARRAY_EXTENT as u64
}

fn array_extent_within_limit(extent: u64, max_extent: usize) -> bool {
    extent <= max_extent as u64
}

fn is_bounded_inline_body(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    let (basic_blocks, statements) = inline_body_size(tcx, def_id);
    inline_body_within_budget(basic_blocks, statements)
}

fn is_bounded_array_callback_body(
    tcx: TyCtxt<'_>,
    def_id: rustc_hir::def_id::DefId,
    array_extent: u64,
) -> bool {
    let (basic_blocks, statements) = inline_body_size(tcx, def_id);
    array_callback_body_within_budget(basic_blocks, statements, array_extent)
}

fn array_callback_body_within_budget(
    basic_blocks: usize,
    statements: usize,
    array_extent: u64,
) -> bool {
    if inline_body_within_budget(basic_blocks, statements) {
        return true;
    }
    if !(1..=MAX_DEVICE_LINK_WIDE_CALLBACK_ARRAY_EXTENT).contains(&array_extent)
        || basic_blocks > MAX_DEVICE_LINK_WIDE_CALLBACK_BLOCKS
        || statements > MAX_DEVICE_LINK_WIDE_CALLBACK_STATEMENTS
    {
        return false;
    }
    let array_extent = array_extent as usize;
    basic_blocks
        .checked_mul(array_extent)
        .is_some_and(|total| total <= MAX_DEVICE_LINK_WIDE_CALLBACK_TOTAL_BLOCKS)
        && statements
            .checked_mul(array_extent)
            .is_some_and(|total| total <= MAX_DEVICE_LINK_WIDE_CALLBACK_TOTAL_STATEMENTS)
}

fn inline_body_size(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> (usize, usize) {
    let body = tcx.optimized_mir(def_id);
    let statement_count = body
        .basic_blocks
        .iter()
        .map(|block| block.statements.len())
        .sum();
    (body.basic_blocks.len(), statement_count)
}

fn inline_body_within_budget(basic_blocks: usize, statements: usize) -> bool {
    basic_blocks <= MAX_DEVICE_LINK_INLINE_BLOCKS && statements <= MAX_DEVICE_LINK_INLINE_STATEMENTS
}

fn is_concrete_array_builder_path(path: &str) -> bool {
    matches!(
        path,
        "core::array::from_fn"
            | "std::array::from_fn"
            | "core::array::try_from_fn"
            | "std::array::try_from_fn"
    )
}

fn is_erased_array_builder_helper_path(path: &str) -> bool {
    matches!(
        path,
        "core::array::try_from_fn_erased" | "std::array::try_from_fn_erased"
    )
}

#[cfg(test)]
mod inline_plan_tests {
    use super::{
        array_callable_type, array_callback_body_within_budget, array_extent_within_budget,
        array_extent_within_limit, array_output_size_within_budget, inline_body_within_budget,
        is_concrete_array_builder_path, is_erased_array_builder_helper_path,
        retain_uncontested_closures,
    };
    use std::collections::HashSet;

    #[test]
    fn recognizes_only_concrete_core_array_builders() {
        for path in [
            "core::array::from_fn",
            "std::array::from_fn",
            "core::array::try_from_fn",
            "std::array::try_from_fn",
        ] {
            assert!(
                is_concrete_array_builder_path(path),
                "expected concrete array builder: {path}"
            );
        }

        for path in [
            "core::array::try_from_fn_erased",
            "core::array::from_ref",
            "core::array::iter",
            "user_crate::array::from_fn",
            "user_crate::std::array::try_from_fn",
            "<F as core::ops::FnMut<(A,)>>::call_mut",
            "<core::ops::try_trait::Wrapped<T, A, F> as core::ops::FnMut<(A,)>>::call_mut",
        ] {
            assert!(
                !is_concrete_array_builder_path(path),
                "non-root helper must not seed callback promotion: {path}"
            );
        }
    }

    #[test]
    fn recognizes_only_the_erased_core_array_builder_helper() {
        for path in [
            "core::array::try_from_fn_erased",
            "std::array::try_from_fn_erased",
        ] {
            assert!(is_erased_array_builder_helper_path(path));
        }
        for path in [
            "core::array::try_from_fn",
            "core::array::from_fn",
            "user_crate::array::try_from_fn_erased",
        ] {
            assert!(!is_erased_array_builder_helper_path(path));
        }
    }

    #[test]
    fn enforces_both_inline_body_limits() {
        assert!(inline_body_within_budget(24, 128));
        assert!(!inline_body_within_budget(25, 128));
        assert!(!inline_body_within_budget(24, 129));
    }

    #[test]
    fn bounds_wide_callbacks_by_their_three_element_trip_count() {
        assert!(array_callback_body_within_budget(24, 128, 128));
        assert!(array_callback_body_within_budget(66, 751, 3));
        assert!(!array_callback_body_within_budget(66, 751, 4));
        assert!(!array_callback_body_within_budget(97, 751, 3));
        assert!(!array_callback_body_within_budget(66, 1025, 3));
        assert!(!array_callback_body_within_budget(usize::MAX, 1, 3));
        assert!(!array_callback_body_within_budget(1, usize::MAX, 3));
    }

    #[test]
    fn bounds_concrete_array_extents() {
        assert!(array_extent_within_budget(0));
        assert!(array_extent_within_budget(128));
        assert!(!array_extent_within_budget(129));
        assert!(!array_extent_within_budget(u64::MAX));
        assert!(array_extent_within_limit(8, 8));
        assert!(!array_extent_within_limit(9, 8));
        assert!(array_output_size_within_budget(32 * 1024));
        assert!(!array_output_size_within_budget(32 * 1024 + 1));
        assert!(!array_output_size_within_budget(64 * 1024));
    }

    #[test]
    fn selects_only_the_concrete_builder_callable_type() {
        let generic_types = ["result_closure", "callable"];
        assert_eq!(
            array_callable_type("core::array::try_from_fn", &generic_types),
            Some(&"callable"),
            "a closure-valued result type must not be selected"
        );

        assert_eq!(
            array_callable_type("core::array::try_from_fn_erased", &generic_types),
            None,
            "implementation scaffolds must not seed promotion"
        );
    }

    #[test]
    fn a_rejected_use_wins_for_a_shared_callback() {
        let mut accepted = HashSet::from(["accepted_only", "shared"]);
        let rejected = HashSet::from(["rejected_only", "shared"]);

        assert_eq!(
            retain_uncontested_closures(&mut accepted, &rejected),
            1,
            "the shared callback is the single conflict"
        );
        assert_eq!(accepted, HashSet::from(["accepted_only"]));
    }
}

fn is_tiny_inline_helper(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    if !tcx.is_mir_available(def_id) {
        return false;
    }

    // libNVVM's bounded -opt=1 pipeline often leaves even explicitly inlined
    // generic leaf helpers as device calls. Requiring inlining only for very
    // small MIR bodies preserves vectorized loads and scalar shims without the
    // compile-time explosion caused by forcing every `#[inline(always)]` body.
    let body = tcx.optimized_mir(def_id);
    body.basic_blocks.len() <= 2
        && body
            .basic_blocks
            .iter()
            .map(|block| block.statements.len())
            .sum::<usize>()
            <= 16
}

fn is_primitive_float_operator(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    let Some(impl_def_id) = tcx.trait_impl_of_assoc(def_id) else {
        return false;
    };
    let trait_ref = tcx.impl_trait_ref(impl_def_id).instantiate_identity();
    if !matches!(trait_ref.self_ty().kind(), TyKind::Float(_)) {
        return false;
    }

    let lang = tcx.lang_items();
    [
        lang.add_trait(),
        lang.sub_trait(),
        lang.mul_trait(),
        lang.div_trait(),
        lang.rem_trait(),
        lang.neg_trait(),
        lang.add_assign_trait(),
        lang.sub_assign_trait(),
        lang.mul_assign_trait(),
        lang.div_assign_trait(),
        lang.rem_assign_trait(),
    ]
    .into_iter()
    .flatten()
    .any(|operator_trait| operator_trait == trait_ref.def_id)
}

/// Convert a Rust device-extern type to the LLVM type supported at the
/// external function boundary.
///
/// Raw-pointer pointees are preserved recursively. Unsupported C ABI types
/// return an error instead of being treated as an arbitrary pointer.
///
/// Integer types smaller than 32 bits keep their NARROW IR type (`i8`,
/// `i16`, `i1` for `bool`) and carry a `signext`/`zeroext` ABI attribute,
/// exactly like clang's NVPTXABIInfo and rustc's nvptx64 callconv; the NVPTX
/// backend performs the `.param.b32` widening. This keeps the emitted
/// `declare` byte-for-byte compatible with clang/nvcc-compiled LTOIR
/// definitions and lets cuda-oxide's own narrow SSA values flow into the
/// call with no inserted conversions. `f16` is passed as `half` directly
/// since NVPTX has native f16 support.
fn rustc_ty_to_device_extern_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    position: DeviceExternTypePosition,
) -> Result<mir_importer::DeviceExternType, String> {
    use mir_importer::DeviceExternType as E;

    if ty.is_c_void(tcx) {
        return if position == DeviceExternTypePosition::Pointee {
            // LLVM spells a C void pointer as `i8*` in typed-pointer IR.
            Ok(E::Integer(8))
        } else {
            Err("`c_void` is only supported behind a pointer".to_string())
        };
    }

    // For pointer pointees, all integer widths are valid without extension.
    // For by-value parameters/returns, sub-32-bit integers keep their narrow
    // type and gain the sign/zero extension ABI attribute at that width.
    let signed_integer = |bits: u32| {
        if position == DeviceExternTypePosition::Pointee || matches!(bits, 32 | 64) {
            Ok(E::Integer(bits))
        } else if matches!(bits, 8 | 16) {
            // NVPTX ABI: narrow type with signext (clang NVPTXABIInfo shape).
            Ok(E::SignExtInteger(bits))
        } else {
            Err(format!(
                "`i{bits}` is not supported by value in a device extern; use i32/i64 or pass a pointer"
            ))
        }
    };

    let unsigned_integer = |bits: u32| {
        if position == DeviceExternTypePosition::Pointee || matches!(bits, 32 | 64) {
            Ok(E::Integer(bits))
        } else if matches!(bits, 8 | 16) {
            // NVPTX ABI: narrow type with zeroext (clang NVPTXABIInfo shape).
            Ok(E::ZeroExtInteger(bits))
        } else {
            Err(format!(
                "`u{bits}` is not supported by value in a device extern; use u32/u64 or pass a pointer"
            ))
        }
    };

    match ty.kind() {
        TyKind::Int(int_ty) => match int_ty {
            rustc_middle::ty::IntTy::I8 => signed_integer(8),
            rustc_middle::ty::IntTy::I16 => signed_integer(16),
            rustc_middle::ty::IntTy::I32 => signed_integer(32),
            rustc_middle::ty::IntTy::I64 => signed_integer(64),
            rustc_middle::ty::IntTy::I128 => signed_integer(128),
            rustc_middle::ty::IntTy::Isize => signed_integer(64), // nvptx64
        },
        TyKind::Uint(uint_ty) => match uint_ty {
            rustc_middle::ty::UintTy::U8 => unsigned_integer(8),
            rustc_middle::ty::UintTy::U16 => unsigned_integer(16),
            rustc_middle::ty::UintTy::U32 => unsigned_integer(32),
            rustc_middle::ty::UintTy::U64 => unsigned_integer(64),
            rustc_middle::ty::UintTy::U128 => unsigned_integer(128),
            rustc_middle::ty::UintTy::Usize => unsigned_integer(64), // nvptx64
        },
        TyKind::Float(float_ty) => match float_ty {
            // NVPTX supports native f16 (LLVM `half`) in all positions.
            rustc_middle::ty::FloatTy::F16 => Ok(E::Float16),
            rustc_middle::ty::FloatTy::F32 => Ok(E::Float32),
            rustc_middle::ty::FloatTy::F64 => Ok(E::Float64),
            rustc_middle::ty::FloatTy::F128 => {
                Err("f128 device externs are not supported".to_string())
            }
        },
        TyKind::RawPtr(pointee, _) | TyKind::Ref(_, pointee, _) => {
            let pointee = if matches!(pointee.kind(), TyKind::Tuple(fields) if fields.is_empty()) {
                // Rust's `*mut ()` is its common spelling for a void pointer.
                E::Integer(8)
            } else {
                rustc_ty_to_device_extern_type(
                    tcx,
                    *pointee,
                    DeviceExternTypePosition::Pointee,
                )?
            };
            Ok(E::pointer_to(pointee, 0))
        }
        TyKind::Array(element, len) if position == DeviceExternTypePosition::Pointee => {
            let len = len.try_to_target_usize(tcx).ok_or_else(|| {
                format!("device-extern array length for `{ty}` is not a concrete constant")
            })?;
            let element = rustc_ty_to_device_extern_type(
                tcx,
                *element,
                DeviceExternTypePosition::Pointee,
            )?;
            Ok(E::Array {
                element: Box::new(element),
                len,
            })
        }
        TyKind::Tuple(fields)
            if fields.is_empty() && position == DeviceExternTypePosition::Result =>
        {
            Ok(E::Void)
        }
        TyKind::Bool => {
            if position == DeviceExternTypePosition::Pointee {
                // Behind a pointer, bool is just i8 (Rust's bool is 1 byte).
                Ok(E::Integer(8))
            } else {
                // NVPTX ABI: bool stays i1 with zeroext, matching both
                // clang's `zeroext i1` and cuda-oxide's own i1 SSA values.
                Ok(E::ZeroExtInteger(1))
            }
        }
        TyKind::Char => Err(
            "Rust `char` is not supported in device extern signatures; use `u32` in a C-compatible wrapper".to_string(),
        ),
        TyKind::Never => Err("never-returning device externs are not yet supported".to_string()),
        _ => Err(format!(
            "unsupported device-extern ABI type `{ty}`; use scalar C types or raw pointers to supported scalar/array pointees"
        )),
    }
}

/// Result of device code generation.
///
/// Contains paths to generated artifacts and the payload selected for
/// embedding in the host binary.
pub struct DeviceCodegenResult {
    /// Path to generated PTX assembly file.
    ///
    /// In NVVM IR modes this is the would-be PTX path and may not exist.
    pub ptx_path: PathBuf,
    /// Path to generated LLVM IR file.
    pub ll_path: PathBuf,
    /// Explicit owner-level PTX bundle path for a partitioned materialization.
    pub ptx_bundle_path: Option<PathBuf>,
    /// GPU target architecture used (e.g., "sm_80", "sm_90a", "sm_100a").
    ///
    /// Auto-detected based on GPU features used, or overridden via
    /// `CUDA_OXIDE_TARGET` environment variable.
    pub target: String,
    /// PTX content as a string, ready for embedding in the host binary.
    ///
    /// NVVM IR / LTOIR flows intentionally skip PTX generation.
    pub ptx_content: Option<String>,
    /// Ordered device artifact partitions selected for finalization and
    /// embedding. Ordinary and small-owner builds contain at most one item.
    /// Large AOT owners contain several PTX and/or NVVM IR inputs that must be
    /// linked, in this order, into one final cubin.
    pub artifacts: Vec<DeviceCodegenArtifact>,
    /// Whether later compilation stages may contract ordinary floating-point
    /// multiply/add expressions.
    pub allow_fma_contraction: bool,
    /// Debug policy used when exporting the NVVM IR. Later materialization
    /// stages must preserve this policy instead of silently compiling with
    /// their own defaults.
    pub debug_kind: llvm_export::export::DebugKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCodegenArtifactKind {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

pub struct DeviceCodegenArtifact {
    pub kind: DeviceCodegenArtifactKind,
    pub name: String,
    /// On-disk compiler artifact. Partition materialization consumes these
    /// paths sequentially instead of retaining every large buffer.
    pub path: PathBuf,
    /// Exact PTX path used directly or populated by bounded NVVM compilation.
    pub ptx_sidecar_path: PathBuf,
    /// In-memory payload for the unchanged single-module path.
    pub bytes: Option<Vec<u8>>,
}

/// Configuration for device codegen.
///
/// Controls output paths and diagnostic output during compilation.
pub struct DeviceCodegenConfig {
    /// Output directory for generated files (.ll, .ptx).
    pub output_dir: PathBuf,
    /// Base name for output files (e.g., "kernel" → kernel.ll, kernel.ptx).
    pub output_name: String,
    /// Print verbose progress to stderr.
    pub verbose: bool,
    /// Dump raw rustc MIR before translation.
    pub dump_rustc_mir: bool,
    /// Dump the `dialect-mir` module during compilation.
    pub dump_mir_dialect: bool,
    /// Dump the LLVM dialect module during compilation.
    pub dump_llvm_dialect: bool,
    /// Partition large owner closures before LLVM/libNVVM optimization.
    ///
    /// This is normally enabled by build-time cubin materialization. An
    /// explicitly selected monolithic owner stays on the LLVM O3 path.
    pub partition_large_owner: bool,
}

impl Default for DeviceCodegenConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            output_name: "kernel".to_string(),
            verbose: false,
            dump_rustc_mir: false,
            dump_mir_dialect: false,
            dump_llvm_dialect: false,
            partition_large_owner: false,
        }
    }
}

const OWNER_PARTITION_TARGET_WEIGHT: usize = 12 * 1024 * 1024;
const OWNER_PARTITION_MAX_WEIGHT: usize = OWNER_PARTITION_TARGET_WEIGHT * 6 / 5;

#[derive(Clone, Copy)]
struct OwnerPartitionPolicy {
    target_weight: usize,
    max_weight: usize,
}

impl Default for OwnerPartitionPolicy {
    fn default() -> Self {
        Self {
            target_weight: OWNER_PARTITION_TARGET_WEIGHT,
            max_weight: OWNER_PARTITION_MAX_WEIGHT,
        }
    }
}

/// Path-independent structural proxy for the amount of IR a MIR body tends to
/// produce. Deliberately use only collection sizes, never MIR debug text,
/// spans, source paths, or traversal order: those would make a partition
/// boundary depend on the checkout or diagnostic formatting.
fn structural_mir_weight(mir: &rustc_middle::mir::Body<'_>) -> usize {
    const BASE: usize = 512;
    const LOCAL: usize = 96;
    const SOURCE_SCOPE: usize = 48;
    const DEBUG_VALUE: usize = 48;
    const BASIC_BLOCK: usize = 192;
    const STATEMENT: usize = 256;
    const SUCCESSOR_EDGE: usize = 32;

    let blocks = mir.basic_blocks.len();
    let statements = mir
        .basic_blocks
        .iter()
        .map(|block| block.statements.len())
        .sum::<usize>();
    let successor_edges = mir
        .basic_blocks
        .iter()
        .map(|block| block.terminator().successors().count())
        .sum::<usize>();

    [
        BASE,
        mir.local_decls.len().saturating_mul(LOCAL),
        mir.source_scopes.len().saturating_mul(SOURCE_SCOPE),
        mir.var_debug_info.len().saturating_mul(DEBUG_VALUE),
        blocks.saturating_mul(BASIC_BLOCK),
        statements.saturating_mul(STATEMENT),
        successor_edges.saturating_mul(SUCCESSOR_EDGE),
    ]
    .into_iter()
    .fold(0usize, usize::saturating_add)
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_ascii_whitespace().nth(1)?.parse().ok()
}

struct OwnerPartitionOutputGuard {
    hidden_root: PathBuf,
    path: PathBuf,
    keep: bool,
}

impl OwnerPartitionOutputGuard {
    fn prepare(
        output_dir: &Path,
        output_name: &str,
        partitioned_owner: bool,
    ) -> std::io::Result<Option<Self>> {
        let mut components = Path::new(output_name).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("owner output name is not one path component: {output_name:?}"),
            ));
        }

        let hidden_root = output_dir.join(".cuda-oxide-partitions");
        match std::fs::symlink_metadata(&hidden_root) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "owner partition root is not a regular directory: {}",
                        hidden_root.display()
                    ),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir_all(&hidden_root)?;
                let metadata = std::fs::symlink_metadata(&hidden_root)?;
                if !metadata.is_dir() || metadata.file_type().is_symlink() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "owner partition root became unsafe while creating it: {}",
                            hidden_root.display()
                        ),
                    ));
                }
            }
            Err(error) => return Err(error),
        }
        let path = hidden_root.join(output_name);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                std::fs::remove_dir_all(&path)?;
            }
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "owner partition output is not a regular directory: {}",
                        path.display()
                    ),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }

        if !partitioned_owner {
            return Ok(None);
        }
        std::fs::create_dir(&path)?;
        Ok(Some(Self {
            hidden_root,
            path,
            keep: false,
        }))
    }

    fn keep(&mut self) {
        self.keep = true;
    }
}

impl Drop for OwnerPartitionOutputGuard {
    fn drop(&mut self) {
        let root_is_safe = std::fs::symlink_metadata(&self.hidden_root)
            .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink());
        let path_is_safe = std::fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink());
        if !self.keep && root_is_safe && path_is_safe {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OwnerPartitionPlan {
    plan_id: String,
    root_symbols: Vec<String>,
    function_symbols: Vec<String>,
    estimated_mir_weight: usize,
    exceeds_max_weight: bool,
}

#[derive(Default)]
struct WorkingOwnerPartition {
    root_symbols: BTreeSet<String>,
    function_symbols: BTreeSet<String>,
    estimated_mir_weight: usize,
}

fn plan_owner_partitions(
    function_mir_weights: &BTreeMap<String, usize>,
    families: &[DeviceFunctionFamily],
    policy: OwnerPartitionPolicy,
) -> Vec<OwnerPartitionPlan> {
    let all_function_symbols = function_mir_weights.keys().cloned().collect::<Vec<_>>();
    let all_estimated_mir_weight = function_mir_weights.values().copied().sum::<usize>();
    if all_estimated_mir_weight <= policy.max_weight || families.len() <= 1 {
        return vec![finalize_owner_partition(
            families
                .iter()
                .map(|family| family.root_symbol.clone())
                .collect(),
            all_function_symbols,
            function_mir_weights,
            policy.max_weight,
        )];
    }

    let canonical_families = families
        .iter()
        .map(|family| {
            let function_symbols = family
                .function_symbols
                .iter()
                .filter(|symbol| function_mir_weights.contains_key(*symbol))
                .cloned()
                .collect::<BTreeSet<_>>();
            let estimated_mir_weight = function_symbols
                .iter()
                .map(|symbol| function_mir_weights[symbol])
                .sum::<usize>();
            (
                family.root_symbol.clone(),
                function_symbols,
                estimated_mir_weight,
            )
        })
        .collect::<Vec<_>>();

    // A root may itself be reachable from another root. Roots are emitted as
    // strong entry definitions, so overlapping root families must remain in
    // one indivisible component. Ordinary shared helpers may still be
    // duplicated across components because their linkage is coalescible or
    // module-private in partitioned owners.
    let root_owner = canonical_families
        .iter()
        .enumerate()
        .map(|(index, (root, _, _))| (root.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut family_parent = (0..canonical_families.len()).collect::<Vec<_>>();
    for (family_index, (_, function_symbols, _)) in canonical_families.iter().enumerate() {
        for function_symbol in function_symbols {
            if let Some(&root_index) = root_owner.get(function_symbol.as_str()) {
                union_family_components(&mut family_parent, family_index, root_index);
            }
        }
    }

    let mut components = BTreeMap::<usize, WorkingOwnerPartition>::new();
    for (family_index, (root_symbol, function_symbols, _)) in
        canonical_families.into_iter().enumerate()
    {
        let component_index = find_family_component(&mut family_parent, family_index);
        let component = components.entry(component_index).or_default();
        component.root_symbols.insert(root_symbol);
        component.function_symbols.extend(function_symbols);
    }
    for component in components.values_mut() {
        component.estimated_mir_weight = component
            .function_symbols
            .iter()
            .map(|symbol| function_mir_weights[symbol])
            .sum();
    }
    let mut canonical_components = components.into_values().collect::<Vec<_>>();
    canonical_components.sort_by(|left, right| {
        right
            .estimated_mir_weight
            .cmp(&left.estimated_mir_weight)
            .then_with(|| left.root_symbols.cmp(&right.root_symbols))
    });

    let mut partitions = Vec::<WorkingOwnerPartition>::new();
    for component in canonical_components {
        let candidate = partitions
            .iter()
            .enumerate()
            .filter_map(|(index, partition)| {
                let additional_bytes = component
                    .function_symbols
                    .iter()
                    .filter(|symbol| !partition.function_symbols.contains(*symbol))
                    .map(|symbol| function_mir_weights[symbol])
                    .sum::<usize>();
                let combined_bytes = partition.estimated_mir_weight + additional_bytes;
                (combined_bytes <= policy.max_weight).then_some((
                    combined_bytes.abs_diff(policy.target_weight),
                    index,
                    additional_bytes,
                ))
            })
            .min_by_key(|(distance, index, _)| (*distance, *index));

        let (partition, additional_bytes) = if let Some((_, index, additional_bytes)) = candidate {
            (&mut partitions[index], additional_bytes)
        } else {
            partitions.push(WorkingOwnerPartition::default());
            (
                partitions.last_mut().expect("just pushed a partition"),
                component.estimated_mir_weight,
            )
        };
        partition.root_symbols.extend(component.root_symbols);
        partition
            .function_symbols
            .extend(component.function_symbols);
        partition.estimated_mir_weight += additional_bytes;
    }

    // Collection should make every definition reachable from at least one
    // root. Preserve any future collector-only support definition anyway,
    // assigning it deterministically without changing the exported root set.
    let assigned = partitions
        .iter()
        .flat_map(|partition| partition.function_symbols.iter().cloned())
        .collect::<BTreeSet<_>>();
    for symbol in function_mir_weights
        .keys()
        .filter(|symbol| !assigned.contains(*symbol))
    {
        let bytes = function_mir_weights[symbol];
        let index = partitions
            .iter()
            .enumerate()
            .filter(|(_, partition)| partition.estimated_mir_weight + bytes <= policy.max_weight)
            .min_by_key(|(index, partition)| (partition.estimated_mir_weight, *index))
            .map(|(index, _)| index)
            .unwrap_or_else(|| {
                partitions.push(WorkingOwnerPartition::default());
                partitions.len() - 1
            });
        partitions[index].function_symbols.insert(symbol.clone());
        partitions[index].estimated_mir_weight += bytes;
    }

    let mut result = partitions
        .into_iter()
        .map(|partition| {
            finalize_owner_partition(
                partition.root_symbols.into_iter().collect(),
                partition.function_symbols.into_iter().collect(),
                function_mir_weights,
                policy.max_weight,
            )
        })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| left.plan_id.cmp(&right.plan_id));
    result
}

fn find_family_component(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        parent[index] = find_family_component(parent, parent[index]);
    }
    parent[index]
}

fn union_family_components(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find_family_component(parent, left);
    let right_root = find_family_component(parent, right);
    if left_root == right_root {
        return;
    }
    let (canonical, other) = if left_root < right_root {
        (left_root, right_root)
    } else {
        (right_root, left_root)
    };
    parent[other] = canonical;
}

fn finalize_owner_partition(
    mut root_symbols: Vec<String>,
    mut function_symbols: Vec<String>,
    function_mir_weights: &BTreeMap<String, usize>,
    max_weight: usize,
) -> OwnerPartitionPlan {
    root_symbols.sort();
    root_symbols.dedup();
    function_symbols.sort();
    function_symbols.dedup();
    let estimated_mir_weight = function_symbols
        .iter()
        .map(|symbol| {
            function_mir_weights
                .get(symbol)
                .copied()
                .unwrap_or_default()
        })
        .sum();
    let mut digest = Sha256::new();
    digest.update(b"cuda-oxide-owner-partition-v1\0");
    for root in &root_symbols {
        digest.update((root.len() as u64).to_le_bytes());
        digest.update(root.as_bytes());
    }
    digest.update(b"\0functions\0");
    for function in &function_symbols {
        digest.update((function.len() as u64).to_le_bytes());
        digest.update(function.as_bytes());
    }
    let digest = digest.finalize();
    let plan_id = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    OwnerPartitionPlan {
        plan_id,
        root_symbols,
        function_symbols,
        estimated_mir_weight,
        exceeds_max_weight: estimated_mir_weight > max_weight,
    }
}

pub(crate) fn device_debug_source_scope_map<'tcx>(
    tcx: TyCtxt<'tcx>,
    func: &CollectedFunction<'tcx>,
) -> DebugSourceScopeMap {
    let mir = tcx.instance_mir(func.instance.def);
    let scopes = mir
        .source_scopes
        .iter_enumerated()
        .map(|(scope, data)| {
            let inlined = data.inlined.map(|(callee, callsite)| {
                let callee = tcx.instantiate_and_normalize_erasing_regions(
                    func.instance.args,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(callee),
                );
                let callsite = hygiene::walk_chain_collapsed(callsite, mir.span);
                DebugInlinedScope {
                    callee_name: rustc_middle::ty::print::with_no_trimmed_paths!(
                        callee.to_string()
                    ),
                    callsite: debug_position_from_span(tcx, callsite),
                }
            });

            DebugSourceScope {
                id: scope.as_u32(),
                parent: data.parent_scope.map(|parent| parent.as_u32()),
                span: debug_position_from_span(tcx, data.span.source_callsite()),
                inlined,
            }
        })
        .collect();

    let mut locations = Vec::new();
    for block in mir.basic_blocks.iter() {
        for stmt in &block.statements {
            if let Some(pos) = debug_position_from_span(tcx, stmt.source_info.span) {
                locations.push(DebugSourceScopeLocation {
                    pos,
                    scope: stmt.source_info.scope.as_u32(),
                });
            }
        }

        let terminator = block.terminator();
        if let Some(pos) = debug_position_from_span(tcx, terminator.source_info.span) {
            locations.push(DebugSourceScopeLocation {
                pos,
                scope: terminator.source_info.scope.as_u32(),
            });
        }
    }
    locations.sort_by(|lhs, rhs| {
        (&lhs.pos.file, lhs.pos.line, lhs.pos.column, lhs.scope).cmp(&(
            &rhs.pos.file,
            rhs.pos.line,
            rhs.pos.column,
            rhs.scope,
        ))
    });
    locations.dedup();

    DebugSourceScopeMap { scopes, locations }
}

fn debug_position_from_span(tcx: TyCtxt<'_>, span: Span) -> Option<DebugSourcePosition> {
    let (file, line, column, _, _) = tcx.sess.source_map().span_to_location_info(span);
    let file = file?;
    if line == 0 || column == 0 {
        return None;
    }

    Some(DebugSourcePosition {
        file: file.name.prefer_local_unconditionally().to_string().into(),
        line: line as i32,
        column: column as i32,
    })
}

/// Errors that can occur during device code generation.
#[derive(Debug)]
pub enum DeviceCodegenError {
    /// No kernels were found to compile.
    NoKernels,
    /// Failed to enter or exit stable_mir context.
    StableMirError(String),
    /// MIR to Pliron IR translation failed.
    Translation(String),
    /// PTX generation (llc invocation) failed.
    PtxGeneration(String),
    /// A `#[device] extern` signature could not be represented exactly.
    InvalidDeviceExternSignature(String),
    /// IO error (file read/write).
    Io(std::io::Error),
}

impl std::fmt::Display for DeviceCodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoKernels => write!(f, "No kernel functions found"),
            Self::StableMirError(msg) => write!(f, "stable_mir error: {}", msg),
            Self::Translation(msg) => write!(f, "Translation failed: {}", msg),
            Self::PtxGeneration(msg) => write!(f, "PTX generation failed: {}", msg),
            Self::InvalidDeviceExternSignature(msg) => {
                write!(f, "Invalid device-extern signature: {msg}")
            }
            Self::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for DeviceCodegenError {}

impl From<std::io::Error> for DeviceCodegenError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn report_owner_partition_calibration(
    function_mir_weights: &BTreeMap<String, usize>,
    function_families: &[DeviceFunctionFamily],
) {
    const MIB: usize = 1024 * 1024;
    for target_mib in [6, 7, 8, 9, 10, 12, 24, 48, 96] {
        let target_weight = target_mib * MIB;
        let max_weight = target_weight.saturating_mul(6) / 5;
        let plans = plan_owner_partitions(
            function_mir_weights,
            function_families,
            OwnerPartitionPolicy {
                target_weight,
                max_weight,
            },
        );
        let mut plan_weights = plans
            .iter()
            .map(|plan| plan.estimated_mir_weight)
            .collect::<Vec<_>>();
        plan_weights.sort_unstable();
        let percentile = |numerator: usize, denominator: usize| {
            let index = plan_weights
                .len()
                .saturating_mul(numerator)
                .div_ceil(denominator)
                .saturating_sub(1);
            plan_weights
                .get(index.min(plan_weights.len().saturating_sub(1)))
                .copied()
                .unwrap_or_default()
        };
        let unique_weight = function_mir_weights.values().copied().sum::<usize>();
        let summed_weight = plan_weights.iter().copied().sum::<usize>();
        let oversize_plans = plans.iter().filter(|plan| plan.exceeds_max_weight).count();
        eprintln!(
            "[rustc_codegen_cuda] owner partition calibration: target_mib={target_mib} \
             target_weight={target_weight} max_weight={max_weight} partitions={} \
             oversize_plans={oversize_plans} unique_mir_weight={unique_weight} \
             summed_partition_mir_weight={summed_weight} min_plan_weight={} \
             p50_plan_weight={} p90_plan_weight={} max_plan_weight={}",
            plans.len(),
            plan_weights.first().copied().unwrap_or_default(),
            percentile(1, 2),
            percentile(9, 10),
            plan_weights.last().copied().unwrap_or_default(),
        );
    }
}

/// Generates PTX for device functions using the cuda-oxide pipeline.
///
/// This is the main entry point for device codegen. It bridges between
/// rustc's internal types (`rustc_middle`) and mir-importer's stable_mir-based
/// pipeline (`rustc_public`).
///
/// ## Parameters
///
/// - `tcx`: The type context from rustc
/// - `functions`: Collected device functions from the collector module
/// - `config`: Output and diagnostic configuration
///
/// ## Returns
///
/// - `Ok(DeviceCodegenResult)`: Paths to generated .ll and .ptx files
/// - `Err(DeviceCodegenError)`: Description of what went wrong
///
/// ## Pipeline Stages
///
/// ```text
/// CollectedFunction<'tcx>
///         │
///         ├──▶ rustc_internal::stable() ──▶ rustc_public::Instance
///         │
///         └──▶ mir_importer::run_pipeline()
///                     │
///                     ├──▶ `dialect-mir` (alloca form)
///                     │
///                     ├──▶ `dialect-mir` (mem2reg → SSA)
///                     ├──▶ annotated loop unroll
///                     │
///                     ├──▶ LLVM dialect
///                     │
///                     ├──▶ textual LLVM IR (.ll)
///                     │
///                     └──▶ PTX (.ptx) via `llc`
/// ```
pub fn generate_device_code<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
    function_families: &[DeviceFunctionFamily],
    device_externs: &[DeviceExternDecl],
    config: &DeviceCodegenConfig,
) -> Result<DeviceCodegenResult, DeviceCodegenError> {
    use rustc_public::rustc_internal;

    if functions.is_empty() {
        return Err(DeviceCodegenError::NoKernels);
    }

    if config.verbose {
        eprintln!(
            "[device_codegen] Compiling {} functions, {} device externs to PTX via cuda-oxide pipeline",
            functions.len(),
            device_externs.len()
        );
        for func in functions {
            eprintln!(
                "[device_codegen]   {} {}",
                if func.is_kernel { "kernel" } else { "device" },
                func.export_name
            );
        }
        for decl in device_externs {
            eprintln!(
                "[device_codegen]   extern {} (convergent={}, pure={}, readonly={})",
                decl.export_name,
                decl.attrs.is_convergent,
                decl.attrs.is_pure,
                decl.attrs.is_readonly
            );
        }
    }

    // Prepare data we need to pass into the stable_mir closure
    // (closures can't capture references to local TyCtxt data)
    let export_names: Vec<(String, bool)> = functions
        .iter()
        .map(|f| (f.export_name.clone(), f.is_kernel))
        .collect();
    let function_symbols = functions
        .iter()
        .map(|function| tcx.symbol_name(function.instance).name.to_string())
        .collect::<Vec<_>>();
    let debug_scope_maps: Vec<_> = functions
        .iter()
        .map(|function| device_debug_source_scope_map(tcx, function))
        .collect();
    let partition_stats = std::env::var_os("CUDA_OXIDE_PARTITION_STATS").is_some();
    let partition_plan_started = std::time::Instant::now();
    let function_mir_weights = if config.partition_large_owner {
        Some(
            functions
                .iter()
                .zip(function_symbols.iter())
                .map(|(function, symbol)| {
                    let mir = tcx.instance_mir(function.instance.def);
                    (symbol.clone(), structural_mir_weight(mir))
                })
                .collect::<BTreeMap<_, _>>(),
        )
    } else {
        None
    };
    let partition_plans = if let Some(function_mir_weights) = &function_mir_weights {
        plan_owner_partitions(
            function_mir_weights,
            function_families,
            OwnerPartitionPolicy::default(),
        )
    } else {
        Vec::new()
    };
    let partitioned_owner = partition_plans.len() > 1;
    if partition_stats {
        let estimated_mir_weight = partition_plans
            .iter()
            .map(|partition| partition.estimated_mir_weight)
            .sum::<usize>();
        let oversize_plans = partition_plans
            .iter()
            .filter(|partition| partition.exceeds_max_weight)
            .count();
        eprintln!(
            "[rustc_codegen_cuda] owner partition plan: enabled={} partitioned={} \
             functions={} roots={} partitions={} oversize_plans={} summed_partition_mir_weight={} \
             target_weight={} max_weight={} elapsed={:?} peak_rss_kib={:?}",
            config.partition_large_owner,
            partitioned_owner,
            functions.len(),
            function_families.len(),
            partition_plans.len().max(1),
            oversize_plans,
            estimated_mir_weight,
            OWNER_PARTITION_TARGET_WEIGHT,
            OWNER_PARTITION_MAX_WEIGHT,
            partition_plan_started.elapsed(),
            process_peak_rss_kib(),
        );
        for (index, partition) in partition_plans.iter().enumerate() {
            eprintln!(
                "[rustc_codegen_cuda] owner partition plan item: index={index} \
                 plan_id={} roots={} functions={} estimated_mir_weight={} exceeds_max_weight={}",
                partition.plan_id,
                partition.root_symbols.len(),
                partition.function_symbols.len(),
                partition.estimated_mir_weight,
                partition.exceeds_max_weight,
            );
        }
    }
    let partition_diagnostic_selected = std::env::var_os("CUDA_OXIDE_PARTITION_PLAN_OWNER")
        .is_none_or(|owner| owner == std::ffi::OsStr::new(&config.output_name));
    if partition_diagnostic_selected
        && std::env::var_os("CUDA_OXIDE_PARTITION_CALIBRATION").is_some()
    {
        let function_mir_weights = function_mir_weights.as_ref().ok_or_else(|| {
            DeviceCodegenError::PtxGeneration(
                "partition calibration requires owner partitioning".to_string(),
            )
        })?;
        report_owner_partition_calibration(function_mir_weights, function_families);
    }
    if partition_diagnostic_selected && std::env::var_os("CUDA_OXIDE_PARTITION_PLAN_ONLY").is_some()
    {
        return Err(DeviceCodegenError::PtxGeneration(
            "owner partition plan-only diagnostic completed".to_string(),
        ));
    }
    drop(function_mir_weights);
    let output_dir = config.output_dir.clone();
    let output_name = config.output_name.clone();
    let mut partition_output_guard = if config.partition_large_owner {
        OwnerPartitionOutputGuard::prepare(&output_dir, &output_name, partitioned_owner)?
    } else {
        None
    };
    let partition_output_dir = partition_output_guard
        .as_ref()
        .map(|guard| guard.path.clone());

    // Convert device externs to mir-importer format
    // We extract signature info from rustc here since we have access to TyCtxt
    let stable_device_externs: Vec<mir_importer::DeviceExternDecl> = device_externs
        .iter()
        .map(|decl| {
            // Get function signature from rustc
            let fn_sig = tcx.fn_sig(decl.def_id).instantiate_identity();
            let fn_sig = fn_sig.skip_binder();

            if !matches!(fn_sig.abi, rustc_abi::ExternAbi::C { unwind: false }) {
                return Err(DeviceCodegenError::InvalidDeviceExternSignature(format!(
                    "`{}` uses ABI {:?}; device externs must use `extern \"C\"` without unwinding",
                    decl.export_name, fn_sig.abi
                )));
            }

            if fn_sig.c_variadic {
                return Err(DeviceCodegenError::InvalidDeviceExternSignature(format!(
                    "`{}` is variadic; variadic device externs are not supported",
                    decl.export_name
                )));
            }

            let param_types = fn_sig
                .inputs()
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    rustc_ty_to_device_extern_type(tcx, *ty, DeviceExternTypePosition::Parameter)
                        .map_err(|reason| {
                            DeviceCodegenError::InvalidDeviceExternSignature(format!(
                                "`{}` parameter {} (`{ty}`): {reason}",
                                decl.export_name, index
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let result_ty = fn_sig.output();
            let return_type =
                rustc_ty_to_device_extern_type(tcx, result_ty, DeviceExternTypePosition::Result)
                    .map_err(|reason| {
                        DeviceCodegenError::InvalidDeviceExternSignature(format!(
                            "`{}` result (`{result_ty}`): {reason}",
                            decl.export_name
                        ))
                    })?;

            Ok(mir_importer::DeviceExternDecl {
                export_name: decl.export_name.clone(),
                param_types,
                return_type,
                attrs: mir_importer::DeviceExternAttrs {
                    is_convergent: decl.attrs.is_convergent,
                    is_pure: decl.attrs.is_pure,
                    is_readonly: decl.attrs.is_readonly,
                },
            })
        })
        .collect::<Result<_, _>>()?;

    let verbose = config.verbose;
    let show_rustc_mir = config.dump_rustc_mir;
    let show_mir = config.dump_mir_dialect;
    let show_llvm = config.dump_llvm_dialect;
    let debug_kind = device_debug_kind(tcx.sess.opts.debuginfo);

    // Print raw rustc MIR if requested (before conversion to stable_mir)
    if show_rustc_mir {
        use rustc_middle::ty::print::with_no_trimmed_paths;

        eprintln!();
        eprintln!("=== Rustc MIR (before translation) ===");
        for func in functions {
            let mir = tcx.instance_mir(func.instance.def);
            eprintln!();
            eprintln!("fn {} {{", func.export_name);

            // Print locals
            eprintln!(
                "    let mut _0: {:?};",
                mir.local_decls[rustc_middle::mir::Local::from_u32(0)].ty
            );
            for (local, decl) in mir.local_decls.iter_enumerated().skip(1) {
                let mutability = if decl.mutability == rustc_middle::mir::Mutability::Mut {
                    "mut "
                } else {
                    ""
                };
                eprintln!("    let {}_{}:  {:?};", mutability, local.index(), decl.ty);
            }

            // Print debug info
            for debug_info in &mir.var_debug_info {
                with_no_trimmed_paths!(eprintln!(
                    "    debug {:?} => {:?};",
                    debug_info.name, debug_info.value
                ));
            }

            // Print basic blocks
            for (bb_idx, bb_data) in mir.basic_blocks.iter_enumerated() {
                eprintln!("    bb{}: {{", bb_idx.index());
                for stmt in &bb_data.statements {
                    with_no_trimmed_paths!(eprintln!("        {:?}", stmt));
                }
                with_no_trimmed_paths!(eprintln!("        {:?}", bb_data.terminator().kind));
                eprintln!("    }}");
            }
            eprintln!("}}");
        }
        eprintln!();
    }

    // Enter stable_mir context and run the pipeline.
    //
    // rustc_internal::run() does the following:
    // 1. Creates Tables for type/instance interning
    // 2. Sets up thread-local CompilerCtxt
    // 3. Runs our closure with access to stable() conversion
    // 4. Tears down the context and returns our result
    // Pre-compute inline attributes before entering the stable_mir
    // context, since the query lives on `rustc_middle::TyCtxt` and is not
    // exposed through stable_mir. Preserving this hint avoids making helper
    // boundaries depend entirely on later optimizer heuristics.
    // Partitioned owners stop before libNVVM so the finalizer can compile,
    // link, and release one bounded source partition at a time.
    let emit_nvvm_ir = partitioned_owner || std::env::var_os("CUDA_OXIDE_EMIT_NVVM_IR").is_some();
    let inline_stats = std::env::var_os("CUDA_OXIDE_INLINE_STATS").is_some();
    let inline_plan_started = inline_stats.then(std::time::Instant::now);
    // The shared lowering pipeline can discover libdevice calls and select
    // NVVM IR after this rustc-facing phase. Build the plan independently of
    // an explicit NVVM request so that auto-selected device linking receives
    // the same intent. Direct PTX export deliberately ignores the orthogonal
    // `device_link_alwaysinline` attribute.
    let inline_plan = build_device_inline_plan(tcx, functions);
    if let Some(inline_plan_started) = inline_plan_started {
        let inline_plan_elapsed = inline_plan_started.elapsed();
        let existing_device_always = functions
            .iter()
            .filter(|function| {
                matches!(
                    inline_attr_for_device_function(tcx, function.instance),
                    mir_importer::InlineAttr::DeviceAlways
                )
            })
            .count();
        let matched_array_closures = functions
            .iter()
            .filter(|function| inline_plan.array_closures.contains(&function.instance))
            .count();
        let matched_array_builders = functions
            .iter()
            .filter(|function| inline_plan.array_builders.contains(&function.instance))
            .count();
        let matched_borrowed_kernel_closures = functions
            .iter()
            .filter(|function| {
                inline_plan
                    .borrowed_kernel_closures
                    .contains(&function.instance)
            })
            .count();
        let matched_borrowed_always_inline_helper_closures = functions
            .iter()
            .filter(|function| {
                inline_plan
                    .borrowed_always_inline_helper_closures
                    .contains(&function.instance)
            })
            .count();
        let matched_deferred_full_unroll_helpers = functions
            .iter()
            .filter(|function| {
                inline_plan
                    .deferred_full_unroll_helpers
                    .contains(&function.instance)
            })
            .count();
        eprintln!(
            "[rustc_codegen_cuda] inline plan: elapsed={inline_plan_elapsed:?} \
             existing_device_always={existing_device_always} \
             considered_array_builders={} rejected_array_builders={} \
             conflicting_array_closures={} array_builders={} \
             matched_array_builders={matched_array_builders} array_closures={} \
             matched_array_closures={matched_array_closures} \
             conflicting_deferred_unroll_closures={} \
             deferred_full_unroll_helpers={} \
             matched_deferred_full_unroll_helpers={matched_deferred_full_unroll_helpers} \
             borrowed_kernel_closures={} \
             matched_borrowed_kernel_closures={matched_borrowed_kernel_closures} \
             borrowed_always_inline_helper_closures={} \
             matched_borrowed_always_inline_helper_closures={matched_borrowed_always_inline_helper_closures} \
             strategy=bounded_callbacks_deferred_array_unroll_and_borrowed_frames",
            inline_plan.considered_array_builders,
            inline_plan.rejected_array_builders,
            inline_plan.conflicting_array_closures,
            inline_plan.array_builders.len(),
            inline_plan.array_closures.len(),
            inline_plan.conflicting_deferred_unroll_closures,
            inline_plan.deferred_full_unroll_helpers.len(),
            inline_plan.borrowed_kernel_closures.len(),
            inline_plan.borrowed_always_inline_helper_closures.len()
        );
    }
    let function_codegen_policies =
        device_function_codegen_policies_from_plan(tcx, functions, &inline_plan);
    let device_mono_reachability: Vec<crate::collector::DeviceMonoReachability> = functions
        .iter()
        .map(|func| crate::collector::device_mono_reachability(tcx, func.instance))
        .collect();
    let core_index_trait = tcx.lang_items().index_trait();

    let result = rustc_internal::run(tcx, || {
        let stable_core_index_trait = core_index_trait.map(rustc_internal::stable);
        // Convert internal Instance<'tcx> to stable_mir Instance.
        // Drop glue instances whose bodies are provably no-ops are filtered
        // out: the mir-importer's translate_drop fast-path emits a plain
        // branch for them and never references the function, so translating
        // their (potentially complex) shim bodies is both unnecessary and
        // can fail on constructs the device pipeline does not support.
        // The collector already skips these at discovery time with the
        // same shared predicate (collector::process_drop_place), so this
        // filter is a final guard that keeps translation in lockstep with
        // emission if a future collection path forgets the check.
        let mut stable_functions = Vec::<(String, mir_importer::CollectedFunction)>::new();
        for (index, func) in functions.iter().enumerate() {
            // Use rustc_internal::stable() to convert the Instance.
            // This is the key bridge between rustc_middle and rustc_public types.
            let stable_instance = rustc_internal::stable(func.instance);

            // Skip no-op drop glue: the mir-importer lowers these as plain
            // branches (via drop_glue_is_noop) and never emits a call, so the
            // function body is dead. Translating it would fail on IntoIter and
            // similar stdlib shims whose MIR contains constructs the device
            // pipeline does not support.
            if matches!(func.instance.def, InstanceKind::DropGlue(..))
                && mir_importer::drop_instance_is_noop(&stable_instance)
            {
                continue;
            }

            let (export_name, is_kernel) = &export_names[index];
            let reachability = &device_mono_reachability[index];
            stable_functions.push((
                function_symbols[index].clone(),
                mir_importer::CollectedFunction {
                    instance: stable_instance,
                    rustc_mir_block_count: reachability.block_count,
                    rustc_mono_successors: reachability.successors.clone(),
                    is_kernel: *is_kernel,
                    export_name: export_name.clone(),
                    debug_source_scopes: Some(debug_scope_maps[index].clone()),
                    inline_attr: function_codegen_policies[index].inline_attr,
                    device_link_always: function_codegen_policies[index].device_link_always,
                    device_link_inline_candidate: function_codegen_policies[index]
                        .device_link_inline_candidate,
                    deferred_full_unroll: function_codegen_policies[index].deferred_full_unroll,
                    core_index_trait: stable_core_index_trait,
                },
            ));
        }

        if verbose {
            eprintln!(
                "[device_codegen] Converted {} functions to stable_mir format",
                stable_functions.len()
            );
            if emit_nvvm_ir {
                eprintln!("[device_codegen] NVVM IR mode enabled");
            }
        }

        let target_arch = std::env::var("CUDA_OXIDE_TARGET").ok();
        let device_arch_hint = std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok();
        let allow_fma_contraction = std::env::var_os("CUDA_OXIDE_NO_FMA").is_none();

        if verbose && !allow_fma_contraction {
            eprintln!("[device_codegen] FMA contraction disabled");
        }

        let stable_functions_by_symbol =
            stable_functions.iter().cloned().collect::<BTreeMap<_, _>>();
        let run_partition = |partition_output_dir: PathBuf,
                             partition_output_name: String,
                             partition_functions: Vec<mir_importer::CollectedFunction>,
                             partitioned: bool,
                             partition_index: usize| {
            let pipeline_config = mir_importer::PipelineConfig {
                output_dir: partition_output_dir,
                output_name: partition_output_name,
                verbose,
                show_mir_dialect: show_mir,
                show_llvm_dialect: show_llvm,
                emit_nvvm_ir,
                target_arch: target_arch.clone(),
                target_arch_source: "CUDA_OXIDE_TARGET",
                device_arch_hint: device_arch_hint.clone(),
                debug_kind,
                allow_fma_contraction,
                partitioned_owner: partitioned,
            };
            let kernel_count = partition_functions
                .iter()
                .filter(|function| function.is_kernel)
                .count();
            let started = std::time::Instant::now();
            let result = mir_importer::run_pipeline(
                &partition_functions,
                &stable_device_externs,
                &pipeline_config,
            );
            if partition_stats {
                match result.as_ref() {
                    Ok(compilation) => {
                        let llvm_bytes = std::fs::metadata(&compilation.ll_path)
                            .map(|metadata| metadata.len())
                            .unwrap_or_default();
                        let artifact_bytes = std::fs::metadata(&compilation.artifact_path)
                            .map(|metadata| metadata.len())
                            .unwrap_or_default();
                        let artifact_limit_bytes = match compilation.artifact_kind {
                            mir_importer::CompilationArtifactKind::NvvmIr => {
                                cuda_artifact_finalizer::MAX_PARTITION_NVVM_IR_BYTES
                            }
                            mir_importer::CompilationArtifactKind::Ptx => {
                                cuda_artifact_finalizer::MAX_PARTITION_PTX_BYTES
                            }
                            mir_importer::CompilationArtifactKind::Ltoir
                            | mir_importer::CompilationArtifactKind::Cubin => 0,
                        };
                        eprintln!(
                            "[rustc_codegen_cuda] owner partition codegen: index={partition_index} \
                             name={} functions={} kernels={kernel_count} llvm_bytes={llvm_bytes} \
                             artifact_bytes={artifact_bytes} artifact_kind={:?} \
                             partition_source_limit_bytes={} exceeds_partition_source_limit={} elapsed={:?} \
                             peak_rss_kib={:?}",
                            pipeline_config.output_name,
                            partition_functions.len(),
                            compilation.artifact_kind,
                            artifact_limit_bytes,
                            artifact_bytes > artifact_limit_bytes,
                            started.elapsed(),
                            process_peak_rss_kib(),
                        );
                    }
                    Err(error) => {
                        eprintln!(
                            "[rustc_codegen_cuda] owner partition codegen failed: \
                             index={partition_index} name={} functions={} kernels={kernel_count} \
                             elapsed={:?} peak_rss_kib={:?} error={error}",
                            pipeline_config.output_name,
                            partition_functions.len(),
                            started.elapsed(),
                            process_peak_rss_kib(),
                        );
                    }
                }
            }
            result
        };

        // Run the cuda-oxide pipeline!
        // Rust MIR → `dialect-mir` → mem2reg → unroll → LLVM dialect → LLVM IR → PTX.
        // Device externs are emitted as `declare` statements in LLVM IR.
        if partitioned_owner {
            partition_plans
                .iter()
                .enumerate()
                .map(|(index, partition)| {
                    let partition_functions = partition
                        .function_symbols
                        .iter()
                        .map(|symbol| {
                            stable_functions_by_symbol.get(symbol).cloned().unwrap_or_else(|| {
                                panic!(
                                    "owner partition {} references missing stable MIR function {symbol}",
                                    partition.plan_id
                                )
                            })
                        })
                        .collect();
                    run_partition(
                        partition_output_dir
                            .clone()
                            .expect("partitioned owner prepared its hidden output directory"),
                        format!("part-{}", partition.plan_id),
                        partition_functions,
                        true,
                        index,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            run_partition(
                output_dir.clone(),
                output_name.clone(),
                stable_functions
                    .into_iter()
                    .map(|(_, function)| function)
                    .collect(),
                false,
                0,
            )
            .map(|result| vec![result])
        }
    });

    // Handle the result from rustc_internal::run.
    // We have nested Results: outer from run(), inner from run_pipeline().
    match result {
        Ok(pipeline_result) => match pipeline_result {
            Ok(compilation_results) => {
                let Some(first) = compilation_results.first() else {
                    return Err(DeviceCodegenError::PtxGeneration(
                        "device pipeline produced no owner partition".to_string(),
                    ));
                };
                if compilation_results.iter().any(|result| {
                    result.target != first.target
                        || result.allow_fma_contraction != first.allow_fma_contraction
                }) {
                    return Err(DeviceCodegenError::PtxGeneration(
                        "owner partitions selected inconsistent CUDA targets or FMA policies"
                            .to_string(),
                    ));
                }

                let mut artifacts = Vec::with_capacity(compilation_results.len());
                for compilation_result in &compilation_results {
                    if let Some(artifact) = read_compilation_artifact(
                        compilation_result,
                        !partitioned_owner,
                        partitioned_owner,
                    )? {
                        if config.verbose {
                            eprintln!(
                                "[device_codegen] Embeddable artifact generated: {} ({:?}, target: {})",
                                artifact.name, artifact.kind, compilation_result.target
                            );
                        }
                        artifacts.push(artifact);
                    } else if config.verbose {
                        eprintln!(
                            "[device_codegen] No embeddable artifact found for {} (target: {})",
                            compilation_result.ll_path.display(),
                            compilation_result.target
                        );
                    }
                }
                let ptx_content = match artifacts.as_slice() {
                    [artifact] if artifact.kind == DeviceCodegenArtifactKind::Ptx => Some(
                        String::from_utf8(
                            artifact
                                .bytes
                                .clone()
                                .expect("single PTX artifact retains its payload"),
                        )
                        .map_err(|error| {
                            DeviceCodegenError::PtxGeneration(format!(
                                "generated PTX is not valid UTF-8: {error}"
                            ))
                        })?,
                    ),
                    _ => None,
                };
                if let Some(guard) = partition_output_guard.as_mut() {
                    guard.keep();
                }

                Ok(DeviceCodegenResult {
                    ptx_path: first.ptx_path.clone(),
                    ll_path: first.ll_path.clone(),
                    ptx_bundle_path: partitioned_owner.then(|| {
                        config
                            .output_dir
                            .join(format!("{}.ptx.bundle", config.output_name))
                    }),
                    target: first.target.clone(),
                    ptx_content,
                    artifacts,
                    allow_fma_contraction: first.allow_fma_contraction,
                    debug_kind,
                })
            }
            Err(pipeline_err) => Err(DeviceCodegenError::PtxGeneration(format!(
                "{}",
                pipeline_err
            ))),
        },
        Err(stable_mir_err) => Err(DeviceCodegenError::StableMirError(format!(
            "{:?}",
            stable_mir_err
        ))),
    }
}

pub(crate) fn device_debug_kind(rustc_debug: DebugInfo) -> llvm_export::export::DebugKind {
    device_debug_kind_with_override(
        rustc_debug,
        std::env::var("CUDA_OXIDE_DEBUG").ok().as_deref(),
    )
}

fn device_debug_kind_with_override(
    rustc_debug: DebugInfo,
    override_value: Option<&str>,
) -> llvm_export::export::DebugKind {
    if let Some(value) = override_value {
        match value.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "none" => return llvm_export::export::DebugKind::Off,
            "1" | "line" | "lines" | "line-tables" | "line-tables-only" => {
                return llvm_export::export::DebugKind::LineTables;
            }
            "2" | "full" => return llvm_export::export::DebugKind::Full,
            _ => {}
        }
    }

    match rustc_debug {
        DebugInfo::None => llvm_export::export::DebugKind::Off,
        DebugInfo::LineDirectivesOnly
        | DebugInfo::LineTablesOnly
        | DebugInfo::Limited
        | DebugInfo::Full => llvm_export::export::DebugKind::LineTables,
    }
}

fn read_compilation_artifact(
    result: &mir_importer::CompilationResult,
    retain_bytes: bool,
    required: bool,
) -> Result<Option<DeviceCodegenArtifact>, DeviceCodegenError> {
    let kind = match result.artifact_kind {
        mir_importer::CompilationArtifactKind::Ptx => DeviceCodegenArtifactKind::Ptx,
        mir_importer::CompilationArtifactKind::NvvmIr => DeviceCodegenArtifactKind::NvvmIr,
        mir_importer::CompilationArtifactKind::Ltoir => DeviceCodegenArtifactKind::Ltoir,
        mir_importer::CompilationArtifactKind::Cubin => DeviceCodegenArtifactKind::Cubin,
    };

    match std::fs::metadata(&result.artifact_path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(DeviceCodegenArtifact {
            kind,
            name: result
                .artifact_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("device-artifact")
                .to_string(),
            path: result.artifact_path.clone(),
            ptx_sidecar_path: result.ptx_path.clone(),
            bytes: retain_bytes
                .then(|| std::fs::read(&result.artifact_path))
                .transpose()?,
        })),
        Ok(_) if required => Err(DeviceCodegenError::PtxGeneration(format!(
            "required owner partition artifact is not a regular file: {}",
            result.artifact_path.display()
        ))),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && required => {
            Err(DeviceCodegenError::PtxGeneration(format!(
                "required owner partition artifact was not produced: {}",
                result.artifact_path.display()
            )))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(DeviceCodegenError::Io(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeviceCodegenConfig::default();
        assert!(!config.verbose);
        assert_eq!(config.output_name, "kernel");
        assert!(!config.partition_large_owner);
    }

    #[test]
    fn owner_partition_plan_and_export_inventory_ignore_input_order() {
        let function_weights = BTreeMap::from([
            ("kernel_a".to_string(), 4),
            ("kernel_b".to_string(), 4),
            ("kernel_c".to_string(), 2),
            ("leaf_a".to_string(), 3),
            ("leaf_b".to_string(), 3),
            ("leaf_c".to_string(), 2),
            ("shared".to_string(), 2),
        ]);
        let mut families = vec![
            DeviceFunctionFamily {
                root_symbol: "kernel_b".to_string(),
                root_export_name: "kernel_b".to_string(),
                function_symbols: vec![
                    "shared".to_string(),
                    "leaf_b".to_string(),
                    "kernel_b".to_string(),
                ],
            },
            DeviceFunctionFamily {
                root_symbol: "kernel_a".to_string(),
                root_export_name: "kernel_a".to_string(),
                function_symbols: vec![
                    "kernel_a".to_string(),
                    "shared".to_string(),
                    "leaf_a".to_string(),
                ],
            },
            DeviceFunctionFamily {
                root_symbol: "kernel_c".to_string(),
                root_export_name: "kernel_c".to_string(),
                function_symbols: vec!["leaf_c".to_string(), "kernel_c".to_string()],
            },
        ];
        let policy = OwnerPartitionPolicy {
            target_weight: 10,
            max_weight: 12,
        };
        let baseline = plan_owner_partitions(&function_weights, &families, policy);

        families.reverse();
        for family in &mut families {
            family.function_symbols.reverse();
        }
        let mut reversed_weights = function_weights.into_iter().collect::<Vec<_>>();
        reversed_weights.reverse();
        let reversed_weights = reversed_weights.into_iter().collect();
        let reordered = plan_owner_partitions(&reversed_weights, &families, policy);

        assert_eq!(baseline, reordered);
        assert!(baseline.len() > 1);
        let exported_roots = baseline
            .iter()
            .flat_map(|partition| partition.root_symbols.iter().cloned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            exported_roots,
            BTreeSet::from([
                "kernel_a".to_string(),
                "kernel_b".to_string(),
                "kernel_c".to_string(),
            ])
        );
        for root in ["kernel_a", "kernel_b"] {
            let partition = baseline
                .iter()
                .find(|partition| {
                    partition
                        .root_symbols
                        .iter()
                        .any(|candidate| candidate == root)
                })
                .unwrap();
            assert!(
                partition
                    .function_symbols
                    .iter()
                    .any(|symbol| symbol == root)
            );
            assert!(
                partition
                    .function_symbols
                    .iter()
                    .any(|symbol| symbol == "shared")
            );
        }
    }

    #[test]
    fn owner_partition_plan_co_locates_overlapping_root_definitions() {
        let function_weights = BTreeMap::from([
            ("kernel_a".to_string(), 4),
            ("kernel_b".to_string(), 4),
            ("kernel_c".to_string(), 4),
            ("leaf_c".to_string(), 2),
        ]);
        let families = vec![
            DeviceFunctionFamily {
                root_symbol: "kernel_a".to_string(),
                root_export_name: "kernel_a".to_string(),
                function_symbols: vec!["kernel_a".to_string(), "kernel_b".to_string()],
            },
            DeviceFunctionFamily {
                root_symbol: "kernel_b".to_string(),
                root_export_name: "kernel_b".to_string(),
                function_symbols: vec!["kernel_b".to_string()],
            },
            DeviceFunctionFamily {
                root_symbol: "kernel_c".to_string(),
                root_export_name: "kernel_c".to_string(),
                function_symbols: vec!["kernel_c".to_string(), "leaf_c".to_string()],
            },
        ];
        let partitions = plan_owner_partitions(
            &function_weights,
            &families,
            OwnerPartitionPolicy {
                target_weight: 6,
                max_weight: 7,
            },
        );

        assert_eq!(partitions.len(), 2);
        let overlapping = partitions
            .iter()
            .find(|partition| partition.root_symbols.iter().any(|root| root == "kernel_a"))
            .unwrap();
        assert_eq!(
            overlapping.root_symbols,
            vec!["kernel_a".to_string(), "kernel_b".to_string()]
        );
        assert!(
            overlapping.exceeds_max_weight,
            "overlapping strong roots are indivisible and must be marked oversize"
        );
        for root in ["kernel_a", "kernel_b", "kernel_c"] {
            assert_eq!(
                partitions
                    .iter()
                    .filter(|partition| {
                        partition
                            .function_symbols
                            .iter()
                            .any(|symbol| symbol == root)
                    })
                    .count(),
                1,
                "{root} must have exactly one strong definition"
            );
        }
    }

    #[test]
    fn owner_partition_oversize_marking_distinguishes_exact_cap_and_cap_plus_one() {
        let family = [DeviceFunctionFamily {
            root_symbol: "kernel".to_string(),
            root_export_name: "kernel".to_string(),
            function_symbols: vec!["kernel".to_string(), "leaf".to_string()],
        }];
        let policy = OwnerPartitionPolicy {
            target_weight: 10,
            max_weight: 12,
        };
        let exact_cap = plan_owner_partitions(
            &BTreeMap::from([("kernel".to_string(), 7), ("leaf".to_string(), 5)]),
            &family,
            policy,
        );
        assert_eq!(exact_cap.len(), 1);
        assert_eq!(exact_cap[0].estimated_mir_weight, 12);
        assert!(!exact_cap[0].exceeds_max_weight);

        let cap_plus_one = plan_owner_partitions(
            &BTreeMap::from([("kernel".to_string(), 7), ("leaf".to_string(), 6)]),
            &family,
            policy,
        );
        assert_eq!(cap_plus_one.len(), 1);
        assert_eq!(cap_plus_one[0].estimated_mir_weight, 13);
        assert!(
            cap_plus_one[0].exceeds_max_weight,
            "a single-family owner cannot be split but must be marked honestly"
        );
    }

    #[test]
    fn device_debug_kind_follows_rustc_debuginfo() {
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, None),
            llvm_export::export::DebugKind::Off
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::LineTablesOnly, None),
            llvm_export::export::DebugKind::LineTables
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::Full, None),
            llvm_export::export::DebugKind::LineTables
        );
    }

    #[test]
    fn device_debug_kind_env_override_wins() {
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::Full, Some("off")),
            llvm_export::export::DebugKind::Off
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, Some(" Line-Tables ")),
            llvm_export::export::DebugKind::LineTables
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, Some("full")),
            llvm_export::export::DebugKind::Full
        );
    }

    #[test]
    fn read_compilation_artifact_uses_declared_nvvm_ir_path() {
        let temp_dir = unique_temp_dir("cuda-codegen-artifact");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ll_path = temp_dir.join("demo.ll");
        let ptx_path = temp_dir.join("demo.ptx");
        std::fs::write(&ll_path, b"nvvm ir").unwrap();
        std::fs::write(&ptx_path, b"stale ptx").unwrap();

        let result = mir_importer::CompilationResult {
            ll_path: ll_path.clone(),
            ptx_path,
            artifact_path: ll_path.clone(),
            artifact_kind: mir_importer::CompilationArtifactKind::NvvmIr,
            target: "sm_90".to_string(),
            allow_fma_contraction: false,
        };

        let artifact = read_compilation_artifact(&result, true, false)
            .unwrap()
            .unwrap();
        assert_eq!(artifact.kind, DeviceCodegenArtifactKind::NvvmIr);
        assert_eq!(artifact.name, "demo.ll");
        assert_eq!(artifact.path, ll_path);
        assert_eq!(artifact.bytes.as_deref(), Some(&b"nvvm ir"[..]));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn read_compilation_artifact_reads_declared_cubin_path() {
        let temp_dir = unique_temp_dir("cuda-codegen-artifact");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ll_path = temp_dir.join("demo.ll");
        let cubin_path = temp_dir.join("demo.cubin");
        let ptx_path = temp_dir.join("demo.ptx");
        std::fs::write(&cubin_path, b"cubin").unwrap();

        let result = mir_importer::CompilationResult {
            ll_path,
            ptx_path,
            artifact_path: cubin_path,
            artifact_kind: mir_importer::CompilationArtifactKind::Cubin,
            target: "sm_90".to_string(),
            allow_fma_contraction: true,
        };

        let artifact = read_compilation_artifact(&result, true, false)
            .unwrap()
            .unwrap();
        assert_eq!(artifact.kind, DeviceCodegenArtifactKind::Cubin);
        assert_eq!(artifact.name, "demo.cubin");
        assert_eq!(artifact.bytes.as_deref(), Some(&b"cubin"[..]));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn required_partition_artifact_fails_closed_when_missing() {
        let temp_dir = unique_temp_dir("cuda-codegen-missing-partition");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let missing_path = temp_dir.join("part-missing.ll");
        let result = mir_importer::CompilationResult {
            ll_path: missing_path.clone(),
            ptx_path: temp_dir.join("part-missing.ptx"),
            artifact_path: missing_path.clone(),
            artifact_kind: mir_importer::CompilationArtifactKind::NvvmIr,
            target: "sm_90".to_string(),
            allow_fma_contraction: true,
        };

        assert!(
            read_compilation_artifact(&result, false, false)
                .unwrap()
                .is_none()
        );
        let error = match read_compilation_artifact(&result, false, true) {
            Ok(_) => panic!("missing required partition artifact was accepted"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("required owner partition artifact")
                && error.to_string().contains("part-missing.ll"),
            "unexpected error: {error}"
        );

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[cfg(unix)]
    #[test]
    fn owner_partition_cleanup_rejects_symlinked_hidden_root() {
        use std::os::unix::fs::symlink;

        let temp_dir = unique_temp_dir("cuda-codegen-partition-symlink");
        let output_dir = temp_dir.join("output");
        let outside_dir = temp_dir.join("outside");
        let outside_owner = outside_dir.join("owner");
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::create_dir_all(&outside_owner).unwrap();
        let sentinel = outside_owner.join("must-survive");
        std::fs::write(&sentinel, b"outside").unwrap();
        symlink(&outside_dir, output_dir.join(".cuda-oxide-partitions")).unwrap();

        let error = match OwnerPartitionOutputGuard::prepare(&output_dir, "owner", true) {
            Ok(_) => panic!("symlinked hidden partition root was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"outside");

        std::fs::remove_file(output_dir.join(".cuda-oxide-partitions")).unwrap();
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{}-{nanos}", std::process::id()))
    }
}
