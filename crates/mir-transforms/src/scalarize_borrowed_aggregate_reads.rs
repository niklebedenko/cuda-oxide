/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonicalize bounded read-only aggregate projections before LLVM lowering.
//!
//! rustc MIR aggregate values are commonly imported through stack slots:
//!
//! ```text
//! %slot = mir.alloca
//! mir.store %slot, %aggregate
//! ...
//! %field = mir.field_addr %slot, field
//! %elem = mir.array_element_addr %field, %index
//! %value = mir.load %elem
//! ```
//!
//! `mir.field_addr` is intentionally non-promotable, so the ordinary mem2reg
//! pass cannot recover the aggregate as an SSA value. For a compiler-owned slot
//! initialized by one whole-value store, this pass validates the complete
//! pointer-use graph and rewrites read-only array loads to value operations:
//!
//! ```text
//! %aggregate = mir.load %slot
//! %array = mir.extract_field %aggregate, field
//! %value = mir.extract_array_element %array, %index
//! ```
//!
//! A slot initialized from an entry-block argument and read only in dominated
//! blocks uses that argument directly. Other compiler-owned locals retain a
//! whole-value load at the original scalar read point. The immediately
//! following mem2reg pass promotes that load and store while preserving loop-
//! carried and control-flow-dependent assignment semantics.
//!
//! This pass canonicalizes pointer-based read-only access independently of the
//! runtime index shape. The later `mir.extract_array_element` lowering owns the
//! profitability decision: a bounded `urem value, constant` becomes fixed
//! `extractvalue` candidates plus a select chain, while unsupported indices keep
//! the ordinary memory fallback.
//!
//! The pre-mem2reg phase fails closed on pointer provenance and mutation. It
//! rejects additional stores, volatile accesses, mutable derived pointers,
//! calls, pointer casts, pointer PHIs/selects, unknown users, and non-array
//! fields.
//!
//! A second, post-mem2reg phase handles immutable aggregate pointer arguments
//! such as an `&self` device helper. It accepts only an exact single-use chain:
//!
//! ```text
//! %field = mir.field_addr %aggregate_ptr, field
//! %elem = mir.array_element_addr %field, %index
//! %value = mir.load %elem
//! ```
//!
//! The rewrite retains the constant field projection, loads only that array
//! field as a value, and replaces the dynamic scalar address with
//! `mir.extract_array_element`.
//!
//! The index must either be a bounded unsigned remainder or be guarded by the
//! unique predecessor `mir.assert(mir.lt(index, constant))`. Guarded indices are
//! canonicalized to an equivalent remainder in the assertion-success block so
//! the existing typed `mir.extract_array_element` lowering can scalarize them.
//!
//! The second phase widens one dynamic element load into a load of the whole
//! array field, which is only legal when the pointed-to aggregate lives in
//! caller-private memory. Borrowed pointers that may reference global, shared,
//! or otherwise external memory must keep exactly one dynamic memory access
//! (issue #400, following the #398 precedent). The phase therefore proves
//! caller provenance before rewriting a helper: every `mir.call` of the helper
//! in the module must pass a pointer traceable, through pointer-identity
//! casts, to a caller-local `mir.alloca` of exactly the helper's aggregate
//! type. Kernel pointer parameters, phi/select merges, pointers forwarded
//! from another helper's parameter, shared/global allocations, externally
//! callable device exports, and helpers without a single visible call site
//! all fail closed and keep the original dynamic load.

use std::collections::HashMap;

use dialect_mir::{
    attributes::{FieldIndexAttr, MirCastKindAttr},
    ops::{
        MAX_SCALARIZED_CANDIDATES, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirCallOp,
        MirCastOp, MirConstantOp, MirEqOp, MirExtractArrayElementOp, MirExtractFieldOp,
        MirFieldAddrOp, MirFuncOp, MirLoadOp, MirLtOp, MirNeOp, MirRemOp, MirReturnOp,
        MirStorageDeadOp, MirStorageLiveOp, MirStoreOp,
    },
    types::{MirArrayType, MirPtrType, MirStructType},
};
use pliron::{
    builtin::op_interfaces::SymbolOpInterface,
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    graph::ControlFlowGraph,
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};

#[derive(Clone)]
struct LoadRewrite {
    load: Ptr<Operation>,
    field_index: u32,
    index: Value,
    array_type: TypeHandle,
    element_type: TypeHandle,
    nested_fields: Vec<FieldExtraction>,
    result_type: TypeHandle,
}

#[derive(Clone, Copy)]
struct FieldExtraction {
    field_index: u32,
    result_type: TypeHandle,
}

#[derive(Clone, Copy)]
struct ArrayProjection {
    field_index: u32,
    index: Value,
    array_type: TypeHandle,
    element_type: TypeHandle,
}

struct ProjectionAnalysis<'a> {
    nested_field_addrs: &'a mut Vec<Ptr<Operation>>,
    loads: &'a mut Vec<LoadRewrite>,
}

struct AllocaPlan {
    source: AggregateSource,
    field_addrs: Vec<Ptr<Operation>>,
    array_addrs: Vec<Ptr<Operation>>,
    nested_field_addrs: Vec<Ptr<Operation>>,
    loads: Vec<LoadRewrite>,
}

#[derive(Clone, Copy)]
enum AggregateSource {
    Direct(Value),
    Slot {
        pointer: Value,
        aggregate_type: TypeHandle,
    },
}

/// Rewrite read-only indexed compiler-owned aggregate loads before mem2reg.
///
/// Only allocas initialized by exactly one whole-value store are considered.
/// Every pointer use must belong to the exact read-only projection graph
/// accepted by `analyze_alloca`.
///
/// `verbose` is threaded from the pipeline's backend options; the pass itself
/// never reads the environment.
pub fn canonicalize_read_only_aggregate_arguments(
    module: Ptr<Operation>,
    ctx: &mut Context,
    verbose: bool,
) -> usize {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);

    let allocas: Vec<_> = ops
        .into_iter()
        .filter(|op| Operation::get_op::<MirAllocaOp>(*op, ctx).is_some())
        .collect();

    let mut rewritten_loads = 0usize;
    for alloca in allocas {
        let Some(plan) = analyze_alloca(ctx, alloca) else {
            continue;
        };
        rewritten_loads += rewrite_plan(ctx, plan);
    }

    if rewritten_loads > 0 && verbose {
        eprintln!("borrowed-aggregate scalarization: rewrote {rewritten_loads} dynamic load(s)");
    }
    rewritten_loads
}

fn collect_ops(ctx: &Context, root: Ptr<Operation>, output: &mut Vec<Ptr<Operation>>) {
    output.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, output);
            }
        }
    }
}

/// Validate one compiler-owned aggregate slot without mutating the IR.
fn analyze_alloca(ctx: &Context, alloca: Ptr<Operation>) -> Option<AllocaPlan> {
    let alloca_op = Operation::get_op::<MirAllocaOp>(alloca, ctx)?;
    let pointee = alloca_op.pointee_type(ctx);
    pointee.deref(ctx).downcast_ref::<MirStructType>()?;

    let alloca_block = alloca.deref(ctx).get_parent_block()?;
    let root = alloca.deref(ctx).get_result(0);
    let block_arguments: Vec<_> = alloca_block.deref(ctx).arguments().collect();

    let mut initializer = None;
    let mut field_addrs = Vec::new();
    let mut array_addrs = Vec::new();
    let mut nested_field_addrs = Vec::new();
    let mut loads = Vec::new();

    for root_use in root.uses(ctx) {
        let user = root_use.user_op();
        let operand_index = root_use.find_index(ctx);

        if let Some(store) = Operation::get_op::<MirStoreOp>(user, ctx) {
            if operand_index != 0 || store.is_volatile(ctx) || initializer.is_some() {
                return None;
            }

            let stored_value = store.value_opd(ctx);
            initializer = Some((user, stored_value));
            continue;
        }

        let field = Operation::get_op::<MirFieldAddrOp>(user, ctx)?;
        if operand_index != 0 {
            return None;
        }

        analyze_field_path(
            ctx,
            field,
            &mut field_addrs,
            &mut array_addrs,
            &mut nested_field_addrs,
            &mut loads,
        )?;
    }

    let (initializer, aggregate_value) = initializer?;
    let can_use_direct_value = initializer.deref(ctx).get_parent_block() == Some(alloca_block)
        && block_arguments.contains(&aggregate_value)
        && field_addrs
            .iter()
            .all(|field| field.deref(ctx).get_parent_block() != Some(alloca_block));
    let source = if can_use_direct_value {
        AggregateSource::Direct(aggregate_value)
    } else {
        AggregateSource::Slot {
            pointer: root,
            aggregate_type: pointee,
        }
    };

    Some(AllocaPlan {
        source,
        field_addrs,
        array_addrs,
        nested_field_addrs,
        loads: (!loads.is_empty()).then_some(loads)?,
    })
}

fn analyze_field_path(
    ctx: &Context,
    field: MirFieldAddrOp,
    field_addrs: &mut Vec<Ptr<Operation>>,
    array_addrs: &mut Vec<Ptr<Operation>>,
    nested_field_addrs: &mut Vec<Ptr<Operation>>,
    loads: &mut Vec<LoadRewrite>,
) -> Option<()> {
    let field_op = field.get_operation();
    let field_index = field.get_attr_field_index(ctx)?.0;
    let field_pointer = field_op.deref(ctx).get_result(0);
    let field_pointer_type = field_pointer.get_type(ctx);
    let field_pointer_type_ref = field_pointer_type.deref(ctx);
    let field_pointer_type = field_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if field_pointer_type.is_mutable {
        return None;
    }

    let array_type = field_pointer_type.pointee;
    let array_type_ref = array_type.deref(ctx);
    let array_type_info = array_type_ref.downcast_ref::<MirArrayType>()?;
    if array_type_info.size() == 0 {
        return None;
    }
    let element_type = array_type_info.element_type();

    let mut local_array_addrs = Vec::new();
    let mut local_loads = Vec::new();

    for field_use in field_pointer.uses(ctx) {
        let array_op = field_use.user_op();
        if field_use.find_index(ctx) != 0 {
            return None;
        }

        Operation::get_op::<MirArrayElementAddrOp>(array_op, ctx)?;
        let array_pointer = array_op.deref(ctx).get_result(0);
        let array_pointer_type = array_pointer.get_type(ctx);
        let array_pointer_type_ref = array_pointer_type.deref(ctx);
        let array_pointer_type = array_pointer_type_ref.downcast_ref::<MirPtrType>()?;
        if array_pointer_type.is_mutable {
            return None;
        }

        let index = array_op.deref(ctx).get_operand(1);
        let projection = ArrayProjection {
            field_index,
            index,
            array_type,
            element_type,
        };
        let mut analysis = ProjectionAnalysis {
            nested_field_addrs,
            loads: &mut local_loads,
        };
        analyze_array_element_path(ctx, array_pointer, projection, &[], &mut analysis)?;
        local_array_addrs.push(array_op);
    }

    if local_array_addrs.is_empty() || local_loads.is_empty() {
        return None;
    }

    field_addrs.push(field_op);
    array_addrs.extend(local_array_addrs);
    loads.extend(local_loads);
    Some(())
}

fn analyze_array_element_path(
    ctx: &Context,
    pointer: Value,
    projection: ArrayProjection,
    nested_fields: &[FieldExtraction],
    analysis: &mut ProjectionAnalysis<'_>,
) -> Option<()> {
    let uses = pointer.uses(ctx);
    if uses.is_empty() {
        return None;
    }

    for pointer_use in uses {
        if pointer_use.find_index(ctx) != 0 {
            return None;
        }
        let user = pointer_use.user_op();
        if let Some(load) = Operation::get_op::<MirLoadOp>(user, ctx) {
            if load.is_volatile(ctx) {
                return None;
            }
            analysis.loads.push(LoadRewrite {
                load: user,
                field_index: projection.field_index,
                index: projection.index,
                array_type: projection.array_type,
                element_type: projection.element_type,
                nested_fields: nested_fields.to_vec(),
                result_type: user.deref(ctx).get_result(0).get_type(ctx),
            });
            continue;
        }

        let nested_field = Operation::get_op::<MirFieldAddrOp>(user, ctx)?;
        let nested_pointer = user.deref(ctx).get_result(0);
        let nested_pointer_type = nested_pointer.get_type(ctx);
        let nested_pointer_type_ref = nested_pointer_type.deref(ctx);
        let nested_pointer_type = nested_pointer_type_ref.downcast_ref::<MirPtrType>()?;
        if nested_pointer_type.is_mutable {
            return None;
        }
        let extraction = FieldExtraction {
            field_index: nested_field.get_attr_field_index(ctx)?.0,
            result_type: nested_pointer_type.pointee,
        };
        let mut next_fields = nested_fields.to_vec();
        next_fields.push(extraction);
        analysis.nested_field_addrs.push(user);
        analyze_array_element_path(
            ctx,
            nested_pointer,
            projection,
            &next_fields,
            analysis,
        )?;
    }
    Some(())
}

fn rewrite_plan(ctx: &mut Context, plan: AllocaPlan) -> usize {
    let load_count = plan.loads.len();
    let mut rewriter = IRRewriter::<Recorder>::default();

    for rewrite in plan.loads {
        let location = rewrite.load.deref(ctx).loc().clone();
        let aggregate_value = match plan.source {
            AggregateSource::Direct(value) => value,
            AggregateSource::Slot {
                pointer,
                aggregate_type,
            } => {
                let load = Operation::new(
                    ctx,
                    MirLoadOp::get_concrete_op_info(),
                    vec![aggregate_type],
                    vec![pointer],
                    vec![],
                    0,
                );
                load.deref_mut(ctx).set_loc(location.clone());
                load.insert_before(ctx, rewrite.load);
                load.deref(ctx).get_result(0)
            }
        };

        let extract_field = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![rewrite.array_type],
            vec![aggregate_value],
            vec![],
            0,
        );
        extract_field.deref_mut(ctx).set_loc(location.clone());
        MirExtractFieldOp::new(extract_field)
            .set_attr_index(ctx, FieldIndexAttr(rewrite.field_index));
        extract_field.insert_before(ctx, rewrite.load);
        let array_value = extract_field.deref(ctx).get_result(0);

        let extract_element = Operation::new(
            ctx,
            MirExtractArrayElementOp::get_concrete_op_info(),
            vec![rewrite.element_type],
            vec![array_value, rewrite.index],
            vec![],
            0,
        );
        extract_element.deref_mut(ctx).set_loc(location.clone());
        extract_element.insert_before(ctx, rewrite.load);
        let mut replacement = extract_element.deref(ctx).get_result(0);

        for nested_field in rewrite.nested_fields {
            let extract_field = Operation::new(
                ctx,
                MirExtractFieldOp::get_concrete_op_info(),
                vec![nested_field.result_type],
                vec![replacement],
                vec![],
                0,
            );
            extract_field.deref_mut(ctx).set_loc(location.clone());
            MirExtractFieldOp::new(extract_field)
                .set_attr_index(ctx, FieldIndexAttr(nested_field.field_index));
            extract_field.insert_before(ctx, rewrite.load);
            replacement = extract_field.deref(ctx).get_result(0);
        }
        assert!(
            replacement.get_type(ctx) == rewrite.result_type,
            "validated aggregate projection changed the scalar result type"
        );

        let old_result = rewrite.load.deref(ctx).get_result(0);
        old_result.replace_all_uses_with(ctx, &replacement);
        rewriter.erase_operation(ctx, rewrite.load);
    }

    // Loads are gone, so the exact validated pointer chain is dead. Erase it
    // from leaves to root through the rewriter so linked-list bookkeeping and
    // use-list updates remain valid for the immediately following mem2reg pass.
    for nested_field_addr in plan.nested_field_addrs.into_iter().rev() {
        rewriter.erase_operation(ctx, nested_field_addr);
    }
    for array_addr in plan.array_addrs.into_iter().rev() {
        rewriter.erase_operation(ctx, array_addr);
    }
    for field_addr in plan.field_addrs.into_iter().rev() {
        rewriter.erase_operation(ctx, field_addr);
    }

    load_count
}

#[derive(Clone, Copy)]
enum TrivialComparisonKind {
    Eq,
    Ne,
}

/// Inline trivial read-only scalar comparisons fed by an indexed aggregate.
///
/// Rust's primitive `PartialEq` methods take references, even when their body
/// is only two non-volatile scalar loads and one comparison. After the first
/// mem2reg pass, recognizing that exact helper shape exposes the terminal load
/// behind an aggregate's field/index projection. The aggregate canonicalizer
/// can then replace the address chain with SSA extraction before a second
/// mem2reg pass.
///
/// Helpers with additional blocks, operations, pointer uses, or side effects
/// fail closed. Calls without an indexed aggregate operand are left alone.
pub fn canonicalize_trivial_indexed_comparison_calls(
    module: Ptr<Operation>,
    ctx: &mut Context,
    verbose: bool,
) -> usize {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);

    let comparisons: HashMap<_, _> = operations
        .iter()
        .filter_map(|operation| {
            let function = Operation::get_op::<MirFuncOp>(*operation, ctx)?;
            let comparison = analyze_trivial_comparison(ctx, *operation)?;
            Some((String::from(function.get_symbol_name(ctx)), comparison))
        })
        .collect();
    let calls: Vec<_> = operations
        .into_iter()
        .filter(|operation| Operation::get_op::<MirCallOp>(*operation, ctx).is_some())
        .collect();

    let mut rewritten_calls = 0usize;
    for call in calls {
        let call_op = Operation::get_op::<MirCallOp>(call, ctx)
            .expect("call list contains only mir.call operations");
        let Some(callee) = call_op
            .get_attr_callee(ctx)
            .map(|attribute| String::from((*attribute).clone()))
        else {
            continue;
        };
        let Some(&comparison) = comparisons.get(&callee) else {
            continue;
        };
        if call.deref(ctx).get_num_operands() != 2
            || call.deref(ctx).get_num_results() != 1
            || !(0..2).any(|index| {
                pointer_has_indexed_aggregate_projection(ctx, call.deref(ctx).get_operand(index))
            })
        {
            continue;
        }
        let Some(pointee_types) = (0..2)
            .map(|index| {
                let pointer_type = call.deref(ctx).get_operand(index).get_type(ctx);
                let pointer_type_ref = pointer_type.deref(ctx);
                let pointer_type = pointer_type_ref.downcast_ref::<MirPtrType>()?;
                (!pointer_type.is_mutable).then_some(pointer_type.pointee)
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if pointee_types[0] != pointee_types[1] {
            continue;
        }

        rewrite_trivial_comparison_call(ctx, call, comparison, &pointee_types);
        rewritten_calls += 1;
    }

    if rewritten_calls > 0 && verbose {
        eprintln!("indexed scalar comparison canonicalization: rewrote {rewritten_calls} call(s)");
    }
    rewritten_calls
}

fn analyze_trivial_comparison(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Option<TrivialComparisonKind> {
    Operation::get_op::<MirFuncOp>(function, ctx)?;
    let region = function.deref(ctx).get_region(0);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let [block] = blocks.as_slice() else {
        return None;
    };
    let arguments: Vec<_> = block.deref(ctx).arguments().collect();
    let [lhs_pointer, rhs_pointer] = arguments.as_slice() else {
        return None;
    };
    for &pointer in &arguments {
        let pointer_type = pointer.get_type(ctx);
        let pointer_type_ref = pointer_type.deref(ctx);
        let pointer_type = pointer_type_ref.downcast_ref::<MirPtrType>()?;
        if pointer_type.is_mutable || pointer.num_uses(ctx) != 1 {
            return None;
        }
    }

    let mut loads = Vec::new();
    let mut comparison = None;
    let mut return_op = None;
    for operation in block.deref(ctx).iter(ctx) {
        if let Some(load) = Operation::get_op::<MirLoadOp>(operation, ctx) {
            if load.is_volatile(ctx) {
                return None;
            }
            loads.push(operation);
        } else if Operation::get_op::<MirEqOp>(operation, ctx).is_some() {
            if comparison
                .replace((operation, TrivialComparisonKind::Eq))
                .is_some()
            {
                return None;
            }
        } else if Operation::get_op::<MirNeOp>(operation, ctx).is_some() {
            if comparison
                .replace((operation, TrivialComparisonKind::Ne))
                .is_some()
            {
                return None;
            }
        } else if Operation::get_op::<MirReturnOp>(operation, ctx).is_some() {
            if return_op.replace(operation).is_some() {
                return None;
            }
        } else if Operation::get_op::<MirStorageLiveOp>(operation, ctx).is_none()
            && Operation::get_op::<MirStorageDeadOp>(operation, ctx).is_none()
        {
            return None;
        }
    }

    let [lhs_load, rhs_load] = loads.as_slice() else {
        return None;
    };
    let (comparison, kind) = comparison?;
    let return_op = return_op?;
    if lhs_load.deref(ctx).get_operand(0) != *lhs_pointer
        || rhs_load.deref(ctx).get_operand(0) != *rhs_pointer
    {
        return None;
    }
    let lhs = lhs_load.deref(ctx).get_result(0);
    let rhs = rhs_load.deref(ctx).get_result(0);
    let result = comparison.deref(ctx).get_result(0);
    if lhs.num_uses(ctx) != 1
        || rhs.num_uses(ctx) != 1
        || result.num_uses(ctx) != 1
        || comparison.deref(ctx).get_operand(0) != lhs
        || comparison.deref(ctx).get_operand(1) != rhs
        || return_op.deref(ctx).get_num_operands() != 1
        || return_op.deref(ctx).get_operand(0) != result
    {
        return None;
    }
    Some(kind)
}

fn pointer_has_indexed_aggregate_projection(ctx: &Context, mut pointer: Value) -> bool {
    loop {
        let Some(defining_op) = pointer.defining_op() else {
            return false;
        };
        if Operation::get_op::<MirArrayElementAddrOp>(defining_op, ctx).is_some() {
            return true;
        }
        if Operation::get_op::<MirFieldAddrOp>(defining_op, ctx).is_some() {
            pointer = defining_op.deref(ctx).get_operand(0);
            continue;
        }
        if let Some(cast) = Operation::get_op::<MirCastOp>(defining_op, ctx)
            && cast
                .get_attr_cast_kind(ctx)
                .is_some_and(|kind| matches!(*kind, MirCastKindAttr::PtrToPtr))
        {
            pointer = defining_op.deref(ctx).get_operand(0);
            continue;
        }
        return false;
    }
}

fn rewrite_trivial_comparison_call(
    ctx: &mut Context,
    call: Ptr<Operation>,
    comparison: TrivialComparisonKind,
    pointee_types: &[TypeHandle],
) {
    let location = call.deref(ctx).loc().clone();
    let operands: Vec<_> = (0..2)
        .map(|index| call.deref(ctx).get_operand(index))
        .collect();
    let result_type = call.deref(ctx).get_result(0).get_type(ctx);
    let mut loaded = Vec::with_capacity(2);
    for (index, pointee_type) in pointee_types.iter().copied().enumerate() {
        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![pointee_type],
            vec![operands[index]],
            vec![],
            0,
        );
        load.deref_mut(ctx).set_loc(location.clone());
        load.insert_before(ctx, call);
        loaded.push(load.deref(ctx).get_result(0));
    }

    let comparison_info = match comparison {
        TrivialComparisonKind::Eq => MirEqOp::get_concrete_op_info(),
        TrivialComparisonKind::Ne => MirNeOp::get_concrete_op_info(),
    };
    let replacement = Operation::new(ctx, comparison_info, vec![result_type], loaded, vec![], 0);
    replacement.deref_mut(ctx).set_loc(location);
    replacement.insert_before(ctx, call);
    let replacement_value = replacement.deref(ctx).get_result(0);
    let old_result = call.deref(ctx).get_result(0);
    old_result.replace_all_uses_with(ctx, &replacement_value);

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, call);
}

#[derive(Clone, Copy)]
enum BoundedPointerIndex {
    /// The original index is already `mir.rem(value, constant)`.
    Direct(Value),
    /// The assertion-success block proves `index < bound`. Re-materialize the
    /// equivalent remainder so the typed LLVM lowering sees the bounded shape.
    Asserted { index: Value, bound: Value },
}

struct BorrowedPointerPlan {
    field_pointer: Value,
    array_addr: Ptr<Operation>,
    load: Ptr<Operation>,
    array_type: TypeHandle,
    index: BoundedPointerIndex,
    result_type: TypeHandle,
}

/// Rewrite bounded read-only array loads through immutable aggregate pointer
/// arguments after mem2reg.
///
/// This phase is intentionally narrow. The aggregate pointer must be an entry
/// argument of an `alwaysinline` function, every derived pointer must be
/// immutable, and both pointer results must have exactly one use. The index
/// must be bounded either by an unsigned remainder or by the unique predecessor
/// assertion `assert(index < constant)`.
///
/// On top of the helper-local shape, every call site of the helper must prove
/// that the aggregate pointer targets caller-private memory (see
/// `all_call_sites_pass_owned_aggregate`); any unproven call site keeps the
/// helper untouched.
///
/// `verbose` is threaded from the pipeline's backend options; the pass itself
/// never reads the environment.
pub fn canonicalize_bounded_borrowed_pointer_arguments(
    module: Ptr<Operation>,
    ctx: &mut Context,
    verbose: bool,
) {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);

    let mut calls_by_callee: HashMap<String, Vec<Ptr<Operation>>> = HashMap::new();
    let mut array_addrs = Vec::new();
    for operation in operations {
        if let Some(call) = Operation::get_op::<MirCallOp>(operation, ctx) {
            let callee = call
                .get_attr_callee(ctx)
                .map(|attribute| String::from((*attribute).clone()));
            if let Some(callee) = callee {
                calls_by_callee.entry(callee).or_default().push(operation);
            }
            continue;
        }
        if Operation::get_op::<MirArrayElementAddrOp>(operation, ctx).is_some() {
            array_addrs.push(operation);
        }
    }

    let mut provenance_cache: HashMap<(Ptr<Operation>, usize), bool> = HashMap::new();
    let mut rewritten_loads = 0usize;
    for array_addr in array_addrs {
        let Some(plan) =
            analyze_borrowed_pointer_read(ctx, array_addr, &calls_by_callee, &mut provenance_cache)
        else {
            continue;
        };
        rewrite_borrowed_pointer_read(ctx, plan);
        rewritten_loads += 1;
    }

    if rewritten_loads > 0 && verbose {
        eprintln!(
            "borrowed-pointer aggregate scalarization: rewrote \
             {rewritten_loads} dynamic load(s)"
        );
    }
}

fn analyze_borrowed_pointer_read(
    ctx: &Context,
    array_addr: Ptr<Operation>,
    calls_by_callee: &HashMap<String, Vec<Ptr<Operation>>>,
    provenance_cache: &mut HashMap<(Ptr<Operation>, usize), bool>,
) -> Option<BorrowedPointerPlan> {
    Operation::get_op::<MirArrayElementAddrOp>(array_addr, ctx)?;
    let load_block = array_addr.deref(ctx).get_parent_block()?;

    let field_pointer = array_addr.deref(ctx).get_operand(0);
    let field_addr = field_pointer.defining_op()?;
    let field = Operation::get_op::<MirFieldAddrOp>(field_addr, ctx)?;
    if field_addr.deref(ctx).get_parent_block() != Some(load_block)
        || field_pointer.num_uses(ctx) != 1
    {
        return None;
    }
    let field_use = field_pointer.uses(ctx).into_iter().next()?;
    if field_use.user_op() != array_addr || field_use.find_index(ctx) != 0 {
        return None;
    }

    let field_pointer_type = field_pointer.get_type(ctx);
    let field_pointer_type_ref = field_pointer_type.deref(ctx);
    let field_pointer_type = field_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if field_pointer_type.is_mutable {
        return None;
    }
    let array_type = field_pointer_type.pointee;
    let array_type_ref = array_type.deref(ctx);
    let array_size = array_type_ref.downcast_ref::<MirArrayType>()?.size();
    if array_size == 0 {
        return None;
    }

    let element_pointer = array_addr.deref(ctx).get_result(0);
    let element_pointer_type = element_pointer.get_type(ctx);
    let element_pointer_type_ref = element_pointer_type.deref(ctx);
    let element_pointer_type = element_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if element_pointer_type.is_mutable || element_pointer.num_uses(ctx) != 1 {
        return None;
    }

    let element_use = element_pointer.uses(ctx).into_iter().next()?;
    if element_use.find_index(ctx) != 0 {
        return None;
    }
    let load = element_use.user_op();
    let load_op = Operation::get_op::<MirLoadOp>(load, ctx)?;
    if load_op.is_volatile(ctx) || load.deref(ctx).get_parent_block() != Some(load_block) {
        return None;
    }

    let aggregate_pointer = field_addr.deref(ctx).get_operand(0);
    let entry_block = aggregate_pointer.defining_block()?;
    let region = entry_block.deref(ctx).get_parent_region()?;
    if region.deref(ctx).iter(ctx).next() != Some(entry_block) {
        return None;
    }

    let function = entry_block.deref(ctx).get_parent_op(ctx)?;
    Operation::get_op::<MirFuncOp>(function, ctx)?;
    let alwaysinline_key: pliron::identifier::Identifier = "alwaysinline".try_into().ok()?;
    function
        .deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&alwaysinline_key)?;

    let aggregate_pointer_type = aggregate_pointer.get_type(ctx);
    let aggregate_pointer_type_ref = aggregate_pointer_type.deref(ctx);
    let aggregate_pointer_type = aggregate_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if aggregate_pointer_type.is_mutable {
        return None;
    }
    let aggregate_type = aggregate_pointer_type.pointee;
    aggregate_type.deref(ctx).downcast_ref::<MirStructType>()?;

    let argument_index = entry_block
        .deref(ctx)
        .arguments()
        .position(|argument| argument == aggregate_pointer)?;
    let provenance_key = (function, argument_index);
    let caller_owned = match provenance_cache.get(&provenance_key) {
        Some(&caller_owned) => caller_owned,
        None => {
            let caller_owned = all_call_sites_pass_owned_aggregate(
                ctx,
                calls_by_callee,
                function,
                argument_index,
                aggregate_type,
            );
            provenance_cache.insert(provenance_key, caller_owned);
            caller_owned
        }
    };
    if !caller_owned {
        return None;
    }

    let index_value = array_addr.deref(ctx).get_operand(1);
    let index = bounded_pointer_index(ctx, index_value, load_block, array_size)?;

    // Keep the field projection itself. Loading the bounded array field is
    // narrower than loading the complete aggregate and gives LLVM a constant
    // field address to forward after the helper is inlined.
    field.get_attr_field_index(ctx)?;

    Some(BorrowedPointerPlan {
        field_pointer,
        array_addr,
        load,
        array_type,
        index,
        result_type: load.deref(ctx).get_result(0).get_type(ctx),
    })
}

fn bounded_pointer_index(
    ctx: &Context,
    index: Value,
    load_block: Ptr<pliron::basic_block::BasicBlock>,
    array_size: u64,
) -> Option<BoundedPointerIndex> {
    let index_type = index.get_type(ctx);
    let index_type_ref = index_type.deref(ctx);
    let integer_type = index_type_ref.downcast_ref::<IntegerType>()?;
    if integer_type.signedness() != Signedness::Unsigned {
        return None;
    }

    if let Some(defining_op) = index.defining_op()
        && Operation::get_op::<MirRemOp>(defining_op, ctx).is_some()
    {
        let divisor = defining_op.deref(ctx).get_operand(1);
        let candidate_count = integer_constant_u64(ctx, divisor)?;
        validate_candidate_count(candidate_count, array_size)?;
        return Some(BoundedPointerIndex::Direct(index));
    }

    let region = load_block.deref(ctx).get_parent_region()?;
    let predecessors = region.predecessors(ctx, &load_block);
    let [assert_block] = predecessors.as_slice() else {
        return None;
    };
    let terminator = assert_block.deref(ctx).get_terminator(ctx)?;
    Operation::get_op::<MirAssertOp>(terminator, ctx)?;
    if terminator.deref(ctx).get_num_successors() != 1
        || terminator.deref(ctx).get_successor(0) != load_block
    {
        return None;
    }

    let condition = terminator.deref(ctx).get_operand(0);
    let comparison = condition.defining_op()?;
    Operation::get_op::<MirLtOp>(comparison, ctx)?;
    if comparison.deref(ctx).get_parent_block() != Some(*assert_block)
        || comparison.deref(ctx).get_operand(0) != index
    {
        return None;
    }

    let bound = comparison.deref(ctx).get_operand(1);
    if bound.get_type(ctx) != index_type {
        return None;
    }
    let candidate_count = integer_constant_u64(ctx, bound)?;
    validate_candidate_count(candidate_count, array_size)?;
    Some(BoundedPointerIndex::Asserted { index, bound })
}

/// Decide whether every visible call site of `function` passes the pointer
/// argument at `argument_index` into caller-private memory.
///
/// Issue #400 fail-closed rule (the #398 precedent): a borrowed pointer that
/// may reference global, shared, or otherwise external memory must keep
/// exactly one dynamic memory access, so the widened array-field load is only
/// legal when every caller passes the address of a compiler-owned local slot
/// holding exactly the helper's aggregate type. Externally callable device
/// exports have call sites this module cannot see, and a helper without any
/// visible call site proves nothing; both disqualify the helper outright.
fn all_call_sites_pass_owned_aggregate(
    ctx: &Context,
    calls_by_callee: &HashMap<String, Vec<Ptr<Operation>>>,
    function: Ptr<Operation>,
    argument_index: usize,
    aggregate_type: TypeHandle,
) -> bool {
    let Some(function_op) = Operation::get_op::<MirFuncOp>(function, ctx) else {
        return false;
    };
    let symbol = String::from(function_op.get_symbol_name(ctx));
    if reserved_oxide_symbols::is_device_symbol(&symbol) {
        return false;
    }
    let Some(calls) = calls_by_callee.get(&symbol) else {
        return false;
    };
    !calls.is_empty()
        && calls.iter().all(|call| {
            let call_ref = call.deref(ctx);
            if argument_index >= call_ref.get_num_operands() {
                return false;
            }
            let pointer = call_ref.get_operand(argument_index);
            drop(call_ref);
            pointer_is_owned_aggregate_slot(ctx, pointer, aggregate_type)
        })
}

/// Trace `pointer` back to its allocation through pointer-identity casts
/// (an `&mut slot -> &slot` reborrow imports as `mir.cast PtrToPtr`).
///
/// Accept only a function-local `mir.alloca` whose pointee is exactly the
/// helper's aggregate type: reading a whole array field stays inside the
/// allocation only when the slot and the callee agree on the layout. Block
/// arguments (kernel pointer parameters, phi merges, pointers forwarded from
/// another helper's parameter) and every other producer (shared or global
/// allocations, selects, offsets) fail closed.
fn pointer_is_owned_aggregate_slot(
    ctx: &Context,
    mut pointer: Value,
    aggregate_type: TypeHandle,
) -> bool {
    loop {
        let Some(defining_op) = pointer.defining_op() else {
            return false;
        };
        if let Some(cast) = Operation::get_op::<MirCastOp>(defining_op, ctx) {
            let is_pointer_identity_cast = cast
                .get_attr_cast_kind(ctx)
                .is_some_and(|kind| matches!(*kind, MirCastKindAttr::PtrToPtr));
            if !is_pointer_identity_cast {
                return false;
            }
            pointer = defining_op.deref(ctx).get_operand(0);
            continue;
        }
        let Some(alloca) = Operation::get_op::<MirAllocaOp>(defining_op, ctx) else {
            return false;
        };
        return alloca.pointee_type(ctx) == aggregate_type;
    }
}

fn integer_constant_u64(ctx: &Context, value: Value) -> Option<u64> {
    let defining_op = value.defining_op()?;
    let constant = Operation::get_op::<MirConstantOp>(defining_op, ctx)?;
    let attribute = constant.get_attr_value(ctx)?;
    let constant_value = attribute.value();
    // `APInt::to_u64` truncates wider values, so a >64-bit constant could be
    // misread as a small in-range bound. Fail closed on such widths.
    (constant_value.bw() <= 64).then(|| constant_value.to_u64())
}

fn validate_candidate_count(candidate_count: u64, array_size: u64) -> Option<()> {
    (candidate_count > 0
        && candidate_count <= array_size
        && candidate_count <= MAX_SCALARIZED_CANDIDATES)
        .then_some(())
}

fn rewrite_borrowed_pointer_read(ctx: &mut Context, plan: BorrowedPointerPlan) {
    let location = plan.load.deref(ctx).loc().clone();
    let bounded_index = match plan.index {
        BoundedPointerIndex::Direct(index) => index,
        BoundedPointerIndex::Asserted { index, bound } => {
            let remainder = Operation::new(
                ctx,
                MirRemOp::get_concrete_op_info(),
                vec![index.get_type(ctx)],
                vec![index, bound],
                vec![],
                0,
            );
            remainder.deref_mut(ctx).set_loc(location.clone());
            remainder.insert_before(ctx, plan.load);
            remainder.deref(ctx).get_result(0)
        }
    };

    // Load only the addressed array field at the original access point. The
    // source pointer is immutable and the helper is alwaysinline, so LLVM can
    // forward the caller's by-value aggregate after inlining. Keeping the
    // constant field projection avoids widening the access to the whole struct.
    let array_load = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![plan.array_type],
        vec![plan.field_pointer],
        vec![],
        0,
    );
    array_load.deref_mut(ctx).set_loc(location.clone());
    array_load.insert_before(ctx, plan.load);
    let array_value = array_load.deref(ctx).get_result(0);

    let extract_element = Operation::new(
        ctx,
        MirExtractArrayElementOp::get_concrete_op_info(),
        vec![plan.result_type],
        vec![array_value, bounded_index],
        vec![],
        0,
    );
    extract_element.deref_mut(ctx).set_loc(location);
    extract_element.insert_before(ctx, plan.load);
    let replacement = extract_element.deref(ctx).get_result(0);

    let old_result = plan.load.deref(ctx).get_result(0);
    old_result.replace_all_uses_with(ctx, &replacement);

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, plan.load);
    rewriter.erase_operation(ctx, plan.array_addr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        ops::{MirGotoOp, MirReturnOp},
        types::MirArrayType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{IntegerAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::FunctionType,
        },
        region::Region,
        utils::apint::APInt,
    };
    use std::num::NonZeroUsize;

    struct Fixture {
        module: Ptr<Operation>,
        alloca: Ptr<Operation>,
    }

    #[derive(Default)]
    struct FixtureOptions {
        additional_store: bool,
        volatile_load: bool,
        store_in_body: bool,
        nested_scalar_wrapper: bool,
        comparison_call: bool,
    }

    fn build_fixture(
        ctx: &mut Context,
        array_size: u64,
        divisor: Option<u64>,
        options: FixtureOptions,
    ) -> Fixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index_handle: TypeHandle = index_type.into();
        let array_element_type = if options.nested_scalar_wrapper {
            MirStructType::get_with_full_layout(
                ctx,
                "ScalarWrapper".into(),
                vec!["value".into()],
                vec![element_type],
                vec![0],
                vec![0],
                4,
                4,
            )
            .into()
        } else {
            element_type
        };
        let array_type: TypeHandle = MirArrayType::get(ctx, array_element_type, array_size).into();
        let aggregate_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "BorrowedAggregate".into(),
            vec!["values".into()],
            vec![array_type],
            vec![0],
            vec![0],
            array_size * 4,
            4,
        )
        .into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![aggregate_type, index_handle], vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, "kernel".try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![aggregate_type, index_handle]);
        entry.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let aggregate_argument = entry.deref(ctx).get_argument(0);
        let raw_index = entry.deref(ctx).get_argument(1);

        let aggregate_pointer: TypeHandle =
            MirPtrType::get_generic(ctx, aggregate_type, true).into();
        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![aggregate_pointer],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, aggregate_argument],
            vec![],
            0,
        );
        if !options.store_in_body {
            store.insert_at_back(entry, ctx);
        }

        if options.additional_store {
            let second_store = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![slot, aggregate_argument],
                vec![],
                0,
            );
            second_store.insert_at_back(entry, ctx);
        }

        let goto = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![body],
            0,
        );
        goto.insert_at_back(entry, ctx);

        let index = if let Some(divisor) = divisor {
            let divisor_attribute = IntegerAttr::new(
                index_type,
                APInt::from_u64(divisor, NonZeroUsize::new(64).unwrap()),
            );
            let constant = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![index_handle],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(constant).set_attr_value(ctx, divisor_attribute);
            constant.insert_at_back(body, ctx);
            let divisor_value = constant.deref(ctx).get_result(0);

            let rem = Operation::new(
                ctx,
                MirRemOp::get_concrete_op_info(),
                vec![index_handle],
                vec![raw_index, divisor_value],
                vec![],
                0,
            );
            rem.insert_at_back(body, ctx);
            rem.deref(ctx).get_result(0)
        } else {
            raw_index
        };

        if options.store_in_body {
            store.insert_at_back(body, ctx);
        }

        let field_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, false).into();
        let field = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![field_pointer],
            vec![slot],
            vec![],
            0,
        );
        MirFieldAddrOp::new(field).set_attr_field_index(ctx, FieldIndexAttr(0));
        field.insert_at_back(body, ctx);
        let field_value = field.deref(ctx).get_result(0);

        let element_pointer: TypeHandle =
            MirPtrType::get_generic(ctx, array_element_type, false).into();
        let element_address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![field_value, index],
            vec![],
            0,
        );
        element_address.insert_at_back(body, ctx);
        let element_pointer_value = element_address.deref(ctx).get_result(0);

        let scalar_pointer = if options.nested_scalar_wrapper {
            let pointer_type: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
            let field = Operation::new(
                ctx,
                MirFieldAddrOp::get_concrete_op_info(),
                vec![pointer_type],
                vec![element_pointer_value],
                vec![],
                0,
            );
            MirFieldAddrOp::new(field).set_attr_field_index(ctx, FieldIndexAttr(0));
            field.insert_at_back(body, ctx);
            field.deref(ctx).get_result(0)
        } else {
            element_pointer_value
        };

        if options.comparison_call {
            append_trivial_comparison_helper(ctx, module.get_operation(), element_type);
            let call = Operation::new(
                ctx,
                MirCallOp::get_concrete_op_info(),
                vec![IntegerType::get(ctx, 1, Signedness::Signless).into()],
                vec![scalar_pointer, scalar_pointer],
                vec![],
                0,
            );
            MirCallOp::new(call).set_attr_callee(
                ctx,
                pliron::builtin::attributes::StringAttr::new("trivial_ne".to_string()),
            );
            call.insert_at_back(body, ctx);
        } else {
            let load = Operation::new(
                ctx,
                MirLoadOp::get_concrete_op_info(),
                vec![element_type],
                vec![scalar_pointer],
                vec![],
                0,
            );
            if options.volatile_load {
                MirLoadOp::new(load).set_volatile(ctx, true);
            }
            load.insert_at_back(body, ctx);
        }

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        Fixture {
            module: module.get_operation(),
            alloca,
        }
    }

    fn append_trivial_comparison_helper(
        ctx: &mut Context,
        module: Ptr<Operation>,
        element_type: TypeHandle,
    ) {
        let pointer_type: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
        let result_type: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let function_type =
            FunctionType::get(ctx, vec![pointer_type, pointer_type], vec![result_type]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, "trivial_ne".try_into().unwrap());
        let module_region = module.deref(ctx).get_region(0);
        let module_block = module_region
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("fixture module has one block");
        function.insert_at_back(module_block, ctx);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![pointer_type, pointer_type]);
        entry.insert_at_back(region, ctx);
        let lhs_pointer = entry.deref(ctx).get_argument(0);
        let rhs_pointer = entry.deref(ctx).get_argument(1);

        let mut loaded = Vec::new();
        for pointer in [lhs_pointer, rhs_pointer] {
            let load = Operation::new(
                ctx,
                MirLoadOp::get_concrete_op_info(),
                vec![element_type],
                vec![pointer],
                vec![],
                0,
            );
            load.insert_at_back(entry, ctx);
            loaded.push(load.deref(ctx).get_result(0));
        }
        let comparison = Operation::new(
            ctx,
            MirNeOp::get_concrete_op_info(),
            vec![result_type],
            loaded,
            vec![],
            0,
        );
        comparison.insert_at_back(entry, ctx);
        let result = comparison.deref(ctx).get_result(0);
        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );
        return_op.insert_at_back(entry, ctx);
    }

    fn count<T: Op>(ctx: &Context, root: Ptr<Operation>) -> usize {
        let mut operations = Vec::new();
        collect_ops(ctx, root, &mut operations);
        operations
            .into_iter()
            .filter(|operation| Operation::get_op::<T>(*operation, ctx).is_some())
            .count()
    }

    #[test]
    fn bounded_rem_rewrites_large_array_with_small_candidate_set() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 64, Some(3), FixtureOptions::default());

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert!(
            Operation::get_op::<MirAllocaOp>(fixture.alloca, &ctx).is_some(),
            "mem2reg, not this pass, owns erasing the entry slot"
        );
    }

    #[test]
    fn unbounded_index_is_canonicalized_for_lowering_fallback() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 3, None, FixtureOptions::default());

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn oversized_candidate_set_is_canonicalized_for_lowering_fallback() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 64, Some(17), FixtureOptions::default());

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn additional_store_rejects_the_entire_slot() {
        let mut ctx = Context::new();
        let fixture = build_fixture(
            &mut ctx,
            3,
            Some(3),
            FixtureOptions {
                additional_store: true,
                ..FixtureOptions::default()
            },
        );

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn volatile_load_rejects_the_entire_slot() {
        let mut ctx = Context::new();
        let fixture = build_fixture(
            &mut ctx,
            3,
            Some(3),
            FixtureOptions {
                volatile_load: true,
                ..FixtureOptions::default()
            },
        );

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn body_initialized_local_is_canonicalized_through_a_whole_value_load() {
        let mut ctx = Context::new();
        let fixture = build_fixture(
            &mut ctx,
            3,
            Some(3),
            FixtureOptions {
                store_in_body: true,
                ..FixtureOptions::default()
            },
        );

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);

        let mut analyses = pliron::pass::AnalysisManager::default();
        pliron::opts::mem2reg::mem2reg(fixture.module, &mut ctx, &mut analyses)
            .expect("the whole-value access makes the local promotable");

        assert_eq!(count::<MirAllocaOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn indexed_newtype_comparison_becomes_promotable_scalar_extraction() {
        let mut ctx = Context::new();
        let fixture = build_fixture(
            &mut ctx,
            3,
            Some(3),
            FixtureOptions {
                store_in_body: true,
                nested_scalar_wrapper: true,
                comparison_call: true,
                ..FixtureOptions::default()
            },
        );

        assert_eq!(count::<MirCallOp>(&ctx, fixture.module), 1);
        assert_eq!(
            canonicalize_trivial_indexed_comparison_calls(fixture.module, &mut ctx, false),
            1
        );
        assert_eq!(count::<MirCallOp>(&ctx, fixture.module), 0);
        assert_eq!(
            canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false),
            2
        );
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);

        let mut analyses = pliron::pass::AnalysisManager::default();
        pliron::opts::mem2reg::mem2reg(fixture.module, &mut ctx, &mut analyses)
            .expect("nested scalar extraction makes the aggregate local promotable");

        assert_eq!(count::<MirAllocaOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(
            count::<MirLoadOp>(&ctx, fixture.module),
            2,
            "only the two loads inside the comparison helper remain"
        );
    }

    struct BorrowedPointerFixture {
        module: Ptr<Operation>,
    }

    /// Who calls the borrowed-pointer helper, and what backs the pointer.
    #[derive(Clone, Copy)]
    enum CallerShape {
        /// Every call site passes the address of a caller-local slot.
        OwnedSlot,
        /// The single call site forwards the caller's own pointer parameter.
        PointerParameter,
        /// One owned-slot call site plus one forwarded-parameter call site.
        Mixed,
        /// The helper has no call site in the module.
        None,
    }

    fn add_helper_call(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        helper_symbol: &str,
        pointer: Value,
        index: Value,
        element_type: TypeHandle,
    ) {
        let call = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![element_type],
            vec![pointer, index],
            vec![],
            0,
        );
        MirCallOp::new(call).set_attr_callee(
            ctx,
            pliron::builtin::attributes::StringAttr::new(helper_symbol.to_string()),
        );
        call.insert_at_back(block, ctx);

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        return_op.insert_at_back(block, ctx);
    }

    fn add_caller_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        argument_types: Vec<TypeHandle>,
    ) -> Ptr<BasicBlock> {
        let function_type = FunctionType::get(ctx, argument_types.clone(), vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, name.try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, argument_types);
        entry.insert_at_back(region, ctx);
        entry
    }

    /// The type handles a fixture caller needs to call the helper.
    #[derive(Clone, Copy)]
    struct CallerTypes {
        aggregate_type: TypeHandle,
        aggregate_pointer: TypeHandle,
        index_handle: TypeHandle,
        element_type: TypeHandle,
    }

    /// A caller holding the aggregate by value in a local slot, calling the
    /// helper with a `&mut slot -> &slot` reborrow of that slot's address.
    fn add_owned_slot_caller(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        helper_symbol: &str,
        types: CallerTypes,
    ) {
        let CallerTypes {
            aggregate_type,
            aggregate_pointer,
            index_handle,
            element_type,
        } = types;
        let entry = add_caller_function(ctx, module, name, vec![aggregate_type, index_handle]);
        let aggregate_argument = entry.deref(ctx).get_argument(0);
        let index = entry.deref(ctx).get_argument(1);

        let slot_pointer: TypeHandle = MirPtrType::get_generic(ctx, aggregate_type, true).into();
        let slot = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![slot_pointer],
            vec![],
            vec![],
            0,
        );
        slot.insert_at_back(entry, ctx);
        let slot_value = slot.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot_value, aggregate_argument],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);

        let reborrow = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![aggregate_pointer],
            vec![slot_value],
            vec![],
            0,
        );
        MirCastOp::new(reborrow).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        reborrow.insert_at_back(entry, ctx);
        let reborrow_value = reborrow.deref(ctx).get_result(0);

        add_helper_call(
            ctx,
            entry,
            helper_symbol,
            reborrow_value,
            index,
            element_type,
        );
    }

    /// A caller forwarding its own aggregate pointer parameter, i.e. memory
    /// this module cannot prove to be caller-private.
    fn add_pointer_parameter_caller(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        helper_symbol: &str,
        types: CallerTypes,
    ) {
        let entry = add_caller_function(
            ctx,
            module,
            name,
            vec![types.aggregate_pointer, types.index_handle],
        );
        let forwarded_pointer = entry.deref(ctx).get_argument(0);
        let index = entry.deref(ctx).get_argument(1);
        add_helper_call(
            ctx,
            entry,
            helper_symbol,
            forwarded_pointer,
            index,
            types.element_type,
        );
    }

    fn build_borrowed_pointer_fixture(
        ctx: &mut Context,
        asserted_bound: Option<u64>,
        alwaysinline: bool,
        volatile_load: bool,
        caller_shape: CallerShape,
        helper_symbol: &str,
    ) -> BorrowedPointerFixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index_handle: TypeHandle = index_type.into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, 3).into();
        let aggregate_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "BorrowedAggregate".into(),
            vec!["values".into()],
            vec![array_type],
            vec![0],
            vec![0],
            12,
            4,
        )
        .into();
        let aggregate_pointer: TypeHandle =
            MirPtrType::get_generic(ctx, aggregate_type, false).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(
            ctx,
            vec![aggregate_pointer, index_handle],
            vec![element_type],
        );
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, helper_symbol.try_into().unwrap());
        if alwaysinline {
            function.deref_mut(ctx).attributes.set(
                "alwaysinline".try_into().unwrap(),
                pliron::builtin::attributes::StringAttr::new("true".to_string()),
            );
        }
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![aggregate_pointer, index_handle]);
        entry.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let aggregate_argument = entry.deref(ctx).get_argument(0);
        let index = entry.deref(ctx).get_argument(1);

        if let Some(bound) = asserted_bound {
            let bound_attribute = IntegerAttr::new(
                index_type,
                APInt::from_u64(bound, NonZeroUsize::new(64).unwrap()),
            );
            let constant = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![index_handle],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(constant).set_attr_value(ctx, bound_attribute);
            constant.insert_at_back(entry, ctx);
            let bound_value = constant.deref(ctx).get_result(0);

            let i1_type: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
            let comparison = Operation::new(
                ctx,
                MirLtOp::get_concrete_op_info(),
                vec![i1_type],
                vec![index, bound_value],
                vec![],
                0,
            );
            comparison.insert_at_back(entry, ctx);
            let condition = comparison.deref(ctx).get_result(0);

            let assert = Operation::new(
                ctx,
                MirAssertOp::get_concrete_op_info(),
                vec![],
                vec![condition],
                vec![body],
                0,
            );
            assert.insert_at_back(entry, ctx);
        } else {
            let goto = Operation::new(
                ctx,
                MirGotoOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![body],
                0,
            );
            goto.insert_at_back(entry, ctx);
        }

        let field_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, false).into();
        let field = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![field_pointer],
            vec![aggregate_argument],
            vec![],
            0,
        );
        MirFieldAddrOp::new(field).set_attr_field_index(ctx, FieldIndexAttr(0));
        field.insert_at_back(body, ctx);
        let field_value = field.deref(ctx).get_result(0);

        let element_pointer: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
        let element_address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![field_value, index],
            vec![],
            0,
        );
        element_address.insert_at_back(body, ctx);
        let element_pointer_value = element_address.deref(ctx).get_result(0);

        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element_type],
            vec![element_pointer_value],
            vec![],
            0,
        );
        if volatile_load {
            MirLoadOp::new(load).set_volatile(ctx, true);
        }
        load.insert_at_back(body, ctx);
        let result = load.deref(ctx).get_result(0);

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        let caller_types = CallerTypes {
            aggregate_type,
            aggregate_pointer,
            index_handle,
            element_type,
        };
        match caller_shape {
            CallerShape::OwnedSlot => {
                add_owned_slot_caller(ctx, &module, "caller_owned", helper_symbol, caller_types);
            }
            CallerShape::PointerParameter => {
                add_pointer_parameter_caller(
                    ctx,
                    &module,
                    "caller_external",
                    helper_symbol,
                    caller_types,
                );
            }
            CallerShape::Mixed => {
                add_owned_slot_caller(ctx, &module, "caller_owned", helper_symbol, caller_types);
                add_pointer_parameter_caller(
                    ctx,
                    &module,
                    "caller_external",
                    helper_symbol,
                    caller_types,
                );
            }
            CallerShape::None => {}
        }

        BorrowedPointerFixture {
            module: module.get_operation(),
        }
    }

    #[test]
    fn asserted_immutable_pointer_read_is_canonicalized_after_mem2reg() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirRemOp>(&ctx, fixture.module), 1);
        assert_eq!(
            count::<MirLoadOp>(&ctx, fixture.module),
            1,
            "only the bounded array-field load introduced at the original access point remains"
        );
    }

    #[test]
    fn pointer_read_without_exact_assert_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            None,
            true,
            false,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn non_alwaysinline_pointer_helper_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            false,
            false,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn volatile_pointer_read_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            true,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    /// Asserts the helper kept its single dynamic memory access: the pointer
    /// chain survives, no value-level extraction or widened array-field load
    /// was introduced, and no bounded remainder was materialized.
    fn assert_single_dynamic_load_survives(ctx: &Context, module: Ptr<Operation>) {
        assert_eq!(count::<MirExtractArrayElementOp>(ctx, module), 0);
        assert_eq!(count::<MirExtractFieldOp>(ctx, module), 0);
        assert_eq!(count::<MirFieldAddrOp>(ctx, module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(ctx, module), 1);
        assert_eq!(count::<MirRemOp>(ctx, module), 0);
        assert_eq!(
            count::<MirLoadOp>(ctx, module),
            1,
            "the original dynamic element load must survive unwidened"
        );
    }

    #[test]
    fn pointer_parameter_call_site_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::PointerParameter,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }

    #[test]
    fn mixed_call_sites_are_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::Mixed,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }

    #[test]
    fn helper_without_visible_call_site_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::None,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }

    #[test]
    fn device_export_helper_is_left_unchanged() {
        // An exported `#[device]` function is externally callable, so the
        // module-level call scan cannot see every call site. Even an owned
        // in-module call site must not enable the rewrite.
        let mut ctx = Context::new();
        let exported_symbol = reserved_oxide_symbols::device_symbol("helper");
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::OwnedSlot,
            &exported_symbol,
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }
}
