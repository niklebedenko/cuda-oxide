/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Basic block translation: MIR block → Pliron IR block contents.
//!
//! Translates the contents of a single MIR basic block (statements + terminator)
//! into Pliron IR operations. The block itself is created by [`super::body`];
//! this module just populates it.
//!
//! # Translation order
//!
//! 1. Statements are translated in order, emitting loads/stores against the
//!    per-local alloca slots recorded in [`ValueMap`].
//! 2. Terminator is translated last and emits zero-operand control-flow ops
//!    (`mir.goto`, `mir.cond_br`, `mir.switch`, `mir.return`). Cross-block
//!    data flow happens via the slots, not block arguments.

use super::statement;
use super::terminator;
use crate::error::TranslationResult;
use crate::translator::values::ValueMap;
use dialect_mir::ops::MirUnreachableOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::{ConstantKind, RigidTy, TyKind};

fn is_panic_entry_name(name: &str) -> bool {
    name.contains("::panicking::") || name.contains("::rt::panic")
}

fn is_diverging_panic_call(term: &mir::Terminator) -> bool {
    let mir::TerminatorKind::Call {
        func, target: None, ..
    } = &term.kind
    else {
        return false;
    };

    let mir::Operand::Constant(const_op) = func else {
        return false;
    };
    let ConstantKind::ZeroSized = const_op.const_.kind() else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::FnDef(fn_def, _)) = const_op.const_.ty().kind() else {
        return false;
    };

    is_panic_entry_name(fn_def.name().as_str())
}

fn emit_unreachable(
    ctx: &mut Context,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        MirUnreachableOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }
    op
}

/// Translates a MIR basic block's contents into the corresponding Pliron IR block.
///
/// # Arguments
///
/// * `ctx` - Pliron IR context
/// * `body` - The full MIR body (needed for local declarations)
/// * `mir_block` - The MIR block to translate
/// * `_idx` - Block index (unused, kept for debugging)
/// * `block_ptr` - Target Pliron IR block (already created)
/// * `value_map` - MIR local → alloca slot mapping
/// * `block_map` - Block index → Pliron IR block mapping
/// * `legaliser` - Shared identifier legaliser for name uniqueness
/// * `entry_prev_op` - For the entry block only: the last op emitted by
///   `body::translate_body`'s alloca/store setup (see `emit_entry_allocas`),
///   so that statements are appended **after** that setup instead of being
///   inserted at the front. For every other block this must be `None`.
#[allow(clippy::too_many_arguments)]
pub fn translate_block(
    ctx: &mut Context,
    body: &mir::Body,
    mir_block: &mir::BasicBlock,
    _idx: usize,
    block_ptr: Ptr<BasicBlock>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    legaliser: &mut Legaliser,
    entry_prev_op: Option<Ptr<Operation>>,
) -> TranslationResult<()> {
    let mut prev_op: Option<Ptr<Operation>> = entry_prev_op;

    if is_diverging_panic_call(&mir_block.terminator) {
        let loc = Location::Named {
            name: format!("{:?}", mir_block.terminator.span),
            child_loc: Box::new(Location::Unknown),
        };
        emit_unreachable(ctx, block_ptr, prev_op, loc);
        return Ok(());
    }

    for stmt in &mir_block.statements {
        let op_ptr =
            statement::translate_statement(ctx, body, stmt, value_map, block_ptr, prev_op)?;
        prev_op = op_ptr;
    }

    let _term_op_ptr = terminator::translate_terminator(
        ctx,
        body,
        &mir_block.terminator,
        value_map,
        block_ptr,
        prev_op,
        block_map,
        legaliser,
    )?;

    Ok(())
}
