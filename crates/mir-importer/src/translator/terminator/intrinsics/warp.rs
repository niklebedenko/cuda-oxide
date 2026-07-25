/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Manual translation helper for warp reduction.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::MirCastOp;
use dialect_nvvm::ops::InlinePtxOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::utils::apint::APInt;
use rustc_public::mir;
use rustc_public::ty::{ConstantKind, TyConstKind};
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyShuffleMode {
    Up,
    Down,
    Xor,
    Idx,
}

impl LegacyShuffleMode {
    fn ptx_name(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Xor => "bfly",
            Self::Idx => "idx",
        }
    }
}

fn constant_operand_u32(
    operand: &mir::Operand,
    what: &str,
    loc: &Location,
) -> TranslationResult<u32> {
    let mir::Operand::Constant(const_op) = operand else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle expects constant {what}"
            ))
        );
    };

    let value = match const_op.const_.kind() {
        ConstantKind::Allocated(alloc) => alloc.read_uint(),
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            TyConstKind::Value(_, alloc) => alloc.read_uint(),
            other => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "WarpShuffleValue::shuffle {what} must be a value constant, got {other:?}"
                    ))
                );
            }
        },
        other => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "WarpShuffleValue::shuffle {what} must be a constant, got {other:?}"
                ))
            );
        }
    };

    let value = value.map_err(|err| {
        pliron::input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle could not read constant {what}: {err:?}"
            ))
        )
    })?;

    u32::try_from(value).map_err(|_| {
        pliron::input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle {what} value {value} does not fit in u32"
            ))
        )
    })
}

fn legacy_shuffle_mode(arg: &mir::Operand, loc: &Location) -> TranslationResult<LegacyShuffleMode> {
    match constant_operand_u32(arg, "mode", loc)? {
        0 => Ok(LegacyShuffleMode::Up),
        1 => Ok(LegacyShuffleMode::Down),
        2 => Ok(LegacyShuffleMode::Xor),
        3 => Ok(LegacyShuffleMode::Idx),
        other => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle mode {other} is not recognized"
            ))
        ),
    }
}

fn legacy_shuffle_clamp_bits(mode: LegacyShuffleMode, width: u32) -> Option<u32> {
    if width == 0 || width > 32 || !width.is_power_of_two() {
        return None;
    }

    let base = match mode {
        LegacyShuffleMode::Up => 0,
        LegacyShuffleMode::Down | LegacyShuffleMode::Xor | LegacyShuffleMode::Idx => 0x1f,
    };
    Some(((32 - width) << 8) | base)
}

fn legacy_shuffle_clamp(
    mode: LegacyShuffleMode,
    width: u32,
    loc: &Location,
) -> TranslationResult<u32> {
    legacy_shuffle_clamp_bits(mode, width).ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle width {width} must be a non-zero power of two <= 32"
            ))
        )
    })
}

fn emit_transmute(
    ctx: &mut Context,
    value: pliron::value::Value,
    result_ty: pliron::r#type::TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: &Location,
) -> (pliron::value::Value, Ptr<Operation>) {
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![result_ty],
        vec![value],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
    if let Some(prev_op) = prev_op {
        cast_op.insert_after(ctx, prev_op);
    } else {
        cast_op.insert_at_front(block_ptr, ctx);
    }
    (cast_op.deref(ctx).get_result(0), cast_op)
}

/// Lowers the legacy `WarpShuffleValue::shuffle(mode, mask, value, b, width)`
/// trait method directly. Its concrete implementations call a host-only
/// `warp_shuffle_32` stub, so retaining the implementation body would leave an
/// undefined symbol in device code.
#[allow(clippy::too_many_arguments)]
pub fn emit_warp_shuffle_value_trait(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle expects 5 arguments [mode, mask, value, b, width], got {}",
                args.len()
            ))
        );
    }

    let mode = legacy_shuffle_mode(&args[0], &loc)?;
    let width = constant_operand_u32(&args[4], "width", &loc)?;
    let clamp = legacy_shuffle_clamp(mode, width, &loc)?;

    let tuple_ty = types::translate_destination_type(ctx, body, destination, &loc)?;
    let value_ty = {
        let ty_ref = tuple_ty.deref(ctx);
        let Some(tuple_ty_ref) = ty_ref.downcast_ref::<dialect_mir::types::MirTupleType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "WarpShuffleValue::shuffle destination is not a tuple".to_string()
                )
            );
        };
        let fields = tuple_ty_ref.get_types();
        if fields.len() != 2 {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "WarpShuffleValue::shuffle destination tuple has {} fields, expected 2",
                    fields.len()
                ))
            );
        }
        fields[0]
    };

    let is_f32 = value_ty.deref(ctx).is::<FP32Type>();
    let is_f64 = value_ty.deref(ctx).is::<FP64Type>();
    let is_i32 = value_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| ty.width() == 32);
    if !is_f32 && !is_f64 && !is_i32 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "WarpShuffleValue::shuffle direct lowering only supports f32, f64, and 32-bit integers, got {}",
                value_ty.disp(ctx)
            ))
        );
    }

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (mut value, next_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next_op;
    let (lane_or_delta, next_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next_op;

    let physical_ty = if is_f64 {
        IntegerType::get(ctx, 64, Signedness::Unsigned).to_handle()
    } else if is_f32 {
        IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle()
    } else {
        value_ty
    };
    if is_f32 || is_f64 {
        let (bits, cast_op) = emit_transmute(ctx, value, physical_ty, block_ptr, last_op, &loc);
        value = bits;
        last_op = Some(cast_op);
    }

    let template = if is_f64 {
        format!(
            "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; \
             shfl.sync.{}.b32 lo, lo, $2, {clamp}, $3; \
             shfl.sync.{}.b32 hi, hi, $2, {clamp}, $3; \
             mov.b64 $0, {{lo, hi}}; }}",
            mode.ptx_name(),
            mode.ptx_name(),
        )
    } else {
        format!("shfl.sync.{}.b32 $0, $1, $2, {clamp}, $3;", mode.ptx_name())
    };
    let constraints = if is_f64 { "=l,l,r,r" } else { "=r,r,r,r" };
    let shuffle_op = InlinePtxOp::build(
        ctx,
        vec![physical_ty],
        vec![value, lane_or_delta, mask],
        &template,
        constraints,
        false,
        true,
    );
    shuffle_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev_op) = last_op {
        shuffle_op.insert_after(ctx, prev_op);
    } else {
        shuffle_op.insert_at_front(block_ptr, ctx);
    }
    let mut shuffled = shuffle_op.deref(ctx).get_result(0);
    let mut result_op = shuffle_op;

    if is_f32 || is_f64 {
        let (typed, cast_op) =
            emit_transmute(ctx, shuffled, value_ty, block_ptr, Some(result_op), &loc);
        shuffled = typed;
        result_op = cast_op;
    }

    let bool_ty = types::get_bool_type(ctx);
    let false_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![bool_ty.to_handle()],
        vec![],
        vec![],
        0,
    );
    false_op.deref_mut(ctx).set_loc(loc.clone());
    dialect_mir::ops::MirConstantOp::new(false_op).set_attr_value(
        ctx,
        IntegerAttr::new(
            bool_ty,
            APInt::from_u64(0, NonZeroUsize::new(1).expect("1 is non-zero")),
        ),
    );
    false_op.insert_after(ctx, result_op);
    let predicate = false_op.deref(ctx).get_result(0);

    let tuple_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![tuple_ty],
        vec![shuffled, predicate],
        vec![],
        0,
    );
    tuple_op.deref_mut(ctx).set_loc(loc.clone());
    tuple_op.insert_after(ctx, false_op);
    let tuple_value = tuple_op.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        tuple_value,
        target,
        block_ptr,
        tuple_op,
        value_map,
        block_map,
        loc,
        "WarpShuffleValue::shuffle call without target block",
    )
}

/// Emit a warp reduction operation (`redux.sync.{add,min,max,and,or,xor}`).
///
/// Takes 2 operands `[mask, value]` and returns one result. This helper is
/// shared by the whole integer reduction family.
///
/// # Parameters
/// - `redux_opid`: The NVVM opid for the specific reduction variant
/// - `signed`: result signedness — `true` for the signed `min.s32`/`max.s32`
///   variants (result type must match an `i32` destination slot), `false` for
///   `add`, the unsigned `min.u32`/`max.u32`, and the bitwise `and`/`or`/`xor`
///   variants (all `u32`).
/// - `args`: `[mask, value]`
pub fn emit_warp_redux(
    ctx: &mut Context,
    body: &mir::Body,
    redux_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    signed: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp redux expects 2 arguments [mask, value], got {}",
                args.len()
            ))
        );
    }

    // Result signedness must match the destination local's slot type so the
    // store typechecks: `i32` locals are `Signed`, `u32` locals `Unsigned`.
    let signedness = if signed {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    };
    let result_ty = IntegerType::get(ctx, 32, signedness).to_handle();

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (value, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let redux_op = Operation::new(
        ctx,
        redux_opid,
        vec![result_ty],
        vec![mask, value],
        vec![],
        0,
    );
    redux_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        redux_op.insert_after(ctx, prev);
    } else {
        redux_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = redux_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        redux_op,
        value_map,
        block_map,
        loc,
        "warp redux call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::{LegacyShuffleMode, legacy_shuffle_clamp_bits};

    #[test]
    fn legacy_shuffle_clamp_encodes_width_and_mode() {
        assert_eq!(
            legacy_shuffle_clamp_bits(LegacyShuffleMode::Xor, 32),
            Some(0x001f)
        );
        assert_eq!(
            legacy_shuffle_clamp_bits(LegacyShuffleMode::Idx, 16),
            Some(0x101f)
        );
        assert_eq!(
            legacy_shuffle_clamp_bits(LegacyShuffleMode::Up, 16),
            Some(0x1000)
        );
    }

    #[test]
    fn legacy_shuffle_clamp_rejects_invalid_widths() {
        for width in [0, 3, 24, 33] {
            assert_eq!(
                legacy_shuffle_clamp_bits(LegacyShuffleMode::Down, width),
                None
            );
        }
    }
}
