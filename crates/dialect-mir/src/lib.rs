/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR dialect definition.

pub mod attributes;
pub mod const_fold;
pub mod ops;
pub mod rust_intrinsics;
pub mod side_effects;
pub mod types;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

pub const MIR_DIALECT_NAME: &str = "mir";

/// Function attribute requesting deferred full-unroll metadata on its loops.
///
/// The MIR pass turns this into [`LOOP_UNROLL_FULL_ATTR`] on each back-edge
/// after MIR-level transforms have finished. LLVM can then honor the request
/// after helper inlining exposes a constant trip count.
pub const DEFERRED_FULL_UNROLL_FUNC_ATTR: &str = "deferred_full_unroll";

/// Terminator attribute lowered to LLVM `llvm.loop.unroll.full` metadata.
pub const LOOP_UNROLL_FULL_ATTR: &str = "loop_unroll_full";

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(MIR_DIALECT_NAME).expect("valid dialect name"),
    );
    ops::register(ctx);
    types::register(ctx);
    attributes::register(ctx);
}
