/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Preserve bounded array-loop unroll intent until LLVM sees a constant trip count.

use crate::analyses::loop_info::LoopInfo;
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::StringAttr;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::pass_manager::AnalysisManager;
use pliron::result::Result;

/// Mark every back-edge in a function selected for deferred full unrolling.
///
/// Core's erased array builder accepts its extent as a slice length, so the
/// MIR-level unroller cannot prove a trip count. Its monomorphized caller does
/// carry the concrete extent, however. Keeping the request on the back-edge
/// lets LLVM apply it after the builder has been inlined into that caller.
pub fn mark_deferred_full_unroll_loops(module: Ptr<Operation>, ctx: &mut Context) -> Result<()> {
    let function_key: Identifier = dialect_mir::DEFERRED_FULL_UNROLL_FUNC_ATTR
        .try_into()
        .unwrap();
    let loop_key: Identifier = dialect_mir::LOOP_UNROLL_FULL_ATTR.try_into().unwrap();

    let mut next_loop_marker = 0usize;
    for function in collect_functions(module, ctx) {
        if function
            .deref(ctx)
            .attributes
            .get::<StringAttr>(&function_key)
            .is_none()
        {
            continue;
        }

        let region = function.deref(ctx).get_region(0);
        let loop_info = {
            let mut analyses = AnalysisManager::default();
            let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
            let dom = dom_info.get_dom_tree(ctx, region);
            LoopInfo::compute(ctx, region, dom)
        };

        for loop_ in loop_info.loops() {
            let marker = StringAttr::new(format!("deferred_full_unroll_{next_loop_marker}"));
            next_loop_marker += 1;
            for &latch in &loop_.latches {
                if let Some(terminator) = latch.deref(ctx).get_terminator(ctx) {
                    terminator
                        .deref_mut(ctx)
                        .attributes
                        .set(loop_key.clone(), marker.clone());
                }
            }
        }
    }

    Ok(())
}

fn collect_functions(module: Ptr<Operation>, ctx: &Context) -> Vec<Ptr<Operation>> {
    let mut functions = Vec::new();
    let module_region = module.deref(ctx).get_region(0);
    let blocks: Vec<Ptr<BasicBlock>> = module_region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if Operation::get_op::<MirFuncOp>(operation, ctx).is_some() {
                functions.push(operation);
            }
        }
    }
    functions
}
