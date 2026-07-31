/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod common;

use common::{counted_loop, mir_ctx, multi_latch_counted_loop};
use mir_transforms::deferred_unroll::mark_deferred_full_unroll_loops;
use pliron::builtin::attributes::StringAttr;
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;

#[test]
fn selected_function_marks_only_its_loop_backedge() {
    let mut ctx = mir_ctx();
    let loop_ = counted_loop(&mut ctx, 3);
    let module_region = loop_.module.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_key: Identifier = dialect_mir::DEFERRED_FULL_UNROLL_FUNC_ATTR
        .try_into()
        .unwrap();
    function
        .deref_mut(&ctx)
        .attributes
        .set(function_key, StringAttr::new("true".to_string()));

    mark_deferred_full_unroll_loops(loop_.module, &mut ctx).expect("marking succeeds");

    let backedge = loop_
        .latch
        .deref(&ctx)
        .get_terminator(&ctx)
        .expect("loop latch terminator");
    let loop_key: Identifier = dialect_mir::LOOP_UNROLL_FULL_ATTR.try_into().unwrap();
    assert!(
        backedge
            .deref(&ctx)
            .attributes
            .get::<StringAttr>(&loop_key)
            .is_some()
    );
    let header_terminator = loop_
        .header
        .deref(&ctx)
        .get_terminator(&ctx)
        .expect("loop header terminator");
    assert!(
        header_terminator
            .deref(&ctx)
            .attributes
            .get::<StringAttr>(&loop_key)
            .is_none()
    );
}

#[test]
fn multi_latch_loop_reuses_one_stable_marker() {
    let mut ctx = mir_ctx();
    let loop_ = multi_latch_counted_loop(&mut ctx, 4, 1, 1);
    let module_region = loop_.module.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_key: Identifier = dialect_mir::DEFERRED_FULL_UNROLL_FUNC_ATTR
        .try_into()
        .unwrap();
    function
        .deref_mut(&ctx)
        .attributes
        .set(function_key, StringAttr::new("true".to_string()));

    mark_deferred_full_unroll_loops(loop_.module, &mut ctx).expect("marking succeeds");

    let loop_key: Identifier = dialect_mir::LOOP_UNROLL_FULL_ATTR.try_into().unwrap();
    let latch_markers: Vec<String> = [loop_.continue_latch, loop_.normal_latch]
        .into_iter()
        .map(|latch| {
            let terminator = latch
                .deref(&ctx)
                .get_terminator(&ctx)
                .expect("loop latch terminator");
            let terminator_ref = terminator.deref(&ctx);
            let marker = terminator_ref
                .attributes
                .get::<StringAttr>(&loop_key)
                .expect("deferred unroll marker");
            String::from((*marker).clone())
        })
        .collect();
    assert_eq!(latch_markers[0], latch_markers[1]);
    assert_ne!(latch_markers[0], "true");
}
