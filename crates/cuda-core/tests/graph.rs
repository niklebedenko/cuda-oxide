/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{
    CudaContext, CudaGraph, CudaGraphConditionalOptions, DeviceBuffer, launch_kernel_on_stream,
};

const NESTED_CONDITIONAL_TEST_PTX: &str = r#"
.version 7.1
.target sm_86
.address_size 64

.extern .func cudaGraphSetConditional
(
    .param .b64 cudaGraphSetConditional_param_0,
    .param .b32 cudaGraphSetConditional_param_1
)
;

.visible .entry graph_prefix(
    .param .u64 graph_prefix_param_0
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [graph_prefix_param_0];
    ld.global.u32 %r1, [%rd1];
    setp.eq.u32 %p1, %r1, 0;
    selp.u32 %r2, 1, 225, %p1;
    st.global.u32 [%rd1], %r2;
    ret;
}

.visible .entry graph_branch(
    .param .u64 graph_branch_param_0
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [graph_branch_param_0];
    ld.global.u32 %r1, [%rd1];
    setp.eq.u32 %p1, %r1, 1;
    selp.u32 %r2, 2, 226, %p1;
    st.global.u32 [%rd1], %r2;
    ret;
}

.visible .entry graph_suffix(
    .param .u64 graph_suffix_param_0,
    .param .u64 graph_suffix_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [graph_suffix_param_0];
    ld.param.u64 %rd2, [graph_suffix_param_1];
    ld.global.u32 %r1, [%rd1];
    setp.eq.u32 %p1, %r1, 2;
    selp.u32 %r2, 3, 227, %p1;
    st.global.u32 [%rd1], %r2;
    mov.u32 %r3, 0;
    {
        .param .b64 param0;
        .param .b32 param1;
        st.param.b64 [param0], %rd2;
        st.param.b32 [param1], %r3;
        call.uni cudaGraphSetConditional, (param0, param1);
    }
    ret;
}
"#;

#[test]
fn captured_memset_graph_replays_without_host_resubmission() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let initial = [3_u32, 5, 7, 11];
    let mut buffer = DeviceBuffer::from_host(&stream, &initial).expect("upload initial values");

    let capture = stream.begin_capture().expect("begin graph capture");
    buffer
        .zero_async(&stream)
        .expect("capture device-buffer memset");
    let graph = capture.finish().expect("finish graph capture");
    let graph = graph.instantiate().expect("instantiate captured graph");
    graph.upload(&stream).expect("upload executable graph");

    assert_eq!(
        buffer
            .to_host_vec(&stream)
            .expect("observe pre-launch data"),
        initial,
        "capture must record rather than execute the memset",
    );

    graph.launch(&stream).expect("launch captured graph");
    assert_eq!(
        buffer.to_host_vec(&stream).expect("observe first replay"),
        [0; 4],
    );

    buffer
        .copy_from_host(&stream, &initial)
        .expect("restore nonzero values");
    graph.launch(&stream).expect("relaunch captured graph");
    assert_eq!(
        buffer.to_host_vec(&stream).expect("observe second replay"),
        [0; 4],
    );
}

#[test]
fn dropping_unfinished_capture_restores_stream_submission() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let initial = [13_u32, 17, 19, 23];
    let mut buffer = DeviceBuffer::from_host(&stream, &initial).expect("upload initial values");

    {
        let _capture = stream.begin_capture().expect("begin abandoned capture");
        buffer
            .zero_async(&stream)
            .expect("record abandoned device-buffer memset");
    }

    assert_eq!(
        buffer
            .to_host_vec(&stream)
            .expect("stream remains usable after abandoned capture"),
        initial,
        "dropping capture must destroy, not execute, the recorded graph",
    );
    buffer
        .zero_async(&stream)
        .expect("submit ordinary memset after capture cleanup");
    assert_eq!(
        buffer
            .to_host_vec(&stream)
            .expect("observe ordinary memset"),
        [0; 4],
    );
}

#[test]
fn conditional_if_body_replays_captured_device_work() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let initial = [29_u32, 31, 37, 41];
    let mut buffer = DeviceBuffer::from_host(&stream, &initial).expect("upload initial values");
    let mut graph = CudaGraph::new(&ctx).expect("create IF graph");
    let mut if_node = graph
        .add_if_node(
            &[],
            CudaGraphConditionalOptions::new(1).assign_default_on_launch(),
        )
        .expect("add IF node");
    let capture = if_node
        .begin_body_capture(&stream)
        .expect("begin IF body capture");
    buffer.zero_async(&stream).expect("capture IF body memset");
    capture.finish().expect("finish IF body capture");
    drop(if_node);
    let graph = graph.instantiate().expect("instantiate IF graph");

    graph.launch(&stream).expect("launch IF graph");
    assert_eq!(
        buffer.to_host_vec(&stream).expect("observe IF body"),
        [0; 4]
    );
}

#[test]
fn while_body_segments_execute_in_order_around_a_true_nested_if() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let state = DeviceBuffer::from_host(&stream, &[0_u32]).expect("upload initial state");
    let module = ctx
        .load_module_from_ptx_src(NESTED_CONDITIONAL_TEST_PTX)
        .expect("load nested-conditional test module");
    let prefix = module
        .load_function("graph_prefix")
        .expect("load prefix kernel");
    let branch = module
        .load_function("graph_branch")
        .expect("load branch kernel");
    let suffix = module
        .load_function("graph_suffix")
        .expect("load suffix kernel");
    let mut graph = CudaGraph::new(&ctx).expect("create nested graph");
    let mut while_node = graph
        .add_while_node(
            &[],
            CudaGraphConditionalOptions::new(1).assign_default_on_launch(),
        )
        .expect("add outer WHILE node");
    let while_handle = while_node.conditional_handle();

    let prefix_capture = while_node
        .begin_body_capture(&stream)
        .expect("begin WHILE prefix capture");
    let mut state_pointer = state.cu_deviceptr();
    let mut prefix_parameters = [(&mut state_pointer as *mut _) as *mut std::ffi::c_void];
    // SAFETY: `graph_prefix` has one device-pointer parameter, both launch
    // dimensions are one, and `state` outlives graph execution.
    unsafe {
        launch_kernel_on_stream(
            &prefix,
            (1, 1, 1),
            (1, 1, 1),
            0,
            &stream,
            &mut prefix_parameters,
        )
    }
    .expect("capture WHILE prefix kernel");
    let prefix_tail = prefix_capture
        .finish_with_dependencies()
        .expect("finish WHILE prefix capture");
    assert_eq!(prefix_tail.len(), 1);

    let mut if_node = while_node
        .add_if_node(
            &[],
            CudaGraphConditionalOptions::new(1).assign_default_on_launch(),
        )
        .expect("add nested IF node");
    let if_capture = if_node
        .begin_body_capture(&stream)
        .expect("begin nested IF capture");
    let mut state_pointer = state.cu_deviceptr();
    let mut branch_parameters = [(&mut state_pointer as *mut _) as *mut std::ffi::c_void];
    // SAFETY: `graph_branch` has one device-pointer parameter, both launch
    // dimensions are one, and `state` outlives graph execution.
    unsafe {
        launch_kernel_on_stream(
            &branch,
            (1, 1, 1),
            (1, 1, 1),
            0,
            &stream,
            &mut branch_parameters,
        )
    }
    .expect("capture nested IF kernel");
    if_capture.finish().expect("finish nested IF capture");
    let if_tail = [if_node.cu_node()];
    drop(if_node);
    while_node
        .add_dependencies_to(&prefix_tail, if_tail[0])
        .expect("connect WHILE prefix to nested IF");

    let suffix_capture = while_node
        .begin_body_capture_after(&stream, &if_tail)
        .expect("begin WHILE suffix capture");
    let mut state_pointer = state.cu_deviceptr();
    let mut while_handle = while_handle;
    let mut suffix_parameters = [
        (&mut state_pointer as *mut _) as *mut std::ffi::c_void,
        (&mut while_handle as *mut _) as *mut std::ffi::c_void,
    ];
    // SAFETY: `graph_suffix` takes the state pointer and conditional handle,
    // both launch dimensions are one, and all captured arguments remain valid.
    unsafe {
        launch_kernel_on_stream(
            &suffix,
            (1, 1, 1),
            (1, 1, 1),
            0,
            &stream,
            &mut suffix_parameters,
        )
    }
    .expect("capture WHILE suffix kernel");
    let suffix_tail = suffix_capture
        .finish_with_dependencies()
        .expect("finish WHILE suffix capture");
    assert_eq!(suffix_tail.len(), 1);
    drop(while_node);

    let graph = graph
        .instantiate()
        .expect("instantiate WHILE graph containing segmented nested IF");
    graph.launch(&stream).expect("launch nested graph");
    assert_eq!(
        state
            .to_host_vec(&stream)
            .expect("observe ordered nested graph effects"),
        [3],
        "prefix, true IF body, and suffix must execute exactly once in order",
    );
}
