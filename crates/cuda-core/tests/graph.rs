/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, CudaGraph, CudaGraphConditionalOptions, DeviceBuffer};

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
fn while_body_segments_can_surround_a_nested_if() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let mut prefix = DeviceBuffer::from_host(&stream, &[43_u32]).expect("upload prefix");
    let mut branch = DeviceBuffer::from_host(&stream, &[47_u32]).expect("upload branch");
    let mut suffix = DeviceBuffer::from_host(&stream, &[53_u32]).expect("upload suffix");
    let mut graph = CudaGraph::new(&ctx).expect("create nested graph");
    let mut while_node = graph
        .add_while_node(
            &[],
            CudaGraphConditionalOptions::new(0).assign_default_on_launch(),
        )
        .expect("add outer WHILE node");

    let prefix_capture = while_node
        .begin_body_capture(&stream)
        .expect("begin WHILE prefix capture");
    prefix
        .zero_async(&stream)
        .expect("capture WHILE prefix memset");
    let prefix_tail = prefix_capture
        .finish_with_dependencies()
        .expect("finish WHILE prefix capture");
    assert_eq!(prefix_tail.len(), 1);

    let mut if_node = while_node
        .add_if_node(&[], CudaGraphConditionalOptions::new(1))
        .expect("add nested IF node");
    let if_capture = if_node
        .begin_body_capture(&stream)
        .expect("begin nested IF capture");
    branch
        .zero_async(&stream)
        .expect("capture nested IF memset");
    if_capture.finish().expect("finish nested IF capture");
    let if_tail = [if_node.cu_node()];
    drop(if_node);
    while_node
        .add_dependencies_to(&prefix_tail, if_tail[0])
        .expect("connect WHILE prefix to nested IF");

    let suffix_capture = while_node
        .begin_body_capture_after(&stream, &if_tail)
        .expect("begin WHILE suffix capture");
    suffix
        .zero_async(&stream)
        .expect("capture WHILE suffix memset");
    let suffix_tail = suffix_capture
        .finish_with_dependencies()
        .expect("finish WHILE suffix capture");
    assert_eq!(suffix_tail.len(), 1);
    drop(while_node);

    graph
        .instantiate()
        .expect("instantiate WHILE graph containing segmented nested IF");
}
