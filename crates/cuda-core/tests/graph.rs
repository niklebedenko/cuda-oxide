/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer};

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
