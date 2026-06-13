/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::sync::mpsc;
use std::time::Duration;

fn assert_no_completion<T>(rx: &mpsc::Receiver<T>, label: &str) {
    match rx.recv_timeout(Duration::from_millis(100)) {
        Ok(_) => panic!("{label} completed before the gated stream was released"),
        Err(mpsc::RecvTimeoutError::Timeout) => {}
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{label} worker disconnected before the gated stream was released")
        }
    }
}

fn gate_stream(stream: &CudaStream, label: &'static str) -> mpsc::Sender<()> {
    let (tx, rx) = mpsc::channel();
    stream
        .launch_host_function(move || {
            rx.recv().expect(label);
        })
        .expect("failed to enqueue stream gate");
    tx
}

#[test]
fn device_buffer_from_host_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [1_u32, 2, 3, 4, 5];
    let dev_buf =
        DeviceBuffer::from_host(&stream, &data).expect("failed to allocate DeviceBuffer from host");

    assert_eq!(dev_buf.len(), 5);
    assert_eq!(dev_buf.num_bytes(), 20);
    assert!(!dev_buf.is_empty());

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, data);
}

#[test]
fn device_buffer_zeroed_initializes_with_zeros() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<f32>::zeroed(&stream, 4).expect("failed to allocate zeroed DeviceBuffer");

    assert_eq!(dev_buf.len(), 4);
    assert_eq!(dev_buf.num_bytes(), 16);

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn device_buffer_supports_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<u8>::zeroed(&stream, 0).expect("failed to allocate empty device buffer");
    assert_eq!(dev_buf.len(), 0);
    assert_eq!(dev_buf.num_bytes(), 0);
    assert!(dev_buf.is_empty());

    let dev_buf_host = DeviceBuffer::<u8>::from_host(&stream, &[])
        .expect("failed to allocate empty device buffer from empty slice");
    assert_eq!(dev_buf_host.len(), 0);
    assert_eq!(dev_buf_host.num_bytes(), 0);
    assert!(dev_buf_host.is_empty());
}

#[test]
fn device_buffer_async_compat_methods_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [7_u32, 11, 13, 17];
    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate uninitialized device buffer");
    unsafe {
        dev.copy_from_host_async(&data, &stream)
            .expect("failed to copy host data into device buffer");
    }

    let mut clone = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate clone device buffer");
    clone
        .copy_from_device_async(&dev, &stream)
        .expect("failed to copy device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy clone back to host"),
        data
    );

    clone
        .zero_async(&stream)
        .expect("failed to zero device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy zeroed buffer back to host"),
        [0, 0, 0, 0]
    );

    clone
        .drop_async(&stream)
        .expect("failed to async free clone");
    dev.drop_async(&stream)
        .expect("failed to async free source");

    let empty = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 0) }
        .expect("failed to allocate empty uninitialized device buffer");
    empty
        .drop_async(&stream)
        .expect("failed to async free empty buffer");
    stream.synchronize().expect("stream sync failed");
}

#[test]
fn device_buffer_async_allocation_can_outlive_non_default_stream_before_drop() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    let dev = {
        let stream = ctx.new_stream().expect("failed to create CUDA stream");
        let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, 4) }
            .expect("failed to allocate async device buffer");
        dev.zero_async(&stream)
            .expect("failed to enqueue work on allocation stream");
        dev
    };

    drop(dev);
    ctx.synchronize()
        .expect("async buffer must not drop through a destroyed stream handle");
}

#[test]
fn device_buffer_from_raw_parts_async_can_outlive_non_default_stream_before_drop() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    let dev = {
        let stream = ctx.new_stream().expect("failed to create CUDA stream");
        let mut ptr = 0;
        let rc = unsafe {
            cuda_bindings::cuMemAllocAsync(
                &mut ptr,
                4 * std::mem::size_of::<u32>(),
                stream.cu_stream(),
            )
        };
        assert_eq!(rc, 0, "raw cuMemAllocAsync failed: {rc}");

        let mut dev =
            unsafe { DeviceBuffer::<u32>::from_raw_parts_async_on_stream(ptr, 4, &stream) };
        dev.zero_async(&stream)
            .expect("failed to enqueue work on raw allocation stream");
        dev
    };

    drop(dev);
    ctx.synchronize().expect(
        "raw async buffer must retain the allocation stream used by ordinary DeviceBuffer drop",
    );
}

#[test]
fn device_buffer_from_raw_parts_async_raw_handle_synchronizes_before_drop() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    let dev = {
        let stream = ctx.new_stream().expect("failed to create CUDA stream");
        let mut ptr = 0;
        let rc = unsafe {
            cuda_bindings::cuMemAllocAsync(
                &mut ptr,
                4 * std::mem::size_of::<u32>(),
                stream.cu_stream(),
            )
        };
        assert_eq!(rc, 0, "raw cuMemAllocAsync failed: {rc}");

        let mut dev = unsafe {
            DeviceBuffer::<u32>::from_raw_parts_async(ptr, 4, ctx.clone(), stream.cu_stream())
        };
        dev.zero_async(&stream)
            .expect("failed to enqueue work on raw allocation stream");
        dev
    };

    drop(dev);
    ctx.synchronize().expect(
        "raw async compatibility constructor must not free before queued stream work completes",
    );
}

#[test]
fn device_buffer_from_raw_parts_async_drop_async_orders_free_after_raw_stream() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let alloc_stream = ctx
        .new_stream()
        .expect("failed to create allocation stream");
    let free_stream = ctx.new_stream().expect("failed to create free stream");

    let mut ptr = 0;
    let rc = unsafe {
        cuda_bindings::cuMemAllocAsync(
            &mut ptr,
            4 * std::mem::size_of::<u32>(),
            alloc_stream.cu_stream(),
        )
    };
    assert_eq!(rc, 0, "raw cuMemAllocAsync failed: {rc}");

    let mut dev = unsafe {
        DeviceBuffer::<u32>::from_raw_parts_async(ptr, 4, ctx.clone(), alloc_stream.cu_stream())
    };
    let release_gate = gate_stream(&alloc_stream, "raw allocation stream gate was dropped");
    dev.zero_async(&alloc_stream)
        .expect("failed to enqueue work on raw allocation stream");

    dev.drop_async(&free_stream)
        .expect("drop_async should order raw async free after allocation stream");
    let (tx, rx) = mpsc::channel();
    let free_stream_for_thread = free_stream.clone();
    std::thread::spawn(move || {
        tx.send(free_stream_for_thread.synchronize())
            .expect("failed to send free-stream sync result");
    });
    assert_no_completion(&rx, "raw cross-stream async free");

    release_gate
        .send(())
        .expect("failed to release raw allocation stream gate");
    rx.recv_timeout(Duration::from_secs(5))
        .expect("raw cross-stream async free did not complete after releasing gate")
        .expect("raw cross-stream async free failed");
}

#[test]
fn device_buffer_async_drop_waits_for_cross_stream_work() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let alloc_stream = ctx
        .new_stream()
        .expect("failed to create allocation stream");
    let use_stream = ctx.new_stream().expect("failed to create use stream");

    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&alloc_stream, 4) }
        .expect("failed to allocate async device buffer");
    use_stream
        .join(&alloc_stream)
        .expect("failed to order use stream after allocation stream");
    let release_gate = gate_stream(&use_stream, "use stream gate was dropped");
    dev.zero_async(&use_stream)
        .expect("failed to enqueue cross-stream use");

    drop(alloc_stream);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        drop(dev);
        tx.send(()).expect("failed to send drop completion");
    });
    assert_no_completion(&rx, "ordinary async buffer drop");

    release_gate
        .send(())
        .expect("failed to release use stream gate");
    rx.recv_timeout(Duration::from_secs(5))
        .expect("ordinary async buffer drop did not complete after releasing gate");
}

#[test]
fn device_buffer_async_drop_async_orders_free_after_allocation_stream() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let alloc_stream = ctx
        .new_stream()
        .expect("failed to create allocation stream");
    let free_stream = ctx.new_stream().expect("failed to create free stream");

    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&alloc_stream, 4) }
        .expect("failed to allocate async device buffer");
    let release_gate = gate_stream(&alloc_stream, "allocation stream gate was dropped");
    dev.zero_async(&alloc_stream)
        .expect("failed to enqueue work on allocation stream");

    dev.drop_async(&free_stream)
        .expect("drop_async should order free after allocation stream");
    let (tx, rx) = mpsc::channel();
    let free_stream_for_thread = free_stream.clone();
    std::thread::spawn(move || {
        tx.send(free_stream_for_thread.synchronize())
            .expect("failed to send free-stream sync result");
    });
    assert_no_completion(&rx, "cross-stream async free");

    release_gate
        .send(())
        .expect("failed to release allocation stream gate");
    rx.recv_timeout(Duration::from_secs(5))
        .expect("cross-stream async free did not complete after releasing gate")
        .expect("cross-stream async free failed");
}

#[test]
fn device_buffer_sync_allocation_allows_async_drop_after_queued_work() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let mut dev = DeviceBuffer::<u32>::zeroed(&stream, 4).expect("failed to allocate sync buffer");
    dev.zero_async(&stream)
        .expect("failed to enqueue work before async free");
    dev.drop_async(&stream)
        .expect("sync allocation should support stream-ordered free");
    stream
        .synchronize()
        .expect("stream-ordered sync allocation free should complete");
}

#[test]
fn device_buffer_safe_host_copy_synchronizes_before_returning() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let mut dev = DeviceBuffer::<u32>::zeroed(&stream, 4).expect("failed to allocate buffer");
    let mut data = vec![1_u32, 2, 3, 4];
    dev.copy_from_host(&data, &stream)
        .expect("safe host copy should synchronize");

    data.fill(0);
    assert_eq!(
        dev.to_host_vec(&stream).expect("failed to copy back"),
        [1, 2, 3, 4]
    );
}
