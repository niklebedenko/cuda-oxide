/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side control for CUDA Graph conditional nodes.
//!
//! Conditional WHILE body kernels call [`set_conditional`] to choose whether
//! their body graph executes another iteration. Host code obtains the handle
//! from [`cuda_core::CudaGraphWhileNode::conditional_handle`](https://docs.rs/cuda-core/latest/cuda_core/struct.CudaGraphWhileNode.html)
//! and passes it to the kernel as an ordinary `u64` parameter.

use crate::device;

/// CUDA's device-visible graph conditional handle.
///
/// CUDA defines the handle as an unsigned 64-bit token on both the host and
/// device. This alias therefore accepts a [`cuda_core::CudaGraphConditionalHandle`](https://docs.rs/cuda-core/latest/cuda_core/type.CudaGraphConditionalHandle.html)
/// without a conversion or layout wrapper.
pub type CudaGraphConditionalHandle = u64;

#[device]
unsafe extern "C" {
    #[doc(hidden)]
    fn cudaGraphSetConditional(handle: CudaGraphConditionalHandle, value: u32);
}

/// Set the control value for a CUDA Graph conditional node.
///
/// A conditional WHILE node repeats its body while the value is nonzero. This
/// lowers to CUDA's device-runtime `cudaGraphSetConditional` system call; it
/// does not require embedding `libcudadevrt`.
///
/// # Safety
///
/// - `handle` must belong to the graph whose body is currently executing and
///   to the CUDA context running this kernel.
/// - Threads that can reach this call must not race to assign conflicting
///   values to the same handle.
/// - Conditional graph nodes and this system call require CUDA 12.3 or newer.
#[inline(always)]
pub unsafe fn set_conditional(handle: CudaGraphConditionalHandle, value: u32) {
    unsafe { cudaGraphSetConditional(handle, value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_handle_matches_cuda_device_abi() {
        assert_eq!(core::mem::size_of::<CudaGraphConditionalHandle>(), 8);
        assert_eq!(core::mem::align_of::<CudaGraphConditionalHandle>(), 8);
    }
}
