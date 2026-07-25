/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Library crate for the cross_crate_embedded regression test (issue #222).
//!
//! The generic kernel here is the key: `#[cuda_module]` loads every embedded
//! CUDA artifact and resolves each concrete specialization across the resulting
//! module set. A specialization may be owned by the binary crate rather than
//! this library's artifact.

use core::ops::Mul;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = input[i] * factor;
        }
    }
}

/// Own one concrete specialization in this library crate. The library's
/// device artifact therefore contains only a monomorphized generic kernel and
/// exercises backend retention for a generic-only library.
pub fn scale_f32_ptx_name() -> &'static str {
    kernels::scale_ptx_name::<f32>()
}

/// Ordinary host code kept in a separate module from the generated loader.
/// The regression example references this with multiple host CGUs enabled, so
/// the artifact must be carried by whichever archive member is extracted.
pub mod host_probe {
    #[inline(never)]
    pub fn linked_value() -> u64 {
        0x0cda_0a1d_e222
    }
}
