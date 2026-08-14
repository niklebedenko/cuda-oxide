/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device semantic-fingerprint regression for higher-ranked formatter types.
//!
//! Compilation and PTX generation only. Do not launch this kernel.

use cuda_device::DisjointSlice;
use cuda_device::cuda_module;
use cuda_device::kernel;
use cuda_device::thread;

#[cuda_module]
mod kernels {
    use super::*;

    type FormatProbe = for<'a, 'b> fn(&'a mut core::fmt::Formatter<'b>);

    #[inline(never)]
    fn probe(_: core::marker::PhantomData<FormatProbe>) -> u32 {
        7
    }

    #[kernel]
    pub fn semantic_fingerprint_bound_vars(mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = output.get_mut(idx) {
            *slot = probe(core::marker::PhantomData);
        }
    }
}

fn main() {
    println!("PASS: semantic fingerprint bound-vars fixture compiled (kernel not launched)");
}
