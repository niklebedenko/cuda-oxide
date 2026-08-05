/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Extern-crate closure collection regression test.
//!
//! The kernel calls `helper_lib` functions whose bodies contain closures:
//! `gated_load` uses an `if` plus a closure passed to `apply`, while
//! `double_even` uses `bool::then`. Because `helper-lib` is not the kernel's
//! local crate, the device collector's cross-crate kernel check
//! reaches those closure DefIds through
//! `enqueue_callable_trait_receiver_body` → `should_collect_from_crate`,
//! which called `TyCtxt::item_name` unconditionally and ICE'd rustc:
//!
//!   error: internal compiler error: item_name: no name for DefPath ...
//!
//! Closures in the local crate take the early `LOCAL_CRATE` return and never
//! hit the check, which is why kernels with local closures work fine.
//!
//! Run: cargo oxide run extern_crate_closure

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn gated_double(input: &[i32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // Both helper calls route through closures defined in the
            // external helper-lib crate.
            *o = match helper_lib::gated_load(input, i, !i.is_multiple_of(3)) {
                Some(v) => helper_lib::double_even(v).unwrap_or(v),
                None => -1,
            };
        }
    }

    #[kernel]
    pub fn wrapped_pair_affine(input: &[f32], mut out: DisjointSlice<f32>) {
        let pair = thread::index_1d();
        let base = pair.get() * 2;
        if base + 1 < input.len() && base + 1 < out.len() {
            let result = helper_lib::two_slot_affine(
                [
                    helper_lib::Scalar(input[base]),
                    helper_lib::Scalar(input[base + 1]),
                ],
                helper_lib::Scalar(2.5),
                helper_lib::Scalar(-3.0),
            );
            // SAFETY: thread `pair` exclusively owns output slots
            // `[2 * pair, 2 * pair + 1]`, both checked above.
            unsafe {
                *out.get_unchecked_mut(base) = result[0].0;
                *out.get_unchecked_mut(base + 1) = result[1].0;
            }
        }
    }
}

fn expected(input: &[i32], i: usize) -> i32 {
    match helper_lib::gated_load(input, i, !i.is_multiple_of(3)) {
        Some(v) => helper_lib::double_even(v).unwrap_or(v),
        None => -1,
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded module");

    const N: usize = 1000;
    let input: Vec<i32> = (0..N as i32).map(|i| i - 500).collect();
    let in_dev = DeviceBuffer::from_host(&stream, &input).expect("H2D input");
    let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).expect("alloc out");

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.gated_double(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &in_dev,
            &mut out_dev,
        )
    }
    .expect("launch gated_double");
    let out = out_dev.to_host_vec(&stream).expect("D2H out");

    let errors = (0..N).filter(|&i| out[i] != expected(&input, i)).count();
    if errors != 0 {
        eprintln!("FAILED: {errors} mismatches");
        std::process::exit(1);
    }

    let pair_input: Vec<f32> = (0..N).map(|i| i as f32 * 0.25 - 17.0).collect();
    let pair_in_dev = DeviceBuffer::from_host(&stream, &pair_input).expect("H2D pair input");
    let mut pair_out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).expect("alloc pair output");
    // SAFETY: one thread handles each complete pair and both buffers contain N elements.
    unsafe {
        module.wrapped_pair_affine(
            &stream,
            LaunchConfig::for_num_elems((N / 2) as u32),
            &pair_in_dev,
            &mut pair_out_dev,
        )
    }
    .expect("launch wrapped_pair_affine");
    let pair_out = pair_out_dev.to_host_vec(&stream).expect("D2H pair output");
    let pair_errors = (0..N)
        .filter(|&i| (pair_out[i] - (pair_input[i] * 2.5 - 3.0)).abs() > 1e-5)
        .count();
    if pair_errors != 0 {
        eprintln!("FAILED: {pair_errors} wrapped-scalar closure mismatches");
        std::process::exit(1);
    }

    println!("PASSED: all external closure and wrapped-pair results correct");
}
