/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #138.
//!
//! `for x in arr` over a by-value array `[T; N]` desugars to a loop over
//! `core::array::IntoIter<T, N>`, and rustc places a `Drop` terminator
//! for the iterator at the loop exit because `IntoIter` has an
//! `impl Drop`. For element types without drop glue that destructor is
//! provably a no-op (`IntoIter::drop` is `if needs_drop::<T>() { .. }`,
//! which is statically false), so the importer lowers the `Drop`
//! terminator to a plain branch instead of rejecting the kernel.
//!
//! Before the fix the build failed with
//!
//!   Unsupported construct: drop of `...std::array::IntoIter...` is not
//!   supported on the device; cuda-oxide does not yet emit device-side
//!   `drop_in_place` calls.
//!
//! The kernels cover by-value array iteration, array helper closures/function
//! items, and fieldless enum array constants. Results are verified on the host.
//!
//! Run: cargo oxide run array_for_loop

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// A plain Copy struct; an array of these has no drop glue either.
    #[derive(Clone, Copy)]
    pub struct Point {
        pub x: u32,
        pub y: u32,
    }

    #[derive(Clone, Copy)]
    enum Side {
        Low,
        High,
    }

    #[inline(always)]
    fn triple(x: u32) -> u32 {
        x * 3
    }

    /// Sum a by-value `[u32; 4]` with a `for` loop (the issue-138 shape).
    #[kernel]
    pub fn sum_u32_array(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let arr: [u32; 4] = [t, t + 1, t + 2, t + 3];
            let mut acc: u32 = 0;
            for x in arr {
                acc += x;
            }
            *out_elem = acc;
        }
    }

    /// Same loop shape over an array of Copy structs.
    #[kernel]
    pub fn sum_point_array(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let pts: [Point; 4] = [
                Point { x: t, y: 1 },
                Point { x: t + 1, y: 2 },
                Point { x: t + 2, y: 3 },
                Point { x: t + 3, y: 4 },
            ];
            let mut acc: u32 = 0;
            for p in pts {
                acc += p.x * p.y;
            }
            *out_elem = acc;
        }
    }

    /// `from_fn` and `map` exercise zero-capture closure and function-item
    /// constants in `core::array` helpers. `SIDES` covers fieldless enum array
    /// constants such as Impulse's `LowHigh::ALL`.
    #[kernel]
    pub fn array_helper_constants(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let generated: [u32; 4] = core::array::from_fn(|i| t + i as u32);
            let tripled = generated.map(triple);
            let shifted = generated.map(|x| x + 7);

            const SIDES: [Side; 2] = [Side::Low, Side::High];
            let mut side_score = 0;
            for side in SIDES {
                side_score += match side {
                    Side::Low => 11,
                    Side::High => 19,
                };
            }

            *out_elem = tripled[0]
                + tripled[1]
                + tripled[2]
                + tripled[3]
                + shifted[0]
                + shifted[1]
                + shifted[2]
                + shifted[3]
                + side_score;
        }
    }
}

fn main() {
    println!("=== array_for_loop regression (issue #138) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/array_for_loop.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_u32 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches both kernels' indexing model and
    // the 32-element output allocations.
    unsafe { module.sum_u32_array(stream.as_ref(), cfg, &mut d_u32) }
        .expect("launch sum_u32_array");
    let got_u32 = d_u32.to_host_vec(&stream).unwrap();

    let mut d_pts = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.sum_point_array(stream.as_ref(), cfg, &mut d_pts) }
        .expect("launch sum_point_array");
    let got_pts = d_pts.to_host_vec(&stream).unwrap();

    let mut d_helpers = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .array_helper_constants(stream.as_ref(), cfg, &mut d_helpers)
        .expect("launch array_helper_constants");
    let got_helpers = d_helpers.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    for tid in 0..N {
        let t = tid as u32;
        // sum of [t, t+1, t+2, t+3]
        let want_u32 = 4 * t + 6;
        // t*1 + (t+1)*2 + (t+2)*3 + (t+3)*4
        let want_pts = t + (t + 1) * 2 + (t + 2) * 3 + (t + 3) * 4;
        if got_u32[tid] != want_u32 {
            println!(
                "FAIL tid={tid}: sum_u32_array={} expected={want_u32}",
                got_u32[tid]
            );
            failures += 1;
        }
        if got_pts[tid] != want_pts {
            println!(
                "FAIL tid={tid}: sum_point_array={} expected={want_pts}",
                got_pts[tid]
            );
            failures += 1;
        }
        let want_helpers = 16 * t + 82;
        if got_helpers[tid] != want_helpers {
            println!(
                "FAIL tid={tid}: array_helper_constants={} expected={want_helpers}",
                got_helpers[tid]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!("array_for_loop: PASS ({N} threads, array iteration/helpers translated correctly)");
    } else {
        println!("array_for_loop: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
