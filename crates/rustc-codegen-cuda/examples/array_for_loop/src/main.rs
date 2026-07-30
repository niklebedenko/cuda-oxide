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
//! Two kernels cover the shapes from the issue: a `for` loop over a
//! plain `[u32; 4]` and one over an array of Copy structs. Additional kernels
//! cover impossible residual-enum paths in `array::from_fn`/`array::map` and
//! the fork's small-array SSA lowering. All results are verified on the host.
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
    #[repr(C, align(32))]
    struct ScalarPair {
        values: [f32; 2],
        padding: [u8; 24],
    }

    #[derive(Clone, Copy)]
    #[repr(C, align(32))]
    struct VectorPair {
        values: [[f32; 3]; 2],
        padding: [u8; 8],
    }

    #[derive(Clone, Copy)]
    enum Side {
        Low,
        High,
    }

    #[inline(always)]
    fn hdiv_quadratic_helper(values: &[f64; 10], matrix: &[[f64; 10]; 10]) -> f64 {
        let mut quadratic = 0.0_f64;
        for row in 0..10 {
            for column in 0..10 {
                quadratic += values[row] * matrix[row][column] * values[column];
            }
        }
        quadratic
    }

    #[inline(always)]
    fn triple(x: u32) -> u32 {
        x * 3
    }

    #[inline(never)]
    fn scale_pair(source: ScalarPair, factor: f32) -> ScalarPair {
        ScalarPair {
            values: [source.values[0] * factor, source.values[1] * factor],
            padding: [0; 24],
        }
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

    /// `array::from_fn` and `array::map` retain impossible residual-enum
    /// branches in MIR even though these infallible helpers cannot take them.
    #[kernel]
    pub fn map_generated_array(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let generated: [u32; 4] = core::array::from_fn(|index| t + index as u32);
            let mapped = generated.map(|value| value * 3 + 1);
            *out_elem = mapped.into_iter().sum();
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

    /// Keep fixed-axis aggregate arrays in SSA through constant-trip loops.
    #[kernel]
    pub fn fixed_axis_aggregate_arrays(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let index = tid.get();
        let t = index as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let operand = VectorPair {
                values: [[t, t + 1.0, t + 2.0], [t + 3.0, t + 4.0, t + 5.0]],
                padding: [0; 8],
            };
            let components = [
                ScalarPair {
                    values: [operand.values[0][0], operand.values[1][0]],
                    padding: [0; 24],
                },
                ScalarPair {
                    values: [operand.values[0][1], operand.values[1][1]],
                    padding: [0; 24],
                },
                ScalarPair {
                    values: [operand.values[0][2], operand.values[1][2]],
                    padding: [0; 24],
                },
            ];
            let zero_vector = VectorPair {
                values: [[0.0; 3]; 2],
                padding: [0; 8],
            };
            let mut rows = [zero_vector; 3];

            let mut row = 0;
            while row < 3 {
                let mut row_components = components;
                let mut component = 0;
                while component < 3 {
                    let source = components[component];
                    let factor = (row + component + 1) as f32;
                    row_components[component] = scale_pair(source, factor);
                    component += 1;
                }
                rows[row] = VectorPair {
                    values: [
                        [
                            row_components[0].values[0],
                            row_components[1].values[0],
                            row_components[2].values[0],
                        ],
                        [
                            row_components[0].values[1],
                            row_components[1].values[1],
                            row_components[2].values[1],
                        ],
                    ],
                    padding: [0; 8],
                };
                row += 1;
            }

            let mut sum = 0.0;
            let mut row = 0;
            while row < 3 {
                let mut lane = 0;
                while lane < 2 {
                    let mut component = 0;
                    while component < 3 {
                        sum += rows[row].values[lane][component];
                        component += 1;
                    }
                    lane += 1;
                }
                row += 1;
            }
            *out_elem = sum;
        }
    }

    /// A runtime index into a small aggregate array must remain in SSA rather
    /// than creating a per-thread local-memory copy of the whole array.
    #[kernel]
    pub fn runtime_aggregate_array_index(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let index = tid.get();
        let t = index as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let components = [
                ScalarPair {
                    values: [t, t + 3.0],
                    padding: [0; 24],
                },
                ScalarPair {
                    values: [t + 1.0, t + 4.0],
                    padding: [0; 24],
                },
                ScalarPair {
                    values: [t + 2.0, t + 5.0],
                    padding: [0; 24],
                },
            ];
            let selected = components[index % components.len()];
            *out_elem = selected.values[0] + selected.values[1];
        }
    }

    /// `array::map` uses `[MaybeUninit<U>; N]` internally. Keep that union
    /// usable when `U` has stronger alignment than NVPTX's scalar types.
    #[kernel]
    pub fn map_over_aligned_array(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let t = tid.get() as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let components = [0.0_f32, 1.0, 2.0].map(|offset| ScalarPair {
                values: [t + offset, t + offset + 3.0],
                padding: [0; 24],
            });
            *out_elem = components[0].values[0]
                + components[0].values[1]
                + components[1].values[0]
                + components[1].values[1]
                + components[2].values[0]
                + components[2].values[1];
        }
    }

    /// Compiler-only reproduction of the fixed observation arrays in
    /// Impulse's H(div) affine residual. The arrays are populated through a
    /// constant-bound loop and then read through a fixed-size quadratic form.
    #[kernel]
    pub unsafe fn hdiv_quadratic_array(
        input: &[f64],
        matrix: *const [[f64; 10]; 10],
        mut out: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let index = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut values = [0.0_f64; 10];
            for row in 0..10 {
                values[row] = input[index * 10 + row];
            }
            // SAFETY: this compiler-only shape is never launched by the host
            // harness; the pointer models Impulse's validated fixed metric.
            *out_elem = hdiv_quadratic_helper(&values, unsafe { &*matrix });
        }
    }

    /// Runtime reproduction of the repeated
    /// `reaction.iter_mut().enumerate()` loops in Impulse's H(div)
    /// mass-Riesz reaction.
    #[kernel]
    pub fn hdiv_iter_mut_accumulate(input: &[f64], mut out: DisjointSlice<f64>) {
        let tid = thread::index_1d();
        let index = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut reaction = [0.0_f64; 3];
            for row in 0..10 {
                let multiplier = input[index * 40 + row];
                for (component, reaction_component) in reaction.iter_mut().enumerate() {
                    *reaction_component +=
                        multiplier * input[index * 40 + 10 + 3 * row + component];
                }
            }
            for row in 0..4 {
                let multiplier = input[index * 40 + row];
                for (component, reaction_component) in reaction.iter_mut().enumerate() {
                    *reaction_component += multiplier * input[index * 40 + component];
                }
            }
            for (component, reaction_component) in reaction.iter_mut().enumerate() {
                *reaction_component *= input[index * 40 + component];
            }
            *out_elem = reaction[0] + reaction[1] + reaction[2];
        }
    }
}

fn kernel_body<'a>(ptx: &'a str, kernel_prefix: &str) -> &'a str {
    let marker = format!(".entry {kernel_prefix}(");
    let entry = ptx
        .find(&marker)
        .unwrap_or_else(|| panic!("missing PTX entry with prefix {kernel_prefix}"));
    let body_start = ptx[entry..]
        .find('{')
        .map(|offset| entry + offset)
        .expect("kernel entry has a body");
    let mut depth = 0_u32;
    for (offset, byte) in ptx.as_bytes()[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &ptx[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated PTX body for {kernel_prefix}");
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

    let mut d_mapped = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.map_generated_array(stream.as_ref(), cfg, &mut d_mapped) }
        .expect("launch map_generated_array");
    let got_mapped = d_mapped.to_host_vec(&stream).unwrap();

    let mut d_helpers = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.array_helper_constants(stream.as_ref(), cfg, &mut d_helpers) }
        .expect("launch array_helper_constants");
    let got_helpers = d_helpers.to_host_vec(&stream).unwrap();

    let mut d_fixed_axes = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.fixed_axis_aggregate_arrays(stream.as_ref(), cfg, &mut d_fixed_axes) }
        .expect("launch fixed_axis_aggregate_arrays");
    let got_fixed_axes = d_fixed_axes.to_host_vec(&stream).unwrap();

    let mut d_runtime_index = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.runtime_aggregate_array_index(stream.as_ref(), cfg, &mut d_runtime_index) }
        .expect("launch runtime_aggregate_array_index");
    let got_runtime_index = d_runtime_index.to_host_vec(&stream).unwrap();

    let mut d_aligned_map = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.map_over_aligned_array(stream.as_ref(), cfg, &mut d_aligned_map) }
        .expect("launch map_over_aligned_array");
    let got_aligned_map = d_aligned_map.to_host_vec(&stream).unwrap();

    let hdiv_input = (0..N * 40)
        .map(|index| {
            let lane = index % 40;
            let thread = index / 40;
            0.125 + thread as f64 * 0.03125 + lane as f64 * 0.015625
        })
        .collect::<Vec<_>>();
    let d_hdiv_input = DeviceBuffer::from_host(&stream, &hdiv_input)
        .expect("upload H(div) iter_mut regression input");
    let mut d_hdiv_iter_mut = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();
    // SAFETY: every thread owns one 40-value input segment and one output.
    unsafe {
        module.hdiv_iter_mut_accumulate(stream.as_ref(), cfg, &d_hdiv_input, &mut d_hdiv_iter_mut)
    }
    .expect("launch hdiv_iter_mut_accumulate");
    let got_hdiv_iter_mut = d_hdiv_iter_mut.to_host_vec(&stream).unwrap();

    let ptx = std::fs::read_to_string(ptx_path).expect("read generated PTX");
    let fixed_axis_ptx = kernel_body(&ptx, "fixed_axis_aggregate_arrays");
    assert!(
        !fixed_axis_ptx.contains(".local")
            && !fixed_axis_ptx.contains("ld.local")
            && !fixed_axis_ptx.contains("st.local"),
        "constant-trip aggregate indexing must not use local memory:\n{fixed_axis_ptx}"
    );
    let runtime_index_ptx = kernel_body(&ptx, "runtime_aggregate_array_index");
    assert!(
        !runtime_index_ptx.contains(".local")
            && !runtime_index_ptx.contains("ld.local")
            && !runtime_index_ptx.contains("st.local"),
        "small runtime aggregate indexing must not use local memory:\n{runtime_index_ptx}"
    );
    let aligned_map_ptx = kernel_body(&ptx, "map_over_aligned_array");
    assert!(
        !aligned_map_ptx.contains(".local")
            && !aligned_map_ptx.contains("ld.local")
            && !aligned_map_ptx.contains("st.local"),
        "over-aligned array::map must not use local memory:\n{aligned_map_ptx}"
    );
    let hdiv_quadratic_ptx = kernel_body(&ptx, "hdiv_quadratic_array");
    assert!(
        !hdiv_quadratic_ptx.contains(".local")
            && !hdiv_quadratic_ptx.contains("ld.local")
            && !hdiv_quadratic_ptx.contains("st.local"),
        "bounded H(div) quadratic arrays must remain stackless:\n{hdiv_quadratic_ptx}"
    );

    let mut failures = 0usize;
    for tid in 0..N {
        let t = tid as u32;
        // sum of [t, t+1, t+2, t+3]
        let want_u32 = 4 * t + 6;
        // t*1 + (t+1)*2 + (t+2)*3 + (t+3)*4
        let want_pts = t + (t + 1) * 2 + (t + 2) * 3 + (t + 3) * 4;
        // sum([t, t+1, t+2, t+3].map(|value| value * 3 + 1))
        let want_mapped = 12 * t + 22;
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
        if got_mapped[tid] != want_mapped {
            println!(
                "FAIL tid={tid}: map_generated_array={} expected={want_mapped}",
                got_mapped[tid]
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
        let t = t as f32;
        let values = [t, t + 1.0, t + 2.0, t + 3.0, t + 4.0, t + 5.0];
        let want_fixed_axes: f32 = (0..3)
            .map(|row| {
                values
                    .chunks_exact(3)
                    .map(|lane| {
                        lane.iter()
                            .enumerate()
                            .map(|(component, value)| *value * (row + component + 1) as f32)
                            .sum::<f32>()
                    })
                    .sum::<f32>()
            })
            .sum();
        if (got_fixed_axes[tid] - want_fixed_axes).abs() > 1.0e-4 {
            println!(
                "FAIL tid={tid}: fixed_axis_aggregate_arrays={} expected={want_fixed_axes}",
                got_fixed_axes[tid]
            );
            failures += 1;
        }
        let component = tid % 3;
        let want_runtime_index = 2.0 * t + 3.0 + 2.0 * component as f32;
        if (got_runtime_index[tid] - want_runtime_index).abs() > 1.0e-4 {
            println!(
                "FAIL tid={tid}: runtime_aggregate_array_index={} expected={want_runtime_index}",
                got_runtime_index[tid]
            );
            failures += 1;
        }
        let want_aligned_map = 6.0 * t + 15.0;
        if (got_aligned_map[tid] - want_aligned_map).abs() > 1.0e-4 {
            println!(
                "FAIL tid={tid}: map_over_aligned_array={} expected={want_aligned_map}",
                got_aligned_map[tid]
            );
            failures += 1;
        }
        let input = &hdiv_input[tid * 40..(tid + 1) * 40];
        let mut reaction = [0.0_f64; 3];
        for row in 0..10 {
            let multiplier = input[row];
            for (component, reaction_component) in reaction.iter_mut().enumerate() {
                *reaction_component += multiplier * input[10 + 3 * row + component];
            }
        }
        for row in 0..4 {
            let multiplier = input[row];
            for (component, reaction_component) in reaction.iter_mut().enumerate() {
                *reaction_component += multiplier * input[component];
            }
        }
        for (component, reaction_component) in reaction.iter_mut().enumerate() {
            *reaction_component *= input[component];
        }
        let want_hdiv_iter_mut = reaction.iter().sum::<f64>();
        if (got_hdiv_iter_mut[tid] - want_hdiv_iter_mut).abs()
            > 1.0e-11 * want_hdiv_iter_mut.abs().max(1.0)
        {
            println!(
                "FAIL tid={tid}: hdiv_iter_mut_accumulate={} expected={want_hdiv_iter_mut}",
                got_hdiv_iter_mut[tid]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "array_for_loop: PASS ({N} threads, array iteration/helpers translated correctly)"
        );
    } else {
        println!("array_for_loop: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
