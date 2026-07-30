/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for an iterator returning `Option<Axis>`.
//!
//! `Option<Axis>` uses the otherwise-invalid fourth `usize` discriminant as
//! its niche. Extracting the integer-only `Axis` payload through a stack spill
//! hides the fact that its discriminant is one of `0..3`, leaving unreachable
//! bounds traps when the yielded axis indexes `[T; 3]`. The compiler keeps this
//! narrow payload shape in SSA so LLVM can retain the finite discriminant
//! dataflow.
//!
//! Run:
//!   cargo oxide run option_axis_index
//!   ./crates/rustc-codegen-cuda/examples/option_axis_index/verify-code-shape.sh

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    use core::marker::PhantomData;

    #[derive(Clone, Copy)]
    #[repr(usize)]
    enum Axis {
        X,
        Y,
        Z,
    }

    struct AxisIter {
        idx: usize,
        back_idx: usize,
        marker: PhantomData<fn() -> ()>,
    }

    impl AxisIter {
        #[inline]
        fn get(&self, index: usize) -> Option<Axis> {
            match index {
                0 => Some(Axis::X),
                1 => Some(Axis::Y),
                2 => Some(Axis::Z),
                _ => None,
            }
        }
    }

    impl Axis {
        #[inline]
        fn iter() -> AxisIter {
            AxisIter {
                idx: 0,
                back_idx: 0,
                marker: PhantomData,
            }
        }
    }

    impl Iterator for AxisIter {
        type Item = Axis;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.nth(0)
        }

        #[inline]
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            let idx = self.idx + n + 1;
            if idx + self.back_idx > 3 {
                self.idx = 3;
                None
            } else {
                self.idx = idx;
                AxisIter::get(self, idx - 1)
            }
        }
    }

    #[inline(always)]
    fn update_axis_slots(x: u32, y: u32, z: u32, scale: u32, bias: u32) -> u32 {
        let source = [x, y, z];
        let mut result = [11_u32, 13, 17];

        for axis in Axis::iter() {
            let index = axis as usize;
            result[index] = source[index].wrapping_mul(scale).wrapping_add(bias);
        }

        result[0]
            .wrapping_mul(17)
            .wrapping_add(result[1])
            .wrapping_mul(17)
            .wrapping_add(result[2])
    }

    #[kernel]
    pub fn option_axis_index_kernel(
        x: u32,
        y: u32,
        z: u32,
        scale: u32,
        bias: u32,
        mut output: DisjointSlice<u32>,
    ) {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = update_axis_slots(x, y, z, scale, bias);
        }
    }

    /// Positive control: an actually-unbounded runtime index must keep its
    /// bounds trap. The enum fix must not delete genuine bounds checks.
    #[kernel]
    pub fn runtime_index_control(index: usize, mut output: DisjointSlice<u32>) {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            let values = [29_u32, 31, 37];
            *slot = values[index];
        }
    }
}

fn expected(x: u32, y: u32, z: u32, scale: u32, bias: u32) -> u32 {
    let result = [
        x.wrapping_mul(scale).wrapping_add(bias),
        y.wrapping_mul(scale).wrapping_add(bias),
        z.wrapping_mul(scale).wrapping_add(bias),
    ];
    result[0]
        .wrapping_mul(17)
        .wrapping_add(result[1])
        .wrapping_mul(17)
        .wrapping_add(result[2])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let inputs = (5_u32, 7_u32, 11_u32, 13_u32, 17_u32);

    // SAFETY: one thread writes the single-element output, and all scalar
    // arguments remain valid for the duration of the launch.
    unsafe {
        module.option_axis_index_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            inputs.0,
            inputs.1,
            inputs.2,
            inputs.3,
            inputs.4,
            &mut output,
        )
    }?;
    let actual = output.to_host_vec(&stream)?;
    assert_eq!(
        actual,
        vec![expected(inputs.0, inputs.1, inputs.2, inputs.3, inputs.4)]
    );

    // SAFETY: index 1 is valid for the control's three-element local array,
    // and one thread writes the single-element output.
    unsafe {
        module.runtime_index_control(&stream, LaunchConfig::for_num_elems(1), 1, &mut output)
    }?;
    assert_eq!(output.to_host_vec(&stream)?, vec![31]);

    println!("option_axis_index: PASS");
    Ok(())
}
