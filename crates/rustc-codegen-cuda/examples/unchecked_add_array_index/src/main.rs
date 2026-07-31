/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for Rust range iteration over two correlated nine-element arrays.
//!
//! `Range<usize>::next` advances its induction variable with MIR
//! `AddUnchecked`. Preserving that no-overflow promise strengthens LLVM range
//! analysis in larger production loops. This regression pins the promise on
//! the correlated loop while retaining a bounds trap for an unbounded control.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    const SCALAR_COUNT: usize = 9;

    #[inline(always)]
    fn correlate(left: &[u32; SCALAR_COUNT], right: &[u32; SCALAR_COUNT]) -> u32 {
        let mut result = [0_u32; SCALAR_COUNT];
        for scalar in 0..SCALAR_COUNT {
            let mut value = left[scalar].wrapping_add(right[scalar]);
            value = value.wrapping_mul(17).rotate_left(3).wrapping_add(11);
            value = value.wrapping_mul(19).rotate_left(5).wrapping_add(13);
            value = value.wrapping_mul(23).rotate_left(7).wrapping_add(17);
            value = value.wrapping_mul(29).rotate_left(11).wrapping_add(19);
            result[scalar] = value;
        }
        result.into_iter().fold(0, u32::wrapping_add)
    }

    #[kernel]
    pub fn correlated_array_index(seed: u32, mut output: DisjointSlice<u32>) {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            let left = core::array::from_fn(|index| seed.wrapping_add(index as u32));
            let right = core::array::from_fn(|index| {
                seed.wrapping_mul(3).wrapping_add((index as u32) * 5)
            });
            *slot = correlate(&left, &right);
        }
    }

    /// Positive control: a genuinely unbounded index must retain its trap.
    #[kernel]
    pub fn runtime_index_control(index: usize, mut output: DisjointSlice<u32>) {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = [29_u32, 31, 37][index];
        }
    }
}

fn expected(seed: u32) -> u32 {
    let left: [u32; 9] = core::array::from_fn(|index| seed.wrapping_add(index as u32));
    let right: [u32; 9] = core::array::from_fn(|index| {
        seed.wrapping_mul(3).wrapping_add((index as u32) * 5)
    });
    let mut result = [0_u32; 9];
    for scalar in 0..9 {
        let mut value = left[scalar].wrapping_add(right[scalar]);
        value = value.wrapping_mul(17).rotate_left(3).wrapping_add(11);
        value = value.wrapping_mul(19).rotate_left(5).wrapping_add(13);
        value = value.wrapping_mul(23).rotate_left(7).wrapping_add(17);
        value = value.wrapping_mul(29).rotate_left(11).wrapping_add(19);
        result[scalar] = value;
    }
    result.into_iter().fold(0, u32::wrapping_add)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let seed = 41_u32;

    // SAFETY: one thread writes the one-element output buffer.
    unsafe {
        module.correlated_array_index(
            &stream,
            LaunchConfig::for_num_elems(1),
            seed,
            &mut output,
        )
    }?;
    assert_eq!(output.to_host_vec(&stream)?, [expected(seed)]);

    // SAFETY: index 1 is valid for the control's three-element array.
    unsafe {
        module.runtime_index_control(&stream, LaunchConfig::for_num_elems(1), 1, &mut output)
    }?;
    assert_eq!(output.to_host_vec(&stream)?, [31]);

    println!("unchecked_add_array_index: PASS");
    Ok(())
}
