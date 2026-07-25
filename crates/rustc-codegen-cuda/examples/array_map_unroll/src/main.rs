/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for automatic unrolling of a small concrete `array::map`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];
}

#[derive(Clone, Copy)]
struct Pair([f32; 2]);

macro_rules! mix_round {
    ($value:ident, $coefficient:ident) => {{
        $value = $value.mul_add(1.000_244_1, $coefficient * 0.000_976_562_5);
        $coefficient = $coefficient.mul_add(0.999_511_7, 0.000_122_070_31);
    }};
}

#[inline(always)]
fn mix(mut value: f32, mut coefficient: f32) -> f32 {
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    mix_round!(value, coefficient);
    value.mul_add(1.000_244_1, coefficient * 0.000_976_562_5)
}

fn evaluate(input: [f32; 3], bias: f32) -> f32 {
    let mapped = Axis::ALL.map(|axis| {
        let index = axis as usize;
        Pair([
            mix(input[index], bias + index as f32 * 0.125),
            mix(input[index] - bias, 0.5 + index as f32 * 0.25),
        ])
    });
    mapped[0].0[0]
        + mapped[0].0[1]
        + mapped[1].0[0]
        + mapped[1].0[1]
        + mapped[2].0[0]
        + mapped[2].0[1]
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `input` covers three values and `output` covers one value.
    #[kernel]
    pub unsafe fn map_three(input: *const f32, output: *mut f32, bias: f32) {
        if thread::index_1d().get() == 0 {
            // SAFETY: upheld by the kernel contract.
            let values = unsafe { [input.read(), input.add(1).read(), input.add(2).read()] };
            // SAFETY: upheld by the kernel contract.
            unsafe { output.write(evaluate(values, bias)) };
        }
    }
}

fn assert_stackless_entry() {
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/array_map_unroll.ptx");
    let ptx = std::fs::read_to_string(ptx_path).expect("device PTX was not emitted");
    let marker = ".visible .entry map_three(";
    let entry = ptx
        .split_once(marker)
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.split_once("// -- End function").map(|(body, _)| body))
        .expect("map_three PTX entry was not found");
    for forbidden in [".local", "ld.local", "st.local", "trap;"] {
        assert!(
            !entry.contains(forbidden),
            "map_three unexpectedly contains {forbidden:?}"
        );
    }
}

fn main() {
    const BIAS: f32 = 0.375;
    let input = [0.25, -0.75, 1.5];
    let expected = evaluate(input, BIAS);

    let context = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = context.default_stream();
    let input_device = DeviceBuffer::from_host(&stream, &input).expect("failed to upload input");
    let output_device = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("failed to allocate output");
    let module = kernels::load(&context).expect("failed to load CUDA module");

    // SAFETY: the device allocations match the kernel contract and remain live
    // through the synchronized download.
    unsafe {
        module.map_three(
            &stream,
            LaunchConfig::for_num_elems(1),
            input_device.cu_deviceptr() as *const f32,
            output_device.cu_deviceptr() as *mut f32,
            BIAS,
        )
    }
    .expect("map_three launch failed");

    let actual = output_device
        .to_host_vec(&stream)
        .expect("failed to download output")[0];
    let tolerance = 2.0e-5 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "GPU result {actual} differs from host result {expected}"
    );
    assert_stackless_entry();
    println!("array map unroll: PASS");
}
