/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for a directly invoked closure borrowing a by-value kernel
//! aggregate.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
pub struct AggregateArgs {
    input: *const u32,
    output: *mut u32,
    count: u32,
    c00: u32,
    c01: u32,
    c02: u32,
    c03: u32,
    c04: u32,
    c05: u32,
    c06: u32,
    c07: u32,
    c08: u32,
    c09: u32,
    c10: u32,
    c11: u32,
    c12: u32,
    c13: u32,
    c14: u32,
    c15: u32,
}

macro_rules! mix_round {
    ($value:ident, $args:ident) => {{
        $value = $value.wrapping_mul($args.c00 | 1).rotate_left(1) ^ $args.c01;
        $value = $value.wrapping_add($args.c02).rotate_left(3) ^ $args.c03;
        $value = $value.wrapping_mul($args.c04 | 1).rotate_left(5) ^ $args.c05;
        $value = $value.wrapping_add($args.c06).rotate_left(7) ^ $args.c07;
        $value = $value.wrapping_mul($args.c08 | 1).rotate_left(9) ^ $args.c09;
        $value = $value.wrapping_add($args.c10).rotate_left(11) ^ $args.c11;
        $value = $value.wrapping_mul($args.c12 | 1).rotate_left(13) ^ $args.c13;
        $value = $value.wrapping_add($args.c14).rotate_left(15) ^ $args.c15;
    }};
}

macro_rules! mix_all {
    ($value:ident, $args:ident) => {{
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
        mix_round!($value, $args);
    }};
}

macro_rules! mix_heavy {
    ($value:ident, $args:ident) => {{
        mix_all!($value, $args);
        mix_all!($value, $args);
        mix_all!($value, $args);
        mix_all!($value, $args);
        mix_all!($value, $args);
        mix_all!($value, $args);
        mix_all!($value, $args);
        mix_all!($value, $args);
    }};
}

macro_rules! mix_very_heavy {
    ($value:ident, $args:ident) => {{
        mix_heavy!($value, $args);
        mix_heavy!($value, $args);
    }};
}

fn evaluate(args: &AggregateArgs, input: u32) -> u32 {
    let mut value = input;
    mix_very_heavy!(value, args);
    value
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `args.input` and `args.output` must cover `args.count` elements.
    #[kernel]
    pub unsafe fn aggregate_closure(args: AggregateArgs) {
        let index = thread::index_1d().get();
        if index < args.count as usize {
            let evaluate = |input: u32| {
                let mut value = input;
                mix_very_heavy!(value, args);
                value
            };
            // SAFETY: the kernel contract covers this in-range index.
            let input = unsafe { *args.input.add(index) };
            // SAFETY: the kernel contract covers this in-range index.
            unsafe { args.output.add(index).write(evaluate(input)) };
        }
    }
}

fn assert_stackless_entry() {
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/aggregate_closure_inline.ptx");
    let ptx = std::fs::read_to_string(ptx_path).expect("device PTX was not emitted");
    let marker = ".visible .entry aggregate_closure(";
    let entry = ptx
        .split_once(marker)
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.split_once("// -- End function").map(|(body, _)| body))
        .expect("aggregate_closure PTX entry was not found");
    for forbidden in [".local", "ld.local", "st.local", "call."] {
        assert!(
            !entry.contains(forbidden),
            "aggregate_closure unexpectedly contains {forbidden:?}"
        );
    }
}

fn main() {
    const COUNT: usize = 257;
    let input = (0..COUNT)
        .map(|index| (index as u32).wrapping_mul(0x9e37_79b9))
        .collect::<Vec<_>>();

    let context = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = context.default_stream();
    let input_device = DeviceBuffer::from_host(&stream, &input).expect("failed to upload input");
    let output_device =
        DeviceBuffer::<u32>::zeroed(&stream, COUNT).expect("failed to allocate output");
    let args = AggregateArgs {
        input: input_device.cu_deviceptr() as *const u32,
        output: output_device.cu_deviceptr() as *mut u32,
        count: COUNT as u32,
        c00: 0x243f_6a88,
        c01: 0x85a3_08d3,
        c02: 0x1319_8a2e,
        c03: 0x0370_7344,
        c04: 0xa409_3822,
        c05: 0x299f_31d0,
        c06: 0x082e_fa98,
        c07: 0xec4e_6c89,
        c08: 0x4528_21e6,
        c09: 0x38d0_1377,
        c10: 0xbe54_66cf,
        c11: 0x34e9_0c6c,
        c12: 0xc0ac_29b7,
        c13: 0xc97c_50dd,
        c14: 0x3f84_d5b5,
        c15: 0xb547_0917,
    };
    let expected = input
        .iter()
        .map(|&value| evaluate(&args, value))
        .collect::<Vec<_>>();

    let module = kernels::load(&context).expect("failed to load CUDA module");
    // SAFETY: both allocations cover COUNT elements and remain live through
    // the synchronized download.
    unsafe {
        module.aggregate_closure(
            &stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            args,
        )
    }
    .expect("aggregate closure launch failed");

    let actual = output_device
        .to_host_vec(&stream)
        .expect("failed to download output");
    assert_eq!(actual, expected, "GPU aggregate closure result differs");
    assert_stackless_entry();
    println!("aggregate closure inline: PASS");
}
