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

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Field([f32; 20]);

macro_rules! mix_round {
    ($value:ident, $coefficient:ident) => {{
        $value = $value.mul_add(1.000_244_1, $coefficient * 0.000_976_562_5);
        $coefficient = $coefficient.mul_add(0.999_511_7, 0.000_122_070_31);
    }};
}

macro_rules! field_mix_rounds {
    ($field:ident, $coefficient:ident, $($tag:literal),+ $(,)?) => {
        $(
            {
                let node = $tag % 20;
                let neighbour = ($tag * 7 + 3) % 20;
                let own = $field.0[node];
                let other = $field.0[neighbour];
                $field.0[node] = own.mul_add(
                    0.999_511_7 + $tag as f32 * 0.000_000_119_209_29,
                    other * 0.000_244_140_63 + $coefficient,
                );
                $coefficient = $coefficient.mul_add(
                    0.999_755_86,
                    own * 0.000_015_258_789 + $tag as f32 * 0.000_000_953_674_3,
                );
            }
        )+
    };
}

macro_rules! transform_field {
    ($field:expr, $axis:expr, $bias:expr) => {{
        let mut field = $field;
        let mut coefficient = $bias + $axis as usize as f32 * 0.125;
        field_mix_rounds!(
            field,
            coefficient,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        );
        field.0[0] += coefficient * 0.000_030_517_578;
        field
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

#[inline(always)]
fn evaluate_wide(input: [Field; 3], bias: f32) -> f32 {
    let [x, y, z] = Axis::ALL.map(|axis| transform_field!(input[axis as usize], axis, bias));
    checksum_field(x) + checksum_field(y) * 2.0 + checksum_field(z) * 3.0
}

#[inline(always)]
fn checksum_field(field: Field) -> f32 {
    field.0[0]
        + field.0[1] * 2.0
        + field.0[2] * 3.0
        + field.0[3] * 4.0
        + field.0[4] * 5.0
        + field.0[5] * 6.0
        + field.0[6] * 7.0
        + field.0[7] * 8.0
        + field.0[8] * 9.0
        + field.0[9] * 10.0
        + field.0[10] * 11.0
        + field.0[11] * 12.0
        + field.0[12] * 13.0
        + field.0[13] * 14.0
        + field.0[14] * 15.0
        + field.0[15] * 16.0
        + field.0[16] * 17.0
        + field.0[17] * 18.0
        + field.0[18] * 19.0
        + field.0[19] * 20.0
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

    /// # Safety
    ///
    /// `input` covers sixty values and `output` covers one value.
    #[kernel]
    pub unsafe fn map_three_wide(input: *const f32, output: *mut f32, bias: f32) {
        if thread::index_1d().get() == 0 {
            let [x, y, z] = Axis::ALL.map(|axis| {
                let base = axis as usize * 20;
                let field = Field([
                    // SAFETY: upheld by the kernel contract.
                    unsafe { input.add(base).read() },
                    unsafe { input.add(base + 1).read() },
                    unsafe { input.add(base + 2).read() },
                    unsafe { input.add(base + 3).read() },
                    unsafe { input.add(base + 4).read() },
                    unsafe { input.add(base + 5).read() },
                    unsafe { input.add(base + 6).read() },
                    unsafe { input.add(base + 7).read() },
                    unsafe { input.add(base + 8).read() },
                    unsafe { input.add(base + 9).read() },
                    unsafe { input.add(base + 10).read() },
                    unsafe { input.add(base + 11).read() },
                    unsafe { input.add(base + 12).read() },
                    unsafe { input.add(base + 13).read() },
                    unsafe { input.add(base + 14).read() },
                    unsafe { input.add(base + 15).read() },
                    unsafe { input.add(base + 16).read() },
                    unsafe { input.add(base + 17).read() },
                    unsafe { input.add(base + 18).read() },
                    unsafe { input.add(base + 19).read() },
                ]);
                transform_field!(field, axis, bias)
            });
            // SAFETY: upheld by the kernel contract.
            unsafe {
                output.write(checksum_field(x) + checksum_field(y) * 2.0 + checksum_field(z) * 3.0)
            };
        }
    }
}

fn assert_stackless_entry(name: &str) {
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/array_map_unroll.ptx");
    let ptx = std::fs::read_to_string(ptx_path).expect("device PTX was not emitted");
    let marker = format!(".visible .entry {name}(");
    let entry = ptx
        .split_once(&marker)
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.split_once("// -- End function").map(|(body, _)| body))
        .unwrap_or_else(|| panic!("{name} PTX entry was not found"));
    for forbidden in [".local", "ld.local", "st.local", "call.", "trap;"] {
        assert!(
            !entry.contains(forbidden),
            "{name} unexpectedly contains {forbidden:?}"
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

    let wide_input = core::array::from_fn::<_, 3, _>(|field| {
        Field(core::array::from_fn(|node| {
            (field * 20 + node) as f32 * 0.03125 - 0.75
        }))
    });
    let wide_flat = wide_input
        .iter()
        .flat_map(|field| field.0)
        .collect::<Vec<_>>();
    let expected_wide = evaluate_wide(wide_input, BIAS);
    let wide_input_device =
        DeviceBuffer::from_host(&stream, &wide_flat).expect("failed to upload wide input");
    // SAFETY: the input covers sixty values and the output covers one value.
    unsafe {
        module.map_three_wide(
            &stream,
            LaunchConfig::for_num_elems(1),
            wide_input_device.cu_deviceptr() as *const f32,
            output_device.cu_deviceptr() as *mut f32,
            BIAS,
        )
    }
    .expect("map_three_wide launch failed");
    let actual_wide = output_device
        .to_host_vec(&stream)
        .expect("failed to download wide output")[0];
    let wide_tolerance = 2.0e-5 * expected_wide.abs().max(1.0);
    assert!(
        (actual_wide - expected_wide).abs() <= wide_tolerance,
        "wide GPU result {actual_wide} differs from host result {expected_wide}"
    );

    assert_stackless_entry("map_three");
    assert_stackless_entry("map_three_wide");
    println!("array map unroll: PASS");
}
