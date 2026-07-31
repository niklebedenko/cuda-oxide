/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use element_lib::{
    apply_operator, checksum, make_pressure_volume, make_volume, pressure_operator,
};

// Multiple kernels exercise the generated module-set loading path used by Impulse.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn hdiv_like(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if raw_index < input.len()
            && let Some(slot) = output.get_mut(index)
        {
            let base = input[raw_index];
            let lane = ((raw_index & 31) as f32) * 0.03125;
            let coefficient = base.abs().sqrt() + 0.5;
            let first = checksum(apply_operator(make_volume(base, lane), coefficient));
            let second = checksum(apply_operator(
                make_volume(base + first * 0.0001, lane + 0.03125),
                coefficient + 0.125,
            ));
            let third = checksum(apply_operator(
                make_volume(base + second * 0.0001, lane + 0.0625),
                coefficient + 0.25,
            ));
            let fourth = checksum(apply_operator(
                make_volume(base + third * 0.0001, lane + 0.09375),
                coefficient + 0.375,
            ));
            *slot = first + second * 0.5 + third * 0.25 + fourth * 0.125;
        }
    }

    #[kernel]
    pub fn residual_probe(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if raw_index < input.len()
            && let Some(slot) = output.get_mut(index)
        {
            let base = input[raw_index];
            let lane = ((raw_index & 31) as f32) * 0.03125;
            let coefficient = base.abs().sqrt() + 0.75;
            let first = checksum(apply_operator(make_volume(base, lane), coefficient));
            let second = checksum(apply_operator(
                make_volume(base + first * 0.0002, lane + 0.0625),
                coefficient + 0.25,
            ));
            *slot = first - second * 0.375;
        }
    }

    #[kernel]
    pub fn mass_probe(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if raw_index < input.len()
            && let Some(slot) = output.get_mut(index)
        {
            let base = input[raw_index];
            let lane = ((raw_index & 31) as f32) * 0.03125;
            let coefficient = base.abs().sqrt() + 1.0;
            *slot = checksum(apply_operator(make_volume(base, lane), coefficient)) * 0.125;
        }
    }

    #[kernel]
    pub fn flux_probe(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if raw_index < input.len()
            && let Some(slot) = output.get_mut(index)
        {
            let base = input[raw_index];
            let lane = ((raw_index & 31) as f32) * 0.03125;
            let coefficient = base.abs().sqrt() + 1.25;
            let first = checksum(apply_operator(make_volume(base, lane), coefficient));
            let second = checksum(apply_operator(
                make_volume(base - first * 0.0001, lane + 0.046875),
                coefficient + 0.1875,
            ));
            let third = checksum(apply_operator(
                make_volume(base + second * 0.0001, lane + 0.078125),
                coefficient + 0.3125,
            ));
            *slot = first * 0.5 + second * 0.25 - third * 0.125;
        }
    }

    #[kernel]
    pub fn face_frame_pressure(input: &[f64], mut output: DisjointSlice<f64>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if raw_index < input.len()
            && let Some(slot) = output.get_mut(index)
        {
            let base = input[raw_index];
            let lane = ((raw_index & 31) as f64) * 0.03125;
            *slot = pressure_operator(make_pressure_volume(base, lane), base.abs() + 0.5);
        }
    }
}

fn expected_hdiv(input: f32, index: usize) -> f32 {
    let lane = ((index & 31) as f32) * 0.03125;
    let coefficient = input.abs().sqrt() + 0.5;
    let first = checksum(apply_operator(make_volume(input, lane), coefficient));
    let second = checksum(apply_operator(
        make_volume(input + first * 0.0001, lane + 0.03125),
        coefficient + 0.125,
    ));
    let third = checksum(apply_operator(
        make_volume(input + second * 0.0001, lane + 0.0625),
        coefficient + 0.25,
    ));
    let fourth = checksum(apply_operator(
        make_volume(input + third * 0.0001, lane + 0.09375),
        coefficient + 0.375,
    ));
    first + second * 0.5 + third * 0.25 + fourth * 0.125
}

fn expected_residual(input: f32, index: usize) -> f32 {
    let lane = ((index & 31) as f32) * 0.03125;
    let coefficient = input.abs().sqrt() + 0.75;
    let first = checksum(apply_operator(make_volume(input, lane), coefficient));
    let second = checksum(apply_operator(
        make_volume(input + first * 0.0002, lane + 0.0625),
        coefficient + 0.25,
    ));
    first - second * 0.375
}

fn expected_mass(input: f32, index: usize) -> f32 {
    let lane = ((index & 31) as f32) * 0.03125;
    let coefficient = input.abs().sqrt() + 1.0;
    checksum(apply_operator(make_volume(input, lane), coefficient)) * 0.125
}

fn expected_flux(input: f32, index: usize) -> f32 {
    let lane = ((index & 31) as f32) * 0.03125;
    let coefficient = input.abs().sqrt() + 1.25;
    let first = checksum(apply_operator(make_volume(input, lane), coefficient));
    let second = checksum(apply_operator(
        make_volume(input - first * 0.0001, lane + 0.046875),
        coefficient + 0.1875,
    ));
    let third = checksum(apply_operator(
        make_volume(input + second * 0.0001, lane + 0.078125),
        coefficient + 0.3125,
    ));
    first * 0.5 + second * 0.25 - third * 0.125
}

fn expected_face_frame_pressure(input: f64, index: usize) -> f64 {
    let lane = ((index & 31) as f64) * 0.03125;
    pressure_operator(
        make_pressure_volume(input, lane),
        input.abs() + 0.5,
    )
}

fn validate_output(label: &str, output: &[f32], input: &[f32], expected: fn(f32, usize) -> f32) {
    for (index, (&got, &input)) in output.iter().zip(input).enumerate() {
        let want = expected(input, index);
        assert!(
            (got - want).abs() <= 2.0e-4 * want.abs().max(1.0),
            "{label} index {index}: GPU {got}, host {want}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const COUNT: usize = 1024;
    let input: Vec<f32> = (0..COUNT)
        .map(|index| index as f32 * 0.001953125 - 0.75)
        .collect();
    let input_device = DeviceBuffer::from_host(&stream, &input)?;
    let mut output_device = DeviceBuffer::<f32>::zeroed(&stream, COUNT)?;

    // SAFETY: the 1-D launch covers both buffers and matches `index_1d`.
    unsafe {
        module.hdiv_like(
            &stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &input_device,
            &mut output_device,
        )
    }?;

    validate_output(
        "hdiv_like",
        &output_device.to_host_vec(&stream)?,
        &input,
        expected_hdiv,
    );

    // SAFETY: every launch uses the same matching, COUNT-element buffers.
    unsafe {
        module.residual_probe(
            &stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &input_device,
            &mut output_device,
        )
    }?;
    validate_output(
        "residual_probe",
        &output_device.to_host_vec(&stream)?,
        &input,
        expected_residual,
    );

    // SAFETY: every launch uses the same matching, COUNT-element buffers.
    unsafe {
        module.mass_probe(
            &stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &input_device,
            &mut output_device,
        )
    }?;
    validate_output(
        "mass_probe",
        &output_device.to_host_vec(&stream)?,
        &input,
        expected_mass,
    );

    // SAFETY: every launch uses the same matching, COUNT-element buffers.
    unsafe {
        module.flux_probe(
            &stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &input_device,
            &mut output_device,
        )
    }?;
    validate_output(
        "flux_probe",
        &output_device.to_host_vec(&stream)?,
        &input,
        expected_flux,
    );

    let pressure_input: Vec<f64> = input.iter().copied().map(f64::from).collect();
    let pressure_input_device = DeviceBuffer::from_host(&stream, &pressure_input)?;
    let mut pressure_output_device = DeviceBuffer::<f64>::zeroed(&stream, COUNT)?;
    // SAFETY: the 1-D launch covers both matching COUNT-element f64 buffers.
    unsafe {
        module.face_frame_pressure(
            &stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &pressure_input_device,
            &mut pressure_output_device,
        )
    }?;
    let pressure_output = pressure_output_device.to_host_vec(&stream)?;
    for (index, (&got, &input)) in pressure_output.iter().zip(&pressure_input).enumerate() {
        let want = expected_face_frame_pressure(input, index);
        assert!(
            (got - want).abs() <= 2.0e-11 * want.abs().max(1.0),
            "face_frame_pressure index {index}: GPU {got}, host {want}"
        );
    }

    println!("SUCCESS: five H(div)-shaped kernels agree for {COUNT} aggregate-helper results each");
    Ok(())
}
