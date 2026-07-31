/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal range-proof coverage for fieldless enums loaded from device memory.

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Axis {
    X,
    Y,
    Z,
}

// SAFETY: `Axis` is a fieldless `repr(usize)` enum whose valid values are POD.
unsafe impl DeviceCopy for Axis {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SparseTag {
    Zero = 0,
    Far = 7,
}

// SAFETY: `SparseTag` is a fieldless `repr(u32)` enum whose valid values are POD.
unsafe impl DeviceCopy for SparseTag {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TetFace {
    LowX,
    LowY,
    LowZ,
    HighX,
}

impl TetFace {
    #[inline(always)]
    fn from_axis_high_low(axis: Axis, high: bool) -> Self {
        match (axis, high) {
            (Axis::X, false) => Self::LowX,
            (Axis::X, true) => Self::HighX,
            (Axis::Y, false) => Self::LowY,
            (Axis::Z, false) => Self::LowZ,
            (Axis::Y | Axis::Z, true) => unreachable!(),
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn select_axis(values: [u32; 3], axis: Axis) -> u32 {
        values[axis as usize]
    }

    #[inline(always)]
    fn select_tet_face(values: [u32; 4], axis: Axis, high: bool) -> u32 {
        values[TetFace::from_axis_high_low(axis, high) as usize]
    }

    /// Reproduction: a valid fieldless enum loaded from global memory has a
    /// finite range which should discharge the three-element bounds check.
    #[kernel]
    pub fn loaded_axis_index(axes: &[Axis], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let (Some(&axis), Some(slot)) = (axes.get(idx.get()), output.get_mut(idx)) {
            *slot = select_axis([11, 13, 17], axis);
        }
    }

    /// The X-specialized tetrahedral path is valid for both boolean sides and
    /// should select only LowX or HighX without fallback or bounds traps.
    #[kernel]
    pub fn literal_x_tet_face(high: bool, mut output: DisjointSlice<u32>) {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = select_tet_face([19, 23, 29, 31], Axis::X, high);
        }
    }

    /// Positive control: a genuinely unbounded integer index must keep its
    /// bounds trap.
    #[kernel]
    pub fn runtime_index_control(index: usize, mut output: DisjointSlice<u32>) {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = [37, 41, 43][index];
        }
    }

    /// Positive control: sparse enum discriminants are not a zero-based
    /// contiguous range. `SparseTag::Far` is a valid enum value but an invalid
    /// index here, so this kernel must retain its bounds trap.
    #[kernel]
    pub fn sparse_enum_control(tags: &[SparseTag], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let (Some(&tag), Some(slot)) = (tags.get(idx.get()), output.get_mut(idx)) {
            *slot = [47, 53, 59][tag as usize];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let axes = [Axis::X, Axis::Y, Axis::Z];
    let axes = DeviceBuffer::from_host(&stream, &axes)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, 3)?;

    // SAFETY: the input and output each cover the three launched threads.
    unsafe {
        module.loaded_axis_index(&stream, LaunchConfig::for_num_elems(3), &axes, &mut output)
    }?;
    assert_eq!(output.to_host_vec(&stream)?, vec![11, 13, 17]);

    // SAFETY: one output element exists and both boolean values are valid.
    unsafe {
        module.literal_x_tet_face(&stream, LaunchConfig::for_num_elems(1), false, &mut output)
    }?;
    assert_eq!(output.to_host_vec(&stream)?[0], 19);
    unsafe {
        module.literal_x_tet_face(&stream, LaunchConfig::for_num_elems(1), true, &mut output)
    }?;
    assert_eq!(output.to_host_vec(&stream)?[0], 31);

    // SAFETY: index 1 is valid for the control's three-element array.
    unsafe {
        module.runtime_index_control(&stream, LaunchConfig::for_num_elems(1), 1, &mut output)
    }?;
    assert_eq!(output.to_host_vec(&stream)?[0], 41);

    let sparse = DeviceBuffer::from_host(&stream, &[SparseTag::Zero])?;
    // SAFETY: the live control uses the in-range `Zero` discriminant. Its
    // valid-but-out-of-range `Far` variant remains covered by the PTX check.
    unsafe {
        module.sparse_enum_control(
            &stream,
            LaunchConfig::for_num_elems(1),
            &sparse,
            &mut output,
        )
    }?;
    assert_eq!(output.to_host_vec(&stream)?[0], 47);

    println!("enum_face_index: PASS");
    Ok(())
}
