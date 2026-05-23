/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::{
    address_space, gpu_only,
    warp::{WarpShuffleMode, WarpShuffleValue},
};

#[address_space(shared)]
static mut SCRATCH: [u32; 32] = [0; 32];

#[gpu_only]
fn device_only_identity(value: u32) -> u32 {
    unsafe {
        SCRATCH[0] = value;
    }
    value
}

fn assert_shuffle_value<T: WarpShuffleValue>() {}

fn main() {
    assert_shuffle_value::<u32>();
    assert_shuffle_value::<i32>();
    assert_shuffle_value::<f32>();

    let _: unsafe fn(WarpShuffleMode, u32, u32, u32, u32) -> (u32, bool) =
        <u32 as WarpShuffleValue>::shuffle;

    let _ = device_only_identity as fn(u32) -> u32;
}
