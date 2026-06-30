// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::{NonZeroU32, Wrapping};

use cuda_core::DeviceCopy;

fn assert_device_copy<T: DeviceCopy>() {}

#[test]
fn device_copy_covers_core_parity_types() {
    assert_device_copy::<PhantomData<String>>();
    assert_device_copy::<MaybeUninit<u32>>();
    assert_device_copy::<Wrapping<u64>>();

    // This fork keeps cust_core parity for these validity-narrow types. The
    // enum derive still validates the concrete enum's all-zero pattern instead
    // of assuming that field-level `DeviceCopy` is enough.
    assert_device_copy::<bool>();
    assert_device_copy::<char>();
    assert_device_copy::<NonZeroU32>();
    assert_device_copy::<Option<u32>>();
    assert_device_copy::<Result<u32, u16>>();
}
