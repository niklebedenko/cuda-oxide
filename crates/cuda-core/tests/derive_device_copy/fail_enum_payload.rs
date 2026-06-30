// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use core::num::NonZeroU32;

use cuda_core::DeviceCopy;

// The only variant contains a `NonZeroU32`, so the all-zero bit pattern is not
// a valid `BadPayload` even though the payload type currently implements
// `DeviceCopy` for compatibility.
#[derive(Copy, Clone, DeviceCopy)]
enum BadPayload {
    Scalar(NonZeroU32),
}

fn main() {
    let _ = core::mem::size_of::<BadPayload>();
}
