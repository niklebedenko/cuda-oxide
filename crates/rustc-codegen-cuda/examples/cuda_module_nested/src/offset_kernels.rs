/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// Included nested module: out[i] = a[i] + 10.
pub mod offset {
    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn offset_by(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(elem) = out.get_mut(idx) {
            *elem = a[idx_raw] + 10.0;
        }
    }
}
