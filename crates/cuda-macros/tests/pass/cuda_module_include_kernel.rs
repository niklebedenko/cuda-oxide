// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{CudaStream, LaunchConfig};
use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    pub fn root(value: u32) {
        let _ = value;
    }

    include!("cuda_module_include_kernel_items.rs");
}

fn included_launcher(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
) {
    // SAFETY: this type-checking fixture supplies the raw launch proof.
    let _ = unsafe { module.from_include(stream, config, 1u32) };
}

fn main() {}
