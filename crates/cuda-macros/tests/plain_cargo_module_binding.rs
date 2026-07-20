// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn no_op() {}
}

#[test]
fn plain_cargo_from_modules_links_without_an_artifact_anchor() {
    let loaded = kernels::from_modules(Vec::new()).expect("empty module set is structurally valid");
    drop(loaded);
}
