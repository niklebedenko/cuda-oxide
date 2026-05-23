/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[test]
fn cuda_std_compat_api_typechecks() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/cuda_std_compat_api.rs");
}
