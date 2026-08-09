/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA-codegen adapters for the shared LLVM toolchain resolver.

use crate::options::BackendOptions;
use std::path::PathBuf;

pub(crate) use cuda_artifact_finalizer::{LlvmToolchain, OptTool, describe_tool, probe_runnable};

fn selection_options(options: &BackendOptions) -> cuda_artifact_finalizer::LlvmToolchainOptions {
    cuda_artifact_finalizer::LlvmToolchainOptions {
        no_opt: options.no_opt,
        llc_override: options.llc_override.clone(),
        opt_override: options.opt_override.clone(),
        llvm_link_override: std::env::var_os("CUDA_OXIDE_LLVM_LINK").map(PathBuf::from),
        llvm_link_disabled: std::env::var_os(reserved_oxide_symbols::LLVM_LINK_DISABLED_ENV)
            .is_some(),
    }
}

pub(crate) fn resolve_toolchain(options: &BackendOptions) -> Option<LlvmToolchain> {
    LlvmToolchain::resolve(&selection_options(options))
}

pub(crate) fn libdevice_ir_linking_available(options: &BackendOptions) -> bool {
    cuda_artifact_finalizer::libdevice_ir_linking_available(&selection_options(options))
}

pub(crate) fn resolve_sibling_tool(
    tool: &str,
    environment_variable: &str,
    llc_path: &str,
    llc_major: Option<u32>,
) -> Option<OptTool> {
    if std::env::var_os(reserved_oxide_symbols::LLVM_LINK_DISABLED_ENV).is_some() {
        return None;
    }
    let explicit = std::env::var_os(environment_variable).map(PathBuf::from);
    cuda_artifact_finalizer::resolve_sibling_tool(tool, explicit.as_deref(), llc_path, llc_major)
}
