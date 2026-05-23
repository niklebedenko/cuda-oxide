/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[gpu_only]` attribute macro.
//!
//! Faithful port of NVIDIA Rust-CUDA's `cuda_std_macros::gpu_only` (itself
//! derived from rust-gpu's `gpu_only`). The legacy `impulse_*` tree applies
//! `#[gpu_only]` to device-only helpers (e.g. the `warp_shuffle_N` shuffle
//! primitives). Port v2 (Impulse epic #198) remaps `cuda_std` paths to
//! `cuda_device`, so the attribute must resolve unchanged.
//!
//! Behaviour (verbatim from cuda_std): the macro splits the function into two
//! cfg-gated copies —
//!
//! * **non-`nvptx64` (host)**: the body is replaced with an `unimplemented!`
//!   stub. This is what lets `cuda_device` build on the host target despite
//!   the device-only bodies (e.g. the `__nvvm_warp_shuffle` extern call), which
//!   matches how the cuda-oxide fork compiles device code on the host and lets
//!   the codegen backend intercept the real call sites.
//! * **`nvptx64`**: the original body is kept verbatim.

use proc_macro::TokenStream;

/// Creates a cpu version of the function which panics and cfg-gates the
/// function for only nvptx/nvptx64.
pub fn gpu_only_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let syn::ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = syn::parse_macro_input!(item as syn::ItemFn);

    let mut cloned_attrs = attrs.clone();
    cloned_attrs.retain(|a| a.path().segments[0].ident != "nvvm_internal");

    let fn_name = sig.ident.clone();

    let sig_cpu = syn::Signature {
        abi: None,
        ..sig.clone()
    };

    let output = quote::quote! {
        #[cfg(not(target_arch="nvptx64"))]
        #[allow(unused_variables)]
        #(#cloned_attrs)* #vis #sig_cpu {
            unimplemented!(concat!("`", stringify!(#fn_name), "` can only be used on the GPU with the cuda-oxide codegen backend"))
        }

        #[cfg(target_arch="nvptx64")]
        #(#attrs)* #vis #sig {
            #block
        }
    };

    output.into()
}
