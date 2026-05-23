/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[address_space(...)]` attribute macro.
//!
//! Faithful port of NVIDIA Rust-CUDA's `cuda_std_macros::address_space`. The
//! legacy `impulse_*` tree applies `#[cuda_std::address_space(shared)]` to
//! `static mut` shared-memory buffers; Port v2 (Impulse epic #198) remaps the
//! path to `cuda_device::address_space`, so the attribute must resolve
//! unchanged.
//!
//! Behaviour (verbatim from cuda_std): takes a single argument
//! (`global`/`shared`/`constant`/`local`), and on the `cuda` target appends a
//! `nvvm_internal::addrspace(N)` attribute to the `static`. On the host it does
//! nothing — exactly what lets `cuda_device` build on the host target.

use proc_macro::TokenStream;
use quote::ToTokens;
use syn::{Ident, parse_quote};

/// Notifies the codegen to put a `static`/`static mut` inside of a specific
/// memory address space. Takes a single argument which can be `global`,
/// `shared`, `constant`, or `local`. Does nothing on the CPU.
pub fn address_space_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut global = syn::parse_macro_input!(item as syn::ItemStatic);
    let input = syn::parse_macro_input!(attr as Ident);

    let addrspace_num = match input.to_string().as_str() {
        "global" => 1,
        // what did you do to address space 2 libnvvm??
        "shared" => 3,
        "constant" => 4,
        "local" => 5,
        addr => panic!("Invalid address space `{}`", addr),
    };

    let new_attr =
        parse_quote!(#[cfg_attr(target_os = "cuda", nvvm_internal::addrspace(#addrspace_num))]);
    global.attrs.push(new_attr);

    global.into_token_stream().into()
}
