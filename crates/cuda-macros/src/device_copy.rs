// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `#[derive(DeviceCopy)]` proc-macro.
//!
//! Emits the `unsafe impl DeviceCopy` plus a hidden field-type-check function
//! that fails to compile if any field is not itself `DeviceCopy`.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{
    Data, DataEnum, DataStruct, DataUnion, DeriveInput, Field, Fields, Generics, parse_quote,
};

pub fn impl_device_copy(input: &DeriveInput, import: TokenStream) -> TokenStream {
    let input_type = &input.ident;

    // Generate the code to type-check all fields of the derived type. We can't perform
    // type checking at expansion-time, so instead we generate a dummy nested function with a
    // type-bound on DeviceCopy and call it with every field type.
    // This will fail to compile if any of the nested types doesn't implement DeviceCopy.
    let (check_types_code, zero_validity_code) = match input.data {
        Data::Struct(ref data_struct) => (type_check_struct(data_struct), quote! {}),
        Data::Union(ref data_union) => (type_check_union(data_union), quote! {}),
        Data::Enum(ref data_enum) => match type_check_enum(input, data_enum) {
            Ok(check_types_code) => {
                let enum_ty = &input.ident;
                (
                    check_types_code,
                    quote! {
                        const _: () = {
                            let _ = unsafe { ::core::mem::zeroed::<#enum_ty>() };
                        };
                    },
                )
            }
            Err(err) => return err.to_compile_error(),
        },
    };

    // We need a function for the type-checking code to live in, so generate a complicated and
    // hopefully-unique name for that. The type identifier is used verbatim (not lowercased) so
    // distinct types differing only in case (e.g. `Foo` and `foo`) get distinct helper names
    // instead of colliding; the `non_snake_case` allow covers the casing.
    let type_test_func_name = format!("__verify_{input_type}_can_implement_devicecopy");
    let type_test_func_ident = Ident::new(&type_test_func_name, Span::call_site());

    // If the struct/enum/union is generic, we need to add the DeviceCopy bound to the generics
    // when implementing DeviceCopy.
    let generics = add_bound_to_generics(&input.generics, import.clone());
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    // Finally, generate the unsafe impl and the type-checking function.
    let generated_code = quote! {
        #zero_validity_code

        unsafe impl #impl_generics #import for #input_type #type_generics #where_clause {}

        #[doc(hidden)]
        #[allow(non_snake_case, dead_code, unused_variables)]
        fn #type_test_func_ident #impl_generics(value: &#input_type #type_generics) #where_clause {
            fn assert_impl<T: #import>() {}
            #check_types_code
        }
    };

    generated_code
}

fn add_bound_to_generics(generics: &Generics, import: TokenStream) -> Generics {
    let mut new_generics = generics.clone();
    let type_params = new_generics
        .type_params()
        .map(|param| param.ident.clone())
        .collect::<Vec<_>>();
    let where_clause = new_generics.make_where_clause();
    for type_param in type_params {
        where_clause
            .predicates
            .push(parse_quote!(#type_param: #import));
    }

    new_generics
}

#[cfg(test)]
mod tests {
    use super::add_bound_to_generics;
    use quote::{ToTokens, quote};
    use syn::{DeriveInput, parse_quote};

    #[test]
    fn device_copy_bounds_share_the_where_clause() {
        let input: DeriveInput = parse_quote!(
            struct Example<T: Copy, U>(T, U)
            where
                U: Send;
        );
        let bounded = add_bound_to_generics(&input.generics, quote!(::cuda_core::DeviceCopy));
        let params = bounded.params.to_token_stream().to_string();
        let where_clause = bounded
            .where_clause
            .expect("DeviceCopy bounds require a where clause")
            .to_token_stream()
            .to_string();

        assert_eq!(params, "T : Copy , U");
        assert_eq!(
            where_clause,
            "where U : Send , T : :: cuda_core :: DeviceCopy , U : :: cuda_core :: DeviceCopy"
        );
    }
}

fn type_check_struct(s: &DataStruct) -> TokenStream {
    let checks = match s.fields {
        Fields::Named(ref named_fields) => {
            let fields: Vec<&Field> = named_fields.named.iter().collect();
            check_fields(&fields)
        }
        Fields::Unnamed(ref unnamed_fields) => {
            let fields: Vec<&Field> = unnamed_fields.unnamed.iter().collect();
            check_fields(&fields)
        }
        Fields::Unit => vec![],
    };
    quote!(
        #(#checks)*
    )
}

fn type_check_enum(input: &DeriveInput, s: &DataEnum) -> syn::Result<TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "`#[derive(DeviceCopy)]` supports enums only when the enum is not generic, because \
             rustc must be able to const-evaluate `mem::zeroed::<Enum>()` for the concrete enum \
             type and reject enums whose all-zero bit pattern is invalid.",
        ));
    }

    let mut checks = vec![];
    for variant in &s.variants {
        match variant.fields {
            Fields::Named(ref named_fields) => {
                let fields: Vec<&Field> = named_fields.named.iter().collect();
                checks.extend(check_fields(&fields));
            }
            Fields::Unnamed(ref unnamed_fields) => {
                let fields: Vec<&Field> = unnamed_fields.unnamed.iter().collect();
                checks.extend(check_fields(&fields));
            }
            Fields::Unit => {}
        }
    }
    Ok(quote!(
        #(#checks)*
    ))
}

fn type_check_union(s: &DataUnion) -> TokenStream {
    let fields: Vec<&Field> = s.fields.named.iter().collect();
    let checks = check_fields(&fields);
    quote!(
        #(#checks)*
    )
}

fn check_fields(fields: &[&Field]) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let field_type = &field.ty;
            quote! {assert_impl::<#field_type>();}
        })
        .collect()
}
