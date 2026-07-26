/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for generic fixed-array `Index<usize>` lowering.
//!
//! An associated storage type hides `[T; N]` from the generic helper's MIR.
//! If core's `Index::index` call survives until LLVM inlining, the dynamic GEP
//! appears too late for CUDA Oxide's small-array scalar selection and a
//! borrowed 48-byte launch aggregate spills to per-thread local memory.
//!
//! Run on the project validation GPU:
//!
//! `cargo oxide run fixed_array_index --arch sm_86`

use core::ops::Index;
use cuda_core::{CudaContext, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::process::Command;
use std::sync::Arc;

const TRAP_CASE_ENV: &str = "CUDA_OXIDE_FIXED_ARRAY_INDEX_TRAP_CASE";
const SENTINEL: f32 = -91_337.0;

#[cuda_module]
mod kernels {
    use super::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct RawField {
        pub ptr: *const f32,
    }

    pub trait HistoryScheme: Clone + Copy {
        type RawStorage<T: Clone + Copy>: Clone + Copy + Index<usize, Output = T>;
    }

    #[derive(Clone, Copy)]
    pub struct FourLevels;

    impl HistoryScheme for FourLevels {
        type RawStorage<T: Clone + Copy> = [T; 4];
    }

    #[derive(Clone, Copy)]
    pub struct ZeroLevels;

    impl HistoryScheme for ZeroLevels {
        type RawStorage<T: Clone + Copy> = [T; 0];
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct GenericHistory<T: Clone + Copy, B: HistoryScheme> {
        pub inner: B::RawStorage<T>,
    }

    impl<T: Clone + Copy, B: HistoryScheme> Index<usize> for GenericHistory<T, B> {
        type Output = T;

        #[inline(always)]
        fn index(&self, index: usize) -> &Self::Output {
            &self.inner[index]
        }
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct GenericNestedHistory<B: HistoryScheme> {
        pub volume: RawField,
        pub ghost: RawField,
        pub history: GenericHistory<RawField, B>,
    }

    impl<B: HistoryScheme> GenericNestedHistory<B> {
        #[inline(always)]
        fn at_time(&self, index: usize) -> RawField {
            self.history[index]
        }
    }

    #[kernel]
    pub unsafe fn generic_borrowed_index<B: HistoryScheme>(
        field: GenericNestedHistory<B>,
        history_index: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d();
        if let Some(slot) = out.get_mut(tid) {
            let selected = field.at_time(history_index as usize);
            *slot = unsafe { *selected.ptr };
        }
    }
}

fn ptx_kernel_bodies<'a>(ptx: &'a str, prefix: &str) -> Vec<&'a str> {
    let marker = format!(".entry {prefix}");
    let mut bodies = Vec::new();
    let mut search_start = 0;
    while let Some(relative_entry) = ptx[search_start..].find(&marker) {
        let entry = search_start + relative_entry;
        let body_start = ptx[entry..]
            .find('{')
            .map(|offset| entry + offset)
            .expect("kernel entry has a body");
        let mut depth = 0_u32;
        let mut body_end = None;
        for (offset, byte) in ptx.as_bytes()[body_start..].iter().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = Some(body_start + offset);
                        break;
                    }
                }
                _ => {}
            }
        }
        let body_end = body_end.expect("kernel body is terminated");
        bodies.push(&ptx[body_start..=body_end]);
        search_start = body_end + 1;
    }
    bodies
}

fn assert_zero_ptx_local_memory(ptx_path: &str) {
    let ptx = std::fs::read_to_string(ptx_path).expect("read generated PTX");
    let bodies = ptx_kernel_bodies(&ptx, "generic_borrowed_index");
    assert_eq!(
        bodies.len(),
        2,
        "expected FourLevels and ZeroLevels monomorphizations"
    );
    for body in bodies {
        assert!(
            !body.contains(".local") && !body.contains("ld.local") && !body.contains("st.local"),
            "generic fixed-array indexing must not use PTX local memory:\n{body}"
        );
        assert!(
            body.contains("trap;"),
            "bounds-checked generic indexing must retain a trap path:\n{body}"
        );
    }
}

fn assert_zero_sm86_native_frame(ptx_path: &str) {
    let cubin_path = std::env::temp_dir().join(format!(
        "cuda_oxide_fixed_array_index_{}_sm86.cubin",
        std::process::id()
    ));
    let assembled = Command::new("ptxas")
        .args(["-arch=sm_86", "-v", ptx_path, "-o"])
        .arg(&cubin_path)
        .output()
        .expect("run ptxas");
    assert!(
        assembled.status.success(),
        "ptxas failed:\n{}",
        String::from_utf8_lossy(&assembled.stderr)
    );
    let ptxas_log = String::from_utf8_lossy(&assembled.stderr);
    let matching_sections: Vec<_> = ptxas_log
        .split("Function properties for ")
        .skip(1)
        .filter(|section| section.starts_with("generic_borrowed_index"))
        .collect();
    assert_eq!(
        matching_sections.len(),
        2,
        "expected native resource reports for both monomorphizations:\n{ptxas_log}"
    );
    for section in matching_sections {
        assert!(
            section.contains("0 bytes stack frame")
                && section.contains("0 bytes spill stores")
                && section.contains("0 bytes spill loads"),
            "generic fixed-array indexing must have a zero native frame:\n{section}"
        );
    }

    let disassembled = Command::new("nvdisasm")
        .arg(&cubin_path)
        .output()
        .expect("run nvdisasm");
    assert!(
        disassembled.status.success(),
        "nvdisasm failed:\n{}",
        String::from_utf8_lossy(&disassembled.stderr)
    );
    let sass = String::from_utf8_lossy(&disassembled.stdout);
    assert!(
        !sass.contains("LDL") && !sass.contains("STL"),
        "final sm_86 code must not contain local loads/stores:\n{sass}"
    );
    std::fs::remove_file(cubin_path).expect("remove temporary cubin");
}

fn load_module(context: &Arc<CudaContext>) -> kernels::LoadedModule {
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixed_array_index.ptx");
    let module = context
        .load_module_from_file(ptx_path)
        .expect("load generated PTX");
    kernels::from_module(module).expect("initialize typed module")
}

fn run_in_bounds() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let module = load_module(&context);
    let stream = context.default_stream();
    let values = DeviceBuffer::<f32>::from_host(&stream, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        .expect("history values");
    let base = values.cu_deviceptr() as *const f32;
    let field = kernels::GenericNestedHistory::<kernels::FourLevels> {
        volume: kernels::RawField { ptr: base },
        ghost: kernels::RawField {
            ptr: unsafe { base.add(1) },
        },
        history: kernels::GenericHistory {
            inner: [
                kernels::RawField {
                    ptr: unsafe { base.add(2) },
                },
                kernels::RawField {
                    ptr: unsafe { base.add(3) },
                },
                kernels::RawField {
                    ptr: unsafe { base.add(4) },
                },
                kernels::RawField {
                    ptr: unsafe { base.add(5) },
                },
            ],
        },
    };
    let mut out = DeviceBuffer::<f32>::from_host(&stream, &[SENTINEL]).expect("output");
    unsafe {
        module
            .generic_borrowed_index::<kernels::FourLevels>(
                &stream,
                LaunchConfig::for_num_elems(1),
                field,
                2,
                &mut out,
            )
            .expect("launch in-range index");
    }
    assert_eq!(out.to_host_vec(&stream).expect("read output"), [50.0]);
}

fn trap_is_bounds_check(error: DriverError) -> bool {
    error.0 != cuda_core::sys::cudaError_enum_CUDA_ERROR_ILLEGAL_ADDRESS
}

fn run_trap_case(case: &str) -> bool {
    let context = CudaContext::new(0).expect("create CUDA context");
    let module = load_module(&context);
    let stream = context.default_stream();
    let values = DeviceBuffer::<f32>::from_host(&stream, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        .expect("history values");
    let base = values.cu_deviceptr() as *const f32;
    let mut out = DeviceBuffer::<f32>::from_host(&stream, &[SENTINEL]).expect("output");

    let launch_result = match case {
        "out-of-bounds" => {
            let field = kernels::GenericNestedHistory::<kernels::FourLevels> {
                volume: kernels::RawField { ptr: base },
                ghost: kernels::RawField {
                    ptr: unsafe { base.add(1) },
                },
                history: kernels::GenericHistory {
                    inner: [
                        kernels::RawField {
                            ptr: unsafe { base.add(2) },
                        },
                        kernels::RawField {
                            ptr: unsafe { base.add(3) },
                        },
                        kernels::RawField {
                            ptr: unsafe { base.add(4) },
                        },
                        kernels::RawField {
                            ptr: unsafe { base.add(5) },
                        },
                    ],
                },
            };
            unsafe {
                module.generic_borrowed_index::<kernels::FourLevels>(
                    &stream,
                    LaunchConfig::for_num_elems(1),
                    field,
                    4,
                    &mut out,
                )
            }
        }
        "zero-length" => {
            let field = kernels::GenericNestedHistory::<kernels::ZeroLevels> {
                volume: kernels::RawField { ptr: base },
                ghost: kernels::RawField {
                    ptr: unsafe { base.add(1) },
                },
                history: kernels::GenericHistory { inner: [] },
            };
            unsafe {
                module.generic_borrowed_index::<kernels::ZeroLevels>(
                    &stream,
                    LaunchConfig::for_num_elems(1),
                    field,
                    0,
                    &mut out,
                )
            }
        }
        _ => panic!("unknown trap case {case}"),
    };

    let observed = launch_result.and_then(|_| out.to_host_vec(&stream).map(|_| ()));
    match observed {
        Err(error) if trap_is_bounds_check(error) => {
            println!("{case}: PASS (kernel trapped: {error})");
            true
        }
        Err(error) => {
            eprintln!("{case}: FAIL (memory fault escaped bounds check: {error})");
            false
        }
        Ok(()) => {
            eprintln!("{case}: FAIL (kernel completed without trapping)");
            false
        }
    }
}

fn run_trap_child(case: &str) {
    let status = Command::new(std::env::current_exe().expect("current executable"))
        .env(TRAP_CASE_ENV, case)
        .status()
        .expect("run isolated trap case");
    assert!(status.success(), "{case} child regression failed");
}

fn main() {
    if let Ok(case) = std::env::var(TRAP_CASE_ENV) {
        std::process::exit(if run_trap_case(&case) { 0 } else { 1 });
    }

    run_in_bounds();
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixed_array_index.ptx");
    assert_zero_ptx_local_memory(ptx_path);
    assert_zero_sm86_native_frame(ptx_path);
    run_trap_child("out-of-bounds");
    run_trap_child("zero-length");
    println!("fixed_array_index: PASS");
}
