/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-crate embedded bundle regression test (issue #222).
//!
//! This example exercises the code path that issue #222 fixed: loading a
//! generic kernel from a library crate via the embedded artifact bundle API
//! rather than from a PTX file on disk.
//!
//! When `kernel_lib::kernels::load(&ctx)` is called, the macro-generated
//! `load` function loads every embedded CUDA artifact because the module
//! contains generic kernels. Its launch handle resolves specializations across
//! those modules regardless of which crate owns their monomorphized entry
//! points. The example deliberately owns `scale::<f32>` in the library as an
//! archive-retention regression and instantiates its other launch types
//! downstream.
//!
//! Run: cargo oxide run cross_crate_embedded
//! No-CUDA linkage check: run the built binary with `--verify-bundles`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use kernel_lib::kernels;

fn main() {
    assert_eq!(kernel_lib::host_probe::linked_value(), 0x0cda_0a1d_e222);
    // Force the generic specialization owned by kernel-lib.
    let _ = kernel_lib::scale_f32_ptx_name();
    if std::env::args().any(|arg| arg == "--verify-bundles") {
        // Verify the selected owner's archive-carried artifact before touching
        // the CUDA driver. This mode is built with an explicit owner filter.
        let bundles = cuda_host::embedded::artifact_bundles_from_current_exe()
            .expect("read embedded artifact bundles");
        assert_eq!(
            bundles.len(),
            1,
            "owner-filtered final ELF must contain exactly one artifact"
        );
        assert_eq!(bundles[0].name, "kernel-lib");
        println!("SUCCESS: generic-only library artifact survived archive linking");
        return;
    }

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Embedded loading path: this is what issue #222 was about.
    let module = kernels::load(&ctx).expect("load embedded module");

    const N: usize = 256;
    let cfg = LaunchConfig::for_num_elems(N as u32);
    let mut errors = 0usize;

    // scale::<f32>
    {
        let factor: f32 = 2.5;
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let in_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.scale::<f32>(&stream, cfg, factor, &in_dev, &mut out_dev) }
            .expect("scale::<f32> launch");
        let out = out_dev.to_host_vec(&stream).unwrap();
        for (i, (&got, &x)) in out.iter().zip(input.iter()).enumerate() {
            if (got - x * factor).abs() > 1e-5 {
                if errors < 5 {
                    eprintln!(
                        "  FAIL scale::<f32>[{}]: got {} want {}",
                        i,
                        got,
                        x * factor
                    );
                }
                errors += 1;
            }
        }
    }

    // scale::<i32>
    {
        let factor: i32 = 3;
        let input: Vec<i32> = (0..N as i32).collect();
        let in_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.scale::<i32>(&stream, cfg, factor, &in_dev, &mut out_dev) }
            .expect("scale::<i32> launch");
        let out = out_dev.to_host_vec(&stream).unwrap();
        for (i, (&got, &x)) in out.iter().zip(input.iter()).enumerate() {
            if got != x * factor {
                if errors < 5 {
                    eprintln!(
                        "  FAIL scale::<i32>[{}]: got {} want {}",
                        i,
                        got,
                        x * factor
                    );
                }
                errors += 1;
            }
        }
    }

    if errors == 0 {
        println!("SUCCESS: cross-crate generic kernels load correctly via embedded bundles");
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }
}
