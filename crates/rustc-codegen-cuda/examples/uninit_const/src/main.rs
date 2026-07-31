// SPDX-License-Identifier: Apache-2.0
//! Regression test: a fully-uninitialized constant allocation
//! (`MaybeUninit::uninit()` of a non-ZST type, every byte uninit, no
//! provenance) must translate as `undef` instead of failing with
//! "Unsupported constant type in translate_constant".
//!
//! Under -O, rustc const-promotes the `MaybeUninit::uninit()` initializer
//! into an `Allocated` constant whose init mask is empty. Aggregate types
//! take the struct/union dispatch, which has no handler for a constant
//! with no defined bytes. Aligned wrapper structs over vector types (the
//! glam `Align16<Vec3>` pattern) hit this in real shader crates.
//!
//! Run: cargo oxide run uninit_const

use core::mem::MaybeUninit;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
#[repr(transparent)]
struct Scalar(f32);

#[derive(Clone, Copy)]
#[repr(usize)]
enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];
}

#[cuda_module]
mod kernels {
    use super::*;

    /// 16-byte aligned aggregate, mirroring glam-style `Align16<Vec3>`.
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub struct Align16([f32; 3]);

    #[kernel]
    pub fn write_through_uninit(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let t = tid.get() as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut slot: MaybeUninit<Align16> = MaybeUninit::uninit();
            slot.write(Align16([t, t + 1.0, t + 2.0]));
            // SAFETY: written just above.
            let v = unsafe { slot.assume_init() };
            *out_elem = v.0[0] + v.0[1] + v.0[2];
        }
    }

    /// Exercises the dynamically dead uninhabited residual arms retained by
    /// `core::array::from_fn` through its infallible `try_from_fn` machinery.
    #[kernel]
    pub fn from_fn_scalar(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let base = tid.get() as f32;
        let values: [Scalar; 4] = core::array::from_fn(|lane| Scalar(base + lane as f32));
        if let Some(out_elem) = out.get_mut(tid) {
            *out_elem = values[0].0 + values[1].0 + values[2].0 + values[3].0;
        }
    }

    /// Exercises bare array constants whose elements are fieldless enums.
    #[kernel]
    pub fn enum_array(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let raw_index = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            *out_elem = Axis::ALL[raw_index % Axis::ALL.len()] as usize as f32;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/uninit_const.ptx");
    let module = ctx.load_module_from_file(ptx_path).expect("load PTX");
    let module = kernels::from_module(module).expect("typed module");
    let stream = ctx.default_stream();
    const N: usize = 32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: 32-thread 1D block matches the 32-element output allocation.
    unsafe { module.write_through_uninit(stream.as_ref(), cfg, &mut out) }.expect("launch");
    let got = out.to_host_vec(&stream).unwrap();
    let mut failures = 0;
    for (tid, &v) in got.iter().enumerate() {
        let want = 3.0 * tid as f32 + 3.0;
        if (v - want).abs() > 1e-6 {
            println!("FAIL tid={tid}: got {v} want {want}");
            failures += 1;
        }
    }

    // SAFETY: the launch shape matches the output allocation.
    unsafe { module.from_fn_scalar(stream.as_ref(), cfg, &mut out) }
        .expect("launch from_fn_scalar");
    for (tid, &value) in out.to_host_vec(&stream).unwrap().iter().enumerate() {
        let expected = 4.0 * tid as f32 + 6.0;
        if value != expected {
            println!("FAIL from_fn tid={tid}: got {value} want {expected}");
            failures += 1;
        }
    }

    // SAFETY: the launch shape matches the output allocation.
    unsafe { module.enum_array(stream.as_ref(), cfg, &mut out) }.expect("launch enum_array");
    for (tid, &value) in out.to_host_vec(&stream).unwrap().iter().enumerate() {
        let expected = (tid % Axis::ALL.len()) as f32;
        if value != expected {
            println!("FAIL enum_array tid={tid}: got {value} want {expected}");
            failures += 1;
        }
    }

    if failures == 0 {
        println!("uninit_const: PASS ({N} threads)");
    } else {
        std::process::exit(1);
    }
}
