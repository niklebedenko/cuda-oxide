// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Minimal reproduction for a rustc-codegen-cuda miscompile: a GENERIC
 * kernel that (transitively) branches on an associated `const bool` keeps
 * the dead arm through device codegen, so a callee that only the dead arm
 * names ends up referenced but never emitted.
 *
 *   trait FaceShape { const INTERPOLATE: bool; fn weights() -> u32; }
 *
 *   fn transfer<S: FaceShape>(x: u32) -> u32 {
 *       if S::INTERPOLATE { x * S::weights() } else { x }
 *   }
 *
 * For a shape with `INTERPOLATE = false`, rustc's monomorphization
 * collector evaluates the constant discriminant and never instantiates
 * `<S as FaceShape>::weights` (documented rustc behavior: dead arms of
 * const-discriminant SwitchInts don't produce mono items, see
 * rustc_monomorphize::collector). If the cuda-oxide pipeline translates
 * the monomorphized MIR as written — both arms — the device module calls
 * a symbol nothing emits.
 *
 * Symptom (current cuda-oxide spike/applied):
 *   error: Verification failed for 'llvm module':
 *     Symbol _R..7weights.. not found
 *
 * The kernel being GENERIC is essential. In a monomorphic kernel the MIR
 * inliner + GVN + SimplifyCfg fold `S::INTERPOLATE` and delete the dead
 * arm in the kernel's own optimized MIR, so nothing downstream ever sees
 * it (the `mono_transfer` kernel below is the control for exactly that).
 * In a generic kernel the flag is unevaluatable in the generic MIR; the
 * arm is only dead after substitution, which is precisely the case
 * rustc's collector prunes and a MIR-faithful importer must prune too.
 *
 * Flavors, from minimal to structurally faithful to the original hit:
 *   - mono_transfer            control; branch folds at MIR-opt time
 *   - generic_explicit::<S>    dead hook is an explicit impl method
 *   - generic_default::<S>     dead hook is an un-overridden trait default
 *   - generic_sumfact::<S>     full shape/backend skeleton: blanket
 *                              `FaceTransferOps` impl branches, dead arm
 *                              calls an intermediate generic fn, which
 *                              calls a hook generic over T returning a
 *                              GAT-projected `&'static` table
 *
 * Expected after fix: every kernel launches and
 *   mono      = [21, 5, 21, 5]
 *   explicit  interp: x*7, plain: x
 *   default   interp: x*7, plain: x
 *   sumfact   tet: x*2.0, cube: x
 *
 * First hit in the wild: Impulse `impulse_detail_nvgpu` face-transfer
 * dedup (2026-07-05) — a `const INTERPOLATE: bool` branch in a blanket
 * `ShapeFaceTransferOps` impl left cube shapes' `face_val_weights`
 * dangling; adding an explicit (non-default) impl of the hook did not
 * help, because the instance is unreferenced by any live mono item either
 * way.
 *
 * Build:
 *   cargo oxide run const_bool_dead_branch
 */

use core::marker::PhantomData;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const N: usize = 4;

/// Shape-tagged scalar. The kernels' generic shape param appears only in
/// buffer element types (as in Impulse's `Vol<N, S, Backend>` fields), so
/// the `#[kernel]` wrapper can infer it from the arguments.
#[repr(transparent)]
pub struct Node<T, S: Shape> {
    pub v: T,
    _shape: PhantomData<S>,
}

// Manual impls: deriving would demand `S: Copy`/`S: Debug`/`S: PartialEq`
// on the shape tags.
impl<T: Copy, S: Shape> Copy for Node<T, S> {}

impl<T: Copy, S: Shape> Clone for Node<T, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: core::fmt::Debug, S: Shape> core::fmt::Debug for Node<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.v.fmt(f)
    }
}

impl<T: PartialEq, S: Shape> PartialEq for Node<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl<T, S: Shape> Node<T, S> {
    #[inline(always)]
    pub fn new(v: T) -> Self {
        Self {
            v,
            _shape: PhantomData,
        }
    }
}

// SAFETY: repr(transparent) over T; PhantomData is zero-sized.
unsafe impl<T: DeviceCopy, S: Shape> DeviceCopy for Node<T, S> {}

pub trait Shape: Sized + 'static {}

// ---------------------------------------------------------------------------
// Minimal flavor: flag + hook on one trait, hook called directly in the
// branch.
// ---------------------------------------------------------------------------

/// Explicit-impl flavor: every impl provides `weights`, but for `Plain` it
/// is dead code behind `INTERPOLATE = false`.
pub trait FaceShape: Shape {
    const INTERPOLATE: bool;
    fn weights() -> u32;
}

pub struct Interp;
pub struct Plain;
impl Shape for Interp {}
impl Shape for Plain {}

impl FaceShape for Interp {
    const INTERPOLATE: bool = true;
    #[inline(always)]
    fn weights() -> u32 {
        7
    }
}

impl FaceShape for Plain {
    const INTERPOLATE: bool = false;
    // Never called: the only user is the `if S::INTERPOLATE` arm that is
    // statically false for this type, so rustc never instantiates this
    // method.
    fn weights() -> u32 {
        unreachable!()
    }
}

/// Default-method flavor: FLAG=false types keep the (never instantiated)
/// default body.
pub trait FaceShapeD: Shape {
    const INTERPOLATE: bool;
    fn weights() -> u32 {
        unreachable!()
    }
}

pub struct InterpD;
pub struct PlainD;
impl Shape for InterpD {}
impl Shape for PlainD {}

impl FaceShapeD for InterpD {
    const INTERPOLATE: bool = true;
    #[inline(always)]
    fn weights() -> u32 {
        7
    }
}

impl FaceShapeD for PlainD {
    const INTERPOLATE: bool = false;
    // Keeps the default `weights` body.
}

// ---------------------------------------------------------------------------
// Structurally faithful flavor: shape/backend skeleton with a GAT weights
// type, a blanket face-transfer impl branching on the flag, and the hook
// called through an intermediate generic fn with its own type param.
// ---------------------------------------------------------------------------

pub struct Tet;
pub struct Cube;
impl Shape for Tet {}
impl Shape for Cube {}

pub struct Backend;

/// Per-shape storage formats on this backend (GAT, as in Impulse's
/// `ShapeLayout::SumfactWeights<T>`).
pub trait Layout<S: Shape> {
    type Weights<T: WeightNum>: Copy + 'static;
}

impl Layout<Tet> for Backend {
    type Weights<T: WeightNum> = &'static [[T; 4]; 4];
}
impl Layout<Cube> for Backend {
    type Weights<T: WeightNum> = &'static [[T; 4]; 4];
}

/// Weight scalar with a static table (as in Impulse's `NodeFloat` Gauss
/// tables).
pub trait WeightNum: Copy + 'static {
    fn table() -> &'static [[Self; 4]; 4];
    fn mul(self, w: Self) -> Self;
}

impl WeightNum for f32 {
    #[inline(always)]
    fn table() -> &'static [[f32; 4]; 4] {
        &[[2.0; 4]; 4]
    }
    #[inline(always)]
    fn mul(self, w: f32) -> f32 {
        self * w
    }
}

/// Per-shape sumfact kernel (analogue of `ShapeSumfactOps`).
pub trait SumfactOps<S: Shape>: Layout<S> {
    fn sumfact<T: WeightNum>(x: T, weights: Self::Weights<T>) -> T;
}

impl SumfactOps<Tet> for Backend {
    #[inline(always)]
    fn sumfact<T: WeightNum>(x: T, weights: &'static [[T; 4]; 4]) -> T {
        x.mul(weights[0][0])
    }
}
impl SumfactOps<Cube> for Backend {
    #[inline(always)]
    fn sumfact<T: WeightNum>(x: T, weights: &'static [[T; 4]; 4]) -> T {
        x.mul(weights[0][0])
    }
}

/// The marker whose blanket impl carries the const-bool branch. Cube keeps
/// the default (never instantiated) `weights` body — this mirrors the
/// original Impulse hit, where the missing symbol was this default method.
pub trait PaddedShape: Shape
where
    Backend: Layout<Self> + SumfactOps<Self>,
{
    const INTERPOLATE: bool;
    /// Weight table in this shape's storage format. Shapes that never
    /// interpolate keep this default body.
    fn weights<T: WeightNum>() -> <Backend as Layout<Self>>::Weights<T> {
        unreachable!()
    }
}

impl PaddedShape for Tet {
    const INTERPOLATE: bool = true;
    #[inline(always)]
    fn weights<T: WeightNum>() -> &'static [[T; 4]; 4] {
        T::table()
    }
}

impl PaddedShape for Cube {
    const INTERPOLATE: bool = false;
    // Keeps the default `weights` body.
}

/// Intermediate generic fn between the branch and the hook (analogue of
/// `vol_componentwise_sumfact`): the dead arm's direct callee is this fn,
/// and the dangling symbol is one level deeper.
#[inline(always)]
fn componentwise_sumfact<T: WeightNum, S: PaddedShape>(x: T) -> T
where
    Backend: Layout<S> + SumfactOps<S>,
{
    let weights = S::weights::<T>();
    <Backend as SumfactOps<S>>::sumfact(x, weights)
}

/// The deduplicated ops trait whose blanket impl branches on the flag
/// (analogue of `ShapeFaceTransferOps`).
pub trait FaceTransferOps<S: Shape> {
    fn face2vol<T: WeightNum>(x: T) -> T;
}

impl<S: PaddedShape> FaceTransferOps<S> for Backend
where
    Backend: Layout<S> + SumfactOps<S>,
{
    #[inline(always)]
    fn face2vol<T: WeightNum>(x: T) -> T {
        if S::INTERPOLATE {
            componentwise_sumfact::<T, S>(x)
        } else {
            x
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn transfer_explicit<S: FaceShape>(x: u32) -> u32 {
        if S::INTERPOLATE { x * S::weights() } else { x }
    }

    #[inline(always)]
    fn transfer_default<S: FaceShapeD>(x: u32) -> u32 {
        if S::INTERPOLATE { x * S::weights() } else { x }
    }

    /// Control: monomorphic kernel. rustc's MIR pipeline folds the flags
    /// and deletes the dead arms in this kernel's own optimized MIR, so it
    /// builds and runs even while the generic kernels below miscompile.
    #[kernel]
    pub fn mono_transfer(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        let v = match i {
            0 => transfer_explicit::<Interp>(3),
            1 => transfer_explicit::<Plain>(5),
            2 => transfer_default::<InterpD>(3),
            _ => transfer_default::<PlainD>(5),
        };
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// Minimal generic flavor, explicit-impl hook.
    #[kernel]
    pub fn generic_explicit<S: FaceShape>(
        input: &[Node<u32, S>],
        mut out: DisjointSlice<Node<u32, S>>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = Node::new(transfer_explicit::<S>(input[i].v));
        }
    }

    /// Minimal generic flavor, default-method hook.
    #[kernel]
    pub fn generic_default<S: FaceShapeD>(
        input: &[Node<u32, S>],
        mut out: DisjointSlice<Node<u32, S>>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = Node::new(transfer_default::<S>(input[i].v));
        }
    }

    /// Structurally faithful flavor: blanket-impl branch, intermediate fn,
    /// GAT-returning hook generic over T.
    #[kernel]
    pub fn generic_sumfact<S: PaddedShape>(
        input: &[Node<f32, S>],
        mut out: DisjointSlice<Node<f32, S>>,
    ) where
        Backend: Layout<S> + SumfactOps<S>,
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = Node::new(<Backend as FaceTransferOps<S>>::face2vol::<f32>(input[i].v));
        }
    }
}

/// Launch one kernel over `input`, compare against `expected`.
fn check<I, O>(
    name: &str,
    stream: &Arc<CudaStream>,
    input: &[I],
    expected: &[O],
    launch: impl FnOnce(LaunchConfig, &DeviceBuffer<I>, &mut DeviceBuffer<O>),
) -> bool
where
    I: DeviceCopy,
    O: DeviceCopy + PartialEq + std::fmt::Debug,
{
    let dev_in = DeviceBuffer::from_host(stream, input).unwrap();
    let mut dev_out = DeviceBuffer::<O>::zeroed(stream, expected.len()).unwrap();
    launch(
        LaunchConfig::for_num_elems(expected.len() as u32),
        &dev_in,
        &mut dev_out,
    );
    let host_out = dev_out.to_host_vec(stream).unwrap();
    let pass = host_out == expected;
    let verdict = if pass { "PASS" } else { "FAIL" };
    println!("  {name:<26}  {verdict}");
    pass
}

/// Tag a plain scalar vec with a shape for the typed kernel buffers.
fn nodes<T: Copy, S: Shape>(xs: &[T]) -> Vec<Node<T, S>> {
    xs.iter().map(|&x| Node::new(x)).collect()
}

fn main() {
    println!("=== const-bool dead-branch symbol resolution repro ===\n");
    println!("Generic `if S::CONST_FLAG` branches must not leave dead-arm");
    println!("callees dangling in the device module.\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");

    let useq: Vec<u32> = vec![1, 2, 3, 4];
    let x7: Vec<u32> = useq.iter().map(|x| x * 7).collect();
    let fseq: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let x2: Vec<f32> = fseq.iter().map(|x| x * 2.0).collect();
    let mut all_pass = true;

    // Control: monomorphic kernel, branch folds at MIR-opt time.
    {
        let mut dev_out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        // SAFETY: the launch covers exactly N elements and the kernel writes
        // only within the provided output buffer.
        unsafe {
            module
                .mono_transfer(&stream, LaunchConfig::for_num_elems(N as u32), &mut dev_out)
                .expect("launch");
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        let pass = host_out == [21, 5, 21, 5];
        println!(
            "  {:<26}  {}",
            "mono_transfer",
            if pass { "PASS" } else { "FAIL" }
        );
        all_pass &= pass;
    }

    // Live arm: FLAG=true shapes must keep working.
    all_pass &= check(
        "generic_explicit::<Interp>",
        &stream,
        &nodes::<_, Interp>(&useq),
        &nodes::<_, Interp>(&x7),
        |cfg, i, o| {
            // SAFETY: `check` launches over the input/output length and the
            // kernel guards accesses by N.
            unsafe {
                module
                    .generic_explicit::<Interp>(&stream, cfg, i, o)
                    .expect("launch")
            }
        },
    );
    all_pass &= check(
        "generic_default::<InterpD>",
        &stream,
        &nodes::<_, InterpD>(&useq),
        &nodes::<_, InterpD>(&x7),
        |cfg, i, o| {
            // SAFETY: `check` launches over the input/output length and the
            // kernel guards accesses by N.
            unsafe {
                module
                    .generic_default::<InterpD>(&stream, cfg, i, o)
                    .expect("launch")
            }
        },
    );
    all_pass &= check(
        "generic_sumfact::<Tet>",
        &stream,
        &nodes::<_, Tet>(&fseq),
        &nodes::<_, Tet>(&x2),
        |cfg, i, o| {
            // SAFETY: `check` launches over the input/output length and the
            // kernel guards accesses by N.
            unsafe {
                module
                    .generic_sumfact::<Tet>(&stream, cfg, i, o)
                    .expect("launch")
            }
        },
    );

    // Dead arm: FLAG=false shapes — these instantiations are the repro.
    all_pass &= check(
        "generic_explicit::<Plain>",
        &stream,
        &nodes::<_, Plain>(&useq),
        &nodes::<_, Plain>(&useq),
        |cfg, i, o| {
            // SAFETY: `check` launches over the input/output length and the
            // kernel guards accesses by N.
            unsafe {
                module
                    .generic_explicit::<Plain>(&stream, cfg, i, o)
                    .expect("launch")
            }
        },
    );
    all_pass &= check(
        "generic_default::<PlainD>",
        &stream,
        &nodes::<_, PlainD>(&useq),
        &nodes::<_, PlainD>(&useq),
        |cfg, i, o| {
            // SAFETY: `check` launches over the input/output length and the
            // kernel guards accesses by N.
            unsafe {
                module
                    .generic_default::<PlainD>(&stream, cfg, i, o)
                    .expect("launch")
            }
        },
    );
    all_pass &= check(
        "generic_sumfact::<Cube>",
        &stream,
        &nodes::<_, Cube>(&fseq),
        &nodes::<_, Cube>(&fseq),
        |cfg, i, o| {
            // SAFETY: `check` launches over the input/output length and the
            // kernel guards accesses by N.
            unsafe {
                module
                    .generic_sumfact::<Cube>(&stream, cfg, i, o)
                    .expect("launch")
            }
        },
    );

    println!();
    if all_pass {
        println!("RESULT: PASS - dead const-bool arms pruned, live hooks intact.");
        std::process::exit(0);
    } else {
        println!("RESULT: FAIL - wrong kernel output (see above).");
        std::process::exit(1);
    }
}
