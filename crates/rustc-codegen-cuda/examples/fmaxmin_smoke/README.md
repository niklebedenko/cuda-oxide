# fmaxmin_smoke

Smoke test for `f32::max` / `f32::min` (and the f64 forms) lowering to
LLVM intrinsics and then PTX.

Both methods lower to the `_nsz` flavor of the rustc maxNum / minNum
intrinsics in MIR:

| Public API | MIR intrinsic | LLVM intrinsic |
| ---------- | ------------- | ---------------- |
| `f32::max` | `core::intrinsics::maximum_number_nsz_f32` | `llvm.maxnum.f32` |
| `f64::max` | `core::intrinsics::maximum_number_nsz_f64` | `llvm.maxnum.f64` |
| `f32::min` | `core::intrinsics::minimum_number_nsz_f32` | `llvm.minnum.f32` |
| `f64::min` | `core::intrinsics::minimum_number_nsz_f64` | `llvm.minnum.f64` |

This example exists as a regression test for that lowering chain. Before
the corresponding entries existed in `dialect-mir::rust_intrinsics`,
`mir-importer`, and `mir-lower`, `f32::max` / `f32::min` fell out of the
pipeline as unresolved intrinsic calls.

Run it with:

```bash
cargo oxide run fmaxmin_smoke
```

## How code reaches the GPU

The kernels in this example use LLVM `maxnum` / `minnum` intrinsics, so
cuda-oxide emits ordinary PTX. [`cuda_host::ltoir::load_kernel_module`]
loads that PTX directly; it only falls back to libNVVM + nvJitLink for
modules that actually emit NVVM IR.

## What the smoke checks

For both `f32` and `f64`:

1. The finite case — `(1.5_f32).max(-2.5_f32) == 1.5_f32` and the matching
   `.min` — confirms the LLVM intrinsic lowering is reached and the result is
   bit-exact.
2. The NaN case — `f32::NAN.max(b) == b` and `f32::NAN.min(b) == b` —
   confirms the maxNum / minNum NaN-suppression rule is honored. The
   NaN payload is passed in as a kernel argument from the host so that
   the example does not double up with how cuda-oxide renders NaN
   constants in LLVM IR.
