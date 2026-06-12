# cuda-oxide Fork Notes

## Impulse Rebase Recovery Note - 2026-06-12

This checkout carries the Impulse-focused fork changes on top of current
`upstream/main`.

- Upstream base used for the careful rebase:
  `cb318ad4e4e37f5e1913ed0a13478af990e857f7`
- New rebased branch:
  `rebase/spike-applied-v0.3`
- Rebased head after the audit:
  `a80b0cb1d768560c32f51a8f724150cab89e2eb4`
- Pre-rebase backup branch:
  `backup/rebase-spike-applied-v0.2-20260612`
- Pre-rebase head:
  `5f98494ab188a5c62e3d9a85094961a09d2b6d14`

If a later Impulse kernel exposes a suspicious miscompile, compare against the
backup branch before assuming the old fork code was dead:

```bash
git log --oneline upstream/main..backup/rebase-spike-applied-v0.2-20260612
git show <old-commit>
git diff rebase/spike-applied-v0.3..backup/rebase-spike-applied-v0.2-20260612 -- <path>
```

### Retained In The New Rebase

The new branch keeps the fork changes that are still required by Impulse's
single-invocation Model A CUDA build:

- `cuda-core` parity: `DeviceCopy` primitive/container impls, derive macro, and
  `DeviceBuffer` async compatibility methods.
- `cuda-device` compatibility: warp shuffle value support, legacy warp free
  functions, `gpu_only`, `address_space`, and the `bf16x2` FMA intrinsic.
- `#[cuda_module]` host/runtime support: lazy function lookup and multi-artifact
  module lookup.
- Codegen ownership filtering via `CUDA_OXIDE_DEVICE_CODEGEN_CRATE`.
- MIR/import/lowering fixes still needed by Impulse: output directory creation,
  inline preservation, fast float intrinsics, export/call-name alignment, array
  constants, nested index assignment, `copy_nonoverlapping`, volatile loads,
  pointer-distance intrinsics, and closure capture type lowering.

Two extra fixes were discovered while verifying this rebase:

- `6c4fdee`: closure capture tuples are the trailing generic argument in current
  stable-MIR, not a fixed slot.
- `f3f2072`: normalize unresolved associated-type projections through `TyCtxt`.
- `a80b0cb`: when walking through a slice data pointer, index by the slice
  element type even if that element is itself an array. Without this, Impulse's
  `vec_dot_generic_gpu` read `[[f32; 64]; 3]` as adjacent `f32`s inside row 0;
  the smoke test produced host `679.936` vs GPU `2248.8325`. The fix restored
  host `679.936` vs GPU about `679.9356`.

### Deliberately Dropped As Obsolete

These old fork areas were not carried forward because upstream/main or Impulse's
current build model made them unnecessary:

- Two-world / embedded wrapper / root-manifest / synthetic root-matrix tooling.
- Old generic-kernel naming and forced generic launch-root discovery. Upstream
  now has type-id generic kernel naming and examples covering generic/cross-crate
  kernels.
- Fork artifact-retention and root-manifest `.oxart` injection. Upstream now
  retains embedded artifacts through anchor symbols.
- Legacy `launch!`, `Dim3`, and `CudaModule::get_function` API surface.
- Old fat-pointer/slice/repr-enum changes that upstream already replaced with
  equivalent or better support.
- Historical host atomic fallbacks and broad workaround patches that were not
  required by the verified Impulse smoke path.

### Verification Used For The Rebase

From `/workspace/cuda-oxide-fork`:

```bash
cargo fmt --check
cargo check --workspace
cargo build --release --manifest-path crates/rustc-codegen-cuda/Cargo.toml
cargo test -p cuda-core device_copy_derive
cargo test -p cuda-core device_buffer -- --nocapture
```

From `/workspace/Impulse2`, against the rebuilt backend:

```bash
bash scripts/model-a host-check
bash scripts/model-a test \
  --test vec_dot_generic_gpu \
  --backend /workspace/cuda-oxide-fork/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so \
  --device-codegen-crate vec_dot_generic_gpu \
  --cargo-target-dir /tmp/impulse-vecdot-target-final \
  -- --nocapture
```
