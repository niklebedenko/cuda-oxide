# H(div)-shaped array-inline probe

This cross-crate example approximates the device-code shape that stresses
Impulse's tetrahedral H(div) operators:

- five executed kernels in one embedded NVVM IR module;
- aligned, nested volume and face aggregates passed and returned by value;
- nested `core::array::from_fn` closures;
- an f64 pressure-face path large enough to retain a concrete array-builder
  call after ordinary libNVVM optimization;
- large helpers carrying explicit `#[inline(always)]` intent; and
- enough scalar work to produce roughly 11 MB of legacy NVVM IR.

Run the correctness check on a CUDA-capable machine with:

```console
cargo oxide run hdiv_array_inline_probe
```

The ordinary command checks runtime correctness. Exercise the NVVM-IR
attributes and device linker explicitly with:

```console
cargo oxide run hdiv_array_inline_probe --emit-nvvm-ir --arch=sm_86
```

The probe guards bounded device-link-only promotion of callbacks and erased
array helpers. Erased helpers also carry bounded deferred-unroll intent.
Concrete `core::array::from_fn` roots retain ordinary inline hints and carry a
deferred-inline marker in NVVM IR. PTX finalization compiles the ordinary form
first, promotes only marked roots which survive as calls, and retains the
promotion only when its PTX local-frame score improves. Direct PTX compilation
does not carry the NVVM-only marker. `emit-ltoir` stops before PTX finalization
and therefore does not apply the deferred promotion.

Inspect a produced cubin with:

```console
cuobjdump --dump-resource-usage image.cubin
cuobjdump --dump-sass image.cubin |
  rg -c 'CALL|STL|LDL'
```

The compiler normally limits callback promotion to 24 MIR blocks / 128 MIR
statements. A concrete one-to-three-element builder may promote a callback up
to 96 blocks / 1024 statements when its fully unrolled block and statement
totals remain bounded. Captures remain limited to 256 bytes, output layouts to
32 KiB, and concrete array extents to 128. Those are compile-work budgets
rather than source-language semantics.
