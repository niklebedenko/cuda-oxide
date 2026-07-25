# H(div)-shaped array-inline probe

This cross-crate example approximates the device-code shape that stresses
Impulse's tetrahedral H(div) operators:

- four executed kernels in one embedded NVVM IR module;
- aligned, nested volume and face aggregates passed and returned by value;
- nested `core::array::from_fn` closures;
- large helpers carrying explicit `#[inline(always)]` intent; and
- enough scalar work to produce roughly 4.3 MB of legacy NVVM IR.

Run the correctness check on a CUDA-capable machine with:

```console
cargo oxide run hdiv_array_inline_probe
```

This deliberately uses the ordinary auto-libdevice path: no
`--emit-nvvm-ir` flag is needed. It guards the backend's bounded promotion of
callbacks passed to concrete `core::array::from_fn` builders to
device-link-only mandatory inline intent. The core construction scaffolds
retain their ordinary inline hints because forcing those boundaries can
trigger invalid native stack-frame lowering in the CUDA 12.9 device linker.

Inspect a produced cubin with:

```console
cuobjdump --dump-resource-usage image.cubin
cuobjdump --dump-sass image.cubin |
  rg -c 'CALL|STL|LDL'
```

The compiler limits callback promotion to bodies no larger than 24 MIR blocks
/ 128 MIR statements, captures no larger than 256 bytes, output layouts no
larger than 32 KiB, and concrete array extents no larger than 128. Those are
compile-work budgets rather than source-language semantics.
