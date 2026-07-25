# Array map unroll

This regression exercises a three-element `array::map` whose callback becomes
large only after device inlining. CUDA Oxide preserves bounded full-unroll
intent on the erased array-builder loop until its concrete extent is visible,
so the fixed-size input, iterator, and result remain in registers instead of
local memory.

Run the end-to-end GPU and PTX check with:

```bash
cargo oxide run array_map_unroll --arch=sm_86
```
