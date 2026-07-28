# Array map unroll

This regression exercises three-element `array::map` callbacks at two code
sizes. The wide case mirrors an H(div) operator: it returns an aligned
twenty-scalar field and has a callback whose optimized MIR exceeds the ordinary
small-helper inline budget. CUDA Oxide uses the known trip count to bound
device-link inlining and preserves full-unroll intent on the erased
array-builder loop until its concrete extent is visible. Both fixed-size
inputs, iterators, and results must remain in registers instead of local
memory.

Run the end-to-end GPU and PTX check with:

```bash
cargo oxide run array_map_unroll --arch=sm_86
```
