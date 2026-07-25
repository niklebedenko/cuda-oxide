# Materialized fixed vector loops

This regression example exercises the small fixed-trip-count loops used by
element-local vector kernels. It loads two `2 x 3` vectors, forms a SAXPY
combination, and computes a dot product.

Run the native-cubin route on an Ampere GPU with:

```bash
cargo oxide run materialized_fixed_vector_loops \
  --materialize-cubin \
  --arch sm_86
```

The program checks both the GPU result and the exact PTX sidecar retained for
the materialized cubin. The fixed loops must optimize without per-thread local
arrays, local loads/stores, or trap instructions. Loading and launching the
module also verifies that the embedded cubin is valid.

The target can be changed to match another installed GPU. The example uses no
Ampere-specific instructions.
