# Aggregate closure inline

This regression exercises a kernel that borrows one large by-value argument
from a directly invoked closure. The closure is intentionally large enough to
remain a separate function under ordinary heuristic inlining.

CUDA Oxide must preserve the launch ABI while inlining that closure into its
kernel entry. Otherwise the borrowed aggregate needs a per-thread local-memory
copy solely to pass a pointer across the helper boundary.

Run the end-to-end GPU and PTX check with:

```bash
cargo oxide run aggregate_closure_inline --arch=sm_86
```
