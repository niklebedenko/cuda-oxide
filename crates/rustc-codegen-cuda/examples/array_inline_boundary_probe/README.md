# Array-inline boundary probe

This compiler fixture checks CUDA Oxide's device-link-only array callback
inline policy:

- the callback passed to `core::array::from_fn` is promoted;
- a closure returned as an array element is not promoted;
- concrete core array roots retain their ordinary inline hints;
- accepted erased array helpers receive device-link-only mandatory inline
  intent plus bounded deferred-unroll intent;
- an oversized callback capture is rejected;
- a function item is not mistaken for a closure;
- a callback shared by accepted and rejected builders is rejected globally;
  and
- a user function merely named `array::from_fn` is not itself promoted.

The smoke harness emits NVVM IR, verifies those attributes directly, and
checks that direct PTX IR is unchanged:

```console
scripts/smoketest.sh --only '^array_inline_boundary_probe$'
```
