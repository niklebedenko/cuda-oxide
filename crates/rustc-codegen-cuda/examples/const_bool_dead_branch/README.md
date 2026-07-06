# `const_bool_dead_branch` — const-dead branch arms must be pruned

In a **generic** device function, `if S::CONST_FLAG { S::hook() } else { x }`
was rejected (or miscompiled) even though the dead arm can never execute:
rustc's monomorphization collector walks only mono-reachable blocks
(`Body::mono_successors` evaluates const-discriminant `SwitchInt`s per
instance), so the dead arm's callees are **never instantiated**. The fork's
collector and MIR importer walked *all* blocks, demanding symbols rustc never
emits.

The kernel being generic is essential: in a monomorphic kernel rustc's MIR
pipeline (inline + GVN + SimplifyCfg) folds the constant and deletes the dead
arm before codegen ever sees it. In a generic kernel the flag is
unevaluatable in generic MIR — the arm only becomes dead after substitution,
exactly when the collector/importer must prune it. `mono_transfer` is the
monomorphic control; the three generic kernels are the repro.

Two manifestations of the same divergence:

- If the dead-arm callee's body is panic-only (a `default` trait hook, or an
  explicit `unreachable!()` impl), the collector rejected it with the
  issue-#76 "body is nothing but a panic" diagnostic.
- Otherwise collection was skipped silently but the importer still translated
  the dead call, leaving a dangling symbol:
  `Verification failed for 'llvm module': Symbol <hash> not found`.

## Run

```bash
cargo oxide run const_bool_dead_branch
```

## Expected output before the fix

```
error: `<Plain as FaceShape>::weights` is called from device code but its
body is nothing but a panic, which cannot be compiled for the GPU
note: called from device code here (reached from kernel `generic_explicit_TID_...`)
```

(build failure — the binary never runs)

## Expected output after the fix

```
  mono_transfer               PASS
  generic_explicit::<Interp>  PASS
  generic_default::<InterpD>  PASS
  generic_sumfact::<Tet>      PASS
  generic_explicit::<Plain>   PASS
  generic_default::<PlainD>   PASS
  generic_sumfact::<Cube>     PASS

RESULT: PASS - dead const-bool arms pruned, live hooks intact.
```

## Root cause and fix

rustc decides reachability *per mono instance*: `traversal::mono_reachable`
const-evaluates a `SwitchInt` discriminant like `<Plain as
FaceShape>::INTERPOLATE` under the instance's substitutions and follows only
the taken edge. Anything walking monomorphized MIR must apply the same rule
or it disagrees with what rustc actually instantiates.

The fix applies it in both fork components that walk bodies:

- **Collector** (`crates/rustc-codegen-cuda/src/collector.rs`): call-graph
  discovery and the panic-machinery check now skip blocks outside
  `traversal::mono_reachable_as_bitset(mir, tcx, instance)` — the exact
  helper rustc's own collector uses.
- **MIR importer** (`crates/mir-importer/src/translator/body.rs`):
  `prune_const_switch_targets` mirrors rustc's
  `Body::try_const_mono_switchint` on `rustc_public` MIR. The bridge's
  `BodyBuilder` has already monomorphized the body and evaluated every const
  operand, so a constant discriminant arrives as an `Allocation` readable
  with `read_uint()` — either directly on the terminator or via the block's
  last non-storage assignment to the switched place. The reachability BFS
  then follows only the taken edge; dead-arm blocks fall out of the
  reachable set and become `mir.unreachable` stubs (the existing mechanism
  for unwind cleanups).

The three generic kernels cover the trait shapes that hit this in practice:

| Kernel             | Dead-arm hook                                    | Flavor                     |
| :----------------- | :----------------------------------------------- | :------------------------- |
| `generic_explicit` | explicit impl with `unreachable!()` body         | minimal                    |
| `generic_default`  | default trait method left unoverridden           | minimal                    |
| `generic_sumfact`  | default hook returning a GAT-like assoc type     | faithful (CFD sum-factor.) |

Live arms (`Interp`/`InterpD`/`Tet`) verify pruning doesn't eat real code;
dead arms (`Plain`/`PlainD`/`Cube`) verify the panic-only / uninstantiated
hooks no longer poison compilation.

Origin: hit in Impulse (2026-07-05) when deduplicating shape-generic
face-transfer kernels behind a trait with a `const INTERPOLATE: bool` switch.
