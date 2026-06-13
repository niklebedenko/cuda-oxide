# Upstreaming checklist for cuda-oxide fork patches

This file is for preparing fork patches before opening PRs against
`NVlabs/cuda-oxide`.

The main lesson from recent upstreamed fixes is: do not send narrow spot-fixes
for the one shape Impulse happened to hit. Upstream reviewers should not need to
generalize the patch, find duplicated lowering paths with the same bug, or add
the obvious edge-case tests. Do that work before the PR.

## Principle

Make the fix general, robust, and owned by the right abstraction.

A good upstream patch answers:

- What compiler/runtime rule was wrong?
- Where is the single abstraction that should enforce that rule?
- Which sibling paths could express the same rule incorrectly?
- Which tests prove the rule, not just the original reproduction?

If the patch only answers "this one repro now passes", it is probably not ready
for upstream.

## Diligence checklist

Before opening a PR, work through this list.

### 1. Define the rule, not only the symptom

Write down the underlying rule in one or two sentences.

Weak:

```text
This fixes `&(*ptr).field[i]`.
```

Better:

```text
Address materialization must preserve every projection needed to reach the
addressed place. It must not silently stop at an aggregate prefix.
```

Weak:

```text
This fixes `#[repr(u32)]` enums.
```

Better:

```text
Enum lowering must use rustc layout facts for tag storage, discriminant values,
size, and alignment. Local variant-count heuristics are not authoritative.
```

The better statement tells you where else to look.

### 2. Find duplicated or parallel paths

Search for every code path that implements the same conceptual operation. Fix
the shared helper or create one if the logic is duplicated.

Examples of questions to ask:

- Are there separate importer and lowerer paths for the same concept?
- Are `Ref`, `AddressOf`, loads, stores, aggregate construction, and ABI
  lowering using separate mappings?
- Are constants and runtime values handled by different code paths?
- Are local-crate and cross-crate monomorphization paths reconstructing the same
  type information differently?
- Are host ABI and device ABI paths making independent layout decisions?

If two paths can drift, prefer a small refactor that gives them one source of
truth. Do not leave a sibling path obviously capable of the same bug unless the
PR explicitly explains why it is safe.

### 3. Prefer authoritative data over heuristics

Use rustc layout/type information when it exists. Avoid re-deriving facts from
surface syntax or approximate local rules.

Examples:

- Prefer rustc layout offsets, size, alignment, ABI, and tag representation over
  ad hoc struct/enum calculations.
- Prefer `ProjectionElem::ty` or equivalent rustc-provided type transitions over
  manually guessing the next place type.
- Prefer a central symbol/name builder over hardcoded string prefixes.

Heuristics are acceptable only when the authoritative information is unavailable
and the PR documents the limitation.

### 4. Make invalid intermediate states hard to express

Good upstream patches often improve the shape of the code, not just the failing
branch.

Look for opportunities to:

- introduce a small typed helper for a repeated lowering operation;
- carry required layout metadata in the IR type instead of recomputing it;
- return an explicit error instead of silently producing a prefix, fallback, or
  placeholder value;
- remove identity/default fallbacks that can hide miscompiles;
- make callers pass the information required for correctness rather than
  allowing them to omit it.

The goal is not abstraction for its own sake. The goal is to make the same bug
unlikely to reappear in a parallel path.

### 5. Test the rule, including edge cases

Every upstream compiler fix should include regression coverage that proves the
general rule.

Minimum bar:

- Add a small example or test that fails before the patch and passes after.
- Verify both sides explicitly: the regression must fail on `upstream/main`
  without the patch, and the same regression must pass on the prepared branch
  with the patch.
- Assert the computed result when possible; do not only check that compilation
  succeeds if the bug is a possible miscompile.
- Include at least one edge case beyond the exact Impulse-triggered shape.

Good tests vary the dimension that exposed the rule:

- layout: padding, alignment, reordered fields, tag width, niche-like shapes;
- places: field, deref, index, nested projection, cross-crate type;
- constants: scalar, array, nested array, address/provenance-carrying constant;
- ABI: kernel boundary, device function boundary, by-value and by-reference;
- control/data flow: direct use and use through a helper or closure if relevant.

Avoid giant application-derived repros as the only test. They are useful for
diagnosis but poor upstream regression tests. Reduce them to a focused example
that names the compiler feature being protected.

### 6. Keep examples repo-native

For examples under `crates/rustc-codegen-cuda/examples/`:

- include `Cargo.lock`;
- use repo conventions for naming and style;
- keep the example focused on one compiler feature;
- add a short README when the bug is subtle;
- avoid unnecessary external dependencies;
- ensure `cargo fmt` and clippy pass for the example.

If adding or changing an intentional `error*` example, update
`crates/rustc-codegen-cuda/STATUS.md` and `scripts/smoketest.sh` in the same
commit.

### 7. Keep PRs reviewable

Prefer one conceptual compiler rule per PR.

Split unrelated work even if Impulse discovered it during the same porting
session. A PR should be easy to review as:

```text
bug mechanism -> abstraction change -> regression tests
```

If a robust fix requires a preparatory refactor, make that refactor mechanical
and explain why it is needed. Do not mix cleanup, unrelated examples, lockfile
churn, and the bug fix unless each part is necessary for the same rule.

### 8. Do the boring checks before opening the PR

Run the smallest relevant checks first, then the broader ones if the change
touches codegen or examples.

Common checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test -p <changed-crate> --all-targets
```

For `rustc-codegen-cuda` and examples:

```bash
cd crates/rustc-codegen-cuda
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cd ../..
scripts/smoketest.sh --compile-only --no-color
```

For `error*` examples:

```bash
scripts/check-error-example-status.sh
```

Also check upstream `CONTRIBUTING.md` before opening the PR:

- commits need DCO signoff (`git commit -s`);
- new source files need the NVIDIA copyright and SPDX header;
- avoid external-author changes under `crates/cuda-bindings/`.

## Upstreaming draft files

For each prepared upstream branch, keep local draft text under an uncommitted
`upstreaming/<branch-topic>/` directory:

- `issue.md` describes the user-visible problem and desired behavior, not the
  exact implementation.
- `pr.md` explains the implementation, diligence, tests, and validation.

Do not commit these draft files unless the upstream repository explicitly wants
them. They are local copy/paste sources for GitHub issues and PRs.

Use this structure for `issue.md`:

```markdown
# Short issue title

## Problem

What fails today? Describe the user-visible behavior or missing capability.

## Why this matters

Why should upstream care? Explain the correctness, usability, or maintenance
impact without referencing Impulse-only context unless it is essential.

## Expected behavior

What should users or the compiler/runtime be able to do after this is fixed?

## Acceptance criteria

- Observable condition 1.
- Regression coverage expectation.
- Edge case or sibling path that should be covered.
```

## PR description template

Use this structure when opening an upstream PR:

```markdown
## Problem

What rule was wrong, and what user-visible failure did it cause?
Include relevant related issues or PRs here, especially if an issue already
tracks the gap or an earlier PR explains why the current behavior exists.

## Fix

Where is the rule now enforced? Why is this the right abstraction boundary?

## Diligence

Which sibling paths did you check for the same bug? Which were unified,
hardened, or deliberately left alone?

## Tests

What regression examples/unit tests were added? What edge cases do they cover?

## Validation

Commands run locally.
```

The "Diligence" section is important. It tells reviewers that the patch is not
just the first green fix for the downstream repro.

Keep the heading shape stable. If there are related issues or PRs, mention them
inside the relevant section rather than adding a new top-level heading. If you
checked and found none, say that briefly instead of leaving reviewers to wonder.

## Quick self-review questions

Before requesting review, ask:

- Would I be surprised if a maintainer adds a commit that generalizes this?
- Did I check every duplicate path I can find?
- Does the fix use rustc's authoritative facts where available?
- Can the same bug still happen through a sibling operation?
- Did I actually run or otherwise verify the regression against `upstream/main`
  and confirm it fails for the right reason before the patch?
- Did I actually run or otherwise verify the same regression on the prepared
  branch and confirm it passes with the patch?
- Does the test cover an edge case, not only the original happy-path repro?
- Is the PR small enough that the mechanism is obvious?

If the answer to any of these is uncomfortable, keep working before opening the
PR.
