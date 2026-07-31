#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/unchecked_add_array_index.ptx"
llvm="${root}/unchecked_add_array_index.ll"
test -s "${ptx}"
test -s "${llvm}"

symbol_body() {
    local symbol="$1"
    awk -v marker="${symbol}(" '
        !emit && !candidate && index($0, marker) &&
            (index($0, ".func") || index($0, ".entry")) { candidate = 1 }
        candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ { candidate = 0; next }
        candidate && $0 ~ /^[[:space:]]*\{[[:space:]]*$/ { emit = 1; candidate = 0 }
        emit { print }
        emit && index($0, "End function") != 0 { exit }
    ' "${ptx}"
}

llvm_symbol_body() {
    local symbol="$1"
    awk -v marker="@${symbol}(" '
        !emit && /^define / && index($0, marker) { emit = 1 }
        emit { print }
        emit && /^}/ { exit }
    ' "${llvm}"
}

entry="$(symbol_body correlated_array_index)"
control="$(symbol_body runtime_index_control)"
correlate_llvm="$(llvm_symbol_body unchecked_add_array_index__kernels__correlate)"
test -n "${entry}"
test -n "${control}"
test -n "${correlate_llvm}"

if grep -Fq 'trap;' <<<"${entry}"; then
    echo "error: correlated nine-element indexing retained a bounds trap" >&2
    exit 1
fi
if ! grep -Fq 'trap;' <<<"${control}"; then
    echo "error: genuinely unbounded runtime index lost its bounds trap" >&2
    exit 1
fi
if ! grep -Eq 'add nuw i(32|64)' <<<"${correlate_llvm}"; then
    echo "error: imported AddUnchecked did not reach LLVM with nuw" >&2
    exit 1
fi

echo "unchecked_add_array_index code shape: PASS"
