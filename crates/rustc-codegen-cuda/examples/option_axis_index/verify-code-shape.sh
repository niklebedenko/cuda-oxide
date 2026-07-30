#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/option_axis_index.ptx"
test -s "${ptx}"

symbol_body() {
    local symbol="$1"

    # Unoptimized PTX may include forward declarations. Wait for the matching
    # header to reach `{`; a prototype reaches `;` and is skipped.
    awk -v marker="${symbol}(" '
        !emit && !candidate && index($0, marker) &&
            (index($0, ".func") || index($0, ".entry")) {
            candidate = 1
        }
        candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ {
            candidate = 0
            next
        }
        candidate && $0 ~ /^[[:space:]]*\{[[:space:]]*$/ {
            emit = 1
            candidate = 0
        }
        emit { print }
        emit && index($0, "End function") != 0 { exit }
    ' "${ptx}"
}

entry="$(symbol_body option_axis_index_kernel)"
if [[ -z "${entry}" ]]; then
    echo "error: missing option_axis_index_kernel in ${ptx}" >&2
    exit 1
fi

if grep -Eq 'trap;|(^|[[:space:]])call(\.|[[:space:]])' <<<"${entry}"; then
    echo "error: Option<Axis> indexing retained a trap or outlined call" >&2
    printf '%s\n' "${entry}" >&2
    exit 1
fi

control="$(symbol_body runtime_index_control)"
if [[ -z "${control}" ]]; then
    echo "error: missing runtime_index_control in ${ptx}" >&2
    exit 1
fi
if ! grep -Fq 'trap;' <<<"${control}"; then
    echo "error: genuinely-unbounded runtime index lost its bounds trap" >&2
    printf '%s\n' "${control}" >&2
    exit 1
fi

echo "option_axis_index code shape: PASS"
