#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/enum_face_index.ptx"
test -s "${ptx}"

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

for symbol in loaded_axis_index literal_x_tet_face; do
    body="$(symbol_body "${symbol}")"
    if [[ -z "${body}" ]]; then
        echo "error: missing ${symbol} in ${ptx}" >&2
        exit 1
    fi
    if grep -Fq 'trap;' <<<"${body}"; then
        echo "error: ${symbol} retained an unreachable trap" >&2
        printf '%s\n' "${body}" >&2
        exit 1
    fi
done

for symbol in runtime_index_control sparse_enum_control; do
    control="$(symbol_body "${symbol}")"
    if [[ -z "${control}" ]] || ! grep -Fq 'trap;' <<<"${control}"; then
        echo "error: ${symbol} lost its genuine bounds trap" >&2
        printf '%s\n' "${control}" >&2
        exit 1
    fi
done

echo "enum_face_index code shape: PASS"
