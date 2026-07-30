#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/warp_reduce.ptx"
test -s "${ptx}"

ptxas="${PTXAS:-}"
if [[ -z "${ptxas}" ]]; then
    if [[ -x /usr/local/cuda/bin/ptxas ]]; then
        ptxas=/usr/local/cuda/bin/ptxas
    else
        ptxas="$(command -v ptxas)"
    fi
fi
cuobjdump="${CUOBJDUMP:-}"
if [[ -z "${cuobjdump}" ]]; then
    if [[ -x /usr/local/cuda/bin/cuobjdump ]]; then
        cuobjdump=/usr/local/cuda/bin/cuobjdump
    else
        cuobjdump="$(command -v cuobjdump)"
    fi
fi

kernel_ptx="$(
    awk '
        /^\.visible \.entry warp_uniform_guarded_reduce\(/ { in_kernel = 1 }
        in_kernel { print }
        in_kernel && /^}/ { exit }
    ' "${ptx}"
)"
test -n "${kernel_ptx}"

vote_count="$(grep -c 'vote\.sync\.all\.pred' <<<"${kernel_ptx}")"
shuffle_count="$(grep -c 'shfl\.sync\.bfly\.b32' <<<"${kernel_ptx}")"
if [[ ${vote_count} -ne 5 || ${shuffle_count} -ne 25 ]]; then
    echo "warp-uniform guard expected 5 votes and 25 butterfly shuffles in PTX; got ${vote_count} and ${shuffle_count}" >&2
    exit 1
fi

scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
"${ptxas}" \
    -arch=sm_86 \
    -O3 \
    -e warp_uniform_guarded_reduce \
    "${ptx}" \
    -o "${scratch}/warp_uniform_guarded_reduce.cubin"
"${cuobjdump}" \
    --dump-sass \
    "${scratch}/warp_uniform_guarded_reduce.cubin" \
    >"${scratch}/warp_uniform_guarded_reduce.sass"

sass="${scratch}/warp_uniform_guarded_reduce.sass"
sass_vote_count="$(grep -c '\bVOTE\.ALL\b' "${sass}")"
sass_shuffle_count="$(grep -c '\bSHFL\.BFLY\b' "${sass}")"
if [[ ${sass_vote_count} -ne 5 || ${sass_shuffle_count} -ne 25 ]]; then
    echo "warp-uniform guard expected 5 VOTE.ALL and 25 native SHFL.BFLY instructions; got ${sass_vote_count} and ${sass_shuffle_count}" >&2
    exit 1
fi
if grep -Eq 'BRA\.DIV|CALL\.(REL|ABS)\.NOINC|__cuda_sm[0-9]+_shflsync' "${sass}"; then
    echo "warp-uniform guard grew a ptxas divergence fallback" >&2
    exit 1
fi

echo "warp_reduce uniform-control code shape: PASS"
