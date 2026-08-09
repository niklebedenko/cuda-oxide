#!/usr/bin/env bash
set -euo pipefail

# Manual real-compiler validation for the cross-target device artifact cache.
#
# Quick mode proves clean-target reuse, external-closure restoration, and
# source/architecture invalidation:
#   scripts/device-artifact-cache-e2e.sh
# Full mode additionally checks type layout, static initializers, backend
# bytes, LLVM program bytes, and CUDA compiler-tool bytes:
#   scripts/device-artifact-cache-e2e.sh --full

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-quick}
if [[ "$mode" != "quick" && "$mode" != "--full" ]]; then
    echo "usage: $0 [--full]" >&2
    exit 2
fi

arch=${CUDA_OXIDE_CACHE_E2E_ARCH:-sm_86}
alternate_arch=${CUDA_OXIDE_CACHE_E2E_ALTERNATE_ARCH:-sm_80}
run_root=$(mktemp -d /tmp/cuda-oxide-device-cache-e2e.XXXXXX)
keep=${CUDA_OXIDE_CACHE_E2E_KEEP:-0}
cleanup() {
    status=$?
    if [[ "$keep" == "1" ]]; then
        echo "device-cache-e2e: retained $run_root"
    else
        rm -rf -- "$run_root"
    fi
    exit "$status"
}
trap cleanup EXIT

cd "$repo_root"
if [[ -n "${CARGO_OXIDE_BIN:-}" ]]; then
    cargo_oxide=$CARGO_OXIDE_BIN
else
    cargo build -p cargo-oxide --bin cargo-oxide
    cargo_oxide=$repo_root/target/debug/cargo-oxide
fi
if [[ ! -x "$cargo_oxide" ]]; then
    echo "device-cache-e2e: cargo-oxide is not executable: $cargo_oxide" >&2
    exit 1
fi

cache=$run_root/cache
fixture=$run_root/vecadd
cp -a crates/rustc-codegen-cuda/examples/vecadd "$fixture"
rm -f -- "$fixture/Cargo.lock"
sed -i \
    -e "s#../../../cuda-device#$repo_root/crates/cuda-device#g" \
    -e "s#../../../cuda-host#$repo_root/crates/cuda-host#g" \
    -e "s#../../../cuda-core#$repo_root/crates/cuda-core#g" \
    "$fixture/Cargo.toml"
perl -0pi -e 's/#\[cuda_module\]\nmod kernels \{/#\[cuda_module\]\nmod kernels {\n    trait DeviceScale {\n        const VALUE: f32;\n    }\n\n    struct UnitDeviceScale;\n\n    impl DeviceScale for UnitDeviceScale {\n        const VALUE: f32 = 1.0;\n    }/' \
    "$fixture/src/main.rs"
sed -i \
    's/\*c_elem = a\[idx_raw\] + b\[idx_raw\];/*c_elem = (a[idx_raw] + b[idx_raw]) * UnitDeviceScale::VALUE;/' \
    "$fixture/src/main.rs"
grep -Fq 'const VALUE: f32 = 1.0;' "$fixture/src/main.rs"
grep -Fq 'UnitDeviceScale::VALUE' "$fixture/src/main.rs"

run_build() {
    label=$1
    target_arch=$2
    target_dir=$3
    log=$run_root/$label.log
    env_args=(
        "CUDA_OXIDE_DEVICE_ARTIFACT_CACHE_DIR=$cache"
        "CUDA_OXIDE_DEVICE_ARTIFACT_CACHE_TRACE=${CUDA_OXIDE_CACHE_E2E_TRACE:-1}"
    )
    if [[ -n "${BACKEND_OVERRIDE:-}" ]]; then
        env_args+=("CUDA_OXIDE_BACKEND=$BACKEND_OVERRIDE")
    fi
    if [[ -n "${LIBNVVM_OVERRIDE:-}" ]]; then
        env_args+=("LIBNVVM_PATH=$LIBNVVM_OVERRIDE")
    fi
    if [[ -n "${LLC_OVERRIDE:-}" ]]; then
        env_args+=("CUDA_OXIDE_LLC=$LLC_OVERRIDE")
    fi
    if ! env "${env_args[@]}" "$cargo_oxide" build \
        --materialize-cubin \
        --arch "$target_arch" \
        --cargo-target-dir "$target_dir" \
        -- \
        --manifest-path "$fixture/Cargo.toml" >"$log" 2>&1; then
        tail -200 "$log" >&2
        echo "device-cache-e2e: $label build failed" >&2
        return 1
    fi
}

event_key() {
    event=$1
    log=$2
    sed -n "s/.*device artifact cache $event: //p" "$log" | tail -1
}

require_event() {
    event=$1
    log=$2
    key=$(event_key "$event" "$log")
    if [[ ! "$key" =~ ^[0-9a-f]{64}$ ]]; then
        tail -100 "$log" >&2
        echo "device-cache-e2e: missing cache $event in $log" >&2
        return 1
    fi
    printf '%s' "$key"
}

run_build cold-a "$arch" "$run_root/target-a"
baseline_key=$(require_event miss "$run_root/cold-a.log")
published_key=$(require_event published "$run_root/cold-a.log")
[[ "$published_key" == "$baseline_key" ]]

run_build warm-b "$arch" "$run_root/target-b"
warm_key=$(require_event hit "$run_root/warm-b.log")
[[ "$warm_key" == "$baseline_key" ]]
[[ $(grep -c 'device artifact cache hit:' "$run_root/warm-b.log") -eq 1 ]]
! grep -Eq 'device artifact cache (miss|published):' "$run_root/warm-b.log"
if ! "$run_root/target-b/debug/vecadd" >"$run_root/warm-b-run.log" 2>&1; then
    cat "$run_root/warm-b-run.log" >&2
    echo "device-cache-e2e: restored warm-B cubin failed on the GPU" >&2
    exit 1
fi
grep -Fq 'SUCCESS: All 1024 elements correct' "$run_root/warm-b-run.log"

sed -i 's/const VALUE: f32 = 1.0;/const VALUE: f32 = 2.0;/' "$fixture/src/main.rs"
grep -Fq 'const VALUE: f32 = 2.0;' "$fixture/src/main.rs"
run_build evaluated-constant-change "$arch" "$run_root/target-b"
constant_key=$(require_event miss "$run_root/evaluated-constant-change.log")
[[ "$constant_key" != "$baseline_key" ]]

sed -i 's/const VALUE: f32 = 2.0;/const VALUE: f32 = 1.0;/' "$fixture/src/main.rs"
run_build restored-constant "$arch" "$run_root/target-b"
restored_constant_key=$(require_event hit "$run_root/restored-constant.log")
[[ "$restored_constant_key" == "$baseline_key" ]]

sed -i 's/a\[idx_raw\] + b\[idx_raw\]/a[idx_raw] - b[idx_raw]/' "$fixture/src/main.rs"
grep -Fq 'a[idx_raw] - b[idx_raw]' "$fixture/src/main.rs"
run_build reachable-code-change "$arch" "$run_root/target-b"
code_key=$(require_event miss "$run_root/reachable-code-change.log")
[[ "$code_key" != "$baseline_key" ]]

sed -i 's/a\[idx_raw\] - b\[idx_raw\]/a[idx_raw] + b[idx_raw]/' "$fixture/src/main.rs"
run_build restored-source "$arch" "$run_root/target-b"
restored_key=$(require_event hit "$run_root/restored-source.log")
[[ "$restored_key" == "$baseline_key" ]]

run_build alternate-arch "$alternate_arch" "$run_root/target-b"
arch_key=$(require_event miss "$run_root/alternate-arch.log")
[[ "$arch_key" != "$baseline_key" ]]

run_fixture_mutation() {
    name=$1
    example=$2
    before=$3
    after=$4
    fixture=$run_root/$name
    cp -a "crates/rustc-codegen-cuda/examples/$example" "$fixture"
    rm -f -- "$fixture/Cargo.lock"
    sed -i \
        -e "s#../../../cuda-device#$repo_root/crates/cuda-device#g" \
        -e "s#../../../cuda-host#$repo_root/crates/cuda-host#g" \
        -e "s#../../../cuda-core#$repo_root/crates/cuda-core#g" \
        "$fixture/Cargo.toml"
    run_build "$name-baseline" "$arch" "$run_root/target-$name"
    base=$(require_event miss "$run_root/$name-baseline.log")
    sed -i "s|$before|$after|" "$fixture/src/main.rs"
    run_build "$name-change" "$arch" "$run_root/target-$name"
    changed=$(require_event miss "$run_root/$name-change.log")
    [[ "$changed" != "$base" ]]
}

run_external_closure() {
    label=$1
    target_dir=$2
    log=$run_root/$label.log
    if ! env \
        "CARGO_TARGET_DIR=$target_dir" \
        "CUDA_OXIDE_DEVICE_ARTIFACT_CACHE_DIR=$cache" \
        "CUDA_OXIDE_DEVICE_ARTIFACT_CACHE_TRACE=${CUDA_OXIDE_CACHE_E2E_TRACE:-1}" \
        "$cargo_oxide" run extern_crate_closure \
        --materialize-cubin --arch "$arch" >"$log" 2>&1; then
        tail -200 "$log" >&2
        echo "device-cache-e2e: $label cross-crate closure build or GPU run failed" >&2
        return 1
    fi
    grep -Fq 'PASSED: all external closure and wrapped-pair results correct' "$log"
}

run_external_closure closure-cold "$run_root/target-extern-crate-closure-a"
closure_key=$(require_event miss "$run_root/closure-cold.log")
closure_published_key=$(require_event published "$run_root/closure-cold.log")
[[ "$closure_published_key" == "$closure_key" ]]

run_external_closure closure-warm "$run_root/target-extern-crate-closure-b"
closure_warm_key=$(require_event hit "$run_root/closure-warm.log")
[[ "$closure_warm_key" == "$closure_key" ]]
[[ $(grep -c 'device artifact cache hit:' "$run_root/closure-warm.log") -eq 1 ]]
! grep -Eq 'device artifact cache (miss|published):' "$run_root/closure-warm.log"

if [[ "$mode" == "--full" ]]; then

    run_fixture_mutation \
        layout \
        field_array_assign \
        '#\[derive(Copy, Clone)\]' \
        '#[derive(Copy, Clone)]\n#[repr(align(32))]'
    run_fixture_mutation \
        static \
        device_global \
        '\[\[0.25, 0.5\]' \
        '[[0.375, 0.5]'

    fixture=$run_root/vecadd

    objcopy_bin=$(command -v llvm-objcopy || command -v objcopy || true)
    if [[ -z "$objcopy_bin" ]]; then
        echo "device-cache-e2e: --full requires llvm-objcopy or objcopy" >&2
        exit 1
    fi
    host=$(rustc -vV | sed -n 's/^host: //p')
    backend=$repo_root/crates/rustc-codegen-cuda/target/$host/debug/librustc_codegen_cuda.so
    if [[ ! -f "$backend" ]]; then
        echo "device-cache-e2e: backend not found after build: $backend" >&2
        exit 1
    fi

    rust_sysroot=$(rustc --print sysroot)
    llc=$rust_sysroot/lib/rustlib/$host/bin/llc
    if [[ ! -x "$llc" ]]; then
        llc=$(command -v llc-22 || command -v llc-21 || command -v llc || true)
    fi
    if [[ -z "$llc" || ! -x "$llc" ]]; then
        echo "device-cache-e2e: --full could not locate the selected llc" >&2
        exit 1
    fi
    llc=$(realpath "$llc")
    llc_variant=$run_root/llc.variant
    printf '#!/bin/sh\n# cache-e2e-llvm-tool-a\nexec "%s" "$@"\n' \
        "$llc" >"$llc_variant"
    chmod +x "$llc_variant"
    LLC_OVERRIDE=$llc_variant run_build \
        llvm-tool-baseline "$arch" "$run_root/target-llvm-tool-change"
    llvm_tool_base_key=$(require_event miss "$run_root/llvm-tool-baseline.log")
    llvm_tool_base_published_key=$(require_event published "$run_root/llvm-tool-baseline.log")
    [[ "$llvm_tool_base_published_key" == "$llvm_tool_base_key" ]]
    sed -i 's/cache-e2e-llvm-tool-a/cache-e2e-llvm-tool-b/' "$llc_variant"
    LLC_OVERRIDE=$llc_variant run_build \
        llvm-tool-change "$arch" "$run_root/target-llvm-tool-change"
    llvm_tool_changed_key=$(require_event miss "$run_root/llvm-tool-change.log")
    llvm_tool_changed_published_key=$(require_event published "$run_root/llvm-tool-change.log")
    [[ "$llvm_tool_changed_published_key" == "$llvm_tool_changed_key" ]]
    [[ "$llvm_tool_changed_key" != "$llvm_tool_base_key" ]]
    sed -i 's/cache-e2e-llvm-tool-b/cache-e2e-llvm-tool-a/' "$llc_variant"
    LLC_OVERRIDE=$llc_variant run_build \
        llvm-tool-restored "$arch" "$run_root/target-llvm-tool-change"
    llvm_tool_restored_key=$(require_event hit "$run_root/llvm-tool-restored.log")
    [[ "$llvm_tool_restored_key" == "$llvm_tool_base_key" ]]
    ! grep -Eq 'device artifact cache (miss|published):' "$run_root/llvm-tool-restored.log"

    backend_variant=$run_root/librustc_codegen_cuda.variant.so
    cp -- "$backend" "$backend_variant"
    printf 'cache-e2e-backend-variant\n' >"$run_root/backend-marker"
    "$objcopy_bin" \
        --add-section ".cuda_oxide_cache_e2e=$run_root/backend-marker" \
        "$backend_variant"
    BACKEND_OVERRIDE=$backend_variant run_build \
        backend-change "$arch" "$run_root/target-backend-change"
    backend_key=$(require_event miss "$run_root/backend-change.log")
    [[ "$backend_key" != "$baseline_key" ]]

    libnvvm=$(find /usr/local/cuda*/nvvm/lib64 -maxdepth 1 -type f \
        -name 'libnvvm.so.*' 2>/dev/null | sort | tail -1)
    if [[ -z "$libnvvm" ]]; then
        echo "device-cache-e2e: --full could not locate libnvvm" >&2
        exit 1
    fi
    libnvvm_variant=$run_root/libnvvm.variant.so
    cp -- "$libnvvm" "$libnvvm_variant"
    printf 'cache-e2e-tool-variant\n' >"$run_root/tool-marker"
    "$objcopy_bin" \
        --add-section ".cuda_oxide_cache_e2e=$run_root/tool-marker" \
        "$libnvvm_variant"
    LIBNVVM_OVERRIDE=$libnvvm_variant run_build \
        tool-change "$arch" "$run_root/target-tool-change"
    tool_key=$(require_event miss "$run_root/tool-change.log")
    [[ "$tool_key" != "$baseline_key" ]]
fi

echo "device-cache-e2e: PASS"
echo "  clean target A miss/publish: $baseline_key"
echo "  clean target B hit:          $warm_key"
echo "  evaluated constant miss:     $constant_key"
echo "  reachable source miss:       $code_key"
echo "  alternate architecture miss: $arch_key"
echo "  cross-crate closure miss:     $closure_key"
echo "  cross-crate closure hit:      $closure_warm_key"
if [[ "$mode" == "--full" ]]; then
    echo "  LLVM tool byte-change miss:   $llvm_tool_changed_key"
fi
