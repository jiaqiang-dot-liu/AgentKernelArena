#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/src/scripts/docker_benchmark.sh"
PINNED_GFX950_IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705"
OLD_GFX950_IMAGE="lmsysorg/sglang:v0.5.12-rocm720-mi35x"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_has() {
    local expected="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" == "$expected" ]] && return 0
    done
    fail "missing Docker argument: $expected"
}

assert_not_has() {
    local unexpected="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" != "$unexpected" ]] || fail "unexpected Docker argument: $unexpected"
    done
}

# Capture the exact argv that the runner would pass to Docker without requiring
# a daemon, GPU devices, or the benchmark images on this host.
docker() {
    printf '%s\n' "$@"
}
export -f docker

run_shell_args() {
    env \
        HOME="$TEST_HOME" \
        AKA_AGENTS=test-noop-agent \
        "$@" \
        bash "$RUNNER" shell 2>/dev/null
}

assert_cache_args_present() {
    local suffix="$1"
    shift
    assert_has "AITER_JIT_DIR=/tmp/aiter-jit${suffix}" "$@"
    assert_has "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-runtime-cache${suffix}" "$@"
    assert_has "/tmp/aiter_configs:rw,uid=$(id -u),gid=$(id -g),mode=1777" "$@"
}

assert_cache_args_absent() {
    assert_not_has "AITER_JIT_DIR=/tmp/aiter-jit" "$@"
    assert_not_has "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-runtime-cache" "$@"
    assert_not_has "/tmp/aiter_configs:rw,uid=$(id -u),gid=$(id -g),mode=1777" "$@"
}

TEST_HOME="$(mktemp -d)"
trap 'rm -rf "$TEST_HOME"' EXIT

bash -n "$RUNNER"

# The gfx950 default resolves to the pinned image and enables writable caches.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950)
assert_has "$PINNED_GFX950_IMAGE" "${args[@]}"
assert_cache_args_present "" "${args[@]}"

# A worker suffix must isolate both runtime cache directories.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_CACHE_SUFFIX=worker/3)
assert_cache_args_present "-worker_3" "${args[@]}"

# Explicitly selecting the same verified tag has the same behavior.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE="$PINNED_GFX950_IMAGE")
assert_cache_args_present "" "${args[@]}"

# Old and custom gfx950 images retain their existing Docker arguments.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE="$OLD_GFX950_IMAGE")
assert_cache_args_absent "${args[@]}"
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE_GFX950=example.invalid/custom:latest)
assert_cache_args_absent "${args[@]}"

# The unchanged gfx942 default does not receive the gfx950-only configuration.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx942)
assert_has "lmsysorg/sglang:v0.5.12-rocm720-mi30x" "${args[@]}"
assert_cache_args_absent "${args[@]}"

# Image equality alone is insufficient: the selected architecture must be gfx950.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx942 AKA_DOCKER_IMAGE="$PINNED_GFX950_IMAGE")
assert_cache_args_absent "${args[@]}"

echo "PASS: docker_benchmark gfx950 runtime argument tests"
