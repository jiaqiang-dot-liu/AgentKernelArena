#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_aka_forge_example.sh — one-click AgentKernelArena demo with the FORGE agent.
#
# Runs ONE kernel-optimization task through KernelForge's autonomous forge-loop
# (baseline -> agent edit -> 5-stage validate -> bench -> keep/revert), wrapped
# as an Arena agent. Arena then re-scores the resulting kernel with its own
# compile/correctness/performance commands.
#
# Usage:
#     bash run_aka_forge_example.sh
#
# Env-overridable, e.g.:
#     TASK=flydsl2flydsl/rmsnorm_kernel bash run_aka_forge_example.sh
#     MODEL=claude-opus-4-7 bash run_aka_forge_example.sh
#     AKA_GPU_ARCH=gfx950 bash run_aka_forge_example.sh   # force arch if autodetect fails
#
# NOTE: the API key is hardcoded below for convenience — treat this file as a
#       SECRET and do NOT commit it.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 1. LLM gateway auth (AMD primus-safe, bearer token) ──────────────────────
export FORGE_API_KEY="${FORGE_API_KEY:-ak-xxxxxxx}"
# The base URL must NOT end in /v1: the claude CLI/SDK appends /v1/messages itself,
# so a trailing /v1 here yields /v1/v1/messages -> 404.
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://project1.tw325.primus-safe.amd.com/api/v1/llm-proxy}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$FORGE_API_KEY}"
# This gateway (BRAIN_Hyperloom) requires a mandatory 'user: <NTID>' header on every
# request, and serves an internal TLS cert not in the system trust store.
export FORGE_USER_NTID="${FORGE_USER_NTID:-jqliu}"
export ANTHROPIC_CUSTOM_HEADERS="${ANTHROPIC_CUSTOM_HEADERS:-user: $FORGE_USER_NTID}"
export NODE_TLS_REJECT_UNAUTHORIZED="${NODE_TLS_REJECT_UNAUTHORIZED:-0}"
unset ANTHROPIC_API_KEY 2>/dev/null || true

# ── 2. Runtime / ROCm / cache env ────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:/opt/venv/bin:$PATH"   # ~/.local/bin holds the claude CLI

# Detect the GPU architecture from the running machine so this script is portable
# across nodes (MI300/MI325 -> gfx942, MI350/MI355 -> gfx950, ...). Export
# AKA_GPU_ARCH beforehand to override the autodetection.
detect_gpu_arch() {
    local arch=""
    if command -v rocminfo >/dev/null 2>&1; then
        arch="$(rocminfo 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true)"
    fi
    if [[ -z "$arch" ]] && command -v rocm-smi >/dev/null 2>&1; then
        arch="$(rocm-smi --showhw 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true)"
    fi
    if [[ -z "$arch" ]]; then
        arch="$(python -c 'import torch
try:
    print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
except Exception:
    pass' 2>/dev/null || true)"
    fi
    printf '%s' "$arch"
}

export AKA_GPU_ARCH="${AKA_GPU_ARCH:-$(detect_gpu_arch)}"
if [[ -z "$AKA_GPU_ARCH" ]]; then
    echo "ERROR: could not detect a ROCm GPU arch (tried rocminfo, rocm-smi, torch). Set AKA_GPU_ARCH manually." >&2
    exit 1
fi
export AGENT_KERNEL_ARENA_GPU_ARCH="$AKA_GPU_ARCH"
export PYTORCH_ROCM_ARCH="$AKA_GPU_ARCH"
export GPU_TARGET="$AKA_GPU_ARCH"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch-extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/agent-cache}"
export IS_SANDBOX=1
export PYTHONUNBUFFERED=1
# Model for the forge agent (also honored by agent_config.yaml default).
export KERNEL_AGENTS_MODEL="${MODEL:-claude-opus-4-8}"

# ── 3. What to run ───────────────────────────────────────────────────────────
TASK="${TASK:-flydsl2flydsl/softmax_kernel}"
# Arena's target_gpu_model, derived from the detected arch (override via GPU_MODEL).
case "$AKA_GPU_ARCH" in
    gfx942) GPU_MODEL="${GPU_MODEL:-MI300}" ;;
    gfx950) GPU_MODEL="${GPU_MODEL:-MI355X}" ;;
    *)      GPU_MODEL="${GPU_MODEL:-MI300}" ;;
esac
# Real card name (display only). The physical card may be a gfx942/gfx950 variant
# (e.g. MI325X) that Arena buckets into the coarser $GPU_MODEL profile above.
GPU_NAME="$(python -c 'import torch
try:
    print(torch.cuda.get_device_name(0))
except Exception:
    pass' 2>/dev/null || true)"
GPU_NAME="${GPU_NAME:-unknown}"
RUN_SUFFIX="${RUN_SUFFIX:-forge_demo}"
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config.forge_example.yaml}"

cat > "$CONFIG_FILE" <<EOF
agent:
  template: forge

tasks:
  - $TASK

target_gpu_model: $GPU_MODEL
log_directory: logs
workspace_directory_prefix: workspace
EOF

echo "── AgentKernelArena FORGE example ───────────────────────"
echo "  repo      : $REPO_ROOT"
echo "  task      : $TASK"
echo "  agent     : forge   model: $KERNEL_AGENTS_MODEL"
echo "  gpu       : $GPU_NAME   arch: $AKA_GPU_ARCH   (arena profile: $GPU_MODEL)"
echo "  gateway   : $ANTHROPIC_BASE_URL"
echo "  config    : $CONFIG_FILE"
echo "─────────────────────────────────────────────────────────"

command -v kernel-agents >/dev/null 2>&1 || { echo "ERROR: kernel-agents (KernelForge) not found in PATH"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "ERROR: claude CLI not found in PATH"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'no ROCm/CUDA device'; print('GPU OK:', torch.cuda.get_device_name(0))"

cd "$REPO_ROOT"
exec python main.py --config_name "$CONFIG_FILE" --run-suffix "$RUN_SUFFIX"
