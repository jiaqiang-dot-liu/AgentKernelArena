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
#     TASK=triton2triton/rocmbench/easy/test_add_kernel bash run_aka_forge_example.sh
#     MODEL=claude-opus-4-7 bash run_aka_forge_example.sh
#
# NOTE: the API key is hardcoded below for convenience — treat this file as a
#       SECRET and do NOT commit it.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 1. LLM gateway auth (AMD core42 / primus-safe, bearer token) ─────────────
export FORGE_API_KEY="${FORGE_API_KEY:-ak-xxxxxxx}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://core42.primus-safe.amd.com:443/api/v1/llm-proxy}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$FORGE_API_KEY}"
unset ANTHROPIC_API_KEY 2>/dev/null || true

# ── 2. Runtime / ROCm / cache env (what the Docker wrapper normally sets) ─────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/opt/venv/bin:$PATH"
export AKA_GPU_ARCH="${AKA_GPU_ARCH:-gfx942}"           # MI300X=gfx942, MI355X=gfx950
export AGENT_KERNEL_ARENA_GPU_ARCH="$AKA_GPU_ARCH"
export PYTORCH_ROCM_ARCH="$AKA_GPU_ARCH"
export GPU_TARGET="$AKA_GPU_ARCH"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch-extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/agent-cache}"
export IS_SANDBOX=1
export PYTHONUNBUFFERED=1
# Model for the forge agent (also honored by agent_config.yaml default).
export KERNEL_AGENTS_MODEL="${MODEL:-claude-opus-4-8}"

# ── 3. What to run ───────────────────────────────────────────────────────────
TASK="${TASK:-triton2triton/rocmbench/easy/test_add_kernel}"
GPU_MODEL="${GPU_MODEL:-MI300}"
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
echo "  agent     : forge   gpu: $GPU_MODEL ($AKA_GPU_ARCH)   model: $KERNEL_AGENTS_MODEL"
echo "  gateway   : $ANTHROPIC_BASE_URL"
echo "  config    : $CONFIG_FILE"
echo "─────────────────────────────────────────────────────────"

command -v kernel-agents >/dev/null 2>&1 || { echo "ERROR: kernel-agents (KernelForge) not found in PATH"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "ERROR: claude CLI not found in PATH"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'no ROCm/CUDA device'; print('GPU OK:', torch.cuda.get_device_name(0))"

cd "$REPO_ROOT"
exec python main.py --config_name "$CONFIG_FILE" --run-suffix "$RUN_SUFFIX"
