# Install AgentKernelArena

AgentKernelArena runs AI coding agents against GPU kernel tasks on an AMD GPU and
evaluates the results. Installation sets up a Python environment with the correct
ROCm PyTorch build, installs the framework dependencies, and makes at least one
agent CLI available.

## Prerequisites

- **Python 3.12+**
- **AMD GPU with ROCm** — ROCm 6.4, 7.0, or 7.1 (the `Makefile` auto-detects the
  installed version under `/opt/rocm-*`)
- **ROCm toolchain for HIP tasks** — `hipcc` and `rocprof-compute`
- **Triton** — required for Triton tasks (installed with the ROCm PyTorch wheel)
- **[uv](https://github.com/astral-sh/uv)** — used by the `Makefile` to create the
  virtual environment
- **Git**
- At least one supported agent CLI and the matching API key (see
  [Configure agents and models](../how-to/agents.md))

## Recommended: `make setup`

From the repository root, the `Makefile` detects your ROCm version, creates a
`.venv` virtual environment with the matching ROCm PyTorch build, and installs
all dependencies.

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

# Full environment setup (venv + deps; includes FlyDSL by default)
make setup

# Or set up without FlyDSL support
make setup WITH_FLYDSL=0
```

Activate the environment:

```bash
make act
# or
source .venv/bin/activate
```

`make setup` will fail with `Could not detect ROCm installation` if no
`/opt/rocm-*` directory is found. Install ROCm first, then re-run.

### FlyDSL tasks

The `flydsl2flydsl` task category requires [FlyDSL](https://github.com/ROCm/FlyDSL).
It is installed by default with `make setup`. To install or verify it on its own:

```bash
make setup-flydsl     # install FlyDSL into the venv and verify
make verify-flydsl    # verify FlyDSL import and ROCm PyTorch GPU access
```

## Manual installation

If you prefer not to use the `Makefile`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Install the ROCm PyTorch build that matches your ROCm version, for example:
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.4

pip install -r requirements.txt
```

## Install agent CLIs

Install whichever agents you plan to evaluate. For example:

```bash
# Cursor Agent CLI
make install-cursor-agent

# Claude Code
npm install -g @anthropic-ai/claude-code

# Codex CLI: follow the official Codex CLI instructions, then ensure
# `codex` is available on PATH.
```

## Configure API keys

Export the keys for the providers you will use:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

To run against a self-hosted model instead of a hosted provider, start a local
vLLM server:

```bash
make vllm
```

This launches a `rocm/vllm` container serving a model on port `30001`. Set
`local_llm_enabled: true` in the relevant agent configuration to use it.

## Verify the installation

Run a single quick task to confirm the framework, GPU, and agent CLI all work
together. Edit `config.yaml` to select one agent and one task, then run:

```bash
python main.py
```

A successful run creates a timestamped workspace directory
(`workspace_<gpu>_<agent>/run_<timestamp>/`), logs to `logs/`, and writes a
`task_result.yaml` per task. See [Run an evaluation](../how-to/run-evaluation.md)
for the full workflow.
