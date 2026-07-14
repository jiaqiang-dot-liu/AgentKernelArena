---
myst:
    html_meta:
        "description": "Install the Docker-first AgentKernelArena experimentation environment, pinned ROCm/SGLang images, and supported agent integrations."
        "keywords": "AgentKernelArena, install, Docker, ROCm, SGLang, AMD GPU, HIP, PyTorch, agent CLI"
---

# Install AgentKernelArena

AgentKernelArena runs controlled agent experiments against GPU kernel tasks on
an AMD GPU. Docker is the supported workflow: each experiment runs inside the
GPU-architecture-specific SGLang image and bind-mounts the required local agent
CLI plus its login state.

## Prerequisites

The following prerequisites are required before running AgentKernelArena.

- **Docker**
- **AMD GPU with ROCm-compatible Docker access** — the runner mounts `/dev/kfd`,
  `/dev/dri`, and `/dev/mem` when present.
- **SGLang runtime image** — `gfx942` uses
  `lmsysorg/sglang:v0.5.12-rocm720-mi30x`; `gfx950` uses
  `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705`. The runner selects from
  `target_gpu_model` for experiment runs and from the visible host GPU for shell
  and smoke commands.
- **Git**
- **Node.js 22+ and npm**, when using the alternative npm installation of Claude
  Code or another npm-installed agent CLI.
- The selected agent CLI installed and authenticated on the host. The Docker
  runner provisions only the configured agent for a normal run. Codex, Claude
  Code, and Cursor Agent are the first-class host-CLI integrations. See
  [Configure agents and models](../how-to/agents.md).

## Docker runner

From the repository root, use the Docker-first Makefile targets. The runner does
not copy credentials into an image; it bind-mounts the existing host login state.

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

# Verify the container runtime and GPU.
make docker-smoke
```

Start an interactive shell in the same environment:

```bash
make docker-shell
```

Install and authenticate the agent selected by your configuration before
starting an experiment; the next section covers the first-class host CLIs.

## Install agent CLIs

Install whichever agent you plan to run. Claude Code recommends its native
installer; npm is also supported by the Docker runner:

```bash
# Recommended native installation on Linux/macOS/WSL:
curl -fsSL https://claude.ai/install.sh | bash

# Alternative npm installation (requires Node.js 22+):
# node --version
# npm --version
# npm install -g @anthropic-ai/claude-code

# Authenticate after either installation.
claude

# Alternatively, install Cursor Agent:
# make install-cursor-agent

# For Codex, follow its official CLI instructions and ensure `codex` is on PATH.
```

The Docker runner supports both npm-installed Claude Code and its native/local
installation. See the
[official Claude Code setup guide](https://code.claude.com/docs/en/installation)
for current installation alternatives. After completing the interactive login,
verify the host CLI with `claude auth status`.

The `geak_v3`, `geak_v3_triton`, and `mini_swe_triton` integrations require
their own runtime dependencies. Review the corresponding directory under
`agents/` before selecting one.

## FlyDSL tasks (optional)

`flydsl2flydsl`, `torch2flydsl`, and `triton2flydsl` tasks need the `flydsl`
package inside the container. The selected image may already ship it
(`make docker-smoke` prints `flydsl=ok <version>` when present). If yours does
not, install it once into the container's persistent pip user-base:

```bash
make docker-setup-flydsl
```

This is a no-op when the image already provides FlyDSL.

## Configure authentication and providers

Cursor, Claude Code, and Codex reuse their host CLI authentication. A normal run
preflights only its selected CLI. For a config that selects one of these CLIs—or
`task_validator`, which resolves to its configured backend—check the same CLI
without starting a task by passing the run config:

```bash
make docker-check-agents CONFIG=config.yaml

# Optional overrides:
make docker-check-agents AGENTS=claude_code,codex
make docker-check-agents AGENTS=all
```

`AGENTS=all` is the explicit strict check for Cursor, Claude Code, and Codex.
Specialized integrations such as GEAK and mini-swe use their own dependency and
authentication checks. They read credentials and provider endpoints from their
own environment/configuration; there is no shared provider field in the root
`config.yaml`.

To run against a self-hosted model instead of a hosted provider, start a local
vLLM server:

```bash
make vllm
```

This launches a `rocm/vllm` container with an OpenAI-compatible endpoint on port
`30001`. Starting the server does not automatically configure an agent; point
the selected integration at the endpoint using that integration's base-URL and
provider settings.

## Verify the installation

Run a single quick task to confirm the framework, GPU, and agent CLI all work
together. Edit `config.yaml` to select one agent and one task, then run:

```bash
make docker-run CONFIG=config.yaml
```

To run across multiple GPUs, list host GPU IDs or omit `GPU_IDS` to discover
them with `rocm-smi --showid`:

```bash
make docker-parallel-run CONFIG=config.yaml GPU_IDS=0,1,2,3
```

A successful run creates a timestamped workspace directory
(`workspace_<gpu>_<agent>/run_<timestamp>/`), logs to `logs/`, and writes a
`task_result.yaml` per task. See [Run an experiment](../how-to/run-evaluation.md)
for the full workflow.
