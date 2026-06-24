---
myst:
    html_meta:
        "description": "Supported and tested hardware, software versions, agent CLIs, and model providers for AgentKernelArena, including ROCm and Python requirements."
        "keywords": "AgentKernelArena, compatibility matrix, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers"
---

# AgentKernelArena compatibility matrix

The following versions are the supported and tested configurations for
AgentKernelArena. Entries marked `TODO (verify)` must be confirmed against a
tested environment before publication.

## Hardware

The following hardware configurations are supported and tested.

| Component | Supported | Notes |
| --- | --- | --- |
| GPU architecture | AMD Instinct™ MI300 series | `target_gpu_model: MI300`. TODO (verify) other architectures. |
| GPU architecture | AMD Instinct MI355X | TODO (verify) |

## Software

The following software versions are required or verified.

| Component | Version | Notes |
| --- | --- | --- |
| ROCm | 6.4, 7.0, 7.1 | Auto-detected by the `Makefile` from `/opt/rocm-*`. |
| Python | 3.12+ | |
| PyTorch | ROCm build matching the detected ROCm version | Installed by `make setup` (nightly for ROCm 7.x). |
| Triton | Bundled with the ROCm PyTorch wheel | Required for Triton task categories. |
| FlyDSL | Latest from PyPI | Required for `flydsl2flydsl` tasks. TODO (verify) pinned version. |
| uv | Latest | Used to create the virtual environment. |
| hipcc | Matches ROCm | Required for HIP tasks. |
| rocprof-compute | Matches ROCm | Required for HIP performance profiling. |

## Agents

The following agent CLIs have been tested with AgentKernelArena.

| Agent | Tested version | Notes |
| --- | --- | --- |
| Cursor Agent CLI | TODO (verify) | Installed using `make install-cursor-agent`. |
| Claude Code | TODO (verify) | `npm install -g @anthropic-ai/claude-code`. |
| Codex CLI | TODO (verify) | Installed per the official Codex CLI instructions. |
| SWE-agent | TODO (verify) | |
| OpenEvolve (GEAK) | TODO (verify) | |

## Model providers

The following model providers are supported.

| Provider | Notes |
| --- | --- |
| OpenAI | Requires `OPENAI_API_KEY`. |
| Anthropic | Requires `ANTHROPIC_API_KEY`. |
| OpenRouter | Requires `OPENROUTER_API_KEY`. |
| Local vLLM | Self-hosted on port `30001` using `make vllm`. |
