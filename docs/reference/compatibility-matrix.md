# Compatibility matrix

The following versions are the supported and tested configurations for
AgentKernelArena. Entries marked `TODO (verify)` must be confirmed against a
tested environment before publication.

## Hardware

| Component | Supported | Notes |
| --- | --- | --- |
| GPU architecture | AMD Instinct MI300 series | `target_gpu_model: MI300`. TODO (verify) other architectures. |
| GPU architecture | AMD Instinct MI355X | TODO (verify) |

## Software

| Component | Version | Notes |
| --- | --- | --- |
| ROCm | 6.4, 7.0, 7.1 | Provided by the SGLang Docker image; arch (gfx942/gfx950) selects the image. |
| Python | Provided by the image (e.g. 3.10) | Bundled in the SGLang image. |
| PyTorch | ROCm build bundled in the image | Provided by the SGLang Docker image. |
| Triton | Bundled with the image's ROCm PyTorch | Required for Triton task categories. |
| FlyDSL | Provided by the image (or `make docker-setup-flydsl`) | Required for `flydsl2flydsl` tasks. |
| uv | Latest | Used to create the virtual environment. |
| hipcc | Matches ROCm | Required for HIP tasks. |
| rocprof-compute | Matches ROCm | Required for HIP performance profiling. |

## Agents

| Agent | Tested version | Notes |
| --- | --- | --- |
| Cursor Agent CLI | TODO (verify) | Installed via `make install-cursor-agent`. |
| Claude Code | TODO (verify) | `npm install -g @anthropic-ai/claude-code`. |
| Codex CLI | TODO (verify) | Installed per the official Codex CLI instructions. |

## Model providers

| Provider | Notes |
| --- | --- |
| OpenAI | Requires `OPENAI_API_KEY`. |
| Anthropic | Requires `ANTHROPIC_API_KEY`. |
| OpenRouter | Requires `OPENROUTER_API_KEY`. |
| Local vLLM | Self-hosted on port `30001` via `make vllm`. |
