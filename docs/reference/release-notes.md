# Release notes

## AgentKernelArena 0.1.0

Initial release of AgentKernelArena, a standardized arena for evaluating AI
coding agents on GPU kernel optimization tasks on AMD GPUs.

### Features

- **Multi-agent arena**: run Cursor, Claude Code, and Codex agents through a
  common evaluation pipeline.
- **Multi-model support**: OpenAI, Anthropic, and additional models via
  OpenRouter or a self-hosted vLLM server.
- **Task categories**: `hip2hip`, `cuda2hip`, `triton2triton`,
  `instruction2triton`, `torch2hip`, and `flydsl2flydsl`, with bundled suites
  from gpumode, vLLM, and ROCmBench.
- **Objective metrics**: automated evaluation of compilation, correctness, and
  real GPU performance speedups, combined into a single comparable score.
- **Workspace isolation**: each task runs in a timestamped duplicate workspace
  for reproducibility.
- **Resumable runs**: resume an interrupted run and skip completed tasks with
  `--resume-run` or `--resume-latest`.
- **A/B testing**: run the same task set with and without an agent-side
  capability to measure its real impact.
- **Task validator**: a dedicated agent that runs 10 automated quality checks on
  tasks and emits a `validation_report.yaml`.
- **Visualization dashboard**: a static dashboard for comparing run reports
  across agents and models.
- **ROCm environment auto-detection**: the `Makefile` detects ROCm 6.4, 7.0, or
  7.1 and installs the matching PyTorch build.

### Known limitations

- Agents can hang during task execution and block test completion.
- The published leaderboard is forthcoming; the live demo is illustrative only.
- Task suites for several categories are being expanded toward 100+ tasks each.
