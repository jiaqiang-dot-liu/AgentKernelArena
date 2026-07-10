---
myst:
    html_meta:
        "description": "Release notes for AgentKernelArena, covering new features, known limitations, supported GPU task categories, and agent integration changes for each release."
        "keywords": "AgentKernelArena, release notes, changelog, ROCm, GPU kernel, HIP, Triton, agents, evaluation"
---

# AgentKernelArena release notes

## AgentKernelArena 0.1.0

Initial release of AgentKernelArena, a standardized arena for evaluating AI
coding agents on GPU kernel optimization tasks on AMD GPUs.

### Features

AgentKernelArena 0.1.0 includes the following features.

- **Multi-agent arena**: run Cursor, Claude Code, and Codex agents through a
  common evaluation pipeline.
- **Multi-model support**: OpenAI, Anthropic, and additional models through
  OpenRouter or a self-hosted vLLM server.
- **Task categories**: `hip2hip`, `cuda2hip`, `triton2triton`,
  `instruction2triton`, `torch2hip`, `flydsl2flydsl`, and repository-level
  tasks, with bundled suites from gpumode, vLLM, and ROCmBench.
- **Docker-first workflow**: run evaluations inside pinned ROCm/SGLang images
  selected from the configured target GPU architecture.
- **Objective metrics**: automated evaluation of compilation, correctness, and
  real GPU performance speedups, combined into a single comparable score.
- **Benchmark methodology metadata**: record timing methods for baseline and
  optimized runs so mixed-method comparisons are visible.
- **Workspace isolation**: each task runs in a timestamped duplicate workspace
  for reproducibility.
- **Performance helper materialization**: keep shared Triton timing helpers in
  `src/tools/perf/` and materialize them into task workspaces at runtime.
- **Resumable runs**: resume an interrupted run and skip completed tasks with
  `--resume-run` or `--resume-latest`.
- **Multi-GPU parallel runs**: use `make docker-parallel-run` to start one
  isolated Docker worker per GPU, claim tasks from a shared queue, and aggregate
  results once after all workers finish.
- **A/B testing**: run the same task set with and without an agent-side
  capability to measure its real impact.
- **Task validator**: a dedicated agent that runs 10 automated quality checks on
  tasks and emits a `validation_report.yaml`.
- **Visualization dashboard**: a static dashboard for comparing run reports
  across agents and models.

### Known limitations

The following limitations are present in this release.

- A single agent task can still run until its configured timeout, but in
  `docker-parallel-run` other GPU workers continue processing the remaining
  queue.
- The published leaderboard is forthcoming; the live demo is illustrative only.
- Task suites for several categories are being expanded toward 100+ tasks each.
