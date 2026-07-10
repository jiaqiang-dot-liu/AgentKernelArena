.. meta::
   :description: Learn what AgentKernelArena is — a standardized AMD evaluation arena for measuring AI coding agent performance on GPU kernel optimization tasks on ROCm.
   :keywords: AgentKernelArena, GPU kernel, optimization, AI agents, ROCm, HIP, Triton, LLM, AMD, evaluation

*************************
What is AgentKernelArena?
*************************

AgentKernelArena is a standardized evaluation arena, built by AMD, that measures
how well AI coding agents perform on real GPU kernel optimization tasks. It runs
LLM-powered agents side-by-side on the same tasks in isolated workspaces and
scores them with objective, reproducible metrics for compilation, correctness,
and GPU performance.

The AgentKernelArena source code is hosted in the
`AMD-AGI/AgentKernelArena <https://github.com/AMD-AGI/AgentKernelArena>`_ GitHub
repository.

What AgentKernelArena does
==========================

AgentKernelArena gives each agent the same kernel task, runs it in a siloed,
timestamped workspace, then evaluates the result through a common pipeline:

* **Compile**: Build the agent's kernel and confirm it compiles cleanly.
* **Validate**: Run the task's correctness check against a reference.
* **Profile**: Measure GPU execution time and compute speedup over a baseline.
* **Score**: Combine the results into a single comparable score.

Key features
============

* **Multi-agent arena**: Cursor Agent, Claude Code, Codex, and custom agent
  integrations.
* **Multi-model support**: OpenAI, Anthropic, and other models through OpenRouter or
  a local vLLM server.
* **Docker-first runtime**: Benchmark runs execute inside pinned ROCm/SGLang
  Docker images selected from the target GPU architecture.
* **Task categories**: HIP (``hip2hip``), CUDA-to-HIP (``cuda2hip``), Triton
  (``triton2triton``, ``instruction2triton``), Torch-to-HIP (``torch2hip``), and
  FlyDSL (``flydsl2flydsl``), plus repository-level tasks.
* **Objective metrics**: Automated compilation, correctness, and real GPU
  performance speedups.
* **Benchmark methodology metadata**: Timing method metadata is recorded for
  baseline and optimized runs so mixed-method comparisons are visible.
* **Workspace isolation**: Each task runs in its own timestamped workspace for
  reproducibility.
* **Multi-GPU parallel runs**: On multi-GPU servers, start one isolated Docker
  worker per GPU and keep idle GPUs busy with a shared task queue.
* **A/B testing**: Run the same task set with and without a new Model Context
  Protocol (MCP) server, skill, prompt, or tool to measure its real impact.
* **Task validator**: An agent that runs 10 automated checks on task quality
  before tasks enter the leaderboard.
* **Visualization dashboard**: A static dashboard for comparing run reports.

Use cases
=========

* Compare AI coding agents head-to-head on GPU kernel optimization.
* Rank models and agent configurations on a standardized leaderboard.
* A/B test whether a new agent capability (MCP server, skill, prompt) improves
  outcomes under identical conditions.
* Run large task suites faster by distributing tasks across all GPUs on a
  multi-GPU server.
* Curate and validate a high-quality, self-contained GPU kernel task suite.
