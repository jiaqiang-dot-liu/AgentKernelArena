.. meta::
   :description: AgentKernelArena is a standardized arena for evaluating AI coding agents on real GPU kernel optimization tasks on AMD GPUs.
   :keywords: AgentKernelArena, ROCm, GPU, kernel, optimization, agent, LLM, HIP, Triton, benchmark, AMD

******************************
AgentKernelArena documentation
******************************

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

* **Compile** -- Build the agent's kernel and confirm it compiles cleanly.
* **Validate** -- Run the task's correctness check against a reference.
* **Profile** -- Measure GPU execution time and compute speedup over a baseline.
* **Score** -- Combine the results into a single comparable score.

Key features
============

* **Multi-agent arena**: Cursor, Claude Code, Codex, SWE-agent, OpenEvolve
  (GEAK), GEAK HIP, single-LLM-call, and custom agents.
* **Multi-model support**: OpenAI, Anthropic, and other models via OpenRouter or
  a local vLLM server.
* **Task categories**: HIP (``hip2hip``), CUDA-to-HIP (``cuda2hip``), Triton
  (``triton2triton``, ``instruction2triton``), Torch-to-HIP (``torch2hip``), and
  FlyDSL (``flydsl2flydsl``).
* **Objective metrics**: automated compilation, correctness, and real GPU
  performance speedups.
* **Workspace isolation**: each task runs in its own timestamped workspace for
  reproducibility.
* **A/B testing**: run the same task set with and without a new MCP server,
  skill, prompt, or tool to measure its real impact.
* **Task validator**: an agent that runs 10 automated checks on task quality
  before tasks enter the leaderboard.
* **Visualization dashboard**: a static dashboard for comparing run reports.

Use cases
=========

* Compare AI coding agents head-to-head on GPU kernel optimization.
* Rank models and agent configurations on a standardized leaderboard.
* A/B test whether a new agent capability (MCP server, skill, prompt) improves
  outcomes under identical conditions.
* Curate and validate a high-quality, self-contained GPU kernel task suite.

Documentation
=============

The AgentKernelArena documentation is organized into the following categories.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install AgentKernelArena <install/install>`

   .. grid-item-card:: Reference

      * :doc:`Release notes <reference/release-notes>`
      * :doc:`Compatibility matrix <reference/compatibility-matrix>`
      * :doc:`Configuration and API reference <reference/api-reference>`

   .. grid-item-card:: How to

      * :doc:`Run an evaluation <how-to/run-evaluation>`
      * :doc:`Configure agents and models <how-to/agents>`
      * :doc:`Add a task <how-to/add-task>`
      * :doc:`Validate tasks <how-to/task-validator>`
      * :doc:`Visualize and compare runs <how-to/visualization>`

   .. grid-item-card:: Examples

      * :doc:`Examples <examples/examples>`

   .. grid-item-card:: About

      * :doc:`License <about/license>`

To contribute to the documentation, see the
`AgentKernelArena GitHub repository <https://github.com/AMD-AGI/AgentKernelArena>`_.

AgentKernelArena is released under the Apache 2.0 license. For details, see the
:doc:`License <about/license>` page.
