---
myst:
    html_meta:
        "description": "Step-by-step examples for running AgentKernelArena evaluations, A/B testing agent capabilities, validating tasks, and resuming interrupted runs."
        "keywords": "AgentKernelArena, examples, evaluation, A/B test, task validation, GPU kernel, ROCm, HIP, Triton"
---

# AgentKernelArena examples

These walkthroughs assume you've completed the [installation](../install/install.md)
and activated the virtual environment (`make act`).

## Example 1: Evaluate one agent on one task

Run a single HIP task with the Cursor agent.

1. Edit `config.yaml`:

    ```yaml
    agent:
      template: cursor

    tasks:
      - hip2hip/gpumode/GELU

    target_gpu_model: MI300
    log_directory: logs
    workspace_directory_prefix: workspace
    ```

2. Run:

    ```bash
    python main.py
    ```

3. Inspect the result:

    ```text
    workspace_MI300_cursor/
    └── run_<timestamp>/
        └── hip2hip_gpumode_GELU_<timestamp>/
            └── task_result.yaml
    ```

A successful `task_result.yaml` looks like:

```yaml
task_name: hip2hip/gpumode/GELU
pass_compilation: true
pass_correctness: true
base_execution_time: 1.82
best_optimized_execution_time: 1.15
speedup_ratio: 1.58
score: 278.0
```

## Example 2: Run a whole task category

Evaluate Claude Code across all vLLM Triton tasks.

```yaml
agent:
  template: claude_code

tasks:
  - triton2triton/vllm

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
python main.py
```

Each task gets its own workspace under the same `run_<timestamp>/` directory, and
post-processing aggregates them into a run report.

## Example 3: A/B test an agent capability

Measure whether a new Model Context Protocol (MCP) server, skill, or prompt change helps. Run the same
task set twice with distinct run suffixes.

```bash
# Baseline (capability disabled in the agent configuration)
python main.py --run-suffix baseline

# Treatment (capability enabled)
python main.py --run-suffix with_capability
```

Both runs land in `workspace_MI300_<agent>/` with distinct run names. Build the
dashboard and compare them side-by-side:

```bash
cd visualization
python backend/scripts/build_dashboard_data.py --include-workspace-runs
python backend/server.py --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> to compare scores. See
[Visualize and compare runs](../how-to/visualization.md) for details.

## Example 4: Validate a new task

Before merging a new task, run the task_validator agent against it.

```yaml
agent:
  template: task_validator

tasks:
  - hip2hip/others/my_new_kernel

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
python3 main.py
```

Open the generated `validation_report.yaml` in the task workspace. The task must
reach **PASS** (or **WARN** with justification) before merging. See
[Validate tasks](../how-to/task-validator.md) for more information.

## Example 5: Resume an interrupted run

If a long run is interrupted, resume it without repeating completed tasks:

```bash
# Resume the most recent run
python main.py --resume-latest

# Or resume a specific run directory
python main.py --resume-run run_20260617_101500
```
