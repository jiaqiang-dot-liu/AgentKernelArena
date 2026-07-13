---
myst:
    html_meta:
        "description": "Complete reference for AgentKernelArena configuration: run config.yaml schema, task config fields, CLI flags, scoring formula, and the agent registry."
        "keywords": "AgentKernelArena, API reference, config.yaml, CLI flags, scoring, agent registry, ROCm, GPU kernel"
---

# AgentKernelArena configuration and API reference

This topic documents the run configuration (`config.yaml`), the per-task
configuration, the command-line flags, the scoring formula, and the agent
registry.

## Run configuration (`config.yaml`)

The root `config.yaml` defines a single evaluation run.

| Field | Type | Description |
| --- | --- | --- |
| `agent.template` | string | Agent to run. One of the [supported agents](../how-to/agents.md#supported-agents). |
| `tasks` | list of strings | Task selectors relative to `tasks/`. Use `all` for every task, a category prefix for a group, or a full path for a single task. |
| `target_gpu_model` | string | Target GPU model, for example `MI300` or `MI355X`. Used to select the Docker image architecture, set `PYTORCH_ROCM_ARCH`, and name the workspace. |
| `log_directory` | string | Directory for run logs. |
| `workspace_directory_prefix` | string | Prefix for the workspace directory. The full name is `<prefix>_<gpu>_<agent>`. |

Example:

```yaml
agent:
  template: cursor

tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

## Command-line flags

The in-container `main.py` entrypoint accepts these flags:

| Flag | Description |
| --- | --- |
| `--config_name <file>` | Config file to load (default `config.yaml`). Use separate files to keep multiple task sets in one folder |
| `--run-suffix <suffix>` | Suffix appended to the run directory name (letters, numbers, `.`, `_`, `-` only). Useful for labeling A/B runs |
| `--resume-run <run_dir>` | Resume a specific run directory, skipping completed tasks |
| `--resume-latest` | Resume the most recent run in the workspace |

These flags are passed to the in-container entrypoint through `make docker-run`
or `make docker-parallel-run` (`CONFIG=` sets `--config_name`; `RUN_ARGS=`
forwards the rest):

```bash
make docker-run CONFIG=config_triton.yaml RUN_ARGS="--run-suffix with_mcp"
make docker-parallel-run CONFIG=config_triton.yaml GPU_IDS=0,1 RUN_ARGS="--run-suffix with_mcp_parallel"
```

The following flags are internal implementation details used by
`docker-parallel-run` and should not be passed manually in normal use:

| Flag | Description |
| --- | --- |
| `--run-name <run_dir>` | Explicit run directory shared by parallel init, workers, and post-processing |
| `--parallel-init` | Initialize the shared `.parallel/` queue |
| `--parallel-worker` | Claim and execute tasks from the shared queue |
| `--worker-id <id>` | Worker identifier used in queue descriptors and logs |
| `--postprocess-only` | Aggregate results once after all workers finish |

## Docker runner Make targets

The following Make targets are available for running evaluations.

| Target | Description |
| --- | --- |
| `make docker-run CONFIG=config.yaml` | Run tasks serially in one Docker container |
| `make docker-parallel-run CONFIG=config.yaml GPU_IDS=0,1` | Run one Docker worker per listed GPU, using a shared dynamic task queue |
| `make docker-smoke` | Verify Docker, ROCm runtime visibility, Python imports, and GPU access |
| `make docker-check-agents` | Verify host agent CLI login reuse inside Docker |
| `make docker-shell` | Open an interactive shell in the benchmark Docker runtime |

`docker-parallel-run` accepts these environment variables:

| Variable | Description |
| --- | --- |
| `GPU_IDS` | Comma- or space-separated host GPU IDs. If omitted, the runner uses `rocm-smi --showid` |
| `RUN_ARGS` | Additional `main.py` flags, such as `--run-suffix`, `--resume-run`, or `--resume-latest` |
| `AKA_LOGICAL_GPU` | Logical GPU index inside a masked worker container. Defaults to `0` and normally should not be changed |

Each parallel worker sets `ROCR_VISIBLE_DEVICES` to the host GPU ID and sets
`HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` to the
logical GPU index inside the masked container. See
[Run tasks in parallel across multiple GPUs](../how-to/parallel-run.md) for the
full scheduling model.

## Task configuration

Each task is defined by a `config.yaml` in its directory. Command fields are
*lists*.

For isolated-kernel tasks (`hip2hip`, `cuda2hip`, `triton2triton`,
`instruction2triton`, `torch2hip`, and `flydsl2flydsl`):

| Field | Required | Description |
| --- | --- | --- |
| `source_file_path` | Yes | Source files containing the kernel, relative to the task root |
| `target_kernel_functions` | Yes | Kernel function names that must be defined in the source |
| `compile_command` | Yes | Command(s) to compile or build-check |
| `correctness_command` | Yes | Command(s) to validate correctness |
| `task_type` | Yes | One of `hip2hip`, `cuda2hip`, `triton2triton`, `instruction2triton`, `torch2hip`, or `flydsl2flydsl` |
| `performance_command` | No | Command(s) to measure performance |
| `task_result_template` | No | Override the result template (`null` = default) |
| `prompt.source_code` | No | Override the prompt's source-code section |
| `prompt.instructions` | No | Custom prompt instructions |
| `prompt.cheatsheet` | No | Reference/cheatsheet content for the prompt |

For repository-level tasks (`task_type: repository`):

| Field | Required | Description |
| --- | --- | --- |
| `repo_url` | Yes | Upstream repository to clone for the task |
| `task_type` | Yes | Must be `repository` |
| `repository_language` | Yes | Primary optimization stack, for example `hip` or `triton` |
| `compile_command` | Yes | Command(s) to compile or build-check |
| `correctness_command` | Yes | Command(s) to validate correctness |
| `performance_command` | No | Command(s) to measure performance |
| `post_clone_install` | No | Setup command(s) to run after cloning the upstream repository |
| `post_clone_install_mode` | No | Controls when `post_clone_install` runs, for example `every_setup` |
| `source_file_path` | No | Optional target source-file hints, relative to the cloned repository root |
| `target_kernel_functions` | No | Optional target function or kernel-symbol hints |
| `prompt.instructions` | No | Custom prompt instructions |
| `prompt.cheatsheet` | No | Reference/cheatsheet content for the prompt |

See [Add a task](../how-to/add-task.md) for layout and authoring rules.

## Result schema (`task_result.yaml`)

Each task produces a `task_result.yaml` in its workspace:

| Field | Description |
| --- | --- |
| `task_name` | `<task_type>/<task_name>` |
| `pass_compilation` | Whether the optimized kernel compiled |
| `compilation_error_message` | Error text if compilation failed, else `null` |
| `pass_correctness` | Whether correctness passed |
| `correctness_error_message` | Error text if correctness failed, else `null` |
| `base_execution_time` | Baseline runtime in ms |
| `best_optimized_execution_time` | Best optimized runtime in ms |
| `speedup_ratio` | Speedup over baseline |
| `baseline_benchmark_methods` | Timing methods observed while measuring the baseline |
| `optimized_benchmark_methods` | Timing methods observed while measuring the optimized kernel |
| `benchmark_method_consistent` | Whether baseline and optimized timing methods matched |
| `valid_baseline_cases` | Number of baseline test cases with usable timing results |
| `valid_optimized_cases` | Number of optimized test cases with usable timing results |
| `speedup_calculation_error_message` | Error text if speedup could not be calculated, else `null` |
| `optimization_summary` | Free-text summary from the agent |
| `score` | Computed score (see below) |

## Scoring

The score is the sum of three components:

| Component | Points | Condition |
| --- | --- | --- |
| Compilation | `20` | The kernel compiles successfully |
| Correctness | `100` | The kernel passes the correctness check |
| Speedup | `speedup_ratio × 100` | Added only when compilation *and* correctness pass |

The rules, expressed as the framework applies them:

- Compilation fails → score `0`.
- Compilation passes, correctness fails → score `20`.
- Both pass → `120 + speedup_ratio × 100`.

**Example**: A kernel that compiles (`20`), is correct (`100`), and achieves a
`1.58×` speedup scores `20 + 100 + 158 = 278`.

The speedup used for scoring prefers the explicit `speedup_ratio` written by the
evaluator (which weights each test case equally for multi-testcase tasks) and
falls back to `base_execution_time / best_optimized_execution_time` when an
explicit ratio is not present.

This is the default scoring scheme; you can define your own in `src/score.py`.

## Agent registry

Agents register themselves into a shared registry with the `register_agent`
decorator, and the framework loads only the selected agent:

```python
from agents import register_agent

@register_agent("your_agent")
def launch_agent(eval_config, task_config_dir, workspace):
    ...
    return result
```

The selectable agent names are defined by the `AgentType` enum in
`src/module_registration.py`. See [Configure agents and models](../how-to/agents.md)
for the integration steps.
