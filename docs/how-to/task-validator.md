# Validate tasks

The **task_validator** agent checks that tasks are correctly configured,
self-contained, and functional. It does not optimize kernels — it audits them.
Use it to validate new tasks before merging and to audit existing tasks before
publishing results to a leaderboard.

## Run the validator

Set the validator as the agent in `config.yaml` and list the tasks to check:

```yaml
agent:
  template: task_validator
tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm
  # - all                     # validate every task
target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

Then run:

```bash
python3 main.py
```

Each task workspace receives a `validation_report.yaml` with per-check results,
and a `validation_summary.yaml` with aggregated statistics is written to the
workspace root.

## Validator configuration

The validator's own backend and limits are set in
`agents/task_validator/agent_config.yaml`:

```yaml
backend: claude_code          # claude_code | codex | cursor
timeout_seconds: 600          # max time per task validation (0 disables the timeout)
python_path: /root/AgentKernelArena/.venv/bin/python3
```

## The 10 checks

| # | Check | What it verifies |
| --- | --- | --- |
| 1 | `config_schema` | All required fields exist with correct types |
| 2 | `source_files_exist` | Every file in `source_file_path` exists |
| 3 | `target_symbols_found` | Every `target_kernel_functions` symbol is defined in source |
| 4 | `compilation` | `compile_command` succeeds (exit 0, within 120s) |
| 5 | `correctness` | `correctness_command` succeeds (exit 0, within 180s) |
| 6 | `performance` | `performance_command` succeeds, if present (within 180s) |
| 7 | `correctness_implementation_review` | The correctness check is meaningful, not trivially passing |
| 8 | `self_contained` | No missing headers/imports or references to external repos/paths |
| 9 | `gpu_hang_check` | No command hangs or times out |
| 10 | `result_template_compatibility` | Output maps to the standard `task_result_template.yaml` |

## Overall status

- **PASS** — all checks passed.
- **WARN** — no failures, but at least one warning (for example, a questionable
  correctness implementation). Acceptable with justification.
- **FAIL** — at least one check failed; the task must be fixed before merging.

## Result template

A validated task's compile → correctness → performance flow must produce results
that populate the standard template:

```yaml
task_name: "<task_type>/<task_name>"
best_optimized_source_file_path:
  - <source files>
best_optimized_kernel_functions:
  - <kernel functions>
pass_compilation: true/false
compilation_error_message: null
pass_correctness: true/false
correctness_error_message: null
base_execution_time: 0.0          # ms
best_optimized_execution_time: 0.0
speedup_ratio: 0.0
optimization_summary: ""
```

For the full author checklist and self-containedness rules, see
`agents/task_validator/README.md` in the repository and
[Add a task](add-task.md).
