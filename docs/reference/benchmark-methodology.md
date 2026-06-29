# Performance benchmark methodology

Performance timing in AgentKernelArena is **not uniform across task categories**.
There are two timing implementations; both use 10 warmup iterations and average the
measured samples, but they differ in how samples are collected.

The shared Triton timing helpers are maintained in `src/tools/perf/`; committed
task sources contain stubs/markers, and `setup_workspace()` materializes the
canonical helpers into each run workspace. See the
[`src/tools/perf/README.md`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/tools/perf/README.md)
helper workflow for maintenance details.

## 1. CUDA-graph timing (Triton tasks)

Used by:

- `triton2triton/vllm/*/scripts/task_runner.py` (118 tasks) — `_benchmark_cuda_graph_or_events()`
- `*/rocmbench/*/performance_utils_pytest.py` (62 tasks, `instruction2triton` + `triton2triton`) — `_measure_times()`

Method:

- 10 warmup iterations.
- A CUDA graph is captured that replays the kernel `n_repeat` times, where `n_repeat`
  is chosen adaptively so one graph replay takes about `target_ms` (≈1 ms), capped at
  `max_graph_repeats`. This amortizes/eliminates per-launch host overhead.
- The graph replay is timed `repetition` (100) times; each sample is
  `elapsed_time / n_repeat` (per-call time). The reported runtime is the average of
  the 100 samples. Metadata records `benchmark_method: cuda_graph` and
  `benchmark_samples`.
- If CUDA-graph capture fails, it falls back to per-call CUDA-event timing
  (`benchmark_method: cuda_event_fallback`).

## 2. CUDA-event timing (HIP tasks)

Used by:

- `hip2hip/gpumode/*/eval_tools/cal_kernel_perf.py` — `cal_hip_latency()`

Method:

- 10 warmup iterations, then 100 measured iterations timed with a single
  start/stop CUDA-event pair; reported runtime is `elapsed_time / 100`.
- `hip2hip/others/ball_query` uses its own `scripts/task_runner.py` event timing
  (correctness was strengthened separately; timing is unchanged).

## Notes

- Both paths satisfy the validator's "10 warmup / 100 measured / averaged" expectation.
  The CUDA-graph path is accepted as an equivalent (warmup + 100 averaged graph-replay
  samples); see `agents/task_validator/validation_prompt.py` Check 6.
- Speedup is computed by the harness (`src/evaluator.py`): it runs the same
  `performance_command` against the original kernel (baseline) and the optimized kernel,
  then `speedup_ratio = base / optimized`. Comparing across categories is only valid
  when both sides used the same timing method — see `benchmark_method` in the results.
- Unifying both categories onto a single shared implementation is tracked as a
  follow-up (extract a shared `src/` benchmark module, then converge methodology).
