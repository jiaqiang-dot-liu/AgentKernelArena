# Canonical performance-benchmark helpers

Benchmark tasks run **standalone** inside per-run workspaces (a copy of the task
folder), so each task ships its own copy of the timing helper instead of importing
a shared module — this keeps tasks self-contained (and keeps the validator's
`self_contained` check happy). To avoid hand-editing ~180 duplicated files, the
helpers live here as the single source of truth and are propagated by a sync script.

## Files
- `performance_utils_pytest.py` — full helper file, copied verbatim to every
  `tasks/*/rocmbench/**/performance_utils_pytest.py` (62 copies).
- `vllm_cuda_graph_block.py` — the two helper functions
  (`_measure_cuda_event_fallback`, `_benchmark_cuda_graph_or_events`), injected
  between the `# >>> AKA-GENERATED ... >>>` / `# <<< AKA-GENERATED <<<` markers in
  every `tasks/triton2triton/vllm/*/scripts/task_runner.py` (118 copies).

## Workflow
1. Edit the canonical file(s) here.
2. Run `make sync-perf-helpers` (or `python tools/sync_perf_helpers.py`).
3. Commit the canonical change **and** the regenerated task copies together.

`make check-perf-helpers` (or `python tools/sync_perf_helpers.py --check`) reports
drift without writing and exits non-zero — suitable for CI.

See `docs/reference/benchmark-methodology.md` for the timing methodology itself.
