# Canonical performance-benchmark helpers

The real timing helpers live here as the single source of truth. Committed task
sources keep only small stubs/markers, and `setup_workspace()` materializes the
canonical helper code into each per-run workspace before benchmark commands run.
This keeps generated code out of normal development diffs while preserving
self-contained task workspaces at runtime.

## Files
- `performance_utils_pytest.py` - full rocmbench pytest timing helper. In task
  sources, `tasks/*/rocmbench/**/performance_utils_pytest.py` is a stub; in run
  workspaces, it is replaced with this file.
- `vllm_cuda_graph_block.py` - the two vLLM helper functions
  (`_measure_cuda_event_fallback`, `_benchmark_cuda_graph_or_events`). In task
  sources, the `# >>> AKA-GENERATED ... >>>` / `# <<< AKA-GENERATED <<<` region
  contains a stub block; in run workspaces, that region is replaced with these
  functions.

## Workflow
1. Edit the canonical file(s) here.
2. Run `make check-perf-helpers` before pushing. This verifies that committed
   task stubs/markers are valid.
3. If you add a new task or change marker/stub structure, run
   `make sync-perf-helpers` to refresh the committed stubs.

Do not hand-edit the committed perf-helper stubs in task directories. Runtime
workspaces are materialized from `src/tools/perf/` by the framework.

Use `make materialize-perf-workspace WORKSPACE=...` to inject helpers into an
existing copied task workspace, or `make materialize-perf-task TASK=tasks/...`
to copy a task to `/tmp/aka-materialized-tasks` and inject helpers there.

See `docs/reference/benchmark-methodology.md` for the timing methodology itself.
