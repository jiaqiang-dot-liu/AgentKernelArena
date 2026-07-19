# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Generic KernelForge driver that reuses an AgentKernelArena task's own
correctness and performance measurement.

Instead of re-implementing (or hardcoding) precision/perf parsing per kernel,
this adapter calls Arena's OWN evaluation functions and translates their results
into the KernelForge driver contract consumed by
``kernel_agents.mcp_server.tools.{test,bench}``:

  * correctness mode (default / --mode smoke|stability|determinism):
        src.evaluator.evaluate_correctness(...) -> "allclose: True/False"
  * bench mode (--bench-mode):
        src.performance.measure_performance(...) -> "mean_ms: <mean-of-cases>"
        (arithmetic mean across test cases, matching Arena's own evaluator
        aggregation — deliberately a MEAN, not a median, so the label is honest)

Reusing Arena's functions means the adapter is task-type aware and not tied to a
single report filename: Arena already searches multiple candidate report files
(``build/performance_report.json``, ``perf/benchmark_results.json``, ...) and
falls back to stdout parsing, per task type. So this one driver works for any
task Arena itself can score — no per-kernel driver and no hardcoded filename.

This module is a LIBRARY: it takes the task paths (workspace / task config /
Arena repo root) as explicit arguments — NO environment variables. The Arena
forge launcher generates a tiny ``forge_driver.py`` shim with those paths baked
in that calls :func:`run`.
"""

import argparse
import statistics
import sys
from pathlib import Path

import yaml


def _load_task_config(task_config: str) -> dict:
    with open(task_config, "r") as f:
        return yaml.safe_load(f) or {}


def do_correctness(workspace: str, task_config: str, arena_root: str = "") -> int:
    """Reuse Arena's correctness evaluation; emit the forge allclose contract."""
    try:
        from src.evaluator import evaluate_correctness
    except Exception as e:  # noqa: BLE001
        print("allclose: False")
        print(f"error: cannot import Arena evaluator (arena_root={arena_root!r}): {e}")
        return 0

    task_config_data = _load_task_config(task_config)
    passed, err = evaluate_correctness(Path(workspace), task_config_data)
    if passed:
        print("allclose: True")
    else:
        print("allclose: False")
        if err:
            print(err[-1500:])
    return 0


def do_bench(workspace: str, task_config: str, arena_root: str = "") -> int:
    """Reuse Arena's performance measurement; emit the forge median_ms contract."""
    try:
        from src.performance import measure_performance
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot import Arena performance module (arena_root={arena_root!r}): {e}")
        return 1

    task_config_data = _load_task_config(task_config)
    # is_baseline=False: bench whatever kernel is currently in the tree (forge
    # benches the pristine kernel and each edited version the same way).
    cases = measure_performance(Path(workspace), task_config_data, is_baseline=False)
    times = [c.execution_time_ms for c in cases
             if getattr(c, "execution_time_ms", None) and c.execution_time_ms > 0]
    if not times:
        print("error: Arena measure_performance returned no usable timing")
        return 1

    # Aggregate the per-case times into the single wall time the forge-loop uses
    # for its keep/revert decision. This is the ARITHMETIC MEAN across cases,
    # matching Arena's own evaluator (sum/len over cases), so forge optimizes the
    # same quantity Arena scores. Reported as ``mean_ms:`` (not ``median_ms:``)
    # so the label reflects the statistic actually computed.
    agg = statistics.mean(times)
    print(f"mean_ms: {agg:.6f}")
    return 0


def run(workspace: str, task_config: str, arena_root: str = "", argv: list[str] | None = None) -> int:
    """Driver entry point. All task paths are passed explicitly (no env).

    Args:
        workspace: dir where the task commands run (the kernel lives here).
        task_config: path to the task's config.yaml.
        arena_root: AgentKernelArena repo root, prepended to sys.path so Arena's
            ``src`` package (evaluator / performance) is importable.
        argv: driver args from KernelForge (``--shape`` / ``--mode`` /
            ``--bench-mode`` / ...); defaults to ``sys.argv[1:]``.
    """
    if arena_root and arena_root not in sys.path:
        sys.path.insert(0, arena_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="default")   # task harness owns its shapes
    ap.add_argument("--mode", default="full")        # all modes -> task correctness
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _unknown = ap.parse_known_args(argv)

    if args.bench_mode:
        return do_bench(workspace, task_config, arena_root)
    return do_correctness(workspace, task_config, arena_root)
