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
        src.performance.measure_performance(...) -> "median_ms: <mean-of-cases>"

Reusing Arena's functions means the adapter is task-type aware and not tied to a
single report filename: Arena already searches multiple candidate report files
(``build/performance_report.json``, ``perf/benchmark_results.json``, ...) and
falls back to stdout parsing, per task type. So this one driver works for any
task Arena itself can score — no per-kernel driver and no hardcoded filename.

Environment (set by the Arena forge launcher):
  FORGE_WORKSPACE    workspace dir (where the task commands run).
  FORGE_TASK_CONFIG  path to the task's config.yaml.
  FORGE_ARENA_ROOT   AgentKernelArena repo root (so `src` is importable). The
                     adapter is copied into the workspace at runtime, so __file__
                     cannot locate the repo — the launcher passes this instead.
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

import yaml

WORKSPACE = os.environ.get("FORGE_WORKSPACE") or os.getcwd()
TASK_CONFIG = os.environ.get("FORGE_TASK_CONFIG") or os.path.join(WORKSPACE, "config.yaml")
ARENA_ROOT = os.environ.get("FORGE_ARENA_ROOT", "")

# Make Arena's `src` package importable so we reuse its evaluation logic.
if ARENA_ROOT and ARENA_ROOT not in sys.path:
    sys.path.insert(0, ARENA_ROOT)


def _load_task_config() -> dict:
    with open(TASK_CONFIG, "r") as f:
        return yaml.safe_load(f) or {}


def do_correctness() -> int:
    """Reuse Arena's correctness evaluation; emit the forge allclose contract."""
    try:
        from src.evaluator import evaluate_correctness
    except Exception as e:  # noqa: BLE001
        print("allclose: False")
        print(f"error: cannot import Arena evaluator (FORGE_ARENA_ROOT={ARENA_ROOT!r}): {e}")
        return 0

    task_config = _load_task_config()
    passed, err = evaluate_correctness(Path(WORKSPACE), task_config)
    if passed:
        print("allclose: True")
    else:
        print("allclose: False")
        if err:
            print(err[-1500:])
    return 0


def do_bench() -> int:
    """Reuse Arena's performance measurement; emit the forge median_ms contract."""
    try:
        from src.performance import measure_performance
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot import Arena performance module (FORGE_ARENA_ROOT={ARENA_ROOT!r}): {e}")
        return 1

    task_config = _load_task_config()
    # is_baseline=False: bench whatever kernel is currently in the tree (forge
    # benches the pristine kernel and each edited version the same way).
    cases = measure_performance(Path(WORKSPACE), task_config, is_baseline=False)
    times = [c.execution_time_ms for c in cases
             if getattr(c, "execution_time_ms", None) and c.execution_time_ms > 0]
    if not times:
        print("error: Arena measure_performance returned no usable timing")
        return 1

    # Aggregate the per-case times into the single wall time the forge-loop
    # uses for its keep/revert decision.
    agg = statistics.mean(times)
    print(f"median_ms: {agg:.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="default")   # task harness owns its shapes
    ap.add_argument("--mode", default="full")        # all modes -> task correctness
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _unknown = ap.parse_known_args()

    if args.bench_mode:
        return do_bench()
    return do_correctness()


if __name__ == "__main__":
    raise SystemExit(main())
