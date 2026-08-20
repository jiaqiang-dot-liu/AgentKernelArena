#!/usr/bin/env python3
"""forge-loop measurement driver for the Kimi-K3 KDA linear-attention task.

Targets (PROTECTED, optimized by forge-loop):
    ``fused_recurrent_kda_packed_decode`` (decode hot kernel k007) and
    ``chunk_kda_with_fused_gate`` (prefill chunk-KDA group), in
    ``vllm/models/kimi_k3/amd/ops/third_party/kda/``.

Why this file exists
--------------------
The Arena forge launcher (``agents/forge/launch_agent.py``) prefers a task-shipped
``scripts/forge_driver.py`` and copies it VERBATIM to the workspace root. Without
it the launcher generates a generic shim that delegates to ``arena_task_adapter``,
which does NOT implement ``--profile-run``; forge-loop then spends an LLM agent
authoring a profiling-capable driver before every run. That is exactly what failed
in the source session (``task_preparation_failed``: driver crashed in both
correctness and bench mode). Shipping this file makes the driver preflight pass on
the first check, so task preparation is skipped entirely.

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        runs the task's own ``run_correctness()`` -- the same per-case cosine,
        normalized max-error, shape and finiteness assertions the scored run
        makes against the harness's independent float64 golden -- and prints
        ``allclose: True|False``.

        Do NOT add an ``SNR: <db> dB`` line here. Both metrics are parsed but
        SNR takes precedence (``mcp_server/tools/test.py``), so an SNR line
        silently overrides this verdict and gates the search on an aggregate L2
        statistic while the score is decided by a per-element bound.

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` per case plus one ``mean_ms: <ms>``
        aggregate (arithmetic mean across cases, matching Arena's own evaluator).
        Timing goes through ``task_runner._benchmark_cuda_graph_or_events``, i.e.
        the call is captured once into a CUDA/HIP graph and REPLAYED per timed
        iteration, so forge-loop's graph-replay probe is satisfied for real.

  * Profiling    ``--profile-run``
        builds one case's inputs and launches ONLY the target entry point: a few
        warmups to settle the Triton JIT, a couple of profiled launches, one
        synchronize, exit 0. No timing is printed.

All measurement logic is REUSED from ``scripts/task_runner.py`` (``_configure`` /
``_prepare`` / ``_run`` / ``run_correctness`` / ``_benchmark_cuda_graph_or_events``),
so the driver measures exactly the same op Arena scores. forge-loop never edits it.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path


def _import_task_runner():
    """Import the task's own harness, whether we run from scripts/ or the root.

    In a real run the launcher copies this file to ``<workspace>/forge_driver.py``
    while task_runner stays at ``<workspace>/scripts/task_runner.py``; run in place
    from the task dir, both sit in ``scripts/``. Search both.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "scripts", here, here.parent / "scripts"):
        runner = cand / "task_runner.py"
        if runner.is_file():
            spec = importlib.util.spec_from_file_location("_forge_task_runner", runner)
            module = importlib.util.module_from_spec(spec)
            sys.modules["_forge_task_runner"] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(f"task_runner.py not found near {here}")


def _case_cost(case: dict) -> int:
    p = case["params"]
    return int(p["num_seqs"]) * int(p["seq_len"]) * int(p["num_heads"]) * int(p["head_dim"])


def _run_correctness(tr) -> int:
    """Delegate to the task's own correctness suite; map any failure to allclose.

    The suite owns every criterion the scored run asserts, orders the golden
    before the mutating launch, and prints the per-case numbers itself.
    Re-deriving a metric here would gate the search on a different statistic
    than the one that decides the score, letting a candidate pass this driver
    and still be rejected by the scorer.
    """
    ok = True
    try:
        tr.run_correctness()  # asserts / raises on any failing case
    except Exception as exc:  # noqa: BLE001 - any failure is a correctness fail
        ok = False
        print(f"# correctness failed: {type(exc).__name__}: {str(exc)[:300]}")
    print(f"allclose: {ok}")
    return 0


def _run_bench(tr, warmup: int, iters: int) -> int:
    """Graph-timed bench: one CUDA-graph mean per case (reuses task_runner)."""
    import torch

    results = []
    for case in tr.CASES:
        inp = tr._prepare(case)
        tr._run(inp)                   # settle the Triton JIT before capture
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        ms, meta = tr._benchmark_cuda_graph_or_events(
            lambda i=inp: tr._run(i),
            warmup=max(1, warmup),
            repetition=max(1, iters),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
        )
        if not math.isfinite(ms) or ms <= 0:
            print(f"error: invalid timing for case {case['id']!r}: {ms!r}", file=sys.stderr)
            return 1
        results.append((case["id"], ms, meta))
    if len(results) != len(tr.CASES) or not results:
        print("error: benchmark did not produce every task case", file=sys.stderr)
        return 1
    for case_id, ms, meta in results:
        print(f"case_ms: {case_id} {ms:.6f}")
        print(f"# bench {case_id}: {ms:.6f} ms method={meta.get('benchmark_method')}"
              f" {meta.get('benchmark_fallback_reason', '')}".rstrip())
    means = [ms for _, ms, _ in results]
    print(f"mean_ms: {sum(means) / len(means):.6f}")
    return 0


def _run_profile(tr) -> int:
    """Kernel-only profiling for one case: warm, a few launches, sync, exit 0."""
    import torch

    case = max(tr.CASES, key=_case_cost)
    inp = tr._prepare(case)
    for _ in range(3):                 # settle Triton JIT / autotune selection
        tr._run(inp)
    torch.cuda.synchronize()
    for _ in range(3):                 # profiled launches
        tr._run(inp)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-K3 KDA forge driver")
    parser.add_argument("--mode", default="full")       # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    tr = _import_task_runner()
    tr._configure()   # gfx950 env + workspace-seeded vllm first on sys.path

    import torch
    if not torch.cuda.is_available():
        print("error: a ROCm GPU (gfx950) is required (torch.cuda.is_available() is False)",
              file=sys.stderr)
        return 1

    if args.profile_run:
        return _run_profile(tr)
    if args.bench_mode:
        return _run_bench(tr, args.warmup, args.iters)
    return _run_correctness(tr)


if __name__ == "__main__":
    sys.exit(main())
