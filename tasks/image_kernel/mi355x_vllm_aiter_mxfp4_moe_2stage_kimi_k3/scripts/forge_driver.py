#!/usr/bin/env python3
"""forge-loop measurement driver for the Kimi-K3 aiter mxfp4 MoE 2-stage task.

Target (PROTECTED, optimized by forge-loop): the aiter routed-expert MoE 2-stage
dispatch -- ``fused_moe`` plus ``ops/flydsl/moe_kernels.py`` and ``ops/shuffle.py``.
The FlyDSL/CK compute cores themselves are prebuilt, so the edit surface is the
Python dispatch that selects, configures and feeds them.

Why this file exists
--------------------
The Arena forge launcher (``agents/forge/launch_agent.py``) prefers a task-shipped
``scripts/forge_driver.py`` and copies it VERBATIM to the workspace root. Without
it the launcher generates a generic ``arena_task_adapter`` shim that does NOT
implement ``--profile-run``, so forge-loop spends an LLM agent authoring a
profiling-capable driver before every run -- the exact failure the source session
hit on this kernel (``task_preparation_failed``: the driver crashed in both
correctness and bench mode).

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        runs the task's own ``run_correctness()`` and prints
        ``allclose: True|False``. The suite owns every criterion the scored run
        asserts: the numeric tolerance taken worst-of-N repeats (the stage2
        kernel reduces with atomics, so a single launch is not representative),
        and -- decisively for this task -- ``_assert_tuned_dispatch`` plus the
        check that the correctness token dispatches the same kernel pair as the
        timed token. Routing the op to a different tuned variant is a scoring
        failure, so the search has to see it as one.

        Do NOT add an ``SNR: <db> dB`` line here. Both metrics are parsed but
        SNR takes precedence (``mcp_server/tools/test.py``), so an SNR line
        silently overrides this verdict -- and an SNR derived from the output
        norm cannot express a dispatch mismatch at all.

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        prints ``case_ms: <case_id> <ms>`` per case plus one ``mean_ms: <ms>``
        aggregate (arithmetic mean across cases, matching Arena's evaluator).
        Timing goes through ``task_runner._benchmark_cuda_graph_or_events``: the
        MoE call is captured once into a CUDA/HIP graph and REPLAYED per timed
        iteration, so forge-loop's graph-replay probe is satisfied for real.

  * Profiling    ``--profile-run``
        builds one case's inputs and launches ONLY ``fused_moe``: a few warmups to
        settle the aiter JIT and tuned-config lookup, a couple of profiled
        launches, one synchronize, exit 0. No timing is printed.

All measurement logic is REUSED from ``scripts/task_runner.py`` (``_configure`` /
``_prepare`` / ``_run`` / ``run_correctness`` / ``_benchmark_cuda_graph_or_events``),
so the driver measures exactly the same op Arena scores. forge-loop never edits it.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys
from pathlib import Path


def _import_task_runner():
    """Import the task's own harness, whether we run from scripts/ or the root."""
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
    return int(p["token"]) * int(p["topk"]) * int(p["model_dim"])


def _scored_cases(tr) -> list[dict]:
    """Return only performance-scored cases; correctness-only buckets are gates."""
    cases = [case for case in tr.CASES if not case.get("correctness_only")]
    if not cases:
        raise RuntimeError("task declares no performance-scored cases")
    return cases


_RUNTIME_TENSOR_KEYS = (
    "hidden",
    "w1",
    "w2",
    "w1_scale",
    "w2_scale",
    "topk_weights",
    "topk_ids",
)


def _profile_cache_path(tr, case_id: str) -> Path:
    """Return this workspace's cache for profiler-safe packed runtime inputs."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id or "default")
    return Path(tr.WORKSPACE) / "build" / "profile_input_cache" / f"{safe}.pt"


def _save_profile_inputs(tr, case: dict, inputs: dict) -> None:
    """Atomically refresh packed inputs after an unprofiled canonical benchmark."""
    import torch

    path = _profile_cache_path(tr, case["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "case_id": case["id"],
        "token": inputs["token"],
        "topk": inputs["topk"],
        "tensors": {key: inputs[key] for key in _RUNTIME_TENSOR_KEYS},
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_profile_inputs(tr, case: dict):
    """Load packed inputs without running quantization/shuffle under counters."""
    import torch

    path = _profile_cache_path(tr, case["id"])
    if not path.is_file():
        print(
            f"error: profile input cache is missing for {case['id']!r}; "
            "run --bench-mode before --profile-run",
            file=sys.stderr,
        )
        return None
    try:
        payload = torch.load(path, map_location="cuda")
    except Exception as error:
        print(f"error: unusable profile input cache {path}: {error}", file=sys.stderr)
        return None
    tensors = payload.get("tensors") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("case_id") != case["id"]
        or not isinstance(tensors, dict)
        or any(key not in tensors for key in _RUNTIME_TENSOR_KEYS)
    ):
        print(f"error: invalid profile input cache schema: {path}", file=sys.stderr)
        return None
    aiter = tr._aiter()
    return {
        **tensors,
        "token": int(payload["token"]),
        "topk": int(payload["topk"]),
        "quant_type": aiter.QuantType.per_1x32,
        "activation": tr._activation(aiter),
    }


def _run_correctness(tr) -> int:
    """Delegate to the task's own correctness suite; map any failure to allclose.

    The suite owns every criterion the scored run asserts -- the worst-of-N
    numeric tolerance, the tuned-dispatch guard and the correctness/timed token
    parity check -- and it prints the per-case numbers itself. Re-deriving a
    subset of them here would gate the search on a weaker bar than the one that
    decides the score, letting a candidate pass this driver and still be
    rejected by the scorer.
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
    """Graph-timed bench: one CUDA-graph mean per scored case."""
    import torch

    cases = _scored_cases(tr)
    profile_case_id = max(cases, key=_case_cost)["id"]
    results = []
    for case in cases:
        inputs = tr._prepare(case, correctness=False)
        tr._run(inputs)                # settle the aiter JIT / tuned-config lookup
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        ms, meta = tr._benchmark_cuda_graph_or_events(
            lambda i=inputs: tr._run(i),
            warmup=max(1, warmup),
            repetition=max(1, iters),
            target_ms=bench.get("target_ms", 1.0),
            max_graph_repeats=bench.get("max_graph_repeats", 100),
        )
        if not math.isfinite(ms) or ms <= 0:
            print(f"error: invalid timing for case {case['id']!r}: {ms!r}", file=sys.stderr)
            return 1
        if case["id"] == profile_case_id:
            try:
                _save_profile_inputs(tr, case, inputs)
            except Exception as error:
                print(
                    f"# profile: could not refresh input cache for "
                    f"{case['id']}: {error}",
                    file=sys.stderr,
                )
        results.append((case["id"], ms, meta))
    if len(results) != len(cases):
        print("error: benchmark did not produce every scored case", file=sys.stderr)
        return 1
    for case_id, ms, meta in results:
        print(f"case_ms: {case_id} {ms:.6f}")
        print(f"# bench {case_id}: {ms:.6f} ms method={meta.get('benchmark_method')}"
              f" {meta.get('benchmark_fallback_reason', '')}".rstrip())
    means = [ms for _, ms, _ in results]
    print(f"mean_ms: {sum(means) / len(means):.6f}")
    return 0


def _run_profile(tr) -> int:
    """Profile one scored case using pre-packed, benchmark-refreshed inputs."""
    import torch

    case = max(_scored_cases(tr), key=_case_cost)
    inputs = _load_profile_inputs(tr, case)
    if inputs is None:
        return 1
    for _ in range(3):                 # settle JIT + tuned-config selection
        tr._run(inputs)
    torch.cuda.synchronize()
    for _ in range(3):                 # profiled launches
        tr._run(inputs)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-K3 aiter mxfp4 MoE forge driver")
    parser.add_argument("--mode", default="full")       # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    tr = _import_task_runner()
    tr._configure()   # gfx950 env + workspace-seeded aiter first on sys.path

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
