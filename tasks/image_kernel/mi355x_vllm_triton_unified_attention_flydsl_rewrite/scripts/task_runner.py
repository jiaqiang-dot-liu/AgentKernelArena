#!/usr/bin/env python3
"""Arena harness for the FlyDSL rewrite of AITER's Triton unified attention.

Same operator and same session cases as
``mi355x_vllm_triton_unified_attention``, but the target language is FlyDSL:
``aiter.ops.triton.attention.unified_attention`` is the oracle and the baseline,
and the agent produces ``kernel.py``.

Both states are valid, which is what makes baseline and optimized runs
comparable:

* no (or stubbed) ``kernel.py`` -- measure the Triton source; baseline run;
* a working ``kernel.py``       -- measure the FlyDSL port.

Correctness always validates whichever implementation is under test against an
fp32 torch reference. That is a strictly stronger oracle than comparing the port
to Triton, because it also catches a bug the port would inherit by imitating the
source. ``scripts/forge_driver.py`` additionally gates the port against the live
Triton output on the SNR threshold the rewrite pipeline enforces.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]

REPO_SUBDIR = "aiter"
OP_NAME = SPEC.get("rewrite_contract", {}).get("op_name", "unified_attention_decode")
BUILDER = SPEC.get("rewrite_contract", {}).get(
    "builder_symbol", f"build_{OP_NAME}_module"
)

# Profiling is a single-shape probe, pinned rather than derived from timings so
# the profiled kernel never drifts between runs.
PROFILE_CASE_ID = SPEC.get("profile_case") or CASES[0]["id"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # Keep AITER's JIT artifacts inside the workspace. No AITER_REBUILD: the
    # Triton source is the protected oracle rather than the edit surface, and
    # Triton re-keys its own JIT on source anyway.
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
    repo = WORKSPACE / REPO_SUBDIR
    if repo.is_dir():
        # Import the seeded copy of aiter so the oracle is the reviewed source.
        sys.path.insert(0, str(WORKSPACE))
        os.environ.setdefault(
            "AITER_META_DIR", "/usr/local/lib/python3.12/dist-packages/aiter_meta"
        )
        # aiter.utility.aiter_types resolves aiter_enum.h relative to the parent
        # of the seeded aiter package, i.e. <WORKSPACE>/aiter_meta/csrc/include.
        # Only the aiter package is seeded, so link the installed companion tree
        # rather than copying another 224 MB into every workspace.
        meta_ws = WORKSPACE / "aiter_meta"
        if not meta_ws.exists():
            installed = Path(os.environ["AITER_META_DIR"])
            if installed.is_dir():
                meta_ws.symlink_to(installed, target_is_directory=True)
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<


def _write_report(rows: list[dict]) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def flydsl_builder():
    """Return the FlyDSL port's builder, or None when it is absent or a stub."""
    path = WORKSPACE / "kernel.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ka_flydsl_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ka_flydsl_kernel"] = module
    spec.loader.exec_module(module)
    return getattr(module, BUILDER, None)


def profile_case() -> dict:
    """The single case profiling runs against (see PROFILE_CASE_ID)."""
    for case in CASES:
        if case["id"] == PROFILE_CASE_ID:
            return case
    raise KeyError(f"profile_case {PROFILE_CASE_ID!r} is not in session_cases.json")


def _make(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    p = dict(case["params"])
    ctx_len = min(p["ctx_len"], 128) if correctness else p["ctx_len"]
    num_seqs = p["q_tokens"]
    hq, hkv = p["num_q_heads"], p["num_kv_heads"]
    hs, bs = p["head_size"], p["block_size"]
    pages = (ctx_len + bs - 1) // bs
    num_blocks = num_seqs * pages

    torch.manual_seed(7)
    query = torch.randn((num_seqs, hq, hs), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((num_blocks, bs, hkv, hs), device="cuda", dtype=torch.bfloat16)
    if p["kv_dtype"] == "fp8":
        key = kv.to(torch.float8_e4m3fn)
        value = (kv * 0.7).to(torch.float8_e4m3fn)
    else:
        key = kv
        value = kv * 0.7

    return {
        "cfg": case,
        "query": query,
        "key": key,
        "value": value,
        "out": torch.empty_like(query),
        "cu_seqlens_q": torch.arange(
            num_seqs + 1, device="cuda", dtype=torch.int32
        ),
        "seq_lens": torch.full(
            (num_seqs,), ctx_len, device="cuda", dtype=torch.int32
        ),
        "block_tables": torch.arange(
            num_blocks, device="cuda", dtype=torch.int32
        ).view(num_seqs, pages),
        "one": torch.ones(1, device="cuda", dtype=torch.float32),
        "hq": hq,
        "hkv": hkv,
        "hs": hs,
        "bs": bs,
        "ctx_len": ctx_len,
        "scale": hs**-0.5,
    }


def _run_triton(t: dict):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    unified_attention(
        t["query"], t["key"], t["value"], t["out"], t["cu_seqlens_q"], 1,
        t["seq_lens"], t["ctx_len"], t["scale"], True, (-1, -1),
        t["block_tables"], 0.0, t["one"], t["one"], t["one"],
    )
    return t["out"]


def _make_runner(t: dict, builder):
    """Bind the implementation under test: the FlyDSL port when present."""
    if builder is None:
        return lambda: _run_triton(t), "triton"
    launch = builder(t["hq"], t["hkv"], t["hs"], t["bs"])

    def run():
        launch(
            t["out"], t["query"], t["key"], t["value"], t["block_tables"],
            t["seq_lens"], t["scale"], t["one"], t["one"],
        )
        return t["out"]

    return run, "flydsl"


def _run(inputs: dict):
    run, _ = _make_runner(inputs, flydsl_builder())
    return run()


def _reference(t: dict):
    torch = _torch()
    query = t["query"].float()
    key = t["key"].float()
    value = t["value"].float()
    outputs = []
    for s in range(query.shape[0]):
        blocks = t["block_tables"][s]
        k = key[blocks].reshape(-1, key.shape[2], key.shape[3])[: t["ctx_len"]]
        v = value[blocks].reshape(-1, value.shape[2], value.shape[3])[: t["ctx_len"]]
        ratio = query.shape[1] // k.shape[1]
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
        scores = torch.einsum("hd,khd->hk", query[s], k) * t["scale"]
        outputs.append(torch.einsum("hk,khd->hd", torch.softmax(scores, dim=-1), v))
    return torch.stack(outputs).to(t["out"].dtype)


def run_compile() -> None:
    inputs = _make(CASES[0], correctness=True)
    run, impl = _make_runner(inputs, flydsl_builder())
    run()
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke ({impl}): PASS")


def run_correctness() -> None:
    torch = _torch()
    builder = flydsl_builder()
    impl = "flydsl" if builder is not None else "triton"
    for case in CASES:
        inputs = _make(case, correctness=True)
        # The output buffer is written, never accumulated: poison it so an
        # implementation that leaves rows untouched cannot pass on allocator
        # leftovers.
        inputs["out"].fill_(float("nan"))
        run, _ = _make_runner(inputs, builder)
        got = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, _reference(inputs), atol=0.08, rtol=0.08)
        print(f"correctness PASS ({impl})", case["id"])


def run_performance() -> None:
    builder = flydsl_builder()
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        run, impl = _make_runner(inputs, builder)
        run()
        _torch().cuda.synchronize()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            run,
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
        )
        metadata = {
            **case["params"],
            "implementation": impl,
            "model": case.get("model"),
            "session_breakdown_id": case.get("session_breakdown_id"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        metadata.update(
            {k: v for k, v in bench_meta.items() if k.startswith("benchmark_")}
        )
        rows.append(
            {
                "test_case_id": case["id"],
                "shape": case.get("trace_input_shapes"),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
        print(
            case["id"],
            f"{execution_time_ms:.6f} ms",
            impl,
            bench_meta.get("benchmark_method"),
            bench_meta.get("benchmark_fallback_reason", ""),
        )
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["compile", "correctness", "performance", "manifest"]
    )
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if mode == "compile":
        run_compile()
    elif mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
