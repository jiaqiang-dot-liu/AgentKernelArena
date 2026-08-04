#!/usr/bin/env python3
"""Harness for vLLM's unquantized Triton fused-MoE expert GEMM
``fused_moe_kernel`` (``vllm/model_executor/layers/fused_moe/fused_moe.py``).

The kernel is loaded from the editable workspace copy of the in-image source tree
so an optimizing agent's edits to fused_moe.py (and to ``configs/``) take effect;
Triton re-keys its JIT on the source, so no explicit rebuild step is needed.

This is the BF16 path, not the int4 ``fused_moe_kernel_gptq_awq`` covered by
``mi355x_vllm_triton_fused_moe_gptq_awq``.
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

REPO_SUBDIR = "vllm_fused_moe"
KERNEL_FILE = "fused_moe.py"
EDIT_MODULE_NAME = "vllm.model_executor.layers.fused_moe._ka_fused_moe"

# Profiling is a single-shape probe, pinned rather than derived from timings so
# the profiled kernel never drifts between runs. The decode shape is the
# session's hot entry at 12.5% of GPU time. Correctness and performance still
# sweep every case in CASES.
PROFILE_CASE_ID = SPEC.get("profile_case") or CASES[0]["id"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
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


def _load_kernel_module():
    # fused_moe.py runs direct_register_custom_op at import; suppress it so loading
    # the editable copy does not clash with the already-registered installed copy.
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa: F401
    import vllm.utils.torch_utils as torch_utils

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    if not path.is_file():
        raise RuntimeError(f"seeded kernel source not found: {path}")
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    original = torch_utils.direct_register_custom_op
    torch_utils.direct_register_custom_op = lambda *a, **k: None
    try:
        spec.loader.exec_module(module)
    finally:
        torch_utils.direct_register_custom_op = original
    return module


def profile_case() -> dict:
    """The single case profiling runs against (see PROFILE_CASE_ID)."""
    for case in CASES:
        if case["id"] == PROFILE_CASE_ID:
            return case
    raise KeyError(
        f"profile_case {PROFILE_CASE_ID!r} is not present in session_cases.json"
    )


def _make(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    p = dict(case["params"])
    if correctness:
        # The reference is an explicit per-token, per-expert loop in float32, so
        # shrink every dimension to keep it tractable while still exercising
        # routing, both GEMMs and the gated activation.
        tokens = min(p["tokens"], 16)
        num_experts = min(p["num_experts"], 8)
        topk = min(p["topk"], 2)
        hidden = min(p["hidden"], 512)
        inter = min(p["inter"], 128)
    else:
        tokens = p["tokens"]
        num_experts = p["num_experts"]
        topk = p["topk"]
        hidden = p["hidden"]
        inter = p["inter"]

    torch.manual_seed(31)
    module = _load_kernel_module()

    # Normalize weights by 1/sqrt(K) so the MoE output stays O(1); a fixed atol
    # against a tiny output would let a broken kernel pass.
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(
        (num_experts, 2 * inter, hidden), device="cuda", dtype=torch.bfloat16
    ) / (hidden**0.5)
    w2 = torch.randn(
        (num_experts, hidden, inter), device="cuda", dtype=torch.bfloat16
    ) / (inter**0.5)

    # Router: Gemma4Router picks top-k over a softmax of fp32 logits.
    logits = torch.randn((tokens, num_experts), device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(logits, dim=-1), topk, dim=-1
    )

    return {
        "cfg": case,
        "module": module,
        "x": x,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights.to(torch.float32).contiguous(),
        "topk_ids": topk_ids.to(torch.int32).contiguous(),
        "num_experts": num_experts,
        "inter": inter,
        "activation": p["activation"],
    }


def _run(inputs: dict):
    return inputs["module"].fused_experts_impl(
        hidden_states=inputs["x"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        activation=inputs["activation"],
        global_num_experts=inputs["num_experts"],
    )


def _reference(inputs: dict):
    torch = _torch()
    import torch.nn.functional as F

    x = inputs["x"].float()  # [M, hidden]
    w1 = inputs["w1"].float()  # [E, 2*inter, hidden]
    w2 = inputs["w2"].float()  # [E, hidden, inter]
    tw = inputs["topk_weights"].float()  # [M, topk]
    tid = inputs["topk_ids"]  # [M, topk]
    inter = inputs["inter"]
    assert inputs["activation"] == "gelu_tanh", inputs["activation"]

    M, hidden = x.shape
    out = torch.zeros((M, hidden), device="cuda", dtype=torch.float32)
    for m in range(M):
        acc = torch.zeros((hidden,), device="cuda", dtype=torch.float32)
        for j in range(tid.shape[1]):
            e = int(tid[m, j])
            gate_up = x[m] @ w1[e].t()  # [2*inter]
            # gelu_tanh_and_mul: gelu_tanh(first half) * second half
            act = F.gelu(gate_up[:inter], approximate="tanh") * gate_up[inter:]
            acc = acc + float(tw[m, j]) * (act @ w2[e].t())
        out[m] = acc
    return out.to(torch.bfloat16)


def run_compile() -> None:
    inputs = _make(CASES[0], correctness=True)
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case, correctness=True)
        got = _run(inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(got).all(), case["id"]
        torch.testing.assert_close(got, _reference(inputs), atol=0.03, rtol=0.03)
        print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        _run(inputs)
        _torch().cuda.synchronize()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=10,
            repetition=100,
            # These cases run 0.16-1.0 ms per call, so the default target_ms=1.0
            # collapses the captured repeat count toward 1 and stops amortizing
            # the fixed graph-replay overhead - measured run-to-run swings of
            # 1.7x, and up to 30x on a cold first run. 10 ms keeps the repeat
            # count comfortably above 1 for every case.
            target_ms=10.0,
            max_graph_repeats=1000,
        )
        metadata = {
            **case["params"],
            "model": case.get("model"),
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
