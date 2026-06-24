#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/moe_fused_gemm (FLAT layout).

The kernel under test is AITER's (vLLM-derived) fused Mixture-of-Experts GEMM
(`fused_moe` / `_fused_moe_kernel`). Each token is routed to `top_k` experts;
the kernel multiplies the token's activations by each selected expert's weight
matrix and (optionally) scales by the routing weight, writing C[M, top_k, N].

The token->expert sort + block padding is produced by the in-source
`moe_align_block_size` (AITER's 4-stage Triton counting-sort).

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output
                    (flydsl-vs-triton comparison added when the FlyDSL target lands)
  --full-benchmark  warmup + cuda-event timing, write build/performance_report.json
"""
import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

SOURCE_FILE = "moe_fused_gemm.py"
ENTRY = "fused_moe"
KERNEL = "_fused_moe_kernel"

# (num_tokens M, hidden K, expert-out N, num_experts E, top_k)
TEST_SHAPES = [
    {"name": "m64_k1024_n512_e8_top2", "M": 64, "K": 1024, "N": 512, "E": 8, "top_k": 2},
    {"name": "m128_k2048_n1024_e8_top2", "M": 128, "K": 2048, "N": 1024, "E": 8, "top_k": 2},
    {"name": "m256_k4096_n512_e16_top1", "M": 256, "K": 4096, "N": 512, "E": 16, "top_k": 1},
    {"name": "m512_k1024_n768_e16_top2", "M": 512, "K": 1024, "N": 768, "E": 16, "top_k": 2},
    {"name": "m1024_k2048_n256_e32_top4", "M": 1024, "K": 2048, "N": 256, "E": 32, "top_k": 4},
]

BLOCK_SIZE_M = 64
SEED = 20260601
WARMUP, ITERS = 10, 50

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("moe_fused_gemm_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, K, N, E, top_k, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    A = (torch.randn((M, K), generator=gen, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    B = (torch.randn((E, N, K), generator=gen, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    # Per-token top_k DISTINCT experts.
    scores = torch.rand((M, E), generator=gen, device=device)
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    topk_weights = torch.rand(
        (M, top_k), generator=gen, device=device, dtype=torch.float32
    ).contiguous()
    return A, B, topk_ids, topk_weights


def _run_kernel(mod, A, B, topk_ids, topk_weights, top_k, mul_routed_weight):
    import torch
    import triton.language as tl

    M, K = A.shape
    E, N, _ = B.shape
    sorted_ids, expert_ids, num_post = mod.moe_align_block_size(topk_ids, BLOCK_SIZE_M, E)
    C = torch.zeros((M, top_k, N), device=A.device, dtype=B.dtype)
    config = {
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    mod.fused_moe(
        A,
        B,
        C,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        mul_routed_weight,
        top_k,
        tl.bfloat16,
        config=config,
    )
    return C


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    assert hasattr(mod, "moe_align_block_size"), "Missing moe_align_block_size"
    print("Compilation: PASS")
    return True


def run_correctness(verbose=True):
    # Runs the Triton kernel on TEST_SHAPES and asserts finite output. No torch
    # comparison: the flydsl-vs-triton comparison is added when the FlyDSL target
    # lands (the Triton kernel is the reference here).
    import torch

    mod = _load_source()
    failures = []
    for shape in TEST_SHAPES:
        for mul in (True, False):
            tag = f"{shape['name']}{'_w' if mul else ''}"
            try:
                A, B, topk_ids, topk_weights = _make_inputs(
                    shape["M"], shape["K"], shape["N"], shape["E"], shape["top_k"]
                )
                C = _run_kernel(mod, A, B, topk_ids, topk_weights, shape["top_k"], mul)
                torch.cuda.synchronize()
                ok = bool(torch.isfinite(C).all().item())
                if verbose:
                    print(
                        f"  {'PASS' if ok else 'FAIL'}: {tag} "
                        f"(M={shape['M']},K={shape['K']},N={shape['N']},E={shape['E']},"
                        f"top_k={shape['top_k']}) out={tuple(C.shape)} finite={ok}"
                    )
                if not ok:
                    failures.append(tag)
            except Exception as e:  # noqa: BLE001
                failures.append(tag)
                if verbose:
                    print(f"  FAIL: {tag} - {str(e)[:160]}")

    total = len(TEST_SHAPES) * 2
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        A, B, topk_ids, topk_weights = _make_inputs(
            shape["M"], shape["K"], shape["N"], shape["E"], shape["top_k"]
        )
        fn = lambda: _run_kernel(  # noqa: E731
            mod, A, B, topk_ids, topk_weights, shape["top_k"], True
        )
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        ms = sorted(times)[len(times) // 2]
        latencies.append(ms)
        flops = 2.0 * shape["M"] * shape["top_k"] * shape["N"] * shape["K"]
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                "params": {k: shape[k] for k in ("M", "K", "N", "E", "top_k")},
                "tflops": flops / (ms * 1e-3) / 1e12,
            }
        )
        if verbose:
            print(f"  {shape['name']}: {ms:.4f} ms")

    build_dir = Path(_HERE) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    geomean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    print(f"Geometric mean latency: {geomean:.4f} ms")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="moe_fused_gemm harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl fused MoE GEMM")
    print("=" * 62)

    if args.compile:
        try:
            run_compile()
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"Compilation: FAIL\nError: {e}")
            sys.exit(1)
    elif args.correctness:
        sys.exit(0 if run_correctness() else 1)
    else:
        run_benchmark()
        sys.exit(0)
