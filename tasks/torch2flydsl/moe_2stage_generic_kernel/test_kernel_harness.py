#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl moe_2stage_generic task.

The op is the generic two-stage fused MoE (bf16, ``QuantType.No``): a softmax
top-k router, a grouped gate/up GEMM with silu-gated activation, a bf16
intermediate, a grouped down GEMM, and a weighted top-k combine; GEMMs
accumulate in fp32 and the output is bf16.

Correctness compares the (b)-faithful PyTorch reference in model.py against the
real AMD runtime op (``aiter.fused_moe`` with ``QuantType.No``) over the same
routing and weights (the harness only adds the host-side weight shuffle the CK
kernel needs). The gate is the normalized worst-element error
``max|ref-gt| / max|ref| <= TOL``. When the FlyDSL kernel.py exists it is
additionally validated against the reference.

Modes:
  --correctness     compare the model.py reference to the aiter ground truth
  --full-benchmark  time the FlyDSL kernel (or the aiter op when no kernel.py),
                    write build/performance_report.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"
KERNEL_ENTRY = "flydsl_moe_2stage_generic"


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, KERNEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, KERNEL_FILE)):
        return cwd
    return here


def _load_module(kernel_dir, filename, alias):
    entry = os.path.join(kernel_dir, filename)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# Real generic two-stage fused-MoE shapes (bf16, g1u1, silu). The E32/k5 row is a
# compact mid-size MoE; the E257/k9 row is the DeepSeek-V3 fmoe shape
# (configs/model_configs/*fmoe*.csv). model_dim and inter_dim*2 are multiples of
# 128 so the grouped GEMM tiles cleanly.
SHAPES = [
    {"name": "e32_t32_d4096_i1024_k5", "tokens": 32, "model_dim": 4096, "inter_dim": 1024, "experts": 32, "topk": 5},
    {"name": "dsv3_t32_e257_k9", "tokens": 32, "model_dim": 7168, "inter_dim": 256, "experts": 257, "topk": 9},
]

# bf16 fused-MoE gate: normalized worst-element error vs the aiter op.
TOL = 1e-2
SEED = 20260401


def _retry(fn, tries=5, what="kernel call"):
    """Retry on transient out-of-memory / HIP errors (shared-GPU friendly)."""
    import torch

    last = None
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as exc:  # noqa: PERF203
            msg = str(exc).lower()
            if "out of memory" not in msg and "hip" not in msg:
                raise
            last = exc
            torch.cuda.empty_cache()
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{what} failed after {tries} retries: {last}")


def _build_model(mmod, shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = (
        mmod.Model(
            model_dim=shape["model_dim"], inter_dim=shape["inter_dim"],
            experts=shape["experts"], topk=shape["topk"],
        )
        .to(device)
        .eval()
    )
    hidden = torch.randn(
        shape["tokens"], shape["model_dim"], dtype=torch.bfloat16, device=device
    )
    return model, hidden


def _aiter_op(mmod, model, hidden, topk):
    """Drive the real aiter fused_moe (generic two-stage, bf16, no quant)."""
    from aiter import QuantType, ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    logits = model.gate(hidden)
    topk_weights, topk_ids = mmod.route_topk(logits, topk)
    act = (
        ActivationType.Gelu if model.activation == "gelu" else ActivationType.Silu
    )
    w1s = shuffle_weight(model.w1.detach(), layout=(16, 16))
    w2s = shuffle_weight(model.w2.detach(), layout=(16, 16))
    return fused_moe(
        hidden, w1s, w2s, topk_weights, topk_ids,
        quant_type=QuantType.No, activation=act,
    )


def _norm_worst(ref, out):
    rf, of = ref.float(), out.float()
    worst = (rf - of).abs().max().item()
    denom = rf.abs().max().item()
    denom = denom if denom > 0 else 1.0
    return worst, worst / denom


def run_correctness(verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if mmod is None:
        print("FAIL: cannot load model.py")
        assert False, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    failures = []
    for shape in SHAPES:
        try:
            model, hidden = _build_model(mmod, shape)
            with torch.no_grad():
                ref = model(hidden)
                gt = _retry(
                    lambda: _aiter_op(mmod, model, hidden, shape["topk"]),
                    what="aiter fused_moe",
                )
            torch.cuda.synchronize()

            worst, norm = _norm_worst(ref, gt)
            ok = norm <= TOL
            note = ""
            if has_kernel:
                logits = model.gate(hidden)
                topk_weights, topk_ids = mmod.route_topk(logits, shape["topk"])
                out = _retry(
                    lambda: kmod.flydsl_moe_2stage_generic(
                        hidden, model.w1.detach(), model.w2.detach(),
                        topk_weights, topk_ids,
                    ),
                    what="flydsl kernel",
                )
                torch.cuda.synchronize()
                _, knorm = _norm_worst(ref, out)
                kok = knorm <= TOL
                ok = ok and kok
                note = f" | kernel norm={knorm:.4g} {'ok' if kok else 'BAD'}"

            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"(D{shape['model_dim']}/I{shape['inter_dim']}/"
                    f"E{shape['experts']}/k{shape['topk']}) "
                    f"worst={worst:.4g} norm={norm:.4g} tol={TOL}{note}"
                )
            if not ok:
                failures.append(shape["name"])
            del model, hidden, gt
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:200]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    label = "FlyDSL" if has_kernel else "aiter"
    latencies, speedups, report = [], [], []
    print(f"{'Config':<26} {'Ref':>10} {label:>10} {'Speedup':>10}")
    print("-" * 60)
    for idx, shape in enumerate(SHAPES):
        model, hidden = _build_model(mmod, shape)
        topk = shape["topk"]
        with torch.no_grad():
            if has_kernel:
                logits = model.gate(hidden)
                topk_weights, topk_ids = mmod.route_topk(logits, topk)

                def device_op():
                    return kmod.flydsl_moe_2stage_generic(
                        hidden, model.w1.detach(), model.w2.detach(),
                        topk_weights, topk_ids,
                    )
            else:
                def device_op():
                    return _aiter_op(mmod, model, hidden, topk)

            _retry(device_op, what="benchmark warmup")
            torch.cuda.synchronize()
            for _ in range(warmup):
                device_op()
            torch.cuda.synchronize()
            ktimes = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); device_op(); e.record(); torch.cuda.synchronize()
                ktimes.append(s.elapsed_time(e))
            kernel_ms = sum(ktimes) / len(ktimes)

            rtimes = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); model(hidden); e.record(); torch.cuda.synchronize()
                rtimes.append(s.elapsed_time(e))
            ref_ms = sum(rtimes) / len(rtimes)

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [shape["tokens"], shape["model_dim"], shape["inter_dim"]],
            "params": {k: shape[k] for k in ("tokens", "model_dim", "inter_dim", "experts", "topk")},
        })
        if verbose:
            print(f"{shape['name']:<26} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x")
        del model, hidden
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 60)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup:.2f}x")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl moe_2stage_generic harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl fused MoE (generic two-stage, bf16)")
    print("=" * 60)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
