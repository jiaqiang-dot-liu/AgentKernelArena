#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl layernorm2d (model-only) task.

`model.py` is the pure-torch reference (bf16 2D LayerNorm, fp32 reduction). No
`kernel.py` ships: a clean standalone FlyDSL layernorm kernel does not exist in
aiter, so FlyDSL is GEAK's target.

Model-only correctness: the reference in `model.py` is validated against the REAL
AMD runtime op `aiter.layer_norm` (CK `layernorm2d_fwd`, the ground truth). The
harness MAY import aiter; `model.py` MUST NOT. The normalized worst-element error
``max|truth - ref| / max|truth|`` must be <= REL_TOL.

Modes:
  --compile         import model.py + aiter and report readiness
  --correctness     assert model.py matches aiter.layer_norm at the tight gate
  --full-benchmark  time the torch reference and aiter op, write perf report
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


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, MODEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, MODEL_FILE)):
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

# Real transformer hidden shapes (m rows, n hidden).
SHAPES = [
    {"name": "m1_n4096", "m": 1, "n": 4096},
    {"name": "m8_n4096", "m": 8, "n": 4096},
    {"name": "m32_n8192", "m": 32, "n": 8192},
    {"name": "m128_n8192", "m": 128, "n": 8192},
    {"name": "m256_n6144", "m": 256, "n": 6144},
    {"name": "m64_n4096", "m": 64, "n": 4096},
]

REL_TOL = 1e-2
SEED = 0
EPS = 1e-5


def _retry(fn, *, tries=5, what="op"):
    """Retry on transient OOM/contention (a 2nd worker may share the GPU)."""
    import torch

    delay = 0.5
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as e:  # noqa: PERF203
            msg = str(e).lower()
            transient = "out of memory" in msg or "hip" in msg or "ran out" in msg
            if not transient or attempt == tries - 1:
                raise
            print(
                f"  [retry] transient GPU error on {what} "
                f"(attempt {attempt + 1}/{tries}): {str(e)[:80]} — backing off {delay:.1f}s"
            )
            torch.cuda.empty_cache()
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _make_inputs(shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    m, n = shape["m"], shape["n"]
    input = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n, dtype=torch.bfloat16, device=device)
    bias = torch.randn(n, dtype=torch.bfloat16, device=device)
    return input, weight, bias


def _norm_max_err(ref, out):
    ref_f, out_f = ref.float(), out.float()
    max_abs = (ref_f - out_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-9
    return max_abs / denom, max_abs, denom


def run_correctness(verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"

    init = mmod.get_init_inputs()
    smoke_model = mmod.Model(*init).to("cuda").eval()
    with torch.no_grad():
        smoke_args = [a.to("cuda") for a in mmod.get_inputs()]
        smoke_out = smoke_model(*smoke_args)
    assert smoke_out.shape == smoke_args[0].shape, "smoke Model forward shape mismatch"
    if verbose:
        print(
            f"  smoke: Model(*get_init_inputs())+get_inputs() OK "
            f"(init={init}, out={tuple(smoke_out.shape)})"
        )

    failures = []
    worst = 0.0
    for shape in SHAPES:
        m, n = shape["m"], shape["n"]
        try:
            model = mmod.Model(EPS).to("cuda").eval()
            input, weight, bias = _make_inputs(shape)

            with torch.no_grad():
                ref = model(input, weight, bias)

            truth = _retry(
                lambda: aiter.layer_norm(input, weight, bias, EPS), what=shape["name"]
            )
            torch.cuda.synchronize()

            err, max_abs, _ = _norm_max_err(truth, ref)
            worst = max(worst, err)
            pct = (
                torch.isclose(truth.float(), ref.float(), atol=1e-2, rtol=1e-2)
                .float()
                .mean()
                .item()
                * 100
            )
            ok = err <= REL_TOL
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} (m{m}/n{n}) "
                    f"norm_max_err={err:.6f} (tol={REL_TOL}) max_abs={max_abs:.5f} "
                    f"close%@1e-2={pct:.2f}"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"worst normalized max error across all shapes: {worst:.6f} (tol={REL_TOL})")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=50, verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"

    latencies, report = [], []
    print(f"{'Config':<20} {'TorchRef':>12} {'aiter':>12}")
    print("-" * 48)
    for idx, shape in enumerate(SHAPES):
        m, n = shape["m"], shape["n"]
        model = mmod.Model(EPS).to("cuda").eval()
        input, weight, bias = _make_inputs(shape)

        def run_ref():
            with torch.no_grad():
                return model(input, weight, bias)

        def run_truth():
            return aiter.layer_norm(input, weight, bias, EPS)

        _retry(run_truth, what=shape["name"])
        torch.cuda.synchronize()

        def _median(fn):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            return sorted(ts)[len(ts) // 2]

        ref_ms = _median(run_ref)
        aiter_ms = _median(run_truth)
        latencies.append(ref_ms)
        # bytes moved: input + output (bf16) + weight + bias (bf16).
        bytes_total = (m * n * 2 * 2) + (n * 2 * 2)
        gbps = bytes_total / (ref_ms * 1e-3) / 1e9
        report.append(
            {
                "test_case_id": f"test_case_{idx}",
                "execution_time_ms": ref_ms,
                "shape": [m, n],
                "params": {"m": m, "n": n, "eps": EPS, "dtype": "bf16"},
                "aiter_ms": aiter_ms,
                "gbps": gbps,
            }
        )
        if verbose:
            print(f"{shape['name']:<20} {ref_ms:>10.4f}ms {aiter_ms:>10.4f}ms")
        del model, input, weight, bias
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("-" * 48)
    print(f"Geometric mean torch-reference latency: {geomean_latency:.4f} ms")
    return {"geomean_latency_ms": geomean_latency}


def run_compile():
    import torch  # noqa: F401
    import aiter  # noqa: F401

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    assert hasattr(mmod, "Model") and hasattr(mmod, "get_inputs"), "model.py contract"
    print("compile ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl layernorm2d harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl layernorm2d (bf16, model-only)")
    print("=" * 60)

    if args.compile:
        run_compile()
        sys.exit(0)
    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
