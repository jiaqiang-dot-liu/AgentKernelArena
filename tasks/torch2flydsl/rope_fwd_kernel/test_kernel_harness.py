#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl rope_fwd (model-only) task.

`model.py` is the pure-torch reference (bf16 cached-cos/sin RoPE, sbhd layout,
fp32 rotation). No `kernel.py` ships: a clean standalone FlyDSL RoPE kernel does
not exist in aiter (only the already-done fused qk-norm-rope path), so FlyDSL is
GEAK's target.

Model-only correctness: the reference in `model.py` is validated against the REAL
AMD runtime op `aiter.rope_cached_fwd` (the ground truth). The harness MAY import
aiter; `model.py` MUST NOT. The normalized worst-element error
``max|truth - ref| / max|truth|`` must be <= REL_TOL.

The sweep covers both rotate styles (NEOX/GPT-J), full and partial rotary, the
reuse-freqs-front-part on/off cases, and a nope-first case.

Modes:
  --compile         import model.py + aiter and report readiness
  --correctness     assert model.py matches aiter.rope_cached_fwd at the gate
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

ROTATE_NEOX = 0
ROTATE_GPTJ = 1


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

# sbhd shapes + RoPE variants. rotary_percent<1.0 leaves a no-position tail (or
# head when nope_first). reuse=True stores one entry per freq pair (rotary_dim =
# cos.shape[-1]*2). The primary variant is NEOX / reuse / full-rotary.
SHAPES = [
    {"name": "neox_full_reuse", "s": 2048, "b": 2, "h": 8, "d": 128,
     "rotary_pct": 1.0, "rotate_style": ROTATE_NEOX, "reuse": True, "nope_first": False},
    {"name": "gptj_full_reuse", "s": 2048, "b": 2, "h": 8, "d": 128,
     "rotary_pct": 1.0, "rotate_style": ROTATE_GPTJ, "reuse": True, "nope_first": False},
    {"name": "neox_half_noreuse", "s": 1024, "b": 4, "h": 8, "d": 128,
     "rotary_pct": 0.5, "rotate_style": ROTATE_NEOX, "reuse": False, "nope_first": False},
    {"name": "gptj_half_reuse", "s": 1024, "b": 4, "h": 8, "d": 128,
     "rotary_pct": 0.5, "rotate_style": ROTATE_GPTJ, "reuse": True, "nope_first": False},
    {"name": "neox_half_nopefirst", "s": 1024, "b": 2, "h": 16, "d": 128,
     "rotary_pct": 0.5, "rotate_style": ROTATE_NEOX, "reuse": True, "nope_first": True},
    {"name": "neox_full_noreuse", "s": 512, "b": 2, "h": 16, "d": 64,
     "rotary_pct": 1.0, "rotate_style": ROTATE_NEOX, "reuse": False, "nope_first": False},
]

REL_TOL = 1e-2
SEED = 0


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
    s, b, h, d = shape["s"], shape["b"], shape["h"], shape["d"]
    ratio = 2 if shape["reuse"] else 1
    rot = int(d * shape["rotary_pct"])
    freq_dim = rot // ratio
    input = torch.randn(s, b, h, d, dtype=torch.bfloat16, device=device)
    freqs = torch.randn(s, 1, 1, freq_dim, dtype=torch.bfloat16, device=device)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return input, cos, sin


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
        try:
            model = mmod.Model(
                shape["rotate_style"], shape["reuse"], shape["nope_first"]
            ).to("cuda").eval()
            input, cos, sin = _make_inputs(shape)

            with torch.no_grad():
                ref = model(input, cos, sin)

            truth = _retry(
                lambda: aiter.rope_cached_fwd(
                    input, cos, sin,
                    shape["rotate_style"], shape["reuse"], shape["nope_first"], False,
                ),
                what=shape["name"],
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
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"(s{shape['s']}/b{shape['b']}/h{shape['h']}/d{shape['d']}) "
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
    print(f"{'Config':<22} {'TorchRef':>12} {'aiter':>12}")
    print("-" * 50)
    for idx, shape in enumerate(SHAPES):
        s, b, h, d = shape["s"], shape["b"], shape["h"], shape["d"]
        model = mmod.Model(
            shape["rotate_style"], shape["reuse"], shape["nope_first"]
        ).to("cuda").eval()
        input, cos, sin = _make_inputs(shape)

        def run_ref():
            with torch.no_grad():
                return model(input, cos, sin)

        def run_truth():
            return aiter.rope_cached_fwd(
                input, cos, sin,
                shape["rotate_style"], shape["reuse"], shape["nope_first"], False,
            )

        _retry(run_truth, what=shape["name"])
        torch.cuda.synchronize()

        def _median(fn):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                fn()
                ev1.record()
                torch.cuda.synchronize()
                ts.append(ev0.elapsed_time(ev1))
            return sorted(ts)[len(ts) // 2]

        ref_ms = _median(run_ref)
        aiter_ms = _median(run_truth)
        latencies.append(ref_ms)
        # bytes moved: input + output (bf16); cos/sin caches are small.
        bytes_total = s * b * h * d * 2 * 2
        gbps = bytes_total / (ref_ms * 1e-3) / 1e9
        report.append(
            {
                "test_case_id": f"test_case_{idx}",
                "execution_time_ms": ref_ms,
                "shape": [s, b, h, d],
                "params": {
                    "s": s, "b": b, "h": h, "d": d,
                    "rotary_pct": shape["rotary_pct"],
                    "rotate_style": shape["rotate_style"],
                    "reuse": shape["reuse"],
                    "nope_first": shape["nope_first"],
                    "dtype": "bf16",
                },
                "aiter_ms": aiter_ms,
                "gbps": gbps,
            }
        )
        if verbose:
            print(f"{shape['name']:<22} {ref_ms:>10.4f}ms {aiter_ms:>10.4f}ms")
        del model, input, cos, sin
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("-" * 50)
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
    parser = argparse.ArgumentParser(description="torch2flydsl rope_fwd harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl rope_fwd (cached cos/sin, sbhd, bf16, model-only)")
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
