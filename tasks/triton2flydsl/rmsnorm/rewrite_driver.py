#!/usr/bin/env python3
"""BYOD measurement driver for the rmsnorm triton2flydsl rewrite task.

SELF-CONTAINED single file: depends only on the Python stdlib + torch (imported
lazily). No dependency on KernelForge / kernel_agents, so the task is portable —
it can be run by forge-rewrite, another agent, the task_validator, or by hand.

Compares the FlyDSL candidate (`flydsl/kernel.py`, built via `build_rmsnorm_module`)
against the ORIGINAL Triton RMSNorm (`rmsnorm.py`, host entry `rmsnorm`) used as a
live oracle + performance baseline. RMSNorm has TWO inputs (x[M,N], weight[N]).

FlyDSL contract: build_rmsnorm_module(M, N, dtype_str) -> launch(x, weight, out, M)

Stdout contract (what forge's test/bench tools parse):
  * correctness (default)   -> "SNR: <db> dB"
  * --bench-mode            -> "median_ms: <ms>" (+ one "case_ms: <id> <ms>")
  * --ref-bench-mode        -> "median_ms: <ms>" (the source baseline)
  * --profile-run           -> candidate only, no reference, minimal iters
"""

import argparse
import importlib.util
import math
import os
import sys

# ── measurement primitives (self-contained; stdlib + torch only) ─────────────

_SNR_CAP_DB = 200.0
_TORCH_DTYPE = {
    "fp16": "float16", "f16": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp32": "float32", "f32": "float32", "float32": "float32",
}
_FLYDSL_DTYPE = {
    "fp16": "f16", "f16": "f16", "float16": "f16",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp32": "f32", "f32": "f32", "float32": "f32",
}


def _load_module(path, alias):
    """Import a .py file by path under ``alias`` (its dir goes on sys.path)."""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d and d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _torch_dtype(dt, torch):
    name = _TORCH_DTYPE.get(str(dt).lower())
    if name is None:
        raise ValueError(f"unsupported dtype {dt!r}; expected one of fp16/bf16/fp32")
    return getattr(torch, name)


def _flydsl_dtype(dt):
    out = _FLYDSL_DTYPE.get(str(dt).lower())
    if out is None:
        raise ValueError(f"unsupported dtype {dt!r}; expected one of fp16/bf16/fp32")
    return out


def _snr_db(ref, out, torch):
    """SNR(dB) of ``out`` vs ``ref`` (fp32 accumulation), clamped to a finite cap."""
    ref = ref.float()
    out = out.float()
    signal = ref.pow(2).sum().item()
    noise = (out - ref).pow(2).sum().item()
    if noise <= 0.0:
        return _SNR_CAP_DB
    if signal <= 0.0:
        return -_SNR_CAP_DB
    return max(-_SNR_CAP_DB, min(_SNR_CAP_DB, 10.0 * math.log10(signal / noise)))


def _time_ms(fn, warmup, iters, torch):
    """Median wall time (ms) of ``fn`` over ``iters`` timed runs (GPU-synced)."""
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _resolve_shape(shape_str, default_shapes):
    """"M=8192,N=8192,dtype=fp16" -> dict; falls back to default_shapes[0]."""
    if shape_str and shape_str != "default":
        out = {}
        for part in shape_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        if out:
            return out
    return dict(default_shapes[0]) if default_shapes else {}


# ── operator-specific: how to build inputs / call the source + candidate ─────

_HERE = os.path.dirname(os.path.abspath(__file__))

# Fallback shapes (forge normally passes --shape explicitly).
SHAPES = [
    {"M": 2048, "N": 2048, "dtype": "fp16"},
    {"M": 8192, "N": 8192, "dtype": "fp16"},
]

_src = None
_fly = None


def _source():
    global _src
    if _src is None:
        _src = _load_module(os.path.join(_HERE, "rmsnorm.py"), "rewrite_source")
    return _src


def _flydsl():
    global _fly
    if _fly is None:
        _fly = _load_module(os.path.join(_HERE, "flydsl", "kernel.py"), "rewrite_flydsl")
    return _fly


def make_inputs(shape, mode, torch):
    M, N = int(shape["M"]), int(shape["N"])
    tdt = _torch_dtype(shape.get("dtype", "fp16"), torch)
    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=tdt)
    w = torch.randn(N, device="cuda", dtype=tdt)
    if mode == "stability":
        # Large magnitudes: sum(x^2) must accumulate in fp32. A port that reduces
        # in fp16 overflows (fp16 max 65504) and fails; plain randn never does.
        x = (x * 50.0).to(tdt)
    return {"x": x, "w": w, "M": M}


def reference(inputs, shape, torch):
    # Live oracle + baseline: the original Triton RMSNorm host entry.
    return _source().rmsnorm(inputs["x"], inputs["w"])


def build_candidate(shape, torch):
    M, N = int(shape["M"]), int(shape["N"])
    return _flydsl().build_rmsnorm_module(M, N, _flydsl_dtype(shape.get("dtype", "fp16")))


def run_candidate(launch, inputs, torch):
    out = torch.empty_like(inputs["x"])
    launch(inputs["x"], inputs["w"], out, inputs["M"])
    return out


# ── main: mode dispatch (correctness / bench / ref-bench / profile) ──────────

def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="default")
    ap.add_argument("--mode", default="full")  # smoke/stability/determinism/full
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--ref-bench-mode", action="store_true")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--profile-case", default=None)
    ap.add_argument("--snr-threshold", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _unknown = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    shape = _resolve_shape(args.shape, SHAPES)

    if args.ref_bench_mode:
        inputs = make_inputs(shape, "full", torch)
        med = _time_ms(lambda: reference(inputs, shape, torch), args.warmup, args.iters, torch)
        print(f"median_ms: {med:.6f}")
        return 0

    if args.profile_run:
        inputs = make_inputs(shape, "full", torch)
        launch = build_candidate(shape, torch)
        run_candidate(launch, inputs, torch)
        torch.cuda.synchronize()
        return 0

    if args.bench_mode:
        inputs = make_inputs(shape, "full", torch)
        launch = build_candidate(shape, torch)
        med = _time_ms(lambda: run_candidate(launch, inputs, torch), args.warmup, args.iters, torch)
        print(f"median_ms: {med:.6f}")
        print(f"case_ms: shape0 {med:.6f}")
        return 0

    inputs = make_inputs(shape, args.mode, torch)
    ref = reference(inputs, shape, torch)
    launch = build_candidate(shape, torch)
    out = run_candidate(launch, inputs, torch)
    torch.cuda.synchronize()
    snr = _snr_db(ref, out, torch)
    print(f"SNR: {snr:.2f} dB")
    # Arena scores from this PASS/FAIL; forge's test tool reads the SNR line above.
    # Exit 0 either way so a low-SNR result reads as FAIL, not a driver crash.
    print(f"correctness: {'PASS' if snr >= args.snr_threshold else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
