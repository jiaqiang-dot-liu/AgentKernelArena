#!/usr/bin/env python3
"""Test harness for the triton2flydsl/aiter/fp8_mqa_logits task.

Loads the standalone Triton source (`fp8_mqa_logits.py`) and runs it over a set
of FP8 MQA-logits shapes.

Modes:
  --compile        ast-parse + import the source, assert entry/kernel symbols exist
  --correctness    run the Triton kernel on TEST_SHAPES, assert finite in-window output
  --full-benchmark warmup + cuda-event timing, write build/performance_report.json

The flydsl-vs-triton comparison will be added when the FlyDSL target lands.
"""
import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

SOURCE_FILE = "fp8_mqa_logits.py"
ENTRY = "fp8_mqa_logits"
KERNEL = "_fp8_mqa_logits_kernel"

# (seq_len, seq_len_kv, num_heads, head_size, window)
#   window: "full"    -> [0, seq_len_kv) for every row
#           "causal"  -> row s sees [0, end_s) with end_s growing across rows
#           "band"    -> row s sees a sliding band [start_s, end_s)
# num_heads / head_size are powers of 2 (kernel asserts this). num_heads is the
# MFMA M dim, so it must be a multiple of matrix_instr_nonkdim (16 for
# seq_len<=1024, else 32). seq_len_kv values include non-128 multiples to
# exercise the masked tail.
TEST_SHAPES = [
    (64, 512, 32, 128, "full"),     # basic full window
    (128, 1000, 64, 128, "full"),   # masked tail (1000 % 128 != 0)
    (256, 2048, 32, 64, "causal"),  # causal windows, head_size=64
    (96, 777, 16, 128, "band"),     # sliding band, ragged size
    (2048, 1024, 64, 128, "full"),  # seq_len>1024 -> matrix_instr_nonkdim=32
]

SEED = 20260617
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def _resolve_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, SOURCE_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, SOURCE_FILE)):
        return cwd
    return here


_TASK_DIR = _resolve_dir()


def load_module():
    entry = os.path.join(_TASK_DIR, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("fp8_mqa_logits_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_windows(seq_len, seq_len_kv, window, device):
    import torch

    if window == "full":
        starts = torch.zeros(seq_len, dtype=torch.int32)
        ends = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)
    elif window == "causal":
        # row s sees [0, end_s); end_s grows monotonically to seq_len_kv
        rows = torch.arange(1, seq_len + 1, dtype=torch.float64)
        ends = (rows / seq_len * seq_len_kv).ceil().clamp(1, seq_len_kv).to(torch.int32)
        starts = torch.zeros(seq_len, dtype=torch.int32)
    elif window == "band":
        band = max(seq_len_kv // 3, 1)
        rows = torch.arange(1, seq_len + 1, dtype=torch.float64)
        ends = (rows / seq_len * seq_len_kv).ceil().clamp(1, seq_len_kv).to(torch.int32)
        starts = (ends - band).clamp(min=0).to(torch.int32)
    else:
        raise ValueError(f"unknown window {window!r}")
    return starts.to(device), ends.to(device)


def make_inputs(seq_len, seq_len_kv, num_heads, head_size, window, mod, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    e4m3 = mod.e4m3_dtype

    # Keep magnitudes small so the fp8 (e4m3) quantization of Q/KV does not
    # saturate and the Q.KV dot stays well-conditioned.
    q = (torch.randn(seq_len, num_heads, head_size, generator=gen, device=device) * 0.3).to(e4m3)
    kv = (torch.randn(seq_len_kv, head_size, generator=gen, device=device) * 0.3).to(e4m3)
    kv_scales = (torch.rand(seq_len_kv, generator=gen, device=device, dtype=torch.float32) + 0.5)
    weights = torch.rand(seq_len, num_heads, generator=gen, device=device, dtype=torch.float32)
    cu_starts, cu_ends = _build_windows(seq_len, seq_len_kv, window, device)
    return q, kv, kv_scales, weights, cu_starts, cu_ends


def _window_mask(seq_len_kv, cu_starts, cu_ends, device):
    import torch

    col = torch.arange(seq_len_kv, device=device)[None, :]
    start = cu_starts.clamp(min=0).long()[:, None]
    end = cu_ends.clamp(max=seq_len_kv).long()[:, None]
    return (col >= start) & (col < end)


def run_compile():
    try:
        with open(os.path.join(_TASK_DIR, SOURCE_FILE)) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, ENTRY), f"Missing {ENTRY} entry"
        assert hasattr(mod, KERNEL), f"Missing {KERNEL} kernel"
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def run_correctness(verbose=True):
    import torch

    try:
        mod = load_module()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot load {SOURCE_FILE}: {e}")
        return {"correct": False, "details": []}

    details = []
    failures = []
    for i, (s, skv, h, d, window) in enumerate(TEST_SHAPES):
        try:
            q, kv, kv_scales, weights, cu_starts, cu_ends = make_inputs(
                s, skv, h, d, window, mod
            )
            out = mod.fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends,
                                     clean_logits=True)
            torch.cuda.synchronize()

            # In-window positions must be finite; no NaNs anywhere. Out-of-window
            # positions are intentionally -inf (causal masking), so they are excluded.
            in_window = _window_mask(skv, cu_starts, cu_ends, out.device)
            no_nan = bool((~torch.isnan(out)).all().item())
            in_window_finite = bool(torch.isfinite(out[in_window]).all().item()) if in_window.any() else True
            ok = no_nan and in_window_finite
            details.append({
                "shape_id": i + 1,
                "shape": [s, skv, h, d, window],
                "no_nan": no_nan,
                "in_window_finite": in_window_finite,
                "passed": bool(ok),
            })
            if verbose:
                print(f"  {'PASS' if ok else 'FAIL'}: shape {i+1} "
                      f"(seq={s}, kv={skv}, H={h}, D={d}, {window}) "
                      f"no_nan={no_nan} in_window_finite={in_window_finite}")
            if not ok:
                failures.append(i + 1)
        except Exception as e:  # noqa: BLE001
            details.append({"shape_id": i + 1, "shape": [s, skv, h, d, window],
                            "error": str(e)})
            failures.append(i + 1)
            if verbose:
                print(f"  FAIL: shape {i+1} (seq={s}, kv={skv}, H={h}, D={d}, {window}) "
                      f"- {str(e)[:120]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(TEST_SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return {"correct": not failures, "details": details}


def run_benchmark(verbose=True):
    import torch

    try:
        mod = load_module()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot load {SOURCE_FILE}: {e}")
        return {"geomean_latency_ms": -1.0}

    report = []
    latencies = []
    print(f"{'shape (seq,kv,H,D,win)':<34} {'latency(ms)':>12}")
    print("-" * 48)
    for idx, (s, skv, h, d, window) in enumerate(TEST_SHAPES):
        try:
            q, kv, kv_scales, weights, cu_starts, cu_ends = make_inputs(
                s, skv, h, d, window, mod
            )
            for _ in range(WARMUP_ITERATIONS):
                mod.fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends)
            torch.cuda.synchronize()

            starts = [torch.cuda.Event(enable_timing=True) for _ in range(BENCHMARK_ITERATIONS)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(BENCHMARK_ITERATIONS)]
            for j in range(BENCHMARK_ITERATIONS):
                starts[j].record()
                mod.fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends)
                ends[j].record()
            torch.cuda.synchronize()
            times = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            ms = sum(times) / len(times)
        except Exception as e:  # noqa: BLE001
            ms = -1.0
            if verbose:
                print(f"  shape {idx+1} error: {str(e)[:120]}")
        latencies.append(ms)
        report.append({
            "test_case_id": f"perf{idx + 1}",
            "execution_time_ms": ms,
            "params": {"seq_len": s, "seq_len_kv": skv, "num_heads": h,
                       "head_size": d, "window": window},
        })
        if verbose:
            print(f"(seq={s:>5}, kv={skv:>5}, H={h:>3}, D={d:>3}, {window:<6}) {ms:>12.4f}")
        del q, kv, kv_scales, weights, cu_starts, cu_ends
        torch.cuda.empty_cache()

    build_dir = Path(_TASK_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    valid = [x for x in latencies if x > 0]
    geomean = math.exp(sum(math.log(x) for x in valid) / len(valid)) if valid else -1.0
    print("-" * 48)
    print(f"Geometric mean latency: {geomean:.4f} ms ({len(valid)}/{len(TEST_SHAPES)} measured)")
    return {"geomean_latency_ms": geomean}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triton2flydsl fp8_mqa_logits harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 48)
    print("triton2flydsl FP8 MQA logits")
    print("=" * 48)

    if args.compile:
        ok, err = run_compile()
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.correctness:
        result = run_correctness()
        sys.exit(0 if result.get("correct", False) else 1)
    else:
        run_benchmark()
        sys.exit(0)
