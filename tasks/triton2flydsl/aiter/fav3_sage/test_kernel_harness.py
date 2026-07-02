#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/fav3_sage.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton source on TEST_SHAPES, assert finite output. No torch
                   comparison: the flydsl-vs-triton comparison is added when the FlyDSL
                   target lands (the Triton kernel is the reference here).
  - performance  : warmup + cuda-event timing, write build/performance_report.json

The kernel under test is SageAttention v1 (INT8 Q/K + FP8 V flash attention).
Public entry: `fav3_sage(q, k, v, ...)` (high-precision BF16/FP16/FP32 in -> BF16 out;
quantizes internally). @triton.jit kernels: `sage_fwd` (attention) and
`sage_quant_kernel` (smooth-K + per-block INT8 Q/K + per-channel FP8 V quantization).
"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/aiter/fav3_sage"
SOURCE_FILE = os.path.join(TASK_DIR, "fav3_sage.py")

# Small kernel config so several seqlen blocks are exercised cheaply on a shared GPU.
# BLKQ == BLOCK_M and BLKK == BLOCK_N must stay consistent (the descale tables are
# indexed per BLOCK_M / BLOCK_N block).
CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "waves_per_eu": 2,
    "PRE_LOAD_V": False,
    "num_stages": 2,
    "num_warps": 4,
}

# (batch, seqlen, num_q_heads, num_kv_heads, head_dim, causal, window)
# window > 0 selects a causal sliding window of `window` keys; 0 disables it.
TEST_SHAPES = [
    (1, 64, 4, 4, 64, True, 0),     # single block, causal, MHA, d=64
    (1, 128, 8, 8, 64, True, 0),    # 2 blocks, causal
    (2, 128, 8, 2, 64, True, 0),    # GQA causal
    (1, 128, 8, 8, 128, True, 0),   # d=128, causal
    (1, 128, 8, 8, 64, False, 0),   # non-causal (full) attention
    (1, 256, 8, 8, 64, True, 0),    # 4 blocks, causal
    (2, 128, 8, 8, 64, True, 32),   # causal sliding window (32 keys)
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def load_module():
    spec = importlib.util.spec_from_file_location("fav3_sage_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(batch, seqlen, hq, hk, d, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, hq, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, hk, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, hk, d, device=device, dtype=dtype)
    scale = 1.0 / (d ** 0.5)
    return q, k, v, scale


def _window_size(window):
    """Causal sliding window of `window` keys -> (left=window-1, right=0)."""
    if window and window > 0:
        return (window - 1, 0)
    return (-1, -1)


def _call_kernel(mod, q, k, v, scale, causal, window):
    return mod.fav3_sage(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        window_size=_window_size(window),
        layout="bshd",
        return_lse=False,
        smooth_k=True,
        config=dict(CONFIG),
    )


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fav3_sage"), "Missing fav3_sage entry"
        assert hasattr(mod, "sage_fwd"), "Missing sage_fwd kernel"
        assert hasattr(mod, "sage_quant_kernel"), "Missing sage_quant_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    # Runs the Triton kernel on TEST_SHAPES and asserts finite output. No torch
    # comparison: the flydsl-vs-triton comparison is added when the FlyDSL target
    # lands (the Triton kernel is the reference here).
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    details = []

    for i, (b, s, hq, hk, d, causal, window) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, k, v, scale = make_test_data(b, s, hq, hk, d, device)

            result = _call_kernel(mod, q, k, v, scale, causal, window)
            torch.cuda.synchronize()

            ok = bool(torch.isfinite(result).all().item())
            details.append({
                "shape_id": i + 1,
                "shape": [b, s, hq, hk, d, causal, window],
                "out_shape": list(result.shape),
                "finite": ok,
                "passed": ok,
            })
            if not ok:
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: non-finite output", details
        except Exception as e:
            import traceback
            details.append({
                "shape_id": i + 1,
                "shape": [b, s, hq, hk, d, causal, window],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}\n{traceback.format_exc()}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (b, s, hq, hk, d, causal, window) in enumerate(TEST_SHAPES):
        params = {
            "batch": b, "seqlen": s, "num_q_heads": hq, "num_kv_heads": hk,
            "head_dim": d, "causal": causal, "window": window,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, k, v, scale = make_test_data(b, s, hq, hk, d, device)

            for _ in range(WARMUP_ITERATIONS):
                _call_kernel(mod, q, k, v, scale, causal, window)
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                start_events[j].record()
                _call_kernel(mod, q, k, v, scale, causal, window)
                end_events[j].record()

            torch.cuda.synchronize()
            times = [s_e.elapsed_time(e_e) for s_e, e_e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": params,
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": params,
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
            "details": details,
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for dd in details:
            if "finite" in dd:
                print(f"  shape {dd['shape_id']} {dd['shape']}: out={dd['out_shape']} "
                      f"finite={dd['finite']} -> {'PASS' if dd['passed'] else 'FAIL'}")
            elif "error" in dd:
                print(f"  shape {dd['shape_id']} {dd['shape']}: ERROR {dd['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
