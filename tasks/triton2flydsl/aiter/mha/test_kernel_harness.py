#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/mha (FORWARD only).

Self-contained harness mirroring the triton2flydsl/unified_attention template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton forward on TEST_SHAPES, assert finite output
                   (fp16), causal + non-causal, incl. GQA. No torch comparison:
                   the flydsl-vs-triton comparison is added when the FlyDSL target
                   lands (the Triton kernel is the reference here).
  - performance  : warmup + cuda-event timing, write build/performance_report.json

The kernel under test is the Triton MHA forward (`_attn_fwd`). Forward-only:
the backward kernels were intentionally dropped (out of scope for this task).
Public entry: `flash_attn_func(q, k, v, softmax_scale, causal, window_size, ...)`.
"""
import sys
import os
import json
import time
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/aiter/mha"
SOURCE_FILE = os.path.join(TASK_DIR, "mha.py")

# Test configurations:
# (batch, seqlen, num_query_heads, num_kv_heads, head_size, causal)
# seqlen_q == seqlen_k keeps causal masking aligned with torch SDPA is_causal.
TEST_SHAPES = [
    (2, 128, 8, 8, 64, False),    # MHA, non-causal
    (2, 128, 8, 8, 64, True),     # MHA, causal
    (1, 256, 16, 16, 128, True),  # MHA, causal, hs=128
    (4, 64, 16, 4, 64, True),     # GQA (4:1), causal
    (1, 512, 8, 2, 128, False),   # GQA (4:1), non-causal, hs=128
    (2, 384, 12, 12, 64, True),   # non-power-of-2 heads, causal
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

MAX_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("mha_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(batch, seqlen, nqh, nkvh, hs, device="cuda", dtype=None):
    """Q,K,V in bshd layout: [batch, seqlen, nheads, head_size]."""
    import torch
    if dtype is None:
        dtype = torch.float16
    q = torch.randn(batch, seqlen, nqh, hs, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, nkvh, hs, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, nkvh, hs, device=device, dtype=dtype)
    scale = 1.0 / (hs ** 0.5)
    return q, k, v, scale


def _call_kernel(mod, q, k, v, scale, causal):
    return mod.flash_attn_func(
        q, k, v,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1),
    )


def _with_oom_retry(fn):
    """Retry on transient CUDA OOM (other workers share the GPU)."""
    import torch
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            last = e
            torch.cuda.empty_cache()
            time.sleep(2.0 * (attempt + 1))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                last = e
                torch.cuda.empty_cache()
                time.sleep(2.0 * (attempt + 1))
            else:
                raise
    raise last


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "flash_attn_func"), "Missing flash_attn_func entry"
        assert hasattr(mod, "_attn_fwd"), "Missing _attn_fwd kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.float16
    details = []

    for i, (batch, seqlen, nqh, nkvh, hs, causal) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, k, v, scale = make_test_data(batch, seqlen, nqh, nkvh, hs, device, dtype)

            result = _with_oom_retry(lambda: _call_kernel(mod, q, k, v, scale, causal))
            torch.cuda.synchronize()

            ok = bool(torch.isfinite(result).all().item())
            details.append({
                "shape_id": i + 1,
                "shape": [batch, seqlen, nqh, nkvh, hs, causal],
                "out_shape": list(result.shape),
                "finite": ok,
                "passed": ok,
            })
            if not ok:
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: non-finite output", details
        except Exception as e:
            details.append({
                "shape_id": i + 1,
                "shape": [batch, seqlen, nqh, nkvh, hs, causal],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (batch, seqlen, nqh, nkvh, hs, causal) in enumerate(TEST_SHAPES):
        params = {
            "batch": batch, "seqlen": seqlen, "num_query_heads": nqh,
            "num_kv_heads": nkvh, "head_size": hs, "causal": causal,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, k, v, scale = make_test_data(batch, seqlen, nqh, nkvh, hs, device, dtype)

            for _ in range(WARMUP_ITERATIONS):
                _with_oom_retry(lambda: _call_kernel(mod, q, k, v, scale, causal))
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                start_events[j].record()
                _call_kernel(mod, q, k, v, scale, causal)
                end_events[j].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
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
        for d in details:
            if "finite" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: out={d['out_shape']} "
                      f"finite={d['finite']} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
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
