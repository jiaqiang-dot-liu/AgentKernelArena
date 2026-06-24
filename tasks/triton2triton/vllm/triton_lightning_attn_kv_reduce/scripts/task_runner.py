#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lightning_attn_kv_reduce"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_lightning_attn_kv_reduce"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lightning_attn_kv_reduce.py")

# Test configurations: (batch, heads, seq, d_model, e_model)
TEST_SHAPES = [
    (1, 4, 64, 64, 64),
    (2, 8, 128, 64, 64),
    (1, 4, 256, 128, 128),
    (2, 4, 128, 64, 32),
    (4, 2, 64, 32, 32),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def _measure_cuda_event_fallback(fn, repetition):
    import time
    import torch

    repetition = max(1, int(repetition))
    if not torch.cuda.is_available():
        times_ms = []
        for _ in range(repetition):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
        return times_ms

    times_ms = []
    for _ in range(repetition):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    return times_ms


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=10,
    repetition=100,
    target_ms=20.0,
    n_retries=5,
    estimate_reps=5,
    max_graph_repeats=1000,
    use_cuda_graph=True,
    fallback_reason=None,
):
    import torch

    for _ in range(max(0, int(warmup))):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    max_graph_repeats = max(1, int(max_graph_repeats))
    metadata = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_retries": int(n_retries),
        "benchmark_max_repeats": int(max_graph_repeats),
    }

    if not torch.cuda.is_available():
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cpu_timer_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_unavailable",
        })
        return sum(times) / len(times), metadata

    if not use_cuda_graph:
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_graph_disabled",
        })
        return sum(times) / len(times), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_reps = max(1, int(estimate_reps))
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(estimate_reps):
                    fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()

            estimate_ms = start_event.elapsed_time(end_event) / estimate_reps
            if estimate_ms == 0:
                n_repeat = max_graph_repeats
            else:
                n_repeat = min(max_graph_repeats, max(1, int(float(target_ms) / estimate_ms)))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(n_repeat):
                    fn()
            torch.cuda.synchronize()

            retry_times = []
            for _ in range(max(1, int(n_retries))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                retry_times.append(start_event.elapsed_time(end_event) / n_repeat)

        retry_times = sorted(retry_times)
        metadata.update({
            "benchmark_method": "cuda_graph",
            "benchmark_effective_repeats": int(n_repeat),
        })
        return retry_times[len(retry_times) // 2], metadata
    except Exception as exc:
        torch.cuda.synchronize()
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}",
        })
        return sum(times) / len(times), metadata

def load_module():
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_kv_reduce(s, kv, kv_history, n, BLOCK=256):
    """
    PyTorch reference for KV reduce (prefix sum with decay).
    s: [H], kv: [B, H, NUM_BLOCK, D, E], kv_history: [B, H, D, E]
    Returns: (kv_reduced, kv_history_updated) -- both modified
    """
    import torch
    B, H, NUM_BLOCK, D, E = kv.shape
    kv_out = kv.clone()
    kv_hist_out = kv_history.clone()

    for b_idx in range(B):
        for h_idx in range(H):
            slope = s[h_idx].item()
            kv_pre = kv_hist_out[b_idx, h_idx].float().clone()

            for blk in range(NUM_BLOCK):
                block_size = min(n - blk * BLOCK, BLOCK)
                block_decay = float(torch.exp(torch.tensor(-slope * block_size)))

                kv_cur = kv_out[b_idx, h_idx, blk].float().clone()
                kv_out[b_idx, h_idx, blk] = kv_pre
                kv_pre = block_decay * kv_pre + kv_cur

            kv_hist_out[b_idx, h_idx] = kv_pre

    return kv_out, kv_hist_out


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lightning_attn_kv_reduce_forward"), "Missing lightning_attn_kv_reduce_forward"
        assert hasattr(mod, "_fwd_kv_reduce"), "Missing _fwd_kv_reduce"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    """Run correctness checks against PyTorch reference."""
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"

    for i, (B, H, N, D, E) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            BLOCK = 256
            NUM_BLOCK = (N + BLOCK - 1) // BLOCK
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01

            kv = torch.randn(B, H, NUM_BLOCK, D, E, device=device, dtype=torch.float32)
            kv_history = torch.randn(B, H, D, E, device=device, dtype=torch.float32) * 0.1

            # Reference (on clones)
            ref_kv, ref_hist = reference_kv_reduce(s, kv.clone(), kv_history.clone(), N, BLOCK)

            # Run Triton kernel (modifies in-place)
            kv_triton = kv.clone()
            hist_triton = kv_history.clone()
            mod.lightning_attn_kv_reduce_forward(s, kv_triton, hist_triton, N, BLOCK)
            torch.cuda.synchronize()

            if not torch.allclose(kv_triton, ref_kv, atol=1e-2, rtol=1e-2):
                max_diff = (kv_triton - ref_kv).abs().max().item()
                return False, (
                    f"Shape {i+1} KV mismatch (B={B}, H={H}, N={N}, D={D}, E={E}): "
                    f"max diff = {max_diff:.6f}"
                )
            if not torch.allclose(hist_triton, ref_hist, atol=1e-2, rtol=1e-2):
                max_diff = (hist_triton - ref_hist).abs().max().item()
                return False, (
                    f"Shape {i+1} History mismatch (B={B}, H={H}, N={N}, D={D}, E={E}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (B={B}, H={H}, N={N}, D={D}, E={E}): exception: {e}"
            )

    return True, None


def run_performance():
    """Measure kernel execution time."""
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (B, H, N, D, E) in enumerate(TEST_SHAPES):
        try:
            BLOCK = 256
            NUM_BLOCK = (N + BLOCK - 1) // BLOCK

            torch.manual_seed(0)
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
            kv = torch.randn(B, H, NUM_BLOCK, D, E, device=device, dtype=torch.float32)
            kv_history = torch.zeros(B, H, D, E, device=device, dtype=torch.float32)

            # Warmup
            for _ in range(WARMUP_ITERATIONS):
                kv_c = kv.clone()
                h_c = kv_history.clone()
                mod.lightning_attn_kv_reduce_forward(s, kv_c, h_c, N, BLOCK)
            torch.cuda.synchronize()

            # Benchmark
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                kv_c = kv.clone()
                h_c = kv_history.clone()
                start_events[j].record()
                mod.lightning_attn_kv_reduce_forward(s, kv_c, h_c, N, BLOCK)
                end_events[j].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)
            benchmark_metadata = {
                "benchmark_method": "cuda_event_fallback",
                "benchmark_target_ms": 20.0,
                "benchmark_retries": 1,
                "benchmark_max_repeats": 1000,
                "benchmark_effective_repeats": n_iter,
                "benchmark_fallback_reason": "per_iteration_prepare_or_state_reset",
            }

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "batch": B,
                    "heads": H,
                    "seq": N,
                    "d_model": D,
                    "e_model": E
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": B,
                    "heads": H,
                    "seq": N,
                    "d_model": D,
                    "e_model": E
                }
            })

    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
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
        ok, err = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
