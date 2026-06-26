#!/usr/bin/env python3
"""Task runner for triton2triton/triton_correct_attn_cp_out"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_correct_attn_cp_out"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_correct_attn_cp_out.py")

# Test configurations: (B, H, D, N)
# B=batch, H=heads, D=head_dim, N=num_ranks
TEST_SHAPES = [
    (4, 8, 64, 2),
    (8, 16, 64, 4),
    (16, 32, 128, 2),
    (2, 8, 128, 8),
    (32, 16, 64, 4),
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
    target_ms=1.0,
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
        "benchmark_samples": int(repetition),
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
            for _ in range(max(1, int(repetition))):
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
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_correct_attn_cp_out(out, lses, lse_idx, is_base_e=True):
    """CPU/PyTorch reference for correct_attn_cp_out."""
    import torch
    import math

    B, H, D = out.shape
    N = lses.shape[0]

    # Compute global LSE over N ranks for each (B, H)
    # lses: [N, B, H]
    lses_float = lses.float()

    # Handle NaN and inf
    lses_clean = torch.where(
        torch.isnan(lses_float) | (lses_float == float("inf")),
        torch.tensor(-float("inf"), device=lses.device, dtype=torch.float32),
        lses_float,
    )

    lse_max = lses_clean.max(dim=0).values  # [B, H]
    lse_max = torch.where(lse_max == -float("inf"), torch.zeros_like(lse_max), lse_max)

    shifted = lses_clean - lse_max.unsqueeze(0)

    if is_base_e:
        exp_vals = torch.exp(shifted)
        acc = exp_vals.sum(dim=0)
        global_lse = torch.log(acc) + lse_max
    else:
        exp_vals = torch.pow(2.0, shifted)
        acc = exp_vals.sum(dim=0)
        global_lse = torch.log2(acc) + lse_max

    # Compute correction factor
    local_lse = lses_float[lse_idx]  # [B, H]
    lse_diff = local_lse - global_lse

    # Clean up
    lse_diff = torch.where(
        torch.isnan(lse_diff) | (lse_diff == float("inf")),
        torch.tensor(-float("inf"), device=lse_diff.device, dtype=torch.float32),
        lse_diff,
    )

    if is_base_e:
        factor = torch.exp(lse_diff)
    else:
        factor = torch.pow(2.0, lse_diff)

    corrected = out.float() * factor.unsqueeze(-1)
    return corrected.to(out.dtype), global_lse


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "correct_attn_cp_out"), "Missing correct_attn_cp_out"
        assert hasattr(mod, "_correct_attn_cp_out_kernel"), "Missing _correct_attn_cp_out_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"

    for i, (B, H, D, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            out = torch.randn(B, H, D, device=device, dtype=torch.float16)
            lses = torch.randn(N, B, H, device=device, dtype=torch.float32)
            lse_idx = 0

            corrected, final_lse = mod.correct_attn_cp_out(out, lses, lse_idx, is_base_e=True)
            torch.cuda.synchronize()

            ref_corrected, ref_lse = reference_correct_attn_cp_out(out, lses, lse_idx, is_base_e=True)

            if not torch.allclose(final_lse.float(), ref_lse.float(), atol=1e-2, rtol=1e-2):
                max_diff = (final_lse.float() - ref_lse.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, H={H}, D={D}, N={N}): "
                    f"lse max diff = {max_diff:.6f}"
                )

            if not torch.allclose(corrected.float(), ref_corrected.float(), atol=1e-2, rtol=1e-2):
                max_diff = (corrected.float() - ref_corrected.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, H={H}, D={D}, N={N}): "
                    f"output max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (B={B}, H={H}, D={D}, N={N}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (B, H, D, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            out = torch.randn(B, H, D, device=device, dtype=torch.float16)
            lses = torch.randn(N, B, H, device=device, dtype=torch.float32)

            def _bench_fn():
                mod.correct_attn_cp_out(out, lses, 0, is_base_e=True)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "B": B,
                    "H": H,
                    "D": D,
                    "N": N
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "B": B,
                    "H": H,
                    "D": D,
                    "N": N
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
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
