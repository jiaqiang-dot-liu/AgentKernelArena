#!/usr/bin/env python3
"""Task runner for triton2triton/triton_silu_mul_quant_fp8"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_silu_mul_quant_fp8"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_silu_mul_quant_fp8.py")

# Test configs: (M, N) where N must be divisible by 256 (GROUP_SIZE*2), M by 128
TEST_SHAPES = [
    (128, 256),
    (128, 512),
    (256, 512),
    (256, 1024),
    (512, 1024),
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
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_silu_mul_quant_fp8(input_t, fp8_dtype):
    """CPU reference: silu(x[:,:N/2]) * x[:,N/2:], then quantize per group."""
    import torch
    GROUP_SIZE = 128
    M, N = input_t.shape
    N_2 = N // 2

    x = input_t.cpu().float()
    gate = x[:, :N_2]
    up = x[:, N_2:]

    # SiLU + mul
    silu_out = gate / (1.0 + torch.exp(-gate))
    y = silu_out * up

    if fp8_dtype == torch.float8_e4m3fnuz:
        fp8_min, fp8_max = -240.0, 240.0
    else:
        finfo = torch.finfo(fp8_dtype)
        fp8_min, fp8_max = finfo.min, finfo.max

    num_groups = N_2 // GROUP_SIZE
    y_q = torch.zeros_like(y)
    # Column-major scales: shape [M, num_groups]
    y_s = torch.zeros(M, num_groups, dtype=torch.float32)

    for row in range(M):
        for g in range(num_groups):
            start = g * GROUP_SIZE
            end = start + GROUP_SIZE
            group = y[row, start:end]
            absmax = max(group.abs().max().item(), 1e-10)
            scale = absmax / fp8_max
            y_s[row, g] = scale
            y_q[row, start:end] = (group / scale).clamp(fp8_min, fp8_max)

    return y_q.to(fp8_dtype), y_s


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "silu_mul_per_token_group_quant_fp8_colmajor"), \
            "Missing silu_mul_per_token_group_quant_fp8_colmajor"
        assert hasattr(mod, "_silu_mul_per_token_group_quant_fp8_colmajor"), \
            "Missing _silu_mul_per_token_group_quant_fp8_colmajor"
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
    fp8_dtype = mod._get_fp8_dtype()

    for i, (M, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            y_q, y_s = mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
            torch.cuda.synchronize()

            ref_q, ref_s = reference_silu_mul_quant_fp8(x, fp8_dtype)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            # Check scales
            if not torch.allclose(y_s, ref_s, atol=1e-2, rtol=1e-1):
                max_diff = (y_s - ref_s).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}): scale max diff = {max_diff:.6f}"
                )

            # Check quantized via dequant
            N_2 = N // 2
            GROUP_SIZE = 128
            y_dq = y_q.float() * y_s.repeat_interleave(GROUP_SIZE, dim=-1)
            ref_dq = ref_q.float().to(device) * ref_s.repeat_interleave(GROUP_SIZE, dim=-1)

            if not torch.allclose(y_dq, ref_dq, atol=5e-1, rtol=1e-1):
                max_diff = (y_dq - ref_dq).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}): dequant max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, N={N}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (M, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            def _bench_fn():
                mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
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
                    "M": M,
                    "N": N
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
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
