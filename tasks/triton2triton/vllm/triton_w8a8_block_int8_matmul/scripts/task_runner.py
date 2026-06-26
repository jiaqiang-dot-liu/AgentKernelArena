#!/usr/bin/env python3
"""Task runner for triton2triton/triton_w8a8_block_int8_matmul"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_w8a8_block_int8_matmul"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_w8a8_block_int8_matmul.py")

# Test configs: (M, N, K, block_n, block_k)
TEST_SHAPES = [
    (64, 128, 128, 128, 128),
    (128, 256, 256, 128, 128),
    (64, 128, 256, 128, 128),
    (256, 512, 512, 128, 128),
    (128, 256, 512, 128, 128),
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


def reference_w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    """CPU reference: block-wise dequantize INT8 then matmul."""
    import torch
    block_n, block_k = block_size
    M, K = A.shape
    N = B.shape[0]

    A_f = A.cpu().float()
    B_f = B.cpu().float()
    As_f = As.cpu().float()
    Bs_f = Bs.cpu().float()

    # Dequantize A
    A_dq = torch.zeros_like(A_f)
    for m in range(M):
        for kg in range(As_f.shape[1]):
            start_k = kg * block_k
            end_k = min(start_k + block_k, K)
            A_dq[m, start_k:end_k] = A_f[m, start_k:end_k] * As_f[m, kg]

    # Dequantize B
    B_dq = torch.zeros_like(B_f)
    for ng in range(Bs_f.shape[0]):
        for kg in range(Bs_f.shape[1]):
            start_n = ng * block_n
            end_n = min(start_n + block_n, N)
            start_k = kg * block_k
            end_k = min(start_k + block_k, K)
            B_dq[start_n:end_n, start_k:end_k] = B_f[start_n:end_n, start_k:end_k] * Bs_f[ng, kg]

    result = A_dq @ B_dq.T
    return result.to(output_dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "w8a8_block_int8_matmul"), "Missing w8a8_block_int8_matmul"
        assert hasattr(mod, "_w8a8_block_int8_matmul"), "Missing _w8a8_block_int8_matmul"
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

    for i, (M, N, K, block_n, block_k) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            import triton as _triton

            # Create INT8 tensors
            A = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
            B = torch.randint(-128, 127, (N, K), device=device, dtype=torch.int8)

            # Scales
            As = torch.rand(M, _triton.cdiv(K, block_k), device=device, dtype=torch.float32) * 0.1 + 0.01
            Bs = torch.rand(_triton.cdiv(N, block_n), _triton.cdiv(K, block_k),
                           device=device, dtype=torch.float32) * 0.1 + 0.01

            result = mod.w8a8_block_int8_matmul(
                A, B, As, Bs, [block_n, block_k], output_dtype=torch.float16
            )
            torch.cuda.synchronize()

            ref = reference_w8a8_block_int8_matmul(
                A, B, As, Bs, [block_n, block_k], torch.float16
            ).to(device)

            if not torch.allclose(result, ref, atol=1e-1, rtol=1e-1):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}, K={K}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, N={N}, K={K}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    import triton as _triton
    test_cases = []

    for test_idx, (M, N, K, block_n, block_k) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            A = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
            B = torch.randint(-128, 127, (N, K), device=device, dtype=torch.int8)
            As = torch.rand(M, _triton.cdiv(K, block_k), device=device, dtype=torch.float32) + 0.01
            Bs = torch.rand(_triton.cdiv(N, block_n), _triton.cdiv(K, block_k),
                            device=device, dtype=torch.float32) + 0.01

            def _bench_fn():
                mod.w8a8_block_int8_matmul(A, B, As, Bs, [block_n, block_k])
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
                    "N": N,
                    "K": K,
                    "block_n": block_n,
                    "block_k": block_k
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "N": N,
                    "K": K,
                    "block_n": block_n,
                    "block_k": block_k
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
