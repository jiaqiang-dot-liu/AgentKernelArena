#!/usr/bin/env python3
"""Task runner for triton2triton/triton_mrope"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_mrope"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_mrope.py")

# Test configurations: (num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim, mrope_section)
TEST_SHAPES = [
    (32, 8, 8, 64, 64, [16, 8, 8]),
    (64, 16, 4, 64, 64, [16, 8, 8]),
    (128, 32, 8, 128, 64, [16, 8, 8]),
    (256, 16, 16, 64, 64, [16, 8, 8]),
    (16, 8, 2, 128, 64, [16, 8, 8]),
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


def reference_mrope(q, k, cos, sin, mrope_section, head_size, rotary_dim):
    """CPU/PyTorch reference for MRoPE.

    q: [num_tokens, num_q_heads * head_size]
    k: [num_tokens, num_kv_heads * head_size]
    cos: [3, num_tokens, rotary_dim // 2]
    sin: [3, num_tokens, rotary_dim // 2]
    mrope_section: [t, h, w]
    """
    import torch

    num_tokens = q.shape[0]
    n_q_head = q.shape[1] // head_size
    n_kv_head = k.shape[1] // head_size
    half_rd = rotary_dim // 2

    # Build combined cos/sin from sections (non-interleaved)
    t_sec, h_sec, w_sec = mrope_section
    # cos/sin shape: [3, num_tokens, rotary_dim // 2]
    # Section t: indices [0, t_sec), from cos[0]
    # Section h: indices [t_sec, t_sec+h_sec), from cos[1]
    # Section w: indices [t_sec+h_sec, half_rd), from cos[2]
    combined_cos = torch.zeros(num_tokens, half_rd, device=q.device, dtype=cos.dtype)
    combined_sin = torch.zeros(num_tokens, half_rd, device=q.device, dtype=sin.dtype)

    combined_cos[:, :t_sec] = cos[0, :, :t_sec]
    combined_sin[:, :t_sec] = sin[0, :, :t_sec]
    combined_cos[:, t_sec:t_sec + h_sec] = cos[1, :, t_sec:t_sec + h_sec]
    combined_sin[:, t_sec:t_sec + h_sec] = sin[1, :, t_sec:t_sec + h_sec]
    combined_cos[:, t_sec + h_sec:half_rd] = cos[2, :, t_sec + h_sec:half_rd]
    combined_sin[:, t_sec + h_sec:half_rd] = sin[2, :, t_sec + h_sec:half_rd]

    # Apply rotary embedding to q
    q_out = q.clone()
    for h in range(n_q_head):
        offset = h * head_size
        x1 = q_out[:, offset:offset + half_rd].float()
        x2 = q_out[:, offset + half_rd:offset + rotary_dim].float()
        c = combined_cos.float()
        s = combined_sin.float()
        q_out[:, offset:offset + half_rd] = (x1 * c - x2 * s).to(q.dtype)
        q_out[:, offset + half_rd:offset + rotary_dim] = (x2 * c + x1 * s).to(q.dtype)

    # Apply rotary embedding to k
    k_out = k.clone()
    for h in range(n_kv_head):
        offset = h * head_size
        x1 = k_out[:, offset:offset + half_rd].float()
        x2 = k_out[:, offset + half_rd:offset + rotary_dim].float()
        c = combined_cos.float()
        s = combined_sin.float()
        k_out[:, offset:offset + half_rd] = (x1 * c - x2 * s).to(k.dtype)
        k_out[:, offset + half_rd:offset + rotary_dim] = (x2 * c + x1 * s).to(k.dtype)

    return q_out, k_out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "triton_mrope"), "Missing triton_mrope"
        assert hasattr(mod, "_triton_mrope_forward"), "Missing _triton_mrope_forward"
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
    dtype = torch.float16

    for i, (num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
            k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
            cos = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
            sin = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)

            # Clone for reference
            q_ref = q.clone()
            k_ref = k.clone()

            # Triton kernel (in-place)
            q_triton = q.clone()
            k_triton = k.clone()
            mod.triton_mrope(
                q_triton, k_triton, cos, sin, mrope_section,
                head_size, rotary_dim, False
            )
            torch.cuda.synchronize()

            # Reference
            q_expected, k_expected = reference_mrope(
                q_ref, k_ref, cos, sin, mrope_section, head_size, rotary_dim
            )

            if not torch.allclose(q_triton, q_expected, atol=1e-2, rtol=1e-2):
                max_diff = (q_triton - q_expected).abs().max().item()
                return False, (
                    f"Shape {i + 1} q mismatch: max diff = {max_diff:.6f}"
                )
            if not torch.allclose(k_triton, k_expected, atol=1e-2, rtol=1e-2):
                max_diff = (k_triton - k_expected).abs().max().item()
                return False, (
                    f"Shape {i + 1} k mismatch: max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i + 1}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
            k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
            cos = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
            sin = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)

            for _ in range(WARMUP_ITERATIONS):
                q_tmp = q.clone()
                k_tmp = k.clone()
                mod.triton_mrope(q_tmp, k_tmp, cos, sin, mrope_section, head_size, rotary_dim, False)
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                q_tmp = q.clone()
                k_tmp = k.clone()
                start_events[j].record()
                mod.triton_mrope(q_tmp, k_tmp, cos, sin, mrope_section, head_size, rotary_dim, False)
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
                    "num_tokens": num_tokens,
                    "num_q_heads": n_qh,
                    "num_kv_heads": n_kh,
                    "head_size": head_size,
                    "rotary_dim": rotary_dim
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_tokens": num_tokens,
                    "num_q_heads": n_qh,
                    "num_kv_heads": n_kh,
                    "head_size": head_size,
                    "rotary_dim": rotary_dim
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
