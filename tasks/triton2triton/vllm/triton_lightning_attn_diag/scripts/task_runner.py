#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lightning_attn_diag"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_lightning_attn_diag"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lightning_attn_diag.py")

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


def reference_diag_attention(q, k, v, s, BLOCK=256, CBLOCK=32):
    """
    PyTorch reference for diagonal block attention with exponential decay.
    q: [B, H, N, D], k: [B, H, N, D], v: [B, H, N, E], s: [H]
    Returns: o [B, H, N, E]
    """
    import torch
    B, H, N, D = q.shape
    E = v.shape[-1]
    o = torch.zeros(B, H, N, E, dtype=torch.float32, device=q.device)
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK
    NUM_CBLOCK = BLOCK // CBLOCK

    for b_idx in range(B):
        for h_idx in range(H):
            slope = s[h_idx].item()
            for blk in range(NUM_BLOCK):
                block_start = blk * BLOCK
                block_end = min(block_start + BLOCK, N)
                for ci in range(NUM_CBLOCK):
                    q_start = block_start + ci * CBLOCK
                    q_end = min(q_start + CBLOCK, N)
                    if q_start >= N:
                        break
                    q_len = q_end - q_start
                    q_slice = q[b_idx, h_idx, q_start:q_end].float()

                    acc = torch.zeros(q_len, E, device=q.device, dtype=torch.float32)
                    for cj in range(ci + 1):
                        kv_start = block_start + cj * CBLOCK
                        kv_end = min(kv_start + CBLOCK, N)
                        kv_len = kv_end - kv_start

                        k_slice = k[b_idx, h_idx, kv_start:kv_end].float()
                        v_slice = v[b_idx, h_idx, kv_start:kv_end].float()

                        # Compute decay matrix
                        q_idx = torch.arange(ci * CBLOCK, ci * CBLOCK + q_len, device=q.device)
                        kv_idx = torch.arange(cj * CBLOCK, cj * CBLOCK + kv_len, device=q.device)
                        diff = q_idx[:, None] - kv_idx[None, :]
                        s_index = slope * diff
                        s_index = torch.where(diff >= 0, -s_index, torch.tensor(float('-inf'), device=q.device))
                        decay = torch.exp(s_index)

                        qk = (q_slice @ k_slice.T) * decay
                        acc += qk @ v_slice

                    o[b_idx, h_idx, q_start:q_end] = acc

    return o


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lightning_attn_diag_forward"), "Missing lightning_attn_diag_forward"
        assert hasattr(mod, "_fwd_diag_kernel"), "Missing _fwd_diag_kernel"
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
    dtype = torch.float16

    for i, (B, H, N, D, E) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01

            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, E, device=device, dtype=dtype)

            # Run Triton kernel
            out = mod.lightning_attn_diag_forward(q, k, v, s)
            torch.cuda.synchronize()

            # Run reference (in float32)
            ref = reference_diag_attention(q, k, v, s)

            # Compare in float32 to avoid extra fp16 rounding noise.
            # Use tolerances that accommodate hardware matrix-core
            # precision differences (fp16 tensor cores vs f32 GEMM).
            if not torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-3):
                max_diff = (out.float() - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, H={H}, N={N}, D={D}, E={E}): "
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
    dtype = torch.float16
    test_cases = []

    for test_idx, (B, H, N, D, E) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, E, device=device, dtype=dtype)

            # Warmup
            def _bench_fn():
                mod.lightning_attn_diag_forward(q, k, v, s)
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
