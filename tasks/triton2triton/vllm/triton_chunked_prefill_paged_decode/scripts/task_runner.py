#!/usr/bin/env python3
"""Task runner for triton2triton/triton_chunked_prefill_paged_decode"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_chunked_prefill_paged_decode"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_chunked_prefill_paged_decode.py")

# Test configs: (num_seqs, seq_len_k, num_query_heads, num_kv_heads, head_size, block_size, x_factor)
TEST_SHAPES = [
    (4, 64, 8, 8, 64, 16, 8),
    (2, 128, 16, 4, 64, 16, 8),
    (8, 256, 32, 8, 128, 16, 8),
    (4, 128, 16, 16, 64, 32, 8),
    (2, 512, 8, 8, 128, 32, 8),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers — edit tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
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

        metadata.update({
            "benchmark_method": "cuda_graph",
            "benchmark_effective_repeats": int(n_repeat),
        })
        return sum(retry_times) / len(retry_times), metadata
    except Exception as exc:
        torch.cuda.synchronize()
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}",
        })
        return sum(times) / len(times), metadata
# <<< AKA-GENERATED <<<

def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(num_seqs, seq_len_k, num_query_heads, num_kv_heads,
                   head_size, block_size, x_factor, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.float16

    # Each seq has 1 query token (decode)
    total_tokens = num_seqs
    query = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)

    num_blocks_per_seq = (seq_len_k + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4

    # 5D K cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    key_cache = torch.randn(total_blocks, num_kv_heads, head_size // x_factor,
                            block_size, x_factor, device=device, dtype=dtype)
    # 4D V cache: [num_blocks, num_kv_heads, head_size, block_size]
    value_cache = torch.randn(total_blocks, num_kv_heads, head_size, block_size,
                              device=device, dtype=dtype)

    block_table = torch.zeros(num_seqs, num_blocks_per_seq, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b

    seq_lens = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)

    # query_start_loc for decode: each seq has 1 token
    query_start_loc = torch.arange(0, num_seqs + 1, device=device, dtype=torch.int32)

    output = torch.zeros_like(query)
    scale = 1.0 / (head_size ** 0.5)

    return query, output, key_cache, value_cache, block_table, seq_lens, query_start_loc, scale


def reference_attention(query, key_cache, value_cache, block_table, seq_lens,
                        scale, block_size, x_factor):
    """CPU reference for paged attention with 5D K / 4D V cache."""
    import torch
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads

    output = torch.zeros_like(query, dtype=torch.float32)

    for s in range(num_seqs):
        k_len = seq_lens[s].item()

        # Gather K from 5D cache
        k_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=query.device, dtype=query.dtype)
        v_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=query.device, dtype=query.dtype)

        for t in range(k_len):
            bi = t // block_size
            bo = t % block_size
            pb = block_table[s, bi].item()
            # K: [num_blocks, num_kv_heads, head_size//x, block_size, x]
            for kv_h in range(num_kv_heads):
                for d in range(head_size):
                    d_outer = d // x_factor
                    d_inner = d % x_factor
                    k_gathered[t, kv_h, d] = key_cache[pb, kv_h, d_outer, bo, d_inner]
                # V: [num_blocks, num_kv_heads, head_size, block_size]
                v_gathered[t, kv_h, :] = value_cache[pb, kv_h, :, bo]

        for h in range(num_query_heads):
            kv_h = h // num_queries_per_kv
            Q_h = query[s, h, :].float()
            K_h = k_gathered[:, kv_h, :].float()
            V_h = v_gathered[:, kv_h, :].float()

            S = (Q_h @ K_h.T) * scale
            # Decode: no causal mask needed beyond seq_len (already bounded)
            # Just ensure positions beyond seq_len are masked
            S_max = S.max()
            P = torch.exp(S - S_max)
            P = P / P.sum()
            output[s, h, :] = P @ V_h

    return output.to(query.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "chunked_prefill_paged_decode"), "Missing chunked_prefill_paged_decode"
        assert hasattr(mod, "kernel_paged_attention_2d"), "Missing kernel_paged_attention_2d"
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

    for i, (num_seqs, slk, nqh, nkvh, hs, bs, xf) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale = \
                make_test_data(num_seqs, slk, nqh, nkvh, hs, bs, xf, device, dtype)

            mod.chunked_prefill_paged_decode(
                query, output, key_cache, value_cache, block_table,
                seq_lens, qsl, scale, filter_by_query_len=False,
            )
            torch.cuda.synchronize()

            ref = reference_attention(
                query, key_cache, value_cache, block_table,
                seq_lens, scale, bs, xf,
            )

            if not torch.allclose(output.float(), ref.float(), atol=1e-2, rtol=1e-2):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

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

    for test_idx, (num_seqs, slk, nqh, nkvh, hs, bs, xf) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale = \
                make_test_data(num_seqs, slk, nqh, nkvh, hs, bs, xf, device, dtype)

            def _bench_fn():
                mod.chunked_prefill_paged_decode(
                    query, output, key_cache, value_cache, block_table,
                    seq_lens, qsl, scale, filter_by_query_len=False,
                )
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
                    "num_seqs": num_seqs,
                    "seq_len_k": slk,
                    "num_query_heads": nqh,
                    "num_kv_heads": nkvh,
                    "head_size": hs,
                    "block_size": bs,
                    "x_factor": xf
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_seqs": num_seqs,
                    "seq_len_k": slk,
                    "num_query_heads": nqh,
                    "num_kv_heads": nkvh,
                    "head_size": hs,
                    "block_size": bs,
                    "x_factor": xf
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
