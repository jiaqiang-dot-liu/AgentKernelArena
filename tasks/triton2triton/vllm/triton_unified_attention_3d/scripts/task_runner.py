#!/usr/bin/env python3
"""Task runner for triton2triton/triton_unified_attention_3d"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_unified_attention_3d"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_unified_attention_3d.py")

# Test configs: (num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads, head_size, block_size, num_segments)
TEST_SHAPES = [
    (2, 1, 128, 8, 8, 64, 16, 2),
    (1, 1, 256, 16, 4, 64, 16, 4),
    (4, 1, 512, 32, 8, 128, 16, 4),
    (2, 1, 128, 16, 16, 64, 32, 2),
    (1, 1, 1024, 8, 8, 128, 32, 4),
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


def make_test_data(num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads,
                   head_size, block_size, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.float16

    total_tokens = num_seqs * seq_len_q
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)

    num_blocks_per_seq = (seq_len_k + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4
    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size,
                            device=device, dtype=dtype)
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size,
                              device=device, dtype=dtype)

    block_table = torch.zeros(num_seqs, num_blocks_per_seq, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b

    cu_seqlens_q = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        cu_seqlens_q[s + 1] = cu_seqlens_q[s] + seq_len_q

    seqused_k = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)
    scale = 1.0 / (head_size ** 0.5)

    return q, key_cache, value_cache, block_table, cu_seqlens_q, seqused_k, scale


def reference_attention_3d(q, key_cache, value_cache, block_table, cu_seqlens_q,
                           seqused_k, scale, block_size, num_segments):
    """CPU reference: compute per-segment partial attention outputs."""
    import torch
    import triton
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    num_kv_heads = key_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    total_tokens = q.shape[0]

    segm_output = torch.zeros(total_tokens, num_query_heads, num_segments, head_size_padded,
                              device=q.device, dtype=torch.float32)
    segm_max = torch.full((total_tokens, num_query_heads, num_segments),
                          float("-inf"), device=q.device, dtype=torch.float32)
    segm_expsum = torch.zeros(total_tokens, num_query_heads, num_segments,
                              device=q.device, dtype=torch.float32)

    TILE_SIZE = 16

    for s in range(num_seqs):
        q_start = cu_seqlens_q[s].item()
        q_end = cu_seqlens_q[s + 1].item()
        q_len = q_end - q_start
        k_len = seqused_k[s].item()

        # Gather K, V
        k_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=q.device, dtype=q.dtype)
        v_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=q.device, dtype=q.dtype)
        for t in range(k_len):
            bi = t // block_size
            bo = t % block_size
            pb = block_table[s, bi].item()
            k_gathered[t] = key_cache[pb, bo]
            v_gathered[t] = value_cache[pb, bo]

        tiles_per_segment = (k_len + num_segments * TILE_SIZE - 1) // (num_segments * TILE_SIZE)
        context_len = k_len - q_len

        for qi_local in range(q_len):
            qi_global = q_start + qi_local
            for h in range(num_query_heads):
                kv_h = h // num_queries_per_kv
                Q_h = q[qi_global, h, :].float()

                for seg in range(num_segments):
                    tile_lo = seg * tiles_per_segment * TILE_SIZE
                    tile_hi = min((seg + 1) * tiles_per_segment * TILE_SIZE, k_len)
                    if tile_lo >= k_len:
                        continue

                    K_seg = k_gathered[tile_lo:tile_hi, kv_h, :].float()
                    V_seg = v_gathered[tile_lo:tile_hi, kv_h, :].float()

                    scores = (Q_h @ K_seg.T) * scale

                    # Causal mask
                    for ki in range(scores.shape[0]):
                        abs_ki = tile_lo + ki
                        if abs_ki > context_len + qi_local:
                            scores[ki] = float("-inf")

                    max_s = scores.max().item()
                    if max_s == float("-inf"):
                        continue
                    exp_s = torch.exp(scores - max_s)
                    sum_exp = exp_s.sum().item()

                    out_seg = (exp_s @ V_seg)  # [head_size]

                    segm_output[qi_global, h, seg, :head_size] = out_seg
                    segm_max[qi_global, h, seg] = max_s
                    segm_expsum[qi_global, h, seg] = sum_exp

    return segm_output, segm_max, segm_expsum


def reduce_segments_ref(segm_output, segm_max, segm_expsum, head_size):
    """Reduce segment partials to final output using logsumexp."""
    import torch
    total_tokens = segm_output.shape[0]
    num_heads = segm_output.shape[1]
    output = torch.zeros(total_tokens, num_heads, head_size, device=segm_output.device, dtype=torch.float32)

    overall_max = segm_max.max(dim=-1).values  # [tokens, heads]
    rescaled_expsum = segm_expsum * torch.exp(segm_max - overall_max.unsqueeze(-1))
    overall_expsum = rescaled_expsum.sum(dim=-1)  # [tokens, heads]

    rescaled_output = segm_output * torch.exp(segm_max - overall_max.unsqueeze(-1)).unsqueeze(-1)
    summed = rescaled_output.sum(dim=2)  # [tokens, heads, head_size_padded]

    safe_denom = overall_expsum.unsqueeze(-1)
    safe_denom = torch.where(safe_denom == 0, torch.ones_like(safe_denom), safe_denom)
    output = summed[:, :, :head_size] / safe_denom

    return output


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "unified_attention_3d"), "Missing unified_attention_3d"
        assert hasattr(mod, "kernel_unified_attention_3d"), "Missing kernel_unified_attention_3d"
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

    for i, (num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, nseg) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, key_cache, value_cache, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, device, dtype)

            segm_out, segm_max_out, segm_expsum_out = mod.unified_attention_3d(
                q, key_cache, value_cache, block_table,
                cu_seqlens_q, seqused_k, scale, num_segments=nseg,
            )
            torch.cuda.synchronize()

            # Reduce to final output
            result = reduce_segments_ref(segm_out, segm_max_out, segm_expsum_out, hs)

            # Reference: full attention
            ref_segm_out, ref_segm_max, ref_segm_expsum = reference_attention_3d(
                q, key_cache, value_cache, block_table,
                cu_seqlens_q, seqused_k, scale, bs, nseg,
            )
            ref = reduce_segments_ref(ref_segm_out, ref_segm_max, ref_segm_expsum, hs)

            if not torch.allclose(result.float(), ref.float(), atol=1e-2, rtol=1e-2):
                max_diff = (result.float() - ref.float()).abs().max().item()
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

    for test_idx, (num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, nseg) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            q, key_cache, value_cache, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, device, dtype)

            def _bench_fn():
                mod.unified_attention_3d(
                    q, key_cache, value_cache, block_table,
                    cu_seqlens_q, seqused_k, scale, num_segments=nseg,
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
                    "seq_len_q": seq_len_q,
                    "seq_len_k": seq_len_k,
                    "num_query_heads": nqh,
                    "num_kv_heads": nkvh,
                    "head_size": hs,
                    "block_size": bs,
                    "num_segments": nseg
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_seqs": num_seqs,
                    "seq_len_q": seq_len_q,
                    "seq_len_k": seq_len_k,
                    "num_query_heads": nqh,
                    "num_kv_heads": nkvh,
                    "head_size": hs,
                    "block_size": bs,
                    "num_segments": nseg
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
