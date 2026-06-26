#!/usr/bin/env python3
"""Task runner for triton2triton/triton_decode_attn_stage1"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_decode_attn_stage1"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "_fwd_kernel_stage1.py")

# Test configurations: (bs, num_heads, num_kv_heads, head_dim, max_seq, num_kv_splits, page_size)
TEST_SHAPES = [
    (1, 8, 8, 64, 128, 4, 16),
    (4, 16, 4, 64, 256, 8, 16),
    (2, 32, 8, 128, 512, 4, 32),
    (1, 8, 1, 64, 64, 2, 16),    # MQA
    (8, 8, 8, 64, 128, 4, 16),
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
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_stage1(q, k_buffer, v_buffer, req_to_tokens, b_seqlen,
                     num_kv_splits, sm_scale, page_size):
    """
    CPU/PyTorch reference for decode attention stage1.

    For each (batch, head, kv_split), compute partial attention over
    the assigned KV range and return partial output + logsumexp.
    """
    import torch
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    Lv = v_buffer.shape[-1]

    att_out = torch.zeros(batch, num_heads, num_kv_splits, Lv + 1,
                          device=q.device, dtype=torch.float32)

    for b in range(batch):
        seq_len = b_seqlen[b].item()
        kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_vec = q[b, h, :].float()  # [head_dim]

            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, seq_len)
                if end <= start:
                    continue

                # Gather K and V via paged token table
                positions = torch.arange(start, end, device=q.device)
                page_nums = req_to_tokens[b, positions // page_size]
                kv_locs = page_nums * page_size + positions % page_size

                k_vals = k_buffer[kv_locs, kv_h, :].float()  # [length, head_dim]
                v_vals = v_buffer[kv_locs, kv_h, :].float()  # [length, Lv]

                # Q @ K^T * sm_scale
                scores = (k_vals @ q_vec) * sm_scale  # [length]

                # Numerically stable softmax
                max_score = scores.max()
                exp_scores = torch.exp(scores - max_score)
                sum_exp = exp_scores.sum()

                # Partial output
                partial_out = (exp_scores.unsqueeze(-1) * v_vals).sum(0) / sum_exp

                att_out[b, h, s, :Lv] = partial_out
                att_out[b, h, s, Lv] = max_score + torch.log(sum_exp)

    return att_out


def make_inputs(bs, num_heads, num_kv_heads, head_dim, max_seq, num_kv_splits,
                page_size, device="cuda", dtype=None):
    """Create test inputs for the stage1 kernel."""
    import torch
    if dtype is None:
        dtype = torch.float16

    torch.manual_seed(42)

    q = torch.randn(bs, num_heads, head_dim, device=device, dtype=dtype)

    # Total tokens in KV buffer: use enough pages
    max_pages_per_seq = (max_seq + page_size - 1) // page_size
    total_pages = bs * max_pages_per_seq
    total_tokens = total_pages * page_size

    k_buffer = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_buffer = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Build req_to_tokens: identity mapping for simplicity (page i -> page i)
    max_seq_padded = max_pages_per_seq * page_size
    req_to_tokens = torch.zeros(bs, max_seq_padded, device=device, dtype=torch.int32)
    for b in range(bs):
        for pos in range(max_seq):
            page_idx = b * max_pages_per_seq + pos // page_size
            req_to_tokens[b, pos] = page_idx

    b_seqlen = torch.full((bs,), max_seq, device=device, dtype=torch.int32)

    # att_out: [batch, num_heads, num_kv_splits, head_dim + 1]
    att_out = torch.zeros(bs, num_heads, num_kv_splits, head_dim + 1,
                          device=device, dtype=torch.float32)

    sm_scale = 1.0 / (head_dim ** 0.5)

    return q, k_buffer, v_buffer, att_out, req_to_tokens, b_seqlen, sm_scale


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "decode_att_m_fwd"), "Missing decode_att_m_fwd"
        assert hasattr(mod, "_fwd_kernel_stage1"), "Missing _fwd_kernel_stage1"
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

    for i, (bs, nh, nkv, hd, max_seq, num_splits, ps) in enumerate(TEST_SHAPES):
        try:
            q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen, sm_scale = \
                make_inputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype)

            mod.decode_att_m_fwd(
                q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen,
                num_splits, sm_scale, ps, logit_cap=0.0,
            )
            torch.cuda.synchronize()

            ref = reference_stage1(q, k_buf, v_buf, req_to_tokens, b_seqlen,
                                   num_splits, sm_scale, ps)

            if not torch.allclose(att_out, ref, atol=1e-2, rtol=1e-2):
                max_diff = (att_out - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, "
                    f"seq={max_seq}, splits={num_splits}, ps={ps}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, "
                f"seq={max_seq}, splits={num_splits}, ps={ps}): "
                f"exception: {e}"
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

    for test_idx, (bs, nh, nkv, hd, max_seq, num_splits, ps) in enumerate(TEST_SHAPES):
        try:
            q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen, sm_scale = \
                make_inputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype)

            # Warmup
            for _ in range(WARMUP_ITERATIONS):
                att_out.zero_()
                mod.decode_att_m_fwd(
                    q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen,
                    num_splits, sm_scale, ps, logit_cap=0.0,
                )
            torch.cuda.synchronize()

            # Benchmark
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                att_out.zero_()
                start_events[j].record()
                mod.decode_att_m_fwd(
                    q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen,
                    num_splits, sm_scale, ps, logit_cap=0.0,
                )
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
                    "batch_size": bs,
                    "num_heads": nh,
                    "num_kv_heads": nkv,
                    "head_dim": hd,
                    "max_seq": max_seq,
                    "num_kv_splits": num_splits,
                    "page_size": ps
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch_size": bs,
                    "num_heads": nh,
                    "num_kv_heads": nkv,
                    "head_dim": hd,
                    "max_seq": max_seq,
                    "num_kv_splits": num_splits,
                    "page_size": ps
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
