#!/usr/bin/env python3
"""Task runner for triton2triton/triton_prepare_prefill_inputs"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_prepare_prefill_inputs"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_prefill_inputs.py")

# Test configurations: (num_reqs, max_seq_len, query_len)
TEST_SHAPES = [
    (4, 128, 32),
    (8, 256, 64),
    (16, 512, 128),
    (32, 1024, 256),
    (64, 2048, 512),
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


def reference_prepare_prefill_inputs(
    idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens
):
    """CPU reference implementation."""
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1].item())
    input_ids = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")
    next_prefill_tokens = torch.zeros(idx_mapping.max().item() + 1, dtype=torch.int32, device="cpu")

    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        plen = prefill_len[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()
        if num_computed >= plen:
            continue
        qstart = query_start_loc[b].item()
        qend = query_start_loc[b + 1].item()
        qlen = qend - qstart
        for k in range(qlen):
            input_ids[qstart + k] = all_token_ids[req_state_idx, num_computed + k]
        next_pos = num_computed + qlen
        if next_pos < plen:
            next_prefill_tokens[req_state_idx] = all_token_ids[req_state_idx, next_pos]

    return input_ids, next_prefill_tokens


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_prefill_inputs"), "Missing prepare_prefill_inputs"
        assert hasattr(mod, "_prepare_prefill_inputs_kernel"), "Missing _prepare_prefill_inputs_kernel"
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

    for i, (num_reqs, max_seq_len, query_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 16

            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len

            total_tokens = int(query_start_loc[-1].item())
            all_token_ids = torch.randint(0, 32000, (max_num_reqs, max_seq_len), dtype=torch.int32, device=device)
            prefill_len = torch.full((max_num_reqs,), max_seq_len, dtype=torch.int32, device=device)
            num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
            next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            mod.prepare_prefill_inputs(
                input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                all_token_ids, prefill_len, num_computed_tokens,
            )
            torch.cuda.synchronize()

            ref_ids, ref_next = reference_prepare_prefill_inputs(
                idx_mapping.cpu(), query_start_loc.cpu(), all_token_ids.cpu(),
                prefill_len.cpu(), num_computed_tokens.cpu(),
            )

            if not torch.equal(input_ids.cpu(), ref_ids):
                return False, f"Shape {i+1}: input_ids mismatch"
            if not torch.equal(next_prefill_tokens.cpu()[:num_reqs], ref_next[:num_reqs]):
                return False, f"Shape {i+1}: next_prefill_tokens mismatch"

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
    test_cases = []

    for test_idx, (num_reqs, max_seq_len, query_len) in enumerate(TEST_SHAPES):
        try:
            max_num_reqs = num_reqs + 16
            torch.manual_seed(42 + test_idx)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())
            all_token_ids = torch.randint(0, 32000, (max_num_reqs, max_seq_len), dtype=torch.int32, device=device)
            prefill_len = torch.full((max_num_reqs,), max_seq_len, dtype=torch.int32, device=device)
            num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
            next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            def _bench_fn():
                mod.prepare_prefill_inputs(
                    input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                    all_token_ids, prefill_len, num_computed_tokens,
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
                    "num_reqs": num_reqs,
                    "max_seq_len": max_seq_len,
                    "query_len": query_len
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "max_seq_len": max_seq_len,
                    "query_len": query_len
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
