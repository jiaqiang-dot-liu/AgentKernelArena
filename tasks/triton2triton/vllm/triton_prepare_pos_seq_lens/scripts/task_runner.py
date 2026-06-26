#!/usr/bin/env python3
"""Task runner for triton2triton/triton_prepare_pos_seq_lens"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_prepare_pos_seq_lens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_pos_seq_lens.py")

# (num_reqs, max_num_reqs, query_len, num_computed_base)
TEST_SHAPES = [
    (4, 32, 16, 0),
    (8, 64, 32, 10),
    (16, 128, 64, 50),
    (32, 256, 128, 100),
    (64, 512, 256, 200),
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


def reference_prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, max_num_reqs):
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1].item())
    pos = torch.zeros(total_tokens, dtype=torch.int64, device="cpu")
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device="cpu")

    for r in range(num_reqs):
        req_state_idx = idx_mapping[r].item()
        nc = num_computed_tokens[req_state_idx].item()
        start = query_start_loc[r].item()
        end = query_start_loc[r + 1].item()
        qlen = end - start
        seq_lens[r] = nc + qlen
        for k in range(qlen):
            pos[start + k] = nc + k

    return pos, seq_lens


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_pos_seq_lens"), "Missing prepare_pos_seq_lens"
        assert hasattr(mod, "_prepare_pos_seq_lens_kernel"), "Missing _prepare_pos_seq_lens_kernel"
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
    for i, (num_reqs, max_num_reqs, query_len, nc_base) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())
            num_computed_tokens = torch.full((max_num_reqs,), nc_base, dtype=torch.int32, device=device)
            pos = torch.zeros(total_tokens, dtype=torch.int64, device=device)
            seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            mod.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
            torch.cuda.synchronize()

            ref_pos, ref_seq = reference_prepare_pos_seq_lens(
                idx_mapping.cpu(), query_start_loc.cpu(), num_computed_tokens.cpu(), max_num_reqs
            )

            if not torch.equal(pos.cpu(), ref_pos):
                return False, f"Shape {i+1}: pos mismatch"
            if not torch.equal(seq_lens.cpu(), ref_seq):
                return False, f"Shape {i+1}: seq_lens mismatch"
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

    for test_idx, (num_reqs, max_num_reqs, query_len, nc_base) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())
            num_computed_tokens = torch.full((max_num_reqs,), nc_base, dtype=torch.int32, device=device)
            pos = torch.zeros(total_tokens, dtype=torch.int64, device=device)
            seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            def _bench_fn():
                mod.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
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
                    "max_num_reqs": max_num_reqs,
                    "query_len": query_len,
                    "num_computed_base": nc_base
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "max_num_reqs": max_num_reqs,
                    "query_len": query_len,
                    "num_computed_base": nc_base
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
