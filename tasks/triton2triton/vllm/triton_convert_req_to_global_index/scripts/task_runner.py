#!/usr/bin/env python3
"""Task runner for triton2triton/triton_convert_req_to_global_index"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_convert_req_to_global_index"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_convert_req_to_global_index.py")

# Test configs: (num_tokens, num_requests, max_blocks_per_req, num_topk_tokens, block_size, block_n)
TEST_SHAPES = [
    (16, 4, 8, 128, 64, 128),
    (32, 8, 16, 256, 64, 128),
    (64, 16, 32, 512, 64, 128),
    (128, 32, 16, 256, 32, 128),
    (256, 64, 32, 512, 64, 128),
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


def make_test_data(num_tokens, num_requests, max_blocks_per_req, num_topk_tokens,
                   block_size, device="cuda"):
    import torch
    # req_id: each token assigned to a random request
    req_id = torch.randint(0, num_requests, (num_tokens,), device=device, dtype=torch.int32)

    # block_table: random physical block indices
    block_table = torch.randint(0, 1000, (num_requests, max_blocks_per_req),
                                device=device, dtype=torch.int32)

    # token_indices: random token indices, some set to -1
    max_token_idx = max_blocks_per_req * block_size
    token_indices = torch.randint(0, max_token_idx, (num_tokens, num_topk_tokens),
                                  device=device, dtype=torch.int32)
    # Set ~10% to -1 (invalid)
    mask = torch.rand(num_tokens, num_topk_tokens, device=device) < 0.1
    token_indices[mask] = -1

    return req_id, block_table, token_indices


def reference_convert(req_id, block_table, token_indices, block_size):
    """CPU reference implementation."""
    import torch
    num_tokens = req_id.shape[0]
    num_topk = token_indices.shape[1]
    max_blocks = block_table.shape[1]
    out = torch.empty_like(token_indices)

    for i in range(num_tokens):
        r = req_id[i].item()
        for j in range(num_topk):
            tok = token_indices[i, j].item()
            if tok < 0:
                out[i, j] = -1
                continue
            block_id = tok // block_size
            inblock_off = tok % block_size
            if block_id < 0 or block_id >= max_blocks:
                out[i, j] = -1
            else:
                base = block_table[r, block_id].item()
                out[i, j] = base * block_size + inblock_off

    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "convert_req_to_global_index"), "Missing convert_req_to_global_index"
        assert hasattr(mod, "_convert_req_index_to_global_index_kernel"), \
            "Missing _convert_req_index_to_global_index_kernel"
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

    for i, (nt, nr, mbpr, ntk, bs, bn) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            req_id, block_table, token_indices = make_test_data(nt, nr, mbpr, ntk, bs, device)

            result = mod.convert_req_to_global_index(
                req_id, block_table, token_indices,
                BLOCK_SIZE=bs, BLOCK_N=bn,
            )
            torch.cuda.synchronize()

            ref = reference_convert(req_id, block_table, token_indices, bs)

            if not torch.equal(result, ref):
                diff_mask = result != ref
                num_diffs = diff_mask.sum().item()
                return False, f"Shape {i+1}: {num_diffs} mismatched entries"

            # Also test with valid counts
            result2, counts = mod.convert_req_to_global_index(
                req_id, block_table, token_indices,
                BLOCK_SIZE=bs, BLOCK_N=bn, return_valid_counts=True,
            )
            torch.cuda.synchronize()

            ref_counts = (ref != -1).sum(dim=1).to(torch.int32)
            if not torch.equal(counts, ref_counts):
                return False, f"Shape {i+1}: valid counts mismatch"

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

    for test_idx, (nt, nr, mbpr, ntk, bs, bn) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            req_id, block_table, token_indices = make_test_data(nt, nr, mbpr, ntk, bs, device)

            def _bench_fn():
                mod.convert_req_to_global_index(
                    req_id, block_table, token_indices,
                    BLOCK_SIZE=bs, BLOCK_N=bn,
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
                    "num_tokens": nt,
                    "num_requests": nr,
                    "max_blocks_per_req": mbpr,
                    "num_topk_tokens": ntk,
                    "block_size": bs,
                    "block_n": bn
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_tokens": nt,
                    "num_requests": nr,
                    "max_blocks_per_req": mbpr,
                    "num_topk_tokens": ntk,
                    "block_size": bs,
                    "block_n": bn
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
