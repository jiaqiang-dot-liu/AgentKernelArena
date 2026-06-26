#!/usr/bin/env python3
"""Task runner for triton_eagle_prepare_next_token_padded"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_eagle_prepare_next_token_padded"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_eagle_prepare_next_token_padded.py")

# (num_reqs, num_sampled_tokens_per_req, vocab_size)
TEST_SHAPES = [
    (4, 4, 32000),
    (8, 6, 32000),
    (16, 8, 50000),
    (32, 5, 32000),
    (64, 7, 128000),
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


def make_inputs(num_reqs, num_sampled, vocab_size, device="cpu"):
    import torch
    torch.manual_seed(42)
    # Sampled token IDs: mix of valid tokens and -1 (rejected)
    sampled = torch.randint(0, vocab_size, (num_reqs, num_sampled), dtype=torch.int32)
    # Randomly mark some tokens as rejected (-1)
    reject_mask = torch.rand(num_reqs, num_sampled) < 0.3
    sampled[reject_mask] = -1
    # Some requests fully rejected
    if num_reqs > 2:
        sampled[0] = -1  # all rejected

    discard_mask = torch.zeros(num_reqs, dtype=torch.bool)
    if num_reqs > 3:
        discard_mask[1] = True

    backup = torch.randint(0, vocab_size, (num_reqs,), dtype=torch.int32)

    if device != "cpu":
        sampled = sampled.to(device)
        discard_mask = discard_mask.to(device)
        backup = backup.to(device)
    return sampled, discard_mask, backup


def reference(sampled, discard_mask, backup, vocab_size):
    import torch
    num_reqs, num_sampled = sampled.shape
    next_tokens = torch.empty(num_reqs, dtype=torch.int32)
    valid_counts = torch.empty(num_reqs, dtype=torch.int32)

    for i in range(num_reqs):
        if discard_mask[i]:
            next_tokens[i] = backup[i]
            valid_counts[i] = 0
        else:
            valid_indices = []
            for j in range(num_sampled):
                tid = sampled[i, j].item()
                if tid != -1 and tid < vocab_size:
                    valid_indices.append(j)
            vc = len(valid_indices)
            valid_counts[i] = vc
            if vc > 0:
                last_idx = max(valid_indices)
                next_tokens[i] = sampled[i, last_idx].item()
            else:
                next_tokens[i] = backup[i]
    return next_tokens, valid_counts


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "eagle_prepare_next_token_padded"), "Missing wrapper"
        assert hasattr(mod, "eagle_prepare_next_token_padded_kernel"), "Missing kernel"
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
    for i, (nr, ns, vs) in enumerate(TEST_SHAPES):
        try:
            sampled, dm, backup = make_inputs(nr, ns, vs, device)
            res_next, res_vc = mod.eagle_prepare_next_token_padded(sampled, dm, backup, vs)
            torch.cuda.synchronize()

            ref_next, ref_vc = reference(sampled.cpu(), dm.cpu(), backup.cpu(), vs)
            if not torch.equal(res_next.cpu(), ref_next):
                return False, f"Shape {i+1}: next_token_ids mismatch"
            if not torch.equal(res_vc.cpu(), ref_vc):
                return False, f"Shape {i+1}: valid_count mismatch"
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

    for test_idx, (nr, ns, vs) in enumerate(TEST_SHAPES):
        try:
            sampled, dm, backup = make_inputs(nr, ns, vs, device)

            def _bench_fn():
                mod.eagle_prepare_next_token_padded(sampled, dm, backup, vs)
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
                    "num_reqs": nr,
                    "num_sampled_tokens_per_req": ns,
                    "vocab_size": vs
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": nr,
                    "num_sampled_tokens_per_req": ns,
                    "vocab_size": vs
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
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
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
