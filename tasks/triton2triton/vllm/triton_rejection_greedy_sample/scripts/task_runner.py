#!/usr/bin/env python3
"""Task runner for triton2triton/triton_rejection_greedy_sample"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_rejection_greedy_sample"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_rejection_greedy_sample.py")

TEST_SHAPES = [
    (4, 3, 8),   # (batch_size, max_draft_tokens, max_spec_len)
    (8, 5, 16),
    (16, 4, 16),
    (32, 6, 32),
    (64, 8, 64),
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

def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f: source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "rejection_greedy_sample"), "Missing rejection_greedy_sample"
        assert hasattr(mod, "rejection_greedy_sample_kernel"), "Missing rejection_greedy_sample_kernel"
        return True, None
    except Exception as e:
        return False, str(e)

def cpu_reference(draft_token_ids_list, target_argmax_list, bonus_token_ids, max_spec_len):
    """CPU reference for greedy rejection sampling."""
    import torch
    batch_size = len(draft_token_ids_list)
    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32)
    for b in range(batch_size):
        draft = draft_token_ids_list[b]
        target = target_argmax_list[b]
        rejected = False
        for pos in range(len(draft)):
            if not rejected:
                output[b, pos] = target[pos]
                if draft[pos] != target[pos]:
                    rejected = True
        if not rejected:
            output[b, len(draft)] = bonus_token_ids[b]
    return output

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, max_draft, max_spec_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            vocab_size = 100
            num_draft_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_draft_per_req, dtype=torch.int32), dim=0).to(device)
            total_tokens = sum(num_draft_per_req)
            draft_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
            target_argmax = draft_ids.clone()
            # Make some mismatches
            for b in range(batch_size):
                start = sum(num_draft_per_req[:b])
                if max_draft > 1:
                    pos = torch.randint(0, max_draft, (1,)).item()
                    target_argmax[start + pos] = (draft_ids[start + pos] + 1) % vocab_size
            bonus = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
            output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
            mod.rejection_greedy_sample(output, cu, draft_ids, target_argmax, bonus, None, max_spec_len)
            torch.cuda.synchronize()
            # CPU ref
            draft_list, target_list = [], []
            for b in range(batch_size):
                start = sum(num_draft_per_req[:b])
                end = start + num_draft_per_req[b]
                draft_list.append(draft_ids[start:end].cpu().tolist())
                target_list.append(target_argmax[start:end].cpu().tolist())
            ref = cpu_reference(draft_list, target_list, bonus.cpu().tolist(), max_spec_len).to(device)
            if not torch.equal(output, ref):
                return False, f"Shape {i+1}: mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch_size, max_draft, max_spec_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            vocab_size = 100
            num_draft_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_draft_per_req, dtype=torch.int32), dim=0).to(device)
            total_tokens = sum(num_draft_per_req)
            draft_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
            target_argmax = draft_ids.clone()
            bonus = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
            output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
            for _ in range(WARMUP_ITERATIONS):
                mod.rejection_greedy_sample(output, cu, draft_ids, target_argmax, bonus, None, max_spec_len)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                output.fill_(-1)
                start_events[j].record()
                mod.rejection_greedy_sample(output, cu, draft_ids, target_argmax, bonus, None, max_spec_len)
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
                    "batch_size": batch_size,
                    "max_draft_tokens": max_draft,
                    "max_spec_len": max_spec_len
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch_size": batch_size,
                    "max_draft_tokens": max_draft,
                    "max_spec_len": max_spec_len
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
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__":
    main()
