#!/usr/bin/env python3
"""Task runner for triton2triton/triton_sample_recovered_tokens"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_sample_recovered_tokens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_sample_recovered_tokens.py")

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
        assert hasattr(mod, "sample_recovered_tokens"), "Missing sample_recovered_tokens"
        assert hasattr(mod, "sample_recovered_tokens_kernel"), "Missing sample_recovered_tokens_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 3, 64),   # (batch, max_draft, vocab)
    (8, 5, 128),
    (16, 4, 256),
    (32, 6, 512),
    (64, 8, 1024),
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

def reference_sample_recovered_tokens(
    cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, q, vocab_size
):
    import torch
    batch_size = cu_num_draft_tokens.shape[0]
    out = torch.empty_like(draft_token_ids)
    start = 0
    for req in range(batch_size):
        end = cu_num_draft_tokens[req].item()
        for idx in range(start, end):
            if draft_probs is None:
                prob = target_probs[idx].clone()
                prob[draft_token_ids[idx]] = 0
            else:
                prob = torch.maximum(target_probs[idx] - draft_probs[idx], torch.zeros_like(target_probs[idx]))
            out[idx] = torch.argmax(prob / q[req]).to(out.dtype)
        start = end
    return out

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, max_draft, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_per_req = [(max_draft - (j % 2)) for j in range(batch_size)]
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32), dim=0).to(device)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            result = mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
            ref = reference_sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, vocab_size)
            if not torch.equal(result, ref):
                return False, f"Shape {i+1}: mismatch with draft_probs path"

            # Also test NO_DRAFT_PROBS path.
            result_no_draft = mod.sample_recovered_tokens(cu, draft_ids, None, target_probs, q, max_draft, vocab_size)
            ref_no_draft = reference_sample_recovered_tokens(cu, draft_ids, None, target_probs, q, vocab_size)
            if not torch.equal(result_no_draft, ref_no_draft):
                return False, f"Shape {i+1}: mismatch with NO_DRAFT_PROBS path"
            torch.cuda.synchronize()
            assert result.shape == (total,), f"Wrong shape: {result.shape}"
            assert result.min() >= 0 and result.max() < vocab_size
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch_size, max_draft, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            num_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32), dim=0).to(device)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            def _bench_fn():
                mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
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
                    "batch": batch_size,
                    "max_draft": max_draft,
                    "vocab": vocab_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch_size,
                    "max_draft": max_draft,
                    "vocab": vocab_size
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

if __name__ == "__main__": main()
