#!/usr/bin/env python3
"""Task runner for triton2triton/triton_bad_words"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_bad_words"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_bad_words.py")

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
        assert hasattr(mod, "apply_bad_words"), "Missing apply_bad_words"
        assert hasattr(mod, "_bad_words_kernel"), "Missing _bad_words_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256, 2),   # (batch, vocab, num_bad_words)
    (8, 1024, 4),
    (16, 4096, 8),
    (32, 8192, 16),
    (64, 16384, 8),
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

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab, nbw) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            ref = logits.clone()
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            # Simple: single-token bad words (no prefix matching needed)
            max_tokens = nbw
            bad_word_ids = torch.randint(0, vocab, (batch, max_tokens), dtype=torch.int32, device=device)
            offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
            for b in range(batch):
                for j in range(nbw + 1):
                    offsets[b, j] = j
            num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
            all_token_ids = torch.zeros(batch, 128, dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), 10, dtype=torch.int32, device=device)
            total_len = torch.full((batch,), 20, dtype=torch.int32, device=device)
            input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
            torch.cuda.synchronize()
            # For single-token bad words, the last token should be masked
            for b in range(batch):
                for j in range(nbw):
                    tid = bad_word_ids[b, j].item()
                    ref[b, tid] = float("-inf")
            if not torch.allclose(logits, ref, atol=1e-2, rtol=1e-2):
                diff = (logits - ref).abs().max().item()
                return False, f"Shape {i+1}: max diff = {diff}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    # Multi-token bad words: exercise the prefix-matching loop (prefix_len >= 1),
    # which the single-token cases above never trigger. A bad word [t0..t_{L-1}] must
    # mask its final token t_{L-1} iff the last (L-1) output tokens equal [t0..t_{L-2}].
    multi_cases = [
        (4, 256, 2, 2),    # (batch, vocab, num_bad_words, word_len)
        (8, 512, 3, 2),
        (6, 1024, 3, 3),
    ]
    OUTPUT_LEN = 10
    PROMPT_LEN = 10
    for i, (batch, vocab, nbw, word_len) in enumerate(multi_cases):
        try:
            torch.manual_seed(1234 + i)
            prefix_len = word_len - 1
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            max_tokens = nbw * word_len
            bad_word_ids = torch.randint(0, vocab, (batch, max_tokens), dtype=torch.int32, device=device)
            # Each bad word occupies a contiguous span of length `word_len`.
            offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
            for b in range(batch):
                for j in range(nbw + 1):
                    offsets[b, j] = j * word_len
            num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
            all_token_ids = torch.randint(0, vocab, (batch, 128), dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), PROMPT_LEN, dtype=torch.int32, device=device)
            total_len = torch.full((batch,), PROMPT_LEN + OUTPUT_LEN, dtype=torch.int32, device=device)
            input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
            # Force a prefix match for one bad word per request so the match branch
            # is hit; the other bad words almost surely do not match (no-match branch).
            for b in range(batch):
                jj = b % nbw
                start = jj * word_len
                for t in range(prefix_len):
                    seq_pos = PROMPT_LEN + (OUTPUT_LEN - prefix_len + t)
                    all_token_ids[b, seq_pos] = bad_word_ids[b, start + t]

            # Reference from the intended semantics (pos=0, no spec input).
            ref = logits.clone()
            for b in range(batch):
                for j in range(nbw):
                    start = j * word_len
                    matched = True
                    for t in range(prefix_len):
                        expected = int(bad_word_ids[b, start + t].item())
                        actual = int(all_token_ids[b, PROMPT_LEN + (OUTPUT_LEN - prefix_len + t)].item())
                        if expected != actual:
                            matched = False
                            break
                    if matched:
                        last_token = int(bad_word_ids[b, start + word_len - 1].item())
                        ref[b, last_token] = float("-inf")

            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw,
                                all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
            torch.cuda.synchronize()
            if not torch.allclose(logits, ref, atol=1e-2, rtol=1e-2):
                diff = (logits - ref).abs().max().item()
                return False, f"Multi-token case {i+1} (word_len={word_len}): max diff = {diff}"
        except Exception as e:
            return False, f"Multi-token case {i+1}: exception: {e}"

    return True, None

def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (batch, vocab, nbw) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            bad_word_ids = torch.randint(0, vocab, (batch, nbw), dtype=torch.int32, device=device)
            offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
            for b in range(batch):
                for j in range(nbw + 1):
                    offsets[b, j] = j
            num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
            all_token_ids = torch.zeros(batch, 128, dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), 10, dtype=torch.int32, device=device)
            total_len = torch.full((batch,), 20, dtype=torch.int32, device=device)
            input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            for _ in range(WARMUP_ITERATIONS):
                mod.apply_bad_words(logits.clone(), idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                l = logits.clone()
                start_events[j].record()
                mod.apply_bad_words(l, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
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
                    "batch": batch,
                    "vocab": vocab,
                    "num_bad_words": nbw
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "vocab": vocab,
                    "num_bad_words": nbw
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
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__": main()
