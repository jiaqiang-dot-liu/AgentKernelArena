#!/usr/bin/env python3
"""Task runner for triton2triton/triton_topk_topp"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_topk_topp"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_topk_topp.py")

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
        assert hasattr(mod, "apply_top_k_top_p_triton"), "Missing apply_top_k_top_p_triton"
        assert hasattr(mod, "_topk_topp_kernel"), "Missing _topk_topp_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256),    # (batch, vocab)
    (8, 1024),
    (16, 4096),
    (32, 8192),
    (64, 16384),
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

def reference_apply_top_k_top_p(logits, k, p):
    import torch
    out = logits.clone()
    batch, _ = out.shape
    for b in range(batch):
        row = out[b]
        if k is not None:
            kv = int(k[b].item())
            if kv < row.numel():
                topk_vals, _ = torch.topk(row, kv)
                kth = topk_vals[-1]
                row = torch.where(row >= kth, row, torch.tensor(float("-inf"), device=row.device, dtype=row.dtype))
        if p is not None:
            pv = float(p[b].item())
            if pv < 1.0:
                sorted_vals, sorted_idx = torch.sort(row, descending=True)
                probs = torch.softmax(sorted_vals, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > pv
                remove[0] = False
                row[sorted_idx[remove]] = float("-inf")
        out[b] = row
    return out


def compare_masked_logits(got, ref, vocab_size, max_mask_mismatch):
    import torch

    got_mask = torch.isfinite(got)
    ref_mask = torch.isfinite(ref)

    for b in range(got.shape[0]):
        mismatch = (got_mask[b] ^ ref_mask[b]).sum().item()
        if mismatch > max_mask_mismatch:
            return False, f"row {b}: mask mismatch {mismatch} > {max_mask_mismatch}"

    common = got_mask & ref_mask
    if common.any():
        if not torch.allclose(got[common], ref[common], atol=1e-4, rtol=1e-4):
            max_diff = (got[common] - ref[common]).abs().max().item()
            return False, f"common finite values max diff={max_diff}"
    return True, None

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
            k = torch.full((batch_size,), min(50, vocab_size), dtype=torch.int32, device=device)
            p = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)

            logits_topk = logits.clone()
            ref_topk = reference_apply_top_k_top_p(logits.clone(), k, None)
            mod.apply_top_k_top_p_triton(logits_topk, k, None)
            torch.cuda.synchronize()
            ok, msg = compare_masked_logits(logits_topk, ref_topk, vocab_size, max_mask_mismatch=1)
            if not ok:
                return False, f"Shape {i+1}: top-k mismatch ({msg})"

            logits_topkp = logits.clone()
            ref_topkp = reference_apply_top_k_top_p(logits.clone(), k, p)
            mod.apply_top_k_top_p_triton(logits_topkp, k, p)
            torch.cuda.synchronize()
            # Pivot-based GPU implementation may differ slightly at boundary tokens.
            max_mismatch = max(4, vocab_size // 500)
            ok, msg = compare_masked_logits(logits_topkp, ref_topkp, vocab_size, max_mask_mismatch=max_mismatch)
            if not ok:
                return False, f"Shape {i+1}: top-k + top-p mismatch ({msg})"

            # Invariant: top-k + top-p should keep no more tokens than top-k only.
            kept_topk = torch.isfinite(logits_topk).sum(dim=-1)
            kept_topkp = torch.isfinite(logits_topkp).sum(dim=-1)
            if torch.any(kept_topkp > kept_topk):
                return False, f"Shape {i+1}: top-k+topp kept more tokens than top-k"
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

    for test_idx, (batch_size, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            for _ in range(WARMUP_ITERATIONS):
                logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
                k = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
                mod.apply_top_k_top_p_triton(logits, k, None)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
                k = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
                start_events[j].record()
                mod.apply_top_k_top_p_triton(logits, k, None)
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
                    "vocab_size": vocab_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch_size": batch_size,
                    "vocab_size": vocab_size
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
