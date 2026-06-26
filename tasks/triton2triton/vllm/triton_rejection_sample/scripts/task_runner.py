#!/usr/bin/env python3
"""Task runner for triton_rejection_sample"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_rejection_sample"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_rejection_sample.py")

# (num_reqs, num_speculative_steps)
TEST_SHAPES = [
    (4, 3),
    (8, 5),
    (16, 4),
    (32, 6),
    (64, 5),
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


def make_inputs(num_reqs, num_spec_steps, device="cpu"):
    import torch
    torch.manual_seed(42)
    # Each request has (num_spec_steps + 1) logits/tokens
    tokens_per_req = num_spec_steps + 1
    total_tokens = num_reqs * tokens_per_req

    # Target sampled tokens
    target_sampled = torch.randint(0, 100, (total_tokens,), dtype=torch.int64)
    # Input IDs (draft tokens); make some match target to test acceptance
    input_ids = target_sampled.clone()
    # Introduce mismatches at random positions
    for r in range(num_reqs):
        base = r * tokens_per_req
        # Reject at a random position (or not at all)
        reject_pos = torch.randint(0, tokens_per_req + 1, (1,)).item()
        if reject_pos < tokens_per_req:
            for j in range(reject_pos, tokens_per_req):
                input_ids[base + j] = (target_sampled[base + j].item() + 1) % 100

    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32)
    for r in range(num_reqs):
        cu_num_logits[r + 1] = cu_num_logits[r] + tokens_per_req

    if device != "cpu":
        target_sampled = target_sampled.to(device)
        input_ids = input_ids.to(device)
        cu_num_logits = cu_num_logits.to(device)
    return target_sampled, input_ids, cu_num_logits


def reference(target_sampled, input_ids, cu_num_logits, num_spec_steps):
    import torch
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.empty(num_reqs, num_spec_steps + 1, dtype=target_sampled.dtype)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32)

    for r in range(num_reqs):
        start = cu_num_logits[r].item()
        end = cu_num_logits[r + 1].item()
        nt = end - start
        ns = 0
        rejected = False
        for i in range(nt - 1):
            if not rejected:
                ts = target_sampled[start + i].item()
                ds = input_ids[start + i + 1].item()
                sampled[r, i] = ts
                ns += 1
                if ts != ds:
                    rejected = True
        if not rejected:
            ts = target_sampled[start + nt - 1].item()
            sampled[r, nt - 1] = ts
            ns += 1
        num_sampled[r] = ns
    return sampled, num_sampled


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "rejection_sample"), "Missing wrapper"
        assert hasattr(mod, "_rejection_sample_kernel"), "Missing kernel"
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
    for i, (nr, nss) in enumerate(TEST_SHAPES):
        try:
            ts, ids, cu = make_inputs(nr, nss, device)
            res_sampled, res_ns = mod.rejection_sample(ts, ids, cu, nss)
            torch.cuda.synchronize()

            ts_c, ids_c, cu_c = make_inputs(nr, nss, "cpu")
            ref_sampled, ref_ns = reference(ts_c, ids_c, cu_c, nss)

            if not torch.equal(res_ns.cpu(), ref_ns):
                return False, f"Shape {i+1}: num_sampled mismatch: {res_ns.cpu()} vs {ref_ns}"
            # Only check sampled values up to num_sampled for each request
            for r in range(nr):
                ns = ref_ns[r].item()
                if not torch.equal(res_sampled[r, :ns].cpu(), ref_sampled[r, :ns]):
                    return False, f"Shape {i+1}: sampled mismatch at req {r}"
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

    for test_idx, (nr, nss) in enumerate(TEST_SHAPES):
        try:
            ts, ids, cu = make_inputs(nr, nss, device)

            def _bench_fn():
                mod.rejection_sample(ts, ids, cu, nss)
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
                    "num_speculative_steps": nss
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": nr,
                    "num_speculative_steps": nss
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
