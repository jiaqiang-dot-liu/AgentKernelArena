#!/usr/bin/env python3
"""Task runner for triton_prepare_eagle_inputs"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_prepare_eagle_inputs"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_eagle_inputs.py")

# (num_reqs, tokens_per_req, num_rejected_max)
TEST_SHAPES = [
    (4, 16, 2),
    (8, 32, 3),
    (16, 24, 2),
    (32, 20, 4),
    (64, 32, 3),
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


def make_inputs(num_reqs, tokens_per_req, num_rej_max, device="cpu"):
    import torch
    torch.manual_seed(42)
    total_tokens = num_reqs * tokens_per_req
    target_input_ids = torch.randint(0, 32000, (total_tokens,), dtype=torch.int32)
    target_positions = torch.zeros(total_tokens, dtype=torch.int32)
    for r in range(num_reqs):
        start = r * tokens_per_req
        for t in range(tokens_per_req):
            target_positions[start + t] = t

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    max_num_reqs = num_reqs + 8
    last_sampled = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int64)
    next_prefill_tokens = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int32)
    num_sampled = torch.ones(num_reqs, dtype=torch.int32)  # all have sampled tokens
    num_rejected = torch.randint(0, min(num_rej_max + 1, tokens_per_req - 1), (num_reqs,), dtype=torch.int32)

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + tokens_per_req

    if device != "cpu":
        target_input_ids = target_input_ids.to(device)
        target_positions = target_positions.to(device)
        idx_mapping = idx_mapping.to(device)
        last_sampled = last_sampled.to(device)
        next_prefill_tokens = next_prefill_tokens.to(device)
        num_sampled = num_sampled.to(device)
        num_rejected = num_rejected.to(device)
        query_start_loc = query_start_loc.to(device)
    return (target_input_ids, target_positions, idx_mapping, last_sampled,
            next_prefill_tokens, num_sampled, num_rejected, query_start_loc)


def reference(target_input_ids, target_positions, idx_mapping, last_sampled,
              next_prefill_tokens, num_sampled, num_rejected, query_start_loc):
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = target_input_ids.shape[0]
    eagle_input_ids = torch.zeros_like(target_input_ids)
    eagle_positions = torch.zeros_like(target_positions)
    last_token_indices = torch.empty(num_reqs, dtype=torch.int64)

    for b in range(num_reqs):
        req_idx = idx_mapping[b].item()
        qs = query_start_loc[b].item()
        qe = query_start_loc[b + 1].item()
        ql = qe - qs
        nrej = num_rejected[b].item()
        ql -= nrej

        ns = num_sampled[b].item()
        if ns > 0:
            nt = int(last_sampled[req_idx].item())
        else:
            nt = next_prefill_tokens[req_idx].item()

        # Shift input ids
        for j in range(1, ql):
            eagle_input_ids[qs + j - 1] = target_input_ids[qs + j]
        lti = qs + ql - 1
        last_token_indices[b] = lti
        eagle_input_ids[lti] = nt

        # Copy positions
        for j in range(ql):
            eagle_positions[qs + j] = target_positions[qs + j]

    return last_token_indices, eagle_input_ids, eagle_positions


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_eagle_inputs"), "Missing wrapper"
        assert hasattr(mod, "_prepare_eagle_inputs_kernel"), "Missing kernel"
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
    for i, (nr, tpr, nrm) in enumerate(TEST_SHAPES):
        try:
            inputs = make_inputs(nr, tpr, nrm, device)
            res_lti, res_eids, res_epos = mod.prepare_eagle_inputs(*inputs)
            torch.cuda.synchronize()

            cpu_inputs = make_inputs(nr, tpr, nrm, "cpu")
            ref_lti, ref_eids, ref_epos = reference(*cpu_inputs)

            if not torch.equal(res_lti.cpu(), ref_lti):
                return False, f"Shape {i+1}: last_token_indices mismatch"
            if not torch.equal(res_eids.cpu(), ref_eids):
                return False, f"Shape {i+1}: eagle_input_ids mismatch"
            if not torch.equal(res_epos.cpu(), ref_epos):
                return False, f"Shape {i+1}: eagle_positions mismatch"
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

    for test_idx, (nr, tpr, nrm) in enumerate(TEST_SHAPES):
        try:
            inputs = make_inputs(nr, tpr, nrm, device)

            def _bench_fn():
                mod.prepare_eagle_inputs(*inputs)
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
                    "tokens_per_req": tpr,
                    "num_rejected_max": nrm
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": nr,
                    "tokens_per_req": tpr,
                    "num_rejected_max": nrm
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
