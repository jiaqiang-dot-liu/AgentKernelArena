#!/usr/bin/env python3
"""Task runner for triton_ssd_chunk_scan"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_chunk_scan.py")

# (seqlen, nheads, headdim, ngroups, dstate, chunk_size)
TEST_SHAPES = [
    (128, 4, 32, 2, 16, 64),
    (256, 8, 64, 4, 32, 64),
    (512, 4, 32, 2, 16, 128),
    (256, 4, 64, 2, 16, 64),
    (384, 8, 32, 4, 32, 128),
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
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, seq_idx, chunk_size):
    import torch

    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = C.shape
    ratio = nheads // ngroups
    nchunks = seqlen // chunk_size

    cb_c = cb.float().cpu()
    x_c = x.float().cpu()
    dt_c = dt.float().cpu()
    dA_c = dA_cumsum.float().cpu()
    C_c = C.float().cpu()
    states_c = states.float().cpu()
    seq_c = seq_idx.cpu()

    out = torch.zeros(seqlen, nheads, headdim, dtype=torch.float32)
    for c in range(nchunks):
        for h in range(nheads):
            g = h // ratio
            if c == 0 or seq_c[c].item() != seq_c[c - 1].item():
                prev_state = torch.zeros(headdim, dstate, dtype=torch.float32)
            else:
                prev_state = states_c[c - 1, h]
            for t in range(chunk_size):
                tok = c * chunk_size + t
                if tok >= seqlen:
                    break
                dA_t = dA_c[h, c, t]
                acc = torch.matmul(prev_state, C_c[tok, g]) * torch.exp(dA_t)
                for k in range(t + 1):
                    tok_k = c * chunk_size + k
                    coeff = cb_c[c, g, t, k] * torch.exp(dA_t - dA_c[h, c, k]) * dt_c[h, c, k]
                    acc = acc + coeff * x_c[tok_k, h]
                out[tok, h] = acc
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_chunk_scan_fwd_kernel")
        assert hasattr(mod, "chunk_scan_fwd")
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Load failed: {e}"
    device = "cuda"
    for i, (seqlen, nheads, headdim, ngroups, dstate, chunk_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            nchunks = seqlen // chunk_size
            cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=torch.float16) * 0.01
            x = torch.randn(seqlen, nheads, headdim, device=device, dtype=torch.float16)
            C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=torch.float16)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32) * 0.01
            seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            out = torch.zeros(seqlen, nheads, headdim, device=device, dtype=torch.float32)
            mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx)
            torch.cuda.synchronize()
            ref = reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, seq_idx, chunk_size).to(device)
            if not torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2):
                diff = (out.float() - ref.float()).abs().max().item()
                return False, f"Shape {i}: max diff = {diff:.6f}"
        except Exception as e:
            return False, f"Shape {i}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []
    for test_idx, (seqlen, nheads, headdim, ngroups, dstate, chunk_size) in enumerate(TEST_SHAPES):
        try:
            nchunks = seqlen // chunk_size
            torch.manual_seed(0)
            cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=torch.float16) * 0.01
            x = torch.randn(seqlen, nheads, headdim, device=device, dtype=torch.float16)
            C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=torch.float16)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32) * 0.01
            seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            out = torch.zeros(seqlen, nheads, headdim, device=device, dtype=torch.float32)
            def _bench_fn():
                mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx)
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
                    "seqlen": seqlen,
                    "nheads": nheads,
                    "headdim": headdim,
                    "ngroups": ngroups,
                    "dstate": dstate,
                    "chunk_size": chunk_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "seqlen": seqlen,
                    "nheads": nheads,
                    "headdim": headdim,
                    "ngroups": ngroups,
                    "dstate": dstate,
                    "chunk_size": chunk_size
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser()
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
        report = {"status": "ok" if ok else "fail", "error": err}
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
