#!/usr/bin/env python3
"""Task runner for triton_selective_scan_update"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_selective_scan_update.py")

# (batch, nheads, dim, dstate, ngroups, has_D, has_z)
TEST_SHAPES = [
    (4, 8, 64, 16, 4, True, False),
    (8, 4, 128, 32, 2, True, True),
    (2, 16, 64, 64, 8, False, False),
    (4, 8, 128, 16, 4, True, True),
    (16, 4, 64, 32, 2, True, False),
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


def reference(state, x, dt, A, B, C, D, z):
    import torch
    batch, nheads, dim = x.shape
    dstate = state.shape[-1]
    ngroups = B.shape[1]
    nheads_per_group = nheads // ngroups
    state_f = state.float()
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    out = torch.zeros(batch, nheads, dim, dtype=torch.float32)
    for b in range(batch):
        for h in range(nheads):
            g = h // nheads_per_group
            for d in range(dim):
                for s in range(dstate):
                    dA_val = torch.exp(A_f[h, d, s] * dt_f[b, h, d])
                    dB_val = B_f[b, g, s] * dt_f[b, h, d]
                    state_f[b, h, d, s] = state_f[b, h, d, s] * dA_val + dB_val * x_f[b, h, d]
            for d in range(dim):
                val = 0.0
                for s in range(dstate):
                    val += state_f[b, h, d, s] * C_f[b, g, s]
                if D is not None:
                    val += x_f[b, h, d] * D.float()[h, d]
                if z is not None:
                    zv = z.float()[b, h, d]
                    val *= zv * torch.sigmoid(torch.tensor(zv))
                out[b, h, d] = val
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_selective_scan_update_kernel")
        assert hasattr(mod, "selective_state_update")
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
    for i, (batch, nheads, dim, dstate, ngroups, has_D, has_z) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=torch.float32) * 0.1
            x = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32)
            dt = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) * 0.1
            A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32)
            B = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
            C = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
            D = torch.randn(nheads, dim, device=device, dtype=torch.float32) if has_D else None
            z = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) if has_z else None
            out = torch.empty(batch, nheads, dim, device=device, dtype=torch.float32)
            state_copy = state.clone()
            ref = reference(state_copy.cpu(), x.cpu(), dt.cpu(), A.cpu(), B.cpu(), C.cpu(),
                          D.cpu() if D is not None else None,
                          z.cpu() if z is not None else None)
            mod.selective_state_update(state, x, dt, A, B, C, D=D, z=z, out=out)
            torch.cuda.synchronize()
            if not torch.allclose(out.cpu(), ref, atol=1e-2, rtol=1e-2):
                diff = (out.cpu() - ref).abs().max().item()
                return False, f"Shape {i}: max diff={diff}"
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

    for test_idx, (batch, nheads, dim, dstate, ngroups, has_D, has_z) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=torch.float32) * 0.1
            x = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32)
            dt = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) * 0.1
            A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32)
            B = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
            C = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
            D = torch.randn(nheads, dim, device=device, dtype=torch.float32) if has_D else None
            z = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) if has_z else None
            out = torch.empty(batch, nheads, dim, device=device, dtype=torch.float32)
            for _ in range(WARMUP_ITERATIONS):
                mod.selective_state_update(state.clone(), x, dt, A, B, C, D=D, z=z, out=out)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                starts[j].record()
                mod.selective_state_update(state.clone(), x, dt, A, B, C, D=D, z=z, out=out)
                ends[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            elapsed_ms = sum(times) / len(times)
            benchmark_metadata = {
                "benchmark_method": "cuda_event_fallback",
                "benchmark_target_ms": 20.0,
                "benchmark_retries": 1,
                "benchmark_max_repeats": 1000,
                "benchmark_effective_repeats": n_iter,
                "benchmark_fallback_reason": "timed_clone_or_fresh_tensor",
            }

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "batch": batch,
                    "nheads": nheads,
                    "dim": dim,
                    "dstate": dstate,
                    "ngroups": ngroups,
                    "has_D": has_D,
                    "has_z": has_z
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "nheads": nheads,
                    "dim": dim,
                    "dstate": dstate,
                    "ngroups": ngroups,
                    "has_D": has_D,
                    "has_z": has_z
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
