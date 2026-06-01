#!/usr/bin/env python3
"""Test harness for the FP8 block-scale GEMM kernel.

Timing and correctness live HERE, not in kernel.py — the agent edits kernel.py,
so an embedded benchmark there could be gamed. The harness owns the measurement
and only imports the kernel-side building blocks (kernels, wrappers, inputs).
"""
import argparse
import math
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import torch

from kernel import (
    EVAL_CONFIGS,
    RTOL,
    ATOL,
    get_inputs,
    fp8_blockwise_mm_triton,
    fp8_blockwise_mm_pytorch,
)

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

ALL_CONFIGS = EVAL_CONFIGS


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _label(cfg):
    return "M={} N={} K={}".format(cfg["m"], cfg["n"], cfg["k"])


def check_correctness(cfg):
    a, b, a_scale, b_scale, c_triton = get_inputs(**cfg)
    c_ref = c_triton.clone()
    fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c_triton)
    fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c_ref)
    torch.cuda.synchronize()
    return torch.allclose(c_triton.float(), c_ref.float(), rtol=RTOL, atol=ATOL)


def _bench_one(cfg, warmup, iters):
    a, b, a_scale, b_scale, c = get_inputs(**cfg)
    for _ in range(warmup):
        fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c.clone())
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c.clone())
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_correctness(indices):
    torch.manual_seed(42)
    print("Running correctness on {} configs ...".format(len(indices)))
    all_ok = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        try:
            ok = check_correctness(cfg)
        except Exception as e:  # noqa: BLE001
            print("  [{}] {}  FAIL: {}".format(idx, _label(cfg), str(e)[:80]))
            all_ok = False
            continue
        if ok:
            print("  [{}] {}  PASS".format(idx, _label(cfg)))
        else:
            print("  [{}] {}  FAIL".format(idx, _label(cfg)))
            all_ok = False
    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_ok:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("All correctness checks passed.")


def run_benchmark(indices, warmup, iters):
    torch.manual_seed(42)
    print("Running benchmark on {} configs ...".format(len(indices)))
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ms = _bench_one(cfg, warmup, iters)
        latencies.append(ms)
        print("  [{}] {}  {:.4f}ms".format(idx, _label(cfg), ms))
    geo = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo))


def run_profile(indices):
    torch.manual_seed(42)
    print("Running profile on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        a, b, a_scale, b_scale, c = get_inputs(**cfg)
        for _ in range(3):
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c.clone())
        torch.cuda.synchronize()
        print("  [{}] {}  done".format(idx, _label(cfg)))
    print("GEAK_SHAPES_USED={}".format(indices))


def main():
    parser = argparse.ArgumentParser(description="Test harness for fp8 blockwise mm")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()
    iters = args.iterations if args.iterations is not None else ITERATIONS

    if args.correctness:
        run_correctness(_pick(ALL_CONFIGS, 25))
    elif args.benchmark:
        run_benchmark(_pick(ALL_CONFIGS, 25), args.warmup, iters)
    elif args.full_benchmark:
        run_benchmark(list(range(len(ALL_CONFIGS))), args.warmup, iters)
    elif args.profile:
        run_profile(_pick(ALL_CONFIGS, 5))


if __name__ == "__main__":
    main()
