#!/usr/bin/env python3
"""Task runner for triton2flydsl/aiter/mla.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the triton kernel on TEST_SHAPES, assert finite output (bf16)
  - performance  : warmup + cuda-event timing, write build/performance_report.json

Multi-head Latent Attention (MLA) decode / "absorb" path. Public entry:
`mla_decode_fwd(...)`; @triton.jit kernels: `_mla_decode_fwd_kernel`,
`_mla_decode_fwd_reduce_kernel`, `_mla_prefill_fwd_kernel`.

The flydsl-vs-triton comparison will be added when the FlyDSL target lands.
"""
import sys
import os
import json
import time
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/aiter/mla"
SOURCE_FILE = os.path.join(TASK_DIR, "mla.py")

# Test configurations (decode / absorb path):
# (num_seqs, num_tokens_per_seq, num_query_heads, num_kv_heads, kv_lora_rank,
#  qk_rope_head_dim, block_size, seq_len_k)
# kv_lora_rank, qk_rope_head_dim and block_size must be powers of two (Triton
# arange constraints). num_tokens_per_seq == 1 -> ALL_DECODE; > 1 exercises the
# non-ALL_DECODE branch. The decode path always splits into >1 segments on CDNA,
# so the reduce kernel is exercised by every shape.
TEST_SHAPES = [
    (1, 1, 16, 1, 128, 64, 64, 128),    # ALL_DECODE, single seq
    (2, 1, 16, 1, 128, 64, 64, 256),    # ALL_DECODE, 2 seqs
    (4, 1, 16, 1, 256, 64, 64, 512),    # ALL_DECODE, larger lora rank
    (1, 1, 128, 1, 512, 64, 64, 1024),  # DeepSeek-like decode (128 heads, lora 512)
    (2, 1, 32, 1, 128, 64, 32, 192),    # ALL_DECODE, smaller block_size
    (1, 4, 16, 1, 128, 64, 64, 256),    # num_tokens_per_seq=4 (non ALL_DECODE)
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Shared-GPU resilience: two other workers contend for the device, so retry the
# kernel launch on transient OOM / contention with exponential backoff.
_OOM_RETRIES = 6
_OOM_BACKOFF_S = 1.5


def load_module():
    spec = importlib.util.spec_from_file_location("mla_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_transient_gpu_error(exc):
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("out of memory", "oom", "hip error", "resource", "busy", "contention")
    )


def _retry_gpu(fn):
    """Run fn() with retry/backoff on transient OOM/contention (shared GPU)."""
    import torch

    last = None
    for attempt in range(_OOM_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_transient_gpu_error(e) or attempt == _OOM_RETRIES - 1:
                raise
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(_OOM_BACKOFF_S * (2 ** attempt))
    if last is not None:
        raise last


def make_test_data(num_seqs, num_tokens_per_seq, num_query_heads, num_kv_heads,
                   kv_lora_rank, qk_rope_head_dim, block_size, seq_len_k,
                   device="cuda", dtype=None):
    """Create test tensors for mla_decode_fwd (paged latent KV cache)."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16

    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    total_tokens = num_seqs * num_tokens_per_seq

    # Packed Q tensor: [total_tokens, num_query_heads, kv_lora_rank + qk_rope_head_dim]
    q = torch.randn(total_tokens, num_query_heads, qk_head_dim, device=device, dtype=dtype)

    # Paged latent KV buffer: [num_blocks, block_size, num_kv_heads, qk_head_dim]
    num_blocks_per_seq = (seq_len_k + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4  # extra padding blocks
    kv_buffer = torch.randn(total_blocks, block_size, num_kv_heads, qk_head_dim,
                            device=device, dtype=dtype)

    # Block table: each seq uses contiguous blocks
    block_table = torch.zeros(num_seqs, num_blocks_per_seq, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b

    # cu_seqlens_q: cumulative query token counts [num_seqs + 1]
    cu_seqlens_q = torch.arange(0, (num_seqs + 1) * num_tokens_per_seq, num_tokens_per_seq,
                                device=device, dtype=torch.int32)

    # seqused_k: K sequence lengths [num_seqs]
    seqused_k = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)

    out = torch.empty(total_tokens, num_query_heads, kv_lora_rank, device=device, dtype=dtype)
    scale = 1.0 / (qk_head_dim ** 0.5)

    return q, kv_buffer, out, block_table, cu_seqlens_q, seqused_k, scale


def _call_kernel(mod, q, kv_buffer, out, cu_seqlens_q, seqused_k, seq_len_k,
                 block_table, scale, kv_lora_rank, qk_rope_head_dim):
    return mod.mla_decode_fwd(
        q,
        kv_buffer,
        out,
        cu_seqlens_q,
        seqused_k,
        seq_len_k,            # max_seqlen_kv
        block_table,
        scale,                # softmax_scale
        kv_lora_rank,
        qk_rope_head_dim,
        True,                 # causal
        None,                 # q_descale
        None,                 # kv_descale
    )


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "mla_decode_fwd"), "Missing mla_decode_fwd entry"
        assert hasattr(mod, "mla_prefill_fwd"), "Missing mla_prefill_fwd entry"
        assert hasattr(mod, "_mla_decode_fwd_kernel"), "Missing _mla_decode_fwd_kernel"
        assert hasattr(mod, "_mla_prefill_fwd_kernel"), "Missing _mla_prefill_fwd_kernel"
        assert hasattr(mod, "_mla_decode_fwd_reduce_kernel"), "Missing _mla_decode_fwd_reduce_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.bfloat16
    details = []

    for i, (ns, nt, nqh, nkvh, lora, rope, bs, slk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, kv_buffer, out, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(ns, nt, nqh, nkvh, lora, rope, bs, slk, device, dtype)

            result = _retry_gpu(lambda: _call_kernel(
                mod, q, kv_buffer, out, cu_seqlens_q, seqused_k, slk,
                block_table, scale, lora, rope,
            ))
            torch.cuda.synchronize()

            ok = bool(torch.isfinite(result.float()).all().item())
            details.append({
                "shape_id": i + 1,
                "shape": [ns, nt, nqh, nkvh, lora, rope, bs, slk],
                "out_shape": list(result.shape),
                "finite": ok,
                "passed": bool(ok),
            })
            if not ok:
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: non-finite output", details
        except Exception as e:
            details.append({
                "shape_id": i + 1,
                "shape": [ns, nt, nqh, nkvh, lora, rope, bs, slk],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.bfloat16
    test_cases = []

    for test_idx, (ns, nt, nqh, nkvh, lora, rope, bs, slk) in enumerate(TEST_SHAPES):
        params = {
            "num_seqs": ns, "num_tokens_per_seq": nt, "num_query_heads": nqh,
            "num_kv_heads": nkvh, "kv_lora_rank": lora, "qk_rope_head_dim": rope,
            "block_size": bs, "seq_len_k": slk,
        }
        try:
            torch.manual_seed(42 + test_idx)
            q, kv_buffer, out, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(ns, nt, nqh, nkvh, lora, rope, bs, slk, device, dtype)

            for _ in range(WARMUP_ITERATIONS):
                _retry_gpu(lambda: _call_kernel(
                    mod, q, kv_buffer, out, cu_seqlens_q, seqused_k, slk,
                    block_table, scale, lora, rope,
                ))
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

            for j in range(n_iter):
                start_events[j].record()
                _call_kernel(mod, q, kv_buffer, out, cu_seqlens_q, seqused_k, slk,
                             block_table, scale, lora, rope)
                end_events[j].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": params,
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": params,
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
            "details": details,
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "finite" in d:
                print(f"  shape {d['shape_id']} {d['shape']} -> out {d['out_shape']}: "
                      f"finite={d['finite']} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
