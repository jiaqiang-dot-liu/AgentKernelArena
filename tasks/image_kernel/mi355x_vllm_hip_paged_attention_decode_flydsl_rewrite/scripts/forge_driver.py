#!/usr/bin/env python3
"""Self-contained measurement driver: AITER paged-attention decode (oracle +
baseline) vs a FlyDSL port in ``kernel.py``.

This driver is embedded verbatim in the port agent's prompt, so it is the
authoritative statement of the operator's I/O. It is standalone on purpose: it
imports nothing from the task's other scripts.

Modes (forge stdout contract):
  --ref-bench-mode   time the AITER source            -> the speedup baseline
  --bench-mode       time the FlyDSL port in kernel.py
  --profile-run      one launch of the FlyDSL port on the primary case
  (no flag)          correctness: FlyDSL vs AITER, prints ``SNR`` + ``allclose``

FlyDSL contract -- ``kernel.py`` next to this file must expose::

    build_paged_attention_decode_module(
        num_q_heads, num_kv_heads, head_size, block_size) -> launch_fn
    launch_fn(out, query, key_cache, value_cache, block_tables, seq_lens, scale)

with, for x = 16 // itemsize = 8 and bf16 throughout::

    query        (num_seqs, num_q_heads, head_size)
    key_cache    (num_blocks, num_kv_heads, head_size // x, block_size, x)
    value_cache  (num_blocks, num_kv_heads, head_size, block_size)
    block_tables (num_seqs, max_blocks_per_seq)   int32
    seq_lens     (num_seqs,)                      int32
    out          (num_seqs, num_q_heads, head_size)   written in place

Semantics: decode paged attention, one query token per sequence. For sequence s
and query head h, attend over the first ``seq_lens[s]`` KV positions gathered
through ``block_tables[s]``, softmax scale ``scale``, GQA sharing
``num_q_heads // num_kv_heads`` query heads per KV head. No ALiBi, no sliding
window, no FP8 KV cache. The port picks its own KV partitioning: unlike the
AITER source it is not handed any split-K scratch buffers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PARTITION_SIZE = 256
OP_NAME = "paged_attention_decode"
BUILDER = f"build_{OP_NAME}_module"


def _workspace() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here):
        if (cand / "config.yaml").is_file():
            return cand
    return here.parent


WORKSPACE = _workspace()
_SPEC_PATH = next(
    (p for p in (WORKSPACE / "session_cases.json", Path(__file__).with_name("session_cases.json"))
     if p.is_file()),
    None,
)
SPEC = json.loads(_SPEC_PATH.read_text()) if _SPEC_PATH else {"cases": []}
CASES = SPEC["cases"]
PROFILE_CASE_ID = SPEC.get("profile_case") or (CASES[0]["id"] if CASES else "")


def _torch():
    import torch

    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)", file=sys.stderr)
        raise SystemExit(1)
    return torch


def _load_flydsl():
    """Import the FlyDSL port; return its builder or None when absent/stubbed."""
    for cand in (WORKSPACE / "kernel.py", Path(__file__).with_name("kernel.py")):
        if not cand.is_file():
            continue
        spec = importlib.util.spec_from_file_location("ka_flydsl_kernel", cand)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ka_flydsl_kernel"] = module
        spec.loader.exec_module(module)
        return getattr(module, BUILDER, None)
    return None


def _build(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    p = case["params"]
    num_seqs = min(p["num_seqs"], 8) if correctness else p["num_seqs"]
    ctx_len = min(p["ctx_len"], 256) if correctness else p["ctx_len"]
    hq, hkv = p["num_query_heads"], p["num_kv_heads"]
    hs, bs = p["head_size"], p["block_size"]
    dtype = torch.bfloat16

    torch.manual_seed(29)
    query = torch.randn((num_seqs, hq, hs), device="cuda", dtype=dtype)
    key = torch.randn((num_seqs, ctx_len, hkv, hs), device="cuda", dtype=dtype)
    value = torch.randn((num_seqs, ctx_len, hkv, hs), device="cuda", dtype=dtype)

    pages = (ctx_len + bs - 1) // bs
    num_blocks = num_seqs * pages + 1
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache = torch.zeros((num_blocks, hkv, hs // x, bs, x), device="cuda", dtype=dtype)
    value_cache = torch.zeros((num_blocks, hkv, hs, bs), device="cuda", dtype=dtype)

    block_tables = torch.arange(num_seqs * pages, device="cuda", dtype=torch.int32).view(num_seqs, pages)
    si = torch.arange(num_seqs, device="cuda").view(-1, 1).expand(-1, ctx_len)
    pos = torch.arange(ctx_len, device="cuda").view(1, -1).expand(num_seqs, -1)
    blk = block_tables[si, pos // bs].long().reshape(-1)
    off = (pos % bs).reshape(-1)
    k_flat = key.reshape(-1, hkv, hs)
    key_cache[blk, :, :, off, :] = k_flat.view(-1, hkv, hs // x, x)
    value_cache[blk, :, :, off] = value.reshape(-1, hkv, hs)

    parts = (ctx_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    return {
        "query": query, "key": key, "value": value,
        "key_cache": key_cache, "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": torch.full((num_seqs,), ctx_len, device="cuda", dtype=torch.int32),
        "out": torch.empty_like(query),
        "exp_sums": torch.empty((num_seqs, hq, parts), device="cuda", dtype=torch.float32),
        "max_logits": torch.empty((num_seqs, hq, parts), device="cuda", dtype=torch.float32),
        "tmp_out": torch.empty((num_seqs, hq, parts, hs), device="cuda", dtype=dtype),
        "one": torch.ones(1, device="cuda", dtype=torch.float32),
        "hq": hq, "hkv": hkv, "hs": hs, "bs": bs,
        "ctx_len": ctx_len, "scale": hs**-0.5,
    }


def _run_aiter(t: dict):
    import aiter

    aiter.paged_attention_rocm(
        t["out"], t["exp_sums"], t["max_logits"], t["tmp_out"], t["query"],
        t["key_cache"], t["value_cache"], t["hkv"], t["scale"], t["block_tables"],
        t["seq_lens"], t["bs"], t["ctx_len"], None, "auto", t["one"], t["one"],
    )
    return t["out"]


def _make_flydsl(builder, t: dict):
    launch = builder(t["hq"], t["hkv"], t["hs"], t["bs"])

    def run():
        launch(t["out"], t["query"], t["key_cache"], t["value_cache"],
               t["block_tables"], t["seq_lens"], t["scale"])
        return t["out"]

    return run


def _event_ms(fn, repeat: int) -> float:
    """Per-launch event timing; includes host overhead. Graph fallback only."""
    torch = _torch()
    samples = []
    for _ in range(max(1, repeat)):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    samples.sort()
    return samples[len(samples) // 2]


def _time_ms(fn, warmup=10, repeat=50, target_ms=10.0) -> tuple[float, str]:
    """Median ms per call. Prefer CUDA graph so host launch overhead is excluded.

    A port that syncs with the host (``.item()``, a python-side length read)
    cannot be captured. That is a legitimate, if slower, implementation, so fall
    back to event timing instead of failing the whole benchmark.
    """
    torch = _torch()
    try:
        return _graph_ms(fn, warmup, repeat, target_ms), "cuda_graph"
    except Exception:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for _ in range(min(3, max(1, warmup))):
            fn()
        torch.cuda.synchronize()
        return _event_ms(fn, repeat), "cuda_event_fallback"


def _graph_ms(fn, warmup=10, repeat=50, target_ms=10.0):
    """CUDA-graph timing so host launch overhead is excluded."""
    torch = _torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        est = torch.cuda.CUDAGraph()
        with torch.cuda.graph(est):
            for _ in range(5):
                fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(stream); est.replay(); e.record(stream)
        torch.cuda.synchronize()
        per = max(s.elapsed_time(e) / 5, 1e-6)
        n = max(1, min(1000, int(target_ms / per)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(n):
                fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeat):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(stream); graph.replay(); e.record(stream)
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) / n)
    samples.sort()
    return samples[len(samples) // 2]


def _snr_db(ref, test) -> float:
    torch = _torch()
    ref, test = ref.float(), test.float()
    noise = (ref - test).pow(2).sum()
    return float(10 * torch.log10(ref.pow(2).sum() / noise.clamp(min=1e-30)))


def _bench(label: str, use_flydsl: bool, warmup: int, iters: int) -> int:
    torch = _torch()
    builder = _load_flydsl() if use_flydsl else None
    if use_flydsl and builder is None:
        print(f"error: {BUILDER} not found in kernel.py", file=sys.stderr)
        return 1
    times, methods = [], set()
    for case in CASES:
        t = _build(case)
        fn = _make_flydsl(builder, t) if use_flydsl else (lambda t=t: _run_aiter(t))
        try:
            fn()
        except NotImplementedError as exc:
            print(f"error: FlyDSL port not implemented: {exc}", file=sys.stderr)
            return 1
        torch.cuda.synchronize()
        ms, method = _time_ms(fn, warmup=warmup, repeat=iters)
        times.append(ms)
        methods.add(method)
        print(f"case_ms: {case['id']} {ms:.6f}")
    if not times:
        print("error: no cases", file=sys.stderr)
        return 1
    times_sorted = sorted(times)
    print(f"# timing: {'+'.join(sorted(methods))}")
    print(f"# bench mode: {label}")
    print(f"median_ms: {times_sorted[len(times_sorted) // 2]:.6f}")
    print(f"mean_ms: {sum(times) / len(times):.6f}")
    return 0


def _correctness() -> int:
    torch = _torch()
    builder = _load_flydsl()
    if builder is None:
        print(f"error: kernel.py does not expose {BUILDER}", file=sys.stderr)
        print("allclose: False")
        return 0
    worst_snr, ok = float("inf"), True
    for case in CASES:
        t = _build(case, correctness=True)
        try:
            got = _make_flydsl(builder, t)()
        except NotImplementedError as exc:
            print(f"# FlyDSL port not implemented: {exc}")
            print("allclose: False")
            return 0
        torch.cuda.synchronize()
        got = got.clone()
        ref = _run_aiter(t)
        torch.cuda.synchronize()
        snr = _snr_db(ref, got)
        close = bool(torch.allclose(got, ref, atol=2e-2, rtol=2e-2))
        ok = ok and close
        worst_snr = min(worst_snr, snr)
        print(f"# {case['id']}: SNR={snr:.2f} dB allclose={close}")
    print(f"SNR: {worst_snr:.2f} dB")
    print(f"allclose: {ok}")
    return 0


def _profile() -> int:
    torch = _torch()
    builder = _load_flydsl()
    case = next((c for c in CASES if c["id"] == PROFILE_CASE_ID), CASES[0])
    t = _build(case)
    fn = _make_flydsl(builder, t) if builder else (lambda: _run_aiter(t))
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="paged-attention decode FlyDSL rewrite driver")
    ap.add_argument("--bench-mode", action="store_true", help="time the FlyDSL port")
    ap.add_argument("--ref-bench-mode", action="store_true", help="time the AITER source")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _unknown = ap.parse_known_args()

    if args.profile_run:
        return _profile()
    if args.ref_bench_mode:
        return _bench("aiter", False, args.warmup, args.iters)
    if args.bench_mode:
        return _bench("flydsl", True, args.warmup, args.iters)
    return _correctness()


if __name__ == "__main__":
    sys.exit(main())
