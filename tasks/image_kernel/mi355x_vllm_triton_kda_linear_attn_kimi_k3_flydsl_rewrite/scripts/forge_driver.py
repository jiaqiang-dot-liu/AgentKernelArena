#!/usr/bin/env python3
"""Self-contained measurement driver: vLLM's vendored Triton KDA (oracle +
baseline) vs a FlyDSL port in ``kernel.py``.

This driver is embedded verbatim in the port agent's prompt, so it is the
authoritative statement of the operator's I/O. It is standalone on purpose: it
imports nothing from the task's other scripts.

Modes (forge stdout contract):
  --ref-bench-mode   time the Triton source           -> the speedup baseline
  --bench-mode       time the FlyDSL port in kernel.py
  --profile-run      one launch of the FlyDSL port on the primary case
  (no flag)          correctness: FlyDSL vs Triton, prints ``SNR`` + ``allclose``

One entry, two source entries
-----------------------------
The source exposes ``fused_recurrent_kda_packed_decode`` (decode, hot kernel
k007) and ``chunk_kda_with_fused_gate`` (the prefill chunk-KDA kernel group).
They evaluate the SAME gated delta-rule recurrence -- one token-serial, one
blockwise -- so the port implements a single varlen entry driven by
``cu_seqlens`` and ``state_indices``, and decode is the degenerate case of one
token per segment. ``--ref-bench-mode`` still calls the two real Triton entries
with their native signatures, so the baseline is the framework's actual path.

FlyDSL contract -- ``kernel.py`` next to this file must expose::

    build_kda_linear_attn_module(num_heads, head_dim, chunk_size) -> launch_fn
    launch_fn(out, q, k, v, raw_g, raw_beta, A_log, dt_bias,
              state, state_indices, cu_seqlens, scale, lower_bound)

with::

    out            (T, H, D)      bf16, written in place
    q, k, v        (T, H, D)      bf16. The row stride is NOT H * D in the decode
                                  case: q/k/v are strided views into the packed
                                  mixed_qkv block, so the row stride is 3 * H * D.
                                  Read the strides.
    raw_g          (T, H, D)      fp32, pre-gate
    raw_beta       (T, H)         fp32, PRE-sigmoid
    A_log          (H,)           fp32
    dt_bias        (H * D,)       fp32, indexed as [H, D]
    state          (S, H, D, D)   fp32, laid out [slot, head, v, k], updated IN PLACE
    state_indices  (N,)           int32, the state slot of each segment
    cu_seqlens     (N + 1,)       int32, segment boundaries into T
    scale          python float,  D ** -0.5, applied to the l2-normalised q
    lower_bound    python float,  -5.0, selects the safe-gate branch

Semantics -- for each segment ``n``, with ``S = state[state_indices[n]]`` and
``t`` running over ``[cu_seqlens[n], cu_seqlens[n + 1])``::

    g   = lower_bound * sigmoid(exp(A_log)[h] * (raw_g[t, h, :] + dt_bias[h, :]))
    qn  = l2norm(q[t]) * scale ;  kn = l2norm(k[t])
          with l2norm(x) = x / sqrt(sum(x ** 2) + 1e-6)
    S  *= exp(g)                     # decay per k-column, broadcast over v
    vt  = v[t] - (S * kn).sum(-1)    # v - S @ k          -> [H, V]
    vt *= sigmoid(raw_beta[t])
    S  += outer(vt, kn)
    out[t] = (S * qn).sum(-1)        # S @ q

``S`` stays in fp32 and is left updated at the end of the segment -- the decode
path feeds it straight back in on the next step, so a port that gets ``out``
right and the final state wrong is broken. ``chunk_size`` is the blocking the
source uses (64); the recurrence is exact for any blocking, so it constrains
performance, not semantics.

Two contract details that are easy to get wrong, both read off the sources:
  * ``lower_bound = -5.0`` is not a clamp, it selects a different gate function.
    The softplus branch ``-exp(A_log) * softplus(raw_g + dt_bias)`` is what the
    source uses when the bound is unset; K3 never takes it
    (fused_recurrent.py:513-521, chunk.py:507-515).
  * ``raw_beta`` is PRE-sigmoid. The source applies sigmoid internally
    (fused_recurrent.py:525, chunk.py:470), so applying it here would square it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

OP_NAME = "kda_linear_attn"
BUILDER = f"build_{OP_NAME}_module"

# Correctness runs the FlyDSL port against live Triton, so the cost is a GPU
# launch either way -- but a freshly ported kernel may still be token-serial, so
# cap the token count. 320 spans 5 chunks at chunk_size=64 and therefore still
# exercises the cross-chunk path.
CORRECTNESS_MAX_TOKENS = 320


def _workspace() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here):
        if (cand / "config.yaml").is_file():
            return cand
    return here.parent


WORKSPACE = _workspace()
_SPEC_PATH = next(
    (p for p in (WORKSPACE / "session_cases.json",
                 Path(__file__).with_name("session_cases.json")) if p.is_file()),
    None,
)
SPEC = json.loads(_SPEC_PATH.read_text()) if _SPEC_PATH else {"cases": []}
CASES = SPEC["cases"]
KDA_CONFIG = SPEC.get("kda_config", {})
GATE_LOWER_BOUND = KDA_CONFIG.get("gate_lower_bound", -5.0)
CHUNK_SIZE = KDA_CONFIG.get("chunk_size", 64)
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
    """Inputs shaped exactly as the vLLM KDA layer hands them to the kernel.

    The generation order matches mi355x_vllm_triton_kda_linear_attn_kimi_k3 so
    that, for a given case and seed, both tasks time the same numbers.
    """
    torch = _torch()
    p = case["params"]
    H, D = p["num_heads"], p["head_dim"]
    mode = p["mode"]
    num_seqs = p["num_seqs"]
    seq_len = min(p["seq_len"], CORRECTNESS_MAX_TOKENS) if correctness else p["seq_len"]
    total_t = num_seqs * seq_len

    gen = torch.Generator(device="cuda").manual_seed(int(case.get("seed", 23)))

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen) * scale

    A_log = torch.rand(H, device="cuda", dtype=torch.float32, generator=gen) * 2.0 - 4.0
    dt_bias = torch.rand(H * D, device="cuda", dtype=torch.float32, generator=gen) * 0.1
    raw_g = rnd(1, total_t, H, D, dtype=torch.float32)
    raw_beta = rnd(1, total_t, H, dtype=torch.float32)  # pre-sigmoid on purpose

    t = {
        "cfg": case, "mode": mode, "H": H, "D": D,
        "num_seqs": num_seqs, "seq_len": seq_len, "total_t": total_t,
        "raw_g": raw_g, "raw_beta": raw_beta, "A_log": A_log, "dt_bias": dt_bias,
        "scale": D ** -0.5, "lower_bound": GATE_LOWER_BOUND,
        "chunk_size": CHUNK_SIZE,
        "out": torch.empty(total_t, H, D, device="cuda", dtype=torch.bfloat16),
        # The FlyDSL entry is varlen, so both modes describe themselves the same
        # way; only the segment layout differs.
        "g_flat": raw_g[0], "beta_flat": raw_beta[0],
    }

    if mode == "chunk":
        t["q"] = rnd(1, total_t, H, D, scale=0.5)
        t["k"] = rnd(1, total_t, H, D, scale=0.5)
        t["v"] = rnd(1, total_t, H, D, scale=0.5)
        # chunk takes one state per sequence, [N, H, V, K].
        t["state"] = rnd(num_seqs, H, D, D, dtype=torch.float32, scale=0.1).contiguous()
        t["cu_seqlens"] = torch.arange(
            0, (num_seqs + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32
        )
        t["state_indices"] = torch.arange(num_seqs, device="cuda", dtype=torch.int32)
        t["q_flat"], t["k_flat"], t["v_flat"] = t["q"][0], t["k"][0], t["v"][0]
    else:
        # packed decode consumes the post-conv fused QKV block, [B, 3 * H * D],
        # laid out q | k | v with each part head-major (fused_recurrent.py:491-502).
        mixed = rnd(total_t, 3 * H * D, scale=0.5).contiguous()
        t["mixed_qkv"] = mixed
        # A state cache with slot 0 reserved as the NULL slot; indices start at 1
        # because <= 0 makes the source emit zeros (fused_recurrent.py:481).
        t["state"] = rnd(
            num_seqs + 1, H, D, D, dtype=torch.float32, scale=0.1
        ).contiguous()
        t["state_indices"] = torch.arange(
            1, num_seqs + 1, device="cuda", dtype=torch.int32
        )
        # One token per segment.
        t["cu_seqlens"] = torch.arange(
            total_t + 1, device="cuda", dtype=torch.int32
        )
        # Strided [T, H, D] views of the three thirds -- no copy, so the port
        # sees the same memory traffic the Triton kernel does.
        base = mixed.storage_offset()
        t["q_flat"], t["k_flat"], t["v_flat"] = (
            torch.as_strided(mixed, (total_t, H, D), (3 * H * D, D, 1), base + off)
            for off in (0, H * D, 2 * H * D)
        )
    return t


def _run_triton(t: dict):
    """Call the source entries with their native signatures.

    Returns ``(out[T, H, D], final_state)``. The final state is taken from the
    return value rather than from the in-place buffer so this works whether the
    source updates ``initial_state`` in place or allocates a new tensor.
    """
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda_packed_decode,
    )

    if t["mode"] == "chunk":
        out, state = chunk_kda_with_fused_gate(
            q=t["q"], k=t["k"], v=t["v"],
            raw_g=t["raw_g"], raw_beta=t["raw_beta"], A_log=t["A_log"],
            g_bias=t["dt_bias"], lower_bound=t["lower_bound"],
            initial_state=t["state"], output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=t["cu_seqlens"],
        )
    else:
        out, state = fused_recurrent_kda_packed_decode(
            mixed_qkv=t["mixed_qkv"], raw_g=t["raw_g"], raw_beta=t["raw_beta"],
            A_log=t["A_log"], dt_bias=t["dt_bias"], lower_bound=t["lower_bound"],
            initial_state=t["state"], state_indices=t["state_indices"],
        )
    return out.reshape(t["total_t"], t["H"], t["D"]), state


def _make_flydsl(builder, t: dict):
    launch = builder(t["H"], t["D"], t["chunk_size"])

    def run():
        launch(
            t["out"], t["q_flat"], t["k_flat"], t["v_flat"],
            t["g_flat"], t["beta_flat"], t["A_log"], t["dt_bias"],
            t["state"], t["state_indices"], t["cu_seqlens"],
            t["scale"], t["lower_bound"],
        )
        return t["out"]

    return run


def _budget(case: dict, warmup_cli: int, iters_cli: int) -> dict:
    """Per-case benchmark budget, tightened by whatever the caller asked for.

    The long cases carry small repetition counts on purpose: a token-serial port
    at T=32768 is orders of magnitude slower than the blocked source, and the
    rewrite pipeline benches the whole suite under one timeout.
    """
    b = case.get("benchmark") or {}
    return {
        "warmup": max(1, min(int(b.get("warmup", warmup_cli)), int(warmup_cli))),
        "repeat": max(1, min(int(b.get("repetition", iters_cli)), int(iters_cli))),
        "target_ms": float(b.get("target_ms", 2.0)),
        "cap": max(1, int(b.get("max_graph_repeats", 50))),
    }


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


def _graph_ms(fn, warmup: int, repeat: int, target_ms: float, cap: int) -> float:
    torch = _torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        est_n = max(1, min(5, cap))
        est = torch.cuda.CUDAGraph()
        with torch.cuda.graph(est):
            for _ in range(est_n):
                fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(stream); est.replay(); e.record(stream)
        torch.cuda.synchronize()
        per = max(s.elapsed_time(e) / est_n, 1e-6)
        n = max(1, min(cap, int(target_ms / per)))
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


def _time_ms(fn, budget: dict) -> tuple[float, str]:
    """Median ms per call. Prefer CUDA graph so host launch overhead is excluded.

    A port that syncs with the host cannot be captured. That is a legitimate, if
    slower, implementation, so fall back to event timing rather than failing the
    whole benchmark.
    """
    torch = _torch()
    try:
        ms = _graph_ms(fn, budget["warmup"], budget["repeat"],
                       budget["target_ms"], budget["cap"])
        return ms, "cuda_graph"
    except Exception:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for _ in range(min(3, budget["warmup"])):
            fn()
        torch.cuda.synchronize()
        return _event_ms(fn, budget["repeat"]), "cuda_event_fallback"


def _snr_db(ref, test) -> float:
    torch = _torch()
    ref, test = ref.float(), test.float()
    noise = (ref - test).pow(2).sum()
    return float(10 * torch.log10(ref.pow(2).sum() / noise.clamp(min=1e-30)))


def _parity(ref, test) -> tuple[float, float]:
    """Cosine and max error normalized by the reference scale, as the source task.

    Plain allclose is the wrong gate for a bf16 chunked recurrence: the source
    itself only reproduces its own float64 golden to ~7e-3 relative.
    """
    torch = _torch()
    a, b = test.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs().max() / b.abs().max().clamp_min(1e-8)).item()
    return cos, rel


def _bench(label: str, use_flydsl: bool, warmup: int, iters: int) -> int:
    torch = _torch()
    builder = _load_flydsl() if use_flydsl else None
    if use_flydsl and builder is None:
        print(f"error: {BUILDER} not found in kernel.py", file=sys.stderr)
        return 1
    times, methods = [], set()
    for case in CASES:
        t = _build(case)
        fn = _make_flydsl(builder, t) if use_flydsl else (lambda t=t: _run_triton(t)[0])
        try:
            fn()
        except NotImplementedError as exc:
            print(f"error: FlyDSL port not implemented: {exc}", file=sys.stderr)
            return 1
        torch.cuda.synchronize()
        ms, method = _time_ms(fn, _budget(case, warmup, iters))
        times.append(ms)
        methods.add(method)
        print(f"case_ms: {case['id']} {ms:.6f}")
    if not times:
        print("error: no cases", file=sys.stderr)
        return 1
    ordered = sorted(times)
    print(f"# timing: {'+'.join(sorted(methods))}")
    print(f"# bench mode: {label}")
    print(f"median_ms: {ordered[len(ordered) // 2]:.6f}")
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
        # Both implementations advance the same state buffer, so the second one
        # to run must start from the same place as the first.
        state0 = t["state"].clone()
        # The output buffer is written, never accumulated: poison it so a port
        # that leaves rows untouched cannot pass on allocator leftovers.
        t["out"].fill_(float("nan"))
        try:
            got = _make_flydsl(builder, t)()
        except NotImplementedError as exc:
            print(f"# FlyDSL port not implemented: {exc}")
            print("allclose: False")
            return 0
        torch.cuda.synchronize()
        got, got_state = got.clone(), t["state"].clone()

        t["state"].copy_(state0)
        ref, ref_state = _run_triton(t)
        torch.cuda.synchronize()

        tol = case["params"]
        min_cos = tol.get("min_cosine", 0.999)
        max_rel = tol.get("max_rel_err", 0.03)

        snr = _snr_db(ref, got)
        cos, rel = _parity(ref, got)
        case_ok = bool(torch.isfinite(got).all()) and cos > min_cos and rel < max_rel
        detail = f"SNR={snr:.2f} dB cos={cos:.6f} rel_max_err={rel:.4f}"

        # The recurrent state is an output too: for a single decode token, out
        # only observes it through the q projection, so a state error orthogonal
        # to q would otherwise pass. Compared only when the source's returned
        # state is laid out like the cache the port writes.
        if ref_state is not None and tuple(ref_state.shape) == tuple(got_state.shape):
            s_snr = _snr_db(ref_state, got_state)
            s_cos, s_rel = _parity(ref_state, got_state)
            case_ok = case_ok and s_cos > min_cos and s_rel < max_rel
            snr = min(snr, s_snr)
            detail += f" | state SNR={s_snr:.2f} dB cos={s_cos:.6f} rel_max_err={s_rel:.4f}"
        else:
            detail += " | state not comparable (shape mismatch), out only"

        ok = ok and case_ok
        worst_snr = min(worst_snr, snr)
        print(f"# {case['id']}: {detail} pass={case_ok}")
    print(f"SNR: {worst_snr:.2f} dB")
    print(f"allclose: {ok}")
    return 0


def _profile() -> int:
    torch = _torch()
    builder = _load_flydsl()
    case = next((c for c in CASES if c["id"] == PROFILE_CASE_ID), CASES[0])
    t = _build(case)
    fn = _make_flydsl(builder, t) if builder else (lambda: _run_triton(t)[0])
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="KDA linear attention FlyDSL rewrite driver")
    ap.add_argument("--bench-mode", action="store_true", help="time the FlyDSL port")
    ap.add_argument("--ref-bench-mode", action="store_true", help="time the Triton source")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _unknown = ap.parse_known_args()

    if args.profile_run:
        return _profile()
    if args.ref_bench_mode:
        return _bench("triton", False, args.warmup, args.iters)
    if args.bench_mode:
        return _bench("flydsl", True, args.warmup, args.iters)
    return _correctness()


if __name__ == "__main__":
    sys.exit(main())
