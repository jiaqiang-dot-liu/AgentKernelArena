#!/usr/bin/env python3
"""Arena harness for the FlyDSL rewrite of Kimi-K3 KDA linear attention.

Same operator and same session cases as
``mi355x_vllm_triton_kda_linear_attn_kimi_k3``, but the target language is
FlyDSL: the vendored Triton implementation under
``vllm/models/kimi_k3/amd/ops/third_party/kda`` is the oracle and the baseline,
and the agent produces ``kernel.py``.

Both states are valid, which is what makes baseline and optimized runs
comparable:

* no (or stubbed) ``kernel.py`` -- measure the Triton source; baseline run;
* a working ``kernel.py``       -- measure the FlyDSL port.

Correctness always validates whichever implementation is under test against an
independent float64 golden. That is a strictly stronger oracle than comparing
the port to Triton, because it also catches a bug the port would inherit by
imitating the source. ``scripts/forge_driver.py`` additionally gates the port
against the live Triton output on the SNR threshold the rewrite pipeline
enforces.

Contract details that the harness must honour (all read off the kernel sources):
  * ``A_log`` is 1-D of length ``local_num_heads``; ``dt_bias`` is
    ``local_num_heads * head_dim`` (kimi_gdn_linear_attn.py:238,266).
  * ``raw_beta`` is PRE-sigmoid -- the source applies ``sigmoid`` internally
    (fused_recurrent.py:525, chunk.py:470).
  * K3 sets ``gate_lower_bound = -5.0``, which selects the *safe gate* branch
    ``gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``. This is a
    different function from the softplus branch used when the bound is unset
    (fused_recurrent.py:513-521, chunk.py:507-515).
  * ``state_indices`` entries must be > 0 in the decode cache; ``<= 0`` is the
    NULL slot and makes the source emit zeros (fused_recurrent.py:481).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]
KDA_CONFIG = SPEC["kda_config"]
# K3 linear_attn_config.gate_lower_bound; selects the safe-gate branch.
GATE_LOWER_BOUND = KDA_CONFIG["gate_lower_bound"]
CHUNK_SIZE = KDA_CONFIG["chunk_size"]

REPO_SUBDIR = "vllm"
CONTRACT = SPEC.get("rewrite_contract", {})
OP_NAME = CONTRACT.get("op_name", "kda_linear_attn")
BUILDER = CONTRACT.get("builder_symbol", f"build_{OP_NAME}_module")

# Profiling is a single-shape probe, pinned rather than derived from timings so
# the profiled kernel never drifts between runs.
PROFILE_CASE_ID = SPEC.get("profile_case") or CASES[0]["id"]

# The golden is an O(T) float64 recurrence, so correctness caps the token count.
# 320 still spans 5 chunks at chunk_size=64 and exercises the cross-chunk path.
# Unlike the sibling Triton task the segment count is NOT capped: a rewrite gets
# varlen segment indexing wrong far more easily than the in-tree source does,
# and 62 extra python-loop steps cost nothing.
CORRECTNESS_MAX_TOKENS = 320


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # The workspace-seeded copy of vllm has to shadow the in-image install so the
    # oracle is the reviewed source. Without this every `import vllm` resolves to
    # /usr/local/lib/python3.12/dist-packages/vllm.
    if (WORKSPACE / REPO_SUBDIR / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    # Triton keys its JIT cache on the kernel source, so an edit already forces a
    # recompile. Pinning the cache inside the workspace additionally guarantees a
    # run can never serve a binary compiled from a different workspace's source.
    os.environ.setdefault("TRITON_CACHE_DIR", str(WORKSPACE / "build" / "triton_cache"))
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def flydsl_builder():
    """Return the FlyDSL port's builder, or None when it is absent or a stub."""
    path = WORKSPACE / "kernel.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ka_flydsl_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ka_flydsl_kernel"] = module
    spec.loader.exec_module(module)
    return getattr(module, BUILDER, None)


def profile_case() -> dict:
    """The single case profiling runs against (see PROFILE_CASE_ID)."""
    for case in CASES:
        if case["id"] == PROFILE_CASE_ID:
            return case
    raise KeyError(f"profile_case {PROFILE_CASE_ID!r} is not in session_cases.json")


def _kda_ops():
    """AMD/ROCm vendored KDA entry points (the copy kimi_gdn_linear_attn.py picks)."""
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda_packed_decode,
    )

    return chunk_kda_with_fused_gate, fused_recurrent_kda_packed_decode


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict, correctness: bool = False) -> dict:
    """Inputs shaped exactly as the vLLM KDA layer holds them.

    The generation order matches mi355x_vllm_triton_kda_linear_attn_kimi_k3 so
    that, for a given case and seed, both tasks measure the same numbers.
    """
    torch = _torch()
    p = dict(case["params"])
    H = p["num_heads"]          # per-rank heads (K3 num_heads=96, TP=8 -> 12)
    D = p["head_dim"]           # d_k = d_v = 128
    mode = p["mode"]            # "chunk" (prefill) | "packed_decode" (k007)
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

    inp = {
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
        inp["q"] = rnd(1, total_t, H, D, scale=0.5)
        inp["k"] = rnd(1, total_t, H, D, scale=0.5)
        inp["v"] = rnd(1, total_t, H, D, scale=0.5)
        # chunk takes one state per sequence, [N, H, V, K].
        state = rnd(num_seqs, H, D, D, dtype=torch.float32, scale=0.1).contiguous()
        inp["cu_seqlens"] = torch.arange(
            0, (num_seqs + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32
        )
        inp["state_indices"] = torch.arange(
            num_seqs, device="cuda", dtype=torch.int32
        )
        inp["q_flat"] = inp["q"][0]
        inp["k_flat"] = inp["k"][0]
        inp["v_flat"] = inp["v"][0]
        inp["seg_state0"] = [state[n].double().clone() for n in range(num_seqs)]
        inp["segments"] = [(n * seq_len, (n + 1) * seq_len) for n in range(num_seqs)]
    else:
        # packed decode consumes the post-conv fused QKV block, [B, 3 * H * D],
        # laid out q | k | v with each part head-major (fused_recurrent.py:491-502).
        mixed = rnd(total_t, 3 * H * D, scale=0.5).contiguous()
        inp["mixed_qkv"] = mixed
        # A state cache with slot 0 reserved as the NULL slot; indices start at 1.
        state = rnd(num_seqs + 1, H, D, D, dtype=torch.float32, scale=0.1).contiguous()
        inp["state_indices"] = torch.arange(
            1, num_seqs + 1, device="cuda", dtype=torch.int32
        )
        # One token per segment.
        inp["cu_seqlens"] = torch.arange(
            total_t + 1, device="cuda", dtype=torch.int32
        )
        # Strided [T, H, D] views of the three thirds -- no copy, so the port
        # sees the same memory traffic the Triton kernel does.
        base = mixed.storage_offset()
        inp["q_flat"], inp["k_flat"], inp["v_flat"] = (
            torch.as_strided(mixed, (total_t, H, D), (3 * H * D, D, 1), base + off)
            for off in (0, H * D, 2 * H * D)
        )
        inp["seg_state0"] = [state[n + 1].double().clone() for n in range(num_seqs)]
        inp["segments"] = [(n, n + 1) for n in range(num_seqs)]

    # Both implementations update the state in place, so the golden's starting
    # state is snapshotted above (seg_state0) BEFORE anything touches it.
    inp["state"] = state
    return inp


def _run_triton(inp: dict):
    chunk_kda_with_fused_gate, fused_recurrent_kda_packed_decode = _kda_ops()
    if inp["mode"] == "chunk":
        out, _state = chunk_kda_with_fused_gate(
            q=inp["q"], k=inp["k"], v=inp["v"],
            raw_g=inp["raw_g"], raw_beta=inp["raw_beta"], A_log=inp["A_log"],
            g_bias=inp["dt_bias"], lower_bound=inp["lower_bound"],
            initial_state=inp["state"], output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu_seqlens"],
        )
    else:
        out, _state = fused_recurrent_kda_packed_decode(
            mixed_qkv=inp["mixed_qkv"], raw_g=inp["raw_g"], raw_beta=inp["raw_beta"],
            A_log=inp["A_log"], dt_bias=inp["dt_bias"],
            lower_bound=inp["lower_bound"],
            initial_state=inp["state"], state_indices=inp["state_indices"],
        )
    return out.reshape(inp["total_t"], inp["H"], inp["D"])


def _make_runner(inp: dict, builder):
    """Bind the implementation under test: the FlyDSL port when present."""
    if builder is None:
        return (lambda: _run_triton(inp)), "triton"
    launch = builder(inp["H"], inp["D"], inp["chunk_size"])

    def run():
        launch(
            inp["out"], inp["q_flat"], inp["k_flat"], inp["v_flat"],
            inp["g_flat"], inp["beta_flat"], inp["A_log"], inp["dt_bias"],
            inp["state"], inp["state_indices"], inp["cu_seqlens"],
            inp["scale"], inp["lower_bound"],
        )
        return inp["out"]

    return run, "flydsl"


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def _golden(inp: dict):
    """Independent float64 transcription of the KDA gated-delta-rule recurrence.

    Taken directly from ``fused_recurrent_kda_packed_decode_kernel``
    (fused_recurrent.py:504-533); ``chunk_kda_with_fused_gate`` computes the same
    recurrence blockwise, so one reference covers both modes:

        g_t   = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))   # safe gate
        q     = l2norm(q_t) * scale ;  k = l2norm(k_t) ;  v = v_t
        S     = S * exp(g_t)              # decay per k-column
        v     = v - S @ k                 # delta-rule "remove old value"
        v     = v * sigmoid(raw_beta_t)
        S     = S + outer(v, k)
        o_t   = S @ q

    Returns ``(out[T, H, D], final_states)`` where ``final_states[n]`` is the
    float64 state each segment ends on.
    """
    torch = _torch()
    H, D, scale = inp["H"], inp["D"], inp["scale"]

    q = inp["q_flat"].double()
    k = inp["k_flat"].double()
    v = inp["v_flat"].double()

    raw_g = inp["raw_g"][0].double()
    a = torch.exp(inp["A_log"].double()).view(1, H, 1)
    pre_gate = raw_g + inp["dt_bias"].reshape(H, D).double().view(1, H, D)
    if GATE_LOWER_BOUND is None:
        gate = -a * torch.nn.functional.softplus(pre_gate)
    else:
        gate = GATE_LOWER_BOUND * torch.sigmoid(a * pre_gate)
    beta = torch.sigmoid(inp["raw_beta"][0].double())

    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    out = torch.zeros(inp["total_t"], H, D, dtype=torch.float64, device="cuda")
    final_states = []
    for (bos, eos), S0 in zip(inp["segments"], inp["seg_state0"]):
        S = S0.clone()                                     # [H, V, K]
        for t in range(bos, eos):
            S = S * torch.exp(gate[t]).unsqueeze(1)        # decay k-columns
            vt = v[t] - (S * kn[t].unsqueeze(1)).sum(-1)   # v - S @ k  -> [H, V]
            vt = vt * beta[t].unsqueeze(-1)
            S = S + vt.unsqueeze(2) * kn[t].unsqueeze(1)   # + outer(v, k)
            out[t] = (S * qn[t].unsqueeze(1)).sum(-1)      # S @ q
        final_states.append(S)
    return out, final_states


def _check_final_state(inp: dict, ref_states: list, tol: dict, impl: str,
                       case_id: str):
    """Assert the port left the right recurrent state behind.

    The state is an output, not scratch: decode feeds it straight back in on the
    next step, so a kernel that gets ``out`` right and the state wrong is broken
    in a way ``out`` alone cannot see (with one token per segment, ``out`` only
    observes the state through the q projection).

    Checked for the FlyDSL port only, whose contract fixes the update to
    ``state[state_indices[n]]``. The Triton source is the oracle rather than the
    subject, and how it surfaces the final state is its own business.
    """
    if impl != "flydsl":
        return None
    torch = _torch()
    slots = inp["state_indices"].tolist()
    worst = 0.0
    for n, slot in enumerate(slots):
        got = inp["state"][slot].double()
        gold = ref_states[n]
        denom = gold.abs().max().clamp_min(1e-8)
        rel = ((got - gold).abs().max() / denom).item()
        worst = max(worst, rel)
    assert worst < tol.get("max_rel_err", 0.03), (
        case_id, f"final state normalized max err {worst:.4f} too high"
    )
    return worst


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inp = _prepare(CASES[0], correctness=True)
    run, impl = _make_runner(inp, flydsl_builder())
    out = run()
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke ({impl}): PASS  out={tuple(out.shape)}")


def run_correctness() -> None:
    torch = _torch()
    builder = flydsl_builder()
    impl = "flydsl" if builder is not None else "triton"
    for case in CASES:
        inp = _prepare(case, correctness=True)
        # BEFORE the kernel: the state is mutated in place.
        ref, ref_states = _golden(inp)
        # The output buffer is written, never accumulated: poison it so an
        # implementation that leaves rows untouched cannot pass on allocator
        # leftovers.
        inp["out"].fill_(float("nan"))
        run, _ = _make_runner(inp, builder)
        out = run()
        torch.cuda.synchronize()

        assert torch.isfinite(out).all(), (case["id"], "non-finite output")
        expected_shape = (inp["total_t"], inp["H"], inp["D"])
        assert tuple(out.shape) == expected_shape, (
            case["id"], tuple(out.shape), expected_shape
        )

        got = out.double().flatten()
        gold = ref.flatten()
        cos = torch.nn.functional.cosine_similarity(got, gold, dim=0).item()
        denom = gold.abs().max().clamp_min(1e-8)
        rel_max = ((got - gold).abs().max() / denom).item()
        tol = case["params"]
        assert cos > tol.get("min_cosine", 0.999), (
            case["id"], f"cosine {cos:.6f} vs float64 golden too low"
        )
        assert rel_max < tol.get("max_rel_err", 0.03), (
            case["id"], f"normalized max err {rel_max:.4f} too high"
        )

        state_rel = _check_final_state(inp, ref_states, tol, impl, case["id"])
        print(
            f"correctness PASS ({impl})", case["id"],
            f"cos={cos:.6f} rel_max_err={rel_max:.4f} |o|={got.norm().item():.3f}",
            f"state_rel_max_err={state_rel:.4f}" if state_rel is not None else
            "state=source-owned",
        )


def run_performance() -> None:
    builder = flydsl_builder()
    rows = []
    for case in CASES:
        inp = _prepare(case, correctness=False)
        run, impl = _make_runner(inp, builder)
        run()
        _torch().cuda.synchronize()
        bench = case.get("benchmark", {})
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            run,
            warmup=bench.get("warmup", 3),
            repetition=bench.get("repetition", 20),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
        )
        metadata = {
            **case["params"],
            "implementation": impl,
            "model": case.get("model"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": case.get("trace_input_shapes"),
            "execution_time_ms": exec_ms,
            # Flat, not nested: src/testcases.py reads benchmark_method from the
            # top level of each row when building TestCaseResult.metadata.
            **{k: v for k, v in meta.items() if k.startswith("benchmark_")},
            "metadata": metadata,
        })
        print(case["id"], f"{exec_ms:.6f} ms", impl, meta.get("benchmark_method"),
              meta.get("benchmark_fallback_reason", ""))
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if mode == "compile":
        run_compile()
    elif mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
