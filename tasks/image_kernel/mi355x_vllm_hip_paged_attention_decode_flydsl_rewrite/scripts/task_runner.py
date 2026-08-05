#!/usr/bin/env python3
"""Arena harness for the FlyDSL rewrite of ROCm paged-attention decode.

Same operator and same session cases as
``mi355x_vllm_hip_paged_attention_decode``, but the target language is FlyDSL:
AITER's ``paged_attention_rocm`` is the oracle and the baseline, and the agent
produces ``kernel.py``.

Both states are valid, which is what makes baseline and optimized runs
comparable:

* no (or stubbed) ``kernel.py`` -- measure AITER; this is the baseline run;
* a working ``kernel.py``       -- measure the FlyDSL port.

Correctness always validates whichever implementation is under test against an
fp32 torch reference. That is a strictly stronger oracle than comparing the port
to AITER, because it also catches a bug the port would inherit by imitating the
source. ``scripts/forge_driver.py`` additionally gates the port against the live
AITER output on the SNR threshold the rewrite pipeline enforces.
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

REPO_SUBDIR = "aiter_meta"
PARTITION_SIZE = 256
OP_NAME = SPEC.get("rewrite_contract", {}).get("op_name", "paged_attention_decode")
BUILDER = SPEC.get("rewrite_contract", {}).get(
    "builder_symbol", f"build_{OP_NAME}_module"
)

# Profiling is a single-shape probe, pinned rather than derived from timings so
# the profiled kernel never drifts between runs.
PROFILE_CASE_ID = SPEC.get("profile_case") or CASES[0]["id"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # Keep AITER's template-op build cache inside the workspace so parallel runs
    # cannot serve each other's kernels. No AITER_REBUILD here: the AITER source
    # is the protected oracle, not the edit surface, so forcing a rebuild every
    # step would cost ~15 s for nothing.
    os.environ.setdefault("AITER_ROOT_DIR", str(WORKSPACE / "build" / "aiter_root"))
    repo = WORKSPACE / REPO_SUBDIR
    if repo.is_dir():
        os.environ["AITER_META_DIR"] = str(repo)
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


def _write_report(rows: list[dict]) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def _import_aiter():
    import aiter

    seeded = os.environ.get("AITER_META_DIR")
    if seeded:
        import csrc.cpp_itfs.utils as cpp_itfs_utils

        core = Path(cpp_itfs_utils.AITER_CORE_DIR).resolve()
        if core != Path(seeded).resolve():
            raise RuntimeError(
                f"AITER template ops resolved to {core}, not the seeded tree "
                f"{seeded}; the oracle would not be the reviewed source."
            )
    return aiter


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


def _make(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    aiter = _import_aiter()
    p = dict(case["params"])
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
    # vLLM ROCm paged layout: the key cache splits head_size into (head_size//x, x)
    # with x = 16 bytes / element size, and the value cache is block-minor.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache = torch.zeros(
        (num_blocks, hkv, hs // x, bs, x), device="cuda", dtype=dtype
    )
    value_cache = torch.zeros((num_blocks, hkv, hs, bs), device="cuda", dtype=dtype)

    block_tables = torch.arange(
        num_seqs * pages, device="cuda", dtype=torch.int32
    ).view(num_seqs, pages)
    seq_idx = torch.arange(num_seqs, device="cuda").view(-1, 1).expand(-1, ctx_len)
    pos = torch.arange(ctx_len, device="cuda").view(1, -1).expand(num_seqs, -1)
    blk = block_tables[seq_idx, pos // bs].long().reshape(-1)
    off = (pos % bs).reshape(-1)
    key_cache[blk, :, :, off, :] = key.reshape(-1, hkv, hs).view(-1, hkv, hs // x, x)
    value_cache[blk, :, :, off] = value.reshape(-1, hkv, hs)

    parts = (ctx_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    return {
        "cfg": case,
        "aiter": aiter,
        "query": query,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": torch.full(
            (num_seqs,), ctx_len, device="cuda", dtype=torch.int32
        ),
        "out": torch.empty_like(query),
        "exp_sums": torch.empty(
            (num_seqs, hq, parts), device="cuda", dtype=torch.float32
        ),
        "max_logits": torch.empty(
            (num_seqs, hq, parts), device="cuda", dtype=torch.float32
        ),
        "tmp_out": torch.empty(
            (num_seqs, hq, parts, hs), device="cuda", dtype=dtype
        ),
        "one": torch.ones(1, device="cuda", dtype=torch.float32),
        "hq": hq,
        "hkv": hkv,
        "hs": hs,
        "bs": bs,
        "ctx_len": ctx_len,
        "scale": hs**-0.5,
    }


def _run_aiter(t: dict):
    t["aiter"].paged_attention_rocm(
        t["out"], t["exp_sums"], t["max_logits"], t["tmp_out"], t["query"],
        t["key_cache"], t["value_cache"], t["hkv"], t["scale"], t["block_tables"],
        t["seq_lens"], t["bs"], t["ctx_len"], None, "auto", t["one"], t["one"],
    )
    return t["out"]


def _make_runner(t: dict, builder):
    """Bind the implementation under test: the FlyDSL port when present."""
    if builder is None:
        return lambda: _run_aiter(t), "aiter"
    launch = builder(t["hq"], t["hkv"], t["hs"], t["bs"])

    def run():
        launch(
            t["out"], t["query"], t["key_cache"], t["value_cache"],
            t["block_tables"], t["seq_lens"], t["scale"],
        )
        return t["out"]

    return run, "flydsl"


def _run(inputs: dict):
    run, _ = _make_runner(inputs, flydsl_builder())
    return run()


def _reference(t: dict):
    torch = _torch()
    query = t["query"].float()
    key = t["key"].float()
    value = t["value"].float()
    ratio = query.shape[1] // key.shape[2]
    outputs = []
    for s in range(query.shape[0]):
        k = key[s].repeat_interleave(ratio, dim=1)
        v = value[s].repeat_interleave(ratio, dim=1)
        scores = torch.einsum("hd,khd->hk", query[s], k) * t["scale"]
        outputs.append(torch.einsum("hk,khd->hd", torch.softmax(scores, dim=-1), v))
    return torch.stack(outputs).to(t["out"].dtype)


def run_compile() -> None:
    inputs = _make(CASES[0], correctness=True)
    run, impl = _make_runner(inputs, flydsl_builder())
    run()
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke ({impl}): PASS")


def run_correctness() -> None:
    torch = _torch()
    builder = flydsl_builder()
    impl = "flydsl" if builder is not None else "aiter"
    for case in CASES:
        inputs = _make(case, correctness=True)
        # The output buffer is written, never accumulated: poison it so an
        # implementation that leaves rows untouched cannot pass on allocator
        # leftovers.
        inputs["out"].fill_(float("nan"))
        run, _ = _make_runner(inputs, builder)
        got = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, _reference(inputs), atol=0.02, rtol=0.02)
        print(f"correctness PASS ({impl})", case["id"])


def run_performance() -> None:
    builder = flydsl_builder()
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        run, impl = _make_runner(inputs, builder)
        run()
        _torch().cuda.synchronize()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            run,
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
        )
        metadata = {
            **case["params"],
            "implementation": impl,
            "model": case.get("model"),
            "session_id": case.get("session_id"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        metadata.update(
            {k: v for k, v in bench_meta.items() if k.startswith("benchmark_")}
        )
        rows.append(
            {
                "test_case_id": case["id"],
                "shape": case.get("trace_input_shapes"),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
        print(
            case["id"],
            f"{execution_time_ms:.6f} ms",
            impl,
            bench_meta.get("benchmark_method"),
            bench_meta.get("benchmark_fallback_reason", ""),
        )
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["compile", "correctness", "performance", "manifest"]
    )
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
