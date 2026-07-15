#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

TASK_NAME = "repository/aiter/pa_prefill"
REPO_SUBDIR = "aiter"
VENV_PACKAGES = [
    "psutil",
    "pytest",
    "pybind11",
    "ninja",
    "pyyaml",
    "pandas",
    "einops",
    "packaging",
]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _workspace_root() / REPO_SUBDIR


def _report_root() -> Path:
    return _workspace_root() / "build"


def _venv_dir() -> Path:
    return _workspace_root() / ".task-venv"


def _venv_python() -> Path:
    return _venv_dir() / "bin" / "python"


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


# Extra runtime imports the task needs beyond VENV_PACKAGES (the kernel is
# Triton and always needs torch); relied on the container previously.
_RUNTIME_IMPORTS = ("torch", "triton")
# pip package names that differ from their import name.
_IMPORT_NAME_OVERRIDES = {"pyyaml": "yaml"}


def _current_interp_has_deps() -> bool:
    """True if the active interpreter can already import every dependency.

    In a fully provisioned container the deps are already present, so building a
    separate venv is redundant. It is also actively BROKEN inside a venv-based
    container (e.g. /opt/venv): a venv created with system_site_packages=True
    chains to sys.base_prefix (/usr), NOT the active venv, so torch installed
    under /opt/venv becomes invisible and `import torch` fails. Running in place
    avoids that trap.
    """
    import importlib.util

    required = list(_RUNTIME_IMPORTS) + [
        _IMPORT_NAME_OVERRIDES.get(p, p) for p in VENV_PACKAGES
    ]
    for name in required:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except Exception:
            return False
    return True


def _active_site_packages() -> list[str]:
    """site-packages dirs of the ACTIVE interpreter (the running venv, if any)."""
    import site
    import sysconfig

    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    purelib = sysconfig.get_path("purelib")
    if purelib:
        candidates.append(purelib)
    result: list[str] = []
    for p in candidates:
        if p and p not in result:
            result.append(p)
    return result


def _ensure_venv() -> None:
    # Fast path: if the active interpreter already provides every dependency, run
    # in place — a fresh venv is redundant and would break in a venv-based
    # container (see _current_interp_has_deps).
    if _current_interp_has_deps():
        return

    venv_dir = _venv_dir()
    venv_python = _venv_python()
    ready_marker = venv_dir / ".ready"
    script_path = Path(__file__).resolve()

    if not venv_python.exists():
        builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
        builder.create(str(venv_dir))

    if not ready_marker.exists():
        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-q",
                *VENV_PACKAGES,
            ],
            cwd=_workspace_root(),
        )
        ready_marker.write_text("\n".join(VENV_PACKAGES) + "\n")

    current_python = Path(sys.executable).resolve()
    if current_python != venv_python.resolve():
        env = os.environ.copy()
        env.setdefault("ENABLE_CK", "0")
        env.setdefault("AITER_LOG_LEVEL", "WARNING")
        # PYTHONPATH for the re-exec'd venv: repo first, then the active
        # interpreter's site-packages so a venv-based container's torch/triton
        # (which system_site_packages cannot reach) stay importable.
        parts = [str(_repo_root()), *_active_site_packages()]
        existing = env.get("PYTHONPATH", "")
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        os.execve(
            str(venv_python),
            [str(venv_python), str(script_path), *sys.argv[1:]],
            env,
        )


def _configure_runtime() -> None:
    os.environ.setdefault("ENABLE_CK", "0")
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)


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


def _write_performance_report(results: list[dict]) -> None:
    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "performance_report.json").write_text(json.dumps(results, indent=2))


def _make_case(
    *,
    BS: int,
    MAX_SEQ_LEN: int,
    MAX_CTX_LEN: int,
    cache_size: int,
    block_size: int,
    max_block_per_request: int,
    num_heads: int,
    head_size: int,
    num_queries_per_kv: int,
    sliding_window: int,
):
    import torch
    from op_tests.triton_tests.attention.test_pa_prefill import input_helper

    (
        query,
        k,
        v,
        output,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        _,
    ) = input_helper(
        BS=BS,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        MAX_CTX_LEN=MAX_CTX_LEN,
        cache_size=cache_size,
        block_size=block_size,
        max_block_per_request=max_block_per_request,
        num_heads=num_heads,
        head_size=head_size,
        num_queries_per_kv=num_queries_per_kv,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        device="cuda:0",
        use_alibi_slope=False,
    )
    # input_helper draws q/k/v in [-1e-3, 1e-3]. At that magnitude the attention
    # logits are ~0 so softmax is ~uniform and the output is ~1e-3 — far below the
    # correctness tolerance — which makes the check non-discriminating (an all-zero
    # or mis-scaled output still "passes"). Scale inputs up to a realistic
    # magnitude so the softmax is non-degenerate and the output is O(0.1-1); the
    # reference below is computed in fp32 so this does not overflow.
    _CORR_INPUT_SCALE = 1000.0
    query = query * _CORR_INPUT_SCALE
    k = k * _CORR_INPUT_SCALE
    v = v * _CORR_INPUT_SCALE
    k_cache = k_cache * _CORR_INPUT_SCALE
    v_cache = v_cache * _CORR_INPUT_SCALE
    return {
        "params": {
            "BS": BS,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "MAX_CTX_LEN": MAX_CTX_LEN,
            "cache_size": cache_size,
            "block_size": block_size,
            "max_block_per_request": max_block_per_request,
            "num_heads": num_heads,
            "head_size": head_size,
            "num_queries_per_kv": num_queries_per_kv,
            "sliding_window": sliding_window,
        },
        "query": query,
        "k": k,
        "v": v,
        "output": output,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "b_start_loc": b_start_loc,
        "b_seq_len": b_seq_len,
        "max_input_len": max_input_len,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def _run_kernel(case: dict) -> None:
    from aiter.ops.triton.attention.pa_prefill import context_attention_fwd

    context_attention_fwd(
        case["query"],
        case["k"],
        case["v"],
        case["output"],
        "auto",
        case["k_cache"],
        case["v_cache"],
        case["block_table"],
        case["b_start_loc"],
        case["b_seq_len"],
        case["max_input_len"],
        case["k_scale"],
        case["v_scale"],
        sliding_window=case["params"]["sliding_window"],
    )


def run_compile() -> None:
    case = _make_case(
        BS=2,
        MAX_SEQ_LEN=32,
        MAX_CTX_LEN=32,
        cache_size=64,
        block_size=16,
        max_block_per_request=4,
        num_heads=64,
        head_size=24,
        num_queries_per_kv=64,
        sliding_window=0,
    )
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def _reference_paged_prefill(case: dict):
    """Correct fp32 reference for aiter's paged-prefill attention.

    op_tests' ``context_attention_fwd_torch`` is unusable as a reference here: it
    (a) reads the paged KV cache by sequential physical slot, ignoring the block
    table (so with a randomized block table it reads the wrong blocks), and
    (b) adds two independently-normalized softmaxes (context + causal self)
    instead of folding them into one — both wrong, leaving the pristine kernel
    ~92% off the reference while still "passing" only because the [-1e-3, 1e-3]
    inputs make the output tiny vs the tolerance.

    This reference gathers the context KV via the block table and computes a
    single softmax over [context; causal self] with the kernel's sliding-window
    semantics, in fp32.
    """
    import torch

    p = case["params"]
    device = case["query"].device
    query = case["query"].float()
    k = case["k"].float()
    v = case["v"].float()
    k_cache = case["k_cache"].float()
    v_cache = case["v_cache"].float()
    block_table = case["block_table"]
    b_start_loc = case["b_start_loc"]
    b_seq_len = case["b_seq_len"]
    block_size = p["block_size"]
    head_size = p["head_size"]
    num_heads = p["num_heads"]
    num_queries_per_kv = p["num_queries_per_kv"]
    sliding_window = p["sliding_window"]
    sm_scale = 1.0 / (head_size ** 0.5)

    out = torch.zeros_like(query)
    for b in range(p["BS"]):
        qs = int(b_start_loc[b])
        qe = int(b_start_loc[b + 1])
        q_len = qe - qs
        ctx_len = int(b_seq_len[b]) - q_len
        n_blk = (ctx_len + block_size - 1) // block_size
        phys = block_table[b, :n_blk].long()
        # Absolute positions: context is [0, ctx_len), self is [ctx_len, ctx_len+q_len).
        qpos = ctx_len + torch.arange(q_len, device=device)
        kpos = torch.arange(ctx_len + q_len, device=device)
        disallow = kpos[None, :] > qpos[:, None]  # causal
        if sliding_window and sliding_window > 0:
            disallow = disallow | ((qpos[:, None] - kpos[None, :]) >= sliding_window)
        for h in range(num_heads):
            kv_h = h // num_queries_per_kv
            qh = query[qs:qe, h]  # [q_len, D]
            if ctx_len > 0:
                kc = k_cache[phys, kv_h].permute(0, 2, 1, 3).reshape(-1, head_size)[:ctx_len]
                vc = v_cache[phys, kv_h].permute(0, 2, 1).reshape(-1, head_size)[:ctx_len]
            else:
                kc = query.new_zeros((0, head_size))
                vc = query.new_zeros((0, head_size))
            key = torch.cat([kc, k[qs:qe, kv_h]], dim=0)  # [ctx_len+q_len, D]
            val = torch.cat([vc, v[qs:qe, kv_h]], dim=0)
            scores = torch.matmul(qh, key.transpose(0, 1)) * sm_scale
            scores = scores.masked_fill(disallow, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[qs:qe, h] = torch.matmul(probs, val)
    return out


def run_correctness() -> None:
    cases = [
        _make_case(
            BS=2,
            MAX_SEQ_LEN=32,
            MAX_CTX_LEN=32,
            cache_size=64,
            block_size=16,
            max_block_per_request=4,
            num_heads=64,
            head_size=24,
            num_queries_per_kv=64,
            sliding_window=0,
        ),
        _make_case(
            BS=2,
            MAX_SEQ_LEN=64,
            MAX_CTX_LEN=64,
            cache_size=128,
            block_size=16,
            max_block_per_request=8,
            num_heads=16,
            head_size=64,
            num_queries_per_kv=1,
            sliding_window=16,
        ),
    ]

    for idx, case in enumerate(cases):
        _run_kernel(case)
        ref = _reference_paged_prefill(case).float()
        got = case["output"].float()
        # Scale-relative L2 error: magnitude-robust and discriminating — a zeroed
        # or mis-scaled kernel output yields ~1.0, a correct kernel ~1e-3.
        rel = ((got - ref).norm() / ref.norm().clamp_min(1e-8)).item()
        if rel >= 2e-2:
            raise AssertionError(
                f"Correctness case {idx}: relative L2 error {rel:.4e} exceeds 2e-2"
            )
        print(f"Correctness case {idx}: PASS (rel_l2={rel:.2e})")


def run_performance() -> None:
    benchmark_cases = [
        (
            "pa_prefill_small",
            _make_case(
                BS=2,
                MAX_SEQ_LEN=64,
                MAX_CTX_LEN=64,
                cache_size=128,
                block_size=16,
                max_block_per_request=8,
                num_heads=16,
                head_size=64,
                num_queries_per_kv=1,
                sliding_window=0,
            ),
        ),
        (
            "pa_prefill_medium",
            _make_case(
                BS=4,
                MAX_SEQ_LEN=128,
                MAX_CTX_LEN=128,
                cache_size=256,
                block_size=16,
                max_block_per_request=16,
                num_heads=64,
                head_size=24,
                num_queries_per_kv=64,
                sliding_window=32,
            ),
        ),
    ]

    results: list[dict] = []
    for test_case_id, case in benchmark_cases:
        _run_kernel(case)
        time_ms, bench_meta = _benchmark_cuda_graph_or_events(lambda: _run_kernel(case))
        params = case["params"]
        results.append(
            {
                "test_case_id": test_case_id,
                "shape": [
                    params["BS"],
                    params["num_heads"],
                    params["head_size"],
                    params["MAX_SEQ_LEN"],
                    params["MAX_CTX_LEN"],
                ],
                "execution_time_ms": time_ms,
                "metadata": params,
                "benchmark_method": bench_meta.get("benchmark_method"),
            }
        )
        if bench_meta.get("benchmark_fallback_reason"):
            results[-1]["benchmark_fallback_reason"] = bench_meta["benchmark_fallback_reason"]
        print(f"{test_case_id}: {time_ms:.4f} ms [{bench_meta.get('benchmark_method')}]")

    _write_performance_report(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    _ensure_venv()
    _configure_runtime()

    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
