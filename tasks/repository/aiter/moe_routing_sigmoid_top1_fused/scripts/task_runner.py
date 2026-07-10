#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import venv
from pathlib import Path

TASK_NAME = "repository/aiter/moe_routing_sigmoid_top1_fused"
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


def _benchmark_ms(fn, warmup: int = 10, rep: int = 30) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _write_performance_report(results: list[dict]) -> None:
    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "performance_report.json").write_text(json.dumps(results, indent=2))


def _make_case(*, M: int, N: int, K: int):
    import torch

    torch.manual_seed(7)
    dtype = torch.bfloat16
    x = torch.randint(-2, 3, (M, K), device="cuda").to(dtype)
    w = torch.randint(-2, 3, (K, N), device="cuda").to(dtype)
    dummy_ids = torch.ones((M, 1), dtype=torch.int32, device="cuda") * N
    dummy_weights = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    return {
        "params": {"M": M, "N": N, "K": K},
        "x": x,
        "w": w,
        "dummy_ids": dummy_ids,
        "dummy_weights": dummy_weights,
    }


def _run_kernel(case: dict):
    from aiter.ops.triton.moe.moe_routing_sigmoid_top1_fused import routing_sigmoid_top1

    return routing_sigmoid_top1(
        case["x"],
        case["w"],
        1,
        fused_shared_experts=True,
    )


def run_compile() -> None:
    case = _make_case(M=256, N=32, K=64)
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch
    from op_tests.triton_tests.moe.test_moe_routing_sigmoid_top1_fused import (
        torch_routing_sigmoid_top1,
    )

    cases = [
        _make_case(M=256, N=32, K=64),
        _make_case(M=1024, N=128, K=128),
    ]

    for idx, case in enumerate(cases):
        ids, weights = _run_kernel(case)
        ref_ids, ref_weights = torch_routing_sigmoid_top1(
            case["x"],
            case["w"],
            1,
            fused_shared_experts=True,
            dummy_ids=case["dummy_ids"],
            dummy_weights=case["dummy_weights"],
        )
        torch.testing.assert_close(ids, ref_ids, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weights, ref_weights, atol=1e-5, rtol=1e-5)
        print(f"Correctness case {idx}: PASS")


def run_performance() -> None:
    benchmark_cases = [
        ("moe_routing_small", _make_case(M=1024, N=16, K=128)),
        ("moe_routing_medium", _make_case(M=4096, N=128, K=128)),
    ]

    results: list[dict] = []
    for test_case_id, case in benchmark_cases:
        _run_kernel(case)
        time_ms = _benchmark_ms(lambda: _run_kernel(case))
        params = case["params"]
        results.append(
            {
                "test_case_id": test_case_id,
                "shape": [params["M"], params["N"], params["K"]],
                "execution_time_ms": time_ms,
                "metadata": params,
            }
        )
        print(f"{test_case_id}: {time_ms:.4f} ms")

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
