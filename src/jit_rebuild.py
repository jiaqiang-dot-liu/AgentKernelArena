# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Force JIT-compiled kernels to rebuild from the CURRENT source.

aiter ships a prebuilt in-tree ``.so`` in the image and, by default, loads it
WITHOUT checking whether the kernel source changed. An agent's edit to a HIP
``.cu``/``.cuh`` would then be silently ignored — the benchmark keeps running the
ORIGINAL kernel, correctness validates the original, and the speedup never moves.

This is a centralized safety net so individual task authors never have to
remember a per-task rebuild tweak. It is applied before every compile /
correctness / performance step (see ``src/evaluator.py`` + ``src/performance.py``),
so it holds identically for the baseline, any agent (cursor / claude / forge),
and the final re-score.

Scope: aiter HIP (C/C++) kernels only. Triton / Python kernels re-key their JIT
on the source and recompile on edit, so they are left untouched — and forcing an
aiter rebuild for a Triton task would trigger a slow, pointless C++ recompile.

Note: sglang (tvm-ffi) tasks are intentionally NOT handled here yet; that path
needs a kernel-name-agnostic cache-clear and is deferred.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# C/C++ (HIP/CUDA) kernel sources — these are the ones aiter ships prebuilt and
# can shadow with a stale .so. Triton/Python (.py) are not listed: their JIT
# keys by source and recompiles on edit.
_CPP_EXTS = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp")


def _task_paths(task_config: dict[str, Any]) -> list[str]:
    """Kernel/source + repo path strings used to detect the framework + language."""
    paths: list[str] = []
    sfp = task_config.get("source_file_path") or []
    if isinstance(sfp, str):
        sfp = [sfp]
    paths.extend(str(s) for s in sfp if s)
    for key in ("image_repo_path", "repo_url", "repo_subdir"):
        val = task_config.get(key)
        if val:
            paths.append(str(val))
    return paths


def force_jit_rebuild(task_config: dict[str, Any], logger: logging.Logger | None = None) -> None:
    """Best-effort: make the task's JIT kernel recompile from the current source.

    No-op unless the task is an aiter HIP (C/C++) kernel. Never raises — a
    rebuild-hint failure must not break evaluation.
    """
    lg = logger or log
    try:
        apply_jit_rebuild(_task_paths(task_config))
    except Exception as exc:  # noqa: BLE001 - best-effort
        lg.debug("force_jit_rebuild skipped: %r", exc)


def apply_jit_rebuild(paths) -> None:
    """Framework-keyed rebuild forcing from a list of path strings.

    aiter: set ``AITER_REBUILD=1`` so aiter rebuilds the kernel from source
    instead of loading the prebuilt in-tree ``.so``. Env is inherited by every
    build subprocess spawned afterwards.
    """
    strs = [str(p).lower() for p in paths if p]
    if not strs:
        return
    # Only C/C++ HIP kernels have the prebuilt-.so shadowing problem; forcing a
    # rebuild for a Triton (.py) task would just recompile aiter's C++ for nothing.
    if not any(s.endswith(_CPP_EXTS) for s in strs):
        return
    joined = " ".join(strs)

    if "aiter" in joined:
        os.environ.setdefault("AITER_REBUILD", "1")
