"""Repro + regression tests for the `image_kernel` task type and the forge
repo-subdir kernel-path resolution fix.

Run: python3 -m pytest tests/test_image_kernel.py -q
(These are pure-Python tests: no GPU, no network, no torch required.)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

LOG = logging.getLogger("test_image_kernel")


# --------------------------------------------------------------------------
# 1. forge kernel-file resolution must be repo-subdir aware (PR #52 fix).
#    Repository / image_kernel tasks put the source under a repo subdir, so a
#    kernel path given relative to the repo root must still resolve. Plain
#    workspace-root files (legacy triton2triton etc.) must keep working.
# --------------------------------------------------------------------------
def _mk_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "aiter" / "csrc" / "cpp_itfs" / "pa").mkdir(parents=True)
    (ws / "aiter" / "csrc" / "cpp_itfs" / "pa" / "pa_kernels.cuh").write_text("// kernel\n")
    (ws / "naive_softmax.py").write_text("# kernel\n")  # legacy workspace-root file
    return ws


def test_forge_resolve_workspace_relative(tmp_path):
    from agents.forge.launch_agent import _resolve_kernel_file

    ws = _mk_workspace(tmp_path)
    # workspace-relative path that includes the repo subdir
    got = _resolve_kernel_file(str(ws), ["aiter/csrc/cpp_itfs/pa/pa_kernels.cuh"], {})
    assert got == (ws / "aiter/csrc/cpp_itfs/pa/pa_kernels.cuh").resolve()


def test_forge_resolve_repo_root_relative(tmp_path):
    from agents.forge.launch_agent import _resolve_kernel_file

    ws = _mk_workspace(tmp_path)
    # path given relative to the repo root; repo_url implies the subdir name.
    cfg = {"repo_url": "https://github.com/ROCm/aiter.git"}
    got = _resolve_kernel_file(str(ws), ["csrc/cpp_itfs/pa/pa_kernels.cuh"], cfg)
    assert got == (ws / "aiter/csrc/cpp_itfs/pa/pa_kernels.cuh").resolve()


def test_forge_resolve_image_kernel_subdir(tmp_path):
    from agents.forge.launch_agent import _resolve_kernel_file

    ws = _mk_workspace(tmp_path)
    # image_kernel tasks: repo_subdir derived from image_repo_path basename.
    cfg = {"image_repo_path": "/sgl-workspace/aiter"}
    got = _resolve_kernel_file(str(ws), ["csrc/cpp_itfs/pa/pa_kernels.cuh"], cfg)
    assert got == (ws / "aiter/csrc/cpp_itfs/pa/pa_kernels.cuh").resolve()


def test_forge_resolve_legacy_root_file(tmp_path):
    from agents.forge.launch_agent import _resolve_kernel_file

    ws = _mk_workspace(tmp_path)
    got = _resolve_kernel_file(str(ws), ["naive_softmax.py"], {})
    assert got == (ws / "naive_softmax.py").resolve()


def test_forge_resolve_missing_raises(tmp_path):
    from agents.forge.launch_agent import _resolve_kernel_file

    ws = _mk_workspace(tmp_path)
    with pytest.raises(RuntimeError):
        _resolve_kernel_file(str(ws), ["does/not/exist.cu"], {})


# --------------------------------------------------------------------------
# 2. preprocessing: image_kernel seeds the repo from an in-image path
#    (copy, no clone), excluding .git, idempotently.
# --------------------------------------------------------------------------
def _mk_fake_image_repo(tmp_path: Path) -> Path:
    src = tmp_path / "image_aiter"
    (src / "csrc").mkdir(parents=True)
    (src / "csrc" / "k.cuh").write_text("// k\n")
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("[core]\n")
    return src


def test_seed_from_image_copies_without_git(tmp_path):
    from src.preprocessing import _ensure_repo_seeded_from_image

    src = _mk_fake_image_repo(tmp_path)
    dst = tmp_path / "tasks" / "aiter"
    did = _ensure_repo_seeded_from_image(src, dst, LOG)
    assert did is True
    assert (dst / "csrc" / "k.cuh").exists()
    assert not (dst / ".git").exists()  # .git excluded


def test_seed_from_image_idempotent(tmp_path):
    from src.preprocessing import _ensure_repo_seeded_from_image

    src = _mk_fake_image_repo(tmp_path)
    dst = tmp_path / "tasks" / "aiter"
    assert _ensure_repo_seeded_from_image(src, dst, LOG) is True
    # second call: already seeded -> no re-copy
    assert _ensure_repo_seeded_from_image(src, dst, LOG) is False


def test_seed_from_image_missing_raises(tmp_path):
    from src.preprocessing import _ensure_repo_seeded_from_image

    with pytest.raises(RuntimeError):
        _ensure_repo_seeded_from_image(tmp_path / "nope", tmp_path / "dst", LOG)


# --------------------------------------------------------------------------
# 3. image_kernel task-type prompt + prompt_builder wiring.
# --------------------------------------------------------------------------
def test_image_kernel_prompt_nonempty():
    from src.prompts import task_type

    txt = task_type.image_kernel_task_type()
    assert isinstance(txt, str) and len(txt) > 50


def test_prompt_builder_accepts_image_kernel(tmp_path):
    import yaml
    from src.prompt_builder import prompt_builder

    task_dir = tmp_path / "img_task"
    task_dir.mkdir()
    cfg = {
        "task_type": "image_kernel",
        "image_repo_path": "/sgl-workspace/aiter",
        "repository_language": "hip",
        "source_file_path": ["csrc/cpp_itfs/pa/pa_kernels.cuh"],
        "target_kernel_functions": ["paged_attention_ll4mi_QKV_mfma16_kernel"],
        "compile_command": ["python3 scripts/task_runner.py compile"],
        "correctness_command": ["python3 scripts/task_runner.py correctness"],
        "performance_command": ["python3 scripts/task_runner.py performance"],
    }
    cfg_path = task_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    prompt = prompt_builder(str(cfg_path), task_dir, {"target_gpu_model": "MI325X"}, LOG)
    assert isinstance(prompt, str) and len(prompt) > 0
    # must have selected the image_kernel task-type prompt (not raised "Unknown task type")
    assert "image" in prompt.lower()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
