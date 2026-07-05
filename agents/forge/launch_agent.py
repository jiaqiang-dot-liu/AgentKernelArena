# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Forge agent — bridges AgentKernelArena to KernelForge's `forge-loop`.

KernelForge's autonomous optimization loop (baseline -> agent edit -> 5-stage
validate -> bench -> keep/revert) runs as a standalone, hard-killable subprocess
via `kernel-agents forge-loop`. This launcher adapts an Arena task workspace to
that loop's contract:

  1. Resolve the kernel file Arena copied into the workspace (task's
     ``source_file_path[0]``).
  2. Materialize a driver shim implementing the KernelForge driver contract
     (prints ``SNR: <db> dB`` for correctness and ``wall_ms: <ms>`` for bench).
  3. ``git init`` + initial commit the workspace (the loop uses git keep/revert).
  4. Generate a ``forge_program.md`` from the task prompt for agent guidance.
  5. Shell out to ``kernel-agents forge-loop`` (streaming output), which leaves
     the workspace at the best-kept kernel.

After this returns, Arena re-materializes its perf helpers and re-scores the
kernel with the task's own compile/correctness/performance commands.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent

# AMD GPU model (Arena) -> gfx arch (KernelForge / ROCm).
_GPU_ARCH_MAP = {
    "MI300": "gfx942",
    "MI300X": "gfx942",
    "MI325": "gfx942",
    "MI350": "gfx950",
    "MI355X": "gfx950",
    "MI355": "gfx950",
}


def _resolve_gpu_arch(eval_config: dict[str, Any]) -> str:
    """Map the Arena target_gpu_model to a gfx arch (env override wins)."""
    env_arch = os.environ.get("AGENT_KERNEL_ARENA_GPU_ARCH") or os.environ.get("AKA_GPU_ARCH")
    if env_arch:
        return env_arch
    model = str(eval_config.get("target_gpu_model", "")).upper()
    return _GPU_ARCH_MAP.get(model, "gfx942")


def _forge_max_hours(agent_config: dict[str, Any]) -> float:
    """Derive the forge-loop ``--max-hours`` budget from the run's timeout.

    bootstrap patches ``timeout_seconds`` from the run (KA_TIMEOUT_SECONDS) but
    NOT ``max_hours`` — so a static ``max_hours`` would cap the loop long before a
    larger ``timeout_seconds`` takes effect (e.g. a 32h run was still stopped at
    the default 8h). Track ``timeout_seconds`` with a small margin so the loop
    self-stops (BUDGET EXHAUSTED) just before the hard process-wait kill. The
    default timeout (29700s) yields ~8h, matching the previous static cap.
    """
    timeout_s = float(agent_config.get("timeout_seconds", 3600))
    return round(max(0.1, timeout_s / 3600.0 - 0.25), 3)


def _repo_subdir_name(task_config: dict[str, Any]) -> str | None:
    """Best-effort name of the repo subdir a repository/image_kernel task lives in."""
    explicit = task_config.get("repo_subdir")
    if explicit:
        return str(explicit)
    image_repo_path = task_config.get("image_repo_path")
    if image_repo_path:
        return Path(str(image_repo_path)).name
    repo_url = task_config.get("repo_url")
    if repo_url:
        url = str(repo_url).rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url.rsplit("/", 1)[-1]
    return None


def _resolve_kernel_file(workspace: str, source_files: list, task_config: dict[str, Any]) -> Path:
    """Locate the kernel file the agent must edit, robust to task layout.

    `source_file_path[0]` may be given either workspace-relative (legacy snippet
    tasks copy the file to the workspace root) or repo-root-relative (repository /
    image_kernel tasks put the sources under a repo subdir). Resolution order:
      1. as given, relative to the workspace root (preserves legacy behavior);
      2. under the repo subdir (repo_subdir / image_repo_path / repo_url basename);
      3. a unique match anywhere in the workspace whose path ends with the given
         suffix (last-resort, ignores .git).
    """
    rel = str(source_files[0])
    ws = Path(workspace)

    p = (ws / rel).resolve()
    if p.exists():
        return p

    subdir = _repo_subdir_name(task_config)
    if subdir:
        p2 = (ws / subdir / rel).resolve()
        if p2.exists():
            return p2

    tail = Path(rel)
    matches = [
        m for m in ws.rglob(tail.name)
        if str(m).endswith(rel) and ".git" not in m.parts
    ]
    if len(matches) == 1:
        return matches[0].resolve()

    raise RuntimeError(f"Kernel file not found in workspace: {ws / rel}")


# KernelForge fellows available to the single-fellow forge-loop path
# (see kernel_agents.fellows.base.build_single_fellow_prompt).
_VALID_FELLOW_BACKENDS = {"ck", "flydsl", "triton", "aiter", "hip", "hipblaslt"}


def _infer_backend(task_config: dict[str, Any]) -> str:
    """Infer the target backend from the task's task_type.

    Arena task types are "<source>2<target>" (e.g. triton2triton, hip2hip,
    cuda2hip, torch2hip, flydsl2flydsl, instruction2triton). The kernel the loop
    optimizes is in the TARGET language, so the backend is the part after the
    last '2'. Falls back to a keyword scan, then to triton.
    """
    task_type = str(task_config.get("task_type") or "").lower().strip()
    target = task_type.rsplit("2", 1)[-1] if "2" in task_type else task_type
    if target in _VALID_FELLOW_BACKENDS:
        return target
    for backend in _VALID_FELLOW_BACKENDS:
        if backend in task_type:
            return backend
    return "triton"


def _resolve_fellow(task_config: dict[str, Any], agent_config: dict[str, Any]) -> str:
    """Pick the fellow: explicit agent_config override wins, else inferred."""
    override = agent_config.get("fellow")
    if override:
        return str(override)
    return f"{_infer_backend(task_config)}-fellow"


def _ensure_rtk_shim() -> str:
    """Provide a no-op `rtk` passthrough on PATH if rtk is not installed.

    KernelForge's agent prompt instructs prefixing shell commands with `rtk`
    (a token-filtering proxy). When rtk is absent, those Bash calls would fail
    with 'command not found' and waste agent turns. A trivial passthrough shim
    (`exec "$@"`) keeps the commands working. Returns the bin dir to prepend.
    """
    if shutil.which("rtk"):
        return ""
    shim_dir = Path("/tmp/aka-forge-bin")
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "rtk"
    if not shim.exists():
        shim.write_text('#!/usr/bin/env bash\nexec "$@"\n')
        shim.chmod(0o755)
    return str(shim_dir)


def _git(workspace: str, *args: str, logger: logging.Logger) -> None:
    """Run a git command in the workspace, tolerating non-zero exit."""
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.debug(f"git {' '.join(args)} -> {result.returncode}: {result.stderr.strip()}")


# Build artifacts / regenerated reports / forge scaffolding must NOT be tracked:
# if they are, a validation or benchmark run that regenerates them dirties the
# tree and makes the loop's `git revert` fail — leaking a reverted (often broken)
# edit into the final tree. Only source is tracked, matching the loop's own
# `git add -u` philosophy.
_GITIGNORE = """\
__pycache__/
*.pyc
*.pyo
*.so
*.o
*.hsaco
*.pt
build/
perf/
*_perf.yaml
performance_report.json
perf_report.json
forge_experiments/
forge_driver.py
forge_program.md
.pytest_cache/
*.log
"""


def _init_git_workspace(workspace: str, logger: logging.Logger) -> None:
    """Initialize a git repo with an initial commit (required by forge-loop).

    Writes a .gitignore first so build artifacts and regenerated perf reports
    stay untracked — otherwise later tool runs dirty the tree and break the
    loop's keep/revert (git revert aborts on unstaged changes).
    """
    if not (Path(workspace) / ".git").exists():
        _git(workspace, "init", logger=logger)
    gitignore = Path(workspace) / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_GITIGNORE)
    # Local identity so commits succeed without global git config.
    _git(workspace, "config", "user.email", "forge-loop@local", logger=logger)
    _git(workspace, "config", "user.name", "forge-loop", logger=logger)
    # Untrack anything already staged/committed that the .gitignore now excludes
    # (e.g. build/ created by Arena's baseline step before this init).
    _git(workspace, "rm", "-r", "--cached", "--quiet", ".", logger=logger)
    _git(workspace, "add", "-A", logger=logger)
    _git(workspace, "commit", "-m", "forge: initial workspace snapshot", logger=logger)


def _write_program_md(task_config: dict[str, Any], target_funcs, gpu_arch: str, backend: str, dest: Path) -> None:
    """Generate a program.md for the forge agent from the Arena task prompt."""
    prompt_cfg = task_config.get("prompt") or {}
    instructions = prompt_cfg.get("instructions") or ""
    funcs = ", ".join(target_funcs) if target_funcs else "the target kernel"
    dest.write_text(
        f"""# Program: optimize {funcs}

**GPU**: {gpu_arch}
**Backend**: {backend}

## Objective
Optimize the body of `{funcs}` for maximum performance on {gpu_arch} while
keeping numerical results correct (the loop gates on an SNR threshold).

## Modification Rules
1. Make ONE logical change per iteration with a clear hypothesis.
2. Do NOT change the kernel's function signature or parameter list.
3. Do NOT remove imports or helper utilities in the file.
4. Correctness, validation, and benchmarking are run automatically by the loop
   after your edit — do not invoke build/test/bench tools yourself.

## Task instructions (from AgentKernelArena)
{instructions}
"""
    )


def _render_driver_shim(drivers_dir: str, workspace: str, task_config: str, arena_root: str) -> str:
    """Generate a forge_driver.py that bakes the task paths in (no env needed).

    The shim adds the shared adapter's dir to sys.path and delegates to
    ``arena_task_adapter.run``, passing workspace / task config / arena root as
    explicit arguments. KernelForge invokes this file as ``python forge_driver.py
    <args>`` (see kernel_agents.mcp_server.tools.{test,bench}), so a ``__main__``
    entry is all that is required.
    """
    return f"""\
#!/usr/bin/env python3
# Auto-generated by the AgentKernelArena forge launcher. Task paths are baked in
# below so the driver needs NO environment variables. It delegates to the shared
# arena_task_adapter, which reuses Arena's own correctness/performance eval.
import sys

sys.path.insert(0, {drivers_dir!r})
import arena_task_adapter as adapter

WORKSPACE = {workspace!r}
TASK_CONFIG = {task_config!r}
ARENA_ROOT = {arena_root!r}

if __name__ == "__main__":
    raise SystemExit(adapter.run(WORKSPACE, TASK_CONFIG, ARENA_ROOT, sys.argv[1:]))
"""


@register_agent("forge")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Run one KernelForge forge-loop over the Arena task workspace.

    Args:
        eval_config: Arena run config (includes target_gpu_model).
        task_config_dir: Path to the task's config.yaml (for source/target fields).
        workspace: Isolated task workspace Arena prepared; the kernel lives here.

    Returns:
        Combined streamed output of the forge-loop subprocess.
    """
    logger = logging.getLogger(__name__)

    forge_bin = shutil.which("kernel-agents")
    if not forge_bin:
        raise RuntimeError(
            "Command 'kernel-agents' not found. Install KernelForge "
            "(pip install -e KernelForge) so the forge-loop CLI is on PATH."
        )

    # Agent config
    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}

    # Task config: locate the kernel file + target function(s).
    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    source_files = task_config.get("source_file_path") or []
    if not source_files:
        raise RuntimeError(f"Task config has no source_file_path: {task_config_dir}")
    kernel_file = _resolve_kernel_file(workspace, source_files, task_config)
    target_funcs = task_config.get("target_kernel_functions") or []

    gpu_arch = _resolve_gpu_arch(eval_config)
    fellow = _resolve_fellow(task_config, agent_config)
    backend = fellow.split("-")[0]

    # Materialize the driver: prefer a task-shipped scripts/forge_driver.py
    # (used verbatim); otherwise generate a shim that delegates to the shared
    # arena_task_adapter with the task paths baked in — so the driver needs NO
    # environment variables.
    arena_root = str(Path(__file__).resolve().parents[2])
    driver_dest = Path(workspace) / "forge_driver.py"
    task_driver = Path(workspace) / "scripts" / "forge_driver.py"
    if task_driver.exists():
        shutil.copy2(task_driver, driver_dest)
        logger.info(f"Forge: using task-provided driver {task_driver}")
    else:
        drivers_dir = str(Path(__file__).parent / "drivers")
        workspace_task_config = Path(workspace) / "config.yaml"
        task_config_path = str(
            workspace_task_config if workspace_task_config.exists() else task_config_dir
        )
        driver_dest.write_text(_render_driver_shim(
            drivers_dir=drivers_dir,
            workspace=str(workspace),
            task_config=task_config_path,
            arena_root=arena_root,
        ))
        logger.info(f"Forge: generated driver shim -> {driver_dest} (task paths baked in)")

    # program.md for agent guidance.
    program_md = Path(workspace) / "forge_program.md"
    _write_program_md(task_config, target_funcs, gpu_arch, backend, program_md)

    # The loop needs a git repo for the keep/revert pattern.
    _init_git_workspace(workspace, logger)

    experiments_dir = Path(workspace) / "forge_experiments"
    result_json = experiments_dir / "forge_result.json"

    model = str(agent_config.get("model", "claude-opus-4-8"))
    permission_mode = str(agent_config.get("permission_mode", "acceptEdits"))

    # Build the forge-loop command. All task-specific config is passed as flags
    # (model / permission-mode / gpu-target / fellow / paths) so nothing relies
    # on forge-specific environment variables.
    cmd_parts = [
        forge_bin, "forge-loop",
        "--kernel", str(kernel_file),
        "--driver", str(driver_dest),
        "--workspace", str(workspace),
        "--experiments-dir", str(experiments_dir),
        "--result-json", str(result_json),
        "--snr-threshold", str(agent_config.get("snr_threshold", 30.0)),
        "--max-iters", str(agent_config.get("max_iters", 2)),
        "--max-hours", str(_forge_max_hours(agent_config)),
        "--gpu-target", gpu_arch,
        "--fellow", fellow,
        "--git-branch", "forge-optimize",
        "--program-md-file", str(program_md),
        "--model", model,
        "--permission-mode", permission_mode,
    ]
    # shapes_json is only meaningful for per-kernel drivers that parse --shape.
    # The generic arena_task_adapter ignores --shape (the task's pytest owns its
    # shapes), so we omit it unless a task explicitly configures one — the
    # forge-loop CLI defaults to "{}" (one default-shape sweep).
    shapes_json = agent_config.get("shapes_json")
    if shapes_json:
        cmd_parts += ["--shapes-json", str(shapes_json)]
    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    # Environment: inherit the gateway auth (ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN)
    # from the parent. Only two non-inherited vars remain, both genuinely
    # environmental (not forge config): IS_SANDBOX (required by the claude CLI's
    # bypassPermissions under root) and PYTHONUNBUFFERED (streaming). Everything
    # else is a CLI flag or baked into the generated driver.
    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    # The gateway uses bearer auth; ensure x-api-key auth isn't picked instead.
    env.pop("ANTHROPIC_API_KEY", None)
    shim_dir = _ensure_rtk_shim()
    if shim_dir:
        env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"

    logger.info("Forge Preflight")
    logger.info(f"  forge bin:   {forge_bin}")
    logger.info(f"  kernel:      {kernel_file}")
    logger.info(f"  driver:      {driver_dest}")
    logger.info(f"  gpu target:  {gpu_arch}")
    logger.info(f"  model:       {model}")
    logger.info(f"  fellow:      {fellow} (inferred from task_type={task_config.get('task_type')!r})")
    logger.info(f"  budget:      {agent_config.get('max_iters')} iters / {_forge_max_hours(agent_config)}h")
    logger.info(f"  gateway:     {env.get('ANTHROPIC_BASE_URL', '<unset>')}")
    logger.info(f"Running command: {cmd}")
    logger.info("=" * 80)
    logger.info("Forge Output (streaming):")
    logger.info("=" * 80)

    timeout_seconds = int(agent_config.get("timeout_seconds", 3600))

    process = subprocess.Popen(
        cmd,
        shell=True,  # nosec B602 -- launch the forge-loop subprocess
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        env=env,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stream(stream, sink, prefix, log_func):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                text = line.rstrip()
                if text:
                    sink.append(text)
                    log_func(f"{prefix} {text}")
        finally:
            stream.close()

    threads = [
        threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "[FORGE]", logger.info), daemon=True),
        threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "[FORGE STDERR]", logger.warning), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Forge loop timed out after {timeout_seconds}s; terminating")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing forge loop process")
            process.kill()

    for t in threads:
        t.join(timeout=1)

    logger.info("=" * 80)
    logger.info(f"Forge loop completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    # Restore the workspace working tree to the loop's final (best-kept) state.
    # The loop runs on the 'forge-optimize' branch; ensure no partial/uncommitted
    # revert leaves the tree dirty before Arena re-scores.
    _git(workspace, "checkout", "--", ".", logger=logger)

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)
    return output
