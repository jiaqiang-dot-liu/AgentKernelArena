"""Record and compare the actual runtime used for a frozen v2 baseline."""
import json
import os
import subprocess
import sys
from src.runtime_env import build_subprocess_env
from src.task_spec import required_gpu_arches

# Queried in a child process: a GPU hang ends every process on the machine that holds a GPU open, and the arena's
# own process has to outlive a candidate that hangs one.
_PROBE = """\
import importlib.metadata
import json
import sys

import torch

if not torch.cuda.is_available() or not torch.version.hip:
    sys.exit("V2 GPU tasks require an available ROCm GPU in the scoring runtime")
properties = torch.cuda.get_device_properties(0)
identity = {"python": sys.version.split()[0], "torch": str(torch.__version__), "hip": torch.version.hip,
            "gpu_name": properties.name, "gpu_arch": properties.gcnArchName.split(":", 1)[0]}
for name in ("triton", "flydsl", "aiter"):
    try:
        identity[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        identity[name] = None
print(json.dumps(identity))
"""
_PROBE_TIMEOUT_SEC = 300


def _runtime_identity() -> dict:
    try:
        completed = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                                   env=build_subprocess_env(), timeout=_PROBE_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Scoring runtime query did not finish within {_PROBE_TIMEOUT_SEC}s") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"Scoring runtime query failed: {detail or f'exit status {completed.returncode}'}")
    identity = json.loads(completed.stdout.strip().splitlines()[-1])
    for name in ("AKA_SCORING_IMAGE_RUNTIME_REF", "AKA_SCORING_IMAGE_REFERENCE",
                 "AKA_EVAL_TOOLS_SCORING_IMAGE_ID", "AKA_SCORING_IMAGE_ID", "AKA_DOCKER_IMAGE"):
        if name in os.environ:
            identity[name] = os.environ[name]
    return identity


def bind_session_runtime(session) -> dict:
    runtime = _runtime_identity()
    required_arch = required_gpu_arches(session.spec.to_mapping().get("platform_support"))
    if required_arch and runtime["gpu_arch"] not in required_arch:
        raise RuntimeError(f"Task requires {required_arch}; actual GPU is {runtime['gpu_arch']}")
    path = session.state_directory / "runtime_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != runtime:
            raise RuntimeError("Scoring runtime changed since baseline capture; start a separate run")
    else:
        with path.open("x") as handle:
            json.dump(runtime, handle, indent=2, sort_keys=True, allow_nan=False)
    return runtime
