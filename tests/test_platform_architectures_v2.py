"""Architecture eligibility and runtime identity; CPU fixtures, not GPU evidence."""
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.quality_loop.orchestrator import QualityLoop
from main import should_run_task_for_platform
from src.task_runtime import bind_session_runtime
from src.task_spec import TaskConfigError, TaskSpec


def task(required):
    return TaskSpec.from_mapping({
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": ["kernel.hip"]},
        "evaluation": {"runner": ["python3", "evaluate.py"]},
        "platform_support": {"required_arch": required},
    }, task_id="example/operator")


@pytest.mark.parametrize("required", ["gfx950", ["gfx950"], ["gfx942", "gfx950"]])
@pytest.mark.parametrize("actual", ["gfx942", "gfx950", "gfx1100", None])
def test_scheduling_quality_loop_and_runtime_agree(required, actual, tmp_path, monkeypatch):
    spec = task(required)
    allowed = required if isinstance(required, list) else [required]
    expected = actual in allowed
    assert spec.to_mapping()["platform_support"]["required_arch"] == required
    assert should_run_task_for_platform(spec.task_id, spec.to_mapping(), actual,
                                        logging.getLogger(__name__)) is expected
    assert QualityLoop._platform_matches(spec.to_mapping(), actual) is expected
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": actual})
    session = SimpleNamespace(spec=spec, state_directory=tmp_path)
    if expected:
        assert bind_session_runtime(session)["gpu_arch"] == actual
    else:
        with pytest.raises(RuntimeError, match="Task requires"):
            bind_session_runtime(session)
        assert not (tmp_path / "runtime_identity.json").exists()


@pytest.mark.parametrize("invalid", [None, [], ["gfx950", "gfx950"], ["gfx950", 950],
                                     [""], ["gfx950", ["gfx942"]], True, "gfx9*", "gfx942 gfx950"])
def test_invalid_architecture_declarations_fail_before_execution(invalid):
    with pytest.raises(TaskConfigError, match="required_arch"):
        task(invalid)


def test_allowed_architectures_do_not_allow_reusing_baseline_on_other_gpu(tmp_path, monkeypatch):
    session = SimpleNamespace(spec=task(["gfx942", "gfx950"]), state_directory=tmp_path)
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx942"})
    bind_session_runtime(session)
    recorded = (tmp_path / "runtime_identity.json").read_bytes()
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx950"})
    with pytest.raises(RuntimeError, match="Scoring runtime changed"):
        bind_session_runtime(session)
    assert (tmp_path / "runtime_identity.json").read_bytes() == recorded
    assert json.loads(recorded)["gpu_arch"] == "gfx942"


IMAGE_VARIABLES = ("AKA_SCORING_IMAGE_RUNTIME_REF", "AKA_SCORING_IMAGE_REFERENCE",
                   "AKA_EVAL_TOOLS_SCORING_IMAGE_ID", "AKA_SCORING_IMAGE_ID", "AKA_DOCKER_IMAGE")


def test_runtime_identity_is_what_a_child_process_reports(tmp_path, monkeypatch):
    for name in IMAGE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AKA_SCORING_IMAGE_ID", "image-fixture")
    monkeypatch.setattr("src.task_runtime._PROBE",
                        'import json, os; print(json.dumps({"gpu_arch": "gfx950", "pid": os.getpid()}))')
    runtime = bind_session_runtime(SimpleNamespace(spec=task("gfx950"), state_directory=tmp_path))
    assert runtime["gpu_arch"] == "gfx950" and runtime["AKA_SCORING_IMAGE_ID"] == "image-fixture"
    assert runtime["pid"] != os.getpid()
    assert json.loads((tmp_path / "runtime_identity.json").read_text()) == runtime


def test_a_failed_runtime_query_reports_the_child_process_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.task_runtime._PROBE",
                        'import sys; sys.exit("V2 GPU tasks require an available ROCm GPU in the scoring runtime")')
    with pytest.raises(RuntimeError, match="require an available ROCm GPU"):
        bind_session_runtime(SimpleNamespace(spec=task("gfx950"), state_directory=tmp_path))
    assert not (tmp_path / "runtime_identity.json").exists()


def test_binding_a_session_never_loads_torch_into_the_arena_process(tmp_path):
    """A GPU hang ends every process holding a GPU open, so the arena's own process must not be one of them."""
    script = textwrap.dedent(f"""
        import json, sys
        from pathlib import Path
        from types import SimpleNamespace
        from src import task_runtime
        from src.task_spec import TaskSpec
        task_runtime._PROBE = 'import json; print(json.dumps({{"gpu_arch": "gfx950"}}))'
        spec = TaskSpec.from_mapping({{
            "schema_version": 2,
            "candidate": {{"language": "hip", "editable": ["kernel.hip"]}},
            "evaluation": {{"runner": ["python3", "evaluate.py"]}},
            "platform_support": {{"required_arch": "gfx950"}},
        }}, task_id="example/operator")
        task_runtime.bind_session_runtime(SimpleNamespace(spec=spec, state_directory=Path({str(tmp_path)!r})))
        print(json.dumps(sorted(name for name in sys.modules if name.split(".")[0] == "torch")))
    """)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
                               timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []
    assert (tmp_path / "runtime_identity.json").exists()


def test_explicit_skip_overrides_architecture_match():
    config = task(["gfx942", "gfx950"]).to_mapping()
    config["platform_support"].update(status="skip", skip_reason="Pending task maintenance")
    assert not should_run_task_for_platform("example/operator", config, "gfx950", logging.getLogger(__name__))
    assert not QualityLoop._platform_matches(config, "gfx950")
