# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Invariants for the SIKL same-language FlyDSL optimization tasks.

One task per workload case of the GLM-5.2 MXFP4 MoE definition, each optimizing
the FlyDSL kernels aiter ships, in place. Arena copies every task into its own
workspace, so a task cannot import from a sibling and each one carries the whole
harness; these tests hold the copies identical and the measurement honest.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIKL = ROOT / "tasks/SIKL-task"
GENERATOR = SIKL / "generate_shape_tasks.py"
SHAPES = (8, 16, 32, 64, 128, 256)
TASKS = tuple(SIKL / f"glm52_moe_mxfp4_per1x32_t{n}_flydsl_opt" for n in SHAPES)

# Present in every task and required to be byte-identical: a divergent copy
# would silently score one shape differently from the rest.
SHARED_FILES = (
    "test_kernel_harness.py",
    "scripts/forge_driver.py",
    "scripts/task_inputs.py",
    "scripts/task_reference.py",
    "scripts/task_baseline.py",
)

# aiter's MXFP4 quantizer, used by the correctness reference. An agent that
# could edit it would move the operator and its oracle together.
REFERENCE_QUANTIZER = "aiter/ops/quant.py"


def _config(task: Path) -> dict:
    return yaml.safe_load((task / "config.yaml").read_text())


def _inputs_module(task: Path):
    scripts = str(task / "scripts")
    if scripts in sys.path:
        sys.path.remove(scripts)
    sys.path.insert(0, scripts)
    sys.modules.pop("task_inputs", None)
    import task_inputs

    return task_inputs


def test_every_schema_workload_has_a_task():
    for task in TASKS:
        assert task.is_dir(), f"missing task: {task.name}"


@pytest.mark.parametrize("shared", SHARED_FILES)
def test_the_family_ships_one_implementation(shared):
    digests = {(task / shared).read_bytes() for task in TASKS}
    assert len(digests) == 1, f"{shared} differs across the family"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_workload_matches_the_task_name(task):
    workload = json.loads((task / "workload.json").read_text())
    expected = int(task.name.split("_t")[1].split("_")[0])
    assert workload["num_tokens"] == expected
    assert workload["definition"] == (
        "aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128"
    )
    # A gate without its measured floor and a reason is a number nobody can
    # audit later.
    assert workload["max_relerr"] >= workload["measured"]["baseline_relerr_vs_reference"]
    assert workload["tolerance_reason"].strip()


def test_generated_tasks_are_up_to_date():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_task_ships_a_driver_and_a_harness(task):
    assert (task / "test_kernel_harness.py").is_file()
    assert (task / "scripts" / "forge_driver.py").is_file()
    # The canonical helper is materialized at run time and must never be
    # committed under tasks/.
    assert not (task / "_aka_benchmark.py").exists()
    assert not (task / "scripts" / "_aka_benchmark.py").exists()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_driver_and_harness_measure_the_same_way(task):
    # Timing must be CUDA-graph based and shared: eager timing of this operator
    # is dominated by per-call host dispatch, so a candidate that only avoids
    # dispatch would report a speedup that does not exist in a graph-captured
    # server, and the loop would optimize something the score does not measure.
    harness = (task / "test_kernel_harness.py").read_text()
    driver = (task / "scripts" / "forge_driver.py").read_text()
    for source in (harness, driver):
        assert "benchmark_cuda_graph_or_events" in source
        assert "task_inputs.BENCH_TARGET_MS" in source


def test_tasks_declare_one_workload_contract():
    modules = [_inputs_module(task) for task in TASKS]
    for attribute in (
        "MODEL_DIM", "INTER_DIM", "NUM_EXPERTS", "TOPK", "SEED",
        "ACTIVATION", "DOWEIGHT_STAGE1",
        "BENCH_WARMUP", "BENCH_REPETITION", "BENCH_TARGET_MS",
    ):
        values = {getattr(module, attribute) for module in modules}
        assert len(values) == 1, f"{attribute} differs across the family: {values}"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_task_edits_the_seeded_flydsl_sources_only(task):
    config = _config(task)
    assert config["task_type"] == "image_kernel"
    assert config["image_repo_path"] == "/sgl-workspace/aiter"
    assert config["repo_subdir"] == "aiter"
    assert config["repository_language"] == "flydsl"
    assert config["kernel_identity"]["kernel_kind"] == "flydsl"

    # The forge anchor is passed as --kernel and forge-loop's task statement
    # names it, so it has to hold the compute core, not the dispatcher.
    assert config["source_file_path"][0].endswith("kernels/mixed_moe_gemm_2stage.py")

    scope = [*config["source_file_path"], *config["editable_sources"]]
    assert all(path.startswith("aiter/") for path in scope), scope
    # Same-language task: no HIP sources, and the reference's quantizer stays out.
    assert not [path for path in scope if path.startswith("csrc/")]
    assert not [path for path in scope if path.endswith((".cu", ".cpp", ".h", ".hpp"))]
    assert REFERENCE_QUANTIZER not in [path[len("aiter/"):] for path in scope]


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_task_fails_closed_on_the_in_image_aiter(task):
    # If the import resolves past the workspace copy the agent's edits are
    # invisible and every number the task reports describes the shipped kernel.
    harness = (task / "test_kernel_harness.py").read_text()
    driver = (task / "scripts" / "forge_driver.py").read_text()
    for source in (harness, driver):
        assert "task_inputs.use_workspace_aiter()" in source
        assert "assert_aiter_is_workspace_copy" in source
        # The shadowing has to be installed before anything imports aiter.
        assert source.index("use_workspace_aiter()") < source.index("import torch")
