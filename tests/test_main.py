import logging
import tempfile
import unittest
from pathlib import Path

from main import run_task, should_run_task_for_platform
from src.module_registration import AgentType


class TaskValidatorWorkspaceTests(unittest.TestCase):
    def test_validator_does_not_treat_copied_report_as_current_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task_dir = root / "task"
            run_directory = root / "run"
            task_dir.mkdir()
            run_directory.mkdir()
            config_path = task_dir / "config.yaml"
            config_path.write_text("task_type: flydsl2flydsl\n")
            (task_dir / "validation_report.yaml").write_text("overall_status: WARN\n")

            launcher_called = False

            def launcher(*, eval_config, task_config_dir, workspace):
                nonlocal launcher_called
                launcher_called = True
                report_path = Path(workspace) / "validation_report.yaml"
                self.assertFalse(report_path.exists())
                report_path.write_text("overall_status: PASS\n")

            completed, workspace = run_task(
                eval_config={},
                agent=AgentType.TASK_VALIDATOR,
                agent_launcher=launcher,
                task_name="flydsl2flydsl/example",
                task_config_dir=str(config_path),
                run_directory=run_directory,
                timestamp="20260721_000000",
                logger=logging.getLogger(__name__),
                task_index=1,
                total_tasks=1,
            )

            self.assertTrue(launcher_called)
            self.assertTrue(completed)
            self.assertIsNotNone(workspace)
            self.assertEqual(
                (workspace / "validation_report.yaml").read_text(),
                "overall_status: PASS\n",
            )


class PlatformSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.platform")
        self.logger.disabled = True

    def test_status_skip_prevents_task_from_running(self) -> None:
        self.assertFalse(
            should_run_task_for_platform(
                "example",
                {"platform_support": {"status": "skip"}},
                "gfx950",
                self.logger,
            )
        )

    def test_required_arch_must_match_exactly(self) -> None:
        config = {
            "platform_support": {
                "status": "active",
                "required_arch": "gfx942",
            }
        }
        self.assertTrue(
            should_run_task_for_platform("example", config, "gfx942", self.logger)
        )
        self.assertFalse(
            should_run_task_for_platform("example", config, "gfx950", self.logger)
        )

    def test_active_without_required_arch_runs_on_current_arch(self) -> None:
        self.assertTrue(
            should_run_task_for_platform(
                "example",
                {"platform_support": {"status": "active"}},
                "gfx950",
                self.logger,
            )
        )


if __name__ == "__main__":
    unittest.main()
