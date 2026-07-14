# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from src.evaluator import evaluate_correctness, evaluate_kernel
from src.evaluator_utils import (
    find_unimplemented_target_stubs,
    inspect_target_definitions,
)


class EvaluatorStubGuardTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temporary_directory.name)
        self.config = {
            "task_type": "torch2flydsl",
            "source_file_path": ["kernel.py"],
            "target_kernel_functions": ["target", "builder"],
            "compile_command": ["compile"],
            "correctness_command": ["correctness"],
        }

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write_kernel(self, source):
        (self.workspace / "kernel.py").write_text(
            textwrap.dedent(source), encoding="utf-8"
        )

    def test_finds_only_declared_top_level_unconditional_stubs(self):
        self.write_kernel(
            '''
            def target():
                """Starter target."""
                pass
                raise NotImplementedError("implement me")

            def builder(value):
                if value is None:
                    raise NotImplementedError("unsupported configuration")
                return value

            def unrelated():
                raise NotImplementedError
            '''
        )

        self.assertEqual(
            find_unimplemented_target_stubs(self.workspace, self.config),
            ["target"],
        )

    def test_ignores_conditional_nested_and_non_notimplemented_raises(self):
        self.write_kernel(
            '''
            def target(value):
                def nested():
                    raise NotImplementedError("nested")
                if value < 0:
                    raise NotImplementedError("unsupported shape")
                return value

            def builder():
                raise RuntimeError("broken, but not a recognized starter")
            '''
        )

        self.assertEqual(
            find_unimplemented_target_stubs(self.workspace, self.config), []
        )

    def test_last_top_level_definition_is_effective(self):
        self.write_kernel(
            '''
            def target():
                raise NotImplementedError("old definition")

            def target():
                return "implemented"

            def builder():
                return "implemented"
            '''
        )

        self.assertEqual(
            find_unimplemented_target_stubs(self.workspace, self.config), []
        )

    @mock.patch("src.evaluator.measure_performance")
    @mock.patch("src.evaluator.evaluate_correctness")
    @mock.patch("src.evaluator.evaluate_compilation", return_value=(True, None))
    def test_missing_declared_target_fails_before_correctness_command(
        self, _compilation, correctness, performance
    ):
        self.write_kernel(
            '''
            def target():
                return "implemented"

            builder = target

            def container():
                def builder():
                    return "nested is not a declared top-level target"
                return builder
            '''
        )

        result = evaluate_kernel(self.workspace, self.config, [], logger=mock.Mock())

        self.assertTrue(result["pass_compilation"])
        self.assertFalse(result["pass_correctness"])
        self.assertIn("missing declared top-level target", result["correctness_error_message"])
        self.assertIn("builder", result["correctness_error_message"])
        correctness.assert_not_called()
        performance.assert_not_called()

    @mock.patch("src.evaluator.measure_performance", return_value=[])
    @mock.patch("src.evaluator.evaluate_correctness", return_value=(True, None))
    @mock.patch("src.evaluator.evaluate_compilation", return_value=(True, None))
    def test_implemented_targets_across_multiple_sources_are_not_rejected(
        self, _compilation, correctness, _performance
    ):
        self.write_kernel(
            '''
            def target():
                return "implemented"
            '''
        )
        (self.workspace / "builders.py").write_text(
            "def builder():\n    return 'implemented'\n", encoding="utf-8"
        )
        self.config["source_file_path"] = ["kernel.py", "builders.py"]

        self.assertEqual(
            inspect_target_definitions(self.workspace, self.config), ([], [])
        )
        result = evaluate_kernel(self.workspace, self.config, [], logger=mock.Mock())

        self.assertTrue(result["pass_correctness"])
        correctness.assert_called_once()

    @mock.patch("src.evaluator.measure_performance")
    @mock.patch("src.evaluator.evaluate_correctness")
    @mock.patch("src.evaluator.evaluate_compilation", return_value=(True, None))
    def test_optimized_torch2flydsl_submission_fails_before_correctness_command(
        self, _compilation, correctness, performance
    ):
        self.write_kernel(
            '''
            def target():
                raise NotImplementedError("implement me")

            def builder():
                return object()
            '''
        )

        result = evaluate_kernel(self.workspace, self.config, [], logger=mock.Mock())

        self.assertTrue(result["pass_compilation"])
        self.assertFalse(result["pass_correctness"])
        self.assertIn("target", result["correctness_error_message"])
        self.assertIn("unimplemented", result["correctness_error_message"])
        correctness.assert_not_called()
        performance.assert_not_called()

    @mock.patch("src.evaluator.run_command", return_value=(True, "correctness: pass", ""))
    def test_standalone_correctness_command_is_not_the_optimization_guard(self, run_command):
        self.write_kernel(
            '''
            def target():
                raise NotImplementedError("starter")

            def builder():
                raise NotImplementedError("starter")
            '''
        )

        passed, error = evaluate_correctness(self.workspace, self.config)

        self.assertTrue(passed)
        self.assertIsNone(error)
        run_command.assert_called_once()

    @mock.patch("src.evaluator.measure_performance", return_value=[])
    @mock.patch("src.evaluator.evaluate_correctness", return_value=(True, None))
    @mock.patch("src.evaluator.evaluate_compilation", return_value=(True, None))
    def test_conditional_notimplemented_reaches_optimized_correctness(
        self, _compilation, correctness, _performance
    ):
        self.write_kernel(
            '''
            def target(value):
                if value is None:
                    raise NotImplementedError("unsupported configuration")
                return value

            def builder():
                return object()
            '''
        )

        result = evaluate_kernel(self.workspace, self.config, [], logger=mock.Mock())

        self.assertTrue(result["pass_correctness"])
        correctness.assert_called_once()

    @mock.patch("src.evaluator.measure_performance", return_value=[])
    @mock.patch("src.evaluator.evaluate_correctness", return_value=(True, None))
    @mock.patch("src.evaluator.evaluate_compilation", return_value=(True, None))
    def test_other_task_types_are_unchanged(
        self, _compilation, correctness, _performance
    ):
        self.write_kernel(
            '''
            def target():
                raise NotImplementedError("starter")

            def builder():
                raise NotImplementedError("starter")
            '''
        )
        self.config["task_type"] = "flydsl2flydsl"

        result = evaluate_kernel(self.workspace, self.config, [], logger=mock.Mock())

        self.assertTrue(result["pass_correctness"])
        correctness.assert_called_once()


if __name__ == "__main__":
    unittest.main()
