"""
Unit tests for the MUSA task driver.

The driver is what turns a one-configuration problem file into a run over a case
list, and it is the only place that knows how a case maps onto the problem's
module-level defaults. These tests pin that mapping down without a device:

- a wrong `case_parameters` mapping is caught before anything reaches hardware;
- the appended overrides actually re-bind the reference's defaults, verified by
  executing the specialized source against a stub torch;
- the static report says yes for a valid submission and no for a broken one.

Stdlib only.

Run with either:
    python -m unittest musa_operator_eval.tests.test_run_task -v
    python musa_operator_eval/tests/test_run_task.py -v
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent

_spec = importlib.util.spec_from_file_location("run_task", ROOT / "tools" / "run_task.py")
assert _spec is not None and _spec.loader is not None
run_task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_task)

TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"

VALID_SUBMISSION = """
import json
import os

import torch
import torch_musa
import mudnn


class ModelNew(torch.nn.Module):
    def __init__(self, scale, causal, window_left, window_right):
        super(ModelNew, self).__init__()
        self.dispatch = self._probe_runtime_capability()

    def _probe_runtime_capability(self):
        return {"selected_path": "fused_library", "probe_status": "accepted"}

    def forward(self, q, k, v):
        record = dict(self.dispatch)
        record["case_id"] = os.environ["KB_DISPATCH_CASE_ID"]
        with open(os.environ["KB_DISPATCH_TRACE"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")
        return torch_musa.sdpa(q, k, v)
"""


class _StubModule:
    """Stand-in for torch.nn.Module, good enough to be subclassed."""


def _stub_torch_modules():
    """Minimal torch/torch.nn so a problem file can be exec'd without a GPU stack.

    The reference imports torch and annotates with torch.Tensor / torch.device,
    both of which are evaluated at definition time, so the stubs have to cover
    more than the class base.
    """
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = object
    torch_module.device = object
    torch_module.nn = types.ModuleType("torch.nn")
    torch_module.nn.Module = _StubModule
    return {"torch": torch_module, "torch.nn": torch_module.nn}


def _execute(source: str) -> dict:
    namespace: dict = {}
    with mock.patch.dict(sys.modules, _stub_torch_modules()):
        exec(source, namespace)
    return namespace


class LoadTaskTests(unittest.TestCase):
    def test_loads_contract_cases_and_reference(self):
        task, cases, problem_source = run_task.load_task(TASK_DIR)
        self.assertEqual(task["tier"], "B_library")
        self.assertIn("class Model", problem_source)
        self.assertEqual(len(cases["cases"]), 5)
        self.assertEqual(cases["visibility"], "public")

    def test_missing_problem_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary)
            (task_dir / "task.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
            (task_dir / "public_cases.json").write_text(json.dumps({"cases": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_task.load_task(task_dir)


class ResolveCaseFieldTests(unittest.TestCase):
    def test_resolves_a_nested_path(self):
        case = {"case_id": "c", "shape": {"B": 2}, "attributes": {"causal": True}}
        self.assertEqual(run_task.resolve_case_field(case, "shape.B"), 2)
        self.assertIs(run_task.resolve_case_field(case, "attributes.causal"), True)

    def test_missing_field_names_the_case_and_the_path(self):
        with self.assertRaises(KeyError) as raised:
            run_task.resolve_case_field({"case_id": "smoke_009"}, "attributes.scale")
        self.assertIn("smoke_009", str(raised.exception))
        self.assertIn("attributes.scale", str(raised.exception))


class CaseSourceTests(unittest.TestCase):
    def setUp(self):
        self.task, self.cases, self.problem_source = run_task.load_task(TASK_DIR)
        self.case_parameters = self.task["case_parameters"]

    def test_specialized_source_compiles(self):
        for case in self.cases["cases"]:
            with self.subTest(case=case["case_id"]):
                source = run_task.case_source(self.problem_source, case, self.case_parameters)
                compile(source, f"<{case['case_id']}>", "exec")

    def test_overrides_rebind_the_reference_defaults(self):
        """Executing the specialized source must yield the case's configuration."""
        for case in self.cases["cases"]:
            with self.subTest(case=case["case_id"]):
                source = run_task.case_source(self.problem_source, case, self.case_parameters)
                namespace = _execute(source)

                shape = case["shape"]
                self.assertEqual(namespace["batch_size"], shape["B"])
                self.assertEqual(namespace["num_query_heads"], shape["H_q"])
                self.assertEqual(namespace["num_kv_heads"], shape["H_kv"])
                self.assertEqual(namespace["sequence_length_query"], shape["S_q"])
                self.assertEqual(namespace["sequence_length_key"], shape["S_kv"])
                self.assertEqual(namespace["head_dimension"], shape["D"])

                self.assertEqual(
                    namespace["get_init_inputs"](),
                    [
                        case["attributes"]["scale"],
                        case["attributes"]["causal"],
                        case["attributes"]["window_left"],
                        case["attributes"]["window_right"],
                    ],
                )

    def test_the_reference_body_is_left_untouched(self):
        case = self.cases["cases"][0]
        source = run_task.case_source(self.problem_source, case, self.case_parameters)
        self.assertTrue(source.startswith(self.problem_source.rstrip()))
        self.assertIn("class Model", source)

    def test_defaults_without_overrides_are_unchanged(self):
        """The bare problem file must still describe its own default case."""
        namespace = _execute(self.problem_source)
        self.assertEqual(namespace["batch_size"], 1)
        self.assertEqual(namespace["sequence_length_query"], 16)

    def test_an_empty_mapping_leaves_the_source_alone(self):
        case = self.cases["cases"][0]
        source = run_task.case_source(self.problem_source, case, {})
        namespace = _execute(source)
        self.assertEqual(namespace["batch_size"], 1)


class VerifyCaseParametersTests(unittest.TestCase):
    def setUp(self):
        self.task, self.cases, _ = run_task.load_task(TASK_DIR)
        self.cases_list = self.cases["cases"]

    def test_the_shipped_mapping_is_valid(self):
        self.assertEqual(run_task.verify_case_parameters(self.cases_list, self.task["case_parameters"]), [])

    def test_an_unknown_case_field_is_reported(self):
        errors = run_task.verify_case_parameters(self.cases_list, {"attributes.does_not_exist": "x"})
        self.assertTrue(errors)
        self.assertTrue(any("attributes.does_not_exist" in message for message in errors), errors)

    def test_an_invalid_variable_name_is_reported(self):
        errors = run_task.verify_case_parameters(self.cases_list, {"shape.B": "not a name"})
        self.assertTrue(any("not a valid name" in message for message in errors), errors)

    def test_an_empty_mapping_is_reported(self):
        errors = run_task.verify_case_parameters(self.cases_list, {})
        self.assertTrue(any("case_parameters" in message for message in errors), errors)


class StaticReportTests(unittest.TestCase):
    def setUp(self):
        self.task, self.cases, self.problem_source = run_task.load_task(TASK_DIR)

    def test_valid_submission_passes(self):
        report = run_task.static_report(self.task, self.cases["cases"], self.problem_source, VALID_SUBMISSION)
        self.assertTrue(report["summary"]["passed_overall"], report["summary"]["errors"])
        self.assertTrue(report["static_audit"]["passed"])
        self.assertEqual(report["case_sources"]["count"], 5)
        self.assertEqual(report["case_sources"]["errors"], [])
        self.assertEqual(report["tier"], "B_library")

    def test_submission_without_a_trace_fails(self):
        broken = "import torch_musa\n\ndef f(q, k, v):\n    return torch_musa.sdpa(q, k, v)\n"
        report = run_task.static_report(self.task, self.cases["cases"], self.problem_source, broken)
        self.assertFalse(report["summary"]["passed_overall"])
        self.assertTrue(any("dispatch trace" in message for message in report["summary"]["errors"]))

    def test_submission_importing_a_foreign_library_fails(self):
        broken = "import triton\n" + VALID_SUBMISSION
        report = run_task.static_report(self.task, self.cases["cases"], self.problem_source, broken)
        self.assertFalse(report["summary"]["passed_overall"])
        self.assertTrue(any("triton" in message for message in report["summary"]["errors"]))

    def test_a_broken_case_mapping_is_reported_without_a_device(self):
        task = dict(self.task)
        task["case_parameters"] = {"attributes.not_a_field": "x"}
        report = run_task.static_report(task, self.cases["cases"], self.problem_source, VALID_SUBMISSION)
        self.assertFalse(report["summary"]["passed_overall"])
        self.assertTrue(report["case_parameters"]["errors"])


class CliTests(unittest.TestCase):
    def test_static_only_cli_returns_zero_for_a_valid_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "model_new.py"
            submission.write_text(VALID_SUBMISSION, encoding="utf-8")
            report_path = Path(temporary) / "report.json"

            argv = sys.argv
            sys.argv = [
                "run_task.py",
                "--task-dir", str(TASK_DIR),
                "--submission", str(submission),
                "--static-only",
                "--output", str(report_path),
            ]
            try:
                exit_code = run_task.main()
            finally:
                sys.argv = argv

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["summary"]["passed_overall"])

    def test_static_only_cli_returns_one_for_a_broken_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "model_new.py"
            submission.write_text("import triton\n", encoding="utf-8")

            argv = sys.argv
            sys.argv = [
                "run_task.py",
                "--task-dir", str(TASK_DIR),
                "--submission", str(submission),
                "--static-only",
                "--output", str(Path(temporary) / "report.json"),
            ]
            try:
                exit_code = run_task.main()
            finally:
                sys.argv = argv

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
