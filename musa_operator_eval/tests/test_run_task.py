"""
Unit tests for the MUSA task driver.

The driver is what turns a one-configuration problem file into a run over a case
list, and it is the only place that knows how a case maps onto the problem's
module-level defaults. These tests pin that mapping down without a device:

- a wrong `case_parameters` mapping is caught before anything reaches hardware;
- the appended overrides actually re-bind the reference's defaults, verified by
  executing the specialized source against a stub torch;
- the static report says yes for a valid submission and no for a broken one;
- a failing stage short-circuits and returns the named code for that stage;
- a case's `golden` field is actually cross-checked, and the ways it can fail
  (absent, corrupt, contradicted by a recomputation) are told apart.

Stdlib only; the golden tests use the real generator, which needs numpy.

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


def _load_tool(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_task = _load_tool("run_task", ROOT / "tools" / "run_task.py")
exit_codes = _load_tool("exit_codes", ROOT / "tools" / "exit_codes.py")

TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"
GENERATE_CASES_PATH = ROOT / "tools" / "generate_cases.py"

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

    def test_an_explicit_case_manifest_replaces_the_public_one(self):
        """The hidden set lives outside the task package, so it must be passable."""
        hidden = {"visibility": "hidden", "cases": [{"case_id": "hidden_gap_001"}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.private.json"
            path.write_text(json.dumps(hidden), encoding="utf-8")
            _, cases, _ = run_task.load_task(TASK_DIR, path)
        self.assertEqual(cases["visibility"], "hidden")
        self.assertEqual([case["case_id"] for case in cases["cases"]], ["hidden_gap_001"])

    def test_an_explicit_manifest_does_not_need_to_live_in_the_task_dir(self):
        """A private manifest must be readable without being copied into the package."""
        hidden = {"cases": [{"case_id": "c1"}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "anywhere.json"
            path.write_text(json.dumps(hidden), encoding="utf-8")
            self.assertFalse(str(path).startswith(str(TASK_DIR)))
            _, cases, _ = run_task.load_task(TASK_DIR, path)
        self.assertEqual(len(cases["cases"]), 1)


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


class RunCaseWiringTests(unittest.TestCase):
    """What run_case actually asks the evaluator for.

    `eval_kernel_against_ref` defaults `measure_performance` to False, and when
    it is left there the evaluator returns runtime -1.0 regardless of
    num_perf_trials. A report full of passing cases and no latency is a silent
    failure, so the wiring is pinned rather than assumed.
    """

    def _drive(self, num_perf_trials):
        task, cases, problem_source = run_task.load_task(TASK_DIR)
        captured = {}

        class _Result:
            compiled = True
            correctness = True
            dispatch_trace_passed = True
            runtime = 12.5
            ref_runtime = 25.0
            metadata = {"tier": "B_library"}

        def _record(*args, **kwargs):
            captured.update(kwargs)
            return _Result()

        fake_eval = types.ModuleType("kernelbench.eval")
        fake_eval.eval_library_dispatch_against_ref = _record
        fake_eval.get_torch_dtype_from_string = lambda name: name
        fake_package = types.ModuleType("kernelbench")

        with mock.patch.dict(sys.modules, {"kernelbench": fake_package, "kernelbench.eval": fake_eval}):
            row = run_task.run_case(
                task, cases["cases"][0], problem_source, VALID_SUBMISSION,
                backend="musa", precision="fp16",
                num_correct_trials=1, num_perf_trials=num_perf_trials, verbose=False,
            )
        return captured, row

    def test_performance_measurement_is_requested(self):
        captured, _ = self._drive(num_perf_trials=25)
        self.assertTrue(captured.get("measure_performance"), "timing must be switched on explicitly")
        self.assertEqual(captured["num_perf_trials"], 25)

    def test_zero_perf_trials_disables_measurement(self):
        captured, _ = self._drive(num_perf_trials=0)
        self.assertFalse(captured.get("measure_performance"))

    def test_case_identity_and_expected_path_are_forwarded(self):
        captured, _ = self._drive(num_perf_trials=5)
        self.assertEqual(captured["dispatch_case_id"], "smoke_001")
        self.assertEqual(captured["expected_dispatch_path"], "fused_library")

    def test_timings_reach_the_report(self):
        _, row = self._drive(num_perf_trials=5)
        self.assertEqual(row["runtime"], 12.5)
        self.assertEqual(row["ref_runtime"], 25.0)
        self.assertTrue(row["passed"])


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

    def test_static_only_cli_returns_the_forbidden_api_code_for_a_broken_submission(self):
        """A bare 1 would not say *why*; the code names the failing audit category."""
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

            self.assertEqual(exit_code, exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API)

    def test_missing_submission_is_an_input_failure_not_a_traceback(self):
        argv = sys.argv
        sys.argv = [
            "run_task.py",
            "--task-dir", str(TASK_DIR),
            "--submission", "/nonexistent/model_new.py",
            "--static-only",
        ]
        try:
            exit_code = run_task.main()
        finally:
            sys.argv = argv
        self.assertEqual(exit_code, exit_codes.EXIT_INPUT_UNAVAILABLE)

    def test_the_report_carries_the_code_and_its_meaning(self):
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "model_new.py"
            submission.write_text("import triton\n", encoding="utf-8")
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
                run_task.main()
            finally:
                sys.argv = argv

            summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
            self.assertEqual(summary["exit_code"], exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API)
            self.assertFalse(summary["passed_overall"])
            self.assertIn("forbids", summary["exit_code_description"])


class ExitCodeTests(unittest.TestCase):
    def test_every_named_code_is_distinct(self):
        """Two failure categories sharing a number would defeat the whole table."""
        named = {
            name: value
            for name, value in vars(exit_codes).items()
            if name.startswith("EXIT_") and not name.startswith("EXIT_CODE_") and isinstance(value, int)
        }
        self.assertEqual(len(set(named.values())), len(named), f"duplicate exit code values: {named}")

    def test_codes_fit_a_process_status_and_reserve_zero_for_success(self):
        for name, value in vars(exit_codes).items():
            if name.startswith("EXIT_") and name != "EXIT_CODE_DESCRIPTIONS" and isinstance(value, int):
                self.assertLess(value, 128, name)
                self.assertGreaterEqual(value, 0, name)
        self.assertEqual(exit_codes.EXIT_OK, 0)

    def test_every_code_has_a_description(self):
        for name, value in vars(exit_codes).items():
            if name.startswith("EXIT_") and name != "EXIT_CODE_DESCRIPTIONS" and isinstance(value, int):
                self.assertIn(value, exit_codes.EXIT_CODE_DESCRIPTIONS, name)

    def test_describe_says_so_for_an_unknown_code(self):
        self.assertIn("unknown", exit_codes.describe(123))

    def test_static_audit_findings_map_onto_named_categories(self):
        cases = [
            ("Imports non-whitelisted library: triton", exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API),
            ("Uses torch computation op: torch.matmul", exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API),
            ("Calls a fused attention entry point: scaled_dot_product_attention",
             exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API),
            ("Dispatches on a version string: __version__", exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API),
            ("Missing __global__ kernel definition", exit_codes.EXIT_STATIC_AUDIT_DEVICE_SOURCE),
            ("Missing load_inline, cpp_extension, or MUSAExtension for compilation",
             exit_codes.EXIT_STATIC_AUDIT_DEVICE_SOURCE),
            ("dispatch trace is missing required fields: selected_path", exit_codes.EXIT_DISPATCH_TRACE_MISMATCH),
            ("Computes the product in host-side numpy", exit_codes.EXIT_STATIC_AUDIT_HOST_COMPUTATION),
            ("Answers from a lookup table of logits", exit_codes.EXIT_STATIC_AUDIT_TABLE_LOOKUP),
            ("Edited a file the starter manifest freezes", exit_codes.EXIT_STATIC_AUDIT_OUT_OF_SCOPE_MODIFICATION),
            ("something nobody classified", exit_codes.EXIT_STATIC_AUDIT_UNCLASSIFIED),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(run_task.classify_static_audit_errors([message]), expected)

    def test_the_first_classified_finding_decides(self):
        code = run_task.classify_static_audit_errors([
            "a finding nobody classified",
            "Imports non-whitelisted library: triton",
        ])
        self.assertEqual(code, exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API)


class ShortCircuitTests(unittest.TestCase):
    """A failed stage must not be followed by the next one (§4.7)."""

    def setUp(self):
        self.task, self.cases, self.problem_source = run_task.load_task(TASK_DIR)
        self.cases_list = self.cases["cases"]

    def test_a_broken_case_mapping_stops_before_the_audit(self):
        """The audit is not run at all, rather than run and aggregated."""
        task = dict(self.task)
        task["case_parameters"] = {"attributes.not_a_field": "x"}
        report = run_task.static_report(task, self.cases_list, self.problem_source, VALID_SUBMISSION)
        self.assertFalse(report["static_audit"]["ran"])
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_CASE_GENERATION_FAILED)
        self.assertNotIn("static_audit", report["stages"])

    def test_a_failed_audit_stops_after_the_audit(self):
        report = run_task.static_report(
            self.task, self.cases_list, self.problem_source, "import triton\n"
        )
        self.assertTrue(report["static_audit"]["ran"])
        self.assertIn("static_audit", report["stages"])
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API)

    def test_a_missing_golden_stops_the_static_pass_before_the_audit(self):
        """A declared golden that was never generated must not read as a pass.

        The static pass is where this is easiest to get wrong: no device is
        involved, so nothing forces the golden to exist, and a report that says
        "passed" while the cross-check never happened is the illusion §4.3's
        field exists to prevent.
        """
        case = dict(self.cases_list[0])
        case["golden"] = "smoke_001/golden/tensors.json"
        with tempfile.TemporaryDirectory() as temporary:
            report = run_task.static_report(
                self.task, [case], self.problem_source, VALID_SUBMISSION,
                generated_dir=Path(temporary),
            )
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_GOLDEN_MISSING)
        self.assertFalse(report["summary"]["passed_overall"])
        self.assertEqual(report["golden"]["counts"], {"missing": 1})
        self.assertFalse(report["static_audit"]["ran"])
        self.assertNotIn("static_audit", report["stages"])

    def test_a_static_pass_without_a_generated_dir_says_so(self):
        report = run_task.static_report(
            self.task, self.cases_list, self.problem_source, VALID_SUBMISSION
        )
        self.assertFalse(report["golden"]["ran"])
        self.assertTrue(report["golden"]["reason"])
        self.assertNotIn("golden", report["stages"])

    def _stub_rows(self, outcomes):
        """Return a run_case stand-in driven by case id -> row overrides."""
        def _run(task, case, *args, **kwargs):
            row = {
                "case_id": case["case_id"],
                "expected_path": case.get("expected_path"),
                "compiled": True,
                "correctness": True,
                "dispatch_trace_passed": True,
                "runtime": 1.0,
                "ref_runtime": 2.0,
                "static_audit_errors": [],
                "dispatch_trace_errors": [],
                "passed": True,
                "errors": [],
            }
            row.update(outcomes.get(case["case_id"], {}))
            if "correctness" in outcomes.get(case["case_id"], {}):
                row["passed"] = (
                    row["correctness"]
                    and row["dispatch_trace_passed"] is not False
                    and not row["errors"]
                )
            return row
        return _run

    def test_a_failed_stage_skips_every_later_stage(self):
        cases = [
            {"case_id": "c1", "tag": "correctness"},
            {"case_id": "b1", "tag": "boundary"},
            {"case_id": "p1", "tag": "perf"},
        ]
        outcomes = {"c1": {"correctness": False}}
        with mock.patch.object(run_task, "run_case", self._stub_rows(outcomes)):
            rows, exit_code = run_task.evaluate_cases(
                self.task, cases, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False,
            )
        self.assertEqual(exit_code, exit_codes.EXIT_CORRECTNESS_FAILED)
        by_id = {row["case_id"]: row for row in rows}
        self.assertFalse(by_id["c1"].get("skipped", False))
        self.assertTrue(by_id["b1"]["skipped"])
        self.assertTrue(by_id["p1"]["skipped"])
        self.assertIn("correctness stage failed", by_id["p1"]["reason"])

    def test_every_case_within_a_failed_stage_still_runs(self):
        """Short-circuit is per stage: one wrong answer must not hide the others."""
        cases = [
            {"case_id": "c1", "tag": "correctness"},
            {"case_id": "c2", "tag": "correctness"},
            {"case_id": "p1", "tag": "perf"},
        ]
        outcomes = {"c1": {"correctness": False}, "c2": {"correctness": False}}
        with mock.patch.object(run_task, "run_case", self._stub_rows(outcomes)):
            rows, exit_code = run_task.evaluate_cases(
                self.task, cases, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False,
            )
        self.assertEqual(exit_code, exit_codes.EXIT_CORRECTNESS_FAILED)
        by_id = {row["case_id"]: row for row in rows}
        self.assertFalse(by_id["c2"].get("skipped", False))
        self.assertTrue(by_id["p1"]["skipped"])

    def test_a_boundary_failure_only_skips_stages_after_it(self):
        cases = [
            {"case_id": "c1", "tag": "correctness"},
            {"case_id": "b1", "tag": "non_aligned"},
            {"case_id": "s1", "tag": "stability"},
        ]
        outcomes = {"b1": {"correctness": False}}
        with mock.patch.object(run_task, "run_case", self._stub_rows(outcomes)):
            rows, exit_code = run_task.evaluate_cases(
                self.task, cases, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False,
            )
        self.assertEqual(exit_code, exit_codes.EXIT_BOUNDARY_FAILED)
        by_id = {row["case_id"]: row for row in rows}
        self.assertFalse(by_id["c1"].get("skipped", False))
        self.assertFalse(by_id["b1"].get("skipped", False))
        self.assertTrue(by_id["s1"]["skipped"])

    def test_a_perf_case_without_a_measurement_reports_the_perf_code(self):
        cases = [{"case_id": "p1", "tag": "perf"}]
        outcomes = {"p1": {"runtime": -1.0, "passed": False}}
        with mock.patch.object(run_task, "run_case", self._stub_rows(outcomes)):
            _, exit_code = run_task.evaluate_cases(
                self.task, cases, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False,
            )
        self.assertEqual(exit_code, exit_codes.EXIT_PERFORMANCE_FAILED)

    def test_the_full_pipeline_stops_at_the_audit(self):
        """evaluate_task must not reach the case loop once the audit rejects."""
        cases_manifest = {"cases": self.cases_list}
        with mock.patch.object(run_task, "golden_stage", return_value={
            "results": [], "counts": {}, "fatal": False, "exit_code": exit_codes.EXIT_OK,
        }), mock.patch.object(run_task, "run_static_audit", return_value=(
            False, ["Calls a fused attention entry point: scaled_dot_product_attention"], [],
        )), mock.patch.object(run_task, "run_case", side_effect=AssertionError("must not run")):
            report = run_task.evaluate_task(
                self.task, cases_manifest, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False, Path("/tmp/unused"),
            )
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API)
        self.assertNotIn("evaluation", report["stages"])

    def test_the_full_pipeline_stops_at_a_fatal_golden_stage(self):
        cases_manifest = {"cases": self.cases_list}
        with mock.patch.object(run_task, "golden_stage", return_value={
            "results": [], "counts": {"missing": 5},
            "fatal": True, "exit_code": exit_codes.EXIT_GOLDEN_MISSING,
        }), mock.patch.object(run_task, "run_static_audit", side_effect=AssertionError("must not run")):
            report = run_task.evaluate_task(
                self.task, cases_manifest, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False, Path("/tmp/unused"),
            )
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_GOLDEN_MISSING)
        self.assertNotIn("static_audit", report["stages"])

    def test_a_missing_device_stack_reports_the_environment_code(self):
        """No torch is a machine fact, not a submission verdict."""
        cases_manifest = {"cases": self.cases_list}
        with mock.patch.object(run_task, "golden_stage", return_value={
            "results": [], "counts": {}, "fatal": False, "exit_code": exit_codes.EXIT_OK,
        }), mock.patch.object(run_task, "run_static_audit", return_value=(True, [], [])), \
                mock.patch.object(run_task, "run_case", side_effect=ImportError("No module named 'torch'")):
            report = run_task.evaluate_task(
                self.task, cases_manifest, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False, Path("/tmp/unused"),
            )
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_ENVIRONMENT_UNAVAILABLE)
        self.assertIn("torch", report["environment"]["error"])

    def test_a_case_row_carries_the_golden_status_it_was_given(self):
        cases = [{"case_id": "c1", "tag": "correctness"}]
        golden_results = {"c1": {"status": "verified", "verified": True}}
        with mock.patch.object(run_task, "run_case", self._stub_rows({})):
            rows, _ = run_task.evaluate_cases(
                self.task, cases, self.problem_source, VALID_SUBMISSION,
                "musa", "fp16", 1, 5, False, golden_results,
            )
        self.assertEqual(rows[0]["golden"], {"status": "verified", "verified": True})

    def test_the_full_pipeline_stops_at_a_bad_precision(self):
        cases_manifest = {"cases": self.cases_list}
        report = run_task.evaluate_task(
            self.task, cases_manifest, self.problem_source, VALID_SUBMISSION,
            "musa", "fp32", 1, 5, False, Path("/tmp/unused"),
        )
        self.assertEqual(report["summary"]["exit_code"], exit_codes.EXIT_PRECISION_CONTRACT_MISMATCH)
        self.assertEqual(report["stages"], [])


class GoldenTests(unittest.TestCase):
    """The `golden` field must be consumed, and honestly reported.

    `make_private_manifest.py` has always written the field; nothing read it.
    These tests use the real generator so the cross-check is exercised against
    the layout it actually produces, not a hand-written stand-in.
    """

    CASE = {
        "case_id": "smoke_001",
        "tag": "smoke",
        "seed": 1101,
        "dtype": "float16",
        "shape": {"B": 1, "H_q": 4, "H_kv": 4, "S_q": 16, "S_kv": 16, "D": 32},
        "attributes": {"scale": 0.1767766952966369, "causal": False, "window_left": -1, "window_right": -1},
        "distribution": "normal",
    }

    def setUp(self):
        self.generator = _load_tool("musa_generate_cases", GENERATE_CASES_PATH)
        self.case = dict(self.CASE)
        self.case["golden"] = "smoke_001/golden/tensors.json"
        self.temporary = tempfile.TemporaryDirectory()
        self.generated_root = Path(self.temporary.name)
        self.generator.generate_case(self.case, self.generated_root)

    def tearDown(self):
        self.temporary.cleanup()

    def _verify(self, case, tool=GENERATE_CASES_PATH):
        return run_task.verify_golden(case, self.generated_root, Path(tool))

    def test_the_generator_and_the_manifest_agree_on_the_layout(self):
        """The path the manifest names is the path the generator writes."""
        self.assertTrue((self.generated_root / self.case["golden"]).is_file())

    def test_a_fresh_recomputation_reproduces_the_stored_golden(self):
        result = self._verify(self.case)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["verified"])
        self.assertEqual(sorted(result["tensors"]), ["lse", "output"])

    def test_a_corrupted_blob_is_unusable_rather_than_verified(self):
        blob = self.generated_root / "smoke_001" / "golden" / "output.bin"
        data = bytearray(blob.read_bytes())
        data[0] ^= 0xFF
        blob.write_bytes(bytes(data))
        result = self._verify(self.case)
        self.assertEqual(result["status"], "unusable")
        self.assertFalse(result["verified"])
        self.assertTrue(any("sha256" in message for message in result["errors"]))

    def test_a_golden_from_a_different_seed_is_a_mismatch(self):
        drifted = dict(self.case)
        drifted["seed"] = self.case["seed"] + 1
        result = self._verify(drifted)
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["verified"])
        self.assertTrue(result["errors"])

    def test_an_absent_golden_is_reported_as_missing(self):
        absent = dict(self.case)
        absent["golden"] = "smoke_001/golden/not_here.json"
        result = self._verify(absent)
        self.assertEqual(result["status"], "missing")
        self.assertFalse(result["verified"])
        self.assertIn("never generated", result["detail"])

    def test_a_case_without_the_field_says_so_instead_of_passing(self):
        result = self._verify({"case_id": "smoke_001"})
        self.assertEqual(result["status"], "not_declared")
        self.assertFalse(result["verified"])

    def test_without_a_generator_the_golden_is_only_checked_for_integrity(self):
        result = run_task.verify_golden(self.case, self.generated_root, None)
        self.assertEqual(result["status"], "integrity_only")
        self.assertFalse(result["verified"], "an unchecked cross-check is not a verified one")
        self.assertIn("not against a recomputation", result["detail"])

    def test_a_manifest_without_a_tool_stays_integrity_only(self):
        result = run_task.verify_golden(self.case, self.generated_root, run_task.resolve_generation_tool({}))
        self.assertEqual(result["status"], "integrity_only")

    def test_the_golden_stage_counts_every_status_and_flags_the_fatal_ones(self):
        absent = dict(self.case)
        absent["case_id"] = "smoke_002"
        absent["golden"] = "smoke_002/golden/tensors.json"
        stage = run_task.golden_stage(
            [self.case, absent],
            self.generated_root,
            {"tool": "musa_operator_eval/tools/generate_cases.py"},
        )
        self.assertEqual(stage["counts"], {"verified": 1, "missing": 1})
        self.assertTrue(stage["fatal"])
        self.assertEqual(stage["exit_code"], exit_codes.EXIT_GOLDEN_MISSING)

    def test_the_golden_stage_is_not_fatal_when_every_declared_golden_verifies(self):
        stage = run_task.golden_stage(
            [self.case],
            self.generated_root,
            {"tool": "musa_operator_eval/tools/generate_cases.py"},
        )
        self.assertFalse(stage["fatal"])
        self.assertEqual(stage["exit_code"], exit_codes.EXIT_OK)

    def test_an_unparsable_golden_is_unusable(self):
        (self.generated_root / "smoke_001" / "golden" / "tensors.json").write_text("{not json", encoding="utf-8")
        result = self._verify(self.case)
        self.assertEqual(result["status"], "unusable")
        self.assertFalse(result["verified"])


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
