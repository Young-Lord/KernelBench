"""
Unit tests for the B-tier dispatch trace plumbing.

The trace is what makes the B tier gradeable: without it, a submission that
quietly falls back to the slow composition path looks identical to one that hit
the fused library. These tests cover the three places it can go wrong — the
transport (env var and file), the parser, and the contract check.

Stdlib only, so this runs anywhere.

Run with either:
    python -m unittest src.kernelbench.unit_tests.test_dispatch_trace -v
    python src/kernelbench/unit_tests/test_dispatch_trace.py -v
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TRACE_PATH = Path(__file__).resolve().parents[1] / "dispatch_trace.py"
_spec = importlib.util.spec_from_file_location("dispatch_trace", _TRACE_PATH)
assert _spec is not None and _spec.loader is not None
_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trace)

CASE_ID_ENV_VAR = _trace.CASE_ID_ENV_VAR
DispatchTraceError = _trace.DispatchTraceError
DISPATCH_PATHS = _trace.DISPATCH_PATHS
REQUIRED_TRACE_FIELDS = _trace.REQUIRED_TRACE_FIELDS
TRACE_ENV_VAR = _trace.TRACE_ENV_VAR
check_trace = _trace.check_trace
current_case_id = _trace.current_case_id
dispatch_trace = _trace.dispatch_trace
read_trace = _trace.read_trace
summarize_trace = _trace.summarize_trace


def _record(case_id="smoke_001", selected_path="fused_library", probe_status="accepted"):
    return {"case_id": case_id, "selected_path": selected_path, "probe_status": probe_status}


class TransportTests(unittest.TestCase):
    """The env var is the only thing tying the submission to the evaluator."""

    def test_env_var_is_set_inside_and_restored_after(self):
        os.environ.pop(TRACE_ENV_VAR, None)
        with dispatch_trace() as path:
            self.assertEqual(os.environ[TRACE_ENV_VAR], str(path))
            self.assertTrue(Path(path).is_file())
        self.assertNotIn(TRACE_ENV_VAR, os.environ, "the env var must not leak past the block")

    def test_a_previous_value_is_restored(self):
        os.environ[TRACE_ENV_VAR] = "/tmp/previous-trace"
        try:
            with dispatch_trace() as path:
                self.assertNotEqual(os.environ[TRACE_ENV_VAR], "/tmp/previous-trace")
            self.assertEqual(os.environ[TRACE_ENV_VAR], "/tmp/previous-trace")
        finally:
            os.environ.pop(TRACE_ENV_VAR, None)

    def test_env_var_is_restored_even_when_the_body_raises(self):
        os.environ.pop(TRACE_ENV_VAR, None)
        with self.assertRaises(RuntimeError):
            with dispatch_trace():
                raise RuntimeError("evaluation blew up")
        self.assertNotIn(TRACE_ENV_VAR, os.environ)

    def test_an_explicit_path_starts_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "trace.jsonl"
            with dispatch_trace(target) as path:
                self.assertEqual(path, target)
                self.assertEqual(path.read_text(encoding="utf-8"), "")
            self.assertIsNotNone(path)


class CaseIdentityTests(unittest.TestCase):
    """The trace must name the case, so the evaluator has to publish which one."""

    def setUp(self):
        os.environ.pop(CASE_ID_ENV_VAR, None)
        self.addCleanup(os.environ.pop, CASE_ID_ENV_VAR, None)

    def test_case_id_is_published_inside_the_block(self):
        with dispatch_trace(case_id="hidden_gap_003"):
            self.assertEqual(current_case_id(), "hidden_gap_003")
            self.assertEqual(os.environ[CASE_ID_ENV_VAR], "hidden_gap_003")

    def test_case_id_is_unset_after_the_block_when_it_was_unset_before(self):
        with dispatch_trace(case_id="hidden_gap_003"):
            pass
        self.assertNotIn(CASE_ID_ENV_VAR, os.environ, "the case identity must not leak")
        self.assertIsNone(current_case_id())

    def test_a_previous_case_id_is_restored(self):
        os.environ[CASE_ID_ENV_VAR] = "previous_case"
        with dispatch_trace(case_id="another_case"):
            self.assertEqual(current_case_id(), "another_case")
        self.assertEqual(current_case_id(), "previous_case")

    def test_case_id_is_restored_even_when_the_body_raises(self):
        with self.assertRaises(RuntimeError):
            with dispatch_trace(case_id="exploding_case"):
                raise RuntimeError("evaluation blew up")
        self.assertNotIn(CASE_ID_ENV_VAR, os.environ)

    def test_consecutive_evaluations_do_not_see_each_others_case(self):
        """Two cases in one process must not be able to label each other's record."""
        with dispatch_trace(case_id="first"):
            self.assertEqual(current_case_id(), "first")
        with dispatch_trace(case_id="second"):
            self.assertEqual(current_case_id(), "second")

    def test_omitting_the_case_id_leaves_it_unset(self):
        """Silently inheriting the previous case would let a submission pass by luck."""
        self.assertIsNone(current_case_id())
        with dispatch_trace():
            self.assertIsNone(current_case_id())

    def test_a_restored_empty_caller_value_is_not_treated_as_set(self):
        with dispatch_trace(case_id="inner"):
            self.assertEqual(current_case_id(), "inner")
        self.assertIsNone(current_case_id())


class ReadTraceTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        handle, name = tempfile.mkstemp(suffix=".jsonl")
        os.close(handle)
        path = Path(name)
        path.write_text(text, encoding="utf-8")
        self.addCleanup(path.unlink)
        return path

    def test_parses_json_lines(self):
        path = self._write(json.dumps(_record()) + "\n" + json.dumps(_record("smoke_002")) + "\n")
        records = read_trace(path)
        self.assertEqual([record["case_id"] for record in records], ["smoke_001", "smoke_002"])

    def test_blank_lines_are_ignored(self):
        path = self._write("\n" + json.dumps(_record()) + "\n\n")
        self.assertEqual(len(read_trace(path)), 1)

    def test_malformed_line_is_a_hard_error(self):
        path = self._write("{not json}\n")
        with self.assertRaises(DispatchTraceError):
            read_trace(path)

    def test_non_object_line_is_a_hard_error(self):
        path = self._write("[1, 2, 3]\n")
        with self.assertRaises(DispatchTraceError):
            read_trace(path)

    def test_missing_file_is_a_hard_error(self):
        with self.assertRaises(DispatchTraceError):
            read_trace(Path("/tmp/definitely-not-a-dispatch-trace.jsonl"))


class CheckTraceTests(unittest.TestCase):
    def test_empty_trace_is_rejected(self):
        errors = check_trace([], expected_path="fused_library")
        self.assertTrue(any("empty" in message for message in errors), errors)

    def test_matching_trace_passes(self):
        self.assertEqual(check_trace([_record()], expected_path="fused_library"), [])

    def test_missing_field_is_reported(self):
        errors = check_trace([{"case_id": "c", "selected_path": "fused_library"}])
        self.assertTrue(any("probe_status" in message for message in errors), errors)

    def test_empty_field_value_is_reported(self):
        errors = check_trace([_record(probe_status="")])
        self.assertTrue(any("probe_status" in message for message in errors), errors)

    def test_wrong_path_is_reported(self):
        errors = check_trace([_record(selected_path="custom_fallback")], expected_path="fused_library")
        self.assertTrue(any("does not match the expected path" in message for message in errors), errors)

    def test_unknown_path_is_reported(self):
        errors = check_trace([_record(selected_path="mudnn_magic")])
        self.assertTrue(any("unknown dispatch paths" in message for message in errors), errors)

    def test_an_unstable_path_is_reported(self):
        """One configuration answered two ways is not a trace the gate can trust."""
        records = [_record(selected_path="fused_library"), _record(selected_path="custom_fallback")]
        errors = check_trace(records, expected_path="fused_library")
        self.assertTrue(any("does not match the expected path" in message for message in errors), errors)

    def test_structural_errors_short_circuit_the_semantic_check(self):
        """A record missing fields should not also produce a confusing path mismatch."""
        errors = check_trace([{"case_id": "c"}], expected_path="fused_library")
        self.assertTrue(all("selected_path" in message or "probe_status" in message for message in errors), errors)

    def test_case_id_filtering_ignores_other_cases(self):
        records = [_record("smoke_001"), _record("smoke_002", selected_path="custom_fallback")]
        self.assertEqual(check_trace(records, expected_path="fused_library", case_id="smoke_001"), [])

    def test_case_id_filtering_reports_a_missing_case(self):
        errors = check_trace([_record("smoke_001")], case_id="smoke_003")
        self.assertTrue(any("smoke_003" in message for message in errors), errors)

    def test_required_fields_can_be_overridden(self):
        records = [{"case_id": "c", "selected_path": "fused_library", "probe_status": "ok", "route": "mudnn"}]
        self.assertEqual(check_trace(records, required_fields=["route"]), [])


class SummarizeTraceTests(unittest.TestCase):
    def test_counts_and_case_ids(self):
        records = [
            _record("smoke_001", "fused_library"),
            _record("smoke_002", "library_composition", "rejected"),
        ]
        summary = summarize_trace(records)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["case_ids"], ["smoke_001", "smoke_002"])
        self.assertEqual(summary["selected_path_counts"], {"fused_library": 1, "library_composition": 1})
        self.assertEqual(summary["probe_statuses"], ["accepted", "rejected"])

    def test_empty_trace(self):
        summary = summarize_trace([])
        self.assertEqual(summary["records"], 0)
        self.assertEqual(summary["selected_path_counts"], {})


class ConstantsTests(unittest.TestCase):
    def test_dispatch_paths_are_in_preference_order(self):
        self.assertEqual(DISPATCH_PATHS, ("fused_library", "library_composition", "custom_fallback"))

    def test_required_fields_match_the_task_contract(self):
        self.assertEqual(REQUIRED_TRACE_FIELDS, ("case_id", "selected_path", "probe_status"))


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
