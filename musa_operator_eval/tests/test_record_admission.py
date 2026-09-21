"""Tests for the tool that records a B-tier admission decision.

Standard library only, like the tool itself. The cases that earn their keep are the
refusals: an `admit` written into a contract is a claim that both halves of the
gate hold, and the point of the tool is to make that claim checkable rather than
asserted. So most of these build a baseline that would justify an admit and then
break exactly one thing in it.

Run with:
    python3 musa_operator_eval/tests/test_record_admission.py -v
"""

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("record_admission", ROOT / "evaluator" / "record_admission.py")
assert _spec is not None and _spec.loader is not None
recorder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorder)

SNAPSHOT = "musa-5f9d7b9dd1233a68"


def task(snapshot=SNAPSHOT):
    return {
        "id": "example_b_v0",
        "target_environment": {"snapshot_id": snapshot},
        "admission": {"status": "pending_device_measurement", "reason": "not measured yet"},
    }


def row(case_id, implementation, latency, **overrides):
    document = {
        "case_id": case_id,
        "implementation": implementation,
        "status": "pass",
        "latency_ms": latency,
        "dtype": "float16",
    }
    document.update(overrides)
    return document


def baseline(cases=("hidden_gap_001", "hidden_gap_002", "hidden_gap_003"), ratios=(2.0, 2.0, 2.0), **overrides):
    results = []
    for case_id, ratio in zip(cases, ratios):
        results.append(row(case_id, recorder.NAIVE_ROLE, 10.0))
        results.append(
            row(
                case_id,
                recorder.EXPERT_ROLE,
                10.0 / ratio,
                dispatch_trace_passed=True,
                recorded_paths=["fused_library"],
            )
        )
    document = {
        "task_id": "example_b_v0",
        "tier": "B_library",
        "environment_snapshot": SNAPSHOT,
        "implementations": [recorder.NAIVE_ROLE, recorder.EXPERT_ROLE],
        "results": results,
        "scoring": {
            "speedup_denominator": recorder.EXPERT_ROLE,
            "reference_implementation": recorder.NAIVE_ROLE,
            "correctness_gate": "all_hidden_cases",
            "minimum_naive_to_expert_ratio": 1.3,
        },
    }
    document.update(overrides)
    return document


def admission(geometric_mean=2.0, decision="admit", threshold=1.3):
    return {
        "decision": decision,
        "threshold": threshold,
        "geometric_mean_ratio": geometric_mean,
        "reason": "measured naive/expert geometric mean",
    }


class AuditTests(unittest.TestCase):
    def test_a_complete_pair_is_accepted(self):
        problems, measurements = recorder.audit(baseline(), admission(), task())
        self.assertEqual(problems, [])
        self.assertAlmostEqual(measurements["geometric_mean_ratio"], 2.0)
        self.assertEqual(len(measurements["hidden_cases"]), 3)

    def test_a_failing_hidden_case_is_refused(self):
        """Half one. Skipping a case is fine for the mean and not for the decision."""
        document = baseline()
        document["results"][0] = row("hidden_gap_001", recorder.NAIVE_ROLE, None, status="fail", reason="wrong")
        problems, _ = recorder.audit(document, admission(geometric_mean=2.0), task())
        self.assertTrue(any("both halves" in problem for problem in problems), problems)

    def test_a_mismatched_trace_is_refused(self):
        """Half two, which nothing was measuring until this session."""
        document = baseline()
        for result in document["results"]:
            if result["implementation"] == recorder.EXPERT_ROLE and result["case_id"] == "hidden_gap_002":
                result["dispatch_trace_passed"] = False
                result["recorded_paths"] = ["custom_fallback"]
        problems, _ = recorder.audit(document, admission(), task())
        self.assertTrue(any("trace did not match" in problem for problem in problems), problems)

    def test_an_unmeasured_expert_trace_is_refused(self):
        """`None` means nothing was recorded, which is not the same as matching."""
        document = baseline()
        for result in document["results"]:
            if result["implementation"] == recorder.EXPERT_ROLE:
                result["dispatch_trace_passed"] = None
        problems, _ = recorder.audit(document, admission(), task())
        self.assertTrue(any("trace did not match" in problem for problem in problems), problems)

    def test_a_baseline_from_another_environment_is_refused(self):
        problems, _ = recorder.audit(baseline(environment_snapshot="musa-0000000000000000"), admission(), task())
        self.assertTrue(any("comparable only within one snapshot" in problem for problem in problems), problems)

    def test_a_baseline_for_another_task_is_refused(self):
        problems, _ = recorder.audit(baseline(task_id="other_v0"), admission(), task())
        self.assertTrue(any("the contract is" in problem for problem in problems), problems)

    def test_a_baseline_with_no_hidden_case_is_refused(self):
        """The gate is defined over the hidden set, not over whatever was measured."""
        problems, _ = recorder.audit(baseline(cases=("smoke_001",), ratios=(2.0,)), admission(), task())
        self.assertTrue(any("no hidden case" in problem for problem in problems), problems)

    def test_a_geometric_mean_that_disagrees_with_its_inputs_is_refused(self):
        """The gate's own arithmetic is re-derived rather than trusted."""
        problems, _ = recorder.audit(baseline(), admission(geometric_mean=9.9), task())
        self.assertTrue(any("re-derived from the baseline" in problem for problem in problems), problems)

    def test_a_decision_that_disagrees_with_its_inputs_is_refused(self):
        problems, _ = recorder.audit(baseline(ratios=(1.1, 1.1, 1.1)), admission(geometric_mean=1.1), task())
        self.assertTrue(any("the decision should be" in problem for problem in problems), problems)

    def test_a_threshold_that_disagrees_with_the_baseline_is_refused(self):
        problems, _ = recorder.audit(baseline(), admission(threshold=1.0), task())
        self.assertTrue(any("the baseline declares" in problem for problem in problems), problems)

    def test_the_contract_block_carries_no_hidden_case_id(self):
        """§2 hands the agent the contract, so the hidden set stays out of it.

        A ratio table keyed by hidden case ids says which configuration is the gap
        and what the fused path is worth when it applies. That is the finding the
        task exists to make a submission discover, so it belongs in the private
        record, and the contract keeps only the gate's outcome.
        """
        problems, measurements = recorder.audit(baseline(), admission(), task())
        self.assertEqual(problems, [])
        contract_block, evidence = recorder.admission_blocks(
            measurements, baseline(), admission(), Path("private/example/admission.json")
        )
        self.assertNotIn("ratios", contract_block)
        self.assertNotIn("baseline", contract_block)
        for value in contract_block.values():
            self.assertNotIn("hidden_", json.dumps(value, default=str))
        self.assertEqual(sorted(evidence["ratios"]), sorted(measurements["ratios"]))
        self.assertEqual(sorted(contract_block), sorted({"status", "decision", "measured_geometric_mean", "threshold", "measured_in"}))

    def test_the_geometric_mean_is_a_mean_of_ratios_not_of_latencies(self):
        """A long case and a short case have to weigh the same, which is the point."""
        document = baseline(ratios=(4.0, 1.0, 1.0))
        _, measurements = recorder.audit(document, admission(geometric_mean=4.0 ** (1 / 3)), task())
        self.assertAlmostEqual(measurements["geometric_mean_ratio"], 4.0 ** (1 / 3))
        self.assertNotAlmostEqual(measurements["geometric_mean_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
