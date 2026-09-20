"""Unit tests for the B-tier admission gate.

The gate decides whether a task is published at all, and it decides it from
measured latencies, so the ways it can go wrong are the ways any measurement
gate goes wrong: it accepts a ratio it should not, it silently ignores a case it
could not pair, or it reports a number computed from the wrong pairs.

Stdlib only, so this runs anywhere.

Run with either:
    python -m unittest musa_operator_eval.tests.test_candidate_gate -v
    python musa_operator_eval/tests/test_candidate_gate.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("candidate_gate", ROOT / "tools" / "candidate_gate.py")
assert _spec is not None and _spec.loader is not None
candidate_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(candidate_gate)

evaluate = candidate_gate.evaluate


def measurement(case_id, implementation, latency_ms, status="pass"):
    return {
        "case_id": case_id, "implementation": implementation,
        "status": status, "latency_ms": latency_ms,
    }


def baseline(results, threshold=1.3):
    return {"results": results, "scoring": {"minimum_naive_to_expert_ratio": threshold}}


class DecisionTests(unittest.TestCase):
    def test_admits_when_the_ratio_clears_the_threshold(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 2.0),
            measurement("c1", "expert_dispatch", 1.0),
        ]))
        self.assertEqual(result["decision"], "admit")
        self.assertAlmostEqual(result["geometric_mean_ratio"], 2.0)

    def test_rejects_when_the_ratio_falls_short(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 1.2),
            measurement("c1", "expert_dispatch", 1.0),
        ]))
        self.assertEqual(result["decision"], "reject")
        self.assertLess(result["geometric_mean_ratio"], 1.3)

    def test_exactly_at_the_threshold_is_admitted(self):
        """The rule is `>=`, so a ratio of exactly 1.3 must not be rounded away."""
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 1.3),
            measurement("c1", "expert_dispatch", 1.0),
        ]))
        self.assertEqual(result["decision"], "admit")

    def test_the_threshold_comes_from_the_baseline(self):
        results = [
            measurement("c1", "naive_library_composition", 1.5),
            measurement("c1", "expert_dispatch", 1.0),
        ]
        self.assertEqual(evaluate(baseline(results, threshold=1.3))["decision"], "admit")
        self.assertEqual(evaluate(baseline(results, threshold=2.0))["decision"], "reject")


class AggregationTests(unittest.TestCase):
    def test_the_mean_is_geometric_not_arithmetic(self):
        """One huge win must not paper over the cases the expert loses.

        Two cases at 10x and two at 0.1x: the geometric mean is 1.0 (reject),
        while the arithmetic mean is 5.05 (admit). A gate that averaged
        arithmetic would admit on the strength of a single favourable case.
        """
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 10.0),
            measurement("c1", "expert_dispatch", 1.0),
            measurement("c2", "naive_library_composition", 10.0),
            measurement("c2", "expert_dispatch", 1.0),
            measurement("c3", "naive_library_composition", 0.1),
            measurement("c3", "expert_dispatch", 1.0),
            measurement("c4", "naive_library_composition", 0.1),
            measurement("c4", "expert_dispatch", 1.0),
        ]))
        self.assertAlmostEqual(result["geometric_mean_ratio"], 1.0, places=6)
        self.assertEqual(result["decision"], "reject")

    def test_each_case_is_reported_separately(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 1.0),
            measurement("c2", "naive_library_composition", 9.0),
            measurement("c2", "expert_dispatch", 1.0),
        ]))
        self.assertEqual(result["ratios"], {"c1": 4.0, "c2": 9.0})
        self.assertAlmostEqual(result["geometric_mean_ratio"], 6.0)


class PairingTests(unittest.TestCase):
    def test_a_case_missing_the_expert_is_dropped(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 1.0),
            measurement("c2", "naive_library_composition", 100.0),
        ]))
        self.assertEqual(set(result["ratios"]), {"c1"})

    def test_a_case_missing_the_naive_baseline_is_dropped(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 1.0),
            measurement("c2", "expert_dispatch", 1.0),
        ]))
        self.assertEqual(set(result["ratios"]), {"c1"})

    def test_a_failing_implementation_is_not_paired(self):
        """A ratio against a wrong answer would admit a task for the wrong reason."""
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0, status="fail"),
            measurement("c1", "expert_dispatch", 1.0),
        ]))
        self.assertEqual(result["decision"], "insufficient_data")

    def test_an_errored_implementation_is_not_paired(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 1.0, status="error"),
        ]))
        self.assertEqual(result["decision"], "insufficient_data")

    def test_a_missing_latency_is_not_paired(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", None),
        ]))
        self.assertEqual(result["decision"], "insufficient_data")

    def test_a_zero_latency_is_not_paired(self):
        """A zero would divide by zero, and pretending it is a ratio is worse."""
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 0.0),
        ]))
        self.assertEqual(result["decision"], "insufficient_data")

    def test_no_results_is_insufficient_data(self):
        result = evaluate(baseline([]))
        self.assertEqual(result["decision"], "insufficient_data")
        self.assertIn("no paired", result["reason"])

    def test_other_implementations_are_ignored(self):
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 1.0),
            measurement("c1", "torch_musa_sdpa_blind", 0.01),
            measurement("c1", "upstream_musa", 0.01, status="unavailable"),
        ]))
        self.assertEqual(result["ratios"], {"c1": 4.0})

    def test_unavailable_measurements_do_not_break_the_decision(self):
        """The template names six implementations; three may not exist yet."""
        result = evaluate(baseline([
            measurement("c1", "naive_library_composition", 4.0),
            measurement("c1", "expert_dispatch", 1.0),
            measurement("c1", "upstream_musa", None, status="unavailable"),
            measurement("c1", "mudnn_fused", None, status="unavailable"),
            measurement("c1", "torch_musa_eager", None, status="unavailable"),
        ]))
        self.assertEqual(result["decision"], "admit")
        self.assertEqual(result["ratios"], {"c1": 4.0})


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
