"""Tests for §5 step 8's two tables.

The property worth testing is not the arithmetic but the separation: a B-tier
report must never appear in the A-tier table, a report whose denominator is
missing must be listed rather than dropped, and the unit conversion between the
framework's microseconds and a baseline row's milliseconds must happen exactly
once. A scoreboard that quietly averages the two tracks is the failure mode this
tool exists to avoid, so the tests check it can't happen.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rank = load_module("musa_rank_tests", ROOT / "evaluator" / "rank.py")


def write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def baseline(task_id: str, tier: str, denominator: str, latencies: dict) -> dict:
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "tier": tier,
        "scoring": {"speedup_denominator": denominator, "reference_implementation": "reference_model"},
        "results": [
            {"case_id": case_id, "implementation": denominator, "status": "pass", "latency_ms": latency}
            for case_id, latency in latencies.items()
        ],
    }


def report(task_id: str, tier: str, runtimes_us: dict) -> dict:
    return {
        "schema_version": "1.0.0",
        "mode": "binary",
        "task_id": task_id,
        "tier": tier,
        "cases": [{"case_id": case_id, "runtime": runtime} for case_id, runtime in runtimes_us.items()],
    }


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_a_report_runtime_divides_into_the_baseline_directly(self):
        """Both sides are milliseconds, so a 1.0 ms case against a 2.0 ms baseline is 2x."""
        baselines = {"task": baseline("task", "A_kernel", "upstream_musa", {"c1": 2.0})}
        row = rank.rank_report(report("task", "A_kernel", {"c1": 1.0}), baselines)
        self.assertEqual(row["denominator"], "upstream_musa")
        self.assertEqual(row["ratios"], {"c1": 2.0})
        self.assertEqual(row["geometric_mean_ratio"], 2.0)

    def test_the_geometric_mean_is_used_not_the_arithmetic_mean(self):
        """4x and 1x average to 2x geometrically and 2.5x arithmetically."""
        baselines = {"task": baseline("task", "B_library", "expert_dispatch", {"c1": 4.0, "c2": 1.0})}
        row = rank.rank_report(report("task", "B_library", {"c1": 1.0, "c2": 1.0}), baselines)
        self.assertEqual(row["geometric_mean_ratio"], 2.0)

    def test_a_case_without_a_passing_reference_is_excluded_and_named(self):
        baselines = {
            "task": {
                "task_id": "task",
                "tier": "A_kernel",
                "scoring": {"speedup_denominator": "upstream_musa"},
                "results": [
                    {"case_id": "c1", "implementation": "upstream_musa", "status": "pass", "latency_ms": 2.0},
                    {"case_id": "c2", "implementation": "upstream_musa", "status": "error", "latency_ms": None},
                ],
            }
        }
        row = rank.rank_report(report("task", "A_kernel", {"c1": 1.0, "c2": 0.5}), baselines)
        self.assertEqual(row["cases_compared"], 1)
        self.assertIn("c2", row["excluded"])
        self.assertIn("no passing upstream_musa measurement", row["excluded"]["c2"])

    def test_a_case_without_a_runtime_is_excluded(self):
        baselines = {"task": baseline("task", "A_kernel", "upstream_musa", {"c1": 2.0})}
        row = rank.rank_report(report("task", "A_kernel", {"c1": None}), baselines)
        self.assertEqual(row["cases_compared"], 0)
        self.assertIn("no usable runtime", row["excluded"]["c1"])

    def test_a_report_without_a_baseline_is_listed_rather_than_dropped(self):
        row = rank.rank_report(report("orphan", "A_kernel", {"c1": 1.0}), {})
        self.assertIn("no baseline record", row["unranked"])

    def test_an_unknown_tier_is_not_ranked(self):
        row = rank.rank_report(report("task", "C_other", {"c1": 1.0}), {})
        self.assertIn("does not rank", row["unranked"])

    def test_the_two_tracks_are_never_combined(self):
        a_baseline = write(self.tmp / "a.json", baseline("task_a", "A_kernel", "upstream_musa", {"c1": 4.0}))
        b_baseline = write(self.tmp / "b.json", baseline("task_b", "B_library", "expert_dispatch", {"c1": 1.0}))
        a_report = write(self.tmp / "ra.json", report("task_a", "A_kernel", {"c1": 1.0}))
        b_report = write(self.tmp / "rb.json", report("task_b", "B_library", {"c1": 1.0}))

        result = rank.rank([a_report, b_report], [a_baseline, b_baseline])
        self.assertEqual([row["task_id"] for row in result["tables"]["A_kernel"]["ranked"]], ["task_a"])
        self.assertEqual([row["task_id"] for row in result["tables"]["B_library"]["ranked"]], ["task_b"])
        self.assertEqual(result["unranked"], [])
        text = rank.format_tables(result)
        self.assertIn("x4 over 1 of 1 cases", text)
        self.assertIn("x1 over 1 of 1 cases", text)
        self.assertIn("never combined", text)

    def test_a_small_ratio_is_not_printed_as_zero(self):
        """An A-tier kernel measured against a tuned one is a fraction, and the fraction is the number."""
        baselines = write(self.tmp / "a.json", baseline("task_a", "A_kernel", "upstream_musa", {"c1": 0.0015}))
        report_path = write(self.tmp / "ra.json", report("task_a", "A_kernel", {"c1": 0.558}))
        result = rank.rank([report_path], [baselines])
        text = rank.format_tables(result)
        self.assertIn("x0.002688", text)

    def test_the_tables_sort_by_the_ratio(self):
        baselines = write(self.tmp / "b.json", baseline("task_b", "B_library", "expert_dispatch", {"c1": 1.0}))
        slow = write(self.tmp / "slow.json", report("task_b", "B_library", {"c1": 2.0}))
        fast = {
            "schema_version": "1.0.0",
            "task_id": "task_b",
            "tier": "B_library",
            "cases": [{"case_id": "c1", "runtime": 1.0}],
        }
        fast_path = write(self.tmp / "fast.json", fast)
        result = rank.rank([slow, fast_path], [baselines])
        # Same task id, so one baseline record; both reports rank and the faster sorts first.
        self.assertEqual(result["tables"]["B_library"]["ranked"][0]["geometric_mean_ratio"], 1.0)
        self.assertEqual(result["tables"]["B_library"]["ranked"][1]["geometric_mean_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
