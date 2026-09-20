import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    """Load probe_sdpa.py by path so the tests never import torch."""
    spec = importlib.util.spec_from_file_location("probe_sdpa", ROOT / "tools" / "probe_sdpa.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EnvironmentToolTests(unittest.TestCase):
    def test_capability_plan_has_unique_ids_and_required_dimensions(self):
        plan = json.loads((ROOT / "tasks" / "sdpa_forward_pilot" / "capability_plan.json").read_text(encoding="utf-8"))
        ids = [case["probe_id"] for case in plan["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 12)
        required = {"dtype", "layout", "B", "H_q", "H_kv", "S_q", "S_kv", "D", "causal", "window_left", "window_right"}
        for case in plan["cases"]:
            self.assertTrue(required <= case.keys())

    def test_unavailable_probe_is_explicit(self):
        result = load_probe().unavailable_results({"family": "sdpa_forward", "required_routes": ["x"]}, "no device")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["routes"]["x"], "not_probed")

    def test_declared_record_fields_are_actually_written(self):
        """The plan's record_fields must not drift from what the probe emits."""
        plan = json.loads((ROOT / "tasks" / "sdpa_forward_pilot" / "capability_plan.json").read_text(encoding="utf-8"))
        written = set(load_probe().empty_record(plan["cases"][0]).keys())
        for field in plan["record_fields"]:
            self.assertIn(field, written, f"plan declares {field!r} but the probe never writes it")

    def test_route_resolution_reads_the_dispatched_op(self):
        resolve_route = load_probe().resolve_route
        route, deciding = resolve_route({"scaled_dot_product_attention", "_scaled_dot_product_attention_flash_musa"})
        self.assertEqual(route, "fused_library")
        self.assertEqual(deciding, "_scaled_dot_product_attention_flash_musa")

        route, deciding = resolve_route({"scaled_dot_product_attention", "_scaled_dot_product_attention_math_musa"})
        self.assertEqual(route, "library_composition")
        self.assertEqual(deciding, "_scaled_dot_product_attention_math_musa")

        route, deciding = resolve_route({"scaled_dot_product_attention", "matmul", "softmax"})
        self.assertEqual(route, "custom_fallback")
        self.assertIn("matmul", deciding)

    def test_route_resolution_prefers_fused_over_composite(self):
        route, _ = load_probe().resolve_route({
            "_scaled_dot_product_attention_flash_musa", "matmul", "softmax", "_scaled_dot_product_attention_math_musa"})
        self.assertEqual(route, "fused_library")

    def test_route_resolution_stays_unverified_when_nothing_is_observed(self):
        route, deciding = load_probe().resolve_route(set())
        self.assertEqual(route, "unverified")
        self.assertIsNone(deciding)


if __name__ == "__main__":
    unittest.main()
