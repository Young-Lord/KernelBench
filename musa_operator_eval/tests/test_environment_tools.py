import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        spec = importlib.util.spec_from_file_location("probe_sdpa", ROOT / "tools" / "probe_sdpa.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        result = module.unavailable_results({"family": "sdpa_forward", "required_routes": ["x"]}, "no device")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["routes"]["x"], "not_probed")


if __name__ == "__main__":
    unittest.main()
