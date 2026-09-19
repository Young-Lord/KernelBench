import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reference = load("sdpa_reference", ROOT / "tasks" / "sdpa_forward_pilot" / "reference" / "sdpa_reference.py")
generator = load("generate_cases", ROOT / "tools" / "generate_cases.py")


class ReferenceTests(unittest.TestCase):
    def test_uniform_scores_average_values(self):
        q = np.zeros((1, 1, 2, 2), dtype=np.float32)
        k = np.zeros((1, 1, 2, 2), dtype=np.float32)
        v = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=np.float32)
        output, lse = reference.sdpa_forward(q, k, v)
        np.testing.assert_allclose(output, [[[[3.0, 5.0], [3.0, 5.0]]]])
        np.testing.assert_allclose(lse, np.log(2.0))

    def test_fully_masked_row_is_zero_and_negative_infinity(self):
        q = k = v = np.ones((1, 1, 2, 2), dtype=np.float32)
        mask = np.array([[False, False], [True, False]])
        output, lse = reference.sdpa_forward(q, k, v, explicit_mask=mask)
        np.testing.assert_array_equal(output[0, 0, 0], 0.0)
        self.assertTrue(np.isneginf(lse[0, 0, 0]))

    def test_gqa_repeats_kv_heads(self):
        q = np.ones((1, 4, 2, 2), dtype=np.float32)
        k = v = np.ones((1, 2, 2, 2), dtype=np.float32)
        output, _ = reference.sdpa_forward(q, k, v)
        self.assertEqual(output.shape, q.shape)
        np.testing.assert_allclose(output, 1.0)

    def test_bfloat16_round_trip(self):
        values = np.array([0.0, 1.0, -2.5, 0.33333334], dtype=np.float32)
        encoded = generator.float32_to_bfloat16(values)
        decoded = generator.bfloat16_to_float32(encoded)
        np.testing.assert_allclose(decoded, values, rtol=0.01, atol=0.001)

    def test_case_generation_is_deterministic(self):
        manifest = json.loads((ROOT / "tasks" / "sdpa_forward_pilot" / "public_cases.json").read_text(encoding="utf-8"))
        case = manifest["cases"][0]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generator.generate_case(case, Path(first))
            generator.generate_case(case, Path(second))
            first_manifest = json.loads((Path(first) / case["case_id"] / "input" / "tensors.json").read_text())
            second_manifest = json.loads((Path(second) / case["case_id"] / "input" / "tensors.json").read_text())
            self.assertEqual(first_manifest, second_manifest)


if __name__ == "__main__":
    unittest.main()
