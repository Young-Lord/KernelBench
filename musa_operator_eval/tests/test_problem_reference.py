"""
Cross-check the task's PyTorch reference against the independent NumPy one.

`problem.py:Model` is what a submission is graded against, so it has to agree
with `reference/sdpa_reference.py`, which is the blocked float64 implementation
the golden tensors are generated from. The two are deliberately different code
paths: torch on float32 with a max-shift softmax, NumPy on float64 in query
blocks.

The PyTorch reference accumulates in float32 (that is the contract), so the
float64 reference is only an oracle down to float32 rounding — hence the
tolerances below rather than exact equality.

Skipped when torch is unavailable, so the rest of the suite still runs on a
machine without a GPU stack.

Run with:
    python musa_operator_eval/tests/test_problem_reference.py -v
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1] / "tasks" / "sdpa_forward_pilot"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_module("sdpa_reference", TASK_DIR / "reference" / "sdpa_reference.py")

try:
    import torch

    PROBLEM = _load_module("sdpa_problem", TASK_DIR / "problem.py")
except ImportError:  # torch is not installed
    torch = None
    PROBLEM = None


# Configurations that exercise each clause of the semantic contract. The shape
# tuple is (B, H_q, H_kv, S_q, S_kv, D).
CONFIGS = [
    ("mha_square_fp32", (1, 4, 4, 16, 16, 32), 0.1767766952966369, False, -1, -1),
    ("mha_causal", (2, 4, 4, 33, 33, 64), 0.125, True, -1, -1),
    ("gqa_ratio_2", (1, 8, 4, 32, 32, 64), 0.125, False, -1, -1),
    ("gqa_ratio_4", (2, 16, 4, 64, 64, 128), 0.08838834764831845, False, -1, -1),
    ("cross_attention", (1, 8, 8, 127, 193, 64), 0.125, False, -1, -1),
    ("causal_cross", (1, 4, 4, 65, 129, 64), 0.125, True, -1, -1),
    ("window_both_sides", (1, 4, 4, 64, 64, 64), 0.125, False, 16, 16),
    ("window_left_only", (1, 4, 4, 64, 64, 64), 0.125, True, 32, -1),
    ("window_right_only", (1, 4, 4, 64, 64, 64), 0.125, False, -1, 8),
    ("head_dim_80", (1, 8, 8, 96, 96, 80), 0.11180339887498949, False, -1, -1),
    ("head_dim_96_non_aligned", (1, 12, 12, 127, 127, 96), 0.10206207261596575, False, -1, -1),
    ("non_aligned_sequence", (1, 2, 2, 17, 17, 32), 0.1767766952966369, False, -1, -1),
]


def _to_torch(array_fp32: np.ndarray, dtype):
    """Quantize through the target dtype and hand back the exact float values."""
    tensor = torch.from_numpy(np.ascontiguousarray(array_fp32)).to(dtype)
    return tensor, tensor.float().numpy()


# The contract states atol = rtol = 1e-2. Float32 only needs to cover the gap
# between float32 and float64 accumulation, so it can be checked harder.
def _tolerances_for(dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 1e-2, 1e-2


@unittest.skipUnless(torch is not None, "requires torch")
class ProblemReferenceTests(unittest.TestCase):
    def _compare(self, shape, scale, causal, window_left, window_right, dtype, seed=0):
        batch, h_q, h_kv, s_q, s_kv, d = shape
        atol, rtol = _tolerances_for(dtype)
        rng = np.random.default_rng(seed)
        q_t, q_np = _to_torch(rng.standard_normal((batch, h_q, s_q, d), dtype=np.float32), dtype)
        k_t, k_np = _to_torch(rng.standard_normal((batch, h_kv, s_kv, d), dtype=np.float32), dtype)
        v_t, v_np = _to_torch(rng.standard_normal((batch, h_kv, s_kv, d), dtype=np.float32), dtype)

        expected_output, _ = REFERENCE.sdpa_forward(
            q_np, k_np, v_np,
            scale=scale, causal=causal, window_left=window_left, window_right=window_right,
        )

        model = PROBLEM.Model(scale, causal, window_left, window_right)
        output = model(q_t, k_t, v_t)

        self.assertEqual(tuple(output.shape), (batch, h_q, s_q, d), "output shape")
        self.assertEqual(output.dtype, dtype, "output dtype must follow the input")

        self.assertTrue(
            torch.isfinite(output).all(),
            f"output contains non-finite values for {shape}",
        )
        np.testing.assert_allclose(
            output.float().numpy(), expected_output, atol=atol, rtol=rtol,
            err_msg=f"output mismatch for shape={shape} causal={causal} dtype={dtype}",
        )

    def test_matches_numpy_reference_fp32(self):
        for name, shape, scale, causal, window_left, window_right in CONFIGS:
            with self.subTest(config=name):
                self._compare(shape, scale, causal, window_left, window_right, torch.float32)

    def test_matches_numpy_reference_fp16(self):
        """Half precision keeps the contract tolerance of atol = rtol = 1e-2."""
        for name, shape, scale, causal, window_left, window_right in CONFIGS:
            if shape[5] == 80:  # head_dim 80 is exercised at fp32; keep the runtime sane
                continue
            with self.subTest(config=name):
                batch, h_q, h_kv, s_q, s_kv, d = shape
                rng = np.random.default_rng(7)
                q_t, q_np = _to_torch(rng.standard_normal((batch, h_q, s_q, d), dtype=np.float32), torch.float16)
                k_t, k_np = _to_torch(rng.standard_normal((batch, h_kv, s_kv, d), dtype=np.float32), torch.float16)
                v_t, v_np = _to_torch(rng.standard_normal((batch, h_kv, s_kv, d), dtype=np.float32), torch.float16)

                expected_output, _ = REFERENCE.sdpa_forward(
                    q_np, k_np, v_np,
                    scale=scale, causal=causal, window_left=window_left, window_right=window_right,
                )
                output = PROBLEM.Model(scale, causal, window_left, window_right)(q_t, k_t, v_t)

                np.testing.assert_allclose(
                    output.float().numpy(), expected_output, atol=1e-2, rtol=1e-2,
                    err_msg=f"fp16 output mismatch for {name}",
                )

    def test_bfloat16_output_keeps_input_dtype(self):
        model = PROBLEM.Model(0.125, False, -1, -1)
        q = torch.randn(1, 4, 4, 32, dtype=torch.bfloat16)
        k = torch.randn(1, 4, 4, 32, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 4, 32, dtype=torch.bfloat16)
        output = model(q, k, v)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_reference_returns_a_bare_tensor(self):
        """The harness compares `output.shape`, so a tuple reference cannot be graded.

        This is not a style preference: run_and_check_correctness reads `.shape`
        off the reference's return value, and a tuple raises AttributeError before
        any comparison happens, failing every trial of every case.
        """
        model = PROBLEM.Model(0.125, False, -1, -1)
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 4, 32)
        v = torch.randn(1, 2, 4, 32)
        output = model(q, k, v)
        self.assertIsInstance(output, torch.Tensor, "the reference must return one tensor, not a tuple")
        self.assertEqual(tuple(output.shape), (1, 2, 4, 32))

    def test_public_cases_match_the_reference(self):
        """Every public case must reproduce under the torch reference."""
        manifest = json.loads((TASK_DIR / "public_cases.json").read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            with self.subTest(case=case["case_id"]):
                shape = case["shape"]
                attributes = case["attributes"]
                config = (
                    shape["B"], shape["H_q"], shape["H_kv"],
                    shape["S_q"], shape["S_kv"], shape["D"],
                )
                dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[case["dtype"]]
                self._compare(
                    config,
                    attributes["scale"],
                    attributes["causal"],
                    attributes["window_left"],
                    attributes["window_right"],
                    dtype,
                    seed=case["seed"],
                )

    def test_gqa_expands_each_kv_head_to_a_consecutive_group(self):
        """H_q / H_kv expansion must line query head h up with kv head h // ratio."""
        batch, h_q, h_kv, s, d = 1, 4, 2, 4, 8
        model = PROBLEM.Model(1.0, False, -1, -1)
        rng = np.random.default_rng(11)
        q_t, q_np = _to_torch(rng.standard_normal((batch, h_q, s, d), dtype=np.float32), torch.float32)
        k_t, k_np = _to_torch(rng.standard_normal((batch, h_kv, s, d), dtype=np.float32), torch.float32)
        v_t, v_np = _to_torch(rng.standard_normal((batch, h_kv, s, d), dtype=np.float32), torch.float32)

        output = model(q_t, k_t, v_t)
        expected, _ = REFERENCE.sdpa_forward(q_np, k_np, v_np, scale=1.0)
        np.testing.assert_allclose(output.float().numpy(), expected, atol=1e-3, rtol=1e-3)

        # Discriminating power: the other common expansion, tiling the whole K/V
        # tensors ([0, 1, 0, 1] instead of [0, 0, 1, 1]), must produce a
        # different result, otherwise this test could not catch a wrong expansion.
        tiled_k_t = torch.cat([k_t, k_t], dim=1)
        tiled_v_t = torch.cat([v_t, v_t], dim=1)
        output_from_tiled_kv = PROBLEM.Model(1.0, False, -1, -1)(q_t, tiled_k_t, tiled_v_t)
        self.assertFalse(
            torch.allclose(output, output_from_tiled_kv, atol=1e-6),
            "tiling K/V instead of repeating them should change the result",
        )


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
