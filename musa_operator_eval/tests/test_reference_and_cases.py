import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_library_policy_from_problem():
    """Read LIBRARY_POLICY out of problem.py without importing it.

    problem.py imports torch, which is not available on every machine that runs
    this suite, so the literal is recovered from the AST instead.
    """
    tree = ast.parse((TASK_DIR / "problem.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LIBRARY_POLICY" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("problem.py does not define LIBRARY_POLICY")


reference = load("sdpa_reference", TASK_DIR / "reference" / "sdpa_reference.py")
generator = load("generate_cases", ROOT / "tools" / "generate_cases.py")


class TierContractConsistencyTests(unittest.TestCase):
    """problem.py and task.json both declare the tier policy; they must agree."""

    def setUp(self):
        self.policy = read_library_policy_from_problem()
        self.task = json.loads((TASK_DIR / "task.json").read_text(encoding="utf-8"))
        self.declared = self.task["library_policy"]

    def test_tier_agrees(self):
        self.assertEqual(self.task["tier"], "B_library")

    def test_shared_policy_fields_are_identical(self):
        for field in (
            "allowed_libraries", "allowed_symbol_prefixes", "required_trace_fields",
            "trace_env_var", "case_id_env_var", "minimum_gap_cases", "required_gap_reasons",
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.declared, f"task.json is missing library_policy.{field}")
                self.assertEqual(
                    self.policy[field], self.declared[field],
                    f"{field} differs between problem.py:TIER and task.json:library_policy",
                )

    def test_trace_variables_match_the_transport_constants(self):
        """The contract names env vars the transport actually sets.

        A contract that promises a variable nothing publishes is worse than one
        that promises nothing: the submission reads it, gets None, and the
        failure surfaces as an unreadable trace rather than a missing feature.
        """
        transport = load("dispatch_trace", ROOT.parent / "src" / "kernelbench" / "dispatch_trace.py")
        self.assertEqual(self.policy["trace_env_var"], transport.TRACE_ENV_VAR)
        self.assertEqual(self.policy["case_id_env_var"], transport.CASE_ID_ENV_VAR)

    def test_required_trace_fields_match_the_transport_defaults(self):
        transport = load("dispatch_trace", ROOT.parent / "src" / "kernelbench" / "dispatch_trace.py")
        self.assertEqual(tuple(self.policy["required_trace_fields"]), tuple(transport.REQUIRED_TRACE_FIELDS))

    def test_dispatch_order_matches_the_capability_reference(self):
        capabilities = json.loads((ROOT / "agent_reference" / "attention_capabilities.json").read_text(encoding="utf-8"))
        self.assertEqual(self.policy["dispatch_order"], capabilities["dispatch_contract"]["ordered_paths"])

    def test_gap_reasons_come_from_the_declared_vocabulary(self):
        vocabulary = set(json.loads(
            (ROOT / "agent_reference" / "attention_capabilities.json").read_text(encoding="utf-8")
        )["dispatch_contract"]["failure_reasons"])
        for reason in self.policy["required_gap_reasons"]:
            self.assertIn(reason, vocabulary, f"{reason} is not a declared failure reason")

    def test_problem_file_and_reference_entrypoint_exist(self):
        self.assertTrue((TASK_DIR / self.task["problem_file"]).is_file())
        module_name, _, attribute = self.task["reference"].partition(":")
        self.assertEqual(module_name, self.task["problem_file"])
        self.assertIn(attribute, {"Model", "ModelNew"})


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
