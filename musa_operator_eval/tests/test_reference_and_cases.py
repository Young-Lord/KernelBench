import ast
import importlib.util
import json
import re
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
        self.capabilities = json.loads((ROOT / self.task["capability_reference"]).read_text(encoding="utf-8"))

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
        self.assertEqual(self.policy["dispatch_order"], self.capabilities["dispatch_contract"]["ordered_paths"])

    def test_gap_reasons_come_from_the_declared_vocabulary(self):
        vocabulary = set(self.capabilities["dispatch_contract"]["failure_reasons"])
        for reason in self.policy["required_gap_reasons"]:
            self.assertIn(reason, vocabulary, f"{reason} is not a declared failure reason")

    def test_problem_file_and_reference_entrypoint_exist(self):
        self.assertTrue((TASK_DIR / self.task["problem_file"]).is_file())
        module_name, _, attribute = self.task["reference"].partition(":")
        self.assertEqual(module_name, self.task["problem_file"])
        self.assertIn(attribute, {"Model", "ModelNew"})


class TargetEnvironmentTests(unittest.TestCase):
    """A task names the environment it was measured on; the record describes it.

    The device facts live in `environments/`, not in the task package. One
    machine is described once: two tasks that each carried their own copy of the
    toolkit version could disagree about it, and nothing would notice.
    """

    def setUp(self):
        self.task = json.loads((TASK_DIR / "task.json").read_text(encoding="utf-8"))
        self.target = self.task["target_environment"]
        self.record_path = ROOT / self.target["record"]
        self.record = json.loads(self.record_path.read_text(encoding="utf-8"))

    def test_the_record_lives_outside_the_task_package(self):
        self.assertTrue(self.record_path.is_relative_to(ROOT / "environments"))
        self.assertFalse(self.record_path.is_relative_to(TASK_DIR))

    def test_the_record_is_named_after_its_snapshot_id(self):
        self.assertEqual(self.record_path.name, f"{self.target['snapshot_id']}.public.json")

    def test_the_declared_snapshot_id_is_the_one_inside_the_record(self):
        self.assertEqual(self.target["snapshot_id"], self.record["snapshot_id"])

    def test_the_task_does_not_restate_the_device_facts(self):
        """Restating a fact in two places is how the two places drift apart.

        The task names a snapshot and stops. A device name or a version string
        anywhere in it would be a second copy of something the record already
        says, and the two copies would be free to disagree.
        """
        declared = (TASK_DIR / "task.json").read_text(encoding="utf-8")
        restated = {
            "device name": self.record["hardware"]["device_name"],
            "architecture": self.record["hardware"]["architecture"],
            "toolkit version": self.record["software"]["musa_toolkit"],
            "mudnn version": self.record["software"]["mudnn"],
            "mublas version": self.record["software"]["mublas"],
            "driver version": self.record["software"]["driver"],
        }
        for fact, value in restated.items():
            if value:
                with self.subTest(fact=fact):
                    self.assertNotIn(str(value), declared)

    def test_the_record_carries_no_machine_identity(self):
        self.assertEqual(self.record["visibility"], "redacted")
        for key in ("device_instance", "device_instance_id", "host", "raw_probes", "container"):
            self.assertNotIn(key, self.record)

    def test_the_record_carries_no_field_the_collector_redacts(self):
        """The artifact is checked against the rule, not against a copy of it."""
        collector = load("collect_environment", ROOT / "tools" / "collect_environment.py")
        for key in collector.REDACTED_KEYS:
            self.assertNotIn(key, self.record)
        self.assertNotIn("device_instance_id", self.record)

    def test_the_record_keeps_the_configuration_the_agent_needs(self):
        self.assertTrue(self.record["hardware"]["device_name"])
        self.assertTrue(self.record["hardware"]["architecture"])
        self.assertTrue(self.record["software"]["musa_toolkit"])
        self.assertTrue(self.record["software"]["mudnn"])
        self.assertTrue(self.record["toolkit_components"])


class EnvironmentDirectoryTests(unittest.TestCase):
    """The directory is the index, so an unlisted record is an undiscoverable one."""

    def setUp(self):
        self.directory = ROOT / "environments"
        self.records = sorted(self.directory.glob("*.public.json"))
        self.readme = (self.directory / "README.md").read_text(encoding="utf-8")

    def test_there_is_at_least_one_record(self):
        self.assertTrue(self.records)

    def test_every_record_is_named_after_the_snapshot_id_inside_it(self):
        for path in self.records:
            with self.subTest(record=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(path.name, f"{record['snapshot_id']}.public.json")

    def test_every_record_is_listed_in_the_readme(self):
        for path in self.records:
            with self.subTest(record=path.name):
                self.assertIn(path.name, self.readme)

    def test_every_record_is_described_by_device_and_architecture(self):
        for path in self.records:
            with self.subTest(record=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn(record["hardware"]["device_name"], self.readme)
                self.assertIn(record["hardware"]["architecture"], self.readme)


class CapabilityReferenceTests(unittest.TestCase):
    """One capability reference per family, named after the family it describes.

    The reference is family-level because a capability boundary belongs to the
    library and the operation, not to one task: every task in a family probes
    the same surface. Naming the file after the family is what stops a second
    family from quietly inheriting a first family's boundaries.
    """

    def setUp(self):
        self.task = json.loads((TASK_DIR / "task.json").read_text(encoding="utf-8"))
        self.path = ROOT / self.task["capability_reference"]
        self.capabilities = json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_reference_is_named_after_the_family_it_describes(self):
        self.assertEqual(self.path.name, f"{self.task['family']}_capabilities.json")

    def test_the_reference_declares_the_same_family_as_the_task(self):
        self.assertEqual(self.capabilities["family"], self.task["family"])

    def test_the_reference_is_marked_agent_visible(self):
        self.assertEqual(self.capabilities["visibility"], "agent_visible")

    def test_the_reference_does_not_name_the_ops_the_probe_resolves(self):
        """probe_sdpa records which ops answer which configuration.

        Discovering that is the task. A reference that lists the ops hands the
        dispatch decision over before the submission writes a probe.
        """
        probe = load("probe_sdpa", ROOT / "tools" / "probe_sdpa.py")
        blob = self.path.read_text(encoding="utf-8").lower()
        for op in probe.FUSED_LIBRARY_OPS | probe.LIBRARY_COMPOSITION_OPS:
            with self.subTest(op=op):
                self.assertNotIn(op.lower(), blob)

    def test_the_reference_does_not_name_the_installed_component_versions(self):
        """§4.5 forbids version numbers, and the environment record has every one."""
        record = json.loads((ROOT / self.task["target_environment"]["record"]).read_text(encoding="utf-8"))
        blob = self.path.read_text(encoding="utf-8")
        for component in ("driver", "musa_toolkit", "mudnn", "mublas", "driver_commit"):
            version = record["software"].get(component)
            if version:
                with self.subTest(component=component):
                    self.assertNotIn(str(version), blob)

    def test_the_reference_does_not_name_the_libraries(self):
        """§4.5 permits "what kind of capability exists", not which library has it.

        Naming the library collapses the space the probe is supposed to search.
        """
        libraries = (
            "mudnn", "mublas", "torch_musa", "mate", "mutlass", "tilelang",
            "vllm", "paddle", "llama", "flashmla", "flash_attn",
        )
        blob = self.path.read_text(encoding="utf-8").lower()
        for library in libraries:
            with self.subTest(library=library):
                self.assertIsNone(
                    re.search(rf"\b{library}\b", blob),
                    f"{library} appears in the agent-visible capability reference",
                )


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
