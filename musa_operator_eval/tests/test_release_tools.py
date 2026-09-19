import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


private_manifest = load("private_manifest", ROOT / "tools" / "make_private_manifest.py")
candidate_gate = load("candidate_gate", ROOT / "tools" / "candidate_gate.py")
packager = load("packager", ROOT / "tools" / "package_public_task.py")
validator = load("validator_release", ROOT / "tools" / "validate.py")
disclosure_audit = load("disclosure_audit", ROOT / "tools" / "audit_agent_package.py")


class ReleaseToolTests(unittest.TestCase):
    def test_private_manifest_requires_measured_gap_cardinality(self):
        with self.assertRaises(ValueError):
            private_manifest.build_manifest({"gaps": []})

    def test_private_manifest_with_three_gaps_and_two_reasons_validates(self):
        evidence = {"gaps": [
            {"case_id": "hidden_gap_001", "reason": "unsupported_shape", "expected_path": "custom_fallback", "evidence": "runtime probe rejected D=64"},
            {"case_id": "hidden_gap_002", "reason": "unsupported_shape", "expected_path": "custom_fallback", "evidence": "runtime probe rejected D=72"},
            {"case_id": "hidden_gap_003", "reason": "semantic_mismatch", "expected_path": "library_composition", "evidence": "GQA result differs from contract"}
        ]}
        manifest = private_manifest.build_manifest(evidence)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.private.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validator.validate_document(path, "cases"), [])

    def test_candidate_gate_admits_and_rejects(self):
        base = {"scoring": {"minimum_naive_to_expert_ratio": 1.3}, "results": [
            {"case_id": "p1", "implementation": "naive_library_composition", "status": "pass", "latency_ms": 2.0},
            {"case_id": "p1", "implementation": "expert_dispatch", "status": "pass", "latency_ms": 1.0}
        ]}
        self.assertEqual(candidate_gate.evaluate(base)["decision"], "admit")
        base["results"][0]["latency_ms"] = 1.1
        self.assertEqual(candidate_gate.evaluate(base)["decision"], "reject")

    def test_public_package_excludes_maintainer_assets(self):
        task = ROOT / "tasks" / "sdpa_forward_pilot"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "package"
            packager.package(task, output)
            self.assertTrue((output / "task.json").is_file())
            self.assertTrue((output / "PROMPT.md").is_file())
            self.assertTrue((output / "reference" / "sdpa_reference.py").is_file())
            self.assertTrue((output / "agent_reference" / "attention_capabilities.json").is_file())
            self.assertFalse((output / "capability_plan.json").exists())
            self.assertFalse((output / "private").exists())
            self.assertEqual(disclosure_audit.audit(output), [])

    def test_agent_reference_validates(self):
        path = ROOT / "agent_reference" / "attention_capabilities.json"
        self.assertEqual(validator.validate_document(path, "agent_reference"), [])

    def test_disclosure_audit_rejects_source_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "agent_reference").mkdir()
            source = ROOT / "agent_reference" / "attention_capabilities.json"
            (package / "agent_reference" / "attention_capabilities.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            (package / "leak.md").write_text("https://github.com/example/private", encoding="utf-8")
            self.assertTrue(any("repository URL" in error for error in disclosure_audit.audit(package)))


if __name__ == "__main__":
    unittest.main()
