import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "tasks" / "sdpa_forward_pilot" / "starter"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = load("audit", ROOT / "evaluator" / "audit.py")
compare = load("compare", ROOT / "evaluator" / "compare.py")


class EvaluatorTests(unittest.TestCase):
    def test_pristine_starter_passes_audit(self):
        self.assertEqual(audit.audit_submission(STARTER, STARTER), [])

    def test_frozen_modification_fails_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "starter"
            shutil.copytree(STARTER, copy)
            with (copy / "src" / "main.cpp").open("a", encoding="utf-8") as stream:
                stream.write("\n// changed\n")
            self.assertTrue(any("modified frozen file" in error for error in audit.audit_submission(STARTER, copy)))

    def test_forbidden_code_inside_editable_region_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "starter"
            shutil.copytree(STARTER, copy)
            path = copy / "src" / "dispatch.cpp"
            text = path.read_text(encoding="utf-8").replace("// Replace this placeholder", "auto x = dlopen(\"remote.so\", 0);\n  // Replace this placeholder")
            path.write_text(text, encoding="utf-8")
            self.assertTrue(any("dynamic loading" in error for error in audit.audit_submission(STARTER, copy)))

    def test_generated_golden_compares_with_itself(self):
        golden = ROOT / "tasks" / "sdpa_forward_pilot" / "generated" / "smoke_002" / "golden"
        result = compare.compare_directories(golden, golden, atol=0, rtol=0)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
