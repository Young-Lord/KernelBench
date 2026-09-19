import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("musa_eval_validate", ROOT / "tools" / "validate.py")
validate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate)


class ValidationTests(unittest.TestCase):
    def test_all_valid_fixtures_pass(self):
        for path in sorted((ROOT / "fixtures" / "valid").glob("*.json")):
            with self.subTest(path=path.name):
                self.assertEqual(validate.validate_document(path, path.name.split(".")[0]), [])

    def test_all_invalid_fixtures_fail(self):
        for path in sorted((ROOT / "fixtures" / "invalid").glob("*.json")):
            with self.subTest(path=path.name):
                self.assertTrue(validate.validate_document(path, path.name.split(".")[0]))

    def test_all_schemas_are_json_and_versioned(self):
        paths = sorted((ROOT / "schemas").glob("*.schema.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertIn("1.0.0", schema["$id"])


if __name__ == "__main__":
    unittest.main()
