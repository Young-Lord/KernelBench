import json
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "tasks" / "sdpa_forward_pilot" / "starter"


class StarterTests(unittest.TestCase):
    def test_manifest_paths_and_markers_exist(self):
        manifest = json.loads((STARTER / "starter_manifest.json").read_text(encoding="utf-8"))
        for relative in manifest["frozen_files"]:
            self.assertTrue((STARTER / relative).is_file(), relative)
        for item in manifest["editable_files"]:
            text = (STARTER / item["path"]).read_text(encoding="utf-8")
            for region in item["regions"]:
                self.assertEqual(text.count(f"BEGIN_AGENT_EDIT:{region}"), 1)
                self.assertEqual(text.count(f"END_AGENT_EDIT:{region}"), 1)

    @unittest.skipUnless(shutil.which("g++") or shutil.which("clang++"), "C++ compiler unavailable")
    def test_host_io_round_trip(self):
        source = ROOT / "tasks" / "sdpa_forward_pilot" / "generated" / "smoke_001" / "input"
        temp = ROOT.parent / "scratch" / f"starter_test_{uuid.uuid4().hex}"
        try:
            (temp / "output").mkdir(parents=True)
            build_relative = (temp / "build").relative_to(ROOT.parent)
            source_relative = source.relative_to(ROOT.parent)
            output_relative = (temp / "output").relative_to(ROOT.parent)
            subprocess.run(["python", str(STARTER.relative_to(ROOT.parent) / "build.py"), "--host-only", "--output-dir", str(build_relative)], cwd=ROOT.parent, check=True)
            executable = next((temp / "build").glob("host_io_smoke*"))
            subprocess.run([str(executable.relative_to(ROOT.parent)), str(source_relative), str(output_relative), "smoke_001"], cwd=ROOT.parent, check=True)
            before = json.loads((source / "tensors.json").read_text(encoding="utf-8"))
            after = json.loads((temp / "output" / "tensors.json").read_text(encoding="utf-8"))
            self.assertEqual([x["sha256"] for x in before["tensors"]], [x["sha256"] for x in after["tensors"]])
        finally:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
