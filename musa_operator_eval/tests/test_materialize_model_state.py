"""The tool that lets a torch-free submission see a reference's weights.

§4.6's compiled path grades a case out of its input directory, and a reference that
builds weights in `__init__` keeps them out of `get_inputs()`. This tool materializes
them with the framework's own construction, so the submission reads them like any
other tensor. The tests here are the ones that do not need torch: the naming, the
digest, and what the driver does with what the tool prints. The construction itself is
checked on the device, where the framework and a GPU exist.

Run with:
    python musa_operator_eval/tests/test_materialize_model_state.py -v
"""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "materialize_model_state", ROOT / "evaluator" / "materialize_model_state.py"
)
MATERIALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATERIALIZE)

RUN_BINARY_SPEC = importlib.util.spec_from_file_location("musa_run_binary", ROOT / "evaluator" / "run_binary.py")
RUN_BINARY = importlib.util.module_from_spec(RUN_BINARY_SPEC)
RUN_BINARY_SPEC.loader.exec_module(RUN_BINARY)


def record(name: str, payload: bytes) -> dict:
    import hashlib

    return {"name": name, "file": f"{name}.bin", "dtype": "float16", "shape": [1], "sha256": hashlib.sha256(payload).hexdigest()}


class NamingTests(unittest.TestCase):
    """The names are the interface: an agent reads them, so they are the module's own keys."""

    def test_the_state_dict_key_is_kept_verbatim_behind_a_prefix(self):
        self.assertEqual(MATERIALIZE.state_tensor_name("c_attn.weight"), "state_c_attn.weight")
        self.assertEqual(MATERIALIZE.state_file_name("c_attn.weight"), "state_c_attn.weight.bin")

    def test_the_prefix_cannot_collide_with_a_case_input(self):
        """A case's tensors are `input_0..n` (or the pilot's `lse`), never `state_*`."""
        self.assertFalse(MATERIALIZE.state_tensor_name("weight").startswith("input_"))
        self.assertNotIn(MATERIALIZE.state_tensor_name("lse"), ("lse", "input_0"))


class DigestTests(unittest.TestCase):
    """One digest over the whole state, because a drift must be nameable."""

    def test_the_digest_is_order_independent_and_value_sensitive(self):
        first = record("state_a", b"\x01\x02")
        second = record("state_b", b"\x03\x04")
        self.assertEqual(MATERIALIZE.state_digest([first, second]), MATERIALIZE.state_digest([second, first]))
        changed = record("state_b", b"\x03\x05")
        self.assertNotEqual(MATERIALIZE.state_digest([first, second]), MATERIALIZE.state_digest([first, changed]))

    def test_an_empty_state_has_a_digest_too(self):
        """A stateless reference still records one, so a report is uniform."""
        self.assertEqual(MATERIALIZE.state_digest([]), MATERIALIZE.state_digest([]))


class StagingTests(unittest.TestCase):
    """What the driver does with the tool's report, driven through a stand-in materializer."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.generated = self.tmp / "generated"
        self.case_id = "hidden_001"
        self.case_input = self.generated / self.case_id / "input"
        self.case_input.mkdir(parents=True)
        (self.case_input / "input_0.bin").write_bytes(b"\x00\x01")
        (self.case_input / "tensors.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "case_id": self.case_id,
                    "tensors": [
                        {"name": "input_0", "file": "input_0.bin", "dtype": "float16", "shape": [1], "nbytes": 2}
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.task_dir = self.tmp / "task"
        self.task_dir.mkdir()
        self.fake = self.tmp / "fake_materializer.py"
        self.fake.write_text(
            "import argparse, hashlib, json, pathlib, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--seed'); parser.add_argument('--precision')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            "payload = b'\\x02\\x03'\n"
            "path = pathlib.Path(args.output_dir) / 'state_c_attn.weight.bin'\n"
            "path.write_bytes(payload)\n"
            "json.dump({'digest': 'abc', 'dtype': args.precision, 'seed': int(args.seed), 'tensors': [\n"
            "    {'name': 'state_c_attn.weight', 'file': 'state_c_attn.weight.bin', 'dtype': 'float16',\n"
            "     'shape': [1], 'nbytes': 2, 'sha256': hashlib.sha256(payload).hexdigest()}]}, sys.stdout)\n",
            encoding="utf-8",
        )
        self.original = RUN_BINARY.MATERIALIZE_TOOL
        RUN_BINARY.MATERIALIZE_TOOL = self.fake
        self.addCleanup(setattr, RUN_BINARY, "MATERIALIZE_TOOL", self.original)

    def task(self, declared: bool) -> dict:
        cpp = {"model_state": {"source": "problem.py:Model.state_dict()", "materialized_by": "x", "tensor_prefix": "state_"}}
        if not declared:
            cpp["model_state"] = None
        return {"id": "fixture", "tier": "B_library", "starter": {"cpp": cpp}}

    def case(self) -> dict:
        return {"case_id": self.case_id, "seed": 7}

    def test_a_stateless_contract_is_left_alone(self):
        """No declaration means the case's own input directory, unmaterialized and uncopied."""
        directory, block = RUN_BINARY.stage_case_inputs(
            self.task(False), self.task_dir, self.case(), self.generated, self.tmp / "stage", "fp16"
        )
        self.assertEqual(directory, self.case_input)
        self.assertFalse(block["materialized"])
        self.assertFalse((self.tmp / "stage").exists())

    def test_the_state_is_staged_beside_the_case_inputs_and_the_generated_tree_is_untouched(self):
        directory, block = RUN_BINARY.stage_case_inputs(
            self.task(True), self.task_dir, self.case(), self.generated, self.tmp / "stage", "fp16"
        )
        self.assertNotEqual(directory, self.case_input)
        self.assertTrue(block["materialized"])
        self.assertEqual(block["tensors"], 1)
        self.assertEqual(block["digest"], "abc")
        self.assertEqual(block["tensor_prefix"], "state_")
        manifest = json.loads((directory / "tensors.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["name"] for entry in manifest["tensors"]], ["input_0", "state_c_attn.weight"])
        self.assertTrue((directory / "input_0.bin").is_file())
        self.assertTrue((directory / "state_c_attn.weight.bin").is_file())
        # The case's own directory is read, never written.
        self.assertEqual(
            [entry["name"] for entry in json.loads((self.case_input / "tensors.json").read_text())["tensors"]],
            ["input_0"],
        )
        self.assertFalse((self.case_input / "state_c_attn.weight.bin").exists())

    def test_a_failing_materializer_is_a_staging_error(self):
        self.fake.write_text("import sys\nsys.stderr.write('no torch here')\nraise SystemExit(3)\n", encoding="utf-8")
        with self.assertRaises(RUN_BINARY.StagingError) as raised:
            RUN_BINARY.stage_case_inputs(
                self.task(True), self.task_dir, self.case(), self.generated, self.tmp / "stage", "fp16"
            )
        self.assertIn("no torch here", str(raised.exception))

    def test_a_name_outside_the_declared_prefix_is_refused(self):
        """The contract says where the tensors are, and a tool writing elsewhere is a bug."""
        self.fake.write_text(
            "import argparse, json, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--seed'); parser.add_argument('--precision')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            "json.dump({'digest': 'd', 'tensors': [{'name': 'weights.w', 'file': 'w.bin', 'dtype': 'float16',\n"
            "    'shape': [1], 'nbytes': 2, 'sha256': 'x'}]}, sys.stdout)\n",
            encoding="utf-8",
        )
        with self.assertRaises(RUN_BINARY.StagingError) as raised:
            RUN_BINARY.stage_case_inputs(
                self.task(True), self.task_dir, self.case(), self.generated, self.tmp / "stage", "fp16"
            )
        self.assertIn("state_", str(raised.exception))

    def test_a_name_that_collides_with_a_case_tensor_is_refused(self):
        """Two tensors with one name in a manifest is a case that cannot be read, so it stops here.

        The contract is deliberately mis-declared for this: it says the state's names carry the
        case's own prefix, which is exactly the collision the check exists for.
        """
        self.fake.write_text(
            "import argparse, json, pathlib, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--seed'); parser.add_argument('--precision')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            "pathlib.Path(args.output_dir, 'input_0.bin').write_bytes(b'\\x09')\n"
            "json.dump({'digest': 'd', 'tensors': [{'name': 'input_0', 'file': 'input_0.bin', 'dtype': 'float16',\n"
            "    'shape': [1], 'nbytes': 1, 'sha256': 'x'}]}, sys.stdout)\n",
            encoding="utf-8",
        )
        with self.assertRaises(RUN_BINARY.StagingError) as raised:
            RUN_BINARY.stage_case_inputs(
                {"id": "f", "tier": "B_library", "starter": {"cpp": {"model_state": {"tensor_prefix": "input_"}}}},
                self.task_dir,
                self.case(),
                self.generated,
                self.tmp / "stage",
                "fp16",
            )
        self.assertIn("collides", str(raised.exception))


try:
    import torch

    _TORCH = True
except ImportError:  # the device has it; a laptop without a GPU stack does not
    torch = None
    _TORCH = False


@unittest.skipUnless(_TORCH, "requires torch: the construction under test is the framework's")
class DeviceConstructionTests(unittest.TestCase):
    """The tool is only useful if its state is the state the graded model holds.

    The independent side of the comparison is the framework's own loader: it compiles
    `problem.py` into a context and hands back `Model`, which is how the Python path
    builds the reference. The tool imports the same file as a module. If those two ever
    disagree -- a different module namespace, a changed seeding order -- the digests
    differ and this fails, rather than the disagreement surfacing as a wrong number in
    a report that reads like a broken kernel.
    """

    TASKS = ROOT / "tasks"

    def framework_state(self, task_dir: Path, seed: int, dtype):
        """The reference's state, built through the framework's loader."""
        import sys as _sys

        _sys.path.insert(0, str(ROOT.parent / "src"))
        from kernelbench.eval import load_original_model_and_inputs, set_seed

        source = (task_dir / "problem.py").read_text(encoding="utf-8")
        Model, get_init_inputs, _get_inputs = load_original_model_and_inputs(source, {})
        set_seed(int(seed))
        init_inputs = get_init_inputs()
        set_seed(int(seed))
        with torch.no_grad():
            model = Model(*init_inputs).to(dtype=dtype)
        return model.state_dict()

    def records_for(self, state) -> list:
        import hashlib

        records = []
        for key in sorted(state):
            payload = MATERIALIZE.raw_bytes(state[key])
            records.append(
                {
                    "name": MATERIALIZE.state_tensor_name(key),
                    "file": MATERIALIZE.state_file_name(key),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        return records

    def test_the_tool_matches_the_framework_on_a_weight_bearing_reference(self):
        task_dir = self.TASKS / "kb_l3_43_b"
        dtype = torch.float16
        built = MATERIALIZE.build_state(task_dir, 91001, dtype)
        framework = self.framework_state(task_dir, 91001, dtype)
        self.assertEqual(sorted(built), sorted(framework))
        self.assertEqual(
            MATERIALIZE.state_digest(self.records_for(built)),
            MATERIALIZE.state_digest(self.records_for(framework)),
        )

    def test_the_declaration_matches_whether_the_reference_has_state(self):
        for directory in sorted(self.TASKS.iterdir()):
            task_path = directory / "task.json"
            if not task_path.is_file():
                continue
            task = json.loads(task_path.read_text(encoding="utf-8"))
            declared = bool((task.get("starter") or {}).get("cpp", {}).get("model_state"))
            # Any seed will do here: the question is whether the reference has state at all.
            state = self.framework_state(directory, 42, torch.float32)
            with self.subTest(task=task["id"]):
                self.assertEqual(
                    declared,
                    len(state) > 0,
                    f"{task['id']} declares model_state={declared} and its reference holds {len(state)} tensors",
                )

    def test_the_case_seed_is_what_makes_the_state(self):
        """Two cases of one task differ in their weights, which is why state is per case."""
        task_dir = self.TASKS / "kb_l3_43_b"
        first = MATERIALIZE.build_state(task_dir, 91001, torch.float16)
        second = MATERIALIZE.build_state(task_dir, 91002, torch.float16)
        self.assertNotEqual(
            MATERIALIZE.state_digest(self.records_for(first)),
            MATERIALIZE.state_digest(self.records_for(second)),
        )


class DeclarationTests(unittest.TestCase):
    """The contract has to say which packages the compiled path hands state to."""

    def test_every_task_declares_the_field_and_the_tool_it_names_exists(self):
        tasks = sorted((ROOT / "tasks").iterdir())
        declared = []
        for directory in tasks:
            task_path = directory / "task.json"
            if not task_path.is_file():
                continue
            task = json.loads(task_path.read_text(encoding="utf-8"))
            cpp = (task.get("starter") or {}).get("cpp") or {}
            self.assertIn("model_state", cpp, f"{task['id']} does not say whether the compiled path gets state")
            if cpp["model_state"]:
                declared.append(task["id"])
                tool = ROOT / cpp["model_state"]["materialized_by"]
                self.assertTrue(tool.is_file(), f"{task['id']} names {tool}, which is not there")
                self.assertEqual(cpp["model_state"]["tensor_prefix"], "state_")
        self.assertEqual(len(declared), 8, f"the weight-bearing packages are eight, and this found {declared}")


if __name__ == "__main__":
    unittest.main()
