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
        # The staging call carries the manifest the case came from: the reference's
        # construction depends on the case's parameters, so the tool is handed the
        # case's own entry rather than a seed and a hope.
        self.manifest = self.tmp / "cases.json"
        self.manifest.write_text(json.dumps({"cases": [{"case_id": self.case_id, "seed": 7}]}), encoding="utf-8")
        self.fake = self.tmp / "fake_materializer.py"
        self.fake.write_text(
            "import argparse, hashlib, json, pathlib, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--precision')\n"
            "parser.add_argument('--cases'); parser.add_argument('--case')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            # The stand-in refuses a call that does not identify the case: a seed alone
            # built the problem file's default model, which is the defect this fake is
            # here to keep visible.
            "cases = json.loads(pathlib.Path(args.cases).read_text())['cases']\n"
            "entry = [c for c in cases if c['case_id'] == args.case]\n"
            "if not entry:\n"
            "    raise SystemExit('the tool was called for a case the manifest does not declare')\n"
            "payload = b'\\x02\\x03'\n"
            "path = pathlib.Path(args.output_dir) / 'state_c_attn.weight.bin'\n"
            "path.write_bytes(payload)\n"
            "json.dump({'digest': 'abc', 'dtype': args.precision, 'seed': int(entry[0]['seed']), 'tensors': [\n"
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
            self.task(False), self.task_dir, self.case(), self.manifest, self.generated, self.tmp / "stage", "fp16"
        )
        self.assertEqual(directory, self.case_input)
        self.assertFalse(block["materialized"])
        self.assertFalse((self.tmp / "stage").exists())

    def test_the_state_is_staged_beside_the_case_inputs_and_the_generated_tree_is_untouched(self):
        directory, block = RUN_BINARY.stage_case_inputs(
            self.task(True), self.task_dir, self.case(), self.manifest, self.generated, self.tmp / "stage", "fp16"
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

    def test_a_stateless_reference_with_arguments_is_staged_for_its_configuration(self):
        """No state, but the construction still reads the case's own arguments.

        A torch-free submission cannot recover `n_head` or `max_seqlen` from the input
        tensors, and a case re-binds the constants the construction reads, so the
        contract declares them and the evaluator stages them as `case.json` beside the
        case's tensors. This is the pilot's shape: no weights, four arguments.
        """
        self.fake.write_text(
            "import argparse, json, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--precision')\n"
            "parser.add_argument('--cases'); parser.add_argument('--case')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            "json.dump({'digest': 'd', 'case_id': args.case, 'tensors': [],\n"
            "           'init_inputs': [768, 4, 0.0, 0.0, 256]}, sys.stdout)\n",
            encoding="utf-8",
        )
        task = self.task(False)
        task["starter"]["cpp"]["case_configuration"] = {
            "source": "problem.py:get_init_inputs()",
            "args": ["n_embd", "n_head", "attn_pdrop", "resid_pdrop", "max_seqlen"],
        }
        directory, block = RUN_BINARY.stage_case_inputs(
            task, self.task_dir, self.case(), self.manifest, self.generated, self.tmp / "stage", "fp16"
        )
        self.assertNotEqual(directory, self.case_input)
        self.assertFalse(block["materialized"])
        configuration = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        self.assertEqual(configuration["configuration"]["n_head"], 4)
        self.assertEqual(configuration["configuration"]["max_seqlen"], 256)
        # The case's own tensors are there, and no state tensor is claimed for a reference
        # that carries none.
        manifest = json.loads((directory / "tensors.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["name"] for entry in manifest["tensors"]], ["input_0"])

    def test_a_contract_declaring_the_wrong_number_of_arguments_is_refused(self):
        """The names and the values have to line up, or a submission is handed the wrong configuration."""
        self.fake.write_text(
            "import argparse, json, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--precision')\n"
            "parser.add_argument('--cases'); parser.add_argument('--case')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            "json.dump({'digest': 'd', 'tensors': [], 'init_inputs': [768]}, sys.stdout)\n",
            encoding="utf-8",
        )
        task = self.task(False)
        task["starter"]["cpp"]["case_configuration"] = {
            "source": "problem.py:get_init_inputs()",
            "args": ["n_embd", "n_head"],
        }
        with self.assertRaises(RUN_BINARY.StagingError) as raised:
            RUN_BINARY.stage_case_inputs(
                task, self.task_dir, self.case(), self.manifest, self.generated, self.tmp / "stage", "fp16"
            )
        self.assertIn("construction arguments", str(raised.exception))

    def test_a_failing_materializer_is_a_staging_error(self):
        self.fake.write_text("import sys\nsys.stderr.write('no torch here')\nraise SystemExit(3)\n", encoding="utf-8")
        with self.assertRaises(RUN_BINARY.StagingError) as raised:
            RUN_BINARY.stage_case_inputs(
                self.task(True), self.task_dir, self.case(), self.manifest, self.generated, self.tmp / "stage", "fp16"
            )
        self.assertIn("no torch here", str(raised.exception))

    def test_a_name_outside_the_declared_prefix_is_refused(self):
        """The contract says where the tensors are, and a tool writing elsewhere is a bug."""
        self.fake.write_text(
            "import argparse, json, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task-dir'); parser.add_argument('--precision')\n"
            "parser.add_argument('--cases'); parser.add_argument('--case')\n"
            "parser.add_argument('--output-dir')\n"
            "args = parser.parse_args()\n"
            "json.dump({'digest': 'd', 'tensors': [{'name': 'weights.w', 'file': 'w.bin', 'dtype': 'float16',\n"
            "    'shape': [1], 'nbytes': 2, 'sha256': 'x'}]}, sys.stdout)\n",
            encoding="utf-8",
        )
        with self.assertRaises(RUN_BINARY.StagingError) as raised:
            RUN_BINARY.stage_case_inputs(
                self.task(True), self.task_dir, self.case(), self.manifest, self.generated, self.tmp / "stage", "fp16"
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
            "parser.add_argument('--task-dir'); parser.add_argument('--precision')\n"
            "parser.add_argument('--cases'); parser.add_argument('--case')\n"
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
                self.manifest,
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

    def framework_state(self, task_dir: Path, case: dict, task: dict, problem_source: str, dtype):
        """The reference's state for one case, built through the framework's own loader.

        The source is the *case-specialized* one, because that is what the grader and the
        golden generator both run: a case re-binds module constants (`case_parameters`),
        and for `kb_l3_43_b` those constants are `n_embd`/`n_head`/`max_seqlen`, so the
        raw `problem.py` builds a different model with differently shaped weights.
        """
        import sys as _sys

        _sys.path.insert(0, str(ROOT.parent / "src"))
        _sys.path.insert(0, str(ROOT))
        from kernelbench.eval import load_original_model_and_inputs, set_seed
        from tools.case_specialization import case_source

        specialized = case_source(problem_source, case, task.get("case_parameters") or {})
        Model, get_init_inputs, _get_inputs = load_original_model_and_inputs(specialized, {})
        set_seed(int(case["seed"]))
        init_inputs = get_init_inputs()
        set_seed(int(case["seed"]))
        with torch.no_grad():
            model = Model(*init_inputs).to(dtype=dtype)
        return model.state_dict()

    def framework_state_defaults(self, task_dir: Path, dtype):
        """The reference built from `problem.py` as written, for the questions that do not depend on a case."""
        import sys as _sys

        _sys.path.insert(0, str(ROOT.parent / "src"))
        from kernelbench.eval import load_original_model_and_inputs

        source = (task_dir / "problem.py").read_text(encoding="utf-8")
        Model, get_init_inputs, _get_inputs = load_original_model_and_inputs(source, {})
        with torch.no_grad():
            model = Model(*get_init_inputs()).to(dtype=dtype)
        return model.state_dict()

    def case_and_task(self, task_dir: Path, package_id: str, case_id: str):
        from tools.case_specialization import load_task

        manifest = json.loads((ROOT / "private" / package_id / "cases.private.json").read_text(encoding="utf-8"))
        case = [entry for entry in manifest["cases"] if entry["case_id"] == case_id][0]
        task, _cases, problem_source = load_task(task_dir)
        return case, task, problem_source

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
        case, task, source = self.case_and_task(task_dir, "mingpt_causal_attention_b_v0", "hidden_perf_002")
        built, _init = MATERIALIZE.build_state(task_dir, case, task, source, dtype)
        framework = self.framework_state(task_dir, case, task, source, dtype)
        self.assertEqual(sorted(built), sorted(framework))
        self.assertEqual(
            MATERIALIZE.state_digest(self.records_for(built)),
            MATERIALIZE.state_digest(self.records_for(framework)),
        )

    def test_the_state_is_built_for_the_case_and_not_for_the_problem_s_defaults(self):
        """The defect this pair of tests exists for, kept as a test rather than a note.

        `kb_l3_43_b`'s cases re-bind `n_embd`, so a case with a different width needs a
        differently shaped state. Building the raw `problem.py` (the problem's defaults)
        handed eight of ten cases a state belonging to another model -- 2304x768 where
        the case needs 192x64 -- and nothing in the pipeline could tell, because the
        state was reported as materialized with a digest of its own.
        """
        task_dir = self.TASKS / "kb_l3_43_b"
        dtype = torch.float16
        case, task, source = self.case_and_task(task_dir, "mingpt_causal_attention_b_v0", "hidden_extreme_001")
        self.assertEqual(case["shape"]["C"], 64, "the chosen case has to differ from problem.py's n_embd")
        default_case = dict(case)
        default_case["shape"] = {**case["shape"], "C": 768, "H": 8, "S_max": 1024}
        default_case["case_id"] = "the_problems_own_configuration"
        built, init_inputs = MATERIALIZE.build_state(task_dir, case, task, source, dtype)
        defaults, _ = MATERIALIZE.build_state(task_dir, default_case, task, source, dtype)
        self.assertEqual(init_inputs[0], 64, "the case's own n_embd is what the state is built from")
        self.assertNotEqual(
            {key: tuple(value.shape) for key, value in built.items()},
            {key: tuple(value.shape) for key, value in defaults.items()},
            "the case's state and the problem's defaults have the same shapes, so the case's parameters "
            "are not reaching the construction",
        )
        framework = self.framework_state(task_dir, case, task, source, dtype)
        self.assertEqual(
            MATERIALIZE.state_digest(self.records_for(built)),
            MATERIALIZE.state_digest(self.records_for(framework)),
        )
        # And the old behaviour is asserted to be wrong: a state built from the problem
        # file's defaults is not the state this case's golden came from.
        self.assertNotEqual(
            MATERIALIZE.state_digest(self.records_for(defaults)),
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
            state = self.framework_state_defaults(directory, torch.float32)
            with self.subTest(task=task["id"]):
                self.assertEqual(
                    declared,
                    len(state) > 0,
                    f"{task['id']} declares model_state={declared} and its reference holds {len(state)} tensors",
                )

    def test_the_case_seed_is_what_makes_the_state(self):
        """Two cases of one task differ in their weights, which is why state is per case."""
        task_dir = self.TASKS / "kb_l3_43_b"
        case, task, source = self.case_and_task(task_dir, "mingpt_causal_attention_b_v0", "hidden_perf_002")
        other = dict(case)
        other["case_id"] = "the_same_configuration_with_another_seed"
        other["seed"] = int(case["seed"]) + 1
        first, _ = MATERIALIZE.build_state(task_dir, case, task, source, torch.float16)
        second, _ = MATERIALIZE.build_state(task_dir, other, task, source, torch.float16)
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
