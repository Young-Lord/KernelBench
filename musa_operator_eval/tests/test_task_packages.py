"""Structural checks over every task package.

Two rules earn their keep here.

The first is coverage: the attention set decides which entries belong to which
tier, so a set entry that is eligible for a tier and has no package for it is a
row in the ledger that nobody can act on.

The second is that an A-tier reference is a copy. The A tier is the default tier
and an A-tier task *is* a KernelBench problem, so the package carries the problem
rather than restating it. A copy is only safe if drift is detectable, which is
what the byte-for-byte check below is for.
"""

import ast
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
TASKS_DIR = ROOT / "tasks"
SET_PATH = ROOT / "sets" / "attention" / "attention_set.json"

CASE_TAGS = {"smoke", "correctness", "boundary", "non_aligned", "extreme", "perf", "generalization"}
SMOKE_RANGE = (3, 5)
TIERS = ("A_kernel", "B_library")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def packages():
    """Yield (directory, contract) for every package that declares a set entry."""
    for task_json in sorted(TASKS_DIR.glob("*/task.json")):
        yield task_json.parent, json.loads(task_json.read_text(encoding="utf-8"))


def module_level_names(source: str) -> set:
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def all_packages():
    return list(packages())


class SetCoverageTests(unittest.TestCase):
    """The set is the ledger; a package is what makes a row in it actionable."""

    def setUp(self):
        self.entries = json.loads(SET_PATH.read_text(encoding="utf-8"))["entries"]
        self.packaged = {}
        for _, task in all_packages():
            entry = task.get("set_entry")
            if entry:
                self.packaged.setdefault(entry, set()).add(task["tier"])

    def test_every_entry_scoped_package_names_the_entry_it_realises(self):
        """A family-scoped package names the entries it covers instead.

        The pilot generalises the attention core past what any one KernelBench
        entry asks for, so it belongs to the family rather than to an entry, and
        claiming a single entry would misdescribe it.
        """
        for directory, task in all_packages():
            with self.subTest(task=directory.name):
                if task.get("scope") == "family":
                    self.assertIn("covers", task, f"{directory.name} is family-scoped but names no coverage")
                    self.assertTrue(task["covers"]["entries"])
                    self.assertNotIn("set_entry", task)
                else:
                    self.assertIn("set_entry", task, f"{directory.name} does not name a set entry")

    def test_every_named_entry_exists_in_the_set(self):
        known = {entry["entry_id"] for entry in self.entries}
        for entry, _ in self.packaged.items():
            with self.subTest(entry=entry):
                self.assertIn(entry, known)

    def test_every_eligible_entry_has_a_package_for_that_tier(self):
        """This is the check that would have caught ten unbuilt packages.

        The set records an eligibility decision per entry per tier. A decision
        nothing implements is indistinguishable from no decision, so it is
        asserted here rather than left to a reader to notice.
        """
        missing = []
        for entry in self.entries:
            for tier in TIERS:
                if not entry["tier"][tier].get("eligible"):
                    continue
                if tier not in self.packaged.get(entry["entry_id"], set()):
                    missing.append((entry["entry_id"], tier))
        self.assertEqual(missing, [], f"eligible in the set but has no package: {missing}")

    def test_no_package_claims_a_tier_its_entry_is_ineligible_for(self):
        by_entry = {entry["entry_id"]: entry for entry in self.entries}
        for entry_id, tiers in self.packaged.items():
            entry = by_entry[entry_id]
            for tier in tiers:
                with self.subTest(entry=entry_id, tier=tier):
                    self.assertTrue(
                        entry["tier"][tier].get("eligible"),
                        f"{entry_id} is marked ineligible for {tier} but a package claims it",
                    )


class AReferenceIdentityTests(unittest.TestCase):
    """An A-tier reference is the KernelBench problem, byte for byte."""

    def a_tier(self):
        return [(directory, task) for directory, task in all_packages() if task["tier"] == "A_kernel"]

    def test_there_is_at_least_one_a_tier_package(self):
        self.assertTrue(self.a_tier())

    def test_every_a_tier_reference_ends_with_its_source_verbatim(self):
        """The package prepends a provenance header and changes nothing else.

        Retyping a reference is how a copy quietly stops being a copy, and a
        reference that no longer matches the problem it claims to be is worse
        than no package at all.
        """
        for directory, task in self.a_tier():
            source_path = REPO / task["source"]["problem"]
            with self.subTest(task=task["id"]):
                self.assertTrue(source_path.is_file(), f"{source_path} is missing")
                source = source_path.read_text(encoding="utf-8")
                reference = (directory / task["problem_file"]).read_text(encoding="utf-8")
                self.assertTrue(
                    reference.endswith(source.rstrip() + "\n"),
                    f"{task['id']}: problem.py is no longer a copy of {task['source']['problem']}",
                )

    def test_every_a_tier_package_declares_a_source_problem(self):
        for _, task in self.a_tier():
            with self.subTest(task=task["id"]):
                self.assertIn("problem", task["source"])

    def test_no_a_tier_problem_declares_a_library_policy(self):
        """A whitelist is a B-tier concept; its presence in an A-tier problem is a leak."""
        for directory, task in self.a_tier():
            source = (directory / task["problem_file"]).read_text(encoding="utf-8")
            with self.subTest(task=task["id"]):
                self.assertNotIn("LIBRARY_POLICY", source)


class PackageContractTests(unittest.TestCase):
    """Fields every package has to get right, whichever tier it is."""

    def test_every_package_names_a_tier_the_framework_knows(self):
        for directory, task in all_packages():
            with self.subTest(task=directory.name):
                self.assertIn(task["tier"], TIERS)

    def test_every_package_points_at_an_environment_record_that_exists(self):
        for _, task in all_packages():
            target = task["target_environment"]
            with self.subTest(task=task["id"]):
                self.assertTrue((ROOT / target["record"]).is_file())
                record = json.loads((ROOT / target["record"]).read_text(encoding="utf-8"))
                self.assertEqual(target["snapshot_id"], record["snapshot_id"])

    def test_every_package_names_the_problem_file_it_ships(self):
        for directory, task in all_packages():
            with self.subTest(task=task["id"]):
                self.assertTrue((directory / task["problem_file"]).is_file())


class PublicCaseTests(unittest.TestCase):
    """Public cases are the agent's self-test, so their shape is part of the contract."""

    def test_every_package_ships_between_three_and_five_public_cases(self):
        low, high = SMOKE_RANGE
        for directory, task in all_packages():
            cases = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))["cases"]
            with self.subTest(task=task["id"]):
                self.assertGreaterEqual(len(cases), low)
                self.assertLessEqual(len(cases), high)

    def test_every_public_case_tag_comes_from_the_fixed_vocabulary(self):
        """Free-text tags cannot be counted, and classification is what they are for."""
        for directory, task in all_packages():
            cases = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))["cases"]
            for case in cases:
                with self.subTest(task=task["id"], case=case["case_id"]):
                    self.assertIn(case["tag"], CASE_TAGS)

    def test_case_parameters_resolve_for_every_public_case(self):
        runner = load("run_task", ROOT / "tools" / "run_task.py")
        for directory, task in all_packages():
            cases = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))["cases"]
            with self.subTest(task=task["id"]):
                self.assertEqual(runner.verify_case_parameters(cases, task["case_parameters"]), [])

    def test_case_parameters_name_variables_the_problem_defines(self):
        """The driver specialises a case by re-binding a module-level name."""
        for directory, task in all_packages():
            source = (directory / task["problem_file"]).read_text(encoding="utf-8")
            declared = module_level_names(source)
            for path, variable in task["case_parameters"].items():
                with self.subTest(task=task["id"], parameter=path):
                    self.assertIn(variable, declared)

    def test_the_driver_can_specialise_every_case_into_valid_python(self):
        runner = load("run_task", ROOT / "tools" / "run_task.py")
        for directory, task in all_packages():
            _, cases, source = runner.load_task(directory)
            for case in cases["cases"]:
                with self.subTest(task=task["id"], case=case["case_id"]):
                    ast.parse(runner.case_source(source, case, task["case_parameters"]))

    def test_every_public_case_is_deterministic(self):
        for directory, task in all_packages():
            cases = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))["cases"]
            identifiers = [case["case_id"] for case in cases]
            seeds = [case["seed"] for case in cases]
            with self.subTest(task=task["id"]):
                self.assertEqual(len(set(identifiers)), len(identifiers))
                self.assertEqual(len(set(seeds)), len(seeds))


class TierPairTests(unittest.TestCase):
    """An entry that has both tiers must keep the two packages comparable.

    §4.2 of the guide: if the two semantic contracts of one family disagree, the
    goldens and the case generator stop being reusable and the two tiers become
    two incomparable evaluations. The guide states the requirement and leaves it
    to hand discipline; this is the discipline.
    """

    def pairs(self):
        by_entry = {}
        for directory, task in all_packages():
            entry = task.get("set_entry")
            if entry:
                by_entry.setdefault(entry, {})[task["tier"]] = (directory, task)
        for entry, tiers in sorted(by_entry.items()):
            if "A_kernel" in tiers and "B_library" in tiers:
                yield entry, tiers

    def test_there_is_at_least_one_entry_with_both_tiers(self):
        self.assertTrue(list(self.pairs()))

    def test_the_two_tiers_of_an_entry_share_one_semantic_contract(self):
        for entry, tiers in self.pairs():
            documents = []
            for tier in ("A_kernel", "B_library"):
                document = json.loads((tiers[tier][0] / "semantics.json").read_text(encoding="utf-8"))
                documents.append({key: value for key, value in document.items() if key != "note"})
            with self.subTest(entry=entry):
                self.assertEqual(documents[0], documents[1])

    def test_the_two_tiers_of_an_entry_share_one_case_parameterisation(self):
        """Otherwise the same case list means two different things on two tiers."""
        for entry, tiers in self.pairs():
            with self.subTest(entry=entry):
                self.assertEqual(
                    tiers["A_kernel"][1]["case_parameters"],
                    tiers["B_library"][1]["case_parameters"],
                )

    def test_the_two_tiers_of_an_entry_target_the_same_environment(self):
        for entry, tiers in self.pairs():
            with self.subTest(entry=entry):
                self.assertEqual(
                    tiers["A_kernel"][1]["target_environment"]["snapshot_id"],
                    tiers["B_library"][1]["target_environment"]["snapshot_id"],
                )


class LibraryPolicyTests(unittest.TestCase):
    """A B-tier package states its whitelist twice; the two statements must agree."""

    def b_tier(self):
        return [(directory, task) for directory, task in all_packages() if task["tier"] == "B_library"]

    def test_there_is_at_least_one_b_tier_package(self):
        self.assertTrue(self.b_tier())

    def test_the_contract_policy_matches_the_problem_files_library_policy(self):
        for directory, task in self.b_tier():
            tree = ast.parse((directory / task["problem_file"]).read_text(encoding="utf-8"))
            declared = None
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "LIBRARY_POLICY" for target in node.targets
                ):
                    declared = ast.literal_eval(node.value)
            with self.subTest(task=task["id"]):
                self.assertIsNotNone(declared, "the B-tier problem declares no LIBRARY_POLICY")
                self.assertEqual(declared, task["library_policy"])

    def test_every_b_tier_package_declares_a_dispatch_order(self):
        for _, task in self.b_tier():
            with self.subTest(task=task["id"]):
                self.assertEqual(
                    task["library_policy"]["dispatch_order"],
                    ["fused_library", "library_composition", "custom_fallback"],
                )

    def test_every_b_tier_package_declares_its_admission_status(self):
        """An unmeasured task must say so, rather than read as an admitted one."""
        for _, task in self.b_tier():
            with self.subTest(task=task["id"]):
                status = task["admission"]["status"]
                self.assertIn(status, {"pending_device_measurement", "measured"})
                if status == "pending_device_measurement":
                    self.assertTrue(task["admission"]["reason"])

    def test_every_b_tier_capability_reference_is_named_after_its_family(self):
        for _, task in self.b_tier():
            reference = ROOT / task["capability_reference"]
            with self.subTest(task=task["id"]):
                self.assertEqual(reference.name, f"{task['family']}_capabilities.json")
                self.assertEqual(
                    json.loads(reference.read_text(encoding="utf-8"))["family"], task["family"])


class TensorContractTests(unittest.TestCase):
    """The precision a task is graded at is part of its contract, not a CLI default.

    `--precision` used to be a free flag. The A tier was measured at float32 and
    the fused path the B tier exists to exercise does not answer float32, so a run
    at the wrong precision returned a plausible report of a different task's
    behaviour with nothing to indicate it.
    """

    def test_every_package_declares_its_input_dtypes(self):
        for _, task in all_packages():
            contract = task.get("tensor_contract") or {}
            with self.subTest(task=task["id"]):
                self.assertTrue(contract.get("input_dtypes"), "no input_dtypes declared")
                self.assertTrue(contract.get("accumulation_dtype"), "no accumulation_dtype declared")

    def test_declared_dtypes_use_the_drivers_vocabulary(self):
        runner = load("run_task", ROOT / "tools" / "run_task.py")
        known = set(runner.PRECISION_DTYPE_NAMES.values())
        for _, task in all_packages():
            for dtype in task["tensor_contract"]["input_dtypes"]:
                with self.subTest(task=task["id"], dtype=dtype):
                    self.assertIn(dtype, known)

    def test_every_package_permits_at_least_one_precision(self):
        runner = load("run_task", ROOT / "tools" / "run_task.py")
        for _, task in all_packages():
            with self.subTest(task=task["id"]):
                self.assertTrue(runner.allowed_precisions(task))

    def test_the_two_tiers_do_not_share_a_precision_by_accident(self):
        """A is graded in float32 and B in half precision, on the same hardware."""
        for _, task in all_packages():
            permitted = set(task["tensor_contract"]["input_dtypes"])
            with self.subTest(task=task["id"]):
                if task["tier"] == "A_kernel":
                    self.assertEqual(permitted, {"float32"})
                else:
                    self.assertEqual(permitted, {"float16", "bfloat16"})

    def test_the_driver_refuses_a_precision_the_contract_does_not_permit(self):
        """The pilot is a B-tier task; running it at float32 must not start."""
        pilot = TASKS_DIR / "sdpa_forward_pilot"
        submission = ROOT / "private" / "sdpa_forward_b_v0" / "expert" / "model_new.py"
        self.assertTrue(submission.is_file(), "the pilot's expert submission is missing")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "run_task.py"),
             "--task-dir", str(pilot), "--submission", str(submission), "--precision", "fp32"],
            capture_output=True, text=True,
        )
        with self.subTest(exit_code=completed.returncode):
            self.assertEqual(completed.returncode, 2)
            self.assertIn("tensor", completed.stderr)
            self.assertIn("fp32", completed.stderr)


if __name__ == "__main__":
    unittest.main()
