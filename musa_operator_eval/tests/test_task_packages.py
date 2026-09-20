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
import re
import subprocess
import sys
import tempfile
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


class PromptTests(unittest.TestCase):
    """§4.5: the prompt is what the model reads, so its rules are a property of the package.

    Two of them are worth a machine check. The first is the tier direction: §4.5
    says the `必须做` and `禁止做` lists point in opposite directions between the
    tiers and that getting one backwards is worse than leaving it out, because a
    B-tier prompt that forbids the library makes the task unsolvable rather than
    merely underspecified. The second is the disclosure rule: the prompt must not
    hand the model the upstream project, a library function name, or a version
    string, since any of those turns a measurement of the dispatch decision into a
    lookup.

    The budget is checked too, because it is stated twice -- here and in the
    contract -- and a model that budgets against the wrong number is being told
    something the evaluator will not honour.
    """

    #: §4.5's seven sections, in the guide's own words.
    REQUIRED_SECTIONS = ("目标", "交付物", "必须做", "禁止做", "工程提示", "评分口径", "尝试预算")

    #: §4.5 says this must not appear in `工程提示`. Function names are matched
    #: without `muDNN`, which the A tier is required to *forbid* by name and which
    #: therefore appears legitimately in a different section.
    DISCLOSURES = {
        "the upstream project": r"KernelBench|ScalingIntelligence",
        "a library function name": r"scaled_dot_product_attention|MultiheadAttention|mudnn_\w+",
        "a version string": r"\b\d+\.\d+\.\d+\b",
    }

    def prompts(self):
        for path in sorted(TASKS_DIR.glob("*/PROMPT.md")):
            yield path, json.loads((path.parent / "task.json").read_text(encoding="utf-8"))

    @staticmethod
    def sections(text: str) -> dict:
        parts = re.split(r"^##\s+", text, flags=re.M)
        return {part.splitlines()[0].strip(): part for part in parts[1:]}

    def test_every_package_ships_a_prompt(self):
        packaged = {directory.name for directory, _ in all_packages()}
        prompted = {path.parent.name for path, _ in self.prompts()}
        self.assertEqual(packaged, prompted, "a package has no PROMPT.md")

    def test_every_prompt_carries_the_seven_sections(self):
        for path, _ in self.prompts():
            headings = list(self.sections(path.read_text(encoding="utf-8")))
            with self.subTest(prompt=path.parent.name):
                for required in self.REQUIRED_SECTIONS:
                    self.assertTrue(
                        any(required in heading for heading in headings),
                        f"{path.parent.name} has no {required} section; it has {headings}",
                    )

    def test_the_tier_direction_is_not_reversed(self):
        for path, task in self.prompts():
            body = self.sections(path.read_text(encoding="utf-8"))
            required = next((v for k, v in body.items() if "必须做" in k), "")
            forbidden = next((v for k, v in body.items() if "禁止做" in k), "")
            with self.subTest(prompt=path.parent.name):
                if task["tier"] == "A_kernel":
                    # The A tier may not reach for the library at all.
                    self.assertRegex(forbidden, r"muDNN|muBLAS|SDPA|ATen")
                else:
                    # The B tier may not solve it without the library.
                    self.assertRegex(required, r"库|fused_library")
                    self.assertRegex(
                        forbidden, r"降级|朴素实现",
                        "§4.5 lists '全局降级成朴素实现' under the B tier's prohibitions, because "
                        "a submission that never calls the library can still match a "
                        "custom_fallback trace while scoring nothing",
                    )

    def test_the_stated_budget_is_the_contract_budget(self):
        for path, task in self.prompts():
            text = path.read_text(encoding="utf-8")
            stated = re.search(r"最多\s*(\d+)\s*次.*?最长\s*(\d+)\s*秒", text, re.S)
            with self.subTest(prompt=path.parent.name):
                self.assertIsNotNone(stated, f"{path.parent.name} states no budget")
                budget = task["attempt_budget"]
                self.assertEqual(
                    (int(stated.group(1)), int(stated.group(2))),
                    (budget["max_attempts"], budget["wall_time_seconds"]),
                )

    def test_the_engineering_hints_disclose_nothing(self):
        for path, _ in self.prompts():
            body = self.sections(path.read_text(encoding="utf-8"))
            hints = next((v for k, v in body.items() if "工程提示" in k), "")
            for label, pattern in self.DISCLOSURES.items():
                with self.subTest(prompt=path.parent.name, disclosure=label):
                    self.assertEqual(re.findall(pattern, hints), [], f"{path.parent.name} names {label}")

    def test_no_prompt_names_the_upstream_project(self):
        """`kernelbench.musa_extension` is the ABI and has to be named; the project does not.

        The capability references list `repository_name` as a prohibited disclosure,
        and a prompt that says which project a problem came from invites looking the
        reference implementation up instead of reading the contract.
        """
        for path, _ in self.prompts():
            text = path.read_text(encoding="utf-8")
            bare = re.findall(r"KernelBench(?!\.musa_extension)", text)
            with self.subTest(prompt=path.parent.name):
                self.assertEqual(bare, [])


class CapabilityReferenceTests(unittest.TestCase):
    """The family capability references are agent-facing documents, so they are checked.

    They are read by the agent, not by code, which is why nothing noticed when they
    drifted: the newest one gained `semantic_summary` and the other never did, one
    of them carried a relative path that resolved to nothing, and one was left
    describing a family no task asks about. A document whose reader is a person or
    an agent needs the same treatment as a field whose reader is a function.

    §4.2 of the guide is what fixes the content: three facts have to be written down
    rather than left to be inferred, because a golden produced under the wrong one
    of them fails in a way that is hard to trace back. They are stated once per
    family, which is what locks them for every task in it.
    """

    #: §4.2's three mandatory declarations, as the references spell them.
    MANDATORY_SEMANTIC_KEYS = (
        "causal_alignment",       # top_left vs bottom_right for S_q != S_kv
        "window_interval",        # window_size = (left, right), visible j in [i-left, i+right]
        "window_unlimited_value",  # what a direction means when it is unbounded
        "fully_masked_output",    # what a row with no visible key produces
    )

    def references(self):
        for path in sorted((ROOT / "agent_reference").glob("*_capabilities.json")):
            yield path, json.loads(path.read_text(encoding="utf-8"))

    def test_every_b_tier_family_has_a_reference(self):
        families = {task["family"] for _, task in all_packages() if task["tier"] == "B_library"}
        named = {document["family"] for _, document in self.references()}
        self.assertLessEqual(families, named, f"a B-tier family has no reference: {families - named}")

    def test_every_reference_is_claimed_by_a_task(self):
        """An agent-visible reference nothing points at reads as though something does."""
        claimed = {task["family"] for _, task in all_packages() if task["tier"] == "B_library"}
        for path, document in self.references():
            with self.subTest(reference=path.name):
                self.assertIn(document["family"], claimed)

    def test_every_reference_has_the_same_shape(self):
        """Two references of the same kind drifting apart is how one of them goes stale."""
        shapes = {path.name: tuple(document) for path, document in self.references()}
        distinct = set(shapes.values())
        self.assertEqual(len(distinct), 1, f"capability references disagree on their fields: {shapes}")

    def test_every_reference_declares_the_mandatory_semantics(self):
        for path, document in self.references():
            summary = document.get("semantic_summary", {})
            missing = [key for key in self.MANDATORY_SEMANTIC_KEYS if key not in summary]
            with self.subTest(reference=path.name):
                self.assertEqual(missing, [], f"{path.name} does not declare {missing}")

    def test_no_reference_names_a_path(self):
        """A family document cannot point at a task-scoped file, so it must not try.

        `../semantics.json` used to sit in one of these. There is no such file and
        there cannot be one: the semantics contract belongs to a task and this
        belongs to a family, so the path was unresolvable by construction and an
        agent following it got nothing.
        """
        for path, document in self.references():
            with self.subTest(reference=path.name):
                self.assertNotIn("contract_reference", document)


CANONICAL_GAP_REASONS = (
    "unsupported_shape",
    "semantic_mismatch",
    "layout_mismatch",
    "multi_operator_required",
)


class GapReasonVocabularyTests(unittest.TestCase):
    """One vocabulary for library gaps, shared by every surface that names them.

    §4.3 fixes the four reasons a library gap can have. The reasons are restated
    in three places -- the agent-visible capability references, each B package's
    `task.json`, and each B package's `problem.py` -- and the evaluator counts
    gaps by value, so a reason spelled differently in one of them is a gap that
    silently stops being counted. `unsupported_dtype` was removed from the
    vocabulary: a gap is a configuration the library cannot serve that the task
    actually asks for, and the tasks in this set never ask for a dtype the
    libraries reject, so no measured gap ever had that reason.
    """

    def b_tier(self):
        return [(directory, task) for directory, task in all_packages() if task["tier"] == "B_library"]

    def test_every_capability_reference_declares_the_same_four_reasons(self):
        """The reference is what a submission reads, so it cannot drift either."""
        for _, task in self.b_tier():
            reference = ROOT / task["capability_reference"]
            with self.subTest(task=task["id"]):
                declared = json.loads(reference.read_text(encoding="utf-8"))
                self.assertEqual(
                    declared["dispatch_contract"]["failure_reasons"],
                    list(CANONICAL_GAP_REASONS),
                )

    def test_every_declared_gap_reason_is_canonical(self):
        for _, task in self.b_tier():
            with self.subTest(task=task["id"]):
                for reason in task["library_policy"]["required_gap_reasons"]:
                    self.assertIn(reason, CANONICAL_GAP_REASONS)

    def test_every_declared_gap_reason_is_in_the_family_vocabulary(self):
        """A reason the family reference does not list cannot be reported by it."""
        for _, task in self.b_tier():
            reference = ROOT / task["capability_reference"]
            vocabulary = set(
                json.loads(reference.read_text(encoding="utf-8"))["dispatch_contract"]["failure_reasons"]
            )
            with self.subTest(task=task["id"]):
                self.assertLessEqual(set(task["library_policy"]["required_gap_reasons"]), vocabulary)

    def test_a_package_below_the_two_category_floor_says_so(self):
        """§4.3 asks a hidden gap set to cover at least two reason categories.

        A problem that genuinely cannot reach two is allowed to declare fewer
        only if `admission.gap_reason_coverage` records which categories are
        reachable and why the rest are not. The shortfall then reads as a
        recorded decision rather than an oversight.
        """
        for _, task in self.b_tier():
            reasons = task["library_policy"]["required_gap_reasons"]
            if len(reasons) >= 2:
                continue
            with self.subTest(task=task["id"]):
                coverage = task["admission"].get("gap_reason_coverage")
                self.assertIsNotNone(
                    coverage, f"{task['id']} declares {len(reasons)} category without saying why"
                )
                self.assertEqual(coverage["state"], "open")
                self.assertEqual(set(coverage["reachable"]), set(reasons))
                self.assertTrue(coverage["basis"])


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

    def test_every_public_case_pins_a_dtype_its_contract_permits(self):
        """A contract with two dtypes leaves each case to say which one it is.

        The case list is what the harness builds tensors from, and a case with no
        dtype is not runnable at all, so this is the difference between a package
        and a package-shaped file. Every entry-scoped package had this gap at
        once, which is the argument for asserting it rather than reviewing it.
        """
        for directory, task in all_packages():
            permitted = set(task["tensor_contract"]["input_dtypes"])
            cases = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))["cases"]
            for case in cases:
                with self.subTest(task=task["id"], case=case["case_id"]):
                    self.assertIn(case.get("dtype"), permitted)

    def test_every_half_precision_package_exercises_both_contract_dtypes(self):
        """Otherwise the second dtype in the contract is decoration.

        The B tier's fused path is reached differently per dtype, so a package
        that only ever runs float16 has a contract that claims coverage it does
        not take.
        """
        for directory, task in all_packages():
            permitted = set(task["tensor_contract"]["input_dtypes"])
            if len(permitted) < 2:
                continue
            cases = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))["cases"]
            exercised = {case["dtype"] for case in cases}
            with self.subTest(task=task["id"]):
                self.assertEqual(exercised, permitted)

    def test_the_driver_refuses_a_precision_the_contract_does_not_permit(self):
        """The pilot is a B-tier task; running it at float32 must not start.

        The submission is a scratch file rather than the pilot's own expert, which
        lives in the evaluator-only tree. The refusal is decided before the
        submission is evaluated, or the run would have started, so its contents
        cannot affect what is being checked here. Reaching into that tree made this
        test fail on a checkout documented not to carry one, which is exactly the
        checkout an agent gets.
        """
        pilot = TASKS_DIR / "sdpa_forward_pilot"
        with tempfile.TemporaryDirectory() as scratch:
            submission = Path(scratch) / "model_new.py"
            submission.write_text("class ModelNew:\n    pass\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "run_task.py"),
                 "--task-dir", str(pilot), "--submission", str(submission), "--precision", "fp32"],
                capture_output=True, text=True,
            )
        with self.subTest(exit_code=completed.returncode):
            self.assertEqual(completed.returncode, 2)
            self.assertIn("tensor", completed.stderr)
            self.assertIn("fp32", completed.stderr)


CHEATING_SUBMISSION = '''
"""A submission that computes nothing itself."""
import torch
import torch.nn.functional as F
from kernelbench.musa_extension import load_inline

_SOURCE = r"""
#include <musa_runtime.h>
__global__ void scratch_kernel(float* x) { x[0] = 0.0f; }
"""

_scratch = load_inline(name="scratch", cpp_sources="", musa_sources=_SOURCE, functions=[])


class ModelNew(torch.nn.Module):
    def __init__(self, scale, causal, window_left, window_right):
        super().__init__()
        self.scale = scale
        self.causal = causal

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=self.scale)
'''


HONEST_SUBMISSION = '''
"""A submission that writes the attention and never calls the library entry point."""
import torch
from kernelbench.musa_extension import load_inline

_SOURCE = r"""
#include <musa_runtime.h>
__global__ void attention_kernel(const float* q, const float* k, const float* v, float* out) {
    out[threadIdx.x] = q[threadIdx.x] + k[threadIdx.x] + v[threadIdx.x];
}
"""

_ext = load_inline(name="attention", cpp_sources="", musa_sources=_SOURCE, functions=[])


class ModelNew(torch.nn.Module):
    def __init__(self, scale, causal, window_left, window_right):
        super().__init__()
        self.scale = scale


# This one describes the entry point without calling it, which is allowed.
# The check matches a call, not a name.
# scaled_dot_product_attention is deliberately not used above.
'''


class ATierKernelScopeTests(unittest.TestCase):
    """The A tier must not be beatable by calling the fused attention entry point.

    The reference implementation calls the same entry point a submission could,
    so a submission that calls it matches the reference bit for bit and passes
    every correctness check. On `kb_l1_97` that submission would also score about
    2.5x, because the hand-written baseline it is divided by is slower than the
    library call it replaces. Nothing else in the pipeline rejects it: the
    upstream `torch_computation_ops` check ships as a warning.
    """

    def setUp(self):
        self.checker = load("kernel_static_checker", REPO / "src" / "kernelbench" / "kernel_static_checker.py")

    def audit(self, code, tier="A_kernel"):
        return self.checker.static_audit_kernel(code, tier=tier, backend="musa", precision="fp32")

    def test_the_check_that_catches_it_is_in_the_a_tier_set(self):
        self.assertIn("attention_entry_point", self.checker.A_TIER_FORBIDDEN_CHECKS)
        self.assertNotIn("attention_entry_point", self.checker.STRICT_CHECKS)

    def test_a_submission_that_calls_the_fused_entry_point_is_rejected(self):
        valid, errors, _ = self.audit(CHEATING_SUBMISSION)
        with self.subTest(errors=errors):
            self.assertFalse(valid, "a submission calling the fused entry point was accepted")
        self.assertTrue(any("scaled_dot_product_attention" in error for error in errors),
                        f"the rejection does not name the entry point: {errors}")

    def test_a_submission_that_never_calls_it_is_accepted(self):
        valid, errors, _ = self.audit(HONEST_SUBMISSION)
        self.assertTrue(valid, f"an honest submission was rejected: {errors}")

    def test_naming_the_entry_point_without_calling_it_is_not_a_rejection(self):
        """The check matches a call, so describing the prohibition is allowed.

        Without this the prompt could not name what it forbids.
        """
        valid, _errors, _ = self.audit(HONEST_SUBMISSION)
        self.assertTrue(valid)

    def test_the_b_tier_is_unaffected(self):
        """The B tier is built around calling exactly what this check forbids."""
        policy = {
            "allowed_libraries": ["libmusa", "libmudnn"],
            "allowed_symbol_prefixes": ["musa", "mudnn"],
            "required_trace_fields": ["case_id", "selected_path", "probe_status"],
            "trace_env_var": "KB_DISPATCH_TRACE",
            "case_id_env_var": "KB_DISPATCH_CASE_ID",
        }
        valid, errors, _ = self.checker.static_audit_kernel(
            CHEATING_SUBMISSION, tier="B_library", library_policy=policy, backend="musa",
        )
        self.assertFalse(any("attention_entry_point" in error for error in errors),
                         f"the B tier was held to an A-tier check: {errors}")

    def test_the_hand_written_answers_survive_the_tier_audit(self):
        """A check that rejects the answers in this repository would be worthless.

        Those answers compute the attention in a `.mu` kernel and hold the
        surrounding projections as `nn.Linear`, which is why the tier polices the
        entry point rather than every torch op. Only the attention set's own
        answers are checked: the rest of the 107 are outside this tier's scope
        and some of them fail checks that predate this work.
        """
        entries = json.loads(SET_PATH.read_text(encoding="utf-8"))["entries"]
        answers = [
            REPO / entry["tier"]["A_kernel"]["answer"]
            for entry in entries
            if entry["tier"]["A_kernel"].get("eligible") and entry["tier"]["A_kernel"].get("answer")
        ]
        self.assertTrue(answers, "the attention set names no A-tier answers")
        for answer in answers:
            with self.subTest(answer=answer.parent.name):
                self.assertTrue(answer.is_file(), f"{answer} is missing")
                valid, errors, _ = self.audit(answer.read_text(encoding="utf-8"))
                self.assertTrue(valid, f"{answer} is rejected by its own tier's audit: {errors}")


    def test_every_a_tier_contract_declares_its_kernel_scope(self):
        """A reader needs the boundary even where a check cannot draw it.

        The audit rejects the attention entry point. Everything else the tier
        permits or forbids has to be stated, because an `nn.Linear` that holds
        weights and one that computes are the same source.
        """
        scoped = 0
        for directory, task in all_packages():
            if task["tier"] != "A_kernel":
                continue
            scope = task.get("kernel_scope")
            with self.subTest(task=task["id"]):
                self.assertIsNotNone(scope, f"{directory.name} declares no kernel_scope")
                self.assertTrue(scope["must_be_in_kernel"], "no kernel work is named")
                self.assertIsInstance(scope["may_stay_library"], list)
                self.assertIn(scope["level"], {"attention_core", "whole_problem"})
            scoped += 1
        self.assertTrue(scoped, "no A-tier package was checked")

    def test_the_declared_enforcement_matches_the_check_the_audit_runs(self):
        for _, task in all_packages():
            if task["tier"] != "A_kernel":
                continue
            with self.subTest(task=task["id"]):
                self.assertIn(task["kernel_scope"]["enforced_by"],
                              self.checker.A_TIER_FORBIDDEN_CHECKS)

    def test_a_scope_that_permits_nothing_else_admits_it(self):
        """`kb_l1_97` is the attention and nothing more, so it leaves nothing out."""
        task = json.loads((TASKS_DIR / "kb_l1_97_a" / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(task["kernel_scope"]["may_stay_library"], [])


if __name__ == "__main__":
    unittest.main()
