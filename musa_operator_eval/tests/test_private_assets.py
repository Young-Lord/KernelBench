"""The maintainer-side assets must stay out of the agent-visible repository.

This is the whole of the publication separation: the task package is committed
and therefore agent-visible, and the expert dispatch, gap evidence, hidden
manifest, environment snapshot and admission decision live under `private/`,
which `.gitignore` excludes. There is no export step that could filter them out
later, so if one of them is ever tracked it is already published.

These tests check the rule rather than the current file listing, so they still
mean something on a fresh clone where `private/` holds only its README.

Run with either:
    python -m unittest musa_operator_eval.tests.test_private_assets -v
    python musa_operator_eval/tests/test_private_assets.py -v
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent
PRIVATE_PREFIX = "musa_operator_eval/private/"
PRIVATE_README = PRIVATE_PREFIX + "README.md"

# A path that must be ignored if the rule is in place. It does not have to exist:
# `git check-ignore` answers from the pattern, not the filesystem.
PROBE_UNDER_PRIVATE = PRIVATE_PREFIX + "sdpa_forward_b_v0/expert/model_new.py"

# Files that exist only on the maintainer side and must not appear in a task
# package, whatever the ignore rules say.
MAINTAINER_ONLY_NAMES = {"cases.private.json", "baseline.hidden.json", "admission.json", "private_gap_evidence.json"}

#: The hidden lists and the baselines live only on the maintainer side, so the tests
#: that read them skip in the checkout the agent gets rather than failing in it.
MAINTAINER_TREE_MOUNTED = (ROOT / "private" / "mingpt_block_b_v0" / "cases.private.json").is_file()

try:
    import torch
except ImportError:  # torch is not installed
    torch = None


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_TOP, capture_output=True, text=True)


def git_available():
    try:
        return git("rev-parse", "--is-inside-work-tree").returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(git_available(), "requires a git working tree")
class IgnoreRuleTests(unittest.TestCase):
    def test_arbitrary_private_files_are_ignored(self):
        """Checked against the pattern, so this holds on a fresh clone too."""
        self.assertEqual(
            git("check-ignore", "-q", PROBE_UNDER_PRIVATE).returncode, 0,
            f"{PROBE_UNDER_PRIVATE} is not ignored; maintainer assets would be committed",
        )

    def test_the_private_readme_is_deliberately_not_ignored(self):
        self.assertEqual(
            git("check-ignore", "-q", PRIVATE_README).returncode, 1,
            "the private README is the documented exception and must stay tracked",
        )


@unittest.skipUnless(git_available(), "requires a git working tree")
class TrackedFileTests(unittest.TestCase):
    def _tracked_under_private(self):
        result = git("ls-files", "--", "musa_operator_eval/private")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def test_only_the_readme_is_tracked(self):
        tracked = self._tracked_under_private()
        self.assertEqual(
            tracked, [PRIVATE_README],
            f"private assets are tracked and therefore published: {tracked}",
        )

    def test_every_present_private_file_is_ignored(self):
        """Catches a negated pattern added above the directory rule."""
        private_dir = ROOT / "private"
        if not private_dir.is_dir():
            self.skipTest("no private directory on this checkout")
        for path in private_dir.rglob("*"):
            if not path.is_file() or path.name == "__pycache__":
                continue
            relative = path.relative_to(REPO_TOP).as_posix()
            if relative == PRIVATE_README:
                continue
            with self.subTest(path=relative):
                self.assertEqual(
                    git("check-ignore", "-q", relative).returncode, 0,
                    f"{relative} is not ignored",
                )

    def test_maintainer_only_files_are_not_in_task_packages(self):
        for task_dir in sorted((ROOT / "tasks").glob("*")):
            if not task_dir.is_dir():
                continue
            for path in task_dir.rglob("*"):
                if path.name in MAINTAINER_ONLY_NAMES:
                    self.fail(f"{path.relative_to(REPO_TOP)} is a maintainer-only file inside a task package")


class AgentPackageAuditTests(unittest.TestCase):
    """§2: the package the agent is handed holds task definitions and nothing else.

    Each case below plants one of the four ways the separation can fail into a
    copy of a real package, so the audit is tested against the shape it audits
    rather than against a fixture written to be rejected.
    """

    SOURCE = ROOT / "tasks" / "kb_l3_43_b"

    def setUp(self):
        self.audit = _load_audit()
        self.temporary = tempfile.TemporaryDirectory()
        self.package = Path(self.temporary.name) / "kb_l3_43_b"
        shutil.copytree(self.SOURCE, self.package)

    def tearDown(self):
        self.temporary.cleanup()

    def test_the_shipped_packages_pass(self):
        report = self.audit.audit_packages(self.audit.all_packages())
        self.assertTrue(report["passed"], report["findings"])
        self.assertEqual(report["packages"], 11)

    def test_a_copy_of_a_package_passes(self):
        report = self.audit.audit_package(self.package)
        self.assertTrue(report["passed"], report["findings"])

    def test_a_maintainer_only_file_is_caught(self):
        (self.package / "cases.private.json").write_text("{}", encoding="utf-8")
        findings = self.audit.audit_package(self.package)["findings"]
        self.assertTrue(any("maintainer-only file" in finding for finding in findings), findings)

    def test_a_reference_to_the_maintainer_tree_is_caught(self):
        (self.package / "semantics.json").write_text(
            '{"note": "see musa_operator_eval/private/kb_l3_43_b/expert/model_new.py"}', encoding="utf-8"
        )
        findings = self.audit.audit_package(self.package)["findings"]
        self.assertTrue(any("maintainer tree" in finding for finding in findings), findings)

    def test_naming_the_upstream_project_is_caught(self):
        prompt = self.package / "PROMPT.md"
        prompt.write_text(prompt.read_text(encoding="utf-8") + "\nsee KernelBench level3\n", encoding="utf-8")
        findings = self.audit.audit_package(self.package)["findings"]
        self.assertTrue(any("upstream project" in finding for finding in findings), findings)

    def test_a_file_the_inventory_does_not_cover_is_caught(self):
        (self.package / "notes.md").write_text("an undeclared file\n", encoding="utf-8")
        findings = self.audit.audit_package(self.package)["findings"]
        self.assertTrue(any("neither editable nor frozen" in finding for finding in findings), findings)


@unittest.skipUnless(MAINTAINER_TREE_MOUNTED, "the maintainer tree is not mounted: the gap evidence is a hidden asset")
class GapCoverageTests(unittest.TestCase):
    """§4.3 wants at least two gap reason categories; when a task cannot reach two,
    the reason has to be recorded, with every other category accounted for."""

    CATEGORIES = {"unsupported_shape", "semantic_mismatch", "layout_mismatch", "multi_operator_required"}

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.private = self.root / "private"

    def test_a_single_category_task_records_why_the_others_are_out_of_reach(self):
        for directory in sorted(self.private.iterdir()):
            manifest = directory / "cases.private.json"
            if not manifest.is_file():
                continue
            cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
            reasons = {case["library_gap"]["reason"] for case in cases if case.get("library_gap")}
            with self.subTest(task=directory.name, categories=sorted(reasons)):
                # The A tier carries no gap cases at all: §4.3's gap record is how a
                # B-tier case says why the library path is not the expected one.
                if not reasons or len(reasons) >= 2:
                    continue
                evidence_path = directory / "private_gap_evidence.json"
                self.assertTrue(evidence_path.is_file(), f"{directory.name} reaches one category and records no reason")
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                block = evidence.get("exactly_one_reason_category")
                self.assertIsNotNone(block, f"{directory.name}: no exactly_one_reason_category block")
                self.assertEqual(block["state"], "open")
                argued = set(block["why_the_others_are_out_of_reach"])
                self.assertEqual(
                    argued,
                    self.CATEGORIES - reasons,
                    f"{directory.name}: the unreached categories are {sorted(self.CATEGORIES - reasons)} "
                    f"and the record argues {sorted(argued)}",
                )
                for category, text in block["why_the_others_are_out_of_reach"].items():
                    self.assertGreater(len(text), 60, f"{directory.name}/{category}: the reason is a stub")

    def test_the_contract_repeats_the_categories_the_cases_reach(self):
        """Two records of the same fact must not drift apart."""
        for directory in sorted(self.private.iterdir()):
            manifest = directory / "cases.private.json"
            if not manifest.is_file():
                continue
            cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
            reasons = {case["library_gap"]["reason"] for case in cases if case.get("library_gap")}
            task_id = json.loads(manifest.read_text(encoding="utf-8"))["task_id"]
            contract_path = next((path for path in (self.root / "tasks").glob("*/task.json") if json.loads(path.read_text(encoding="utf-8"))["id"] == task_id), None)
            self.assertIsNotNone(contract_path, task_id)
            coverage = (json.loads(contract_path.read_text(encoding="utf-8")).get("admission") or {}).get("gap_reason_coverage")
            with self.subTest(task=task_id, categories=sorted(reasons)):
                if not reasons:
                    # The A tier carries no gap cases, so it has no coverage to record.
                    self.assertIsNone(coverage, f"{task_id}: an A-tier task carries a gap coverage block")
                    continue
                if len(reasons) >= 2:
                    # Nothing to explain: the requirement is met and no coverage
                    # block is needed.
                    self.assertIsNone(coverage, "a task that reaches two categories carries a shortfall block")
                    continue
                self.assertIsNotNone(coverage, f"{task_id}: the contract records no coverage")
                self.assertEqual(coverage["state"], "open")
                self.assertEqual(sorted(coverage["reachable"]), sorted(reasons))
                self.assertEqual(sorted(coverage["unreachable"]), sorted(self.CATEGORIES - reasons))
                self.assertGreater(len(coverage["basis"]), 100, f"{task_id}: the basis is a stub")


@unittest.skipUnless(MAINTAINER_TREE_MOUNTED, "the maintainer tree is not mounted: the hidden lists and baselines live only there")
class CompiledAnswerScopeTests(unittest.TestCase):
    """§4.6's compiled path grades a case out of the tensors its input directory holds.

    That is the whole interface, so a compiled answer is legitimate exactly where the
    task supplies what a torch-free submission cannot build for itself: the case's own
    tensors, plus -- for a reference that keeps weights in `__init__` -- the state the
    evaluator materializes and stages (see `tests/test_materialize_model_state.py`,
    which checks on the device that the declaration matches what each reference holds).
    The rule this holds is mechanical: a package may not ship a compiled answer while
    neither of those is true.

    Skipped when torch is unavailable, since the question it asks about a reference is
    whether that reference carries state at all.
    """

    PRIVATE = ROOT / "private"

    def packages(self):
        for directory in sorted((ROOT / "tasks").iterdir()):
            task_path = directory / "task.json"
            if not task_path.is_file():
                continue
            yield directory, json.loads(task_path.read_text(encoding="utf-8"))

    @staticmethod
    def reference_parameter_count(directory: Path) -> int:
        """How many tensors the reference would have to be handed to reproduce its golden."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"scope_{directory.name}", directory / "problem.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model(*module.get_init_inputs())
        return len(list(model.parameters())) + len(list(model.buffers()))

    @unittest.skipUnless(torch is not None, "requires torch to count the reference's parameters")
    def test_a_compiled_answer_only_exists_where_the_task_supplies_what_it_needs(self):
        answers = sorted(self.PRIVATE.glob("*/compiled/kernel.mu"))
        self.assertTrue(answers, "no compiled answer is mounted: the maintainer tree is absent")
        by_id = {task["id"]: directory for directory, task in self.packages()}
        for answer in answers:
            task_id = answer.parent.parent.name
            task = next(task for _d, task in self.packages() if task["id"] == task_id)
            declared = ((task.get("starter") or {}).get("cpp") or {}).get("model_state")
            with self.subTest(task=task_id):
                count = self.reference_parameter_count(by_id[task_id])
                self.assertTrue(
                    count == 0 or declared,
                    f"{task_id} ships a compiled answer and its reference carries {count} tensors of state, "
                    "which no torch-free submission can reconstruct unless the contract stages it",
                )


class HiddenCaseListTests(unittest.TestCase):
    """§4.3: every package is graded on a hidden list, and the list follows the rules.

    Skipped when the maintainer tree is not mounted, which is the checkout the
    agent gets. The tests above check that it *is* unmounted there; these check
    what it holds where it is.
    """

    PRIVATE = ROOT / "private"

    def packages(self):
        for directory in sorted((ROOT / "tasks").iterdir()):
            task_path = directory / "task.json"
            if not task_path.is_file():
                continue
            yield directory, json.loads(task_path.read_text(encoding="utf-8"))

    def manifest_for(self, task):
        return self.PRIVATE / task["id"] / "cases.private.json"

    def test_every_package_has_a_hidden_list(self):
        for _directory, task in self.packages():
            path = self.manifest_for(task)
            with self.subTest(task=task["id"]):
                self.assertTrue(path.is_file(), f"{task['id']} has no hidden case list")
                manifest = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["visibility"], "private")
                self.assertEqual(manifest["task_id"], task["id"])
                self.assertEqual(manifest["tier"], task["tier"])

    def test_the_hidden_list_covers_the_boundaries_not_the_public_one(self):
        """§4.3 asks the hidden set for the cases the public smoke list cannot show."""
        required = {"boundary", "non_aligned", "extreme", "perf"}
        for _directory, task in self.packages():
            manifest = json.loads(self.manifest_for(task).read_text(encoding="utf-8"))
            tags = {case["tag"] for case in manifest["cases"]}
            with self.subTest(task=task["id"]):
                # A family with one legal shape cannot probe every axis, so the
                # requirement is that it says so rather than that it complies.
                covered = tags | ({"boundary", "non_aligned", "extreme"} if "image" in
                                  next(iter(manifest["cases"]))["shape"] else set())
                self.assertTrue(required <= covered, f"{task['id']} covers {sorted(tags)}")
                self.assertIn("generalization", tags)

    def test_no_hidden_case_repeats_a_public_one_and_no_seed_is_shared(self):
        for directory, task in self.packages():
            public = json.loads((directory / "public_cases.json").read_text(encoding="utf-8"))
            hidden = json.loads(self.manifest_for(task).read_text(encoding="utf-8"))
            with self.subTest(task=task["id"]):
                self.assertFalse(
                    {case["seed"] for case in public["cases"]} & {case["seed"] for case in hidden["cases"]},
                    f"{task['id']} shares a seed between the two lists",
                )
                self.assertFalse(
                    {json.dumps(case["shape"], sort_keys=True) for case in public["cases"]}
                    & {json.dumps(case["shape"], sort_keys=True) for case in hidden["cases"]},
                    f"{task['id']} repeats a public shape in the hidden list",
                )

    def test_every_package_has_a_baseline_record_for_its_hidden_set(self):
        """§4.4's minimum record, checked for presence and for the fields it names."""
        for _directory, task in self.packages():
            path = self.PRIVATE / task["id"] / "baseline.hidden.json"
            with self.subTest(task=task["id"]):
                self.assertTrue(path.is_file(), f"{task['id']} has no baseline record")
                baseline = json.loads(path.read_text(encoding="utf-8"))
                for field in ("environment_snapshot", "measurement_protocol", "implementations",
                              "results", "scoring", "tolerance_source"):
                    self.assertIn(field, baseline, f"{task['id']}'s baseline has no {field}")
                self.assertEqual(baseline["task_id"], task["id"])
                self.assertEqual(baseline["tier"], task["tier"])
                # §4.4 asks for a quantile and a throughput per performance case,
                # not a single number.
                measured = [row for row in baseline["results"] if row.get("latency_ms")]
                self.assertTrue(measured, f"{task['id']}'s baseline measured nothing")
                # One protocol, and it is the framework's: `run_task.py`'s reports are
                # produced by `kernelbench.timing`, so a denominator measured any other
                # way makes every ratio a ratio between two protocols. The record has to
                # say so, and the statistic has to be the one the reports carry.
                protocol = baseline["measurement_protocol"]
                self.assertEqual(
                    protocol.get("source"),
                    "kernelbench.timing.time_execution_with_cuda_event",
                    f"{task['id']}'s baseline does not name the framework's timing loop",
                )
                self.assertEqual(protocol.get("statistic"), "mean")
                self.assertEqual(protocol.get("cache"), "cold")
                for row in measured:
                    self.assertIn("latency_ms_p90", row)
                    self.assertIn("throughput_per_second", row)
                    self.assertIn("latency_ms_samples", row)
                    samples = row["latency_ms_samples"]
                    self.assertTrue(samples, f"{task['id']}/{row['implementation']} records no samples")
                    # The number and the samples agree: `latency_ms` is the mean of the
                    # trials, which is what the framework records as a `runtime`.
                    # `latency_ms` is the framework's own summary of its own trials:
                    # `get_timing_stats` renders the mean as `f"{mean:.3g}"`, which is
                    # three SIGNIFICANT digits, so "1.27" is the mean 1.2719768 and not
                    # a mean that was rounded to two decimals. The samples here are the
                    # same trials to six decimals, hence the relative slack.
                    expected = float(f"{sum(samples) / len(samples):.3g}")
                    self.assertAlmostEqual(
                        row["latency_ms"], expected, delta=abs(expected) * 1e-3,
                        msg=f"{task['id']}/{row['case_id']}/{row['implementation']}",
                    )
                    self.assertEqual(row["latency_ms_stats"]["mean"], row["latency_ms"])
                    self.assertEqual(row["latency_ms_stats"]["num_trials"], len(samples))

    def test_an_a_tier_library_row_is_measured_wherever_the_family_has_a_b_track(self):
        """§4.4's roster, checked against the thing that decides whether it can be built.

        The A tier's roster names `torch_musa_sdpa` and `mudnn_fused` alongside its own
        implementations. Those rows used to carry one shared reason -- "the tier is
        graded at float32, where the fused path does not answer" -- which is true of the
        fused kernel and false of the library's entry point: at float32
        `F.scaled_dot_product_attention` answers, through
        `aten::_scaled_dot_product_attention_math_musa`. So the rule is:

        * `mudnn_fused` is unbuildable on this tier, and its reason has to be the
          measurement rather than a claim about the tier (the library's own words are
          "Unsupport Type FLOAT");
        * a family whose ledger entry has a B track has a library implementation this
          repository can run, so `torch_musa_sdpa` must be a measured row there;
        * a family whose ledger entry has no B track records why, quoting the ledger
          rather than the tier's precision.
        """
        ledger = {
            entry["entry_id"]: entry
            for entry in json.loads((ROOT / "sets" / "attention" / "attention_set.json").read_text(encoding="utf-8"))["entries"]
        }
        checked = 0
        for _directory, task in self.packages():
            if task["tier"] != "A_kernel":
                continue
            entry = ledger[task["set_entry"].split("/")[-1]]
            baseline = json.loads((self.PRIVATE / task["id"] / "baseline.hidden.json").read_text(encoding="utf-8"))
            statuses = {}
            reasons = {}
            for row in baseline["results"]:
                statuses.setdefault(row["implementation"], set()).add(row["status"])
                if row.get("reason"):
                    reasons.setdefault(row["implementation"], row["reason"])
            with self.subTest(task=task["id"]):
                fused = reasons.get("mudnn_fused", "")
                self.assertIn("Unsupport Type FLOAT", fused,
                              f"{task['id']} does not record the measured reason the fused op gives for FLOAT")
                self.assertIn("measured", fused, f"{task['id']}'s fused reason is not a measurement")
                library = statuses.get("torch_musa_sdpa", set())
                if entry["tier"]["B_library"].get("eligible"):
                    self.assertIn("pass", library,
                                  f"{task['id']} has a B track, so the library's entry point answers this "
                                  f"family and its row has to be measured rather than unavailable: {library}")
                else:
                    self.assertEqual(library, {"unavailable"},
                                     f"{task['id']} has no B track, so a measured library row would be a "
                                     f"different operator: {library}")
                    self.assertIn("ledger", reasons.get("torch_musa_sdpa", ""),
                                  f"{task['id']} should quote the ledger's reason for having no B track")
            checked += 1
        self.assertEqual(checked, 6, "the A tier has six packages")

    def test_every_package_names_the_denominator_its_tier_is_scored_against(self):
        """§4.4: the A tier's is the upstream implementation, the B tier's the expert."""
        expected = {"A_kernel": "upstream_musa", "B_library": "expert_dispatch"}
        for _directory, task in self.packages():
            baseline = json.loads(
                (self.PRIVATE / task["id"] / "baseline.hidden.json").read_text(encoding="utf-8")
            )
            with self.subTest(task=task["id"]):
                self.assertEqual(baseline["scoring"]["speedup_denominator"], expected[task["tier"]])

    def test_every_hidden_case_declares_the_golden_it_is_checked_against(self):
        for _directory, task in self.packages():
            manifest = json.loads(self.manifest_for(task).read_text(encoding="utf-8"))
            for case in manifest["cases"]:
                with self.subTest(task=task["id"], case=case["case_id"]):
                    self.assertEqual(case.get("golden"), f"{case['case_id']}/golden/tensors.json")


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_agent_package", ROOT / "evaluator" / "audit_agent_package.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
