"""
Unit tests for the MUSA attention set and its verifier.

The set's whole value is that its claims are checkable, so these tests check the
two things that make that true: the recorded measurements still match the files
they came from, and the generated index is well-formed. A stale number or a
broken table is a defect here, not a cosmetic issue.

The last case is the important one: it tampers with a recorded measurement and
asserts the tool refuses to sign off.

Stdlib only.

Run with either:
    python -m unittest musa_operator_eval.tests.test_attention_set -v
    python musa_operator_eval/tests/test_attention_set.py -v
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent

_spec = importlib.util.spec_from_file_location("collect_attention_set", ROOT / "evaluator" / "collect_attention_set.py")
assert _spec is not None and _spec.loader is not None
collector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collector)

MANIFEST_PATH = ROOT / "sets" / "attention" / "attention_set.json"
INDEX_PATH = ROOT / "sets" / "attention" / "SET.md"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class ManifestShapeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = _manifest()

    def test_set_is_not_empty(self):
        self.assertGreaterEqual(len(self.manifest["entries"]), 8)

    def test_entry_ids_are_unique(self):
        ids = [entry["entry_id"] for entry in self.manifest["entries"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_entry_carries_the_required_fields(self):
        for entry in self.manifest["entries"]:
            with self.subTest(entry=entry["entry_id"]):
                self.assertIn("kernelbench", entry)
                self.assertIn("reference", entry)
                self.assertIn("shapes", entry)
                self.assertIn("tier", entry)
                self.assertIn("A_kernel", entry["tier"])
                self.assertIn("B_library", entry["tier"])

    def test_every_measurement_names_a_known_reader_and_target(self):
        target_ids = {target["target_id"] for target in self.manifest["targets"]}
        for entry in self.manifest["entries"]:
            for measurement in entry.get("measurements", []):
                with self.subTest(measurement=measurement["measurement_id"]):
                    self.assertIn(measurement["reader"], collector.READERS)
                    self.assertIn(measurement["target_id"], target_ids)

    def test_ineligible_tiers_say_why(self):
        """An entry that is not scoreable is only useful if the reason is recorded."""
        for entry in self.manifest["entries"]:
            for tier_name, tier in entry["tier"].items():
                with self.subTest(entry=entry["entry_id"], tier=tier_name):
                    if not tier.get("eligible"):
                        self.assertTrue(tier.get("reason"), f"{entry['entry_id']}/{tier_name} has no reason")

    def test_targets_cover_both_musa_architectures_under_discussion(self):
        arches = {target["musa_arch"] for target in self.manifest["targets"]}
        self.assertIn("mp_22", arches)
        self.assertIn("mp_31", arches)

    def test_warp_size_matches_the_architecture_table(self):
        """mp_31 is the odd one out; getting this wrong invalidates the tuning advice."""
        for target in self.manifest["targets"]:
            with self.subTest(target=target["target_id"]):
                expected = 32 if target["musa_arch"] == "mp_31" else 128
                self.assertEqual(target["warp_size"], expected)


class VerificationTests(unittest.TestCase):
    def test_the_shipped_set_verifies_clean(self):
        """The real integration check: recorded numbers must match their sources."""
        manifest = _manifest()
        problems = []
        for entry in manifest["entries"]:
            _, entry_problems = collector.verify_entry(entry)
            problems.extend(entry_problems)
        self.assertEqual(problems, [], f"attention set has drifted: {problems}")

    def test_a_tampered_measurement_is_caught(self):
        manifest = _manifest()
        entry = manifest["entries"][1]
        entry["measurements"][1]["expected"]["submission_ms"] = entry["measurements"][1]["expected"]["submission_ms"] * 2

        _, problems = collector.verify_entry(entry)
        self.assertTrue(problems, "a doubled latency must not be accepted")
        self.assertTrue(any("submission_ms" in problem for problem in problems), problems)

    def test_a_missing_reference_is_caught(self):
        entry = {
            "entry_id": "bogus",
            "reference": {"path": "KernelBench/level1/does_not_exist.py", "entrypoint": "Model"},
            "tier": {},
        }
        _, problems = collector.verify_entry(entry)
        self.assertTrue(any("reference missing" in problem for problem in problems), problems)

    def test_a_missing_answer_is_caught(self):
        manifest = _manifest()
        entry = manifest["entries"][0]
        entry["tier"]["A_kernel"]["answer"] = "KernelBench/level1_musa/problem_999/model_new.py"

        _, problems = collector.verify_entry(entry)
        self.assertTrue(any("A-tier answer missing" in problem for problem in problems), problems)

    def test_an_unknown_reader_is_caught(self):
        manifest = _manifest()
        entry = manifest["entries"][0]
        entry["measurements"][0]["reader"] = "not_a_reader"

        _, problems = collector.verify_entry(entry)
        self.assertTrue(any("unknown reader" in problem for problem in problems), problems)

    def test_source_paths_are_either_on_the_archive_branch_or_absent(self):
        """A path is either expected on the archive branch or expected nowhere."""
        dangling = collector.verify_dangling(_manifest()["dangling_references"])
        self.assertTrue(dangling)
        for reference in dangling:
            with self.subTest(path=reference["path"]):
                self.assertFalse(
                    reference["exists_locally"],
                    f"{reference['path']} is in the working tree; the archive is not vendored here",
                )
                if reference.get("resolves_on"):
                    self.assertIsNot(
                        reference.get("branch_reachable"), False,
                        f"{reference['path']} is gone from {reference['resolves_on']}",
                    )

    def test_the_archive_branch_really_has_the_paths_that_claim_it(self):
        """Skip-with-a-loud-reason rather than silently pass when the branch is unfetched."""
        dangling = collector.verify_dangling(_manifest()["dangling_references"])
        claimed = [reference for reference in dangling if reference.get("resolves_on")]
        self.assertTrue(claimed)
        if all(reference.get("branch_reachable") is None for reference in claimed):
            self.skipTest(
                "archive branch not fetched; run: git fetch origin "
                "feat/import-musa-attention-set:refs/remotes/origin/feat/import-musa-attention-set"
            )
        for reference in claimed:
            with self.subTest(path=reference["path"]):
                self.assertTrue(
                    reference["branch_reachable"],
                    f"the archive branch is fetched but lacks {reference['path']}",
                )


class ReaderTests(unittest.TestCase):
    def test_attention_perf_reader_picks_the_right_row(self):
        row = collector.read_attn_perf_csv({"problem": 97, "ts": "2026-09-05 01:06:11"})
        self.assertAlmostEqual(row["submission_ms"], 191.026, places=3)
        self.assertAlmostEqual(row["speedup"], 0.3997, places=4)

    def test_attention_perf_reader_rejects_an_unknown_timestamp(self):
        with self.assertRaises(collector.DriftError):
            collector.read_attn_perf_csv({"problem": 97, "ts": "1999-01-01 00:00:00"})

    def test_eval_json_reader_reports_milliseconds(self):
        """The archive's runtime field is milliseconds; the tuning analysis only reconciles that way."""
        record = collector.read_kernelbench_eval_json({"problem": "97", "sample_id": 0})
        self.assertAlmostEqual(record["submission_ms"], 38200.0, places=3)
        self.assertTrue(record["correctness"])

    def test_eval_json_reader_rejects_a_missing_problem(self):
        with self.assertRaises(collector.DriftError):
            collector.read_kernelbench_eval_json({"problem": "999", "sample_id": 0})


class OperatorExtractionTests(unittest.TestCase):
    def test_extracts_each_kind(self):
        source = (
            "import torch\n"
            "import torch.nn as nn\n"
            "x = torch.matmul(a, b)\n"
            "y = F.softmax(x, dim=-1)\n"
            "z = nn.Linear(4, 4)\n"
            "w = x.transpose(-1, -2)\n"
        )
        found = collector.extract_operators(source)
        self.assertIn("matmul", found["torch"])
        self.assertIn("softmax", found["functional"])
        self.assertIn("Linear", found["modules"])
        self.assertIn("transpose", found["tensor_methods"])

    def test_unrelated_method_calls_are_not_reported(self):
        """Only the curated method list counts, otherwise every `.foo(` becomes noise."""
        self.assertNotIn("tensor_methods", collector.extract_operators("y = x.some_custom_thing(3)\n"))

    def test_empty_source_yields_nothing(self):
        self.assertEqual(collector.extract_operators(""), {})


class Level1ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.summary = collector.summarize_level1_archive()

    def test_matches_the_archive_readme(self):
        """Cross-check against the numbers the archive documents for itself."""
        self.assertEqual(self.summary["archived"], 100)
        self.assertEqual(self.summary["correct"], 99)
        self.assertAlmostEqual(self.summary["geometric_mean_speedup"], 0.1262, places=3)
        self.assertAlmostEqual(self.summary["median_speedup"], 0.1693, places=3)

    def test_the_excluded_problem_is_reported_with_a_reason(self):
        excluded = {entry["problem_id"]: entry["reason"] for entry in self.summary["excluded"]}
        self.assertIn(72, excluded)
        self.assertTrue(excluded[72])

    def test_gaps_are_ranked_worst_first(self):
        gaps = self.summary["largest_library_gaps"]
        self.assertEqual(gaps, sorted(gaps, key=lambda row: row["library_gap"], reverse=True))

    def test_the_worst_gap_is_the_attention_problem(self):
        self.assertEqual(self.summary["largest_library_gaps"][0]["problem_id"], 97)


class KernelCandidateTests(unittest.TestCase):
    """The candidate inventory, and the reference discipline between it and the set."""

    def setUp(self):
        self.manifest = _manifest()
        self.record = collector.load_candidate_record(self.manifest)
        self.candidates, self.problems = collector.verify_candidates(self.manifest, self.record)

    def test_the_shipped_record_verifies_clean(self):
        self.assertEqual(self.problems, [], f"candidate record has drifted: {self.problems}")

    def test_record_is_maintainer_only(self):
        self.assertEqual(self.record["visibility"], "maintainer_only")

    def test_candidate_ids_are_unique(self):
        ids = [candidate["candidate_id"] for candidate in self.candidates]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_candidate_declares_at_least_one_target(self):
        for candidate in self.candidates:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertTrue(candidate["targets"])

    def test_every_candidate_of_a_maintainer_record_carries_evidence(self):
        for candidate in self.candidates:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertTrue(
                    collector._has_evidence(candidate),
                    f"{candidate['candidate_id']} points at nothing checkable",
                )

    def test_an_entry_pointing_at_an_unknown_candidate_is_caught(self):
        manifest = _manifest()
        manifest["entries"][0]["candidate_kernels"] = ["not_a_candidate"]
        _, problems = collector.verify_candidates(manifest, self.record)
        self.assertTrue(any("not_a_candidate" in problem for problem in problems), problems)

    def test_an_unknown_target_is_caught(self):
        record = json.loads(json.dumps(self.record))
        record["candidates"][0]["targets"] = ["s9999_fp64"]
        _, problems = collector.verify_candidates(self.manifest, record)
        self.assertTrue(any("s9999_fp64" in problem for problem in problems), problems)

    def test_an_unknown_kind_is_caught(self):
        record = json.loads(json.dumps(self.record))
        record["candidates"][0]["kind"] = "vibes"
        _, problems = collector.verify_candidates(self.manifest, record)
        self.assertTrue(any("vibes" in problem for problem in problems), problems)

    def test_a_claim_without_evidence_is_caught(self):
        record = json.loads(json.dumps(self.record))
        record["candidates"][0].pop("archive_path")
        _, problems = collector.verify_candidates(self.manifest, record)
        self.assertTrue(any("evidence" in problem for problem in problems), problems)

    def test_an_empty_record_is_caught(self):
        _, problems = collector.verify_candidates(self.manifest, {"candidates": []})
        self.assertTrue(problems)


class CandidateFindingTests(unittest.TestCase):
    """The load-bearing facts, including the one that corrected an earlier claim."""

    def setUp(self):
        self.manifest = _manifest()
        self.record = collector.load_candidate_record(self.manifest)
        self.by_id = {candidate["candidate_id"]: candidate for candidate in self.record["candidates"]}

    def test_the_record_carries_the_correction(self):
        """The S5000-only claim was wrong; the correction must stay recorded."""
        self.assertIn("correction_note", self.record)
        self.assertIn("wrong", self.record["correction_note"])

    def test_mate_is_pinned_to_s5000(self):
        mate = self.by_id["mate"]
        self.assertIn("S5000", mate["requirements"]["hardware"])
        self.assertFalse(any(target.startswith("s4000") for target in mate["targets"]))

    def test_mt_flashmla_targets_mp31(self):
        self.assertIn("MP31", self.by_id["mt_flashmla"]["requirements"]["hardware"])

    def test_llama_cpp_is_the_mp22_counterexample(self):
        """The one candidate whose build files name mp_22, and why the old claim died."""
        llama = self.by_id["llama_cpp_fattn"]
        self.assertIn("mp_22", llama["requirements"]["hardware"])
        self.assertIn("s4000_fp32", llama["targets"])
        self.assertEqual(llama["kind"], "fused_library")
        self.assertIn("portable", llama["limits"]["provenance"])

    def test_no_candidate_is_a_handwritten_native_s4000_attention_kernel(self):
        """The part of the corrected claim that survives, and the reason it matters."""
        s4000_capable = [
            candidate for candidate in self.by_id.values()
            if any(target.startswith("s4000") for target in candidate["targets"])
        ]
        self.assertTrue(s4000_capable)
        for candidate in s4000_capable:
            with self.subTest(candidate=candidate["candidate_id"]):
                if candidate["kind"] == "handwritten_kernel":
                    self.assertIn("in-repo", candidate["archive_revision_label"])
                else:
                    self.assertNotEqual(candidate["role"], "native-kernel-library")

    def test_the_head_dim_boundary_on_torch_musa_is_recorded(self):
        """The archive answers half of what used to be an open question here."""
        limits = self.by_id["torch_musa_sdpa"]["limits"]
        self.assertIn("head_dim", limits)
        self.assertIn("only some head dimensions", limits["head_dim"])

    def test_mate_layout_differs_from_the_b_tier_task(self):
        """The B-tier task is BHSD; MATE is BSHD. A dispatch has to transpose."""
        self.assertIn("BSHD", self.by_id["mate"]["limits"]["layout"])

    def test_handwritten_kernels_are_marked_warp_size_sensitive(self):
        limits = self.by_id["kb_level3_musa_handwritten"]["limits"]
        self.assertIn("warp_size_sensitivity", limits)

    def test_the_few_shot_example_is_recorded_as_a_candidate(self):
        """The model under test has already been shown this kernel."""
        example = self.by_id["kb_few_shot_flash_attn"]
        self.assertIn("already been shown this kernel", example["why_it_matters"])
        self.assertEqual(example["serves"], [])

    def test_every_archive_candidate_names_a_revision_or_says_why_not(self):
        for candidate in self.record["candidates"]:
            with self.subTest(candidate=candidate["candidate_id"]):
                if candidate["archive_revision_label"] == "in-repo":
                    self.assertIsNone(candidate["archive_revision"])
                else:
                    self.assertTrue(candidate["archive_revision"])

    def test_the_source_archive_block_points_at_a_branch(self):
        archive = self.record["source_archive"]
        self.assertEqual(archive["branch"], "feat/import-musa-attention-set")
        self.assertTrue(archive["merge_status"])


class ServesAgreementTests(unittest.TestCase):
    """Entries and candidates declare the same relationship; both must agree."""

    def setUp(self):
        self.manifest = _manifest()
        self.record = collector.load_candidate_record(self.manifest)

    def test_one_sided_edit_is_caught(self):
        manifest = _manifest()
        manifest["entries"][0]["candidate_kernels"] = ["mate"]  # drop the rest
        _, problems = collector.verify_candidates(manifest, self.record)
        self.assertTrue(any("disagree" in problem for problem in problems), problems)

    def test_a_candidate_serving_an_unknown_entry_is_caught(self):
        record = json.loads(json.dumps(self.record))
        record["candidates"][0]["serves"] = ["kb_does_not_exist"]
        _, problems = collector.verify_candidates(self.manifest, record)
        self.assertTrue(any("kb_does_not_exist" in problem for problem in problems), problems)


class GeneratedIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = INDEX_PATH.read_text(encoding="utf-8")

    def test_index_is_stamped_as_generated(self):
        self.assertIn("Do not edit by hand", self.index)

    def test_every_entry_appears(self):
        for entry in _manifest()["entries"]:
            self.assertIn(entry["entry_id"], self.index)

    def test_markdown_tables_are_well_formed(self):
        """A separator row with the wrong column count renders as a broken table."""
        lines = self.index.splitlines()
        checked = 0
        index = 0
        while index < len(lines) - 1:
            header, separator = lines[index], lines[index + 1]
            if header.startswith("|") and separator.startswith("|") and set(separator) <= set("|-: "):
                self.assertEqual(
                    header.count("|"),
                    separator.count("|"),
                    f"table starting at line {index + 1} has a mismatched separator",
                )
                checked += 1
                index += 2
                continue
            index += 1
        self.assertGreater(checked, 5, "expected the index to contain several tables")

    def test_no_double_space_artifacts_in_generated_tables(self):
        for line in self.index.splitlines():
            if line.startswith("|") and not set(line) <= set("|-: "):
                self.assertNotIn("  |", line, f"stray padding in table row: {line}")


class CommandLineTests(unittest.TestCase):
    def _run(self, manifest_path: Path) -> int:
        argv = sys.argv
        sys.argv = ["collect_attention_set.py", "--check", "--manifest", str(manifest_path)]
        try:
            return collector.main()
        finally:
            sys.argv = argv

    def test_check_mode_passes_on_the_shipped_manifest(self):
        self.assertEqual(self._run(MANIFEST_PATH), 0)

    def test_check_mode_fails_on_a_tampered_manifest(self):
        manifest = _manifest()
        manifest["entries"][0]["measurements"][0]["expected"]["submission_ms"] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "attention_set.json"
            tampered.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(self._run(tampered), 1)

    def test_check_mode_does_not_rewrite_the_index(self):
        before = INDEX_PATH.read_text(encoding="utf-8")
        self._run(MANIFEST_PATH)
        self.assertEqual(before, INDEX_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
