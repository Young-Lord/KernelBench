"""Tests for the six MUSA evaluation schemas and the validator.

Standard library only, on purpose: the schemas are the gate every companion file
has to pass before it is committed, so the test that proves they work has to run
on a machine with no GPU stack and no third-party packages. When `jsonschema` is
installed the same documents are checked through it as well, so the built-in
engine can be caught disagreeing with a full implementation.

The cases that earn their keep are the tier ones in `TierRuleTests` and
`CaseManifestRuleTests`: §4.8 asks validation to tighten or loosen fields by the
value of `tier`, so an A-tier contract with a library whitelist must fail, and a
B-tier contract with no gap definition must fail.

Run with:
    python3 musa_operator_eval/tests/test_schemas.py -v
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("validate_schemas", ROOT / "tools" / "validate_schemas.py")
assert _spec is not None and _spec.loader is not None
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)

CANONICAL_GAP_REASONS = (
    "unsupported_shape",
    "semantic_mismatch",
    "layout_mismatch",
    "multi_operator_required",
)


def task_document(tier, **overrides):
    """A minimal contract of the requested tier, valid unless overridden."""
    document = {
        "schema_version": "1.0.0",
        "tier": tier,
        "id": "example_v0",
        "name": "Example operator",
        "family": "sdpa_forward",
        "source": {"project": "example-source", "record": "sources/example.json"},
        "target_environment": {
            "snapshot_id": "musa-5f9d7b9dd1233a68",
            "record": "environments/musa-5f9d7b9dd1233a68.public.json",
            "backend": "musa",
        },
        "problem_file": "problem.py",
        "reference": "problem.py:Model",
        "case_parameters": {"shape.B": "batch_size"},
        "tolerances": {"atol": 0.01, "rtol": 0.01, "max_abs_error": 0.05},
        "tensor_contract": {"input_dtypes": ["float16"], "accumulation_dtype": "float32"},
        "prohibited": ["ATen"],
        "submission": {"entrypoint": "ModelNew"},
        "attempt_budget": {"max_attempts": 20, "wall_time_seconds": 7200},
        "outputs": {"required": ["output"]},
    }
    if tier == "B_library":
        document["capability_reference"] = "agent_reference/sdpa_forward_capabilities.json"
        document["library_policy"] = {
            "allowed_libraries": ["libmusa"],
            "allowed_symbol_prefixes": ["musa"],
            "required_trace_fields": ["case_id", "selected_path", "probe_status"],
            "trace_env_var": "KB_DISPATCH_TRACE",
            "case_id_env_var": "KB_DISPATCH_CASE_ID",
            "dispatch_order": ["fused_library", "library_composition", "custom_fallback"],
            "minimum_gap_cases": 3,
            "required_gap_reasons": ["unsupported_shape", "semantic_mismatch"],
        }
    document.update(overrides)
    return document


def case(case_id="smoke_001", tag="smoke", **overrides):
    document = {
        "case_id": case_id,
        "tag": tag,
        "seed": 1101,
        "dtype": "float16",
        "shape": {"B": 1, "H": 4, "S": 64, "D": 64},
    }
    document.update(overrides)
    return document


def manifest_document(tier, visibility, cases, **overrides):
    document = {
        "schema_version": "1.0.0",
        "task_id": "example_b_v0",
        "tier": tier,
        "visibility": visibility,
        "case_generation": {"tool": "musa_operator_eval/tools/generate_cases.py", "deterministic": True},
        "cases": cases,
    }
    document.update(overrides)
    return document


def gap(case_id, reason="unsupported_shape"):
    return {
        "case_id": case_id,
        "tag": "boundary",
        "seed": 91001,
        "dtype": "float16",
        "shape": {"B": 1, "H": 8, "S": 128, "D": 192},
        "golden": f"golden/{case_id}/tensors.json",
        "expected_path": "library_composition",
        "library_gap": {
            "reason": reason,
            "expected_path": "library_composition",
            "evidence": "measured on the target device during admission",
        },
    }


def environment_document(visibility="redacted", **overrides):
    document = {
        "schema_version": "2.0.0",
        "status": "complete",
        "missing_required_fields": [],
        "snapshot_id": "musa-5f9d7b9dd1233a68",
        "visibility": visibility,
        "captured_at": "2026-09-20T12:22:54Z",
        "hardware": {
            "device_name": "MTT S4000",
            "architecture": "mp_22",
            "architecture_source": "torch.musa device properties",
            "device_count": 1,
        },
        "software": {"driver": "2.7.0", "musa_toolkit": "3.1.0", "mudnn": "2.7.0", "mublas": "1.6.0"},
        "build": {"flags": [], "fast_math": False},
    }
    document.update(overrides)
    return document


class SchemaDocumentTests(unittest.TestCase):
    """The schemas themselves, before any instance is checked against them."""

    def schema_paths(self):
        return sorted((ROOT / "schemas").glob("*.schema.json"))

    def test_there_are_six_schemas_and_the_validator_knows_all_of_them(self):
        self.assertEqual(len(self.schema_paths()), 6)
        self.assertEqual(len(validator.SCHEMAS), 6)
        self.assertEqual(
            {path.name for path in self.schema_paths()},
            set(validator.SCHEMAS.values()),
        )

    def test_every_schema_is_draft_2020_12_and_versioned(self):
        for path in self.schema_paths():
            schema = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(schema=path.name):
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertRegex(schema["$id"], r"-(\d+\.\d+\.\d+)\.json$")

    def test_every_schema_has_a_kind_to_load_it_by(self):
        for kind in validator.SCHEMAS:
            with self.subTest(kind=kind):
                self.assertTrue(validator.load_schema(kind))


class RepositoryArtifactTests(unittest.TestCase):
    """Every artifact the repository ships has to pass its own schema."""

    def test_discovery_finds_every_tracked_kind(self):
        """The baseline and the report are private or produced by a run, so they are
        not required to be present on a fresh clone; the other four always are."""
        kinds = {kind for _, kind in validator.discover(ROOT)}
        self.assertTrue({"task_contract", "semantics_contract", "case_manifest", "environment_snapshot"} <= kinds)
        self.assertLessEqual(kinds, set(validator.SCHEMAS))

    def test_every_discovered_artifact_validates(self):
        """The regression test: drift shows up here before it is committed."""
        for path, kind in validator.discover(ROOT):
            with self.subTest(document=str(path.relative_to(ROOT))):
                self.assertEqual(validator.validate_document(path, kind, engine="stdlib"), [])

    def test_the_builtin_engine_agrees_with_jsonschema_when_it_is_installed(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema is not installed; the stdlib engine is the only one here")
        for path, kind in validator.discover(ROOT):
            with self.subTest(document=str(path.relative_to(ROOT))):
                # The verdicts must agree; the message wording will not.
                self.assertEqual(
                    bool(validator.validate_document(path, kind, engine="stdlib")),
                    bool(validator.validate_document(path, kind, engine="jsonschema")),
                )


class TierRuleTests(unittest.TestCase):
    """Guide §4.8: the tier decides which fields exist, so it is validated by value."""

    def test_a_valid_a_tier_contract_passes(self):
        self.assertEqual(validator.validate_instance_stdlib(task_document("A_kernel"), "task_contract"), [])

    def test_a_valid_b_tier_contract_passes(self):
        self.assertEqual(validator.validate_instance_stdlib(task_document("B_library"), "task_contract"), [])

    def test_an_a_tier_contract_may_not_declare_a_library_policy(self):
        """A whitelist is a B-tier concept; carrying one in the A tier is a leak."""
        document = task_document("A_kernel", library_policy={"allowed_libraries": ["libmusa"]})
        errors = validator.validate_instance_stdlib(document, "task_contract")
        self.assertTrue(any("must not match" in error for error in errors), errors)

    def test_a_b_tier_contract_must_declare_a_library_policy(self):
        document = task_document("B_library")
        del document["library_policy"]
        errors = validator.validate_instance_stdlib(document, "task_contract")
        self.assertTrue(any("library_policy" in error for error in errors), errors)

    def test_a_b_tier_policy_must_define_the_gap_requirement(self):
        """§4.1: the minimum gap count and the reasons are the B-tier definition."""
        for missing in ("minimum_gap_cases", "required_gap_reasons"):
            with self.subTest(missing=missing):
                document = task_document("B_library")
                del document["library_policy"][missing]
                errors = validator.validate_instance_stdlib(document, "task_contract")
                self.assertTrue(any(missing in error for error in errors), errors)

    def test_a_reason_outside_the_canonical_vocabulary_is_rejected(self):
        """`unsupported_dtype` was removed from §4.3's four concepts; it must not come back."""
        document = task_document("B_library")
        document["library_policy"]["required_gap_reasons"] = ["unsupported_shape", "unsupported_dtype"]
        errors = validator.validate_instance_stdlib(document, "task_contract")
        self.assertTrue(any("unsupported_dtype" in error for error in errors), errors)

    def test_every_canonical_reason_is_accepted(self):
        document = task_document("B_library")
        document["library_policy"]["required_gap_reasons"] = list(CANONICAL_GAP_REASONS)
        self.assertEqual(validator.validate_instance_stdlib(document, "task_contract"), [])

    def test_one_gap_category_without_a_coverage_note_is_rejected(self):
        """§4.3 asks for two categories; one has to be explained, not just declared."""
        document = task_document("B_library")
        document["library_policy"]["required_gap_reasons"] = ["unsupported_shape"]
        errors = validator.validate_instance_stdlib(document, "task_contract")
        self.assertTrue(any("admission" in error for error in errors), errors)

    def test_one_gap_category_with_a_coverage_note_is_accepted(self):
        document = task_document("B_library")
        document["library_policy"]["required_gap_reasons"] = ["unsupported_shape"]
        document["admission"] = {
            "status": "pending_device_measurement",
            "reason": "measured on the device",
            "gap_reason_coverage": {
                "state": "open",
                "reachable": ["unsupported_shape"],
                "unreachable": ["semantic_mismatch", "layout_mismatch", "multi_operator_required"],
                "basis": "the scale is fixed and there is no window, so no other category is reachable",
            },
        }
        self.assertEqual(validator.validate_instance_stdlib(document, "task_contract"), [])


class CaseManifestRuleTests(unittest.TestCase):
    """Guide §4.3: the tier and the visibility decide which case fields are legal."""

    def test_a_public_a_manifest_without_dispatch_fields_passes(self):
        document = manifest_document("A_kernel", "public", [case()])
        self.assertEqual(validator.validate_instance_stdlib(document, "case_manifest"), [])

    def test_an_a_tier_case_may_not_name_an_expected_path(self):
        document = manifest_document("A_kernel", "public", [case(expected_path="fused_library")])
        errors = validator.validate_instance_stdlib(document, "case_manifest")
        self.assertTrue(any("must not match" in error for error in errors), errors)

    def test_a_public_b_manifest_passes_when_every_case_is_fused(self):
        document = manifest_document("B_library", "public", [case(expected_path="fused_library")])
        self.assertEqual(validator.validate_instance_stdlib(document, "case_manifest"), [])

    def test_a_b_tier_case_must_name_an_expected_path(self):
        document = manifest_document("B_library", "public", [case()])
        errors = validator.validate_instance_stdlib(document, "case_manifest")
        self.assertTrue(any("expected_path" in error for error in errors), errors)

    def test_a_public_b_manifest_may_not_carry_a_library_gap(self):
        """Otherwise the public list tells the submission where the boundary is."""
        document = manifest_document("B_library", "public", [gap("smoke_001")])
        errors = validator.validate_instance_stdlib(document, "case_manifest")
        self.assertTrue(any("must not match" in error for error in errors), errors)

    def test_a_public_b_case_must_expect_the_fused_path(self):
        document = manifest_document("B_library", "public", [case(expected_path="library_composition")])
        errors = validator.validate_instance_stdlib(document, "case_manifest")
        self.assertTrue(any("fused_library" in error for error in errors), errors)

    def test_a_private_b_manifest_meeting_the_floor_passes(self):
        cases = [
            gap("hidden_gap_001", "unsupported_shape"),
            gap("hidden_gap_002", "unsupported_shape"),
            gap("hidden_gap_003", "semantic_mismatch"),
        ]
        document = manifest_document("B_library", "private", cases)
        self.assertEqual(validator.validate_instance_stdlib(document, "case_manifest"), [])

    def test_a_private_b_manifest_below_the_gap_count_is_rejected(self):
        document = manifest_document("B_library", "private", [gap("hidden_gap_001")])
        errors = validator.validate_instance(document, "case_manifest", engine="stdlib")
        self.assertTrue(any("at least 3 library gaps" in error for error in errors), errors)

    def test_a_private_b_manifest_below_the_reason_count_is_rejected(self):
        cases = [gap(f"hidden_gap_{index:03d}", "unsupported_shape") for index in range(1, 4)]
        document = manifest_document("B_library", "private", cases)
        errors = validator.validate_instance(document, "case_manifest", engine="stdlib")
        self.assertTrue(any("2 gap-reason categories" in error for error in errors), errors)

    def test_a_gap_reason_outside_the_vocabulary_is_rejected(self):
        document = manifest_document("B_library", "private", [gap("hidden_gap_001", "unsupported_dtype")])
        errors = validator.validate_instance_stdlib(document, "case_manifest")
        self.assertTrue(any("unsupported_dtype" in error for error in errors), errors)

    def test_a_perf_case_must_carry_its_protocol(self):
        document = manifest_document("B_library", "public", [case(tag="perf", expected_path="fused_library")])
        errors = validator.validate_instance_stdlib(document, "case_manifest")
        self.assertTrue(any("performance" in error for error in errors), errors)

    def test_duplicate_case_ids_are_rejected(self):
        cases = [case("smoke_001", expected_path="fused_library"), case("smoke_001", expected_path="fused_library")]
        document = manifest_document("B_library", "public", cases)
        errors = validator.validate_instance(document, "case_manifest", engine="stdlib")
        self.assertTrue(any("duplicate" in error for error in errors), errors)


class EnvironmentRedactionTests(unittest.TestCase):
    """A redacted record keeps the configuration and drops the machine identity."""

    def test_a_redacted_record_without_identity_passes(self):
        self.assertEqual(validator.validate_instance_stdlib(environment_document("redacted"), "environment_snapshot"), [])

    def test_a_full_record_may_carry_identity(self):
        document = environment_document("full", device_instance_id="gpu-7b930f8e1cf45ad1")
        self.assertEqual(validator.validate_instance_stdlib(document, "environment_snapshot"), [])

    def test_a_redacted_record_may_not_carry_identity(self):
        for key, value in (
            ("device_instance_id", "gpu-7b930f8e1cf45ad1"),
            ("device_instance", {"product_name": "MTT S4000"}),
            ("host", {"kernel": "7.2.6"}),
            ("raw_probes", {"driver_query": {}}),
            ("container", {"platform": "AutoDL"}),
        ):
            with self.subTest(key=key):
                document = environment_document("redacted", **{key: value})
                errors = validator.validate_instance_stdlib(document, "environment_snapshot")
                self.assertTrue(any("must not match" in error for error in errors), errors)

    def test_the_document_version_is_pinned_to_the_collector(self):
        document = environment_document(schema_version="1.0.0")
        errors = validator.validate_instance_stdlib(document, "environment_snapshot")
        self.assertTrue(any("2.0.0" in error for error in errors), errors)


class ReportModeTests(unittest.TestCase):
    """run_task.py emits two report shapes; the mode decides which blocks exist."""

    def static_report(self):
        return {
            "mode": "static",
            "task_id": "sdpa_forward_b_v0",
            "tier": "B_library",
            "static_audit": {"passed": True, "errors": [], "warnings": []},
            "case_parameters": {"count": 5, "errors": []},
            "case_sources": {"count": 5, "errors": []},
            "summary": {"passed_overall": True, "errors": []},
        }

    def full_report(self):
        return {
            "mode": "full",
            "task_id": "sdpa_forward_b_v0",
            "tier": "B_library",
            "backend": "musa",
            "precision": "fp16",
            "submission": "model_new.py",
            "cases": [{"case_id": "smoke_001", "expected_path": "fused_library", "passed": True}],
            "summary": {"cases": 1, "passed": 1, "failed": 0, "passed_overall": True},
        }

    def test_both_shapes_pass(self):
        self.assertEqual(validator.validate_instance_stdlib(self.static_report(), "evaluation_report"), [])
        self.assertEqual(validator.validate_instance_stdlib(self.full_report(), "evaluation_report"), [])

    def test_a_static_report_missing_a_stage_is_rejected(self):
        document = self.static_report()
        del document["case_sources"]
        errors = validator.validate_instance_stdlib(document, "evaluation_report")
        self.assertTrue(any("case_sources" in error for error in errors), errors)

    def test_a_full_report_missing_its_cases_is_rejected(self):
        document = self.full_report()
        del document["cases"]
        errors = validator.validate_instance_stdlib(document, "evaluation_report")
        self.assertTrue(any("cases" in error for error in errors), errors)

    def test_an_unknown_mode_is_rejected(self):
        document = self.static_report()
        document["mode"] = "partial"
        errors = validator.validate_instance_stdlib(document, "evaluation_report")
        self.assertTrue(any("partial" in error for error in errors), errors)

    def test_the_summary_must_say_whether_the_run_passed(self):
        document = self.full_report()
        del document["summary"]["passed_overall"]
        errors = validator.validate_instance_stdlib(document, "evaluation_report")
        self.assertTrue(any("passed_overall" in error for error in errors), errors)

    def test_the_drivers_stage_blocks_and_skipped_rows_are_accepted(self):
        """The driver is the producer and keeps growing stage detail; a report
        that names its stages, its golden cross-check and a skipped case still
        has to validate, because none of that changes what the report means."""
        document = self.full_report()
        document["stages"] = ["case_generation", "golden", "correctness"]
        document["golden"] = {"generated_root": "runs/generated", "results": [], "fatal": False, "exit_code": 0}
        document["cases"] = [
            {
                "case_id": "correctness_001",
                "tag": "correctness",
                "stage": "correctness",
                "correctness": False,
                "static_audit_errors": [],
                "dispatch_trace_errors": [],
                "passed": False,
            },
            {
                "case_id": "perf_001",
                "tag": "perf",
                "stage": "performance",
                "skipped": True,
                "reason": "not evaluated: the correctness stage failed",
            },
        ]
        self.assertEqual(validator.validate_instance_stdlib(document, "evaluation_report"), [])


class BaselineRuleTests(unittest.TestCase):
    """Guide §4.4: the B tier is scored against the expert dispatch, not the reference."""

    def baseline(self, tier):
        return {
            "schema_version": "1.0.0",
            "task_id": "example_b_v0",
            "tier": tier,
            "environment_snapshot": "musa-5f9d7b9dd1233a68",
            "measurement_protocol": {"warmup": 10, "measurements": 100, "rounds": 5, "statistic": "median"},
            "implementations": [
                "reference_model", "upstream_musa", "mudnn_fused",
                "naive_library_composition", "expert_dispatch",
            ],
            "results": [{"case_id": "hidden_gap_001", "implementation": "expert_dispatch", "status": "pass", "latency_ms": 0.1}],
            "scoring": {"speedup_denominator": "expert_dispatch", "correctness_gate": "all_hidden_cases"},
        }

    def test_a_b_baseline_on_the_expert_dispatch_passes(self):
        self.assertEqual(validator.validate_instance_stdlib(self.baseline("B_library"), "baseline"), [])

    def test_a_b_baseline_missing_the_expert_dispatch_is_rejected(self):
        document = self.baseline("B_library")
        document["implementations"] = ["reference_model", "upstream_musa", "mudnn_fused"]
        errors = validator.validate_instance_stdlib(document, "baseline")
        self.assertTrue(any("must not match" in error for error in errors), errors)

    def test_a_b_baseline_scored_against_anything_else_is_rejected(self):
        document = self.baseline("B_library")
        document["scoring"]["speedup_denominator"] = "reference_model"
        errors = validator.validate_instance_stdlib(document, "baseline")
        self.assertTrue(any("expert_dispatch" in error for error in errors), errors)

    def test_a_result_for_an_undeclared_implementation_is_rejected(self):
        document = self.baseline("B_library")
        document["results"].append({"case_id": "hidden_gap_001", "implementation": "mystery", "status": "pass"})
        errors = validator.validate_instance(document, "baseline", engine="stdlib")
        self.assertTrue(any("mystery" in error for error in errors), errors)


class SemanticsContractTests(unittest.TestCase):
    """Guide §4.2: the three conventions that make the goldens reproducible."""

    def sdpa_semantics(self):
        return {
            "schema_version": "1.0.0",
            "family": "sdpa_forward",
            "operation": "sdpa_forward",
            "tensors": [
                {"name": "q", "role": "input", "shape": ["B", "H_q", "S_q", "D"],
                 "dtypes": ["float16"], "layout": "BHSD", "stride": "contiguous"},
            ],
            "dynamic_axes": {"B": [1, 32], "D": [32, 256]},
            "abi": {"passing": "in_process_torch_tensors", "layout": "contiguous", "stream": "current_musa_stream"},
            "attention": {
                "scale": "case_value",
                "causal_alignment": "top_left",
                "window": {"left": "case.window_left", "right": "case.window_right", "unlimited_value": -1},
                "fully_masked_row": {"output": "zeros"},
            },
            "determinism": {"required": True, "seed_source": "case_manifest"},
        }

    def test_a_complete_sdpa_contract_passes(self):
        self.assertEqual(validator.validate_instance_stdlib(self.sdpa_semantics(), "semantics_contract"), [])

    def test_an_sdpa_contract_missing_its_abi_is_rejected(self):
        document = self.sdpa_semantics()
        del document["abi"]
        errors = validator.validate_instance_stdlib(document, "semantics_contract")
        self.assertTrue(any("abi" in error for error in errors), errors)

    def test_an_unknown_causal_alignment_is_rejected(self):
        document = self.sdpa_semantics()
        document["attention"]["causal_alignment"] = "middle"
        errors = validator.validate_instance_stdlib(document, "semantics_contract")
        self.assertTrue(any("middle" in error for error in errors), errors)

    def test_a_nonzero_window_unlimited_value_is_rejected(self):
        document = self.sdpa_semantics()
        document["attention"]["window"]["unlimited_value"] = 0
        errors = validator.validate_instance_stdlib(document, "semantics_contract")
        self.assertTrue(any("window" in error for error in errors), errors)

    def test_a_family_prose_contract_still_passes(self):
        document = {
            "schema_version": "1.0.0",
            "family": "sdpa_forward",
            "set_entry": "kb_l1_97",
            "layout": "BHSD, contiguous",
            "mask": "none",
            "determinism": "no dropout, so the reference is deterministic",
        }
        self.assertEqual(validator.validate_instance_stdlib(document, "semantics_contract"), [])


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
