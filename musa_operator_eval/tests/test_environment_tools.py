import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename, module_name):
    """Load a tool by path so the tests never import torch."""
    # The public flow lives in tools/ and the evaluation service in evaluator/,
    # and this loader is used for both, so the name decides which directory.
    for directory in (ROOT / "tools", ROOT / "evaluator"):
        if (directory / filename).is_file():
            break
    spec = importlib.util.spec_from_file_location(module_name, directory / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# A generated case runs the task's reference, which is a torch model, so the
# cross-tool test that needs one is skipped where torch is missing.
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_manifest_tool():
    return load_tool("make_private_manifest.py", "make_private_manifest")


def load_generator():
    return load_tool("generate_cases.py", "generate_cases")


def load_probe():
    return load_tool("probe_sdpa.py", "probe_sdpa")


def load_collector():
    return load_tool("collect_environment.py", "collect_environment")


class EnvironmentToolTests(unittest.TestCase):
    def test_capability_plan_has_unique_ids_and_required_dimensions(self):
        plan = json.loads((ROOT / "tasks" / "sdpa_forward_pilot" / "capability_plan.json").read_text(encoding="utf-8"))
        ids = [case["probe_id"] for case in plan["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 12)
        required = {"dtype", "layout", "B", "H_q", "H_kv", "S_q", "S_kv", "D", "causal", "window_left", "window_right"}
        for case in plan["cases"]:
            self.assertTrue(required <= case.keys())

    def test_unavailable_probe_is_explicit(self):
        result = load_probe().unavailable_results({"family": "sdpa_forward", "required_routes": ["x"]}, "no device")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["routes"]["x"], "not_probed")

    def test_declared_record_fields_are_actually_written(self):
        """The plan's record_fields must not drift from what the probe emits."""
        plan = json.loads((ROOT / "tasks" / "sdpa_forward_pilot" / "capability_plan.json").read_text(encoding="utf-8"))
        written = set(load_probe().empty_record(plan["cases"][0]).keys())
        for field in plan["record_fields"]:
            self.assertIn(field, written, f"plan declares {field!r} but the probe never writes it")

    def test_route_resolution_reads_the_dispatched_op(self):
        resolve_route = load_probe().resolve_route
        route, deciding = resolve_route({"scaled_dot_product_attention", "_scaled_dot_product_attention_flash_musa"})
        self.assertEqual(route, "fused_library")
        self.assertEqual(deciding, "_scaled_dot_product_attention_flash_musa")

        route, deciding = resolve_route({"scaled_dot_product_attention", "_scaled_dot_product_attention_math_musa"})
        self.assertEqual(route, "library_composition")
        self.assertEqual(deciding, "_scaled_dot_product_attention_math_musa")

        route, deciding = resolve_route({"scaled_dot_product_attention", "matmul", "softmax"})
        self.assertEqual(route, "custom_fallback")
        self.assertIn("matmul", deciding)

    def test_route_resolution_prefers_fused_over_composite(self):
        route, _ = load_probe().resolve_route({
            "_scaled_dot_product_attention_flash_musa", "matmul", "softmax", "_scaled_dot_product_attention_math_musa"})
        self.assertEqual(route, "fused_library")

    def test_route_resolution_stays_unverified_when_nothing_is_observed(self):
        route, deciding = load_probe().resolve_route(set())
        self.assertEqual(route, "unverified")
        self.assertIsNone(deciding)


class ComponentVersionParsingTests(unittest.TestCase):
    SINGLE = 'mudnn:\n{\n\t"version":\t"2.7.0",\n\t"commit id":\t"2fa2f81b"\n}'
    DOUBLE = ('musa_runtime:\n{\n\t"version":\t"3.1.0"\n}\n'
              'driver_dependency:\n{\n\t"commit id":\t"c64ecd8a"\n}')

    def test_single_block(self):
        blocks = load_collector().parse_component_version(self.SINGLE)
        self.assertEqual(list(blocks), ["mudnn"])
        self.assertEqual(blocks["mudnn"]["version"], "2.7.0")

    def test_every_block_is_kept(self):
        """musa_runtime_version prints two blocks; the second must not be dropped."""
        blocks = load_collector().parse_component_version(self.DOUBLE)
        self.assertEqual(list(blocks), ["musa_runtime", "driver_dependency"])
        self.assertEqual(blocks["driver_dependency"]["commit id"], "c64ecd8a")

    def test_unparsable_body_yields_no_blocks(self):
        self.assertEqual(load_collector().parse_component_version("not json at all"), {})

    def test_empty_output_yields_no_blocks(self):
        self.assertEqual(load_collector().parse_component_version(""), {})


class DeviceSummaryTests(unittest.TestCase):
    # `GPU Link Info` sits inside `PCI` in the gmi JSON.
    GPU = {
        "Index": "0", "Product Name": "MTT S4000", "GPU UUID": "bf1da7bf", "Serial Number": "MY1VE",
        "MTBios Version": "3.4.8",
        "PCI": {"Bus ID": "00000000:12:00.0", "Slot ID(Name)": "5(PEEB Slot 5)",
                "GPU Link Info": {"PCIe Generation": {"Max": "5", "Current": "5"},
                                  "Link Width": {"Max": "16x", "Current": "16x"}}},
        "FB Memory Spec": {"Type": "GDDR6", "Vendor": "Samsung", "Speed": "16000Mbps",
                           "Bandwidth": "768GBps", "Bus Width": "384bits"},
        "FB Memory Usage": {"Total": "49152MiB"},
        "Clocks": {"Graphics": "1665MHz"},
    }

    def test_pcie_is_read_from_inside_pci(self):
        summary = load_collector().summarize_device(self.GPU)
        self.assertEqual(summary["pcie_generation"], {"max": "5", "current": "5"})
        self.assertEqual(summary["pcie_link_width"], {"max": "16x", "current": "16x"})

    def test_identity_and_characteristics_are_kept(self):
        summary = load_collector().summarize_device(self.GPU)
        self.assertEqual(summary["uuid"], "bf1da7bf")
        self.assertEqual(summary["mtbios_version"], "3.4.8")
        self.assertEqual(summary["memory"]["bandwidth"], "768GBps")

    def test_missing_nested_blocks_do_not_raise(self):
        summary = load_collector().summarize_device({})
        self.assertIsNone(summary["pcie_generation"]["max"])
        self.assertIsNone(summary["uuid"])


class ArchitectureDerivationTests(unittest.TestCase):
    PROPERTIES = {"status": "ok", "major": 2, "minor": 2}

    def test_derived_from_torch_device_properties(self):
        self.assertEqual(load_collector().derive_architecture(self.PROPERTIES, None),
                         ("mp_22", "torch.musa device properties"))

    def test_override_wins(self):
        self.assertEqual(load_collector().derive_architecture(self.PROPERTIES, "mp_31"),
                         ("mp_31", "command line"))

    def test_absent_everywhere_is_not_guessed(self):
        self.assertEqual(load_collector().derive_architecture({"status": "unavailable"}, None), (None, None))


class ContainerAndRedactionTests(unittest.TestCase):
    def test_digest_is_recorded_when_the_platform_provides_one(self):
        with mock.patch.dict("os.environ", {"MUSA_EVAL_IMAGE_DIGEST": "sha256:abc"}, clear=False):
            container = load_collector().collect_container()
        self.assertEqual(container["image_digest"], "sha256:abc")
        self.assertNotIn("image_digest_unavailable_reason", container)

    def test_absent_digest_always_carries_a_reason(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            container = load_collector().collect_container()
        self.assertIsNone(container["image_digest"])
        self.assertTrue(container["image_digest_unavailable_reason"])

    def test_autodl_identity_is_captured(self):
        with mock.patch.dict("os.environ", {"AutoDLContainerUUID": "abc123", "AutoDLRegion": "sh-A1"}, clear=True):
            container = load_collector().collect_container()
        self.assertEqual(container["platform"], "AutoDL")
        self.assertEqual(container["identity"]["container_uuid"], "abc123")

    def test_redaction_drops_machine_identity_but_keeps_configuration(self):
        snapshot = {
            "snapshot_id": "musa-abc", "device_instance_id": "gpu-def", "device_instance": {"uuid": "x"},
            "host": {"kernel": "6.0"}, "container": {"image_digest": None}, "raw_probes": {"a": 1},
            "hardware": {"architecture": "mp_22"},
        }
        public = load_collector().redact(snapshot)
        for key in ("device_instance", "device_instance_id", "host", "container", "raw_probes"):
            self.assertNotIn(key, public)
        self.assertEqual(public["snapshot_id"], "musa-abc")
        self.assertEqual(public["hardware"]["architecture"], "mp_22")
        self.assertEqual(public["visibility"], "redacted")


class ProbeStatusTests(unittest.TestCase):
    def test_missing_binary_is_reported_as_unavailable_with_a_reason(self):
        result = load_collector().run(["definitely-not-a-real-binary-xyz"])
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("not on PATH", result["reason"])

    def test_failing_command_records_output_and_reason(self):
        result = load_collector().run(["false"])
        self.assertEqual(result["status"], "error")
        self.assertIn("exit code", result["reason"])


class PrivateManifestGoldenPathTests(unittest.TestCase):
    """The private manifest must name golden files the generator actually writes.

    §4.3 requires a golden path per case, and the manifest is the only place a
    hidden case's golden is described. The field was written but the layout it
    named (`golden/<case_id>/tensors.json`) was not the layout the generator
    produces (`<case_id>/golden/tensors.json`), so every path pointed at nothing.
    """

    # A hidden-case plan in the shape `make_private_manifest.py` now reads: the
    # maintainer supplies every case, including the performance and generalisation
    # ones, and the tool refuses a plan that is missing a role.
    EVIDENCE = {"cases": [
        {"case_id": "hidden_gap_001", "role": "gap", "tag": "boundary", "seed": 91, "dtype": "float16",
         "shape": {"B": 1, "H_q": 2, "H_kv": 2, "S_q": 8, "S_kv": 8, "D": 16},
         "attributes": {"scale": 0.25, "causal": False, "window_left": -1, "window_right": -1},
         "distribution": "normal", "expected_path": "library_composition",
         "library_gap": {"reason": "unsupported_shape", "expected_path": "library_composition",
                         "evidence": "recorded on device"}},
        {"case_id": "hidden_gap_002", "role": "gap", "tag": "boundary", "seed": 92, "dtype": "float16",
         "shape": {"B": 1, "H_q": 2, "H_kv": 2, "S_q": 16, "S_kv": 16, "D": 32},
         "attributes": {"scale": 0.1767766952966369, "causal": False, "window_left": -1, "window_right": -1},
         "distribution": "normal", "expected_path": "library_composition",
         "library_gap": {"reason": "unsupported_shape", "expected_path": "library_composition",
                         "evidence": "recorded on device"}},
        {"case_id": "hidden_gap_003", "role": "gap", "tag": "boundary", "seed": 93, "dtype": "float16",
         "shape": {"B": 1, "H_q": 2, "H_kv": 2, "S_q": 32, "S_kv": 32, "D": 64},
         "attributes": {"scale": 0.1, "causal": False, "window_left": -1, "window_right": -1},
         "distribution": "normal", "expected_path": "custom_fallback",
         "library_gap": {"reason": "semantic_mismatch", "expected_path": "custom_fallback",
                         "evidence": "recorded on device"}},
        {"case_id": "hidden_perf_001", "role": "perf", "tag": "perf", "seed": 94, "dtype": "float16",
         "shape": {"B": 1, "H_q": 2, "H_kv": 2, "S_q": 64, "S_kv": 64, "D": 16},
         "attributes": {"scale": 0.25, "causal": False, "window_left": -1, "window_right": -1},
         "distribution": "normal", "expected_path": "fused_library",
         },
        {"case_id": "hidden_probe_001", "role": "probe", "tag": "generalization", "seed": 95, "dtype": "float16",
         "shape": {"B": 1, "H_q": 2, "H_kv": 2, "S_q": 128, "S_kv": 128, "D": 16},
         "attributes": {"scale": 0.25, "causal": False, "window_left": -1, "window_right": -1},
         "distribution": "normal", "expected_path": "fused_library"},
    ]}

    # The task a plan belongs to. Only the fields the tool checks are needed: the
    # contract's required reasons, its id and tier, and -- through the caller's
    # `_public_cases` injection -- the public list it must not repeat.
    TASK = {
        "id": "test_task_b_v0",
        "tier": "B_library",
        "library_policy": {"required_gap_reasons": ["unsupported_shape", "semantic_mismatch"]},
        "admission": {"gap_reason_coverage": {"state": "covered"}},
        "target_environment": {"snapshot_id": "musa-test"},
        "_public_cases": [],
    }

    def manifest(self):
        return load_manifest_tool().build_manifest(self.EVIDENCE["cases"], dict(self.TASK))

    def test_every_case_names_its_golden_under_its_own_case_directory(self):
        manifest = self.manifest()
        self.assertTrue(manifest["cases"])
        for case in manifest["cases"]:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(case["golden"], f"{case['case_id']}/golden/tensors.json")

    @unittest.skipUnless(HAS_TORCH, "a case is built by running the task's reference, which needs torch")
    def test_the_golden_path_is_where_the_generator_writes(self):
        """Cross-tool: the manifest's layout and the generator's layout agree."""
        manifest = self.manifest()
        generator = load_generator()
        case = {"case_id": "hidden_gap_001", "seed": 5, "dtype": "float16",
                "shape": {"B": 1, "H_q": 2, "H_kv": 2, "S_q": 8, "S_kv": 8, "D": 16},
                "attributes": {"scale": 0.25, "causal": False, "window_left": -1, "window_right": -1},
                "distribution": "normal"}
        with tempfile.TemporaryDirectory() as temporary:
            generator.generate_case(case, Path(temporary))
            recorded = next(c["golden"] for c in manifest["cases"] if c["case_id"] == "hidden_gap_001")
            self.assertTrue((Path(temporary) / recorded).is_file(), recorded)

    def test_the_manifest_names_a_generator_a_consumer_can_recompute_with(self):
        manifest = self.manifest()
        tool = manifest["case_generation"]["tool"]
        self.assertTrue((ROOT.parent / tool).is_file(), f"{tool} does not exist")
        self.assertTrue(manifest["case_generation"]["deterministic"])

    def test_the_visibility_is_private(self):
        self.assertEqual(self.manifest()["visibility"], "private")

    def problems(self, plan, task=None):
        """The refusals a plan earns, as a list rather than as an exception.

        `check_plan` returns every problem at once so a maintainer sees all of
        them, which is why the tests assert on the list rather than on a raise.
        """
        return load_manifest_tool().check_plan(plan, dict(task or self.TASK))

    def test_too_few_gaps_is_refused(self):
        thin = [case for case in self.EVIDENCE["cases"]
                if case["role"] != "gap" or case["case_id"] == "hidden_gap_001"]
        self.assertTrue(any("at least 3" in problem for problem in self.problems(thin)))

    def test_one_reason_category_is_refused_unless_the_contract_records_it(self):
        """§4.3 asks for two categories, and most entry-scoped tasks reach one.

        A task whose reference fixes its scale, has no window and is causal has
        exactly one library boundary to find, so the shortfall is a property of the
        task. It is allowed only when the contract says so: a silent single-category
        set is indistinguishable from an unfinished one.
        """
        one_category = [
            {**case, "library_gap": {**case["library_gap"], "reason": "unsupported_shape"}}
            if case["role"] == "gap" else case
            for case in self.EVIDENCE["cases"]
        ]
        silent = dict(self.TASK)
        silent["admission"] = {"gap_reason_coverage": {"state": "covered"}}
        self.assertTrue(any("reason category" in problem for problem in self.problems(one_category, silent)))

        recorded = dict(self.TASK)
        recorded["admission"] = {"gap_reason_coverage": {"state": "open"}}
        # The contract has to agree: a task that reaches one category must require
        # one, or the required-reason check fires on the same plan for a second and
        # unrelated reason.
        recorded["library_policy"] = {**self.TASK["library_policy"],
                                      "required_gap_reasons": ["unsupported_shape"]}
        self.assertEqual(self.problems(one_category, recorded), [])

    def test_a_required_reason_that_no_gap_produces_is_refused(self):
        """A reason the contract requires and the set never produces is a broken promise."""
        without_the_semantic_gap = [case for case in self.EVIDENCE["cases"]
                                    if case["case_id"] != "hidden_gap_003"]
        self.assertTrue(any("requires gap reasons" in problem
                            for problem in self.problems(without_the_semantic_gap)))

    def test_a_gap_with_no_evidence_is_refused(self):
        """An unmeasured gap is a guess wearing a measurement's authority."""
        hollowed = [
            {**case, "library_gap": {**case["library_gap"], "evidence": ""}}
            if case["role"] == "gap" else case
            for case in self.EVIDENCE["cases"]
        ]
        self.assertTrue(any("no evidence" in problem for problem in self.problems(hollowed)))

    def test_a_reason_outside_the_vocabulary_is_refused(self):
        strayed = [
            {**case, "library_gap": {**case["library_gap"], "reason": "unsupported_dtype"}}
            if case["role"] == "gap" else case
            for case in self.EVIDENCE["cases"]
        ]
        self.assertTrue(any("unsupported_dtype" in problem for problem in self.problems(strayed)))

    def test_a_hidden_shape_repeating_a_public_one_is_refused(self):
        """Otherwise passing the public set would be enough to pass the hidden one."""
        task = dict(self.TASK)
        task["_public_cases"] = [{"shape": self.EVIDENCE["cases"][0]["shape"]}]
        self.assertTrue(any("already in the public list" in problem for problem in self.problems(self.EVIDENCE["cases"], task)))

    def test_a_plan_without_a_performance_case_is_refused(self):
        """A set of gaps measures coverage and not speed, and the gate is a speed ratio."""
        no_perf = [case for case in self.EVIDENCE["cases"] if case["role"] != "perf"]
        self.assertTrue(any("no perf case" in problem for problem in self.problems(no_perf)))

    def test_a_plan_without_a_generalisation_probe_is_refused(self):
        no_probe = [case for case in self.EVIDENCE["cases"] if case["role"] != "probe"]
        self.assertTrue(any("no probe case" in problem for problem in self.problems(no_probe)))

    def test_a_free_text_tag_is_refused(self):
        """§4.3 fixes the tag vocabulary so the set can be counted by category."""
        renamed = [dict(case, tag="head_dim_beyond_fused_bound") for case in self.EVIDENCE["cases"]]
        self.assertTrue(any("fixed tags" in problem for problem in self.problems(renamed)))

    def test_a_plan_without_any_cases_is_refused(self):
        with self.assertRaises(ValueError):
            load_manifest_tool().load_plan({})

    def test_the_plan_role_is_not_copied_into_the_manifest(self):
        """`role` is how the tool sorts the plan; a case has a tag, not a role."""
        for case in self.manifest()["cases"]:
            with self.subTest(case=case["case_id"]):
                self.assertNotIn("role", case)


class PrivateManifestSchemaTests(unittest.TestCase):
    """The private manifest has to satisfy `schemas/case_manifest.schema.json` (§4.8).

    Two defects lived here at once: the manifest carried no `case_generation`
    block, and its gap cases were labelled with free text instead of §4.3's tag
    vocabulary. Both are visible from `tools/validate_schemas.py`.
    """

    # §4.3's tag enum, shared by the public and the hidden list.
    CASE_TAGS = {"smoke", "correctness", "boundary", "non_aligned", "extreme", "perf", "generalization"}

    EVIDENCE = PrivateManifestGoldenPathTests.EVIDENCE
    TASK = PrivateManifestGoldenPathTests.TASK

    def manifest(self):
        return load_manifest_tool().build_manifest(self.EVIDENCE["cases"], dict(self.TASK))

    def validator(self):
        path = ROOT / "evaluator" / "validate_schemas.py"
        if not path.is_file():
            self.skipTest("validate_schemas.py is not present")
        return load_tool("validate_schemas.py", "validate_schemas")

    def test_every_case_tag_is_in_the_fixed_vocabulary(self):
        for case in self.manifest()["cases"]:
            with self.subTest(case=case["case_id"]):
                self.assertIn(case["tag"], self.CASE_TAGS)

    def test_the_manifest_declares_its_case_generation(self):
        generation = self.manifest().get("case_generation")
        self.assertIsNotNone(generation, "§4.8 needs the block, and the schema requires it")
        self.assertTrue(generation["deterministic"])

    def test_the_manifest_validates_against_the_case_manifest_schema(self):
        validator = self.validator()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.private.json"
            path.write_text(json.dumps(self.manifest()), encoding="utf-8")
            self.assertEqual(validator.validate_document(path, "case_manifest", engine="stdlib"), [])


if __name__ == "__main__":
    unittest.main()
