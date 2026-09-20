import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename, module_name):
    """Load a tool by path so the tests never import torch."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


if __name__ == "__main__":
    unittest.main()
