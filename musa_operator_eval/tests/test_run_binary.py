"""Tests for the compiled path's driver (§4.6's ABI, §4.7's stages).

The point of these is that the compiled path is held to the same stages as the
Python one without a device: the pieces that need MUSA are the compile and the
launch, and everything around them -- the region rule, the link whitelist, the
tensor comparison, the trace check, the report shape -- is exercised here against
fixtures. A fake build script stands in for mcc and a fake runner for the binary,
so a change that breaks the driver breaks a test rather than a device run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO_TOP = ROOT.parent
TEST_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_binary = load_module("musa_run_binary_tests", ROOT / "evaluator" / "run_binary.py")

PILOT = ROOT / "tasks" / "sdpa_forward_pilot"
A_TASK = ROOT / "tasks" / "kb_l1_97_a"


def write_tensor_dir(directory: Path, tensors, reference: str = "fixture") -> None:
    """Write a manifest and its blobs in the frozen ABI's shape."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, dtype, shape, values in tensors:
        array = np.asarray(values, dtype=np.float32).reshape(shape)
        if dtype == "float32":
            raw = array.astype("<f4").tobytes()
        elif dtype == "float16":
            raw = array.astype("<f2").tobytes()
        elif dtype == "bfloat16":
            raw = (array.view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raise AssertionError(dtype)
        (directory / f"{name}.bin").write_bytes(raw)
        records.append(
            {"name": name, "file": f"{name}.bin", "dtype": dtype, "shape": list(shape), "nbytes": len(raw)}
        )
    (directory / "tensors.json").write_text(
        json.dumps({"schema_version": "1.0.0", "reference": reference, "tensors": records}, indent=2),
        encoding="utf-8",
    )


def failing_task() -> dict:
    return {"id": "fixture_a", "tier": "A_kernel", "starter": {"cpp": {"regions": ["device_kernel"]}}}


def library_task() -> dict:
    return {
        "id": "fixture_b",
        "tier": "B_library",
        "library_policy": {
            "allowed_libraries": ["libmudnn", "libmusa"],
            "required_trace_fields": ["case_id", "selected_path", "probe_status"],
            "trace_env_var": "KB_TRACE",
            "case_id_env_var": "KB_CASE",
        },
        "starter": {
            "cpp": {
                "regions": ["probe_and_dispatch", "custom_fallback", "dispatch_trace", "host_launch", "device_kernel"]
            }
        },
    }


B_REGIONS = ["probe_and_dispatch", "custom_fallback", "dispatch_trace", "host_launch", "device_kernel"]


def with_regions(bodies: dict, regions=None) -> str:
    """Wrap a region body in the markers the inventory declares, so a test that is
    about one region does not trip the marker-versus-inventory check."""
    regions = regions or ["device_kernel"]
    return "".join(
        f"// --- BEGIN {name} ---\n{bodies.get(name, '')}// --- END {name} ---\n" for name in regions
    )


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class SourceAuditTests(unittest.TestCase):
    """§4.6's region rule, which is the one clause a compiled submission can be held to."""

    def test_a_kernel_may_not_name_the_upstream_stacks(self):
        source = with_regions({"device_kernel": "mudnnHandle_t handle;\n"})
        findings = run_binary.audit_compiled_source(failing_task(), source)
        self.assertTrue(findings)
        self.assertIn("prohibited list", findings[0])

    def test_a_comment_may_name_them(self):
        source = with_regions({"device_kernel": "// no muDNN here\nfloat x = 1.0f;\n"})
        self.assertEqual(run_binary.audit_compiled_source(failing_task(), source), [])

    def test_a_library_call_belongs_in_the_dispatch_region(self):
        source = with_regions(
            {"probe_and_dispatch": "mudnnHandle_t handle = mudnnCreate();\n", "device_kernel": "__global__ void k() {}\n"},
            B_REGIONS,
        )
        self.assertEqual(run_binary.audit_compiled_source(library_task(), source), [])

    def test_a_library_call_in_the_kernel_region_is_a_finding(self):
        source = with_regions(
            {"probe_and_dispatch": "int probe() { return 0; }\n", "device_kernel": "__global__ void k() { mudnnFusedAttention(0); }\n"},
            B_REGIONS,
        )
        findings = run_binary.audit_compiled_source(library_task(), source)
        self.assertEqual(len(findings), 1)
        self.assertIn("device_kernel", findings[0])

    def test_a_whitelisted_header_may_be_included_anywhere(self):
        """An include is a declaration; the call it enables is still checked where it sits.

        An agent writing a compiled submission puts its includes at the top of the file,
        which is outside every region, and a rule that failed it for that would fail an
        honest submission for a habit instead of for a call.
        """
        source = (
            "#include <mudnn.h>\n#include \"mudnn_nn.h\"\n"
            + with_regions({"probe_and_dispatch": "int probe() { return 0; }\n"}, B_REGIONS)
        )
        self.assertEqual(run_binary.audit_compiled_source(library_task(), source), [])

    def test_a_header_outside_the_whitelist_is_still_a_finding(self):
        source = (
            "#include <torch/extension.h>\n"
            + with_regions({"probe_and_dispatch": "int probe() { return 0; }\n"}, B_REGIONS)
        )
        findings = run_binary.audit_compiled_source(library_task(), source)
        self.assertEqual(len(findings), 1)
        self.assertIn("region none", findings[0])

    def test_an_include_does_not_excuse_a_call_in_the_kernel_region(self):
        source = (
            "#include <mudnn.h>\n"
            + with_regions(
                {
                    "probe_and_dispatch": "int probe() { return 0; }\n",
                    "device_kernel": "__global__ void k() { mudnnFusedAttention(0); }\n",
                },
                B_REGIONS,
            )
        )
        findings = run_binary.audit_compiled_source(library_task(), source)
        self.assertEqual(len(findings), 1)
        self.assertIn("device_kernel", findings[0])

    def test_the_a_tier_may_not_include_the_stack_either(self):
        source = "#include <mudnn.h>\n" + with_regions({"device_kernel": "__global__ void k() {}\n"})
        findings = run_binary.audit_compiled_source(failing_task(), source)
        self.assertEqual(len(findings), 1)
        self.assertIn("prohibited list", findings[0])

    def test_the_worked_compiled_submission_passes_its_own_audit(self):
        """The B-tier answer this repository keeps under `private/` is held to the rule too."""
        answer = ROOT / "private" / "scaled_dot_product_attention_b_v0" / "compiled" / "kernel.mu"
        if not answer.is_file():
            self.skipTest("the maintainer tree is not mounted")
        task = json.loads((ROOT / "tasks" / "kb_l1_97_b" / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(
            run_binary.audit_compiled_source(task, answer.read_text(encoding="utf-8")), []
        )

    def test_the_markers_must_match_the_inventory(self):
        source = with_regions({"device_kernel": ""}, ["device_kernel", "host_launch"])
        findings = run_binary.audit_compiled_source(failing_task(), source)
        self.assertTrue(any("the inventory says" in finding for finding in findings))

    def test_an_unclosed_region_is_reported(self):
        source = "// --- BEGIN device_kernel ---\n// --- BEGIN host_launch ---\n// --- END host_launch ---\n"
        findings = run_binary.audit_compiled_source(failing_task(), source)
        self.assertTrue(any("still open" in finding for finding in findings))

    def test_the_shipped_starters_pass_their_own_audit(self):
        """The skeletons this repository ships are what §4.6's rule is written against."""
        for directory in sorted((ROOT / "tasks").iterdir()):
            task_path = directory / "task.json"
            if not task_path.is_file():
                continue
            task = json.loads(task_path.read_text(encoding="utf-8"))
            source = (directory / "starter" / "cpp" / "kernel.mu").read_text(encoding="utf-8")
            with self.subTest(task=task["id"]):
                self.assertEqual(run_binary.audit_compiled_source(task, source), [])


class LibraryResolutionTests(unittest.TestCase):
    """The whitelist is written as operations; the linker needs files."""

    def test_resolvable_and_unresolvable_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "lib").mkdir()
            (home / "lib" / "libmudnn.so.2.7.0.0").write_bytes(b"")
            (home / "lib" / "libmudnn.so").write_bytes(b"")
            linkable, missing = run_binary.resolve_libraries(library_task(), home)
            self.assertEqual(linkable, ["mudnn"])
            self.assertEqual(missing, ["libmusa"])

    def test_an_operation_that_is_not_a_library_is_not_an_error(self):
        task = {"library_policy": {"allowed_libraries": ["SDPA", "muDNN"]}}
        with tempfile.TemporaryDirectory() as tmp:
            linkable, missing = run_binary.resolve_libraries(task, Path(tmp))
            self.assertEqual(linkable, [])
            self.assertEqual(sorted(missing), ["SDPA", "muDNN"])


class BuildStageTests(unittest.TestCase):
    """The build stage maps the script's exit onto the report's vocabulary."""

    def _task_with_script(self, script: Path) -> dict:
        return {"id": "fixture", "tier": "A_kernel", "starter": {"cpp": {"build": str(script), "regions": ["device_kernel"]}}}

    def test_a_successful_build_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "build.sh"
            script.write_text("#!/bin/bash\necho built\n", encoding="utf-8")
            script.chmod(0o755)
            result = run_binary.build_submission(self._task_with_script(script), Path(tmp), Path(tmp), Path(tmp) / "build", [])
            self.assertTrue(result["passed"])
            self.assertFalse(result["environment_missing"])

    def test_a_failed_build_is_a_build_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "build.sh"
            script.write_text("#!/bin/bash\necho 'ninja: error' >&2\nexit 1\n", encoding="utf-8")
            script.chmod(0o755)
            result = run_binary.build_submission(self._task_with_script(script), Path(tmp), Path(tmp), Path(tmp) / "build", [])
            self.assertFalse(result["passed"])
            self.assertIn("ninja", result["detail"])

    def test_a_missing_device_stack_is_an_environment_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "build.sh"
            script.write_text("#!/bin/bash\necho 'mcc: not found' >&2\nexit 127\n", encoding="utf-8")
            script.chmod(0o755)
            result = run_binary.build_submission(self._task_with_script(script), Path(tmp), Path(tmp), Path(tmp) / "build", [])
            self.assertFalse(result["passed"])
            self.assertTrue(result["environment_missing"])

    def test_a_task_without_a_compiled_starter_is_refused(self):
        result = run_binary.build_submission({"starter": {}}, Path("."), Path("."), Path("."), [])
        self.assertFalse(result["passed"])
        self.assertIsNone(result["script"])


class TensorComparisonTests(unittest.TestCase):
    def test_a_value_inside_the_tolerance_passes(self):
        expected = {"output": {"dtype": "float32", "shape": [2], "values": np.array([1.0, 2.0], dtype=np.float32)}}
        produced = {"output": {"dtype": "float32", "shape": [2], "values": np.array([1.005, 1.995], dtype=np.float32)}}
        passed, worst, errors = run_binary.compare_tensors(expected, produced, {"atol": 0.01, "rtol": 0.0})
        self.assertTrue(passed, errors)
        self.assertAlmostEqual(worst, 0.005, places=6)

    def test_a_value_outside_the_tolerance_fails_and_names_the_count(self):
        expected = {"output": {"dtype": "float32", "shape": [2], "values": np.array([1.0, 2.0], dtype=np.float32)}}
        produced = {"output": {"dtype": "float32", "shape": [2], "values": np.array([1.5, 2.0], dtype=np.float32)}}
        passed, _, errors = run_binary.compare_tensors(expected, produced, {"atol": 0.01, "rtol": 0.0})
        self.assertFalse(passed)
        self.assertIn("1 of 2 values", errors[0])

    def test_a_missing_or_extra_tensor_is_an_error(self):
        expected = {"output": {"dtype": "float32", "shape": [1], "values": np.array([1.0], dtype=np.float32)}}
        passed, _, errors = run_binary.compare_tensors(expected, {}, {"atol": 0.1, "rtol": 0.1})
        self.assertFalse(passed)
        self.assertIn("the output does not", errors[0])
        passed, _, errors = run_binary.compare_tensors(expected, dict(expected, spare={"dtype": "float32", "shape": [1], "values": np.array([0.0], dtype=np.float32)}), {})
        self.assertFalse(passed)
        self.assertIn("golden does not", errors[0])

    def test_bf16_round_trips_through_the_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_tensor_dir(directory, [("output", "bfloat16", (2,), [1.0, -2.0])])
            tensors = run_binary.read_manifest(directory)
            np.testing.assert_allclose(tensors["output"]["values"], [1.0, -2.0])


FAKE_RUNNER = """#!/usr/bin/env python3
\"\"\"A stand-in for the compiled runner: the same ABI, no device.\"\"\"
import json, os, pathlib, struct, sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if os.environ.get("FAKE_EXIT"):
    print("kernel_entry returned 1", file=sys.stderr)
    raise SystemExit(int(os.environ["FAKE_EXIT"]))
values = [float(value) for value in os.environ.get("FAKE_VALUES", "1 2").split()]
marker = dst.parent / "ran"
if os.environ.get("FAKE_RERUN_SHIFT") and marker.exists():
    values = [value + float(os.environ["FAKE_RERUN_SHIFT"]) for value in values]
marker.touch()
raw = b"".join(struct.pack("<f", value) for value in values)
(dst / "output.bin").write_bytes(raw)
(dst / "tensors.json").write_text(json.dumps({
    "schema_version": "1.0.0",
    "reference": "fake_runner",
    "tensors": [{"name": "output", "file": "output.bin", "dtype": "float32",
                 "shape": [len(values)], "nbytes": len(raw)}],
}))
if os.environ.get("FAKE_TIMING"):
    print(json.dumps({"warmup": int(os.environ.get("FAKE_WARMUP", 0)),
                      "repeat": int(os.environ.get("FAKE_REPEAT", 3)),
                      "discarded": int(os.environ.get("FAKE_DISCARDED", 1)),
                      "timer": os.environ.get("FAKE_TIMER", "musa_event"),
                      "l2_thrash_bytes": int(os.environ.get("FAKE_THRASH", 268435456)),
                      "per_call_ms": [float(v) for v in os.environ["FAKE_TIMING"].split()]}))
if os.environ.get("FAKE_WARM_TIMING"):
    print(json.dumps({"warmup": 0, "repeat": 2, "discarded": 0, "timer": "host_clock",
                      "l2_thrash_bytes": 0,
                      "per_call_ms": [float(v) for v in os.environ["FAKE_WARM_TIMING"].split()]}))
if os.environ.get("KB_TRACE") and not os.environ.get("FAKE_NO_TRACE"):
    with open(os.environ["KB_TRACE"], "a", encoding="utf-8") as trace:
        trace.write(json.dumps({
            "case_id": os.environ.get("KB_CASE"),
            "selected_path": os.environ.get("FAKE_SELECTED", "fused_library"),
            "probe_status": "accepted",
        }) + "\\n")
"""


class ToleranceResolutionTests(unittest.TestCase):
    """One bar per task, resolved from the contract the way the framework resolves it."""

    def test_a_declared_max_abs_error_that_is_looser_becomes_the_bar(self):
        task = {"tolerances": {"atol": 0.01, "rtol": 0.01, "max_abs_error": 0.05}}
        resolved = run_binary.resolve_case_tolerance(task, "fp32")
        self.assertEqual((resolved["atol"], resolved["rtol"]), (0.05, 0.05))
        self.assertEqual(resolved["source"], "task_contract.tolerances.max_abs_error")

    def test_a_declared_max_abs_error_may_not_tighten_the_frameworks_floor(self):
        task = {"tolerances": {"max_abs_error": 1e-9}}
        resolved = run_binary.resolve_case_tolerance(task, "fp16")
        self.assertEqual((resolved["atol"], resolved["rtol"]), (1e-2, 1e-2))

    def test_without_a_declaration_the_frameworks_floor_is_the_bar(self):
        resolved = run_binary.resolve_case_tolerance({}, "fp16")
        self.assertEqual(resolved["atol"], 1e-2)
        self.assertEqual(resolved["source"], "kernelbench.eval.get_tolerance_for_precision")
        self.assertEqual(run_binary.resolve_case_tolerance({}, "fp32")["atol"], 1e-4)

    def test_an_unknown_precision_falls_back_to_the_looser_floor(self):
        """Guessing low would grade a submission against a bar nobody declared."""
        self.assertEqual(run_binary.resolve_case_tolerance({}, "fp8")["atol"], 1e-2)

    @unittest.skipUnless(shutil.which("python") and _torch_available(), "requires the framework's torch")
    def test_the_replica_matches_the_framework(self):
        """The replicated floors must be the framework's, or grading is re-based silently."""
        import kernelbench.eval as framework

        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                self.assertEqual(
                    run_binary.FRAMEWORK_TOLERANCES[precision],
                    framework.get_tolerance_for_precision(precision),
                )
        for declared in (0.05, 1e-9, None):
            with self.subTest(declared=declared):
                self.assertEqual(
                    run_binary.resolve_case_tolerance({"tolerances": {"max_abs_error": declared}}, "fp16")["atol"],
                    framework.resolve_tolerance("fp16", declared),
                )

    def test_the_report_names_the_bar_it_used(self):
        task = {"tolerances": {"max_abs_error": 0.05}}
        self.assertEqual(run_binary.resolve_case_tolerance(task, "fp16")["atol"], 0.05)
        # The comparison itself is one place; a case can no longer declare another.
        passed, _, errors = run_binary.compare_tensors(
            {"output": {"dtype": "float32", "shape": [1], "values": np.array([1.0], dtype=np.float32)}},
            {"output": {"dtype": "float32", "shape": [1], "values": np.array([1.03], dtype=np.float32)}},
            run_binary.resolve_case_tolerance(task, "fp16"),
        )
        self.assertTrue(passed, errors)


class CaseRunTests(unittest.TestCase):
    """One case end to end against a fake runner: no device, real ABI."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)
        self._environment = dict(os.environ)
        self.addCleanup(self._restore_environment)
        self.generated, self.case = self._fixtures()

    def _restore_environment(self):
        os.environ.clear()
        os.environ.update(self._environment)
        self.temporary.cleanup()

    def _fixtures(self, tag: str = "correctness") -> tuple:
        generated = self.tmp / "generated"
        case_id = "correctness_001"
        write_tensor_dir(generated / case_id / "input", [("input_0", "float32", (2,), [1.0, 2.0])])
        write_tensor_dir(
            generated / case_id / "golden", [("output", "float32", (2,), [1.0, 2.0])], reference="cpu_fp64_reference"
        )
        case = {
            "case_id": case_id,
            "tag": tag,
            "golden": f"{case_id}/golden/tensors.json",
        }
        return generated, case

    TOLERANCE = {"atol": 0.01, "rtol": 0.01, "source": "fixture"}

    def _runner(self) -> Path:
        path = self.tmp / "fake_runner"
        path.write_text(FAKE_RUNNER, encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_a_matching_output_passes_and_reports_stability(self):
        row = run_binary.run_binary_case(failing_task(), self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertTrue(row["passed"], row)
        self.assertEqual(
            row["stability"],
            {"reruns": 1, "agrees": True, "reason": "the compiled runner reproduced its own output byte for byte"},
        )
        self.assertEqual(row["max_abs_error"], 0.0)

    def test_a_disagreeing_output_reports_the_tolerance(self):
        os.environ["FAKE_VALUES"] = "1 2.5"
        row = run_binary.run_binary_case(failing_task(), self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertFalse(row["passed"])
        self.assertIn("outside atol", " ".join(row["errors"]))

    def test_a_runner_that_changes_its_answer_on_a_rerun_fails_stability(self):
        os.environ["FAKE_RERUN_SHIFT"] = "1"
        row = run_binary.run_binary_case(failing_task(), self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertFalse(row["passed"])
        self.assertFalse(row["stability"]["agrees"])
        self.assertIn("different bytes", row["stability"]["reason"])

    def test_switching_the_rerun_off_is_reported_rather_than_implied(self):
        row = run_binary.run_binary_case(failing_task(), self._runner(), self.case, self.generated, stability_reruns=0, tolerance=self.TOLERANCE)
        self.assertTrue(row["passed"], row)
        self.assertEqual(row["stability"]["reruns"], 0)
        self.assertIn("switched off", row["stability"]["reason"])

    def test_a_case_without_generated_input_is_skipped_with_a_reason(self):
        shutil.rmtree(self.generated / self.case["case_id"] / "input")
        row = run_binary.run_binary_case(failing_task(), self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertTrue(row["skipped"])
        self.assertIn("no generated input", row["reason"])

    def test_a_runner_that_exits_nonzero_is_reported(self):
        os.environ["FAKE_EXIT"] = "3"
        row = run_binary.run_binary_case(failing_task(), self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertFalse(row["passed"])
        self.assertIn("exited 3", row["errors"][0])

    def test_the_b_tier_trace_is_checked_against_the_expected_path(self):
        self.case["expected_path"] = "fused_library"
        runner = self._runner()
        row = run_binary.run_binary_case(library_task(), runner, self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertTrue(row["dispatch_trace_passed"], row)
        os.environ["FAKE_SELECTED"] = "custom_fallback"
        row = run_binary.run_binary_case(library_task(), runner, self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertFalse(row["passed"])
        self.assertFalse(row["dispatch_trace_passed"])
        self.assertIn("the case expects", " ".join(row["dispatch_trace_errors"]))

    def test_a_missing_trace_is_a_trace_failure_not_a_pass(self):
        self.case["expected_path"] = "fused_library"
        os.environ["FAKE_NO_TRACE"] = "1"
        row = run_binary.run_binary_case(library_task(), self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertFalse(row["passed"])
        self.assertIn("wrote no dispatch trace", row["dispatch_trace_errors"][0])

    def test_a_trace_missing_a_required_field_is_a_failure(self):
        self.case["expected_path"] = "fused_library"
        task = library_task()
        task["library_policy"]["required_trace_fields"] = ["case_id", "selected_path", "probe_status", "kernel_time_ms"]
        row = run_binary.run_binary_case(task, self._runner(), self.case, self.generated, tolerance=self.TOLERANCE)
        self.assertFalse(row["passed"])
        self.assertIn("missing the required field kernel_time_ms", " ".join(row["dispatch_trace_errors"]))


MUSA_BIN = Path("/usr/local/musa/bin")


def _mcc_available() -> bool:
    return (MUSA_BIN / "mcc").exists()


@unittest.skipUnless(_mcc_available(), "requires the MUSA toolchain: this is the one check that must compile")
class LinkWhitelistIntegrationTests(unittest.TestCase):
    """§4.7's B-only link check, exercised through the driver on a real artifact.

    The checker has unit tests; what those cannot show is that the stage stops a run.
    A submission that links something outside the whitelist has to fail before a
    single case is graded, or the run scores an answer to the wrong question.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _package_linking_outside_the_whitelist(self) -> tuple:
        """A package whose build succeeds and produces an artifact that links zlib.

        zlib is a real dependency of a real executable here -- the program calls
        `zlibVersion` -- so the dynamic section names `libz.so.1` rather than merely
        mentioning it, and the contract's whitelist (libmusa, libmudnn) does not
        allow it.
        """
        package = self.tmp / "kb_l3_43_b"
        shutil.copytree(ROOT / "tasks" / "kb_l3_43_b", package)
        task = json.loads((package / "task.json").read_text(encoding="utf-8"))
        task.pop("target_environment", None)
        (self.tmp / "program.cc").write_text(
            'extern "C" const char* zlibVersion(void);\n'
            "int main() {\n"
            "  const char* version = zlibVersion();\n"
            "  return version && version[0] ? 0 : 1;\n"
            "}\n",
            encoding="utf-8",
        )
        script = self.tmp / "build.sh"
        script.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "BUILD_DIR=build\n"
            "while [ $# -gt 0 ]; do case \"$1\" in --build-dir) BUILD_DIR=\"$2\"; shift 2 ;; *) shift ;; esac; done\n"
            "mkdir -p \"$BUILD_DIR\"\n"
            f"${{MUSA_HOME:-/usr/local/musa}}/bin/mcc -O2 {self.tmp / 'program.cc'} -o \"$BUILD_DIR/runner\" -lz\n"
            "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        task["starter"]["cpp"]["build"] = str(script)
        (package / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
        submission = self.tmp / "submission"
        submission.mkdir(exist_ok=True)
        (submission / "kernel.mu").write_text(
            "// A submission whose build script links a library the contract does not\n"
            "// allow. The regions are the B inventory's, so the source audit passes and\n"
            "// the run reaches the stage this test is about.\n"
            "// --- BEGIN probe_and_dispatch ---\n// --- END probe_and_dispatch ---\n"
            "// --- BEGIN custom_fallback ---\n// --- END custom_fallback ---\n"
            "// --- BEGIN dispatch_trace ---\n// --- END dispatch_trace ---\n"
            "// --- BEGIN host_launch ---\n// --- END host_launch ---\n",
            encoding="utf-8",
        )
        manifest = self.tmp / "cases.json"
        manifest.write_text(json.dumps({"cases": []}), encoding="utf-8")
        return package, submission, manifest

    def test_a_submission_linking_outside_the_whitelist_fails_before_any_case_runs(self):
        package, submission, manifest = self._package_linking_outside_the_whitelist()
        code, report = run_binary.evaluate_binary_task(
            task_dir=package,
            submission=submission,
            cases_manifest=manifest,
            generated_dir=self.tmp / "generated",
            build_dir=self.tmp / "build",
            precision="fp16",
        )
        self.assertEqual(code, run_binary.exit_codes.EXIT_LINK_WHITELIST_VIOLATION, report)
        self.assertTrue(report["build"]["passed"], report["build"])
        self.assertIn("link_whitelist_check", report["stages"])
        self.assertNotIn("evaluation", report["stages"])
        linked = [library for artifact in report["link_whitelist"]["artifacts"] for library in artifact.get("needed", [])]
        self.assertIn("libz.so.1", linked, "the fixture did not actually link zlib")


class PrecisionGuardTests(unittest.TestCase):
    """§4.4's precision guard, on the compiled path.

    A run at a precision the task was not measured at reports a different task's
    behaviour, so it is refused before any device work starts. The guard read a key
    no contract has and therefore never fired; this is the test that would have said so.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _run(self, package_name: str, precision: str):
        package = self.tmp / package_name
        shutil.copytree(ROOT / "tasks" / package_name, package)
        task = json.loads((package / "task.json").read_text(encoding="utf-8"))
        task.pop("target_environment", None)
        (package / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
        submission = self.tmp / f"submission_{precision}"
        submission.mkdir()
        (submission / "kernel.mu").write_text("// --- BEGIN host_launch ---\n// --- END host_launch ---\n", encoding="utf-8")
        manifest = self.tmp / f"cases_{precision}.json"
        manifest.write_text(json.dumps({"cases": []}), encoding="utf-8")
        return run_binary.evaluate_binary_task(
            task_dir=package,
            submission=submission,
            cases_manifest=manifest,
            generated_dir=self.tmp / "generated",
            build_dir=self.tmp / f"build_{precision}",
            precision=precision,
        )

    def test_a_b_tier_run_at_float32_is_refused(self):
        code, report = self._run("kb_l3_43_b", "fp32")
        self.assertEqual(code, run_binary.exit_codes.EXIT_PRECISION_CONTRACT_MISMATCH, report)
        self.assertEqual(report["stages"], [])

    def test_an_a_tier_run_at_float16_is_refused(self):
        code, report = self._run("kb_l1_97_a", "fp16")
        self.assertEqual(code, run_binary.exit_codes.EXIT_PRECISION_CONTRACT_MISMATCH, report)

    def test_the_precisions_the_contracts_permit_are_accepted(self):
        for package_name, precision in (("kb_l3_43_b", "fp16"), ("kb_l1_97_a", "fp32")):
            with self.subTest(package=package_name, precision=precision):
                code, report = self._run(package_name, precision)
                self.assertNotEqual(code, run_binary.exit_codes.EXIT_PRECISION_CONTRACT_MISMATCH, report)


class TimingTests(unittest.TestCase):
    """What `runtime` means on the compiled path, and why it is not the wall clock.

    The runner owns the clock and reports one number per timed call; the driver owns
    the statistic. Both halves are here: the mean and the report reader as pure
    functions, and one case end to end with a runner that reports times.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_statistic_is_the_framework_s_mean_in_the_format_it_reports(self):
        """`get_timing_stats` renders `f"{mean:.3g}"`, so this reproduces that."""
        self.assertEqual(run_binary.framework_mean([1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(run_binary.framework_mean([1.0]), 1.0)
        # Three significant digits, not three decimals: 1.2719768 is reported as 1.27.
        self.assertEqual(run_binary.framework_mean([1.2719768, 1.2719768]), 1.27)
        self.assertEqual(run_binary.framework_mean([0.1288391]), 0.129)

    def test_the_runner_report_is_read_from_its_last_json_line(self):
        stdout = ('noise\n{"warmup": 1, "repeat": 2, "discarded": 1, "timer": "musa_event", '
                  '"l2_thrash_bytes": 268435456, "per_call_ms": [1.0, 3.0]}\n')
        report = run_binary.read_timing(stdout)
        self.assertEqual(report["repeat"], 2)
        self.assertEqual(report["per_call_ms"], [1.0, 3.0])
        self.assertEqual(report["timer"], "musa_event")
        self.assertEqual(report["cache"], "cold")
        self.assertIsNone(run_binary.read_timing("no report at all\n"))
        self.assertIsNone(run_binary.read_timing('{"per_call_ms": []}\n'))
        self.assertIsNone(run_binary.read_timing('{"per_call_ms": ["slow"]}\n'))

    def test_a_runner_that_cannot_flush_says_the_number_is_warm(self):
        """A protocol is recorded, not assumed: no flush buffer, no cold-cache claim."""
        report = run_binary.read_timing(
            '{"warmup": 0, "repeat": 1, "timer": "host_clock", "l2_thrash_bytes": 0, "per_call_ms": [2.0]}\n'
        )
        self.assertEqual(report["cache"], "warm")
        self.assertEqual(report["timer"], "host_clock")

    def test_a_case_records_the_steady_state_mean_and_the_wall_clock(self):
        runner = self.tmp / "runner"
        runner.write_text(FAKE_RUNNER, encoding="utf-8")
        runner.chmod(0o755)
        generated = self.tmp / "generated"
        write_tensor_dir(generated / "c1" / "input", [("input_0", "float32", (2,), [1.0, 2.0])])
        write_tensor_dir(generated / "c1" / "golden", [("output", "float32", (2,), [1.0, 2.0])], reference="cpu_fp64_reference")
        os.environ["FAKE_TIMING"] = "10.0 4.0 2.0"
        os.environ["FAKE_REPEAT"] = "3"
        self.addCleanup(os.environ.pop, "FAKE_TIMING", None)
        self.addCleanup(os.environ.pop, "FAKE_REPEAT", None)
        row = run_binary.run_binary_case(
            {"tier": "A_kernel"}, runner, {"case_id": "c1", "golden": "c1/golden/tensors.json"}, generated, stability_reruns=0
        )
        self.assertTrue(row["passed"], row)
        self.assertEqual(row["runtime"], 5.33, "the mean of 10, 4 and 2")
        self.assertEqual(row["timing"], {
            "warmup": 0, "repeat": 3, "discarded": 1, "timer": "musa_event", "cache": "cold",
            "l2_thrash_bytes": 268435456, "mean_ms": 5.33, "min_ms": 2.0, "samples": [10.0, 4.0, 2.0],
        })
        self.assertGreater(row["wall_ms"], 0.0)
        # The number the row grades on is the mean of the samples it carries.
        self.assertEqual(row["runtime"], row["timing"]["mean_ms"])
        self.assertAlmostEqual(row["runtime"], sum(row["timing"]["samples"]) / 3, places=2)

    def test_a_runner_with_no_report_falls_back_to_the_wall_clock(self):
        """An older runner is still graded; the row says which number it is."""
        runner = self.tmp / "runner"
        runner.write_text(FAKE_RUNNER, encoding="utf-8")
        runner.chmod(0o755)
        generated = self.tmp / "generated"
        write_tensor_dir(generated / "c1" / "input", [("input_0", "float32", (2,), [1.0, 2.0])])
        write_tensor_dir(generated / "c1" / "golden", [("output", "float32", (2,), [1.0, 2.0])], reference="cpu_fp64_reference")
        row = run_binary.run_binary_case(
            {"tier": "A_kernel"}, runner, {"case_id": "c1", "golden": "c1/golden/tensors.json"}, generated, stability_reruns=0
        )
        self.assertIsNone(row["timing"])
        self.assertEqual(row["runtime"], row["wall_ms"])


class PipelineTests(unittest.TestCase):
    """The whole compiled path over a copied package, with the device parts faked."""

    def test_the_pipeline_runs_the_stages_in_order_and_grades_the_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package = tmp_path / "kb_l1_97_a"
            shutil.copytree(A_TASK, package)
            task = json.loads((package / "task.json").read_text(encoding="utf-8"))
            task["starter"]["cpp"]["build"] = str(tmp_path / "build.sh")
            # This machine has no MUSA device, so the pre-flight comparison would
            # (correctly) refuse the run before any stage. Dropping the task's
            # snapshot is how a test asks the driver to skip that comparison --
            # which is the same thing a task without a record does.
            task.pop("target_environment", None)
            (package / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")

            build_script = tmp_path / "build.sh"
            build_script.write_text(
                "#!/bin/bash\n"
                "set -e\n"
                "build_dir=''\n"
                "while [ $# -gt 0 ]; do case \"$1\" in --build-dir) build_dir=\"$2\"; shift 2 ;; *) shift ;; esac; done\n"
                f"cp {tmp_path / 'fake_runner'} \"$build_dir/runner\"\n",
                encoding="utf-8",
            )
            build_script.chmod(0o755)
            (tmp_path / "fake_runner").write_text(FAKE_RUNNER, encoding="utf-8")
            (tmp_path / "fake_runner").chmod(0o755)

            generated = tmp_path / "generated"
            case_id = "smoke_001"
            write_tensor_dir(generated / case_id / "input", [("input_0", "float32", (2,), [1.0, 2.0])])
            write_tensor_dir(generated / case_id / "golden", [("output", "float32", (2,), [1.0, 2.0])], reference="cpu_fp64_reference")
            manifest = tmp_path / "cases.json"
            manifest.write_text(
                json.dumps({"cases": [{"case_id": case_id, "tag": "correctness", "golden": f"{case_id}/golden/tensors.json", }]}),
                encoding="utf-8",
            )
            (tmp_path / "submission").mkdir()
            (tmp_path / "submission" / "kernel.mu").write_text(
                "// --- BEGIN device_kernel ---\n__global__ void k() {}\n// --- END device_kernel ---\n"
                "// --- BEGIN host_launch ---\n// --- END host_launch ---\n",
                encoding="utf-8",
            )

            code, report = run_binary.evaluate_binary_task(
                task_dir=package,
                submission=tmp_path / "submission",
                cases_manifest=manifest,
                generated_dir=generated,
                build_dir=tmp_path / "build",
                stability_reruns=0,
            )
            self.assertEqual(code, 0, report)
            self.assertEqual(report["stages"], ["environment", "case_generation", "static_audit", "build", "evaluation"])
            self.assertTrue(report["build"]["passed"], report["build"])
            self.assertTrue(report["summary"]["passed_overall"])
            self.assertEqual(report["summary"]["passed"], 1)

    def test_a_kernel_that_names_a_forbidden_library_never_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package = tmp_path / "kb_l1_97_a"
            shutil.copytree(A_TASK, package)
            submission = tmp_path / "submission"
            submission.mkdir()
            (submission / "kernel.mu").write_text(
                "// --- BEGIN device_kernel ---\nmudnnFusedAttention(0);\n// --- END device_kernel ---\n"
                "// --- BEGIN host_launch ---\n// --- END host_launch ---\n",
                encoding="utf-8",
            )
            task = json.loads((package / "task.json").read_text(encoding="utf-8"))
            task["starter"]["cpp"]["build"] = str(tmp_path / "must_not_run.sh")
            task.pop("target_environment", None)
            (package / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
            (tmp_path / "must_not_run.sh").write_text("#!/bin/bash\ntouch " + str(tmp_path / "ran") + "\n", encoding="utf-8")
            (tmp_path / "must_not_run.sh").chmod(0o755)
            (tmp_path / "cases.json").write_text(json.dumps({"cases": []}), encoding="utf-8")

            code, report = run_binary.evaluate_binary_task(
                task_dir=package,
                submission=submission,
                cases_manifest=tmp_path / "cases.json",
                generated_dir=tmp_path / "generated",
                build_dir=tmp_path / "build",
            )
            self.assertEqual(code, run_binary.exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API)
            self.assertFalse((tmp_path / "ran").exists(), "the audit rejected the source and the build still ran")
            self.assertIn("static_audit", report["stages"])
            self.assertNotIn("build", report["stages"])


if __name__ == "__main__":
    unittest.main()
