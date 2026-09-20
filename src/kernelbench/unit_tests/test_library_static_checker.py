"""
Unit tests for the B-tier (library dispatch) static checker.

The A tier and the B tier have opposite premises: A forbids library compute, B
requires it. These tests pin that inversion down, plus the three rules that only
exist for B: library whitelisting, no version-string dispatch, and dispatch
trace emission.

Run with either:
    python -m unittest src.kernelbench.unit_tests.test_library_static_checker -v
    python src/kernelbench/unit_tests/test_library_static_checker.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

# Loaded by path rather than via `import kernelbench`, because the package
# __init__ imports torch while this checker only needs the standard library.
# That keeps these tests runnable without a GPU stack installed.
_CHECKER_PATH = Path(__file__).resolve().parents[1] / "kernel_static_checker.py"
_spec = importlib.util.spec_from_file_location("kernel_static_checker", _CHECKER_PATH)
assert _spec is not None and _spec.loader is not None
_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checker)

check_dispatch_trace_emission = _checker.check_dispatch_trace_emission
check_library_whitelist = _checker.check_library_whitelist
check_version_string_dispatch = _checker.check_version_string_dispatch
resolve_tier_and_library_policy = _checker.resolve_tier_and_library_policy
static_audit_kernel = _checker.static_audit_kernel
validate_kernel_static = _checker.validate_kernel_static
validate_library_kernel_static = _checker.validate_library_kernel_static


POLICY = {
    "allowed_libraries": ["libmusa", "libmudnn"],
    "allowed_symbol_prefixes": ["musa", "mudnn"],
    "required_trace_fields": ["case_id", "selected_path", "probe_status"],
    "trace_env_var": "KB_DISPATCH_TRACE",
}

# A submission that probes at runtime, dispatches to a whitelisted library, and
# appends the trace where the evaluator told it to. This is what the B tier asks
# for; anything less cannot be graded.
GOOD_SUBMISSION = """
import json
import os

import torch
import torch_musa
import mudnn

class ModelNew(torch.nn.Module):
    def forward(self, q, k, v):
        probe = mudnn.sdpa_supported(q.dtype, q.shape[-1])
        record = {"case_id": case_id, "selected_path": "fused_library", "probe_status": "accepted"}
        with open(os.environ["KB_DISPATCH_TRACE"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")
        return torch_musa.sdpa(q, k, v)
"""

COMPUTE_USING_SUBMISSION = GOOD_SUBMISSION + """
    def extra(self, x):
        return torch.softmax(x, dim=-1)
"""

# Names the trace fields as real string literals but never locates the file, so
# it cannot actually emit a trace: field names alone are one string away from
# meaningless, the env var lookup is the part that has to be there.
FIELDS_ONLY_SUBMISSION = """
import torch_musa

TRACE_FIELDS = ("case_id", "selected_path", "probe_status")

class ModelNew(torch.nn.Module):
    def forward(self, q, k, v):
        return torch_musa.sdpa(q, k, v)
"""


class ReturnShapeTests(unittest.TestCase):
    def test_returns_valid_errors_warnings_tuple(self):
        valid, errors, warnings = validate_library_kernel_static(GOOD_SUBMISSION, POLICY)
        self.assertIsInstance(valid, bool)
        self.assertIsInstance(errors, list)
        self.assertIsInstance(warnings, list)

    def test_good_submission_passes(self):
        valid, errors, _ = validate_library_kernel_static(GOOD_SUBMISSION, POLICY)
        self.assertTrue(valid, f"expected a clean submission, got errors: {errors}")
        self.assertEqual(errors, [])


class ABInversionTests(unittest.TestCase):
    """Library compute is the task for B, and a hack for A."""

    def test_library_compute_is_not_an_error(self):
        valid, errors, warnings = validate_library_kernel_static(COMPUTE_USING_SUBMISSION, POLICY)
        self.assertTrue(valid, f"library compute should not fail the B tier: {errors}")
        self.assertTrue(any("softmax" in message for message in warnings), warnings)

    def test_torch_computation_is_an_error_for_the_a_tier(self):
        code = "import torch\ndef f(x):\n    return torch.softmax(x, dim=-1)\n"
        _, errors, _ = validate_kernel_static(
            code, backend="", precision="fp16", forbidden=["torch_computation_ops"]
        )
        self.assertTrue(any("softmax" in message for message in errors), errors)


class WhitelistTests(unittest.TestCase):
    def test_non_whitelisted_import_is_rejected(self):
        valid, errors, _ = validate_library_kernel_static("import triton\n" + GOOD_SUBMISSION, POLICY)
        self.assertFalse(valid)
        self.assertTrue(any("triton" in message for message in errors), errors)

    def test_whitelisted_library_is_accepted_through_prefix(self):
        """`torch_musa` contains the `musa` prefix, so it is whitelisted."""
        self.assertEqual(check_library_whitelist("import torch_musa", ["libmusa"], ["musa"]), (False, ""))

    def test_lib_prefix_is_normalized_for_imports(self):
        """A contract may name the linker library `libmudnn`; the import is `mudnn`."""
        self.assertEqual(check_library_whitelist("import mudnn", ["libmudnn"], []), (False, ""))

    def test_plumbing_modules_stay_allowed(self):
        for module in ("torch", "torch.nn", "torch.utils.cpp_extension", "numpy", "math", "json"):
            with self.subTest(module=module):
                has_issue, message = check_library_whitelist(f"import {module}", [], [])
                self.assertFalse(has_issue, f"{module} should be plumbing: {message}")

    def test_ctypes_load_of_non_whitelisted_library_is_rejected(self):
        code = 'import ctypes\nlib = ctypes.CDLL("libflashattention.so")\n'
        has_issue, message = check_library_whitelist(code, ["libmusa"], ["musa"])
        self.assertTrue(has_issue)
        self.assertIn("flashattention", message)

    def test_ctypes_load_of_whitelisted_library_is_accepted(self):
        code = 'import ctypes\nlib = ctypes.CDLL("libmudnn.so.2")\n'
        self.assertEqual(check_library_whitelist(code, ["libmudnn"], ["mudnn"]), (False, ""))


class VersionStringDispatchTests(unittest.TestCase):
    def test_version_string_dispatch_is_rejected(self):
        snippets = [
            'if torch.__version__ > "2.0":\n    return 1',
            "torch.version.musa",
            "from packaging import parse_version",
            "LooseVersion(torch.__version__)",
        ]
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                has_issue, message = check_version_string_dispatch(snippet)
                self.assertTrue(has_issue, f"expected a version-string hit in {snippet!r}: {message}")

    def test_runtime_probe_is_not_version_dispatch(self):
        has_issue, message = check_version_string_dispatch(GOOD_SUBMISSION)
        self.assertFalse(has_issue, message)


class DispatchTraceTests(unittest.TestCase):
    def test_missing_trace_fields_are_reported(self):
        has_issue, message = check_dispatch_trace_emission("return output", ["case_id", "selected_path"])
        self.assertTrue(has_issue)
        self.assertIn("case_id", message)
        self.assertIn("selected_path", message)

    def test_complete_trace_fields_pass(self):
        self.assertEqual(
            check_dispatch_trace_emission(GOOD_SUBMISSION, ["case_id", "probe_status"], "KB_DISPATCH_TRACE"),
            (False, ""),
        )

    def test_submission_without_trace_is_rejected(self):
        code = "import torch_musa\ndef f(q, k, v):\n    return torch_musa.sdpa(q, k, v)\n"
        valid, errors, _ = validate_library_kernel_static(code, POLICY)
        self.assertFalse(valid)
        self.assertTrue(any("dispatch trace" in message for message in errors), errors)

    def test_naming_the_fields_without_reading_the_env_var_is_rejected(self):
        """Field names are one string away from meaningless; the env var is not."""
        has_issue, message = check_dispatch_trace_emission(
            FIELDS_ONLY_SUBMISSION, ["case_id", "selected_path", "probe_status"], "KB_DISPATCH_TRACE"
        )
        self.assertTrue(has_issue, "naming the fields alone must not satisfy the trace rule")
        self.assertIn("KB_DISPATCH_TRACE", message)

        valid, errors, _ = validate_library_kernel_static(FIELDS_ONLY_SUBMISSION, POLICY)
        self.assertFalse(valid)
        self.assertTrue(any("KB_DISPATCH_TRACE" in message for message in errors), errors)


class OptionalDeviceKernelTests(unittest.TestCase):
    def test_no_device_kernel_is_fine(self):
        """The fused path needs no custom kernel, so its absence must not be an error."""
        valid, errors, _ = validate_library_kernel_static(GOOD_SUBMISSION, POLICY)
        self.assertTrue(valid, errors)

    def test_present_device_kernel_must_be_a_real_musa_kernel(self):
        code = "import torch_musa\n\n__global__ void bad_kernel() {}\n" + GOOD_SUBMISSION
        valid, errors, _ = validate_library_kernel_static(code, POLICY)
        self.assertFalse(valid)
        self.assertTrue(
            any("musa_runtime.h" in message or "load_inline" in message for message in errors), errors
        )


class OverrideTests(unittest.TestCase):
    def test_promoting_library_compute_to_forbidden_makes_it_an_error(self):
        """A stricter task contract can promote the soft warning to a hard error."""
        code = (
            "import torch_musa\n"
            "def f(x):\n"
            "    return torch.softmax(x, dim=-1)\n"
            'record = {"case_id": 1, "selected_path": "x", "probe_status": "y"}\n'
        )
        valid, errors, _ = validate_library_kernel_static(
            code, POLICY, forbidden=["torch_computation_ops"]
        )
        self.assertFalse(valid)
        self.assertTrue(any("softmax" in message for message in errors), errors)

    def test_warnings_override_suppresses_the_library_compute_warning(self):
        valid, errors, warnings = validate_library_kernel_static(
            COMPUTE_USING_SUBMISSION, POLICY, warnings=[]
        )
        self.assertTrue(valid, errors)
        self.assertEqual(warnings, [])

    def test_b_tier_rules_survive_a_forbidden_override(self):
        """The B-tier rules are the mode itself, not registry entries to be switched off."""
        valid, errors, _ = validate_library_kernel_static(
            "import triton\n" + GOOD_SUBMISSION, POLICY, forbidden=[]
        )
        self.assertFalse(valid)
        self.assertTrue(any("triton" in message for message in errors), errors)


class TierResolutionTests(unittest.TestCase):
    """TIER / LIBRARY_POLICY come from the executed problem namespace."""

    def test_absent_metadata_is_the_a_tier(self):
        self.assertEqual(resolve_tier_and_library_policy({}), ("A_kernel", None))

    def test_explicit_a_tier(self):
        self.assertEqual(resolve_tier_and_library_policy({"TIER": "A_kernel"}), ("A_kernel", None))

    def test_b_tier_returns_the_policy(self):
        tier, policy = resolve_tier_and_library_policy({"TIER": "B_library", "LIBRARY_POLICY": POLICY})
        self.assertEqual(tier, "B_library")
        self.assertIs(policy, POLICY)

    def test_b_tier_without_a_policy_is_a_contract_error(self):
        """A B-tier task with no whitelist cannot be graded, so it must not fall back to A."""
        with self.assertRaises(ValueError):
            resolve_tier_and_library_policy({"TIER": "B_library"})

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_tier_and_library_policy({"TIER": "C_something"})


class TierDispatchTests(unittest.TestCase):
    """static_audit_kernel must route to the rule set the tier asks for."""

    def test_same_submission_fails_the_a_tier_and_passes_the_b_tier(self):
        """The whole point of the dispatch: one source, two verdicts."""
        a_valid, a_errors, _ = static_audit_kernel(
            GOOD_SUBMISSION, tier="A_kernel", backend="musa"
        )
        b_valid, b_errors, _ = static_audit_kernel(
            GOOD_SUBMISSION, tier="B_library", library_policy=POLICY, backend="musa"
        )

        self.assertFalse(a_valid, "an A-tier submission with no device kernel must fail")
        self.assertTrue(any("__global__" in message for message in a_errors), a_errors)
        self.assertTrue(b_valid, f"the same source is a valid B-tier submission: {b_errors}")
        self.assertEqual(b_errors, [])

    def test_a_tier_keeps_its_own_rules(self):
        code = "import torch\ndef f(x):\n    return torch.softmax(x, dim=-1)\n"
        valid, errors, _ = static_audit_kernel(
            code, tier="A_kernel", backend="", forbidden=["torch_computation_ops"]
        )
        self.assertFalse(valid)
        self.assertTrue(any("softmax" in message for message in errors), errors)

    def test_b_tier_uses_the_whitelist_instead(self):
        valid, errors, _ = static_audit_kernel(
            "import triton\n" + GOOD_SUBMISSION, tier="B_library", library_policy=POLICY
        )
        self.assertFalse(valid)
        self.assertTrue(any("triton" in message for message in errors), errors)

    def test_b_tier_audit_requires_a_policy(self):
        with self.assertRaises(ValueError):
            static_audit_kernel("x = 1", tier="B_library")

    def test_unknown_tier_is_rejected_by_the_audit(self):
        with self.assertRaises(ValueError):
            static_audit_kernel("x = 1", tier="C_something")


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
