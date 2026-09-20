"""Drive a MUSA task across its cases on the KernelBench framework.

A task's `problem.py` describes a single configuration through its module-level
defaults; `public_cases.json` lists several configurations. This driver bridges
the two: for each case it appends the overrides that turn those defaults into
that case, evaluates the submission against the reference, and checks the
dispatch trace against the expected path.

The case-to-variable mapping is read from `task.json` under `case_parameters`, so
no code from the task has to be executed to know how a case is assembled. That is
what lets `--static-only` run without a GPU stack: the module level of this file
imports only the standard library.

Each stage of the guide's §4.7 pipeline is run in order and short-circuits: a
stage that fails returns before the next one starts, and returns a named exit
code from `exit_codes.py` rather than a bare 1. The report records which stage
produced the code, so the number identifies the stage and the report explains it.

    # Audit the submission and verify the case overrides. Runs anywhere.
    python musa_operator_eval/tools/run_task.py --submission model_new.py --static-only

    # Full evaluation. Needs the target device.
    python musa_operator_eval/tools/run_task.py \
        --submission model_new.py --backend musa --precision fp16 \
        --generated-dir musa_operator_eval/tasks/sdpa_forward_pilot/generated \
        --output runs/sdpa/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# `run_task.py` is run as a script from anywhere, and loaded by tests through
# `importlib.util.spec_from_file_location`, neither of which puts this directory
# on `sys.path`. Adding it here is what makes the sibling `exit_codes` import
# below work in both cases.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from exit_codes import (  # noqa: E402  (import follows the sys.path fix-up above)
    EXIT_BOUNDARY_FAILED,
    EXIT_BUILD_FAILED,
    EXIT_CASE_GENERATION_FAILED,
    EXIT_CORRECTNESS_FAILED,
    EXIT_DISPATCH_TRACE_MISMATCH,
    EXIT_ENVIRONMENT_UNAVAILABLE,
    EXIT_GOLDEN_MISMATCH,
    EXIT_GOLDEN_MISSING,
    EXIT_GOLDEN_UNUSABLE,
    EXIT_INPUT_UNAVAILABLE,
    EXIT_LINK_WHITELIST_VIOLATION,
    EXIT_OK,
    EXIT_PERFORMANCE_FAILED,
    EXIT_PRECISION_CONTRACT_MISMATCH,
    EXIT_STABILITY_FAILED,
    EXIT_STATIC_AUDIT_DEVICE_SOURCE,
    EXIT_STATIC_AUDIT_FORBIDDEN_API,
    EXIT_STATIC_AUDIT_HOST_COMPUTATION,
    EXIT_STATIC_AUDIT_OUT_OF_SCOPE_MODIFICATION,
    EXIT_STATIC_AUDIT_TABLE_LOOKUP,
    EXIT_STATIC_AUDIT_UNCLASSIFIED,
    EXIT_SUMMARY_FAILED,
    describe as describe_exit_code,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent
DEFAULT_TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"
DEFAULT_GENERATED_SUBDIR = "generated"

CASE_OVERRIDE_HEADER = "# --- case overrides injected by run_task.py for {case_id} ---"

# The driver's precision vocabulary, mapped onto the dtype names a task's
# `tensor_contract` uses. Keeping the mapping in one place is what lets the flag
# be checked against the contract instead of trusted.
PRECISION_DTYPE_NAMES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}

# The file name `generate_cases.py` writes a case's golden tensor manifest under.
GOLDEN_MANIFEST_NAME = "tensors.json"

# Which §4.7 correctness sub-stage a case belongs to, keyed by the §4.3 tag
# vocabulary. The case tag is the only thing that says whether a failure is a
# plain correctness disagreement, a boundary probe or a stability re-run, and the
# exit code is supposed to carry that difference.
CASE_TAG_STAGES = {
    "smoke": "correctness",
    "correctness": "correctness",
    "generalization": "correctness",
    "non_aligned": "boundary",
    "boundary": "boundary",
    "extreme": "boundary",
    "stability": "stability",
    "perf": "performance",
}

# The order the stages run in, matching §5's pipeline
# (`静态审计 → 编译 → 公开 → 隐藏 → 边界 → 稳定性 → 性能`). Once a stage has
# failed, a later one is not entered, which is what the guide's short-circuit
# rule requires. Cases carrying no recognised tag are treated as correctness,
# because a case nobody classified is still a case whose answer must be right.
CASE_STAGE_ORDER = ("correctness", "boundary", "stability", "performance")

STAGE_EXIT_CODES = {
    "correctness": EXIT_CORRECTNESS_FAILED,
    "boundary": EXIT_BOUNDARY_FAILED,
    "stability": EXIT_STABILITY_FAILED,
    "performance": EXIT_PERFORMANCE_FAILED,
}

# §4.7 stages this driver does not run, and the code each would return. The link
# whitelist check reads the build product's export table, which only exists after
# a device build; until it is wired here, naming it is better than leaving a
# reader to wonder why the code never appears.
STAGES_NOT_IMPLEMENTED = {
    "link_whitelist_check": EXIT_LINK_WHITELIST_VIOLATION,
}

# A golden that is absent, corrupt or contradicted means the ground truth cannot
# be trusted, so the run stops there — in the static pass as much as the device
# one, because a static report that says "passed" while a declared golden was
# never generated is the exact illusion §4.3's field is meant to prevent. The
# rule is "declare it and you must deliver it": a manifest that names no golden
# (`not_declared`) or names one that could not be recomputed (`integrity_only`)
# is reported with `verified: False` but does not by itself stop the run.
GOLDEN_FATAL_EXIT_CODES = {
    "missing": EXIT_GOLDEN_MISSING,
    "unusable": EXIT_GOLDEN_UNUSABLE,
    "mismatch": EXIT_GOLDEN_MISMATCH,
}


# ---------------------------------------------------------------------------
# Static audit classification
# ---------------------------------------------------------------------------

# §4.7 names five things the static audit rejects. The checker returns prose
# rather than category labels, so the messages it is known to emit are mapped
# here. Order is precedence: the first pattern that matches an error decides its
# category, which is why the library/torch checks come before the device-source
# ones (`non-whitelisted library: triton` is a forbidden dependency, not a
# missing kernel).
STATIC_AUDIT_CATEGORY_PATTERNS: Tuple[Tuple[int, Tuple[str, ...]], ...] = (
    (EXIT_DISPATCH_TRACE_MISMATCH, (
        "dispatch trace",
    )),
    (EXIT_STATIC_AUDIT_FORBIDDEN_API, (
        "non-whitelisted library",
        "non-whitelisted shared library",
        "whitelist",
        "forbidden",
        "prohibited",
        "torch computation op",
        "torch.nn.functional",
        "torch.nn compute layer",
        "fused attention entry point",
        "version string",
    )),
    (EXIT_STATIC_AUDIT_OUT_OF_SCOPE_MODIFICATION, (
        "out of scope",
        "out-of-scope",
        "freeze",  # covers "frozen", "freezes" and "freezing"
        "not allowed to modify",
        "modified file",
        "immutable",
    )),
    (EXIT_STATIC_AUDIT_HOST_COMPUTATION, (
        "host-side",
        "host side",
        "host computation",
        "computed on the host",
        "cpu tensor",
        "numpy",
    )),
    (EXIT_STATIC_AUDIT_TABLE_LOOKUP, (
        "lookup table",
        "look-up table",
        "table lookup",
        "precomputed table",
        "hardcoded values",
        "hard-coded values",
    )),
    (EXIT_STATIC_AUDIT_DEVICE_SOURCE, (
        "__global__",
        "kernel definition",
        "load_inline",
        "cpp_extension",
        "musa_runtime",
        "cuda_runtime",
        "musaextension",
        "device source",
        "tilelang",
        "triton",
        "hipcc",
    )),
)


def classify_static_audit_errors(errors: List[str]) -> int:
    """Return the exit code for a set of static-audit findings.

    The first finding that matches a named category decides, so the code is
    deterministic and reflects the most prominent problem rather than an
    arbitrary one. A finding that matches nothing gets its own code: silently
    bucketing it into a category it may not belong to would make the report
    claim more than the check found.
    """
    for error in errors:
        lowered = error.lower()
        for exit_code, patterns in STATIC_AUDIT_CATEGORY_PATTERNS:
            if any(pattern in lowered for pattern in patterns):
                return exit_code
    return EXIT_STATIC_AUDIT_UNCLASSIFIED


# ---------------------------------------------------------------------------
# Task and case loading
# ---------------------------------------------------------------------------


def allowed_precisions(task: dict) -> List[str]:
    """The precisions a task's tensor contract permits, in the driver's vocabulary.

    The A tier was measured at float32 and the fused path the B tier exists to
    exercise does not answer float32 at all. `--precision` is a free flag, so
    without this check a run at the wrong precision produces a plausible report
    of a different task's behaviour and nothing says so.
    """
    dtypes = (task.get("tensor_contract") or {}).get("input_dtypes") or []
    return [name for name, dtype in PRECISION_DTYPE_NAMES.items() if dtype in dtypes]


def load_task(task_dir: Path, cases_path: Optional[Path] = None) -> Tuple[dict, dict, str]:
    """Load the contract, the case list and the reference source for a task.

    Args:
        task_dir: the task package
        cases_path: the case manifest to evaluate. Defaults to the task's
            `public_cases.json`; point it at a private manifest to evaluate the
            hidden set. The cases live outside the task package because the
            hidden ones must not be readable from it.

    Returns:
        (task, cases_manifest, problem_source)

    Raises:
        FileNotFoundError: a required task file is missing.
        ValueError: the contract does not name a problem file or case parameters.
    """
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    cases = json.loads((cases_path or task_dir / "public_cases.json").read_text(encoding="utf-8"))

    problem_file = task.get("problem_file")
    if not problem_file:
        raise ValueError("task.json must declare `problem_file`")
    problem_source = (task_dir / problem_file).read_text(encoding="utf-8")

    return task, cases, problem_source


def resolve_case_field(case: dict, dotted_path: str) -> Any:
    """Read a dotted path such as `attributes.scale` out of a case entry.

    Raises:
        KeyError: the path does not exist in this case, which means the task's
            `case_parameters` mapping and its case list disagree.
    """
    value: Any = case
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"case {case.get('case_id')!r} has no field {dotted_path!r}")
        value = value[part]
    return value


def case_source(problem_source: str, case: dict, case_parameters: Dict[str, str]) -> str:
    """Return the problem source specialized to one case.

    The overrides are appended rather than substituted so the reference stays
    byte-identical to what a reader sees in `problem.py`; the appended block just
    re-binds the module-level defaults before `get_init_inputs` / `get_inputs`
    read them. That is the same idiom KernelBench problems already use for their
    constants.
    """
    assignments = "\n".join(
        f"{variable} = {resolve_case_field(case, path)!r}"
        for path, variable in case_parameters.items()
    )
    return (
        problem_source.rstrip()
        + "\n\n\n"
        + CASE_OVERRIDE_HEADER.format(case_id=case["case_id"])
        + "\n"
        + assignments
        + "\n"
    )


def verify_case_parameters(
    cases: List[dict], case_parameters: Dict[str, str]
) -> List[str]:
    """Check the mapping resolves for every case and yields valid Python.

    A typo in `case_parameters` would otherwise surface as a confusing failure
    deep inside an evaluation run, on hardware. This catches it up front, and
    works without a device.
    """
    errors: List[str] = []
    if not case_parameters:
        return ["task.json declares no `case_parameters`, so cases cannot be instantiated"]

    for path, variable in case_parameters.items():
        if not variable.isidentifier():
            errors.append(f"case parameter {path!r} maps to {variable!r}, which is not a valid name")

    for case in cases:
        for path in case_parameters:
            try:
                resolve_case_field(case, path)
            except KeyError as error:
                errors.append(str(error))
    return errors


def verify_case_sources(
    cases: List[dict], problem_source: str, case_parameters: Dict[str, str]
) -> List[str]:
    """Check every case specialises into valid Python.

    Building the source also proves the overrides produce something the loader
    will accept, which is the part of "case generation" that can be checked off
    the device.
    """
    errors: List[str] = []
    for case in cases:
        try:
            specialized = case_source(problem_source, case, case_parameters)
            compile(specialized, f"<case {case['case_id']}>", "exec")
        except (SyntaxError, KeyError, ValueError) as error:
            errors.append(f"{case.get('case_id')}: {error}")
    return errors


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def load_module_from_path(module_name: str, path: Path):
    """Execute a module from a file path without putting it on `sys.path`."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checker():
    """Import the tier-aware static checker without importing the package.

    `import kernelbench` pulls in torch via the package __init__, which would make
    the static path useless on a machine without a GPU stack. The checker itself
    is standard library only.
    """
    path = REPO_TOP / "src" / "kernelbench" / "kernel_static_checker.py"
    return load_module_from_path("kernelbench_static_checker", path)


# ---------------------------------------------------------------------------
# Static audit
# ---------------------------------------------------------------------------


def run_static_audit(
    task: dict, submission_source: str, precision: str
) -> Tuple[bool, List[str], List[str]]:
    """Audit the submission against its tier's rule set.

    Kept separate from the report so the full path can run exactly the same
    audit before it compiles anything, which is what makes an audit failure stop
    the run instead of being re-discovered once per case on the device.
    """
    checker = _load_checker()
    policy = task.get("library_policy") or {}
    tier = task.get("tier", "A_kernel")
    return checker.static_audit_kernel(
        submission_source,
        tier=tier,
        library_policy=policy,
        backend=task.get("target_environment", {}).get("backend", "musa"),
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Golden cross-check (§4.3)
# ---------------------------------------------------------------------------


def golden_manifest_path(case: dict, generated_root: Path) -> Optional[Path]:
    """Resolve a case's `golden` path against the directory the generator wrote to.

    `generate_cases.py` writes `<output-dir>/<case_id>/{input,golden}/`, and the
    manifest records that path relative to `<output-dir>`, so the caller's
    `--generated-dir` has to be the same `<output-dir>` the generator used.
    """
    declared = case.get("golden")
    if not declared:
        return None
    declared_path = Path(declared)
    if declared_path.is_absolute():
        return declared_path
    return Path(generated_root) / declared_path


def check_golden_integrity(case: dict, golden_path: Path, manifest: dict) -> List[str]:
    """Check a stored golden is self-consistent and belongs to this case.

    The digests are what make the check worth doing: a golden whose blob was
    truncated or overwritten reads as valid JSON and would otherwise be graded
    against silently.
    """
    errors: List[str] = []
    manifest_case_id = manifest.get("case_id")
    if manifest_case_id != case.get("case_id"):
        errors.append(
            f"golden manifest names case {manifest_case_id!r}, not {case.get('case_id')!r}"
        )

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        errors.append("golden manifest declares no tensors")
        return errors

    for record in tensors:
        name = record.get("name")
        filename = record.get("file")
        if not filename:
            errors.append(f"golden tensor {name!r} names no file")
            continue
        blob_path = golden_path.parent / filename
        if not blob_path.is_file():
            errors.append(f"golden tensor {name!r} points at a missing file {filename!r}")
            continue
        data = blob_path.read_bytes()
        recorded_length = record.get("nbytes")
        if recorded_length is not None and len(data) != recorded_length:
            errors.append(
                f"golden tensor {name!r}: {len(data)} bytes on disk, {recorded_length} recorded"
            )
        recorded_digest = record.get("sha256")
        actual_digest = hashlib.sha256(data).hexdigest()
        if recorded_digest and actual_digest != recorded_digest:
            errors.append(
                f"golden tensor {name!r}: sha256 {actual_digest[:12]}... on disk, "
                f"{recorded_digest[:12]}... recorded"
            )
    return errors


def recompute_golden(case: dict, generation_tool: Path) -> dict:
    """Regenerate a case's golden from its seed and return the fresh manifest.

    This is the cross-check the `golden` field exists for: it re-runs the same
    deterministic generator a grader would, so a golden produced from a different
    seed, a different reference revision or a changed case definition shows up as
    a disagreement rather than as a plausible-looking number.
    """
    generator = load_module_from_path("musa_case_generator", generation_tool)
    if not hasattr(generator, "generate_case"):
        raise AttributeError(f"{generation_tool} defines no generate_case(case, output_root)")
    with tempfile.TemporaryDirectory(prefix="run_task_golden_") as temporary:
        generator.generate_case(case, Path(temporary))
        regenerated = Path(temporary) / case["case_id"] / "golden" / GOLDEN_MANIFEST_NAME
        return json.loads(regenerated.read_text(encoding="utf-8"))


def compare_golden_manifests(stored: dict, regenerated: dict) -> List[str]:
    """Differences between a stored golden and a freshly generated one."""
    stored_tensors = {record.get("name"): record for record in stored.get("tensors", [])}
    regenerated_tensors = {record.get("name"): record for record in regenerated.get("tensors", [])}

    differences: List[str] = []
    for name in sorted(set(stored_tensors) | set(regenerated_tensors)):
        stored_record = stored_tensors.get(name)
        regenerated_record = regenerated_tensors.get(name)
        if stored_record is None:
            differences.append(f"recomputation produces tensor {name!r}; the stored golden does not have it")
            continue
        if regenerated_record is None:
            differences.append(f"the stored golden has tensor {name!r}; the recomputation does not produce it")
            continue
        for field in ("dtype", "shape", "nbytes", "sha256"):
            if stored_record.get(field) != regenerated_record.get(field):
                differences.append(
                    f"golden tensor {name!r} {field}: stored {stored_record.get(field)!r}, "
                    f"recomputed {regenerated_record.get(field)!r}"
                )
    return differences


def verify_golden(
    case: dict, generated_root: Path, generation_tool: Optional[Path] = None
) -> dict:
    """Cross-check one case's stored golden and describe the outcome.

    Every outcome is named, including the ones where the check did not happen.
    A golden that was never generated and a golden that was generated but not
    recomputed are different facts, and both are recorded as `verified: False`
    rather than being folded into a pass.
    """
    declared = case.get("golden")
    result: dict = {"case_id": case.get("case_id"), "declared_path": declared, "verified": False}

    if not declared:
        result["status"] = "not_declared"
        result["detail"] = "the case records no golden path; §4.3 requires one"
        return result

    golden_path = golden_manifest_path(case, generated_root)
    result["resolved_path"] = str(golden_path)
    if golden_path is None or not golden_path.is_file():
        result["status"] = "missing"
        result["detail"] = "golden was never generated at this path"
        return result

    try:
        manifest = json.loads(golden_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        result["status"] = "unusable"
        result["errors"] = [f"cannot read the golden manifest: {error}"]
        return result

    integrity_errors = check_golden_integrity(case, golden_path, manifest)
    if integrity_errors:
        result["status"] = "unusable"
        result["errors"] = integrity_errors
        return result
    result["tensors"] = [record.get("name") for record in manifest.get("tensors", [])]

    if generation_tool is None:
        result["status"] = "integrity_only"
        result["detail"] = (
            "the case manifest declares no case_generation.tool, so the golden was checked "
            "for internal consistency but not against a recomputation"
        )
        return result

    # Recomputation is best-effort by design: a missing numpy or a generator that
    # cannot handle this family must not turn into a silent pass, but it must not
    # turn into a crash either. The failure is recorded as the reason the golden
    # stayed unverified.
    try:
        regenerated = recompute_golden(case, generation_tool)
    except Exception as error:  # noqa: BLE001 — any failure means "not recomputed"
        result["status"] = "integrity_only"
        result["detail"] = f"recomputation unavailable: {type(error).__name__}: {error}"
        return result

    differences = compare_golden_manifests(manifest, regenerated)
    if differences:
        result["status"] = "mismatch"
        result["errors"] = differences
        result["detail"] = "the stored golden disagrees with a fresh recomputation from the case seed"
        return result

    result["status"] = "verified"
    result["verified"] = True
    result["detail"] = "a reference recomputation from the case seed reproduces the stored golden"
    return result


def resolve_generation_tool(case_generation: Optional[dict]) -> Optional[Path]:
    """Resolve the manifest's declared generation tool against the repo root."""
    declared = (case_generation or {}).get("tool")
    if not declared:
        return None
    tool_path = Path(declared)
    if not tool_path.is_absolute():
        tool_path = REPO_TOP / tool_path
    return tool_path if tool_path.is_file() else None


def golden_stage(
    cases: List[dict], generated_root: Path, case_generation: Optional[dict]
) -> dict:
    """Cross-check every case's golden and report whether any of them is fatal.

    Args:
        cases: the case entries
        generated_root: the `--output-dir` the generator wrote into
        case_generation: the manifest's `case_generation` block, if it has one

    Returns:
        {"results": [...], "exit_code": int, "fatal": bool, "counts": {...}}
    """
    generation_tool = resolve_generation_tool(case_generation)
    results = [verify_golden(case, generated_root, generation_tool) for case in cases]

    counts: Dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    fatal_exit_code = EXIT_OK
    for result in results:
        code = GOLDEN_FATAL_EXIT_CODES.get(result["status"])
        if code is not None:
            fatal_exit_code = code
            break

    return {
        "generated_root": str(generated_root),
        "generation_tool": str(generation_tool) if generation_tool else None,
        "results": results,
        "counts": counts,
        "fatal": fatal_exit_code != EXIT_OK,
        "exit_code": fatal_exit_code,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_summary(exit_code: int, **counts: int) -> dict:
    """Assemble the report's `summary` block.

    `passed_overall` is derived from the exit code rather than recomputed, so a
    report can never say "passed" next to a non-zero code.
    """
    summary = dict(counts)
    summary["passed_overall"] = exit_code == EXIT_OK
    summary["exit_code"] = exit_code
    summary["exit_code_description"] = describe_exit_code(exit_code)
    return summary


def static_report(
    task: dict,
    cases: List[dict],
    problem_source: str,
    submission_source: str,
    precision: str = "fp16",
    generated_dir: Optional[Path] = None,
    case_generation: Optional[dict] = None,
) -> dict:
    """Audit the submission and check the case overrides and goldens, without a device.

    The stages run in §4.7 order — case generation, golden cross-check, static
    audit — and stop at the first failure, so a report from a broken task package
    does not also claim to have audited a submission against it.

    `generated_dir` is optional for direct callers, but omitting it is recorded
    as "the golden stage did not run" rather than being silently skipped: the CLI
    always supplies it, so a run driven through `main` can never report a pass
    while a declared golden went unchecked.
    """
    tier = task.get("tier", "A_kernel")
    case_parameters = task.get("case_parameters") or {}
    report: dict = {
        "mode": "static",
        "task_id": task.get("id"),
        "tier": tier,
        "stages": [],
    }

    parameter_errors = verify_case_parameters(cases, case_parameters)
    source_errors = verify_case_sources(cases, problem_source, case_parameters)
    report["case_parameters"] = {"count": len(case_parameters), "errors": parameter_errors}
    report["case_sources"] = {"count": len(cases), "errors": source_errors}
    report["stages"].append("case_generation")
    if parameter_errors or source_errors:
        report["static_audit"] = {"ran": False, "reason": "case generation failed first"}
        report["summary"] = build_summary(EXIT_CASE_GENERATION_FAILED)
        return report

    if generated_dir is None:
        report["golden"] = {
            "ran": False,
            "reason": "no generated directory was given, so no case's golden was cross-checked",
        }
    else:
        golden = golden_stage(cases, generated_dir, case_generation)
        report["golden"] = golden
        report["stages"].append("golden")
        if golden["fatal"]:
            report["static_audit"] = {"ran": False, "reason": "the golden stage failed first"}
            report["summary"] = build_summary(golden["exit_code"])
            return report

    audit_ok, audit_errors, audit_warnings = run_static_audit(task, submission_source, precision)
    report["static_audit"] = {
        "ran": True,
        "passed": audit_ok,
        "errors": audit_errors,
        "warnings": audit_warnings,
    }
    report["stages"].append("static_audit")

    exit_code = EXIT_OK if audit_ok else classify_static_audit_errors(audit_errors)
    report["summary"] = build_summary(exit_code, errors=audit_errors)
    return report


def run_case(
    task: dict,
    case: dict,
    problem_source: str,
    submission_source: str,
    backend: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    verbose: bool,
) -> dict:
    """Evaluate one case and return its row of the report."""
    sys.path.insert(0, str(REPO_TOP / "src"))
    from kernelbench.eval import eval_library_dispatch_against_ref, get_torch_dtype_from_string

    policy = task.get("library_policy") or {}
    specialized_source = case_source(problem_source, case, task.get("case_parameters") or {})

    result = eval_library_dispatch_against_ref(
        specialized_source,
        submission_source,
        expected_dispatch_path=case.get("expected_path"),
        dispatch_case_id=case.get("case_id"),
        required_trace_fields=policy.get("required_trace_fields"),
        # The contract's tolerance block is the task's one lever on grading, and it
        # only loosens. Without this the field was declared by every package and
        # read by nothing, so a task could state a bar its own cases were graded
        # below without anything noticing.
        tolerance=(task.get("tolerances") or {}).get("max_abs_error"),
        backend=backend,
        precision=get_torch_dtype_from_string(precision),
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        # `measure_performance` defaults to False upstream, and without it the
        # evaluator returns runtime -1.0 no matter what num_perf_trials says.
        # That silently produced reports where every case passed and no number
        # was ever recorded, which is exactly the number the B tier scores.
        measure_performance=num_perf_trials > 0,
        verbose=verbose,
    )

    if result is None:
        return {
            "case_id": case.get("case_id"),
            "expected_path": case.get("expected_path"),
            "compiled": False,
            "correctness": False,
            "dispatch_trace_passed": None,
            "runtime": None,
            "ref_runtime": None,
            "tier": None,
            "dispatch_trace_summary": None,
            "static_audit_errors": [],
            "dispatch_trace_errors": [],
            "passed": False,
            "errors": ["evaluation returned no result (transient compilation lock)"],
        }

    metadata = result.metadata or {}
    static_audit_errors = list(metadata.get("static_audit_errors") or [])
    dispatch_trace_errors = list(metadata.get("dispatch_trace_errors") or [])
    errors = static_audit_errors + dispatch_trace_errors

    passed = bool(result.correctness) and result.dispatch_trace_passed is not False and not errors
    return {
        "case_id": case.get("case_id"),
        "expected_path": case.get("expected_path"),
        "compiled": result.compiled,
        "correctness": result.correctness,
        "dispatch_trace_passed": result.dispatch_trace_passed,
        "runtime": result.runtime,
        "ref_runtime": result.ref_runtime,
        "tier": metadata.get("tier"),
        "dispatch_trace_summary": metadata.get("dispatch_trace_summary"),
        "static_audit_errors": static_audit_errors,
        "dispatch_trace_errors": dispatch_trace_errors,
        "passed": passed,
        "errors": errors,
    }


def stage_for_case(case: dict) -> str:
    """Which §4.7 correctness sub-stage a case belongs to."""
    return CASE_TAG_STAGES.get(str(case.get("tag") or "").lower(), "correctness")


def exit_code_for_case_row(row: dict, case: dict) -> int:
    """The most specific exit code a failing case row justifies.

    Ordered by the pipeline: an audit finding outranks a compile failure, a
    compile failure outranks a wrong answer, and a wrong answer outranks a
    missing measurement. Returning the stage code for a plain disagreement is
    what lets a caller tell "boundary cases fail" from "nothing is right".
    """
    if row.get("passed"):
        return EXIT_OK
    if row.get("static_audit_errors"):
        return classify_static_audit_errors(row["static_audit_errors"])
    if not row.get("compiled"):
        return EXIT_BUILD_FAILED
    if row.get("dispatch_trace_passed") is False:
        return EXIT_DISPATCH_TRACE_MISMATCH

    stage = stage_for_case(case)
    if stage == "performance" and not _usable_runtime(row):
        return EXIT_PERFORMANCE_FAILED
    return STAGE_EXIT_CODES[stage]


def _usable_runtime(row: dict) -> bool:
    runtime = row.get("runtime")
    return isinstance(runtime, (int, float)) and runtime > 0


def evaluate_cases(
    task: dict,
    cases: List[dict],
    problem_source: str,
    submission_source: str,
    backend: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    verbose: bool,
    golden_results: Optional[Dict[str, dict]] = None,
) -> Tuple[List[dict], int]:
    """Run every case, short-circuiting once a stage has failed.

    Cases run in manifest order. A failure records the stage it belongs to and
    silently skips any case in a *later* stage: the guide's pipeline goes
    correctness → boundary → stability → performance, and a boundary probe tells
    you nothing once the answer is known to be wrong. Cases in the same stage all
    run, so a bad submission still reports every correctness disagreement it has.

    Returns:
        (rows, exit_code) with the rows of the cases that ran plus an explicit
        `skipped` row for each case that was not entered.
    """
    rows: List[dict] = []
    failed_stage_rank: Optional[int] = None
    exit_code = EXIT_OK

    for case in cases:
        stage = stage_for_case(case)
        stage_rank = CASE_STAGE_ORDER.index(stage)

        if failed_stage_rank is not None and stage_rank > failed_stage_rank:
            rows.append({
                "case_id": case.get("case_id"),
                "tag": case.get("tag"),
                "stage": stage,
                "skipped": True,
                "reason": (
                    f"not evaluated: the {CASE_STAGE_ORDER[failed_stage_rank]} stage failed, "
                    f"and {stage} comes after it"
                ),
            })
            continue

        row = run_case(
            task,
            case,
            problem_source,
            submission_source,
            backend,
            precision,
            num_correct_trials,
            num_perf_trials,
            verbose,
        )
        row["tag"] = case.get("tag")
        row["stage"] = stage
        golden_result = (golden_results or {}).get(case.get("case_id"))
        if golden_result is not None:
            row["golden"] = {
                "status": golden_result["status"],
                "verified": golden_result["verified"],
            }
        rows.append(row)

        if not row["passed"] and failed_stage_rank is None:
            failed_stage_rank = stage_rank
            exit_code = exit_code_for_case_row(row, case)

    return rows, exit_code


def evaluate_task(
    task: dict,
    cases_manifest: dict,
    problem_source: str,
    submission_source: str,
    backend: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    verbose: bool,
    generated_dir: Path,
) -> dict:
    """Run the full §4.7 pipeline for one submission, short-circuiting per stage."""
    cases = cases_manifest["cases"]
    case_parameters = task.get("case_parameters") or {}
    report: dict = {
        "mode": "full",
        "task_id": task.get("id"),
        "tier": task.get("tier"),
        "backend": backend,
        "precision": precision,
        "stages": [],
    }

    permitted = allowed_precisions(task)
    if precision not in permitted:
        report["contract"] = {"permitted_precisions": permitted}
        report["summary"] = build_summary(EXIT_PRECISION_CONTRACT_MISMATCH)
        return report

    parameter_errors = verify_case_parameters(cases, case_parameters)
    source_errors = verify_case_sources(cases, problem_source, case_parameters)
    report["case_generation"] = {"errors": parameter_errors + source_errors}
    report["stages"].append("case_generation")
    if parameter_errors or source_errors:
        report["summary"] = build_summary(EXIT_CASE_GENERATION_FAILED)
        return report

    # The golden stage exists to keep §4.3's `golden` field from being a promise
    # nothing reads. A manifest that declares a golden must deliver one: the
    # field is part of the task definition, so an absent file is an incomplete
    # evaluation setup, not a case that merely went uncross-checked.
    golden = golden_stage(cases, generated_dir, cases_manifest.get("case_generation"))
    report["golden"] = golden
    report["stages"].append("golden")
    if golden["fatal"]:
        report["summary"] = build_summary(golden["exit_code"])
        return report

    audit_ok, audit_errors, audit_warnings = run_static_audit(task, submission_source, precision)
    report["static_audit"] = {"passed": audit_ok, "errors": audit_errors, "warnings": audit_warnings}
    report["stages"].append("static_audit")
    if not audit_ok:
        report["summary"] = build_summary(classify_static_audit_errors(audit_errors))
        return report

    golden_by_case = {result["case_id"]: result for result in golden["results"]}
    # Importing the device stack is the first thing `run_case` does, and on a
    # machine without it the failure is an ImportError several frames down. That
    # is an environment fact, not a submission verdict, so it gets a code of its
    # own instead of a traceback that reads like a harness bug.
    try:
        rows, case_exit_code = evaluate_cases(
            task,
            cases,
            problem_source,
            submission_source,
            backend,
            precision,
            num_correct_trials,
            num_perf_trials,
            verbose,
            golden_by_case,
        )
    except ImportError as error:
        report["environment"] = {
            "error": str(error),
            "reason": "the device stack (torch / kernelbench) could not be imported",
        }
        report["summary"] = build_summary(EXIT_ENVIRONMENT_UNAVAILABLE)
        return report
    report["cases"] = rows
    report["stages"].append("evaluation")

    passed = sum(1 for row in rows if row.get("passed"))
    skipped = sum(1 for row in rows if row.get("skipped"))
    failed = len(rows) - passed - skipped
    exit_code = EXIT_OK if (failed == 0 and skipped == 0) else (case_exit_code or EXIT_SUMMARY_FAILED)
    report["summary"] = build_summary(
        exit_code, cases=len(rows), passed=passed, failed=failed, skipped=skipped
    )
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--cases", type=Path, help="case manifest to evaluate; defaults to the task's public_cases.json")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--backend", default="musa")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        help=(
            "the directory `generate_cases.py --output-dir` wrote into; a case's `golden` "
            "path is resolved against it. Defaults to <task-dir>/generated."
        ),
    )
    parser.add_argument("--static-only", action="store_true", help="audit and verify case overrides, no device")
    parser.add_argument("--output", type=Path, help="write the report here as JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser


def emit_report(report: dict, output: Optional[Path]) -> None:
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


def main() -> int:
    args = build_parser().parse_args()
    generated_dir = args.generated_dir or (args.task_dir / DEFAULT_GENERATED_SUBDIR)

    try:
        task, cases_manifest, problem_source = load_task(args.task_dir, args.cases)
        submission_source = args.submission.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"run_task: cannot read the task, its cases or the submission: {error}", file=sys.stderr)
        return EXIT_INPUT_UNAVAILABLE

    if args.static_only:
        report = static_report(
            task,
            cases_manifest["cases"],
            problem_source,
            submission_source,
            precision=args.precision,
            generated_dir=generated_dir,
            case_generation=cases_manifest.get("case_generation"),
        )
    else:
        permitted = allowed_precisions(task)
        if args.precision not in permitted:
            print(
                f"refusing to run {task.get('id')} at --precision {args.precision}: its tensor "
                f"contract permits {permitted or 'no precision at all'}. The task was measured at "
                f"the permitted precision, so a run at another one reports a different task's "
                f"behaviour without saying so.",
                file=sys.stderr,
            )
            return EXIT_PRECISION_CONTRACT_MISMATCH
        report = evaluate_task(
            task,
            cases_manifest,
            problem_source,
            submission_source,
            args.backend,
            args.precision,
            args.num_correct_trials,
            args.num_perf_trials,
            args.verbose,
            generated_dir,
        )
        report["submission"] = str(args.submission)

    emit_report(report, args.output)
    return report["summary"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
