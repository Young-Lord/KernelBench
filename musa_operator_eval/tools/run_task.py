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

    # Audit the submission and verify the case overrides resolve. Runs anywhere.
    python musa_operator_eval/tools/run_task.py --submission model_new.py --static-only

    # Full evaluation. Needs the target device.
    python musa_operator_eval/tools/run_task.py \
        --submission model_new.py --backend musa --precision fp16 \
        --output runs/sdpa/report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent
DEFAULT_TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"

CASE_OVERRIDE_HEADER = "# --- case overrides injected by run_task.py for {case_id} ---"


def load_task(task_dir: Path) -> Tuple[dict, dict, str]:
    """Load the contract, the case list and the reference source for a task.

    Returns:
        (task, cases_manifest, problem_source)

    Raises:
        FileNotFoundError: a required task file is missing.
        ValueError: the contract does not name a problem file or case parameters.
    """
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    cases = json.loads((task_dir / "public_cases.json").read_text(encoding="utf-8"))

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


def _load_checker():
    """Import the tier-aware static checker without importing the package.

    `import kernelbench` pulls in torch via the package __init__, which would make
    the static path useless on a machine without a GPU stack. The checker itself
    is standard library only.
    """
    path = REPO_TOP / "src" / "kernelbench" / "kernel_static_checker.py"
    spec = importlib.util.spec_from_file_location("kernelbench_static_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_report(task: dict, cases: List[dict], problem_source: str, submission_source: str) -> dict:
    """Audit the submission and check the case overrides, without a device."""
    checker = _load_checker()
    policy = task.get("library_policy") or {}
    tier = task.get("tier", "A_kernel")

    audit_ok, audit_errors, audit_warnings = checker.static_audit_kernel(
        submission_source,
        tier=tier,
        library_policy=policy,
        backend=task.get("target_environment", {}).get("backend", "musa"),
        precision="fp16",
    )

    case_parameters = task.get("case_parameters") or {}
    parameter_errors = verify_case_parameters(cases, case_parameters)

    # Building each case also proves the overrides produce compilable Python.
    source_errors: List[str] = []
    for case in cases:
        try:
            specialized = case_source(problem_source, case, case_parameters)
            compile(specialized, f"<case {case['case_id']}>", "exec")
        except (SyntaxError, KeyError, ValueError) as error:
            source_errors.append(f"{case.get('case_id')}: {error}")

    errors = audit_errors + parameter_errors + source_errors
    return {
        "mode": "static",
        "task_id": task.get("id"),
        "tier": tier,
        "static_audit": {"passed": audit_ok, "errors": audit_errors, "warnings": audit_warnings},
        "case_parameters": {"count": len(case_parameters), "errors": parameter_errors},
        "case_sources": {"count": len(cases), "errors": source_errors},
        "summary": {"passed_overall": not errors, "errors": errors},
    }


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
        backend=backend,
        precision=get_torch_dtype_from_string(precision),
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
    )

    if result is None:
        return {
            "case_id": case.get("case_id"),
            "expected_path": case.get("expected_path"),
            "passed": False,
            "errors": ["evaluation returned no result (transient compilation lock)"],
        }

    metadata = result.metadata or {}
    errors = list(metadata.get("static_audit_errors") or [])
    errors += list(metadata.get("dispatch_trace_errors") or [])

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
        "passed": passed,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--backend", default="musa")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--static-only", action="store_true", help="audit and verify case overrides, no device")
    parser.add_argument("--output", type=Path, help="write the report here as JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    task, cases_manifest, problem_source = load_task(args.task_dir)
    submission_source = args.submission.read_text(encoding="utf-8")
    cases = cases_manifest["cases"]

    if args.static_only:
        report = static_report(task, cases, problem_source, submission_source)
    else:
        rows = [
            run_case(
                task,
                case,
                problem_source,
                submission_source,
                args.backend,
                args.precision,
                args.num_correct_trials,
                args.num_perf_trials,
                args.verbose,
            )
            for case in cases
        ]
        passed = sum(1 for row in rows if row["passed"])
        report = {
            "mode": "full",
            "task_id": task.get("id"),
            "tier": task.get("tier"),
            "backend": args.backend,
            "precision": args.precision,
            "submission": str(args.submission),
            "cases": rows,
            "summary": {
                "cases": len(rows),
                "passed": passed,
                "failed": len(rows) - passed,
                "passed_overall": passed == len(rows),
            },
        }

    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")

    return 0 if report["summary"]["passed_overall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
