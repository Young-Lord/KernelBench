"""Short-circuit evaluator for the MUSA SDPA pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATE = load_module("musa_validate", ROOT / "tools" / "validate.py")
AUDIT = load_module("musa_audit", Path(__file__).with_name("audit.py"))
COMPARE = load_module("musa_compare", Path(__file__).with_name("compare.py"))


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "build" not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def stage(name: str, status: str, exit_code: int, **details) -> dict:
    return {"name": name, "status": status, "exit_code": exit_code, "details": details}


def validate_trace(path: Path, cases: list[dict]) -> list[str]:
    errors = []
    if not path.is_file():
        return ["dispatch trace is missing"]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_case = {record.get("case_id"): record for record in records}
    for case in cases:
        record = by_case.get(case["case_id"])
        if not record:
            errors.append(f"missing dispatch trace for {case['case_id']}")
        elif record.get("selected_path") != case.get("expected_path"):
            errors.append(f"{case['case_id']}: expected {case.get('expected_path')}, got {record.get('selected_path')}")
        elif not record.get("probe_status"):
            errors.append(f"{case['case_id']}: probe_status is empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=ROOT / "tasks" / "sdpa_forward_pilot")
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--environment-snapshot", default="not-captured")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--control-plane-only", action="store_true")
    args = parser.parse_args()
    submission = args.submission or args.task_dir / "starter"
    template = args.task_dir / "starter"
    task = json.loads((args.task_dir / "task.json").read_text(encoding="utf-8"))
    public_cases = json.loads((args.task_dir / "public_cases.json").read_text(encoding="utf-8"))
    stages = []

    schema_errors = []
    for path, kind in ((args.task_dir / "task.json", "task"), (args.task_dir / "semantics.json", "semantics"), (args.task_dir / "public_cases.json", "cases")):
        schema_errors.extend(f"{path.name}: {error}" for error in VALIDATE.validate_document(path, kind))
    if schema_errors:
        stages.append(stage("schema", "fail", 10, errors=schema_errors))
    else:
        stages.append(stage("schema", "pass", 0))

    if stages[-1]["status"] == "pass":
        audit_errors = AUDIT.audit_submission(template, submission)
        stages.append(stage("static_audit", "fail" if audit_errors else "pass", 20 if audit_errors else 0, errors=audit_errors))

    if stages[-1]["status"] == "pass":
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, str(submission / "build.py"), "--host-only", "--output-dir", str(Path(temp) / "build")]
            built = subprocess.run(command, capture_output=True, text=True)
            stages.append(stage("host_build", "pass" if built.returncode == 0 else "fail", 0 if built.returncode == 0 else 30, stdout=built.stdout, stderr=built.stderr))

    if all(item["status"] == "pass" for item in stages) and args.runner:
        public_errors = []
        trace_path = args.report.parent / "dispatch_trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        for case in public_cases["cases"]:
            case_root = args.task_dir / "generated" / case["case_id"]
            output_dir = args.report.parent / "outputs" / case["case_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            attributes = case["attributes"]
            command = [str(args.runner), str(case_root / "input"), str(output_dir), case["case_id"], str(trace_path), "1" if attributes["causal"] else "0", str(attributes["window_left"]), str(attributes["window_right"]), str(attributes["scale"]), "reserved"]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                public_errors.append(f"{case['case_id']}: runner exit {completed.returncode}: {completed.stderr}")
                break
            comparison = COMPARE.compare_directories(output_dir, case_root / "golden", atol=case["tolerance"]["atol"], rtol=case["tolerance"]["rtol"])
            if not comparison["passed"]:
                public_errors.append(f"{case['case_id']}: {comparison}")
                break
        stages.append(stage("public_correctness", "fail" if public_errors else "pass", 40 if public_errors else 0, errors=public_errors))
        if not public_errors and task["tier"] == "B_library":
            trace_errors = validate_trace(trace_path, public_cases["cases"])
            stages.append(stage("dispatch_trace", "fail" if trace_errors else "pass", 50 if trace_errors else 0, errors=trace_errors))
    elif all(item["status"] == "pass" for item in stages):
        stages.append(stage("public_correctness", "skipped", 0, reason="MUSA runner not supplied"))

    if all(item["status"] == "pass" for item in stages) and args.private_manifest:
        stages.append(stage("hidden_correctness", "skipped", 0, reason="native hidden adapter is not implemented yet"))
    elif not any(item["name"] == "hidden_correctness" for item in stages):
        stages.append(stage("hidden_correctness", "skipped", 0, reason="private manifest not mounted"))
    stages.append(stage("performance", "skipped", 0, reason="MUSA baseline adapter not supplied"))

    failed = any(item["status"] == "fail" for item in stages)
    skipped_required = any(item["status"] == "skipped" for item in stages if item["name"] in {"public_correctness", "hidden_correctness", "performance"})
    status = "fail" if failed else "incomplete" if skipped_required else "pass"
    report = {
        "schema_version": "1.0.0", "task_id": task["id"], "tier": task["tier"],
        "environment_snapshot": args.environment_snapshot, "submission_digest": tree_digest(submission),
        "status": status, "stages": stages,
        "metrics": {"compile_rate": 1.0 if any(x["name"] == "host_build" and x["status"] == "pass" for x in stages) else 0.0, "correctness_rate": 1.0 if any(x["name"] == "public_correctness" and x["status"] == "pass" for x in stages) else 0.0, "coverage_rate": 0.0, "dispatch_efficiency": 0.0, "speedup": None},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.report)
    return 1 if failed else 3 if status == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
