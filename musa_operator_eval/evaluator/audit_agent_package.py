"""§2's publication audit: what the agent is handed holds task definitions only.

§2 splits the evaluation set in two. The task package is handed over; the raw
source snapshot, the hidden cases, the golden tensors, the official baselines and
the expert dispatch are not. §3 puts the package in `tasks/<id>/` and everything
withheld outside it.

That separation is a claim about a directory, so it is checked against the
directory rather than asserted in prose. Four things can go wrong and each of them
is silent on its own:

  * a maintainer-only file is present, which publishes a hidden case list, a
    baseline or the expert dispatch the task exists to measure;
  * a package file names a path under `private/`, which tells a reader where the
    answers are even though it does not contain them;
  * the prompt, the starter or the case list names the upstream project, which
    §2 keeps out of the agent's hands and out of its prompt;
  * a file in the tree is in neither the freeze list nor the editable list, so the
    inventory §4.6 asks for does not describe the package it claims to describe.

    python musa_operator_eval/evaluator/audit_agent_package.py
    python musa_operator_eval/evaluator/audit_agent_package.py musa_operator_eval/tasks/kb_l3_43_b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

# Files that exist only on the maintainer side. A package that has one has
# published it, whatever the ignore rules elsewhere say.
MAINTAINER_ONLY_NAMES = {
    "admission.json",
    "baseline.json",
    "baseline.hidden.json",
    "cases.private.json",
    "private_gap_evidence.json",
    "route_probe.json",
    "unavailable.json",
    "viability.md",
    "environment.snapshot.json",
}

MAINTAINER_ONLY_DIRS = {"expert", "naive", "blind", "golden"}

# A package file that mentions the maintainer tree tells a reader where the
# answers live, even if it does not contain them.
PRIVATE_REFERENCE = re.compile(r"musa_operator_eval/private/|\.private\.json|/private/")

# §2 keeps the upstream project out of the agent's hands. The task contract may
# name it under `source`, which is the provenance record §4.1 asks for; nothing
# else in the package may.
UPSTREAM_NAMES = re.compile(r"KernelBench|ScalingIntelligence")

# The files in a package whose text the agent reads as instructions or as the
# interface, and which therefore must not name the upstream project.
AGENT_FACING_SUFFIXES = (".md",)

TEXT_SUFFIXES = (".md", ".py", ".json", ".txt", ".toml")


def _text_files(package_dir: Path) -> List[Path]:
    return sorted(
        path for path in package_dir.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts
    )


def audit_package(package_dir: Path) -> Dict[str, object]:
    """Check one task package against §2 and §4.6.

    Returns:
        {"package": str, "passed": bool, "findings": [...]}
    """
    package_dir = Path(package_dir)
    findings: List[str] = []
    task_path = package_dir / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    starter = task.get("starter") or {}

    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_dir).as_posix()
        if path.name in MAINTAINER_ONLY_NAMES:
            findings.append(f"{relative}: a maintainer-only file is inside the agent-visible package")
        if set(path.relative_to(package_dir).parts[:-1]) & MAINTAINER_ONLY_DIRS:
            findings.append(f"{relative}: it lives in a maintainer-only directory")

    for path in _text_files(package_dir):
        relative = path.relative_to(package_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        if PRIVATE_REFERENCE.search(text):
            findings.append(f"{relative}: names a path under the maintainer tree")
        if UPSTREAM_NAMES.search(text) and path.name != "task.json":
            findings.append(f"{relative}: names the upstream project outside the contract's source record")
        if path.suffix in AGENT_FACING_SUFFIXES and UPSTREAM_NAMES.search(text):
            findings.append(f"{relative}: names the upstream project in a file the agent reads as instructions")

    # §4.6's inventory has to describe the package it is attached to: a file in
    # neither list is a file nobody declared, and the freeze list is what the
    # audit draws its boundary from.
    editable = {entry["path"] for entry in starter.get("editable", [])} | set(
        (starter.get("cpp") or {}).get("editable", [])
    )
    frozen = set(starter.get("frozen", []))
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_dir).as_posix()
        if relative in frozen or relative in editable or relative.startswith("generated/"):
            continue
        findings.append(f"{relative}: neither editable nor frozen, so §4.6's inventory does not cover it")

    return {"package": str(package_dir), "passed": not findings, "findings": findings}


def audit_packages(packages: List[Path]) -> Dict[str, object]:
    """Audit several packages and summarise."""
    results = [audit_package(package) for package in packages]
    findings = [f"{result['package']}: {finding}" for result in results for finding in result["findings"]]
    return {
        "packages": len(results),
        "passed": not findings,
        "findings": findings,
        "results": results,
    }


def all_packages() -> List[Path]:
    return sorted(path for path in (ROOT / "tasks").iterdir() if path.is_dir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packages", nargs="*", type=Path, help="packages to audit; defaults to every one in tasks/")
    parser.add_argument("--json", action="store_true", help="emit the report instead of the summary line")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    packages = args.packages or all_packages()
    report = audit_packages(packages)
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["passed"]:
        print(f"agent package disclosure audit passed ({report['packages']} packages)")
    else:
        print("agent package disclosure audit failed:", file=sys.stderr)
        for finding in report["findings"]:
            print(f"  {finding}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
