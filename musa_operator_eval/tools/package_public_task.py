"""Export only agent-visible files for a task package."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ENTRIES = ["task.json", "semantics.json", "public_cases.json", "PROMPT.md", "ABI.md", "reference", "starter", "generated"]


def load_auditor():
    path = ROOT / "tools" / "audit_agent_package.py"
    spec = importlib.util.spec_from_file_location("audit_agent_package", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def package(task_dir: Path, output: Path) -> None:
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    for name in PUBLIC_ENTRIES:
        source = task_dir / name
        if not source.exists():
            raise ValueError(f"missing public task entry: {name}")
        target = output / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
    shutil.copytree(ROOT / "agent_reference", output / "agent_reference")
    forbidden_names = {"private", "musa_attention_set", "capability_plan.json"}
    leaked = [path for path in output.rglob("*") if path.name in forbidden_names]
    if leaked:
        raise ValueError(f"private/source material leaked into package: {leaked}")
    disclosure_errors = load_auditor().audit(output)
    if disclosure_errors:
        raise ValueError("agent package disclosure audit failed: " + "; ".join(disclosure_errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=ROOT / "tasks" / "sdpa_forward_pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.task_dir, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
