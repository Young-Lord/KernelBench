"""Reject provenance and hidden-solution leakage from an agent-visible package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".json", ".md", ".py", ".cpp", ".h", ".mu"}
FORBIDDEN_PATTERNS = {
    "repository URL": re.compile(r"https?://|github\.com|gitee\.com", re.IGNORECASE),
    "full commit hash": re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", re.IGNORECASE),
    "maintainer source tree": re.compile(r"musa_attention_set|implementations[/\\]", re.IGNORECASE),
    "source project name": re.compile(r"\b(MUTLASS|MATE|TileLang-MUSA|MT-flashMLA|vLLM-MUSA|Paddle-MUSA)\b", re.IGNORECASE),
    "hidden asset path": re.compile(r"(^|[/\\])private([/\\]|$)|cases\.private", re.IGNORECASE),
}


def audit(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{relative}: leaked {label}")
    reference = root / "agent_reference" / "attention_capabilities.json"
    if reference.is_file():
        data = json.loads(reference.read_text(encoding="utf-8"))
        forbidden_keys = {"repository", "commit", "url", "version", "function_name", "hidden_cases", "exact_gap_configurations"}
        stack = [data]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                leaked_keys = forbidden_keys & value.keys()
                if leaked_keys:
                    errors.append(f"agent_reference/attention_capabilities.json: forbidden keys {sorted(leaked_keys)}")
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    else:
        errors.append("missing agent_reference/attention_capabilities.json")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    errors = audit(args.package)
    for error in errors:
        print(error)
    if errors:
        return 1
    print("agent package disclosure audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
