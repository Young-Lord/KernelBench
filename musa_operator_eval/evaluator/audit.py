"""Submission boundary and source audit for the SDPA pilot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


IGNORED_PARTS = {"build", "__pycache__"}
FORBIDDEN_SOURCE_PATTERNS = {
    "ATen": re.compile(r"\bATen\b|\bat::"),
    "torch": re.compile(r"\btorch::|#\s*include\s*[<\"]torch/"),
    "CUDA residue": re.compile(r"\bcuda[A-Z_]|cuda_runtime|__CUDA"),
    "dynamic loading": re.compile(r"\bdlopen\s*\(|\bLoadLibrary[A-Z]*\s*\("),
    "network access": re.compile(r"\b(curl|wget|socket|connect)\s*\("),
    "version-string dispatch": re.compile(r"\b(version|version_string|MUSA_VERSION|MUDNN_VERSION)\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def outside_regions(text: str, regions: list[str]) -> str:
    result = text
    for region in regions:
        begin = f"BEGIN_AGENT_EDIT:{region}"
        end = f"END_AGENT_EDIT:{region}"
        if result.count(begin) != 1 or result.count(end) != 1:
            raise ValueError(f"invalid markers for {region}")
        prefix, rest = result.split(begin, 1)
        _, suffix = rest.split(end, 1)
        result = prefix + begin + "\n<EDITABLE>\n" + end + suffix
    return result


def audit_submission(template: Path, submission: Path) -> list[str]:
    manifest = json.loads((template / "starter_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((template / "starter_lock.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    allowed = {"starter_manifest.json", "starter_lock.json"}
    allowed.update(manifest["frozen_files"])
    allowed.update(item["path"] for item in manifest["editable_files"])
    for path in submission.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.relative_to(submission).parts):
            continue
        relative = path.relative_to(submission).as_posix()
        if relative not in allowed:
            errors.append(f"unexpected file: {relative}")
    for relative, expected in lock["frozen_sha256"].items():
        path = submission / relative
        if not path.is_file():
            errors.append(f"missing frozen file: {relative}")
        elif sha256(path) != expected:
            errors.append(f"modified frozen file: {relative}")
    for item in manifest["editable_files"]:
        relative = item["path"]
        candidate = submission / relative
        original = template / relative
        if not candidate.is_file():
            errors.append(f"missing editable file: {relative}")
            continue
        try:
            before = outside_regions(original.read_text(encoding="utf-8"), item["regions"])
            after = outside_regions(candidate.read_text(encoding="utf-8"), item["regions"])
            if before != after:
                errors.append(f"modified content outside editable regions: {relative}")
        except ValueError as exc:
            errors.append(f"{relative}: {exc}")
        source = strip_comments(candidate.read_text(encoding="utf-8"))
        for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
            if pattern.search(source):
                errors.append(f"forbidden {label} in {relative}")
    return errors
