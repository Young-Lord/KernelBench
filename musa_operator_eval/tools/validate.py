"""Dependency-free structural and cross-document validation for MUSA eval JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
FIXTURE_DIR = ROOT / "fixtures"
KNOWN_SCHEMAS = {p.stem.split(".")[0]: p for p in SCHEMA_DIR.glob("*.schema.json")}


class ValidationError(ValueError):
    pass


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[expected]


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(value, choice) for choice in choices):
            return [f"{path}: expected type {expected}, got {type(value).__name__}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} is not in {schema['enum']!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string is shorter than minLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: value does not match {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is above maximum {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: array has fewer than {schema['minItems']} items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: array items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(_validate_schema(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, child in value.items():
            if key in properties:
                errors.extend(_validate_schema(child, properties[key], f"{path}.{key}"))
    return errors


def _business_rules(kind: str, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tier = data.get("tier")
    if kind == "task":
        has_policy = "library_policy" in data
        if tier == "A_kernel" and has_policy:
            errors.append("$.library_policy: forbidden for A_kernel")
        if tier == "B_library" and not has_policy:
            errors.append("$.library_policy: required for B_library")
    elif kind == "cases":
        visibility = data.get("visibility")
        seen_ids: set[str] = set()
        gap_reasons: set[str] = set()
        gap_count = 0
        for index, case in enumerate(data.get("cases", [])):
            case_id = case.get("case_id")
            if case_id in seen_ids:
                errors.append(f"$.cases[{index}].case_id: duplicate {case_id!r}")
            seen_ids.add(case_id)
            if case.get("tag") == "perf" and "performance" not in case:
                errors.append(f"$.cases[{index}].performance: required for perf case")
            if tier == "A_kernel" and ("expected_path" in case or "library_gap" in case):
                errors.append(f"$.cases[{index}]: B-only dispatch fields used by A_kernel")
            if tier == "B_library":
                if "expected_path" not in case:
                    errors.append(f"$.cases[{index}].expected_path: required for B_library")
                gap = case.get("library_gap")
                if visibility == "public":
                    if gap is not None:
                        errors.append(f"$.cases[{index}].library_gap: forbidden in public manifest")
                    if case.get("expected_path") != "fused_library":
                        errors.append(f"$.cases[{index}].expected_path: public B case must use fused_library")
                elif gap is not None:
                    gap_count += 1
                    gap_reasons.add(gap.get("reason", ""))
        if tier == "B_library" and visibility == "private":
            if gap_count < 3:
                errors.append("$.cases: private B manifest requires at least 3 library gaps")
            if len(gap_reasons) < 2:
                errors.append("$.cases: private B manifest requires at least 2 library-gap reasons")
    elif kind == "baseline":
        implementations = set(data.get("implementations", []))
        common = {"upstream_musa", "mudnn_fused", "torch_musa_sdpa", "torch_musa_eager"}
        missing = common - implementations
        if missing:
            errors.append(f"$.implementations: missing common baselines {sorted(missing)}")
        if tier == "B_library":
            missing_b = {"naive_library_composition", "expert_dispatch"} - implementations
            if missing_b:
                errors.append(f"$.implementations: missing B baselines {sorted(missing_b)}")
            if data.get("scoring", {}).get("speedup_denominator") != "expert_dispatch":
                errors.append("$.scoring.speedup_denominator: B_library must use expert_dispatch")
    return errors


def validate_document(path: Path, kind: str | None = None) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = kind or path.name.split(".")[0]
    if selected not in KNOWN_SCHEMAS:
        raise ValidationError(f"unknown schema kind {selected!r}; choose from {sorted(KNOWN_SCHEMAS)}")
    schema = json.loads(KNOWN_SCHEMAS[selected].read_text(encoding="utf-8"))
    return _validate_schema(data, schema) + _business_rules(selected, data)


def validate_fixtures() -> int:
    failures = 0
    for expected_valid, directory in ((True, FIXTURE_DIR / "valid"), (False, FIXTURE_DIR / "invalid")):
        for path in sorted(directory.glob("*.json")):
            kind = path.name.split(".")[0]
            errors = validate_document(path, kind)
            passed = not errors
            label = "PASS" if passed == expected_valid else "FAIL"
            print(f"{label} {path.relative_to(ROOT)}")
            if passed != expected_valid:
                failures += 1
                for error in errors:
                    print(f"  {error}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--kind", choices=sorted(KNOWN_SCHEMAS))
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    if args.fixtures:
        return 1 if validate_fixtures() else 0
    if args.path is None:
        parser.error("provide a JSON path or --fixtures")
    errors = validate_document(args.path, args.kind)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"valid: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
