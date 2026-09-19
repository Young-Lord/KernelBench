"""Create a private B-tier case manifest only from recorded gap evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIVATE_CASES = [
    {"case_id": "hidden_gap_001", "tag": "boundary", "seed": 91001, "dtype": "float16", "shape": {"B": 1, "H_q": 8, "H_kv": 8, "S_q": 127, "S_kv": 193, "D": 64}, "attributes": {"scale": 0.125, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}},
    {"case_id": "hidden_gap_002", "tag": "non_aligned", "seed": 91002, "dtype": "float16", "shape": {"B": 1, "H_q": 7, "H_kv": 7, "S_q": 131, "S_kv": 197, "D": 72}, "attributes": {"scale": 0.11785113019775793, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "uniform", "tolerance": {"atol": 0.015, "rtol": 0.015}},
    {"case_id": "hidden_gap_003", "tag": "boundary", "seed": 91003, "dtype": "bfloat16", "shape": {"B": 2, "H_q": 16, "H_kv": 4, "S_q": 257, "S_kv": 257, "D": 128}, "attributes": {"scale": 0.08838834764831845, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "normal", "tolerance": {"atol": 0.02, "rtol": 0.02}},
    {"case_id": "hidden_gap_004", "tag": "boundary", "seed": 91004, "dtype": "float16", "shape": {"B": 1, "H_q": 8, "H_kv": 8, "S_q": 257, "S_kv": 257, "D": 64}, "attributes": {"scale": 0.125, "causal": False, "window_left": 64, "window_right": 0}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}},
    {"case_id": "hidden_perf_001", "tag": "perf", "seed": 92001, "dtype": "float16", "shape": {"B": 4, "H_q": 16, "H_kv": 16, "S_q": 512, "S_kv": 512, "D": 64}, "attributes": {"scale": 0.125, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}, "performance": {"warmup": 10, "measurements": 100, "rounds": 5, "statistic": "median"}, "expected_path": "fused_library"}
]


def build_manifest(evidence: dict) -> dict:
    assignments = {item["case_id"]: item for item in evidence.get("gaps", [])}
    cases = []
    for template in PRIVATE_CASES:
        case = dict(template)
        case["golden"] = f"golden/{case['case_id']}/tensors.json"
        if case["case_id"].startswith("hidden_gap_"):
            if case["case_id"] not in assignments:
                continue
            item = assignments[case["case_id"]]
            case["expected_path"] = item["expected_path"]
            case["library_gap"] = {"reason": item["reason"], "expected_path": item["expected_path"], "evidence": item["evidence"]}
        cases.append(case)
    reasons = {case["library_gap"]["reason"] for case in cases if "library_gap" in case}
    gap_count = sum("library_gap" in case for case in cases)
    if gap_count < 3 or len(reasons) < 2:
        raise ValueError("record at least 3 measured gaps covering at least 2 reason categories")
    return {"schema_version": "1.0.0", "task_id": "sdpa_forward_b_v0", "tier": "B_library", "visibility": "private", "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    manifest = build_manifest(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
