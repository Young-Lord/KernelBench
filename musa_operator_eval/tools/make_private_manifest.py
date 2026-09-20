"""Create a private B-tier case manifest only from recorded gap evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# `generate_cases.py` writes a case to `<output-dir>/<case_id>/{input,golden}/`,
# so a `golden` field is that path relative to `<output-dir>`. The two tools are
# the only place this layout is written down, and they disagreed once: this file
# used to record `golden/<case_id>/tensors.json`, which names no file the
# generator ever produces. Both the case id and the sub-directory name live here
# so a future change to the generator has one obvious place to be mirrored.
GOLDEN_MANIFEST_RELPATH = "{case_id}/golden/tensors.json"

# The tool that owns that layout. Recorded in the manifest so a consumer can
# recompute a golden from the case's seed instead of trusting the stored bytes;
# without it, `run_task.py` can only check the golden's internal consistency.
CASE_GENERATION_TOOL = "musa_operator_eval/tools/generate_cases.py"


# Every gap case here was measured on mp_22 (MTT S4000, MUSA Toolkit 3.1.0,
# torch_musa 1.3.0) before being listed. An earlier draft guessed that
# non-aligned shapes, grouped-query heads and windowed attention would be gaps;
# measurement showed all three are served by the fused path, so they were
# replaced. Do not add a case to this list without a recorded measurement.
#
# `tag` is §4.3's fixed vocabulary for both lists, and a gap is a boundary probe:
# it asks where the library's coverage ends. An earlier draft labelled each gap
# with free text naming the specific boundary ("head_dim_beyond_fused_bound" and
# so on), which `schemas/case_manifest.schema.json` rejects. The specific
# boundary is not lost — it is what `library_gap.reason` classifies and what
# `library_gap.evidence` records.
PRIVATE_CASES = [
    {"case_id": "hidden_gap_001", "tag": "boundary", "seed": 91001, "dtype": "float16", "shape": {"B": 1, "H_q": 8, "H_kv": 8, "S_q": 128, "S_kv": 128, "D": 192}, "attributes": {"scale": 0.07216878364870322, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}},
    {"case_id": "hidden_gap_002", "tag": "boundary", "seed": 91002, "dtype": "float16", "shape": {"B": 1, "H_q": 8, "H_kv": 8, "S_q": 128, "S_kv": 128, "D": 256}, "attributes": {"scale": 0.0625, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "uniform", "tolerance": {"atol": 0.01, "rtol": 0.01}},
    {"case_id": "hidden_gap_003", "tag": "boundary", "seed": 91003, "dtype": "float16", "shape": {"B": 1, "H_q": 8, "H_kv": 8, "S_q": 128, "S_kv": 128, "D": 64}, "attributes": {"scale": 0.05, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}},
    {"case_id": "hidden_gap_004", "tag": "boundary", "seed": 91004, "dtype": "float16", "shape": {"B": 1, "H_q": 8, "H_kv": 8, "S_q": 257, "S_kv": 64, "D": 64}, "attributes": {"scale": 0.125, "causal": False, "window_left": 0, "window_right": 0}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}},
    {"case_id": "hidden_perf_001", "tag": "perf", "seed": 92001, "dtype": "float16", "shape": {"B": 4, "H_q": 16, "H_kv": 16, "S_q": 512, "S_kv": 512, "D": 64}, "attributes": {"scale": 0.125, "causal": False, "window_left": -1, "window_right": -1}, "distribution": "normal", "tolerance": {"atol": 0.01, "rtol": 0.01}, "performance": {"warmup": 10, "measurements": 100, "rounds": 5, "statistic": "median"}, "expected_path": "fused_library"}
]


def build_manifest(evidence: dict) -> dict:
    assignments = {item["case_id"]: item for item in evidence.get("gaps", [])}
    cases = []
    for template in PRIVATE_CASES:
        case = dict(template)
        case["golden"] = GOLDEN_MANIFEST_RELPATH.format(case_id=case["case_id"])
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
    return {
        "schema_version": "1.0.0",
        "task_id": "sdpa_forward_b_v0",
        "tier": "B_library",
        "visibility": "private",
        # The public manifest carries the same block: it is how a consumer knows
        # which tool owns the `golden` layout above and can recompute a golden
        # from the case seed rather than trust the stored bytes.
        "case_generation": {
            "tool": CASE_GENERATION_TOOL,
            "deterministic": True,
            "note": "A private list hides which configuration is a library gap; that is the case the task exists to make the submission discover.",
        },
        "cases": cases,
    }


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
