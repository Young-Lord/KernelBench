"""Apply the B-tier measured-performance admission gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluate(baseline: dict) -> dict:
    threshold = baseline.get("scoring", {}).get("minimum_naive_to_expert_ratio", 1.3)
    by_case: dict[str, dict[str, dict]] = {}
    for result in baseline.get("results", []):
        by_case.setdefault(result["case_id"], {})[result["implementation"]] = result
    ratios = {}
    for case_id, implementations in by_case.items():
        naive = implementations.get("naive_library_composition")
        expert = implementations.get("expert_dispatch")
        if not naive or not expert or naive.get("status") != "pass" or expert.get("status") != "pass":
            continue
        if not naive.get("latency_ms") or not expert.get("latency_ms"):
            continue
        ratios[case_id] = naive["latency_ms"] / expert["latency_ms"]
    if not ratios:
        return {"decision": "insufficient_data", "threshold": threshold, "ratios": {}, "reason": "no paired passing naive/expert measurements"}
    geometric_mean = math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios))
    decision = "admit" if geometric_mean >= threshold else "reject"
    return {"decision": decision, "threshold": threshold, "geometric_mean_ratio": geometric_mean, "ratios": ratios, "reason": "measured naive/expert geometric mean"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.baseline.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["decision"] == "admit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
