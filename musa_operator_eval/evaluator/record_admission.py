"""Record a B-tier admission decision into the task contract, or refuse to.

Guide §4.4 defines the gate as two halves that must both hold: every hidden
correctness case passes, and every dispatch trace matches the path the maintainer
expected. The geometric mean is only computed over cases where both of those are
already true, which is why `candidate_gate.py` skips a case whose naive or expert
did not pass. Skipping is right for the mean and wrong for the decision: a gate
that quietly drops the cases it could not answer can admit an entry on the cases it
happened to be able to answer.

This tool is the other half of that. It reads the baseline and the gate's output,
checks the two halves itself, and writes the result into `task.json` -- refusing to
record an `admit` when a hidden case failed, when a trace did not match, or when
the baseline was measured in a different environment than the contract names.

It writes rather than a human copying numbers out of a log, because a hand-copied
geometric mean is a number nobody can re-derive from the artifact it came from.

    # what would be recorded, without recording it
    python3 musa_operator_eval/evaluator/record_admission.py \
        --task-dir musa_operator_eval/tasks/kb_l3_31_b \
        --baseline musa_operator_eval/private/vision_attention_b_v0/baseline.hidden.json \
        --admission musa_operator_eval/private/vision_attention_b_v0/admission.json

    # and with --apply to write it

Exit codes: 0 recorded (or would be), 1 the gate does not justify the decision,
2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPERT_ROLE = "expert_dispatch"
NAIVE_ROLE = "naive_library_composition"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(baseline: dict, admission: dict, task: dict) -> tuple[list[str], dict]:
    """The two halves of §4.4, checked against the artifacts rather than assumed.

    Returns the problems found and the measurements worth recording.
    """
    problems: list[str] = []

    if baseline.get("task_id") != task.get("id"):
        problems.append(
            f"the baseline is for {baseline.get('task_id')!r}, the contract is {task.get('id')!r}"
        )

    declared = (task.get("target_environment") or {}).get("snapshot_id")
    measured = baseline.get("environment_snapshot")
    if measured != declared:
        problems.append(
            f"the baseline was measured in {measured!r}, the contract names {declared!r}; "
            "latencies are comparable only within one snapshot"
        )

    by_case: dict[str, dict[str, dict]] = {}
    for row in baseline.get("results", []):
        by_case.setdefault(row["case_id"], {})[row["implementation"]] = row

    hidden = sorted(case_id for case_id in by_case if case_id.startswith("hidden_"))
    if not hidden:
        problems.append("the baseline carries no hidden case, so it is not the set the gate is defined over")

    # Half one: every hidden case passes, for everything that has to pass.
    failures = []
    for case_id in hidden:
        rows = by_case[case_id]
        for role in (NAIVE_ROLE, EXPERT_ROLE):
            row = rows.get(role)
            if row is None:
                failures.append(f"{case_id}/{role}: not measured")
            elif row.get("status") != "pass":
                failures.append(f"{case_id}/{role}: {row.get('status')} ({row.get('reason')})")

    # Half two: the expert's trace matched the path the case expected.
    for case_id in hidden:
        row = by_case[case_id].get(EXPERT_ROLE)
        if row is None or row.get("status") != "pass":
            continue  # already reported by half one
        if row.get("dispatch_trace_passed") is not True:
            failures.append(
                f"{case_id}/{EXPERT_ROLE}: the trace did not match; recorded "
                f"{row.get('recorded_paths')!r}"
            )

    if failures:
        problems.append(
            "hidden cases did not satisfy both halves of the gate: " + "; ".join(failures[:6])
        )

    ratios = {}
    for case_id in hidden:
        rows = by_case[case_id]
        naive, expert = rows.get(NAIVE_ROLE), rows.get(EXPERT_ROLE)
        if not naive or not expert:
            continue
        if naive.get("status") != "pass" or expert.get("status") != "pass":
            continue
        if not naive.get("latency_ms") or not expert.get("latency_ms"):
            continue
        ratios[case_id] = naive["latency_ms"] / expert["latency_ms"]

    geometric_mean = None
    if ratios:
        geometric_mean = math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios))

    # The gate's own arithmetic is re-derived rather than trusted, so a decision
    # that disagrees with its inputs is caught here.
    threshold = (baseline.get("scoring") or {}).get("minimum_naive_to_expert_ratio")
    if admission.get("threshold") != threshold:
        problems.append(
            f"the gate used threshold {admission.get('threshold')!r}, the baseline declares {threshold!r}"
        )
    if geometric_mean is not None and admission.get("geometric_mean_ratio") is not None:
        if abs(geometric_mean - admission["geometric_mean_ratio"]) > 1e-9:
            problems.append(
                f"the gate reports a geometric mean of {admission['geometric_mean_ratio']}, "
                f"re-derived from the baseline it is {geometric_mean}"
            )
    if geometric_mean is not None and threshold is not None:
        expected = "admit" if geometric_mean >= threshold else "reject"
        if admission.get("decision") != expected:
            problems.append(
                f"the gate decided {admission.get('decision')!r} at a geometric mean of "
                f"{geometric_mean} against a threshold of {threshold}; the decision should be {expected!r}"
            )

    measurements = {
        "threshold": threshold,
        "geometric_mean_ratio": geometric_mean,
        "ratios": ratios,
        "hidden_cases": hidden,
    }
    return problems, measurements


def admission_blocks(measurements: dict, baseline: dict, admission: dict, baseline_path) -> tuple:
    """Split the measurement into the contract's summary and the private evidence.

    §2 hands the agent the contract, so the contract may say that the gate passed
    and at what threshold, and may not say which hidden case was worth what: a
    ratio table keyed by hidden case ids describes the set the task is graded on,
    down to which configuration is the gap and what the fused path is worth when
    it applies. The per-case numbers belong in the private record the baseline
    already lives in.

    Args:
        measurements: the re-derived ratios and the gate's numbers
        baseline: the measured baseline document
        admission: the gate's own output
        baseline_path: where the baseline is stored

    Returns:
        (contract block, private evidence record)
    """
    decision = admission.get("decision")
    block = {
        "status": "measured",
        "decision": decision,
        "measured_geometric_mean": measurements["geometric_mean_ratio"],
        "threshold": measurements["threshold"],
        "measured_in": baseline.get("environment_snapshot"),
    }
    evidence = {
        "decision": decision,
        "threshold": measurements["threshold"],
        "geometric_mean_ratio": measurements["geometric_mean_ratio"],
        "ratios": measurements["ratios"],
        "hidden_cases": measurements["hidden_cases"],
        "measured_in": baseline.get("environment_snapshot"),
        "measured_over": baseline.get("case_manifest") or "the private case manifest",
        "baseline": str(baseline_path),
        "reason": admission.get("reason") or "measured naive/expert geometric mean",
    }
    return block, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write the block; without it, print it")
    args = parser.parse_args()

    contract_path = args.task_dir / "task.json"
    task = load(contract_path)
    baseline = load(args.baseline)
    admission = load(args.admission)

    problems, measurements = audit(baseline, admission, task)

    block, evidence = admission_blocks(measurements, baseline, admission, args.baseline)

    previously = task.get("admission") or {}
    was_pending = previously.get("status") == "pending_device_measurement"
    for key in (
        "reason",
        "dispatch_note",
        "gap_reason_coverage",
        "notes",
        "also_answered",
        "how_to_read_the_mean",
        "gate",
        "evidence",
        "note",
    ):
        if key not in previously:
            continue
        # `reason` on a pending block explains why nothing could be asserted yet.
        # Carrying it onto a measured block leaves the contract saying both that the
        # gate was never measured and that it was, and a reader has no way to tell
        # which sentence is current. The other keys describe the task rather than
        # the state of the measurement, so they survive the transition.
        if key == "reason" and was_pending:
            continue
        block[key] = previously[key]

    print(json.dumps(block, indent=2, ensure_ascii=False))

    if problems:
        print("\nrefusing to record, because the artifacts do not support it:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if block["decision"] == "admit":
        print("\nboth halves hold: every hidden case passed for the naive and the expert, "
              "and every expert trace matched its case's expected path.")

    if args.apply:
        task["admission"] = block
        contract_path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # The private record is the one that keeps the per-case numbers. Writing
        # both from one place is what keeps them from drifting: a reader can
        # re-derive the contract's geometric mean from this file and get the same
        # number, and nothing in the agent-visible tree carries the cases.
        args.admission.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwritten to {contract_path} (summary) and {args.admission} (per-case evidence)")
    else:
        print("\npass --apply to write this into the contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
