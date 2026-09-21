"""Rank submissions: the A-kernel track and the B-library track, in two tables.

Guide §5's step 8 asks for the two tracks to be split apart and ranked separately.
The splitting is structural -- an A-tier task is graded on a device kernel and a
B-tier task on a library dispatch, and the scoreboard must never average one into
the other -- so this tool produces two tables and no combined number. There is
deliberately no "overall" row: a single figure across both tracks would be a
number about nothing.

The ratio is the tier's own denominator, read from each baseline's
`scoring.speedup_denominator` rather than assumed: `upstream_musa` for the A
tier, `expert_dispatch` for the B tier. It is computed per case over the cases
where the reference has a passing measurement on the same case and the report has
a positive runtime, and summarised as a geometric mean, because a ratio's natural
average is geometric -- the arithmetic mean of ratios treats a doubling and a
halving as equals, which they are not.

Units: both sides are milliseconds. `kernelbench.eval`'s dataclass comment says
`runtime` is in microseconds and the recorded numbers say otherwise -- an expert
module measured at 0.834 in a report and 0.7048 ms in the baseline record for the
same case -- so the baseline is what settles it: `latency_ms` is milliseconds and a
report's `runtime` divides into it directly. A tool that trusted the comment would
report every ratio a thousand times too high.

A ratio is only about the work when both sides measure the work. The Python path's
`runtime` is a warm measurement; the compiled path's is the median of the calls its
runner timed after a warmup, and a row that carries a `timing` block is one of those.
A compiled row without it (`timing: null`) reports a process wall clock -- the device's
first-call initialisation, mapping a library the size of muDNN, and the case's files --
and its ratio is about startup rather than about the submission. Rows like that are
ranked anyway, because dropping a submission silently is worse, but the tables' reading
should start from whether the numbers mean the same thing.

The denominators are measured by `kernelbench.timing` (see
`measure_baseline.py`), so a report whose numbers come from a different protocol -- a
wall clock, a warm cache, a hand-picked statistic -- is not dividing like by like even
when both sides name milliseconds.

Usage:

    rank.py --report /tmp/report_a.json --report /tmp/report_b.json \
        --baseline musa_operator_eval/private/*/baseline.hidden.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TIER_ORDER = ("A_kernel", "B_library")
TIER_TITLES = {
    "A_kernel": "A-tier device kernels (graded against the upstream MUSA implementation)",
    "B_library": "B-tier library dispatches (graded against the expert dispatch)",
}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_rows(baseline: dict, implementation: str) -> Dict[str, float]:
    """The reference's passing latencies per case, keyed by case id."""
    latencies: Dict[str, float] = {}
    for row in baseline.get("results", []):
        if row.get("implementation") != implementation:
            continue
        if row.get("status") != "pass":
            continue
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)) and latency > 0:
            latencies[row["case_id"]] = float(latency)
    return latencies


def rank_report(report: dict, baselines: Dict[str, dict]) -> dict:
    """One report's ratios against its tier's denominator."""
    task_id = report.get("task_id")
    tier = report.get("tier")
    baseline = baselines.get(task_id)
    row: dict = {
        "task_id": task_id,
        "tier": tier,
        "cases_in_report": len(report.get("cases", [])),
        "artifact": report.get("submission_artifact") or report.get("submission"),
    }
    if tier not in TIER_ORDER:
        row["unranked"] = f"the report's tier is {tier!r}, which this scoreboard does not rank"
        return row
    if baseline is None:
        row["unranked"] = "no baseline record for this task, so there is nothing to divide by"
        return row
    denominator = (baseline.get("scoring") or {}).get("speedup_denominator")
    row["denominator"] = denominator
    if not denominator:
        row["unranked"] = "the baseline names no speedup denominator"
        return row

    reference = reference_rows(baseline, denominator)
    ratios: Dict[str, float] = {}
    excluded: Dict[str, str] = {}
    for case_row in report.get("cases", []):
        case_id = case_row.get("case_id")
        runtime_ms = case_row.get("runtime")
        if not isinstance(runtime_ms, (int, float)) or runtime_ms <= 0:
            excluded[case_id] = "the report has no usable runtime for this case"
            continue
        if case_id not in reference:
            excluded[case_id] = f"the baseline has no passing {denominator} measurement for this case"
            continue
        ratios[case_id] = reference[case_id] / float(runtime_ms)

    row["cases_compared"] = len(ratios)
    row["ratios"] = {case_id: round(value, 6) for case_id, value in sorted(ratios.items())}
    row["excluded"] = excluded
    if ratios:
        row["geometric_mean_ratio"] = round(
            math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios)), 6
        )
    return row


def rank(reports: Sequence[Path], baselines: Sequence[Path]) -> dict:
    """Two ranked tables, one per tier, and the reports that could not be ranked."""
    baseline_by_task: Dict[str, dict] = {}
    for path in baselines:
        for match in sorted(glob.glob(str(path))):
            baseline = load_json(Path(match))
            if baseline.get("task_id"):
                baseline_by_task[baseline["task_id"]] = baseline

    rows = [rank_report(load_json(Path(path)), baseline_by_task) for path in reports]
    tables = {}
    for tier in TIER_ORDER:
        ranked = [row for row in rows if row.get("tier") == tier and "geometric_mean_ratio" in row]
        ranked.sort(key=lambda row: row["geometric_mean_ratio"], reverse=True)
        tables[tier] = {
            "title": TIER_TITLES[tier],
            "baselines_found": sum(1 for row in rows if row.get("tier") == tier and "unranked" not in row),
            "ranked": ranked,
        }
    return {
        "unit_note": "ratio = the denominator's latency_ms / the report's runtime, both milliseconds",
        "separate_tracks": "the two tables are never combined; §5's step 8 ranks them apart",
        "tables": tables,
        "unranked": [row for row in rows if "unranked" in row],
    }


def format_tables(result: dict) -> str:
    lines: List[str] = [result["unit_note"], result["separate_tracks"], ""]
    for tier in TIER_ORDER:
        table = result["tables"][tier]
        lines.append(f"== {table['title']}")
        if not table["ranked"]:
            lines.append("   (nothing to rank)")
        for position, row in enumerate(table["ranked"], start=1):
            # Four significant digits: an A-tier submission measured against a
            # hand-written kernel can be a thousandth of the denominator, and "x0.0"
            # would hide exactly the number a reader is looking for.
            lines.append(
                f"   {position}. {row['task_id']}: x{row['geometric_mean_ratio']:.4g} "
                f"over {row['cases_compared']} of {row['cases_in_report']} cases "
                f"(against {row['denominator']})"
            )
        lines.append("")
    if result["unranked"]:
        lines.append("== not ranked")
        for row in result["unranked"]:
            lines.append(f"   {row['task_id']}: {row['unranked']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rank submissions in the two tracks §5 step 8 keeps apart.")
    parser.add_argument("--report", action="append", required=True, help="an evaluation report; repeatable")
    parser.add_argument("--baseline", action="append", default=[], help="a baseline record or a glob; repeatable")
    parser.add_argument("--output", default=None, help="write the tables here (JSON)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the tables")
    arguments = parser.parse_args(argv)

    result = rank(arguments.report, arguments.baseline)
    text = json.dumps(result, indent=2, ensure_ascii=False) if arguments.json else format_tables(result)
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
