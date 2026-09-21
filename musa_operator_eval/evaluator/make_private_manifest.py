"""Build a private B-tier case manifest from a maintainer's hidden-case plan.

The hidden set is the half of §4.3 the agent never sees, and it is the half that
decides the score: a submission that could read these shapes could branch on them
instead of probing the library, which is the one thing the B tier exists to test.

So this tool does not invent cases. Every case comes from a plan the maintainer
wrote and measured, and the tool's job is to refuse a plan that would not do what
the guide asks:

  * every gap names a reason from §4.3's fixed vocabulary, and the reasons a
    task's contract requires must actually appear;
  * a task is allowed one reason category only if its contract says so, which is
    the `admission.gap_reason_coverage` field rather than a silent pass;
  * there is at least one performance case and at least one generalisation probe,
    because a set of gaps alone measures coverage and not speed;
  * no hidden shape repeats a public one, so a submission cannot pass the hidden
    set by passing the public one.

The earlier version of this file held one task's cases in its own source and
stamped that task's id into the output, so it could only ever build that one
manifest. Reading the cases from a plan is what lets a second task have a hidden
set at all.
"""

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

# §4.3's fixed vocabulary for gap reasons. A plan naming anything else is
# describing a boundary the evaluator cannot count, which is the whole point of
# fixing the vocabulary in the first place.
GAP_REASONS = frozenset({"unsupported_shape", "semantic_mismatch", "layout_mismatch", "multi_operator_required"})

# §4.3's fixed tag vocabulary, shared with the public list.
CASE_TAGS = frozenset({"smoke", "correctness", "boundary", "non_aligned", "extreme", "perf", "generalization"})

MINIMUM_GAP_CASES = 3

# A stage a hidden plan must include, and the reason it is not optional.
REQUIRED_ROLES = {
    "gap": "a gap is what distinguishes a submission that probes from one that assumes",
    "perf": "a set of gaps alone cannot measure speedup, and the gate is a speedup ratio",
    "probe": "a probe is a shape in no list, which is what catches a submission that memorised the lists",
}


def load_plan(evidence: dict) -> list:
    """The maintainer's hidden cases, or a refusal naming what is missing."""
    cases = evidence.get("cases")
    if not cases:
        raise ValueError("the plan carries no `cases`; this tool builds a manifest from a plan, not from nothing")
    return cases


def check_plan(plan: list, task: dict) -> list:
    """Everything that would make the manifest say something untrue."""
    problems = []

    roles = {}
    for case in plan:
        roles.setdefault(case.get("role"), []).append(case)

    for role, why in REQUIRED_ROLES.items():
        if not roles.get(role):
            problems.append(f"the plan has no {role} case, and {why}")

    for case in plan:
        tag = case.get("tag")
        if tag not in CASE_TAGS:
            problems.append(f"{case.get('case_id')}: tag {tag!r} is not one of §4.3's fixed tags")

    gap_reasons = set()
    for case in roles.get("gap", []):
        gap = case.get("library_gap") or {}
        reason = gap.get("reason")
        if reason not in GAP_REASONS:
            problems.append(
                f"{case.get('case_id')}: gap reason {reason!r} is not one of §4.3's "
                f"{sorted(GAP_REASONS)}"
            )
            continue
        if not gap.get("evidence"):
            problems.append(f"{case.get('case_id')}: a gap with no evidence is an unmeasured claim")
        if not case.get("expected_path"):
            problems.append(f"{case.get('case_id')}: a gap has to say which path it expects")
        gap_reasons.add(reason)

    if len(roles.get("gap", [])) < MINIMUM_GAP_CASES:
        problems.append(
            f"the plan has {len(roles.get('gap', []))} gaps where §4.3 asks for at least {MINIMUM_GAP_CASES}"
        )

    # §4.3 asks for at least two reason categories, and most entry-scoped tasks
    # cannot reach two: a task whose reference fixes its scale, has no window and
    # is causal has exactly one library boundary to find. That is a real property
    # of the task rather than a gap in the plan, so it is allowed -- but only when
    # the contract says so and says why, which is what `gap_reason_coverage` is.
    coverage = (task.get("admission") or {}).get("gap_reason_coverage") or {}
    if len(gap_reasons) < 2 and coverage.get("state") != "open":
        problems.append(
            f"the gaps cover {len(gap_reasons)} reason category where §4.3 asks for at least 2, "
            "and the contract does not record that this task cannot reach a second under "
            "`admission.gap_reason_coverage`"
        )

    # A reason the contract requires but the plan never produces is a promise the
    # hidden set does not keep.
    required = set((task.get("library_policy") or {}).get("required_gap_reasons") or [])
    missing = required - gap_reasons
    if missing:
        problems.append(f"the contract requires gap reasons {sorted(missing)} that no gap case produces")

    # A hidden shape that repeats a public one lets a submission pass the hidden
    # set by having passed the public one, which defeats hiding it.
    public_shapes = {json.dumps(case["shape"], sort_keys=True) for case in task.get("_public_cases", [])}
    for case in plan:
        if json.dumps(case.get("shape"), sort_keys=True) in public_shapes:
            problems.append(f"{case.get('case_id')}: this shape is already in the public list at {case.get('shape')}")

    seeds = [case.get("seed") for case in plan]
    if len(set(seeds)) != len(seeds):
        problems.append("two hidden cases share a seed; §4.3 asks the two lists not to share seeds")

    return problems


def build_manifest(plan: list, task: dict) -> dict:
    cases = []
    for case in plan:
        stamped = dict(case)
        stamped.pop("role", None)
        stamped["golden"] = GOLDEN_MANIFEST_RELPATH.format(case_id=case["case_id"])
        cases.append(stamped)

    return {
        "schema_version": "1.0.0",
        "task_id": task["id"],
        "tier": task["tier"],
        "visibility": "private",
        "case_generation": {
            "tool": CASE_GENERATION_TOOL,
            "deterministic": True,
            "note": "A private list hides which configuration is a library gap; that is the case the task exists to make the submission discover.",
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, required=True, help="the maintainer's hidden-case plan")
    parser.add_argument("--task-dir", type=Path, required=True,
                        help="the task the plan belongs to, read for its contract and its public cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task = json.loads((args.task_dir / "task.json").read_text(encoding="utf-8"))
    public = args.task_dir / "public_cases.json"
    task["_public_cases"] = json.loads(public.read_text(encoding="utf-8"))["cases"] if public.is_file() else []

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    declared_snapshot = evidence.get("environment_snapshot")
    if declared_snapshot and declared_snapshot != task["target_environment"]["snapshot_id"]:
        print(
            f"refusing to build {task['id']}: the plan was measured in {declared_snapshot} and the task "
            f"targets {task['target_environment']['snapshot_id']}. A gap measured on another "
            "configuration is not evidence about this one.",
        )
        return 2

    plan = load_plan(evidence)
    problems = check_plan(plan, task)
    if problems:
        for problem in problems:
            print(f"  - {problem}")
        print(f"refusing to build {task['id']}: {len(problems)} problem(s) in the plan")
        return 1

    manifest = build_manifest(plan, task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
