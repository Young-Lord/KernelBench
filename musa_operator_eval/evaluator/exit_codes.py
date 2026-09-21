"""Named exit codes for the MUSA task driver (build guide §4.7).

The guide requires one exit code per failed evaluator stage, with `0` meaning
pass, so that a failure can be attributed without opening the report. A bare `1`
cannot do that: it says only "something failed" and leaves the caller to parse
JSON to find out what.

Each value belongs to one stage of the §4.7 pipeline. The static audit gets a
block of its own because §4.7 lists five distinct things it rejects; a submission
that reaches a banned library and one that moves the math onto the host are
different findings and should not share a number.

`0` is the only success code. Callers that need "did it pass?" test
`code == EXIT_OK`; every other value is a specific failure.

All values sit in 0-127 on purpose: a process exit status is truncated to 8 bits
and a shell reports 128+n for a signal, so keeping to the low range means the
number a caller reads is the number returned here.
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------

EXIT_OK = 0

# ---------------------------------------------------------------------------
# Contract (environment collection in §4.7 terms)
# ---------------------------------------------------------------------------

# The task was pinned to a device and a precision when it was measured. Running
# at a precision its tensor contract does not permit reports a different task's
# behaviour, so the run is refused before any device work starts.
#
# The value is 2 rather than an entry of a block below because
# `tests/test_task_packages.py` pins it; changing it would break a test this
# change does not own. It predates the grouped scheme.
EXIT_PRECISION_CONTRACT_MISMATCH = 2

# ---------------------------------------------------------------------------
# Environment and data readiness (§4.7 "环境采集", "Case 生成")
#
# The numbers follow §4.7's row order so the block reads like the pipeline.
# ---------------------------------------------------------------------------

# The device stack could not be imported, so no case could be evaluated. This is
# the machine, not the submission: the guide's environment-collection stage is
# what decides whether a run may start at all.
EXIT_ENVIRONMENT_UNAVAILABLE = 10

# The task package, the case manifest or the submission file could not be read
# at all, so no stage was reached. Distinct from a stage failure: nothing was
# evaluated, and the report says so rather than blaming the submission.
EXIT_INPUT_UNAVAILABLE = 11

# A case's `case_parameters` mapping does not resolve, or a case does not
# specialise into valid Python. A maintainer bug in the task package, not a
# submission failure, and invisible on the device if it is not caught here.
EXIT_CASE_GENERATION_FAILED = 12

# §4.3 requires every case record to name a golden path. The manifest promised
# one and the file is not there, so the ground truth was never generated.
EXIT_GOLDEN_MISSING = 13

# The golden file exists but cannot be trusted: unparsable JSON, a tensor blob
# whose bytes do not match its recorded digest or length, or a record that
# contradicts the case it belongs to.
EXIT_GOLDEN_UNUSABLE = 14

# The golden file is intact but disagrees with a fresh recomputation from the
# case's own seed. Two mutually inconsistent statements of the ground truth mean
# nothing can be graded against either.
EXIT_GOLDEN_MISMATCH = 15

# ---------------------------------------------------------------------------
# Build / compile (§4.7 "构建")
# ---------------------------------------------------------------------------

# The submission did not compile, or compiled but could not be loaded/run. The
# audit had already passed by the time this is returned.
EXIT_BUILD_FAILED = 20

# ---------------------------------------------------------------------------
# Static audit (§4.7 "静态审计")
#
# One value per category the guide names, plus a fallback for a finding that
# does not fit any of them. `classify_static_audit_errors` in run_task.py maps
# the checker's messages onto these.
# ---------------------------------------------------------------------------

# The submission changed a file the task's starter manifest declares frozen.
EXIT_STATIC_AUDIT_OUT_OF_SCOPE_MODIFICATION = 30

# The submission calls a library or primitive the task's contract forbids.
EXIT_STATIC_AUDIT_FORBIDDEN_API = 31

# The submission performs the core computation in host-side tensor math rather
# than on the device.
EXIT_STATIC_AUDIT_HOST_COMPUTATION = 32

# The submission answers from a table of precomputed values instead of
# computing them, so it does not generalise past the cases it was built for.
EXIT_STATIC_AUDIT_TABLE_LOOKUP = 33

# The device source itself is not what it claims: no kernel definition, no
# compilation path, or a backend construct the task does not accept.
EXIT_STATIC_AUDIT_DEVICE_SOURCE = 34

# A finding the static audit produced that maps onto none of the categories
# above. Kept distinct so an unclassified finding is visible rather than folded
# into a category it may not belong to.
EXIT_STATIC_AUDIT_UNCLASSIFIED = 35

# ---------------------------------------------------------------------------
# B-tier structural checks (§4.7 "链接白名单核对", "分派轨迹核对")
# ---------------------------------------------------------------------------

# Reading the build product's export table showed a library linked outside the
# task's whitelist. NOTE: run_task.py does not run this stage yet; the value is
# declared so the stage has a code to return when it is wired.
EXIT_LINK_WHITELIST_VIOLATION = 40

# The submission ran, but its dispatch trace does not match the path the case
# expects, or the trace is structurally incomplete.
EXIT_DISPATCH_TRACE_MISMATCH = 41

# ---------------------------------------------------------------------------
# Correctness, boundary and stability (§4.7 "正确性")
#
# A single case can be a hidden correctness case, a boundary probe or a
# stability re-run; which one it is comes from the case tag, and the three are
# reported with distinct codes so a pattern ("only boundaries fail") is readable
# from the exit status.
# ---------------------------------------------------------------------------

EXIT_CORRECTNESS_FAILED = 50
EXIT_BOUNDARY_FAILED = 51
EXIT_STABILITY_FAILED = 52

# ---------------------------------------------------------------------------
# Performance (§4.7 "性能")
# ---------------------------------------------------------------------------

# A case tagged as a performance case produced no usable measurement, so the
# number the tier scores never materialised.
EXIT_PERFORMANCE_FAILED = 60

# ---------------------------------------------------------------------------
# Summary (§4.7 "汇总")
# ---------------------------------------------------------------------------

# Aggregation itself failed. Distinct from a stage failure: the stages that ran
# passed, but the report could not be assembled.
EXIT_SUMMARY_FAILED = 70


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXIT_CODE_DESCRIPTIONS: Dict[int, str] = {
    EXIT_OK: "all requested stages passed",
    EXIT_PRECISION_CONTRACT_MISMATCH: "the requested --precision is not one the task's tensor contract permits",
    EXIT_ENVIRONMENT_UNAVAILABLE: "the device stack could not be imported, so no case could be evaluated",
    EXIT_INPUT_UNAVAILABLE: "the task package, case manifest or submission file could not be read",
    EXIT_CASE_GENERATION_FAILED: "case parameters do not resolve, or a case does not specialise into valid Python",
    EXIT_GOLDEN_MISSING: "a case declares a golden path but the file was never generated",
    EXIT_GOLDEN_UNUSABLE: "the golden file is unparsable or its tensor bytes contradict its recorded digest/length",
    EXIT_GOLDEN_MISMATCH: "the golden file disagrees with a fresh recomputation from the case seed",
    EXIT_BUILD_FAILED: "the submission failed to compile, or compiled but could not be run",
    EXIT_STATIC_AUDIT_OUT_OF_SCOPE_MODIFICATION: "the submission changed a file the starter manifest freezes",
    EXIT_STATIC_AUDIT_FORBIDDEN_API: "the submission calls a library or primitive its contract forbids",
    EXIT_STATIC_AUDIT_HOST_COMPUTATION: "the submission computes on the host instead of the device",
    EXIT_STATIC_AUDIT_TABLE_LOOKUP: "the submission answers from a precomputed table",
    EXIT_STATIC_AUDIT_DEVICE_SOURCE: "the device source is missing its kernel, compile path or backend construct",
    EXIT_STATIC_AUDIT_UNCLASSIFIED: "the static audit rejected the submission for a reason outside the named categories",
    EXIT_LINK_WHITELIST_VIOLATION: "the build product links a library outside the task's whitelist",
    EXIT_DISPATCH_TRACE_MISMATCH: "the dispatch trace does not match the path the case expects",
    EXIT_CORRECTNESS_FAILED: "the submission's output disagrees with the reference",
    EXIT_BOUNDARY_FAILED: "the submission fails a boundary case (non-aligned, extreme or generalisation probe)",
    EXIT_STABILITY_FAILED: "the submission fails a stability re-run",
    EXIT_PERFORMANCE_FAILED: "a performance case produced no usable measurement",
    EXIT_SUMMARY_FAILED: "the stages that ran passed, but the report could not be assembled",
}


def describe(exit_code: int) -> str:
    """Return the meaning of an exit code, or say that it is not a known code.

    The report carries codes rather than sentences so a consumer can branch on
    them; this is what turns one back into something a human can read without
    keeping the table in their head.
    """
    return EXIT_CODE_DESCRIPTIONS.get(exit_code, f"unknown exit code {exit_code}")
