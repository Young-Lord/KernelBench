"""
Dispatch trace collection and checking for the B_library tier.

The A tier only asks whether the answer is right. The B tier also asks *how* the
answer was reached, so a submission has to say which path it took, and the
evaluator has to check that against the maintainer's expectation. That is what
makes the tier gradeable at all: without a trace, a submission that quietly
falls back to the slow composition path is indistinguishable from one that hit
the fused library.

The transport is deliberately dumb. The evaluator points an environment variable
at a scratch file, tells the submission which case it is running, the submission
appends one JSON object, and the evaluator reads the file back:

    KB_DISPATCH_TRACE=/tmp/.../dispatch_trace.jsonl
    KB_DISPATCH_CASE_ID=hidden_gap_003

A file rather than an attribute on the model, because the custom fallback path
JIT-compiles and may hand work to a helper process; an environment variable and a
file survive that boundary, an in-process attribute does not.

Stdlib only, so the trace checks can run anywhere.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: Environment variable the submission reads to find where to append its trace.
TRACE_ENV_VAR = "KB_DISPATCH_TRACE"

#: Environment variable carrying the case the submission is being evaluated on.
#:
#: The trace requires a `case_id`, and the evaluator checks the record against
#: the case it asked for, so the submission has to be told which case it is
#: answering. It cannot be inferred from the inputs: two cases may share a shape
#: and differ only in an attribute, and a submission must not guess. Passed
#: through the environment for the same reason the trace path is -- the custom
#: fallback path JIT-compiles and may run its work out of process.
CASE_ID_ENV_VAR = "KB_DISPATCH_CASE_ID"

#: Paths the B tier understands, in preference order.
DISPATCH_PATHS = ("fused_library", "library_composition", "custom_fallback")

#: Fields every trace record must carry.
REQUIRED_TRACE_FIELDS = ("case_id", "selected_path", "probe_status")


class DispatchTraceError(ValueError):
    """Raised when a trace file cannot be read or parsed at all."""


@contextmanager
def _environment(variables: Dict[str, Optional[str]]) -> Iterator[None]:
    """Set environment variables for the duration of the block, then restore them.

    Restoring an unset variable to being unset matters: repeated evaluations in
    one process must not leak a previous case's identity into the next one.
    """
    missing = object()
    previous = {name: os.environ.get(name, missing) for name in variables}
    for name, value in variables.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value  # type: ignore[assignment]


@contextmanager
def dispatch_trace(path: Optional[Path] = None, case_id: Optional[str] = None) -> Iterator[Path]:
    """Point the trace env vars at a scratch file for the duration of the block.

    Yields the trace path. The previous values of both variables are restored on
    exit (including being unset), so repeated evaluations in one process do not
    leak state into each other.

    Args:
        path: where to collect the trace. A temporary file is used when omitted.
        case_id: the case being evaluated. Published through CASE_ID_ENV_VAR so
            the submission can label its trace records; when omitted the variable
            is left unset and a submission cannot know which case it is running.
    """
    if path is None:
        handle, name = tempfile.mkstemp(prefix="kb_dispatch_trace_", suffix=".jsonl")
        os.close(handle)
        path = Path(name)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    with _environment({TRACE_ENV_VAR: str(path), CASE_ID_ENV_VAR: case_id}):
        yield path


def current_case_id() -> Optional[str]:
    """The case the evaluator is currently running, as published to submissions.

    Exposed so a submission can read it the same way the checks do rather than
    hard-coding the variable name in two places.
    """
    return os.environ.get(CASE_ID_ENV_VAR)


def read_trace(path: Path) -> List[dict]:
    """Parse a JSON Lines dispatch trace.

    Blank lines are ignored so a submission may flush with trailing newlines.
    A malformed line is a hard error: a trace that cannot be parsed cannot be
    graded, and a half-readable trace is worse than none.

    Raises:
        DispatchTraceError: the file is missing, or a line is not a JSON object.
    """
    path = Path(path)
    if not path.is_file():
        raise DispatchTraceError(f"dispatch trace is missing: {path}")

    records: List[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DispatchTraceError(f"{path}:{line_number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise DispatchTraceError(f"{path}:{line_number} is not a JSON object")
        records.append(record)
    return records


def check_trace(
    records: Sequence[dict],
    expected_path: Optional[str] = None,
    required_fields: Optional[Sequence[str]] = None,
    case_id: Optional[str] = None,
) -> List[str]:
    """Check a trace against the task contract.

    Two things are being asked. Structurally, every record must carry the
    required fields with non-empty values. Semantically, if an expected path is
    given it must be the *only* path the submission took: a submission that
    answers the same configuration through two different paths is either probing
    something it should not, or is unstable, and either way the trace cannot
    stand behind a speedup number.

    Args:
        records: parsed trace records
        expected_path: the maintainer's expected path for this case, if known
        required_fields: fields every record must carry; defaults to
            REQUIRED_TRACE_FIELDS
        case_id: when given, records for other cases are ignored

    Returns:
        A list of human-readable problems; empty means the trace is acceptable.
    """
    fields = tuple(REQUIRED_TRACE_FIELDS if required_fields is None else required_fields)

    errors: List[str] = []
    if not records:
        return ["dispatch trace is empty; the submission never recorded a path"]

    relevant = records
    if case_id is not None:
        relevant = [record for record in records if record.get("case_id") == case_id]
        if not relevant:
            recorded = sorted({str(record.get("case_id")) for record in records})
            return [f"no dispatch trace record for case {case_id!r}; saw {recorded}"]

    for index, record in enumerate(relevant):
        for field in fields:
            value = record.get(field)
            if value is None or value == "":
                errors.append(f"trace record {index} is missing {field!r}")

    if errors:
        return errors

    selected = {record["selected_path"] for record in relevant}
    unknown = sorted(selected - set(DISPATCH_PATHS))
    if unknown:
        errors.append(f"trace reports unknown dispatch paths: {unknown}")

    if expected_path is not None:
        if selected != {expected_path}:
            errors.append(
                "dispatch trace does not match the expected path: "
                f"expected {{{expected_path!r}}}, recorded {sorted(selected)}"
            )

    return errors


def summarize_trace(records: Sequence[dict]) -> dict:
    """Aggregate a trace for the report.

    The counts feed the B-tier metrics that are not covered by speedup: how many
    cases each path served (Coverage Rate) and whether the fused path was taken
    where it was supposed to be (Dispatch Efficiency).
    """
    counts: Dict[str, int] = {}
    for record in records:
        path = str(record.get("selected_path"))
        counts[path] = counts.get(path, 0) + 1

    probe_statuses = sorted({str(record.get("probe_status")) for record in records if record.get("probe_status")})
    return {
        "records": len(records),
        "case_ids": sorted({str(record.get("case_id")) for record in records}),
        "selected_path_counts": counts,
        "probe_statuses": probe_statuses,
    }
