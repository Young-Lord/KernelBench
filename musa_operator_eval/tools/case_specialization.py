"""Turn one case entry into a runnable problem source.

§4.3 keeps a case list to parameters and seeds, so something has to turn
`{"shape": {...}, "attributes": {...}}` back into a problem the reference can run.
The convention is KernelBench's own: a problem carries module-level constants and
a case re-binds them, with the mapping from case field to constant name declared
in `task.json` under `case_parameters`.

Two tools need this and they have to agree byte for byte. `run_task.py` builds the
problem the submission is graded against, and `generate_cases.py` builds the one
its golden is computed from. If they specialised differently, the golden would
describe a case the grader never runs -- and because both would still produce a
plausible tensor, nothing downstream could tell.

Loading a task package lives here too, for the same reason: both tools have to
read the same `task.json` the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

# The overrides are appended rather than substituted, so the reference stays
# byte-identical to what a reader sees in `problem.py` and only the constants move.
CASE_OVERRIDE_HEADER = "# --- case overrides for {case_id}, appended by the case driver ---"


def resolve_case_field(case: dict, dotted_path: str) -> Any:
    """Read a dotted path such as `attributes.scale` out of a case entry.

    Raises:
        KeyError: the path does not exist in this case, which means the task's
            `case_parameters` mapping and its case list disagree.
    """
    value: Any = case
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"case {case.get('case_id')!r} has no field {dotted_path!r}")
        value = value[part]
    return value


def case_source(problem_source: str, case: dict, case_parameters: Dict[str, str]) -> str:
    """Return the problem source specialized to one case.

    The overrides are appended rather than substituted so the reference stays
    byte-identical to what a reader sees in `problem.py`; the appended block just
    re-binds the module-level defaults before `get_init_inputs` / `get_inputs`
    read them. That is the same idiom KernelBench problems already use for their
    constants.
    """
    assignments = "\n".join(
        f"{variable} = {resolve_case_field(case, path)!r}"
        for path, variable in case_parameters.items()
    )
    return (
        problem_source.rstrip()
        + "\n\n\n"
        + CASE_OVERRIDE_HEADER.format(case_id=case["case_id"])
        + "\n"
        + assignments
        + "\n"
    )


def load_task(task_dir: Path, cases_path: Path = None) -> Tuple[dict, dict, str]:
    """Load the contract, the case list and the reference source for a task.

    Args:
        task_dir: the task package
        cases_path: the case manifest to evaluate. Defaults to the task's
            `public_cases.json`; point it at a private manifest to evaluate the
            hidden set. The cases live outside the task package because the
            hidden ones must not be readable from it.

    Returns:
        (task, cases_manifest, problem_source)

    Raises:
        FileNotFoundError: a required task file is missing.
        ValueError: the contract does not name a problem file or case parameters.
    """
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    cases = json.loads((cases_path or task_dir / "public_cases.json").read_text(encoding="utf-8"))

    problem_file = task.get("problem_file")
    if not problem_file:
        raise ValueError("task.json must declare `problem_file`")
    problem_source = (task_dir / problem_file).read_text(encoding="utf-8")

    return task, cases, problem_source
