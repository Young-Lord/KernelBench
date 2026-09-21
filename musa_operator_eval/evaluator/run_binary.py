"""Run a task's compiled path: build it, check what it linked, run it, grade it.

Guide §4.6 asks for a frozen tensor ABI that does not go through libtorch, and
§4.7 asks for one pipeline that produces a verdict. `run_task.py` drives the
Python-module path; this drives the compiled one, over the same stages and the
same report shape, so a task can be graded through either without a second set of
goldens or a second notion of what passing means.

Three things differ from the Python path, and each is why this tool exists:

* The build is a real compile, so §4.7's B-only link check has an artifact to read.
  On the Python path an expert module compiles nothing, and the check honestly
  reports that it checked nothing.
* §4.6 confines library calls to the dispatch file and the Host launch file. A
  Python submission is one file, so that confinement is invisible to a scanner;
  a compiled kernel declares its regions with markers, so this tool reads them and
  rejects a library call that sits outside the regions the contract names.
* The submission runs as a separate process that imports no torch. On the Python
  path the framework itself imports torch into the process the submission runs in,
  which is the deviation `starter.abi.deviation` records.

Usage:

    run_binary.py --task-dir musa_operator_eval/tasks/kb_l1_97_a \
        --submission private/example_submission/kb_l1_97_a \
        --cases private/scaled_dot_product_attention_a_v0/cases.private.json \
        --generated-dir private/generated/scaled_dot_product_attention_a_v0 \
        --build-dir /tmp/build_binary --output /tmp/binary_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EVALUATOR_DIR = Path(__file__).resolve().parent
ROOT = EVALUATOR_DIR.parent
REPO_TOP = ROOT.parent

sys.path.insert(0, str(EVALUATOR_DIR))
import exit_codes  # noqa: E402  (the package's own layout decides the path)

REPORT_SCHEMA_VERSION = "1.0.0"
DEFAULT_PRECISION = "fp32"

# §4.6: a library call belongs in the dispatch file and in the Host launch file.
# In a compiled submission there is one file, so the region is what distinguishes
# them: a probe or a fallback may call a library, and the kernel may not.
LIBRARY_REGIONS = {"probe_and_dispatch", "host_launch"}

# The libraries a submission may reach, as §4.6's whitelist names them. Matched
# against the lowercased line, because a call can arrive as a symbol, an include
# path or a namespace. The MUSA runtime is deliberately absent: musaMalloc and
# musa_runtime.h are the platform the kernel runs on, not a library the kernel is
# asked to dispatch to, and the A tier's prohibited list names muDNN, muBLAS,
# SDPA, ATen and CUDA residue rather than the runtime.
LIBRARY_NAMES = ("mudnn", "mublas", "torch", "at::", "c10", "cuda")

# A preprocessor include is a declaration, not a call: nothing runs when a header is
# read, and the lines that matter -- where a library is actually reached -- are still
# checked wherever they sit. The exemption is narrower than it looks: it applies to
# the B tier only (whose libraries the contract whitelists), and only to headers that
# resolve to one of those whitelisted libraries, so `#include <torch/extension.h>` in
# a submission that may not link libtorch stays a finding.
INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")

MARKER = re.compile(r"^\s*//\s*---\s*(BEGIN|END)\s+([A-Za-z_][A-Za-z0-9_]*)\s*---\s*$", re.MULTILINE)


# The framework's own per-precision floors, replicated so this driver does not have
# to import torch to know them. `test_the_replica_matches_the_framework` compares
# them against `kernelbench.eval.get_tolerance_for_precision` wherever torch is
# importable, so a change there fails a test rather than silently re-basing grading.
FRAMEWORK_TOLERANCES = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 1e-2}
DEFAULT_FRAMEWORK_TOLERANCE = 1e-2


def resolve_case_tolerance(task: dict, precision: str) -> dict:
    """The bar a case is graded at, resolved the way the Python path resolves it.

    §4.1 puts the tolerance in the task contract and `resolve_tolerance` makes it a
    loosen-only override of the framework's per-dtype floor, applied to `atol` and
    `rtol` alike. This driver asks the same question of the same declaration rather
    than inventing a second bar: two graders holding one submission to two different
    numbers is how a case passes on one path and fails on the other.
    """
    declared = (task.get("tolerances") or {}).get("max_abs_error")
    framework = FRAMEWORK_TOLERANCES.get(precision, DEFAULT_FRAMEWORK_TOLERANCE)
    if declared is None:
        threshold = framework
        source = "kernelbench.eval.get_tolerance_for_precision"
    else:
        threshold = max(framework, float(declared))
        source = "task_contract.tolerances.max_abs_error"
    return {"atol": threshold, "rtol": threshold, "source": source, "declared_max_abs_error": declared}


def fail(report: dict, code: int, errors: Sequence[str]) -> Tuple[int, dict]:
    """Record a stage failure in the one shape the report schema requires."""
    report["summary"] = {"errors": list(errors), "exit_code": code, "passed_overall": False}
    return code, report


def load_module(name: str, path: Path):
    """Load a sibling tool by file path, the way this package's tools refer to each other."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(directory: Path) -> Dict[str, dict]:
    """Read a tensor manifest and its blobs into arrays keyed by tensor name."""
    manifest_path = directory / "tensors.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no tensor manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors: Dict[str, dict] = {}
    for record in manifest.get("tensors", []):
        raw = (directory / record["file"]).read_bytes()
        dtype = record["dtype"]
        if dtype == "float32":
            values = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif dtype == "float16":
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dtype == "bfloat16":
            # bfloat16 has no numpy dtype; the top half of a float32 is it.
            values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"tensor {record['name']} has dtype {dtype}, which this reader cannot compare")
        expected = 1
        for axis in record["shape"]:
            expected *= int(axis)
        if values.size != expected:
            raise ValueError(
                f"tensor {record['name']} holds {values.size} values and its shape {record['shape']} needs {expected}"
            )
        tensors[record["name"]] = {
            "dtype": dtype,
            "shape": [int(axis) for axis in record["shape"]],
            "values": values.reshape([int(axis) for axis in record["shape"]]),
        }
    return tensors


def compare_tensors(expected: Dict[str, dict], produced: Dict[str, dict], tolerance: dict) -> Tuple[bool, Optional[float], List[str]]:
    """Compare a produced tensor set against a golden one under one case's tolerance.

    Returns (passed, max_abs_error, errors). Missing or extra tensors are errors
    rather than absences: a submission that writes the right numbers under the
    wrong name has not answered the case.
    """
    errors: List[str] = []
    atol = float(tolerance.get("atol", 0.0))
    rtol = float(tolerance.get("rtol", 0.0))
    max_error: Optional[float] = None

    for name, golden in expected.items():
        if name not in produced:
            errors.append(f"{name}: the golden has this tensor and the output does not")
            continue
        got = produced[name]
        if got["shape"] != golden["shape"]:
            errors.append(f"{name}: shape {got['shape']} != {golden['shape']}")
            continue
        difference = np.abs(got["values"] - golden["values"])
        worst = float(difference.max()) if difference.size else 0.0
        max_error = worst if max_error is None else max(max_error, worst)
        allowed = atol + rtol * np.abs(golden["values"])
        violating = int(np.count_nonzero(difference > allowed))
        if violating:
            errors.append(
                f"{name}: {violating} of {difference.size} values outside atol {atol} + rtol {rtol}, worst {worst:.6g}"
            )
    for name in produced:
        if name not in expected:
            errors.append(f"{name}: the output has this tensor and the golden does not")
    return (not errors, max_error, errors)


def marker_regions(source: str) -> Tuple[List[str], List[str]]:
    """Return the declared region names and any marker problems."""
    regions: List[str] = []
    problems: List[str] = []
    open_region: Optional[str] = None
    for kind, name in MARKER.findall(source):
        if kind == "BEGIN":
            if open_region is not None:
                problems.append(f"region {name} opens while {open_region} is still open")
            open_region = name
            regions.append(name)
        else:
            if open_region is None:
                problems.append(f"region {name} closes without opening")
            elif open_region != name:
                problems.append(f"region {name} closes while {open_region} is open")
            open_region = None
    if open_region is not None:
        problems.append(f"region {open_region} is never closed")
    return regions, problems


def whitelisted_header_stems(task: dict) -> set:
    """The library names the contract whitelists, spelled the way a header spells them.

    `libmudnn.so` -> `mudnn`, so `mudnn.h` and `mudnn_nn.h` are recognisable while
    `torch/extension.h` is not.
    """
    stems = set()
    for entry in (task.get("library_policy") or {}).get("allowed_libraries", []):
        bare = entry.strip().lower()
        if bare.startswith("lib"):
            bare = bare[3:]
        bare = bare.split(".so")[0]
        if bare:
            stems.add(bare)
    return stems


def is_whitelisted_include(line: str, stems: set) -> bool:
    """Whether a line is an include of a header belonging to a whitelisted library."""
    match = INCLUDE.match(line)
    if not match:
        return False
    stem = Path(match.group(1)).name.lower().split(".")[0]
    if stem.startswith("lib"):
        stem = stem[3:]
    return any(stem.startswith(name) for name in stems)


def region_of_line(source: str) -> List[Optional[str]]:
    """The region each line of the source sits in, None outside every region."""
    ownership: List[Optional[str]] = []
    current: Optional[str] = None
    for line in source.splitlines():
        match = MARKER.match(line)
        if match:
            kind, name = match.group(1), match.group(2)
            current = name if kind == "BEGIN" else None
            ownership.append(current)
            continue
        ownership.append(current)
    return ownership


def audit_compiled_source(task: dict, source: str) -> List[str]:
    """§4.6's rules that a compiled submission can actually be held to.

    Two findings, in the vocabulary the exit codes already use: a library call
    outside the regions the contract names, and -- for the A tier, which has no
    library to call -- a reference to the upstream stacks at all.
    """
    findings: List[str] = []
    declared = (task.get("starter", {}).get("cpp") or {}).get("regions", [])
    found, problems = marker_regions(source)
    for problem in problems:
        findings.append(f"region markers: {problem}")
    if declared and set(found) != set(declared):
        findings.append(
            f"the compiled source declares regions {sorted(found)} and the inventory says {sorted(declared)}"
        )

    ownership = region_of_line(source)
    tier = task.get("tier")
    stems = whitelisted_header_stems(task)
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lowered = stripped.lower()
        names_the_stack = any(name in lowered for name in LIBRARY_NAMES)
        if not names_the_stack:
            continue
        # A header the ABI itself ships is not a library call; the frozen half
        # includes nothing from the upstream stacks, so no exclusion is needed
        # beyond the comments filtered above.
        if tier == "A_kernel":
            findings.append(
                f"line {number}: the A tier's kernel may not name {stripped.split()[0][:40]!r}; "
                "the upstream stacks are on its prohibited list"
            )
            continue
        if is_whitelisted_include(line, stems):
            continue
        region = ownership[number - 1]
        if region not in LIBRARY_REGIONS:
            findings.append(
                f"line {number}: a library call sits in region {region or 'none'}, and §4.6 allows one only in "
                f"{sorted(LIBRARY_REGIONS)}"
            )
    return findings


def resolve_libraries(task: dict, musa_home: Path) -> Tuple[List[str], List[str]]:
    """Map the contract's whitelist onto libraries that exist, and name those that do not.

    The whitelist is written in the vocabulary of the contract ("muDNN", "SDPA"),
    while a linker needs a file. Resolving here rather than guessing in the build
    script keeps one list authoritative: what the build links is what the link
    check then verifies against the same whitelist.
    """
    policy = task.get("library_policy") or {}
    linkable: List[str] = []
    unresolvable: List[str] = []
    for entry in policy.get("allowed_libraries", []):
        bare = entry.lower()
        for prefix in ("lib",):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
        bare = bare.split(".so")[0]
        for candidate in (f"{bare}.so", f"lib{bare}.so"):
            if (musa_home / "lib" / candidate).exists():
                linkable.append(bare)
                break
        else:
            unresolvable.append(entry)
    return linkable, unresolvable


def build_submission(
    task: dict, task_dir: Path, submission: Path, build_dir: Path, libraries: Sequence[str], verbose: bool = False
) -> dict:
    """Run the starter's declared build entry point in its own process."""
    script = (task.get("starter", {}).get("cpp") or {}).get("build")
    if not script:
        return {"passed": False, "environment_missing": False, "detail": "the task names no compiled starter to build", "build_dir": str(build_dir), "script": None}
    # A task names its starter's files relative to its own package, the same way
    # `starter.editable` does; an absolute path is taken as given, which is what
    # lets a test drive the stage without writing into the shipped tree.
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = task_dir / script_path
    if not script_path.is_file():
        return {
            "passed": False,
            "environment_missing": False,
            "detail": f"the declared build script {script_path} is not there",
            "build_dir": str(build_dir),
            "script": script,
        }
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [str(script_path), "--submission", str(submission), "--build-dir", str(build_dir)]
    for library in libraries:
        command += ["--extra-library", library]
    environment = dict(os.environ)
    environment.setdefault("MUSA_HOME", "/usr/local/musa")
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=3600, env=environment, cwd=str(REPO_TOP)
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"passed": False, "environment_missing": False, "detail": f"the build script could not be run: {error}", "build_dir": str(build_dir), "script": script}
    detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
    if completed.returncode != 0:
        missing = "not found" in detail.lower() or "No such file" in detail
        return {
            "passed": False,
            "environment_missing": missing,
            "detail": detail,
            "build_dir": str(build_dir),
            "script": script,
            "returncode": completed.returncode,
        }
    return {"passed": True, "environment_missing": False, "detail": detail, "build_dir": str(build_dir), "script": script}


# The framework's measurement protocol, in calls: warm up, then time this many.
# The framework's own call counts: `kernelbench.timing` warms up three times and then
# runs a hundred trials, which is what `measure_baseline.py` measures the denominators
# with and what `run_task.py`'s reports carry. The runner defaults to them so that a
# compiled number and a Python number are the same number measured the same way.
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 100

# The tool that turns a reference's state into tensors a submission can read. It is a
# separate process on purpose: it needs torch, and this driver stays importable
# without it, which is what lets a stateless task's compiled path run on a machine
# that has no framework at all.
MATERIALIZE_TOOL = EVALUATOR_DIR / "materialize_model_state.py"


class StagingError(Exception):
    """A case's inputs could not be produced for the compiled path."""


def framework_mean(values: Sequence[float]) -> float:
    """The mean as `kernelbench.timing.get_timing_stats` reports it.

    That function renders its summary as `f"{value:.3g}"` -- three significant digits,
    not three decimals -- so a baseline's `latency_ms` is a mean in that format. The
    statistic a compiled `runtime` records has to be the same one for the same reason
    the baselines were re-measured: a ratio between a mean and a median is a ratio
    between two summaries. Reproducing the format keeps a compiled row and a baseline
    row readable side by side.
    """
    total = 0.0
    for value in values:
        total += value
    return float(f"{total / len(values):.3g}")


def read_timing(stdout: str) -> Optional[dict]:
    """The per-call times the runner reported, or None if it reported none.

    The clock is in the frozen runner and the statistic is here: a runner that chose
    the number would be choosing a grade, and a driver that timed the whole process
    would be timing the device's first-call initialisation instead of the submission.
    The protocol fields travel with the samples so the record says how the number was
    taken -- a warm-cache host clock and a cold-cache event pair are not the same
    measurement, and only the runner knows which one it made.
    """
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            report = json.loads(stripped)
        except ValueError:
            continue
        calls = report.get("per_call_ms")
        if isinstance(calls, list) and calls and all(isinstance(value, (int, float)) for value in calls):
            thrash = int(report.get("l2_thrash_bytes", 0) or 0)
            return {
                "warmup": int(report.get("warmup", 0)),
                "repeat": int(report.get("repeat", len(calls))),
                "discarded": int(report.get("discarded", 0) or 0),
                "timer": str(report.get("timer", "host_clock")),
                "l2_thrash_bytes": thrash,
                "cache": "cold" if thrash else "warm",
                "per_call_ms": [float(value) for value in calls],
            }
    return None


def stage_case_inputs(
    task: dict,
    task_dir: Path,
    case: dict,
    cases_manifest_path: Path,
    generated_dir: Path,
    stage_root: Path,
    precision: str,
) -> Tuple[Path, dict]:
    """The input directory a compiled submission reads, and what was staged into it.

    A case's own tensors are what they always were. When the contract declares that
    the reference carries state -- weights a torch-free process cannot rebuild -- the
    evaluator materializes that state with the framework's own construction and hands
    it over beside the case's tensors, under names the contract names. The generated
    tree is never written to: the staged copy is what the runner reads.
    """
    case_id = case["case_id"]
    source = generated_dir / case_id / "input"
    if not source.is_dir():
        raise StagingError(f"no generated input at {source}")
    cpp = ((task.get("starter") or {}).get("cpp") or {})
    declaration = cpp.get("model_state")
    configuration = cpp.get("case_configuration") or {}
    # A compiled submission needs what the contract declares it cannot rebuild: the
    # reference's state when there is one, and the case's construction arguments
    # (`n_head`, `max_seqlen`, a default scale) when the construction reads them. Either
    # one means the case's own input directory is not enough, so both go through the
    # stage copy; neither means the case directory is handed over untouched.
    if not declaration and not (configuration.get("args") or []):
        return source, {"materialized": False, "reason": "the contract declares no model state or configuration"}

    staged = stage_root / case_id / "input"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.is_file():
            shutil.copy2(entry, staged / entry.name)

    command = [
        sys.executable,
        str(MATERIALIZE_TOOL),
        "--task-dir",
        str(task_dir),
        # The case, by identity and with the list it came from: the reference's
        # construction depends on the case's parameters, not only on its seed, so a
        # bare seed would build the problem file's default model and hand over a state
        # that belongs to a different case.
        "--cases",
        str(cases_manifest_path),
        "--case",
        str(case_id),
        # The case's own dtype, not the run's flag. A case carries its dtype in the
        # manifest and its tensors were generated in it, so the state has to be cast the
        # same way: passing the run's precision here handed every bfloat16 case a state
        # cast to float16, and the submission read those bytes as bfloat16 -- weights that
        # are not the reference's weights, in a file that says they are.
        "--precision",
        str(case.get("dtype") or precision),
        "--output-dir",
        str(staged),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, cwd=str(REPO_TOP))
    except (OSError, subprocess.SubprocessError) as error:
        raise StagingError(f"the state could not be materialized: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise StagingError(f"materializing the reference's state failed: {detail}")
    try:
        materialized = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise StagingError(f"the materializer's report could not be read: {error}") from error

    if configuration.get("args"):
        names = list(configuration["args"])
        values = materialized.get("init_inputs")
        if not isinstance(values, list) or len(values) != len(names):
            raise StagingError(
                f"the contract declares {len(names)} construction arguments and the reference returns "
                f"{values!r}: a compiled submission would be handed the wrong configuration"
            )
        (staged / "case.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "configuration": {name: value for name, value in zip(names, values)},
                    "source": configuration.get("source"),
                    "note": (
                        "The case's own construction arguments, by the names the contract declares. They are "
                        "not tensors and a torch-free process cannot rebuild them; two cases of one task can "
                        "differ here."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    if declaration is None:
        # Configuration without state: the case's own tensors plus case.json, and no
        # state tensors to merge.
        return staged, {
            "materialized": False,
            "reason": "the contract declares no model state",
            "configuration": {name: value for name, value in zip(list(configuration.get("args") or []), materialized.get("init_inputs") or [])},
        }

    prefix = declaration.get("tensor_prefix", "state_")
    for record in materialized.get("tensors", []):
        if not str(record.get("name", "")).startswith(prefix):
            raise StagingError(
                f"the materializer wrote {record.get('name')!r}, and the contract says its tensors are named {prefix!r}..."
            )
    if materialized.get("tensors"):
        manifest_path = staged / "tensors.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_names = {record["name"] for record in manifest.get("tensors", [])}
        overlap = case_names & {record["name"] for record in materialized["tensors"]}
        if overlap:
            raise StagingError(f"the materialized state collides with the case's own tensors: {sorted(overlap)}")
        manifest["tensors"] = list(manifest.get("tensors", [])) + list(materialized["tensors"])
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return staged, {
            "materialized": True,
            "digest": materialized.get("digest"),
            "tensors": len(materialized["tensors"]),
            "dtype": materialized.get("dtype"),
            "seed": materialized.get("seed"),
            "tensor_prefix": prefix,
            # The case's own entry decided the construction, so the row says which case
            # and which init arguments the state was built from: a case whose
            # configuration differs from the problem file's defaults is the normal
            # situation, and a state built from the defaults would be shaped like a
            # different model.
            "case_id": materialized.get("case_id", case_id),
            "init_inputs": materialized.get("init_inputs"),
        }
    shutil.rmtree(staged, ignore_errors=True)
    return source, {
        "materialized": False,
        "digest": materialized.get("digest"),
        "reason": "the reference holds no state",
    }


def run_binary_case(
    task: dict,
    binary: Path,
    case: dict,
    generated_dir: Path,
    stability_reruns: int = 1,
    timeout: int = 900,
    tolerance: Optional[dict] = None,
    input_dir: Optional[Path] = None,
    model_state: Optional[dict] = None,
    warmup: int = 0,
    repeat: int = 1,
) -> dict:
    """Run one case through the compiled runner and grade it against its golden."""
    case_id = case["case_id"]
    row: dict = {"case_id": case_id, "tag": case.get("tag"), "tier": task.get("tier"), "compiled": True}
    if tolerance is not None:
        row["tolerance_used"] = tolerance
    if model_state is not None:
        row["model_state"] = model_state
    if case.get("expected_path"):
        row["expected_path"] = case["expected_path"]

    input_dir = input_dir or (generated_dir / case_id / "input")
    golden_record = case.get("golden")
    if not golden_record:
        row.update({"skipped": True, "passed": False, "reason": "the case names no golden"})
        return row
    golden_dir = (generated_dir / golden_record).parent
    if not input_dir.is_dir():
        row.update({"skipped": True, "passed": False, "reason": f"no generated input at {input_dir}"})
        return row
    if not golden_dir.is_dir():
        row.update({"skipped": True, "passed": False, "reason": f"no golden at {golden_dir}"})
        return row

    policy = task.get("library_policy") or {}
    trace_path = generated_dir / f"{case_id}.trace.jsonl"
    # A trace file that outlives its run turns a stale record into this run's verdict: the
    # check reads every record for the case, so a path recorded by an earlier submission or
    # by an earlier build of this one would fail a case whose own trace is correct. The
    # trace is this run's evidence, so it starts empty.
    if trace_path.exists():
        trace_path.unlink()
    environment = dict(os.environ)
    if policy.get("trace_env_var"):
        environment[policy["trace_env_var"]] = str(trace_path)
    if policy.get("case_id_env_var"):
        environment[policy["case_id_env_var"]] = case_id
    # The tensors arrive as arguments, so a submission cannot know which directory they
    # came from -- and the staged case directory is where the contract's declaration puts
    # what is not a tensor: the case's construction arguments. Nothing about the path
    # reveals the case's expected path or its golden; those stay with the evaluator.
    environment["KB_CASE_INPUT_DIR"] = str(input_dir)

    output_dir = generated_dir / f"{case_id}.binary_output"
    if output_dir.exists():
        for stale in output_dir.iterdir():
            stale.unlink()

    try:
        expected_tensors = read_manifest(golden_dir)
    except (OSError, ValueError, KeyError) as error:
        row.update({"skipped": True, "passed": False, "reason": f"the golden could not be read: {error}"})
        return row

    digests: List[str] = []
    worst: Optional[float] = None
    errors: List[str] = []
    for attempt in range(max(1, stability_reruns + 1)):
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(binary), str(input_dir), str(output_dir), "--warmup", str(max(0, warmup)), "--repeat", str(max(1, repeat))],
                capture_output=True, text=True, timeout=timeout, env=environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            row.update({"passed": False, "errors": [f"the runner could not be run: {error}"]})
            return row
        # `runtime` is the submission's own work at steady state, in milliseconds, taken
        # the way the framework takes it and a baseline is measured: a MUSA event pair
        # around each call, the L2 cache flushed before each one, the first timed call
        # discarded, and the mean of what is left. `wall_ms` is what the process took
        # end to end -- device initialisation, library mapping, reading the case and
        # writing the answer included -- and it is kept because a reader comparing two
        # numbers deserves to know which one is which.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timing = read_timing(completed.stdout or "")
        if attempt == 0:
            row["wall_ms"] = round(elapsed_ms, 3)
            if timing:
                calls = timing["per_call_ms"]
                row["runtime"] = framework_mean(calls)
                row["timing"] = {
                    "warmup": timing["warmup"],
                    "repeat": timing["repeat"],
                    "discarded": timing["discarded"],
                    "timer": timing["timer"],
                    "cache": timing["cache"],
                    "l2_thrash_bytes": timing["l2_thrash_bytes"],
                    "mean_ms": framework_mean(calls),
                    "min_ms": round(min(calls), 3),
                    "samples": [round(value, 6) for value in calls],
                }
            else:
                row["runtime"] = round(elapsed_ms, 3)
                row["timing"] = None
        if completed.returncode != 0:
            row.update(
                {
                    "passed": False,
                    "errors": [f"the runner exited {completed.returncode}: {(completed.stderr or '').strip()[-400:]}"],
                }
            )
            return row
        try:
            produced_tensors = read_manifest(output_dir)
        except (OSError, ValueError, KeyError) as error:
            row.update({"passed": False, "errors": [f"the output could not be read: {error}"]})
            return row
        passed, max_error, case_errors = compare_tensors(expected_tensors, produced_tensors, tolerance or {})
        # Every run's bytes go into the set, the first one included: a stability
        # check that only compared the re-runs against each other would call a
        # runner stable precisely when it changed between the graded run and the
        # repeat.
        digests.append(blob_digest(output_dir).hex())
        if attempt == 0:
            worst = max_error
            errors = case_errors
        elif not passed:
            errors += [f"re-run {attempt}: {error}" for error in case_errors]

    # The first pass is what grading uses; the re-runs only have to agree with
    # themselves, which is §4.7's stability question asked of an output artifact.
    if stability_reruns > 0 and digests:
        agrees = len(set(digests)) == 1
        row["stability"] = {
            "reruns": stability_reruns,
            "agrees": agrees,
            "reason": (
                "the compiled runner reproduced its own output byte for byte"
                if agrees
                else "the compiled runner produced different bytes on a re-run with the same inputs"
            ),
        }
    else:
        row["stability"] = {
            "reruns": 0,
            "agrees": True,
            "reason": "the stability re-run is switched off",
        }

    row["correctness"] = passed
    if worst is not None:
        row["max_abs_error"] = worst
    if task.get("tier") == "B_library":
        row["dispatch_trace_passed"], trace_errors = check_trace(policy, case, trace_path)
        if trace_errors:
            row["dispatch_trace_errors"] = trace_errors
    row["passed"] = bool(passed and row.get("stability", {}).get("agrees", True) and row.get("dispatch_trace_passed", True) is not False)
    if errors:
        row["errors"] = errors
    return row


def blob_digest(directory: Path) -> bytes:
    """A digest over every produced blob, so a re-run's bytes are what is compared."""

    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.bin")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.digest()


def check_trace(policy: dict, case: dict, trace_path: Path) -> Tuple[Optional[bool], List[str]]:
    """§4.7's dispatch trace: the record must exist, be complete, and name the expected path."""
    expected = case.get("expected_path")
    if not expected:
        return None, []
    if not trace_path.is_file():
        return False, [f"the runner wrote no dispatch trace at {trace_path}"]
    records = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                return False, [f"the trace has a line that is not JSON: {error}"]
    if not records:
        return False, ["the dispatch trace is empty; the submission never recorded a path"]
    required = policy.get("required_trace_fields", [])
    errors: List[str] = []
    matching = [record for record in records if record.get("case_id") == case["case_id"]]
    if not matching:
        errors.append(f"the trace has no record for case {case['case_id']}")
    for record in matching:
        for field in required:
            if field not in record:
                errors.append(f"a trace record is missing the required field {field}")
        if record.get("selected_path") != expected:
            errors.append(
                f"the trace records {record.get('selected_path')!r} and the case expects {expected!r}"
            )
    return (not errors), errors


def evaluate_binary_task(
    task_dir: Path,
    submission: Path,
    cases_manifest: Path,
    generated_dir: Path,
    build_dir: Path,
    precision: str = DEFAULT_PRECISION,
    stability_reruns: int = 1,
    verbose: bool = False,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
) -> Tuple[int, dict]:
    """Drive the compiled path through §4.7's stages and return (exit code, report)."""
    report: dict = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "binary",
        "task_id": None,
        "tier": None,
        "precision": precision,
        "stages": [],
        "cases": [],
        "summary": {},
    }
    try:
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return fail(report, exit_codes.EXIT_INPUT_UNAVAILABLE, [f"the task contract could not be read: {error}"])
    report["task_id"] = task.get("id")
    report["tier"] = task.get("tier")
    report["submission"] = str(submission)
    report["build"] = {"build_dir": str(build_dir)}

    # One rule, one implementation: the permitted precisions come from the tensor
    # contract's `input_dtypes` through the same function the Python driver uses. This
    # read `tolerances.allowed_precisions`, a key no contract has, so the guard never
    # fired and a B-tier task could be graded at a precision its fused path does not
    # answer -- producing a plausible report of a different task's behaviour.
    run_task = load_module("musa_run_task", EVALUATOR_DIR / "run_task.py")
    permitted = run_task.allowed_precisions(task)
    if permitted and precision not in permitted:
        return fail(
            report,
            exit_codes.EXIT_PRECISION_CONTRACT_MISMATCH,
            [f"the tensor contract permits {permitted} and the run asked for {precision}"],
        )
    report["stages"].append("environment")
    environment = run_task.environment_stage(task)
    report["environment"] = environment
    if not environment.get("passed", True):
        return fail(report, exit_codes.EXIT_ENVIRONMENT_UNAVAILABLE, [f"environment mismatch: {environment.get('mismatches')}"])

    report["stages"].append("case_generation")
    try:
        manifest = json.loads(Path(cases_manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return fail(report, exit_codes.EXIT_INPUT_UNAVAILABLE, [f"the case manifest could not be read: {error}"])
    cases = manifest.get("cases", [])
    report["case_sources"] = [str(cases_manifest)]

    kernel_source = submission / "kernel.mu"
    if not kernel_source.is_file():
        return fail(report, exit_codes.EXIT_INPUT_UNAVAILABLE, [f"no kernel.mu under {submission}"])

    report["stages"].append("static_audit")
    findings = audit_compiled_source(task, kernel_source.read_text(encoding="utf-8"))
    report["static_audit"] = {"passed": not findings, "errors": findings}
    if findings:
        return fail(report, exit_codes.EXIT_STATIC_AUDIT_FORBIDDEN_API, findings)

    return _build_check_and_evaluate(
        task, task_dir, submission, build_dir, generated_dir, cases, Path(cases_manifest), report,
        stability_reruns, verbose, warmup, repeat,
    )


def _build_check_and_evaluate(
    task: dict,
    task_dir: Path,
    submission: Path,
    build_dir: Path,
    generated_dir: Path,
    cases: List[dict],
    # The manifest the cases came from travels with them: staging the reference's
    # state needs the case's own entry, and rebuilding it from the problem file's
    # defaults produced a state belonging to a different case.
    cases_manifest: Path,
    report: dict,
    stability_reruns: int,
    verbose: bool,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
) -> Tuple[int, dict]:
    run_task = load_module("musa_run_task", EVALUATOR_DIR / "run_task.py")
    musa_home = Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))
    libraries, unresolvable = resolve_libraries(task, musa_home)

    report["stages"].append("build")
    build = build_submission(task, task_dir, submission, build_dir, libraries)
    build["libraries_linked"] = libraries
    if unresolvable:
        build["whitelist_entries_without_a_library"] = unresolvable
    report["build"] = build
    if not build["passed"]:
        code = exit_codes.EXIT_ENVIRONMENT_UNAVAILABLE if build["environment_missing"] else exit_codes.EXIT_BUILD_FAILED
        return fail(report, code, [build["detail"][-400:]])

    binary = build_dir / "runner"
    if not binary.is_file():
        return fail(report, exit_codes.EXIT_BUILD_FAILED, [f"the build script reported success and {binary} is not there"])
    report["submission_artifact"] = str(binary)

    if task.get("tier") == "B_library":
        report["stages"].append("link_whitelist_check")
        link = run_task.link_whitelist_stage(task, build_dir)
        report["link_whitelist"] = link
        if not link.get("passed", True):
            return fail(report, exit_codes.EXIT_LINK_WHITELIST_VIOLATION, link.get("errors", []))

    report["stages"].append("evaluation")
    tolerance = resolve_case_tolerance(task, report.get("precision") or DEFAULT_PRECISION)
    report["tolerance_used"] = tolerance
    stage_root = build_dir / "cases"
    rows = []
    for case in cases:
        if verbose:
            print(f"[binary] {case['case_id']}", file=sys.stderr)
        try:
            case_inputs, model_state = stage_case_inputs(
                task, task_dir, case, cases_manifest, generated_dir, stage_root,
                report.get("precision") or DEFAULT_PRECISION,
            )
        except StagingError as error:
            return fail(
                report,
                exit_codes.EXIT_CASE_GENERATION_FAILED,
                [f"{case['case_id']}: {error}"],
            )
        rows.append(
            run_binary_case(
                task,
                binary,
                case,
                generated_dir,
                stability_reruns=stability_reruns,
                tolerance=tolerance,
                input_dir=case_inputs,
                model_state=model_state,
                warmup=warmup,
                repeat=repeat,
            )
        )
    report["cases"] = rows

    failed = [row for row in rows if not row.get("passed")]
    passed_count = len(rows) - len(failed)
    report["summary"] = {
        "cases": len(rows),
        "passed": passed_count,
        "failed": len(failed),
        "skipped": len([row for row in rows if row.get("skipped")]),
    }
    if not failed:
        report["summary"]["exit_code"] = exit_codes.EXIT_OK
        report["summary"]["passed_overall"] = True
        return exit_codes.EXIT_OK, report

    code = exit_codes.EXIT_CORRECTNESS_FAILED
    if any(not row.get("stability", {}).get("agrees", True) for row in failed):
        code = exit_codes.EXIT_STABILITY_FAILED
    elif any(row.get("dispatch_trace_passed") is False for row in failed):
        code = exit_codes.EXIT_DISPATCH_TRACE_MISMATCH
    elif all(row.get("tag") == "boundary" for row in failed):
        code = exit_codes.EXIT_BOUNDARY_FAILED
    report["summary"]["exit_code"] = code
    report["summary"]["passed_overall"] = False
    report["summary"]["errors"] = [f"{row['case_id']}: {row.get('errors', row.get('reason', 'failed'))}" for row in failed]
    return code, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a task's compiled path and grade it.")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--submission", required=True, help="the directory holding kernel.mu")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--generated-dir", required=True, help="where each case's input/ and golden/ live")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--stability-reruns", type=int, default=1)
    # The framework's own protocol: ten calls to settle, then a hundred measurements
    # (kernelbench's `num_correct_trials`/`num_perf_trials`). Matching it is what makes
    # a compiled number and a Python number the same kind of number.
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    generated_dir = Path(arguments.generated_dir).resolve()
    build_dir = Path(arguments.build_dir).resolve() if arguments.build_dir else generated_dir / "binary_build"
    code, report = evaluate_binary_task(
        task_dir=Path(arguments.task_dir).resolve(),
        submission=Path(arguments.submission).resolve(),
        cases_manifest=Path(arguments.cases).resolve(),
        generated_dir=generated_dir,
        build_dir=build_dir,
        precision=arguments.precision,
        stability_reruns=arguments.stability_reruns,
        verbose=arguments.verbose,
        warmup=arguments.warmup,
        repeat=arguments.repeat,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if arguments.output:
        Path(arguments.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(
        f"[binary] exit={code} ({exit_codes.describe(code)}) stages={report.get('stages')} "
        f"cases={report['summary'].get('cases')} passed={report['summary'].get('passed')}",
        file=sys.stderr,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
