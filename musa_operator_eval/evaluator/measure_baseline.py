"""Measure the B-tier baseline implementations and emit the admission gate's input.

The gate compares `naive_library_composition` against `expert_dispatch` per case
and takes the geometric mean, so both have to be measured with the same protocol
on the same environment, and both have to actually pass correctness -- a ratio
against a wrong answer would admit a task for the wrong reason.

Three things are inherited from `kernelbench.eval` rather than re-derived, because
the submissions these baselines are measured against are driven by that module and
a denominator measured under a different protocol admits a task for the wrong
reason:

  * weight alignment. The framework builds `Model` and `ModelNew` from the same
    seed with the same init inputs and lets construction reproduce the weights. It
    does not copy a state dict, so neither does this: a state dict is stricter than
    the harness, and a baseline that passed one would be measuring a model the
    harness never builds.
  * input and parameter casting, via `_process_input_tensor`.
  * the correctness tolerance, resolved by `kernelbench.eval.resolve_tolerance`.
    The framework's per-dtype value is the floor and the task's `tolerances` block
    can only loosen it, so a task cannot grade itself leniently without bound; what
    matters more is that this resolves to the same number the grader will use, so
    the gate is not measured at a bar nobody is scored against.

Every implementation is supplied as a module path. The pilot's tooling named three
implementations in its own source, which meant it could only ever drive a problem
whose inputs are already `q, k, v` and whose init inputs are already `scale,
causal, window_left, window_right`. No entry-scoped package is shaped like that:
`kb_l3_43_b` takes one input and five init inputs, and reaches the attention core
through its own projections. So each implementation is a model-level `nn.Module`
here, and reaching the attention core is the model's business.

Implementations that cannot be honestly built on this checkout are recorded as
`unavailable` with a reason rather than being approximated by something else. An
approximation under a real implementation's name is worse than a gap.

    python musa_operator_eval/evaluator/measure_baseline.py \
        --task-dir musa_operator_eval/tasks/kb_l3_43_b \
        --naive  musa_operator_eval/private/mingpt_causal_attention_b_v0/naive/model_new.py \
        --expert musa_operator_eval/private/mingpt_causal_attention_b_v0/expert/model_new.py \
        --unavailable musa_operator_eval/private/mingpt_causal_attention_b_v0/unavailable.json \
        --output musa_operator_eval/private/mingpt_causal_attention_b_v0/baseline.hidden.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from run_task import case_source, load_task

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent

sys.path.insert(0, str(REPO_TOP / "src"))
from kernelbench.dispatch_trace import (  # noqa: E402
    DispatchTraceError,
    check_trace,
    read_trace,
)
from kernelbench.eval import (  # noqa: E402
    _process_input_tensor,
    resolve_tolerance,
    set_seed,
)

# The framework's own protocol, taken from `kernelbench.timing` rather than
# re-derived. This block used to be a hand-rolled one -- warmup 10, five rounds of a
# hundred calls, the median of the round means, and no cache flush -- which meant a
# baseline's `latency_ms` and the `runtime` a `run_task.py` report records for the same
# implementation were two numbers from two protocols, and every ratio between them was
# a ratio between those protocols. Upstream's loop is the one the reports come from, so
# it is the one the denominators are measured with: warm up, empty the allocator, then
# one pass of `measurements` trials, each preceded by an L2 thrash so the number is a
# cold-cache number, with the first trial discarded and the mean taken.
PROTOCOL = {
    "timer": "cuda_event",
    "warmup": 3,
    "measurements": 100,
    "discard_first": 1,
    "statistic": "mean",
    "cache": "cold",
    "rounds": 1,
    "source": "kernelbench.timing.time_execution_with_cuda_event",
    "note": (
        "one pass, as the framework runs it; the samples a record carries are its trials, and "
        "the quantiles are over those trials rather than over round means"
    ),
}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DEVICE = torch.device("musa")
BACKEND = "musa"

# The roster the baseline template publishes. Two of these are supplied as module
# paths and two are usually absent; keeping the names here is what lets a reader
# line the two files up.
REFERENCE_ROLE = "reference_model"
NAIVE_ROLE = "naive_library_composition"
BLIND_ROLE = "torch_musa_sdpa"
EXPERT_ROLE = "expert_dispatch"
# The A tier's denominator: the hand-written MUSA implementation the task's
# speedup is taken against. Independent of whether it is the reference.
UPSTREAM_ROLE = "upstream_musa"


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------


def load_case(task: dict, case: dict, problem_source: str):
    """Build the reference and its inputs the way `eval_kernel_against_ref` does.

    Returned separately because every implementation is rebuilt from the same seed
    and the same init inputs, which is how the framework aligns their weights.
    """
    namespace: dict = {}
    exec(case_source(problem_source, case, task["case_parameters"]), namespace)

    dtype = DTYPES[case["dtype"]]

    set_seed(case["seed"])
    init_inputs = [
        _process_input_tensor(value, DEVICE, BACKEND, dtype) for value in namespace["get_init_inputs"]()
    ]

    with torch.no_grad():
        set_seed(case["seed"])
        reference = namespace["Model"](*init_inputs)
        reference = reference.to(device=DEVICE, dtype=dtype)

    set_seed(case["seed"])
    inputs = [_process_input_tensor(value, DEVICE, BACKEND, dtype) for value in namespace["get_inputs"]()]
    return reference, init_inputs, inputs, dtype


def build_model(model_class, init_inputs, dtype: torch.dtype, seed: int):
    """Instantiate an implementation under the same seed the reference had."""
    with torch.no_grad():
        set_seed(seed)
        model = model_class(*init_inputs)
        return model.to(device=DEVICE, dtype=dtype)


def parse_implementation(value: str) -> tuple:
    """Parse a `role=path[=note]` argument.

    The B tier's roster is fixed by §4.4 and its two denominators are always the
    same two roles, so the flags are named after them. The A tier's roster is
    fixed too and different -- the upstream implementation is the denominator
    there -- so it is supplied as roles instead of having a second set of named
    flags that would drift from this one.
    """
    parts = value.split("=", 2)
    if len(parts) < 2 or not parts[0]:
        raise argparse.ArgumentTypeError(f"{value!r} is not role=path")
    role, path = parts[0], Path(parts[1])
    note = parts[2] if len(parts) > 2 else None
    return role, path, note


def load_model_class(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_implementation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load an implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelNew


def load_unavailable(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------


def percentile(samples, fraction):
    """The nearest-rank percentile of a sorted sample list.

    §4.4 asks a baseline record for quantiles rather than one number. With a
    handful of rounds the higher ones are coarse, and recording them is still
    better than not: a median alone cannot show a run that is bimodal, and a
    bimodal baseline is a baseline whose ratio depends on which mode a submission
    was measured against.
    """
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return ordered[rank - 1]


def time_samples(fn):
    """The framework's timing loop, called rather than reimplemented.

    `kernelbench.timing.time_execution_with_cuda_event` is what `run_task.py`'s
    reports are produced by, so a denominator measured with it is the same kind of
    number as the runtime it will be divided into: same timer, same warmup, same
    trial count, same cache state, same statistic.

    Returns `(samples, peak_before, peak_after)`; the peak counters are read around
    the loop because §4.4 asks the record for memory, and the framework's loop does
    not report it.
    """
    from kernelbench.timing import time_execution_with_cuda_event

    peak_before = torch.musa.max_memory_allocated() if hasattr(torch.musa, "max_memory_allocated") else None
    samples = time_execution_with_cuda_event(
        fn,
        (),
        num_warmup=PROTOCOL["warmup"],
        num_trials=PROTOCOL["measurements"],
        discard_first=PROTOCOL["discard_first"],
        device=DEVICE,
    )
    peak_after = torch.musa.max_memory_allocated() if hasattr(torch.musa, "max_memory_allocated") else None
    return samples, peak_before, peak_after


def time_statistic(fn):
    """What the gate divides by: the framework's own summary of the same trials."""
    from kernelbench.timing import get_timing_stats

    samples, _before, _after = time_samples(fn)
    return float(get_timing_stats(samples)["mean"])


def _measure(name, call, expected, tolerance, trace=None, must_trace=False, required_fields=None):
    """Run one implementation on one case and return its row.

    `trace` is `(trace_dir, case_id, expected_path)` when the implementation is
    expected to record a dispatch, and None when it is not. The trace file is
    truncated before the run rather than appended to, so the records belong to this
    implementation and this case and to nothing earlier.
    """
    row = {"case_id": None, "implementation": name, "status": "pass", "latency_ms": None}
    if trace is not None:
        trace_dir, case_id, _expected_path = trace
        trace_path = Path(trace_dir) / f"{case_id}.{name}.jsonl"
        trace_path.unlink(missing_ok=True)
        os.environ["KB_DISPATCH_CASE_ID"] = case_id
        os.environ["KB_DISPATCH_TRACE"] = str(trace_path)
    try:
        with torch.no_grad():
            actual = call()
        torch.musa.synchronize()
        if tuple(actual.shape) != tuple(expected.shape):
            row.update(status="fail", reason=f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
            return row
        if not torch.allclose(actual.float(), expected.float(), atol=tolerance, rtol=tolerance):
            difference = torch.max(torch.abs(actual.float() - expected.float())).item()
            row.update(status="fail", reason=f"max_abs_diff {difference:.6f} exceeds {tolerance}")
            return row
        with torch.no_grad():
            samples, peak_before, peak_after = time_samples(call)
        from kernelbench.timing import get_timing_stats

        stats = get_timing_stats(samples)
        # The mean, because that is what the framework records as a submission's
        # `runtime`: a denominator has to be the same statistic as the thing it
        # divides, or the ratio carries the difference between two summaries.
        row["latency_ms"] = float(stats["mean"])
        row["latency_ms_stats"] = {key: stats[key] for key in ("mean", "std", "min", "max", "num_trials")}
        row["latency_ms_p90"] = percentile(samples, 0.90)
        row["latency_ms_p10"] = percentile(samples, 0.10)
        row["latency_ms_samples"] = [round(value, 6) for value in samples]
        row["throughput_per_second"] = (
            round(1000.0 / row["latency_ms"], 4) if row["latency_ms"] else None
        )
        if peak_before is not None and peak_after is not None:
            row["peak_memory_bytes"] = max(0, peak_after - peak_before)

        # The dispatch half of the gate. A speedup is only meaningful once it is
        # known which path earned it, so a row whose trace does not match is not a
        # fast row, it is an unanswered one, and it must not reach the ratio.
        if trace is not None:
            _, case_id, expected_path = trace
            problems = _trace_problems(trace_path, case_id, expected_path, required_fields, must_trace)
            row["recorded_paths"] = _recorded_paths(trace_path)
            if problems is None:
                row["dispatch_trace_passed"] = None  # the implementation records nothing
            else:
                row["dispatch_trace_passed"] = not problems
                if problems:
                    row.update(status="fail", reason="; ".join(problems)[:200])
                    return row
    except Exception as exc:
        row.update(status="error", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
    return row


def _recorded_paths(trace_path: Path):
    try:
        return sorted({str(record.get("selected_path")) for record in read_trace(trace_path)})
    except DispatchTraceError:
        return None


def _trace_problems(trace_path, case_id, expected_path, required_fields, must_trace):
    """Problems with this implementation's trace, or None when it records none.

    An implementation that records nothing is not a failure unless it is the expert.
    The naive is the scoring baseline -- it is the number a submission is asked to
    beat, not a submission -- so it is allowed to be an ordinary library call with
    no dispatch story. The expert exists to demonstrate the dispatch decision, and
    the gate's second half reads exactly that, so an expert with no trace leaves the
    half unmeasured and counts against the entry.
    """
    try:
        records = read_trace(trace_path)
    except DispatchTraceError as error:
        return [str(error)] if must_trace else None
    if not records:
        return ["expert recorded no dispatch; the gate's second half cannot be read"] if must_trace else None
    return check_trace(
        records,
        expected_path=expected_path,
        required_fields=required_fields,
        case_id=case_id,
    )


def measure_case(case, task, problem_source, implementations: dict, unavailable: dict, trace_dir: Path):
    reference, init_inputs, inputs, dtype = load_case(task, case, problem_source)
    # Resolved through the same helper the grader uses, so the admission gate is
    # measured at the tolerance submissions are actually scored against.
    tolerance = resolve_tolerance(dtype, (task.get("tolerances") or {}).get("max_abs_error"))

    # Each implementation gets its own trace file, created and truncating inside
    # `_measure`. A single file per case would mix the naive's records with the
    # expert's, and the check would then be comparing two dispatch stories at once.
    with torch.no_grad():
        expected = reference(*inputs)

    # The reference is rebuilt per case by `load_case`, so its own cost is the
    # cost of a fresh model at this case's shapes. Measured for scale: it is the
    # number a submission is being asked to beat by a library call.
    rows = [_measure_existing(REFERENCE_ROLE, reference, inputs, expected, tolerance)]
    expected_path = case.get("expected_path")
    required_fields = (task.get("library_policy") or {}).get("required_trace_fields")
    for name, model_class in implementations.items():
        model = build_model(model_class, init_inputs, dtype, case["seed"])
        rows.append(
            _measure(
                name,
                lambda model=model: model(*inputs),
                expected,
                tolerance,
                trace=(trace_dir, case["case_id"], expected_path),
                must_trace=(name == EXPERT_ROLE),
                required_fields=required_fields,
            )
        )

    for name, reason in unavailable.items():
        rows.append({"case_id": case["case_id"], "implementation": name, "status": "unavailable",
                     "latency_ms": None, "reason": reason})
    return rows


def _measure_existing(name, model, inputs, expected, tolerance):
    return _measure(name, lambda: model(*inputs), expected, tolerance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=None,
                        help="the case manifest to measure. Defaults to the task's public list; point it at "
                             "the private one to measure the set the gate is actually defined over, which "
                             "is the only set an admission decision may be based on.")
    parser.add_argument("--naive", type=Path,
                        help="the task's naive library composition; the gate's numerator")
    parser.add_argument("--expert", type=Path,
                        help="the task's expert dispatch; the speedup denominator")
    parser.add_argument("--blind", type=Path, default=None,
                        help="a submission that calls the library's SDPA and assumes it is fused")
    parser.add_argument("--unavailable", type=Path, default=None,
                        help="JSON mapping an implementation name to why it cannot be built here")
    parser.add_argument("--snapshot-id", default=None,
                        help="defaults to the snapshot the task declares it was measured in")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tier",
        default="B_library",
        choices=["B_library", "A_kernel"],
        help=(
            "the roster and the scoring block follow the tier: the B tier's speedup is against "
            "the expert dispatch, the A tier's against the upstream implementation"
        ),
    )
    parser.add_argument(
        "--implementation",
        action="append",
        type=parse_implementation,
        default=None,
        metavar="ROLE=PATH",
        help="an implementation to measure under this role; repeatable. Required for the A tier.",
    )
    args = parser.parse_args()

    # Removed before the work starts rather than written at the end, so a run that
    # dies partway leaves nothing behind. It used to leave the previous run's file,
    # and `candidate_gate.py` read it and reported a decision for a baseline that
    # had not been measured -- which is worse than no decision, because it looks
    # like one.
    args.output.unlink(missing_ok=True)

    task, cases_manifest, problem_source = load_task(args.task_dir, args.cases)

    if args.tier == "A_kernel":
        # §4.4's A tier: the upstream implementation is the speedup denominator and
        # the library paths are reference points beside it. The roster is supplied
        # because which module stands for `upstream_musa` is a per-package fact.
        implementations = {}
        unavailable = {}
        for role, path, note in args.implementation or []:
            if path.is_file():
                implementations[role] = load_model_class(path)
            else:
                unavailable[role] = note or f"no implementation at {path}"
        if not implementations:
            print("[baseline] the A tier needs at least one --implementation ROLE=PATH", file=sys.stderr)
            return 2
    else:
        missing = [name for name, value in (("--naive", args.naive), ("--expert", args.expert)) if value is None]
        if missing:
            print(f"[baseline] the B tier measures against {', '.join(missing)}", file=sys.stderr)
            return 2
        implementations = {NAIVE_ROLE: load_model_class(args.naive), EXPERT_ROLE: load_model_class(args.expert)}
        if args.blind is not None:
            implementations[BLIND_ROLE] = load_model_class(args.blind)
        unavailable = load_unavailable(args.unavailable)

    snapshot_id = args.snapshot_id or task["target_environment"]["snapshot_id"]

    trace_dir = args.output.parent / "baseline_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases_manifest["cases"]:
        print(f"[baseline] {case['case_id']}  {case['dtype']}", flush=True)
        for row in measure_case(case, task, problem_source, implementations, unavailable, trace_dir):
            row["case_id"] = case["case_id"]
            row["dtype"] = case["dtype"]
            results.append(row)
            latency = f"{row['latency_ms']:.4f}" if row.get("latency_ms") else "-"
            print(f"    {row['implementation']:<28}{row['status']:<12}{latency}", flush=True)

    payload = {
        "schema_version": "1.0.0",
        "task_id": task.get("id"),
        "tier": task.get("tier"),
        "environment_snapshot": snapshot_id,
        "measurement_protocol": PROTOCOL,
        "weight_alignment": "re-seed and rebuild, as kernelbench.eval does; no state dict is copied",
        "tolerance_source": "kernelbench.eval.resolve_tolerance",
        # The block above names the resolver; these two record what it resolved to
        # for this run, which is the part a reader needs in order to judge a
        # `max_abs_diff ... exceeds ...` row. The declared value is the task's own
        # `tolerances.max_abs_error`, and the resolver takes the larger of it and
        # the framework's per-dtype value, so a task can loosen grading and not
        # tighten it.
        "declared_tolerance": (task.get("tolerances") or {}).get("max_abs_error"),
        # Keyed by the manifest's dtype name, resolved from the torch dtype: the
        # framework's string vocabulary is the driver's (`fp16`, `bf16`, `fp32`),
        # which is not the manifest's (`float16`, `bfloat16`, `float32`).
        "resolved_tolerances": {
            name: resolve_tolerance(DTYPES[name], (task.get("tolerances") or {}).get("max_abs_error"))
            for name in sorted({case["dtype"] for case in cases_manifest["cases"]})
        },
        "implementations": sorted({row["implementation"] for row in results}),
        "results": results,
        "scoring": (
            {
                "speedup_denominator": UPSTREAM_ROLE,
                "reference_implementation": REFERENCE_ROLE,
                "correctness_gate": "all_hidden_cases",
                # §4.4 lists torch_musa eager as its own entry and asks it only to
                # cross-check semantics. On this tier the reference model *is* plain
                # PyTorch run eagerly, so the same measurement answers both and a
                # second row under a second name would be the same numbers twice.
                "torch_musa_eager": "the reference model: this tier's reference is plain PyTorch evaluated eagerly",
            }
            if args.tier == "A_kernel"
            else {
                "speedup_denominator": EXPERT_ROLE,
                "reference_implementation": NAIVE_ROLE,
                "correctness_gate": "all_hidden_cases",
                "minimum_naive_to_expert_ratio": 1.3,
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
