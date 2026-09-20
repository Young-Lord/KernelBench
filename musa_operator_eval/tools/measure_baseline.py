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
  * the correctness tolerance, via `get_tolerance_for_precision`. Using the task's
    own `tolerances` block instead would let a task grade its own baseline.

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

    python musa_operator_eval/tools/measure_baseline.py \
        --task-dir musa_operator_eval/tasks/kb_l3_43_b \
        --naive  musa_operator_eval/private/mingpt_causal_attention_b_v0/naive/model_new.py \
        --expert musa_operator_eval/private/mingpt_causal_attention_b_v0/expert/model_new.py \
        --unavailable musa_operator_eval/private/mingpt_causal_attention_b_v0/unavailable.json \
        --output musa_operator_eval/private/mingpt_causal_attention_b_v0/baseline.json
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
from kernelbench.eval import (  # noqa: E402
    _process_input_tensor,
    get_tolerance_for_precision,
    set_seed,
)

PROTOCOL = {"warmup": 10, "measurements": 100, "rounds": 5, "statistic": "median", "timer": "musa_event"}

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


def time_median(fn, warmup=PROTOCOL["warmup"], measurements=PROTOCOL["measurements"], rounds=PROTOCOL["rounds"]):
    """MUSA-event timing, median of per-round means."""
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(measurements):
            fn()
        end.record()
        torch.musa.synchronize()
        samples.append(start.elapsed_time(end) / measurements)
    return sorted(samples)[len(samples) // 2]


def _measure(name, call, expected, tolerance):
    row = {"case_id": None, "implementation": name, "status": "pass", "latency_ms": None}
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
            row["latency_ms"] = time_median(call)
    except Exception as exc:
        row.update(status="error", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
    return row


def measure_case(case, task, problem_source, implementations: dict, unavailable: dict, trace_dir: Path):
    reference, init_inputs, inputs, dtype = load_case(task, case, problem_source)
    tolerance = get_tolerance_for_precision(dtype)

    # The dispatch contract asks a submission to record the path it took per case,
    # and the expert dispatch is a submission. Without these the expert raises on
    # the missing variable; with them the baseline carries the same trace the
    # grading run will, so the two can be compared.
    #
    # The file is removed rather than appended to: a submission records once per
    # process, so a second run of the same case would otherwise leave the first
    # run's record behind it and a reader could not tell which run the file
    # describes.
    trace_path = trace_dir / f"{case['case_id']}.jsonl"
    trace_path.unlink(missing_ok=True)
    os.environ["KB_DISPATCH_CASE_ID"] = case["case_id"]
    os.environ["KB_DISPATCH_TRACE"] = str(trace_path)

    with torch.no_grad():
        expected = reference(*inputs)

    # The reference is rebuilt per case by `load_case`, so its own cost is the
    # cost of a fresh model at this case's shapes. Measured for scale: it is the
    # number a submission is being asked to beat by a library call.
    rows = [_measure_existing(REFERENCE_ROLE, reference, inputs, expected, tolerance)]
    for name, model_class in implementations.items():
        model = build_model(model_class, init_inputs, dtype, case["seed"])
        rows.append(_measure(name, lambda model=model: model(*inputs), expected, tolerance))

    for name, reason in unavailable.items():
        rows.append({"case_id": case["case_id"], "implementation": name, "status": "unavailable",
                     "latency_ms": None, "reason": reason})
    return rows


def _measure_existing(name, model, inputs, expected, tolerance):
    return _measure(name, lambda: model(*inputs), expected, tolerance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--naive", type=Path, required=True,
                        help="the task's naive library composition; the gate's numerator")
    parser.add_argument("--expert", type=Path, required=True,
                        help="the task's expert dispatch; the speedup denominator")
    parser.add_argument("--blind", type=Path, default=None,
                        help="a submission that calls the library's SDPA and assumes it is fused")
    parser.add_argument("--unavailable", type=Path, default=None,
                        help="JSON mapping an implementation name to why it cannot be built here")
    parser.add_argument("--snapshot-id", default=None,
                        help="defaults to the snapshot the task declares it was measured in")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task, cases_manifest, problem_source = load_task(args.task_dir)

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
        "tolerance_source": "kernelbench.eval.get_tolerance_for_precision",
        "implementations": sorted({row["implementation"] for row in results}),
        "results": results,
        "scoring": {
            "speedup_denominator": EXPERT_ROLE,
            "reference_implementation": NAIVE_ROLE,
            "correctness_gate": "all_hidden_cases",
            "minimum_naive_to_expert_ratio": 1.3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
