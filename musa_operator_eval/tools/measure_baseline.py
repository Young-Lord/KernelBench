"""Measure the B-tier baseline implementations and emit the admission gate's input.

The gate compares `naive_library_composition` against `expert_dispatch` per case
and takes the geometric mean, so both have to be measured with the same protocol
on the same environment, and both have to actually pass correctness -- a ratio
against a wrong answer would admit a task for the wrong reason.

Implementations that cannot be honestly built on this checkout are recorded as
`unavailable` with the reason rather than being approximated by something else.
An approximation under a real implementation's name is worse than a gap.

    python musa_operator_eval/tools/measure_baseline.py \
        --task-dir <task> --expert <model_new.py> --output baseline.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from run_task import case_source, load_task

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent
DEFAULT_TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"
DEFAULT_EXPERT = ROOT / "private" / "sdpa_forward_b_v0" / "expert" / "model_new.py"

PROTOCOL = {"warmup": 10, "measurements": 100, "rounds": 5, "statistic": "median", "timer": "musa_event"}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------


def load_case(task: dict, case: dict, problem_source: str):
    """Instantiate the reference and its inputs for one case."""
    namespace: dict = {}
    exec(case_source(problem_source, case, task["case_parameters"]), namespace)

    torch.manual_seed(case["seed"])
    dtype = DTYPES[case["dtype"]]
    device = torch.device("musa")
    inputs = [tensor.to(device=device, dtype=dtype) for tensor in namespace["get_inputs"]()]

    torch.manual_seed(case["seed"])
    reference = namespace["Model"](*namespace["get_init_inputs"]()).to(device=device, dtype=dtype)
    return reference, inputs, dtype


def attention_mask(shape, attributes, device):
    """The contract's mask: top-left causal, window [q-left, q+right], -1 unbounded."""
    s_q, s_kv = shape["S_q"], shape["S_kv"]
    query_index = torch.arange(s_q, device=device).unsqueeze(1)
    key_index = torch.arange(s_kv, device=device).unsqueeze(0)
    allowed = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
    if attributes["causal"]:
        allowed = allowed & (key_index <= query_index)
    if attributes["window_left"] >= 0:
        allowed = allowed & (key_index >= query_index - attributes["window_left"])
    if attributes["window_right"] >= 0:
        allowed = allowed & (key_index <= query_index + attributes["window_right"])
    return allowed


# ----------------------------------------------------------------------------
# Implementations
#
# Each takes the case context and returns the output tensor for the same inputs
# the reference was given.
# ----------------------------------------------------------------------------


def _expand_kv(q, k, v):
    heads_q, heads_kv = q.shape[1], k.shape[1]
    if heads_q == heads_kv:
        return k, v
    repeats = heads_q // heads_kv
    return k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)


def naive_library_composition(q, k, v, context):
    """Compose the library's own primitives, with no capability probing at all.

    This is the baseline the gate compares against: what a submission produces
    when it reaches for the obvious matmul/softmax/matmul and never asks whether
    a fused kernel exists. Accumulation stays in float32, which the contract
    requires, so this is slow-but-correct rather than a strawman.
    """
    k, v = _expand_kv(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * context["scale"]
    mask = context["mask"]
    scores = scores.masked_fill(~mask, float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True)
    finite = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    shifted = torch.where(mask, scores - finite, torch.full_like(scores, float("-inf")))
    weights = torch.exp(shifted)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return (torch.matmul(weights, v.float()) / denominator).to(q.dtype)


def torch_musa_sdpa_blind(q, k, v, context):
    """Call the library's SDPA and assume it is a fused kernel.

    Likely the most common submission shape, and worth measuring separately: it
    is fast wherever the fused path happens to apply and silently slow
    everywhere it does not, with nothing in the trace to say which happened.
    """
    k, v = _expand_kv(q, k, v)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=context["mask"], is_causal=False
    )


def reference_model(q, k, v, context):
    """The task's own reference, for scale."""
    return context["reference"](q, k, v)


IMPLEMENTATIONS = {
    "reference_model": reference_model,
    "naive_library_composition": naive_library_composition,
    "torch_musa_sdpa_blind": torch_musa_sdpa_blind,
}

# Named in the template but not constructible here. Recorded with a reason so the
# gate sees an honest gap instead of a number borrowed from something else.
UNAVAILABLE = {
    "upstream_musa": "no upstream MUSA SDPA implementation is present in this checkout; the archived one lives on the unmerged source branch",
    "mudnn_fused": "requires calling the muDNN attention op directly rather than through torch_musa, which is a separate adapter that has not been written",
    "torch_musa_eager": "no distinct eager SDPA entry point exists on this build; torch_musa exposes SDPA only through the aten op that torch_musa_sdpa_blind already measures",
}


def load_expert(path: Path):
    spec = importlib.util.spec_from_file_location("expert_model_new", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelNew


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


def measure_case(case, task, problem_source, expert_class, trace_dir: Path):
    reference, inputs, dtype = load_case(task, case, problem_source)
    device = inputs[0].device
    q, k, v = inputs

    os.environ["KB_DISPATCH_CASE_ID"] = case["case_id"]
    os.environ["KB_DISPATCH_TRACE"] = str(trace_dir / f"{case['case_id']}.jsonl")

    with torch.no_grad():
        expected = reference(q, k, v)

    context = {
        "scale": case["attributes"]["scale"],
        "mask": attention_mask(case["shape"], case["attributes"], device),
        "reference": reference,
    }

    rows = []
    for name, implementation in IMPLEMENTATIONS.items():
        rows.append(_measure(name, lambda: implementation(q, k, v, context), expected, dtype))

    expert = expert_class(
        case["attributes"]["scale"], case["attributes"]["causal"],
        case["attributes"]["window_left"], case["attributes"]["window_right"],
    ).to(device=device, dtype=dtype)
    rows.append(_measure("expert_dispatch", lambda: expert(q, k, v), expected, dtype))

    for name, reason in UNAVAILABLE.items():
        rows.append({"case_id": case["case_id"], "implementation": name, "status": "unavailable", "latency_ms": None, "reason": reason})

    return rows


def _measure(name, call, expected, dtype):
    row = {"case_id": None, "implementation": name, "status": "pass", "latency_ms": None}
    try:
        with torch.no_grad():
            actual = call()
        torch.musa.synchronize()
        tolerance = 1e-2 if dtype != torch.float32 else 1e-4
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument("--snapshot-id", default="REPLACE_WITH_SNAPSHOT_ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task, cases_manifest, problem_source = load_task(args.task_dir)
    expert_class = load_expert(args.expert)

    trace_dir = args.output.parent / "baseline_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases_manifest["cases"]:
        print(f"[baseline] {case['case_id']}", flush=True)
        for row in measure_case(case, task, problem_source, expert_class, trace_dir):
            row["case_id"] = case["case_id"]
            results.append(row)
            latency = f"{row['latency_ms']:.4f}" if row.get("latency_ms") else "-"
            print(f"    {row['implementation']:<28}{row['status']:<12}{latency}", flush=True)

    payload = {
        "schema_version": "1.0.0",
        "task_id": task.get("id"),
        "tier": task.get("tier"),
        "environment_snapshot": args.snapshot_id,
        "measurement_protocol": PROTOCOL,
        "implementations": sorted({row["implementation"] for row in results}),
        "results": results,
        "scoring": {
            "speedup_denominator": "expert_dispatch",
            "reference_implementation": "naive_library_composition",
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
