#!/usr/bin/env python3
"""Profile one KernelBench reference/DeepSeek pair with torch_musa Kineto.

This is the fallback profiler for the current legacy S4000 environment, where
Moore Perf System/Compute are not installed.  It records one warmed-up forward
pass for each implementation and exports both a compact JSON summary and Chrome
trace files.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch_musa  # noqa: F401 - installs the MUSA profiler integration


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def musa_inputs(reference_module) -> list[torch.Tensor]:
    cpu_inputs = reference_module.get_inputs()
    result = [value.to("musa") if isinstance(value, torch.Tensor) else value for value in cpu_inputs]
    del cpu_inputs
    gc.collect()
    return result


def musa_events(profiler, limit: int = 100) -> list[dict[str, Any]]:
    rows = []
    for event in profiler.key_averages():
        device_us = float(getattr(event, "self_musa_time_total", 0.0))
        if device_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_musa_time_us": device_us,
                "total_musa_time_us": float(getattr(event, "musa_time_total", device_us)),
                "self_cpu_time_us": float(event.self_cpu_time_total),
            }
        )
    rows.sort(key=lambda row: row["self_musa_time_us"], reverse=True)
    return rows[:limit]


def is_device_kernel(row: dict[str, Any]) -> bool:
    """Exclude high-level ATen aggregates that duplicate child kernel time."""
    name = row["name"]
    return not name.startswith("aten::") and not name.endswith("_forward")


def profile_forward(label: str, model, inputs, trace_path: Path) -> dict[str, Any]:
    torch.musa.synchronize()
    with torch.inference_mode(), torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.MUSA,
        ]
    ) as profiler:
        with torch.profiler.record_function(label):
            output = model(*inputs)
            torch.musa.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    events = musa_events(profiler)
    kernels = [row for row in events if is_device_kernel(row)]
    del output
    torch.musa.empty_cache()
    return {
        "trace": str(trace_path.relative_to(ROOT)),
        "device_kernel_count": sum(row["calls"] for row in kernels),
        "device_kernel_time_us": sum(row["self_musa_time_us"] for row in kernels),
        "device_kernels": kernels,
        "top_musa_events_including_aten_aggregates": events[:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_id", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis" / "pytorch_vs_deepseek_musa" / "profiles",
    )
    args = parser.parse_args()

    reference_matches = sorted((ROOT / "KernelBench" / "level1").glob(f"{args.problem_id}_*.py"))
    if len(reference_matches) != 1:
        raise RuntimeError(f"Expected one reference for problem {args.problem_id}: {reference_matches}")
    candidate_path = ROOT / "KernelBench" / "level1_musa" / f"problem_{args.problem_id}" / "model_new.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_module = load_module(f"kb_reference_{args.problem_id}", reference_matches[0])
    candidate_module = load_module(f"kb_candidate_{args.problem_id}", candidate_path)
    init_inputs = reference_module.get_init_inputs()
    reference_model = reference_module.Model(*init_inputs).eval().to("musa")
    candidate_model = candidate_module.ModelNew(*init_inputs).eval().to("musa")
    candidate_model.load_state_dict(reference_model.state_dict(), strict=False)
    inputs = musa_inputs(reference_module)

    with torch.inference_mode():
        for _ in range(args.warmup):
            reference_output = reference_model(*inputs)
            torch.musa.synchronize()
            del reference_output
        for _ in range(args.warmup):
            candidate_output = candidate_model(*inputs)
            torch.musa.synchronize()
            del candidate_output

    prefix = f"problem_{args.problem_id}"
    result = {
        "problem_id": args.problem_id,
        "reference": profile_forward(
            "pytorch_reference_forward",
            reference_model,
            inputs,
            args.output_dir / f"{prefix}_pytorch_trace.json",
        ),
        "deepseek": profile_forward(
            "deepseek_kernel_forward",
            candidate_model,
            inputs,
            args.output_dir / f"{prefix}_deepseek_trace.json",
        ),
    }
    output_path = args.output_dir / f"{prefix}_profile_summary.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
