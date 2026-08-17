#!/usr/bin/env python3
"""Evaluate the submitted Level 1 MUSA kernels as a reproducible API baseline.

Each problem runs in a fresh subprocess so a compiler crash, timeout, or broken
GPU context cannot prevent later problems from being evaluated. Results are
checkpointed after every problem and can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
LEVEL1_DIR = REPO_ROOT / "KernelBench" / "level1"
KERNEL_DIR = REPO_ROOT / "KernelBench" / "level1_musa"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "deepseek_api_level1_musa_s4000"


def problem_file(problem_id: int) -> Path:
    matches = sorted(LEVEL1_DIR.glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one reference for problem {problem_id}, found {len(matches)}"
        )
    return matches[0]


def kernel_file(problem_id: int) -> Path:
    path = KERNEL_DIR / f"problem_{problem_id}" / "model_new.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated kernel: {path}")
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def evaluate_child(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SRC_DIR))
    import torch
    import torch_musa  # noqa: F401 - activates torch.musa

    from kernelbench.eval import eval_kernel_against_ref
    from kernelbench.gpu import resolve_device
    from kernelbench.timing import measure_ref_program_time
    from kernelbench.utils import set_gpu_arch

    problem_id = args.child
    ref_path = problem_file(problem_id)
    generated_path = kernel_file(problem_id)
    result_path = args.output_dir / "problems" / f"problem_{problem_id:03d}.json"
    started = time.time()

    set_gpu_arch(["mp_22"])
    device = resolve_device(0)
    ref_source = ref_path.read_text(encoding="utf-8")
    generated_source = generated_path.read_text(encoding="utf-8")

    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_source,
            custom_model_src=generated_source,
            seed_num=args.seed,
            num_correct_trials=args.num_correct,
            num_perf_trials=args.num_perf,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=args.verbose,
            build_dir=args.output_dir / "build" / f"problem_{problem_id:03d}",
            device=device,
            backend="musa",
            precision=torch.float32,
            check_for_excessive_speedup=True,
            excessive_speedup_threshold=args.excessive_speedup_threshold,
        )
        if result is None:
            raise RuntimeError("Evaluator returned no result (transient build/cache error)")

        ref_runtime = result.ref_runtime
        ref_runtime_stats = result.ref_runtime_stats
        if not ref_runtime or ref_runtime <= 0:
            # The evaluator times the reference only for correct kernels. Keep a
            # complete eager baseline even for compile/correctness failures.
            measured = measure_ref_program_time(
                ref_arch_name=ref_path.name,
                ref_arch_src=ref_source,
                num_trials=args.num_perf,
                timing_method="cuda_event",
                device=device,
                precision=torch.float32,
            )
            if measured:
                ref_runtime = measured["mean"]
                ref_runtime_stats = measured

        runtime = result.runtime
        speedup = (
            ref_runtime / runtime
            if result.correctness and runtime and runtime > 0 and ref_runtime > 0
            else None
        )
        row = {
            "problem_id": problem_id,
            "problem_name": ref_path.name,
            "reference_path": str(ref_path.relative_to(REPO_ROOT)),
            "kernel_path": str(generated_path.relative_to(REPO_ROOT)),
            "sample_id": 0,
            "compiled": result.compiled,
            "correctness": result.correctness,
            "metadata": result.metadata,
            "runtime_ms": runtime,
            "runtime_stats": result.runtime_stats,
            "reference_runtime_ms": ref_runtime,
            "reference_runtime_stats": ref_runtime_stats,
            "speedup_over_eager": speedup,
            "wall_time_seconds": round(time.time() - started, 3),
            "status": "ok",
        }
    except BaseException as exc:
        row = {
            "problem_id": problem_id,
            "problem_name": ref_path.name,
            "reference_path": str(ref_path.relative_to(REPO_ROOT)),
            "kernel_path": str(generated_path.relative_to(REPO_ROOT)),
            "sample_id": 0,
            "compiled": False,
            "correctness": False,
            "metadata": {"runner_error": str(exc), "runner_error_type": type(exc).__name__},
            "runtime_ms": -1.0,
            "runtime_stats": {},
            "reference_runtime_ms": -1.0,
            "reference_runtime_stats": {},
            "speedup_over_eager": None,
            "wall_time_seconds": round(time.time() - started, 3),
            "status": "error",
        }

    atomic_json_dump(result_path, row)
    print(
        f"RESULT problem={problem_id} compiled={row['compiled']} "
        f"correct={row['correctness']} kernel_ms={row['runtime_ms']} "
        f"ref_ms={row['reference_runtime_ms']} speedup={row['speedup_over_eager']}"
    )
    return 0 if row["status"] == "ok" else 1


def load_rows(output_dir: Path, problem_ids: list[int]) -> list[dict[str, Any]]:
    rows = []
    for problem_id in problem_ids:
        path = output_dir / "problems" / f"problem_{problem_id:03d}.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                rows.append(json.load(handle))
    return rows


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize(output_dir: Path, problem_ids: list[int], args: argparse.Namespace) -> dict:
    rows = load_rows(output_dir, problem_ids)
    compiled = sum(bool(row["compiled"]) for row in rows)
    correct = sum(bool(row["correctness"]) for row in rows)
    compiled_problem_ids = [row["problem_id"] for row in rows if row["compiled"]]
    correct_problem_ids = [row["problem_id"] for row in rows if row["correctness"]]
    timed_correct_problem_ids = [
        row["problem_id"]
        for row in rows
        if row["correctness"] and row.get("speedup_over_eager") is not None
    ]
    correct_without_runtime_problem_ids = sorted(
        set(correct_problem_ids) - set(timed_correct_problem_ids)
    )
    speedups = [
        float(row["speedup_over_eager"])
        for row in rows
        if row["correctness"] and row.get("speedup_over_eager") is not None
    ]
    total = len(problem_ids)
    thresholds = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
    fast_p = {
        str(threshold): sum(
            bool(row["correctness"])
            and row.get("speedup_over_eager") is not None
            and row["speedup_over_eager"] > threshold
            for row in rows
        )
        / total
        for threshold in thresholds
    }
    summary = {
        "provenance": {
            "generator": "DeepSeek API",
            "provenance_basis": "User-confirmed external API generation before Git submission",
            "generation_parameters": "not available in repository",
            "sample_count_per_problem": 1,
        },
        "environment": {
            "hardware": "MTT S4000",
            "gpu_count": 1,
            "musa_arch": "mp_22",
            "backend": "musa",
            "precision": "fp32",
            "timing_method": "cuda_event (cold-cache)",
            "num_correct_trials": args.num_correct,
            "num_perf_trials": args.num_perf,
            "seed": args.seed,
        },
        "metrics": {
            "expected_problems": total,
            "evaluated_problems": len(rows),
            "compiled": compiled,
            "correct": correct,
            "compilation_rate": compiled / total,
            "pass_at_1": correct / total,
            "timed_correct": len(timed_correct_problem_ids),
            "performance_coverage": len(timed_correct_problem_ids) / total,
            "performance_coverage_among_correct": (
                len(timed_correct_problem_ids) / correct if correct else 0.0
            ),
            "geometric_mean_speedup_correct": geometric_mean(speedups),
            "median_speedup_correct": statistics.median(speedups) if speedups else None,
            "fast_p": fast_p,
        },
        "problem_sets": {
            "compile_failed_problem_ids": sorted(
                set(problem_ids) - set(compiled_problem_ids)
            ),
            "incorrect_problem_ids": sorted(
                set(problem_ids) - set(correct_problem_ids)
            ),
            "correct_without_runtime_problem_ids": correct_without_runtime_problem_ids,
            "timed_correct_problem_ids": timed_correct_problem_ids,
        },
        "known_issue": {
            "problem_id": 72,
            "description": (
                "Repository documentation reports a torch_musa/muDNN grouped "
                "ConvTranspose3d reference bug; the generated kernel was bit-exact "
                "against CPU but cannot match the faulty MUSA reference."
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    atomic_json_dump(output_dir / "summary.json", summary)
    atomic_json_dump(output_dir / "results.json", rows)

    eval_results = {
        str(row["problem_id"]): [
            {
                "sample_id": 0,
                "compiled": row["compiled"],
                "correctness": row["correctness"],
                "metadata": row["metadata"],
                "runtime": row["runtime_ms"],
                "runtime_stats": row["runtime_stats"],
            }
        ]
        for row in rows
    }
    atomic_json_dump(output_dir / "eval_results.json", eval_results)

    baseline = {
        "level1": {
            row["problem_name"]: row["reference_runtime_stats"] for row in rows
        }
    }
    atomic_json_dump(output_dir / "baseline_time_torch.json", baseline)

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "problem_id", "problem_name", "compiled", "correctness", "runtime_ms",
            "reference_runtime_ms", "speedup_over_eager", "wall_time_seconds", "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metrics = summary["metrics"]
    report = [
        "# DeepSeek API — KernelBench Level 1 MUSA baseline",
        "",
        "- Hardware: MTT S4000 single GPU (`mp_22`)",
        "- Backend / precision: MUSA / FP32",
        f"- Correctness trials: {args.num_correct}; performance trials: {args.num_perf}",
        "- Timing: GPU events with L2 cache thrashing (milliseconds)",
        "- Provenance: user-confirmed DeepSeek API outputs committed as `level1_musa`",
        "- Original API model/version and decoding parameters: unavailable in repository",
        "",
        "## Summary",
        "",
        f"- Evaluated: {metrics['evaluated_problems']} / {metrics['expected_problems']}",
        f"- Compiled: {metrics['compiled']} ({metrics['compilation_rate']:.1%})",
        f"- Correct / pass@1: {metrics['correct']} ({metrics['pass_at_1']:.1%})",
        (
            f"- Correct with valid performance timing: {metrics['timed_correct']} "
            f"({metrics['performance_coverage']:.1%} of all problems; "
            f"{metrics['performance_coverage_among_correct']:.1%} of correct kernels)"
        ),
        f"- Geometric-mean speedup over eager (correct only): {metrics['geometric_mean_speedup_correct']}",
        f"- Median speedup over eager (correct only): {metrics['median_speedup_correct']}",
        f"- Faster than eager: {metrics['fast_p']['1.0']:.1%} of all problems",
        "",
        "## Exceptions and interpretation",
        "",
        (
            "- Problem 72 is reported as incorrect because the MUSA PyTorch grouped "
            "ConvTranspose3d reference is known to be faulty; repository documentation "
            "reports the generated kernel as bit-exact against CPU."
        ),
        (
            "- Problems 63, 76, and 87 pass correctness, but their generated ModelNew "
            "implementations mutate the reusable input tensor storage. They therefore "
            "have no valid repeated-call performance timing and are excluded from "
            "speedup aggregates."
        ),
        "- No generated kernel source was modified while producing this baseline.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def run_parent(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "command": " ".join(sys.argv),
        "problem_ids": args.problem_ids,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, check=True,
        ).stdout.strip(),
    }
    atomic_json_dump(args.output_dir / "run_config.json", metadata)

    for index, problem_id in enumerate(args.problem_ids, start=1):
        result_path = args.output_dir / "problems" / f"problem_{problem_id:03d}.json"
        if result_path.exists() and not args.rerun:
            print(f"[{index}/{len(args.problem_ids)}] SKIP problem {problem_id} (checkpoint exists)")
            continue

        print(f"[{index}/{len(args.problem_ids)}] RUN problem {problem_id}", flush=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--child", str(problem_id),
            "--output-dir", str(args.output_dir), "--num-correct", str(args.num_correct),
            "--num-perf", str(args.num_perf), "--seed", str(args.seed),
            "--excessive-speedup-threshold", str(args.excessive_speedup_threshold),
        ]
        if args.verbose:
            command.append("--verbose")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        log_path = logs_dir / f"problem_{problem_id:03d}.log"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    command, cwd=REPO_ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=args.timeout,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                return_code = 124

        if not result_path.exists():
            atomic_json_dump(
                result_path,
                {
                    "problem_id": problem_id,
                    "problem_name": problem_file(problem_id).name,
                    "reference_path": str(problem_file(problem_id).relative_to(REPO_ROOT)),
                    "kernel_path": str(kernel_file(problem_id).relative_to(REPO_ROOT)),
                    "sample_id": 0,
                    "compiled": False,
                    "correctness": False,
                    "metadata": {
                        "runner_error": "Evaluation timed out" if return_code == 124 else "Child process failed",
                        "child_return_code": return_code,
                        "log_path": str(log_path.relative_to(REPO_ROOT)),
                    },
                    "runtime_ms": -1.0,
                    "runtime_stats": {},
                    "reference_runtime_ms": -1.0,
                    "reference_runtime_stats": {},
                    "speedup_over_eager": None,
                    "wall_time_seconds": args.timeout if return_code == 124 else None,
                    "status": "timeout" if return_code == 124 else "error",
                },
            )
        summary = summarize(args.output_dir, args.problem_ids, args)
        metrics = summary["metrics"]
        print(
            f"[{index}/{len(args.problem_ids)}] DONE problem {problem_id}; "
            f"compiled={metrics['compiled']} correct={metrics['correct']}",
            flush=True,
        )

    summary = summarize(args.output_dir, args.problem_ids, args)
    print(json.dumps(summary["metrics"], indent=2))
    return 0 if summary["metrics"]["evaluated_problems"] == len(args.problem_ids) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-ids", type=int, nargs="+", default=list(range(1, 101)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-correct", type=int, default=5)
    parser.add_argument("--num-perf", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--excessive-speedup-threshold", type=float, default=10.0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.child is not None:
        return evaluate_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
