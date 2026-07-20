#!/usr/bin/env python3
"""Verify hand-written MUSA kernels for KernelBench Level 1 gap problems."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kernelbench.eval import eval_kernel_against_ref
from kernelbench.utils import set_gpu_arch

LEVEL1_DIR = REPO_ROOT / "KernelBench" / "level1"
MUSA_DIR = REPO_ROOT / "KernelBench" / "level1_musa"

# Gap problems identified earlier (no confirmed native MUSA op in ops_list).
DEFAULT_PROBLEM_IDS = [11, 12, 27, 30, 34, 41, 44, 95, 98, 99, 100]


def _problem_file(problem_id: int) -> Path:
    matches = sorted(LEVEL1_DIR.glob(f"{problem_id}_*.py"))
    if not matches:
        raise FileNotFoundError(f"No Level 1 reference for problem {problem_id}")
    return matches[0]


def _kernel_file(problem_id: int) -> Path:
    d = MUSA_DIR / f"problem_{problem_id}"
    path = d / "model_new.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing kernel: {path}")
    return path


def verify_one(problem_id: int, verbose: bool = False) -> dict:
    ref_path = _problem_file(problem_id)
    kernel_path = _kernel_file(problem_id)

    result = eval_kernel_against_ref(
        original_model_src=ref_path.read_text(),
        custom_model_src=kernel_path.read_text(),
        backend="musa",
        precision=torch.float32,
        timing_method="musa_event",
        verbose=verbose,
        measure_performance=False,
    )

    return {
        "problem_id": problem_id,
        "ref": ref_path.name,
        "kernel": str(kernel_path.relative_to(REPO_ROOT)),
        "compiled": result.compiled,
        "correctness": result.correctness,
        "metadata": result.metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-ids",
        type=int,
        nargs="+",
        default=DEFAULT_PROBLEM_IDS,
        help="Level 1 problem ids to verify",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.musa.is_available():
        print("ERROR: MUSA GPU not available")
        return 1

    set_gpu_arch(["mp_22"])

    passed = 0
    failed = 0
    rows = []

    for pid in args.problem_ids:
        try:
            if len(args.problem_ids) > 1:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--problem-ids",
                    str(pid),
                ]
                if args.verbose:
                    cmd.append("--verbose")
                env = os.environ.copy()
                env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
                result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
                ok = result.returncode == 0
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed += 1
                else:
                    failed += 1
                rows.append((status, pid, {"subprocess_exit_code": result.returncode}))
                print(f"[{status}] problem {pid}: subprocess_exit_code={result.returncode}")
                continue

            row = verify_one(pid, verbose=args.verbose)
            ok = row["compiled"] and row["correctness"]
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            rows.append((status, pid, row))
            print(f"[{status}] problem {pid}: compiled={row['compiled']} correctness={row['correctness']}")
            if not ok and row.get("metadata"):
                print(f"       metadata: {row['metadata']}")
        except Exception as exc:
            failed += 1
            rows.append(("ERROR", pid, {"error": str(exc)}))
            print(f"[ERROR] problem {pid}: {exc}")
            if args.verbose:
                traceback.print_exc()
        finally:
            gc.collect()
            if torch.musa.is_available():
                torch.musa.empty_cache()

    print(f"\nSummary: {passed} passed, {failed} failed / {len(args.problem_ids)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
