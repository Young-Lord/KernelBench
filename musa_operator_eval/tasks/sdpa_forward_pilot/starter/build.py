"""The only supported Starter build entry point."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--host-only", action="store_true")
    mode.add_argument("--musa", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=["mp_22", "mp_31"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    include = "include"
    if args.host_only:
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise SystemExit("g++ or clang++ is required for --host-only")
        output = args.output_dir / ("host_io_smoke.exe" if compiler.lower().endswith(".exe") else "host_io_smoke")
        command = [compiler, "-std=c++17", "-O2", f"-I{include}", "src/tensor_io.cpp", "src/host_io_smoke.cpp", "-o", os.path.relpath(output.resolve(), ROOT)]
    else:
        if not args.arch:
            parser.error("--musa requires --arch")
        compiler = shutil.which("mcc")
        if not compiler:
            raise SystemExit("mcc is required for --musa")
        output = args.output_dir / "runner"
        command = [compiler, "-std=c++17", "-O3", f"--cuda-gpu-arch={args.arch}", f"-I{include}", "src/tensor_io.cpp", "src/main.cpp", "src/dispatch.cpp", "src/launch.mu", "src/fallback_kernel.mu", "-lmusa", "-lmudnn", "-o", os.path.relpath(output.resolve(), ROOT)]
    print(" ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
