"""mingpt_causal_attention_b_v0: the build entry point §4.7 runs before any case.

§4.6 gives a starter one build path, and this is it: the evaluator compiles the
submission by running this file, so a submission that builds here builds in the
evaluator. Everything the submission compiles at import time lands in
`--build-dir`, which is the directory the link whitelist check reads.

    python starter/build.py --submission model_new.py --build-dir build/

Exit code is 0 when the submission built, 1 when it did not, and 3 when the device
stack it needs is not installed at all. The last two are different facts: one is a
verdict about the submission and the other is about the machine, and the evaluator
reports them under different exit codes. The compiler's own output is passed
through rather than summarised, because a diagnosis is the point of running a
build separately from an evaluation.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path


def build(submission_path: Path, build_dir: Path) -> int:
    """Execute the submission's module body with its build output redirected.

    The body is what compiles: an A-tier answer calls `load_inline` at module
    level, and a B-tier fallback does the same when it is reached. Executing it in
    a namespace of its own mirrors how the evaluator loads it, so the two agree
    about what a build is.
    """
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build_dir)

    source = Path(submission_path).read_text(encoding="utf-8")
    namespace = {"__name__": "submission_build", "__file__": str(submission_path)}
    exec(compile(source, str(submission_path), "exec"), namespace)

    produced = sorted(
        str(path) for path in build_dir.rglob("*")
        if path.is_file() and (path.suffix in {".so", ".o"} or ".so." in path.name)
    )
    for artifact in produced:
        print("[build] " + artifact)
    if not produced:
        print("[build] nothing was compiled at import time")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return build(args.submission, args.build_dir)
    except ModuleNotFoundError as error:
        # A missing torch is a machine fact, not a verdict on the submission.
        print(f"[build] the device stack is not installed: {error}")
        return 3
    except Exception:  # noqa: BLE001 -- a failed build is a result, not a crash
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
