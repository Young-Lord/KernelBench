"""Collect a reproducible MUSA environment snapshot without fabricating fields."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "unavailable", "command": command, "output": None}
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=30)
    output = (completed.stdout + completed.stderr).strip()
    return {"status": "ok" if completed.returncode == 0 else "error", "command": command, "exit_code": completed.returncode, "output": output}


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def redact(snapshot: dict[str, object]) -> dict[str, object]:
    public = json.loads(json.dumps(snapshot))
    public["visibility"] = "redacted"
    public.pop("host", None)
    public.pop("raw_probes", None)
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["mp_22", "mp_31"])
    parser.add_argument("--device-name")
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--driver")
    parser.add_argument("--toolkit")
    parser.add_argument("--mudnn")
    parser.add_argument("--mublas")
    parser.add_argument("--image-digest", default=os.environ.get("MUSA_EVAL_IMAGE_DIGEST"))
    parser.add_argument("--build-flag", action="append", default=[])
    parser.add_argument("--fast-math", action="store_true")
    args = parser.parse_args()

    captured_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    probes = {
        "musa_smi": run(["musa-smi"]),
        "mcc": run(["mcc", "--version"]),
        "driver": run(["musa-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }
    missing = [
        name
        for name, value in {
            "architecture": args.architecture,
            "device_name": args.device_name,
            "driver": args.driver,
            "musa_toolkit": args.toolkit,
            "mudnn": args.mudnn,
            "mublas": args.mublas,
            "image_digest": args.image_digest,
        }.items()
        if not value
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        probe = {
            "status": "incomplete",
            "captured_at": captured_at,
            "missing_required_fields": missing,
            "host": {"node": platform.node(), "platform": platform.platform(), "python": sys.version},
            "packages": {"torch": package_version("torch"), "torch_musa": package_version("torch_musa")},
            "raw_probes": probes,
        }
        path = args.output_dir / "environment_probe.json"
        path.write_text(json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"incomplete environment probe: {path}")
        print("missing: " + ", ".join(missing))
        return 2

    fingerprint_source = "|".join(
        str(value)
        for value in (
            args.device_name, args.architecture, args.driver, args.toolkit,
            package_version("torch"), package_version("torch_musa"), args.image_digest,
        )
    )
    snapshot_id = "musa-" + hashlib.sha256(fingerprint_source.encode()).hexdigest()[:16]
    snapshot = {
        "schema_version": "1.0.0",
        "snapshot_id": snapshot_id,
        "visibility": "full",
        "captured_at": captured_at,
        "hardware": {"device_name": args.device_name, "architecture": args.architecture, "device_count": args.device_count},
        "software": {
            "driver": args.driver, "musa_toolkit": args.toolkit,
            "torch": package_version("torch") or "unavailable",
            "torch_musa": package_version("torch_musa") or "unavailable",
            "mudnn": args.mudnn, "mublas": args.mublas,
        },
        "compiler": {"mcc_version": str(probes["mcc"].get("output") or "unavailable"), "target_arch": args.architecture},
        "build": {"flags": args.build_flag, "fast_math": args.fast_math},
        "container": {"image_digest": args.image_digest},
        "host": {"node": platform.node(), "platform": platform.platform(), "python": sys.version},
        "raw_probes": probes,
    }
    full_path = args.output_dir / "environment.full.json"
    public_path = args.output_dir / "environment.public.json"
    full_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    public_path.write_text(json.dumps(redact(snapshot), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(full_path)
    print(public_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
