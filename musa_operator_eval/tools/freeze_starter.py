"""Create or verify hashes for frozen Starter files and edit markers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "tasks" / "sdpa_forward_pilot" / "starter"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock() -> dict:
    manifest = json.loads((STARTER / "starter_manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for relative in manifest["frozen_files"]:
        path = STARTER / relative
        if not path.is_file():
            raise ValueError(f"missing frozen file: {relative}")
        hashes[relative] = digest(path)
    for item in manifest["editable_files"]:
        text = (STARTER / item["path"]).read_text(encoding="utf-8")
        for region in item["regions"]:
            if text.count(f"BEGIN_AGENT_EDIT:{region}") != 1 or text.count(f"END_AGENT_EDIT:{region}") != 1:
                raise ValueError(f"invalid edit markers for {item['path']}:{region}")
    return {"schema_version": "1.0.0", "task_id": manifest["task_id"], "frozen_sha256": hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    current = build_lock()
    lock_path = STARTER / "starter_lock.json"
    if args.write:
        lock_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(lock_path)
        return 0
    if not lock_path.exists():
        print("starter lock is missing")
        return 1
    expected = json.loads(lock_path.read_text(encoding="utf-8"))
    if current != expected:
        print("starter frozen files differ from starter_lock.json")
        return 1
    print("starter lock verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
