"""Raw tensor decoding and numeric comparison used by the evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def decode(path: Path, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "float16":
        values = np.fromfile(path, dtype="<f2").astype(np.float32)
    elif dtype == "float32":
        values = np.fromfile(path, dtype="<f4")
    elif dtype == "bfloat16":
        words = np.fromfile(path, dtype="<u2").astype(np.uint32)
        values = (words << np.uint32(16)).view(np.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"element count mismatch: expected {expected}, got {values.size}")
    return values.reshape(shape)


def load_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "tensors.json").read_text(encoding="utf-8"))
    for tensor in manifest["tensors"]:
        path = directory / tensor["file"]
        data = path.read_bytes()
        if len(data) != tensor["nbytes"]:
            raise ValueError(f"nbytes mismatch for {tensor['name']}")
        if hashlib.sha256(data).hexdigest() != tensor["sha256"]:
            raise ValueError(f"sha256 mismatch for {tensor['name']}")
    return manifest


def compare_directories(actual_dir: Path, golden_dir: Path, *, atol: float, rtol: float) -> dict:
    actual = load_manifest(actual_dir)
    golden = load_manifest(golden_dir)
    actual_by_name = {item["name"]: item for item in actual["tensors"]}
    results = {}
    passed = True
    for expected in golden["tensors"]:
        name = expected["name"]
        if name not in actual_by_name:
            results[name] = {"passed": False, "error": "missing output"}
            passed = False
            continue
        observed = actual_by_name[name]
        if observed["dtype"] != expected["dtype"] or observed["shape"] != expected["shape"]:
            results[name] = {"passed": False, "error": "dtype or shape mismatch"}
            passed = False
            continue
        left = decode(actual_dir / observed["file"], observed["dtype"], observed["shape"])
        right = decode(golden_dir / expected["file"], expected["dtype"], expected["shape"])
        finite = np.isfinite(left) & np.isfinite(right)
        nonfinite_equal = np.array_equal(np.isneginf(left), np.isneginf(right)) and np.array_equal(np.isposinf(left), np.isposinf(right)) and np.array_equal(np.isnan(left), np.isnan(right))
        max_abs = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        tensor_passed = nonfinite_equal and bool(np.allclose(left[finite], right[finite], atol=atol, rtol=rtol))
        results[name] = {"passed": tensor_passed, "max_abs_error": max_abs}
        passed &= tensor_passed
    return {"passed": passed, "tensors": results}
