"""Generate deterministic raw-binary SDPA inputs and public golden outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "tasks" / "sdpa_forward_pilot" / "reference" / "sdpa_reference.py"
SPEC = importlib.util.spec_from_file_location("sdpa_reference", REFERENCE_PATH)
REFERENCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFERENCE)


def float32_to_bfloat16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype("<u2")


def bfloat16_to_float32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def quantize(values: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    if dtype == "float16":
        stored = np.asarray(values, dtype="<f2")
        return stored, stored.astype(np.float32)
    if dtype == "float32":
        stored = np.asarray(values, dtype="<f4")
        return stored, stored
    if dtype == "bfloat16":
        stored = float32_to_bfloat16(values)
        return stored, bfloat16_to_float32(stored)
    raise ValueError(f"unsupported dtype {dtype}")


def tensor_record(name: str, filename: str, dtype: str, shape: tuple[int, ...], data: bytes) -> dict:
    return {"name": name, "file": filename, "dtype": dtype, "shape": list(shape), "layout": "BHSD", "byte_order": "little", "nbytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_tensor(directory: Path, name: str, values: np.ndarray, dtype: str) -> tuple[dict, np.ndarray]:
    stored, actual = quantize(values, dtype)
    data = stored.tobytes(order="C")
    filename = f"{name}.bin"
    (directory / filename).write_bytes(data)
    return tensor_record(name, filename, dtype, values.shape, data), actual.reshape(values.shape)


def random_values(rng: np.random.Generator, shape: tuple[int, ...], distribution: str) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(shape, dtype=np.float32)
    if distribution == "uniform":
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if distribution == "adversarial_extremes":
        values = rng.standard_normal(shape, dtype=np.float32)
        flat = values.reshape(-1)
        sentinels = np.asarray([0.0, -0.0, 8.0, -8.0, 1e-3, -1e-3], dtype=np.float32)
        flat[: min(len(flat), len(sentinels))] = sentinels[: min(len(flat), len(sentinels))]
        return values
    raise ValueError(f"unsupported distribution {distribution}")


def generate_case(case: dict, output_root: Path) -> None:
    shape = case["shape"]
    rng = np.random.Generator(np.random.PCG64(case["seed"]))
    q_shape = (shape["B"], shape["H_q"], shape["S_q"], shape["D"])
    kv_shape = (shape["B"], shape["H_kv"], shape["S_kv"], shape["D"])
    case_root = output_root / case["case_id"]
    input_dir, golden_dir = case_root / "input", case_root / "golden"
    input_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    records, actual = [], {}
    for name, tensor_shape in (("q", q_shape), ("k", kv_shape), ("v", kv_shape)):
        values = random_values(rng, tensor_shape, case["distribution"])
        record, quantized = write_tensor(input_dir, name, values, case["dtype"])
        records.append(record)
        actual[name] = quantized
    input_manifest = {"schema_version": "1.0.0", "case_id": case["case_id"], "tensors": records}
    (input_dir / "tensors.json").write_text(json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")
    attrs = case["attributes"]
    output, lse = REFERENCE.sdpa_forward(actual["q"], actual["k"], actual["v"], scale=attrs["scale"], causal=attrs["causal"], window_left=attrs["window_left"], window_right=attrs["window_right"])
    output_record, _ = write_tensor(golden_dir, "output", output, case["dtype"])
    lse_record, _ = write_tensor(golden_dir, "lse", lse, "float32")
    golden_manifest = {"schema_version": "1.0.0", "case_id": case["case_id"], "reference": "cpu_fp64_blocked", "tensors": [output_record, lse_record]}
    (golden_dir / "tensors.json").write_text(json.dumps(golden_manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "tasks" / "sdpa_forward_pilot" / "public_cases.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tasks" / "sdpa_forward_pilot" / "generated")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("visibility") != "public":
        raise ValueError("this public generator refuses private manifests")
    for case in manifest["cases"]:
        generate_case(case, args.output_dir)
        print(f"generated {case['case_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
