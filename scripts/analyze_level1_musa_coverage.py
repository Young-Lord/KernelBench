#!/usr/bin/env python3
"""Map KernelBench Level 1 problems to existing MUSA operator coverage.

Extracts the PyTorch operators used by each Level 1 reference model and checks
them against the operator list of the official torch_musa backend
(third_party/musa-sources/torch_musa/tools/ops_scanner/ops_list.md).

Output: docs/musa/level1_musa_coverage.csv

This is a static coverage analysis; it does not compile or run kernels.
A "covered" verdict means the operator is registered in torch_musa's op list,
which may be a native MUSA kernel, a muDNN/muBLAS bridge, or a combined path.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEVEL1_DIR = REPO_ROOT / "KernelBench" / "level1"
OPS_LIST = (
    REPO_ROOT
    / "third_party"
    / "musa-sources"
    / "torch_musa"
    / "tools"
    / "ops_scanner"
    / "ops_list.md"
)
OUTPUT_CSV = REPO_ROOT / "docs" / "musa" / "level1_musa_coverage.csv"

# Module / functional call -> backend operator keyword(s) to look up in ops_list.
# Keywords are checked as case-insensitive substrings against the op names.
OPS_MAP: dict[str, list[str]] = {
    # matmul family
    "torch.mm": ["mm"],
    "torch.matmul": ["matmul"],
    "torch.bmm": ["bmm"],
    "torch.einsum": ["einsum"],
    "nn.Linear": ["matmul", "addmm"],
    "torch.tensordot": ["tensordot"],
    # activation
    "nn.ReLU": ["relu"],
    "F.relu": ["relu"],
    "torch.relu": ["relu"],
    "nn.LeakyReLU": ["leaky_relu"],
    "F.leaky_relu": ["leaky_relu"],
    "nn.Sigmoid": ["sigmoid"],
    "F.sigmoid": ["sigmoid"],
    "torch.sigmoid": ["sigmoid"],
    "nn.Tanh": ["tanh"],
    "F.tanh": ["tanh"],
    "torch.tanh": ["tanh"],
    "nn.Softmax": ["softmax"],
    "F.softmax": ["softmax"],
    "torch.softmax": ["softmax"],
    "F.log_softmax": ["log_softmax"],
    "torch.log_softmax": ["log_softmax"],
    "nn.SiLU": ["silu"],
    "F.silu": ["silu"],
    "nn.GELU": ["gelu"],
    "F.gelu": ["gelu"],
    "torch.nn.functional.gelu": ["gelu"],
    "nn.SELU": ["selu"],
    "F.selu": ["selu"],
    "torch.selu": ["selu"],
    "nn.Hardsigmoid": ["hardsigmoid"],
    "F.hardsigmoid": ["hardsigmoid"],
    "nn.Softplus": ["softplus"],
    "F.softplus": ["softplus"],
    "nn.Softsign": ["softsign"],
    "F.softsign": ["softsign"],
    "nn.ELU": ["elu"],
    "F.elu": ["elu"],
    "nn.Hardtanh": ["hardtanh"],
    "F.hardtanh": ["hardtanh"],
    "torch.nn.functional.hardtanh": ["hardtanh"],
    # normalization
    "nn.BatchNorm1d": ["batch_norm"],
    "nn.BatchNorm2d": ["batch_norm"],
    "nn.BatchNorm3d": ["batch_norm"],
    "nn.InstanceNorm1d": ["instance_norm"],
    "nn.InstanceNorm2d": ["instance_norm"],
    "nn.InstanceNorm3d": ["instance_norm"],
    "nn.GroupNorm": ["group_norm"],
    "nn.LayerNorm": ["layer_norm"],
    "torch.norm": ["linalg_vector_norm", "norm"],
    "F.normalize": ["normalize"],
    "F.instance_norm": ["instance_norm"],
    "F.layer_norm": ["layer_norm"],
    "F.group_norm": ["group_norm"],
    "F.batch_norm": ["batch_norm"],
    "torch.nn.functional.normalize": ["normalize"],
    # pooling
    "nn.MaxPool1d": ["max_pool1d"],
    "nn.MaxPool2d": ["max_pool2d"],
    "nn.MaxPool3d": ["max_pool3d"],
    "nn.AvgPool1d": ["avg_pool1d"],
    "nn.AvgPool2d": ["avg_pool2d"],
    "nn.AvgPool3d": ["avg_pool3d"],
    "nn.AdaptiveAvgPool1d": ["adaptive_avg_pool1d"],
    "nn.AdaptiveAvgPool2d": ["adaptive_avg_pool2d"],
    "nn.AdaptiveAvgPool3d": ["adaptive_avg_pool3d"],
    "F.max_pool1d": ["max_pool1d"],
    "F.max_pool2d": ["max_pool2d"],
    "F.max_pool3d": ["max_pool3d"],
    "F.avg_pool1d": ["avg_pool1d"],
    "F.avg_pool2d": ["avg_pool2d"],
    "F.avg_pool3d": ["avg_pool3d"],
    # reductions
    "torch.sum": ["sum"],
    "torch.mean": ["mean"],
    "torch.max": ["max"],
    "torch.min": ["min"],
    "torch.argmax": ["argmax"],
    "torch.argmin": ["argmin"],
    # convolution. PyTorch dispatches all conv/conv_transpose variants to the
    # convolution_overrideable backend op, which torch_musa routes to muDNN.
    "nn.Conv1d": ["convolution"],
    "nn.Conv2d": ["convolution"],
    "nn.Conv3d": ["convolution"],
    "nn.ConvTranspose1d": ["convolution"],
    "nn.ConvTranspose2d": ["convolution"],
    "nn.ConvTranspose3d": ["convolution"],
    "F.conv1d": ["convolution"],
    "F.conv2d": ["convolution"],
    "F.conv3d": ["convolution"],
    "F.conv_transpose1d": ["convolution"],
    "F.conv_transpose2d": ["convolution"],
    "F.conv_transpose3d": ["convolution"],
    # scan
    "torch.cumsum": ["cumsum"],
    "torch.cumprod": ["cumprod"],
    # loss
    "nn.MSELoss": ["mse_loss"],
    "F.mse_loss": ["mse_loss"],
    "nn.CrossEntropyLoss": ["cross_entropy", "nll_loss"],
    "F.cross_entropy": ["cross_entropy", "nll_loss"],
    "nn.HuberLoss": ["huber_loss"],
    "F.huber_loss": ["huber_loss"],
    "nn.KLDivLoss": ["kl_div"],
    "F.kl_div": ["kl_div"],
    "nn.TripletMarginLoss": ["triplet_margin"],
    "F.triplet_margin_loss": ["triplet_margin"],
    "nn.HingeEmbeddingLoss": ["hinge_embedding"],
    "F.hinge_embedding_loss": ["hinge_embedding"],
    # attention
    "F.scaled_dot_product_attention": ["scaled_dot_product_attention"],
    # misc
    "torch.pow": ["pow"],
    "torch.sqrt": ["sqrt"],
    "torch.abs": ["abs"],
    "torch.clamp": ["clamp"],
    "torch.flip": ["flip"],
    "torch.cat": ["cat"],
    "torch.masked_fill": ["masked_fill"],
    "torch.where": ["where"],
    "F.pad": ["pad"],
    "F.interpolate": ["interpolate"],
    "torch.reshape": ["reshape", "view"],
    "torch.transpose": ["transpose"],
    "torch.triu": ["triu"],
    "torch.tril": ["tril"],
    "torch.diag": ["diag"],
    "torch.diag_embed": ["diag_embed"],
}


def load_ops_list(path: Path) -> set[str]:
    """Load the normalized set of registered operator names."""
    ops: set[str] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            name = stripped[2:].strip("` ")
            # Skip quantized:: entries; they are quantization-only paths and
            # would otherwise false-positive matches for conv2d etc.
            if "::" in name or "quantized" in name.lower():
                continue
            # strip overload suffix like ".out", ".dim_IntList"
            base = re.split(r"\.", name, maxsplit=1)[0]
            ops.add(base)
    return ops


def extract_calls(source: str) -> list[str]:
    """Extract distinct torch / F / nn API calls used in the model source."""
    # Normalize long-form references (e.g. torch.nn.functional.kl_div -> F.kl_div)
    # so the scan below captures the canonical short form.
    normalized = re.sub(r"torch\.nn\.functional\.", "F.", source)
    normalized = re.sub(r"torch\.nn\.", "nn.", normalized)

    found: set[str] = set()
    for match in re.finditer(r"\b(?:torch|F|nn)\.[A-Za-z_]\w*", normalized):
        found.add(match.group(0))
    # Canonicalize module names that appear in __init__ but not in forward.
    for module_name in re.findall(r"\bnn\.([A-Z]\w+)", normalized):
        found.add("nn." + module_name)
    return sorted(found)


def verdict_for_calls(calls: list[str], ops: set[str]) -> tuple[str, list[str]]:
    """Return (verdict, missing_keywords) for the problem's operator calls."""
    missing: list[str] = []
    for call in calls:
        keywords = OPS_MAP.get(call)
        if keywords is None:
            continue
        if not any(any(k in op for op in ops) for k in keywords):
            missing.append(f"{call} ({', '.join(keywords)})")
    if not missing:
        return "covered", []
    mapped_calls = [call for call in calls if call in OPS_MAP]
    if mapped_calls and len(missing) == len(mapped_calls):
        return "gap", missing
    return "partial", missing


def main() -> int:
    if not OPS_LIST.exists():
        print(f"ERROR: missing ops list: {OPS_LIST}")
        return 1
    ops = load_ops_list(OPS_LIST)
    print(f"torch_musa registered op base names: {len(ops)}")

    rows: list[dict] = []
    stats = {"covered": 0, "partial": 0, "gap": 0}
    for problem in sorted(LEVEL1_DIR.glob("*.py"), key=lambda p: int(p.stem.split("_")[0])):
        problem_id = int(problem.stem.split("_")[0])
        source = problem.read_text()
        calls = extract_calls(source)
        verdict, missing = verdict_for_calls(calls, ops)
        stats[verdict] += 1
        rows.append(
            {
                "problem_id": problem_id,
                "file": problem.name,
                "verdict": verdict,
                "calls": "; ".join(calls),
                "missing": "; ".join(missing),
            }
        )
        marker = {"covered": "[OK]", "partial": "[~]", "gap": "[GAP]"}[verdict]
        print(f"{marker} {problem_id:3d} {problem.name}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["problem_id", "file", "verdict", "calls", "missing"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {stats}")
    print(f"Wrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
