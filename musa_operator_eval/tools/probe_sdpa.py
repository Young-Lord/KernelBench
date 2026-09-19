"""Run the public SDPA capability plan on a MUSA host.

This probe deliberately reports the high-level torch_musa SDPA route only.
Direct muDNN/MATE probes are recorded as not_probed until their task-specific
native adapters are supplied; this prevents a successful PyTorch call from
being mislabeled as a fused-library success.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def unavailable_results(plan: dict, reason: str) -> dict:
    return {
        "schema_version": "1.0.0", "family": plan["family"], "status": "unavailable", "reason": reason,
        "routes": {route: "not_probed" for route in plan["required_routes"]}, "results": []
    }


def make_mask(torch, case, device):
    left, right = case["window_left"], case["window_right"]
    if left == -1 and right == -1:
        return None
    q = torch.arange(case["S_q"], device=device).view(-1, 1)
    k = torch.arange(case["S_kv"], device=device).view(1, -1)
    mask = torch.ones((case["S_q"], case["S_kv"]), dtype=torch.bool, device=device)
    if left != -1:
        mask &= k >= q - left
    if right != -1:
        mask &= k <= q + right
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parents[1] / "tasks" / "sdpa_forward_pilot" / "capability_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=10)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        import torch
        import torch.nn.functional as F
        import torch_musa  # noqa: F401
    except Exception as exc:
        result = unavailable_results(plan, f"MUSA Python stack unavailable: {exc}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["reason"])
        return 2

    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        result = unavailable_results(plan, "torch.musa is unavailable")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["reason"])
        return 2

    results = []
    device = torch.device("musa")
    for case in plan["cases"]:
        record = {"probe_id": case["probe_id"], "route": "torch_musa_sdpa", "accepted": False, "correct": None, "selected_route": "unverified", "latency_ms": None, "peak_memory_bytes": None, "error": None, "evidence": []}
        try:
            dtype = getattr(torch, case["dtype"])
            q_shape = (case["B"], case["H_q"], case["S_q"], case["D"])
            kv_shape = (case["B"], case["H_kv"], case["S_kv"], case["D"])
            q = torch.randn(q_shape, device=device, dtype=dtype)
            k = torch.randn(kv_shape, device=device, dtype=dtype)
            v = torch.randn(kv_shape, device=device, dtype=dtype)
            if case["layout"] == "BSHD":
                q = q.transpose(1, 2).contiguous().transpose(1, 2)
                k = k.transpose(1, 2).contiguous().transpose(1, 2)
                v = v.transpose(1, 2).contiguous().transpose(1, 2)
            mask = make_mask(torch, case, device)
            kwargs = {"attn_mask": mask, "is_causal": case["causal"]}
            if case["H_q"] != case["H_kv"]:
                kwargs["enable_gqa"] = True
            for _ in range(args.warmup):
                F.scaled_dot_product_attention(q, k, v, **kwargs)
            musa.synchronize()
            elapsed = []
            for _ in range(args.measurements):
                start = time.perf_counter()
                output = F.scaled_dot_product_attention(q, k, v, **kwargs)
                musa.synchronize()
                elapsed.append((time.perf_counter() - start) * 1000)
            record.update(accepted=True, latency_ms=sorted(elapsed)[len(elapsed) // 2], evidence=[f"output_shape={list(output.shape)}"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
    payload = {"schema_version": "1.0.0", "family": plan["family"], "status": "complete", "routes": {"torch_musa_sdpa": "probed", "mudnn_fused": "not_probed", "mate_fmha": "not_probed"}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
