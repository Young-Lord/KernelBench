"""Run the public SDPA capability plan on a MUSA host.

The route is resolved from the aten op the dispatcher actually runs, not from a
successful high-level call. A PyTorch call that returns a tensor proves nothing
about which kernel produced it, so every case is profiled and the observed op is
mapped onto the dispatch contract. When profiling is unavailable the route stays
`unverified` rather than being guessed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Maps the op the dispatcher ran onto the dispatch contract's ordered paths.
FUSED_LIBRARY_OPS = {
    "_scaled_dot_product_attention_flash_musa",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_flash_attention_forward",
    "_efficient_attention_forward",
}
LIBRARY_COMPOSITION_OPS = {
    "_scaled_dot_product_attention_math_musa",
    "_scaled_dot_product_attention_math",
}
COMPOSITION_ATEN_OPS = {"matmul", "bmm", "softmax", "mul", "div", "masked_fill"}


def unavailable_results(plan: dict, reason: str) -> dict:
    return {
        "schema_version": "1.0.0", "family": plan["family"], "status": "unavailable", "reason": reason,
        "routes": {route: "not_probed" for route in plan.get("required_routes", [])}, "results": []
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


def build_tensors(torch, case, device, dtype):
    """Build the case tensors, expanding key/value heads when the plan asks for GQA.

    torch 2.2 has no `enable_gqa` argument, so grouped-query attention has to be
    expressed as an explicit head expansion on the key/value side.
    """
    q = torch.randn((case["B"], case["H_q"], case["S_q"], case["D"]), device=device, dtype=dtype)
    k = torch.randn((case["B"], case["H_kv"], case["S_kv"], case["D"]), device=device, dtype=dtype)
    v = torch.randn((case["B"], case["H_kv"], case["S_kv"], case["D"]), device=device, dtype=dtype)
    notes = []
    if case["H_q"] != case["H_kv"]:
        if case["H_q"] % case["H_kv"] != 0:
            raise ValueError("query heads {} is not a multiple of key/value heads {}".format(case["H_q"], case["H_kv"]))
        repeat = case["H_q"] // case["H_kv"]
        k = k.repeat_interleave(repeat, dim=1).contiguous()
        v = v.repeat_interleave(repeat, dim=1).contiguous()
        notes.append("key/value heads expanded {}x to {} (torch 2.2 has no enable_gqa)".format(repeat, case["H_q"]))
    if case["layout"] == "BSHD":
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    return q, k, v, notes


def empty_record(case: dict) -> dict:
    """Build one case's result record, including every field the plan declares."""
    return {
        "probe_id": case["probe_id"], "dtype": case["dtype"],
        "shape": "B{}Hq{}Hkv{}Sq{}Skv{}D{}".format(case["B"], case["H_q"], case["H_kv"], case["S_q"], case["S_kv"], case["D"]),
        "route": "torch_musa_sdpa", "accepted": False, "selected_route": "unverified",
        "deciding_op": None, "latency_ms": None, "peak_memory_bytes": None, "error": None, "evidence": [],
    }


def observed_ops(torch, call):
    """Return the aten op short names the dispatcher ran for one call."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        call()
        torch.musa.synchronize()
    names = set()
    for event in profiler.profiler.function_events:
        if event.name.startswith("aten::"):
            names.add(event.name[len("aten::"):])
    return names


def resolve_route(names: set) -> tuple:
    """Map observed aten ops onto a dispatch path, and say which op decided it."""
    fused = sorted(names & FUSED_LIBRARY_OPS)
    if fused:
        return "fused_library", fused[0]
    library = sorted(names & LIBRARY_COMPOSITION_OPS)
    if library:
        return "library_composition", library[0]
    composition = sorted(names & COMPOSITION_ATEN_OPS)
    if composition:
        return "custom_fallback", "composite:" + ",".join(composition)
    return "unverified", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parents[1] / "tasks" / "sdpa_forward_pilot" / "capability_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=10)
    parser.add_argument("--no-profile", action="store_true", help="skip route resolution and report every route as unverified")
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
        record = empty_record(case)
        try:
            dtype = getattr(torch, case["dtype"])
            q, k, v, notes = build_tensors(torch, case, device, dtype)
            record["evidence"].extend(notes)
            mask = make_mask(torch, case, device)
            kwargs = {"attn_mask": mask, "is_causal": case["causal"]}
            for _ in range(args.warmup):
                F.scaled_dot_product_attention(q, k, v, **kwargs)
            musa.synchronize()

            if args.no_profile:
                record["evidence"].append("route not resolved: --no-profile")
            else:
                observed = observed_ops(torch, lambda: F.scaled_dot_product_attention(q, k, v, **kwargs))
                route, deciding_op = resolve_route(observed)
                record["selected_route"] = route
                record["deciding_op"] = deciding_op
                record["evidence"].append("observed aten ops: " + ", ".join(sorted(observed)))

            elapsed = []
            for _ in range(args.measurements):
                start = time.perf_counter()
                output = F.scaled_dot_product_attention(q, k, v, **kwargs)
                musa.synchronize()
                elapsed.append((time.perf_counter() - start) * 1000)
            record.update(accepted=True, latency_ms=sorted(elapsed)[len(elapsed) // 2])
            record["evidence"].append(f"output_shape={list(output.shape)}")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
    # `routes` stays the per-required-route probing status; `resolved_routes`
    # counts where the cases actually landed.
    probing_status = {route: "not_probed" for route in plan.get("required_routes", [])}
    if probing_status:
        probing_status["torch_musa_sdpa"] = "probed"
    resolved = {}
    for record in results:
        resolved[record["selected_route"]] = resolved.get(record["selected_route"], 0) + 1
    payload = {
        "schema_version": "1.0.0", "family": plan["family"], "status": "complete",
        "routes": probing_status, "resolved_routes": resolved, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
