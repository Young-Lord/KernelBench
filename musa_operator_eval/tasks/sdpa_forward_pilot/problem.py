"""
SDPA forward library dispatch (B_library tier).

The reference Model below is deliberately plain PyTorch: for the B tier the
reference is not the thing under test, the dispatch is. A submission replaces it
with a ModelNew that probes muDNN / torch_musa at runtime, prefers the fused
path, falls back to a library composition, and finally to a custom `.mu` kernel,
while emitting one dispatch trace record per case.

The semantics encoded here are the contract (see `semantics.json`):

- BHSD layout, q is (B, H_q, S_q, D) and k/v are (B, H_kv, S_kv, D);
- causal alignment is top-left, so query i attends to j <= i;
- a window is [query_index - window_left, query_index + window_right], and -1
  means unbounded in that direction;
- scores accumulate in float32 with a max-shift softmax;
- a fully masked query row produces a zero output and a negative-infinite LSE.

Any path a submission takes must reproduce all five.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference SDPA forward, computed with plain PyTorch ops."""

    def __init__(self, scale: float, causal: bool, window_left: int, window_right: int):
        super(Model, self).__init__()
        self.scale = scale
        self.causal = causal
        self.window_left = window_left
        self.window_right = window_right

    def _attention_mask(self, s_q: int, s_kv: int, device: torch.device) -> torch.Tensor:
        query_index = torch.arange(s_q, device=device).unsqueeze(1)
        key_index = torch.arange(s_kv, device=device).unsqueeze(0)
        allowed = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        if self.causal:
            allowed = allowed & (key_index <= query_index)
        if self.window_left >= 0:
            allowed = allowed & (key_index >= query_index - self.window_left)
        if self.window_right >= 0:
            allowed = allowed & (key_index <= query_index + self.window_right)
        return allowed

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q: (B, H_q, S_q, D)
            k: (B, H_kv, S_kv, D)
            v: (B, H_kv, S_kv, D)

        Returns:
            (output, lse) with output shaped like q in q.dtype and lse
            (B, H_q, S_q) in float32.
        """
        num_query_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_query_heads != num_kv_heads:
            repeats = num_query_heads // num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        mask = self._attention_mask(scores.shape[-2], scores.shape[-1], scores.device)

        row_max = scores.amax(dim=-1, keepdim=True)
        # Fully masked rows have a -inf row_max, which would make the subtraction
        # below produce NaN. Substituting 0 keeps them at -inf so that exp -> 0.
        finite_row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        shifted = torch.where(
            mask, scores - finite_row_max, torch.full_like(scores, float("-inf"))
        )
        weights = torch.exp(shifted)

        # For any non-empty row the shifted maximum is exactly 0, so the
        # denominator is at least 1; clamping only affects fully masked rows,
        # where it turns 0/0 into 0 and satisfies the contract.
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        output = torch.matmul(weights, v.float()) / denominator
        lse = row_max.squeeze(-1) + torch.log(denominator.squeeze(-1))

        return output.to(q.dtype), lse


# ============================================================================
# Default case (mirrors smoke_001 in public_cases.json)
# ============================================================================

batch_size = 1
num_query_heads = 4
num_kv_heads = 4
sequence_length_query = 16
sequence_length_key = 16
head_dimension = 32

default_scale = 0.1767766952966369
default_causal = False
default_window_left = -1
default_window_right = -1


def get_init_inputs():
    return [default_scale, default_causal, default_window_left, default_window_right]


def get_inputs():
    q = torch.randn(batch_size, num_query_heads, sequence_length_query, head_dimension)
    k = torch.randn(batch_size, num_kv_heads, sequence_length_key, head_dimension)
    v = torch.randn(batch_size, num_kv_heads, sequence_length_key, head_dimension)
    return [q, k, v]


# ============================================================================
# Task contract metadata
#
# KernelBench has no place for the A/B tier distinction, so the tier and the
# library policy travel with the problem file. `kernel_static_checker`'s
# `validate_library_kernel_static` consumes LIBRARY_POLICY directly.
#
# The module-level defaults above are the case-parameterization surface: the case
# driver appends overrides for them to instantiate each entry of
# public_cases.json. The mapping from case field to variable name lives in
# task.json under `case_parameters`, so the driver can read it without executing
# this file.
# ============================================================================

TIER = "B_library"

LIBRARY_POLICY = {
    "allowed_libraries": ["libmusa", "libmudnn"],
    "allowed_symbol_prefixes": ["musa", "mudnn"],
    "required_trace_fields": ["case_id", "selected_path", "probe_status"],
    "trace_env_var": "KB_DISPATCH_TRACE",
    "dispatch_order": ["fused_library", "library_composition", "custom_fallback"],
    "minimum_gap_cases": 3,
    # `unsupported_dtype` was added after measuring that float16 and bfloat16
    # reach the fused path while float32 and float64 do not. Keep this list
    # identical to `library_policy.required_gap_reasons` in task.json.
    "required_gap_reasons": ["unsupported_dtype", "unsupported_shape", "semantic_mismatch"],
}
