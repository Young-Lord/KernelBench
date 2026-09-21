"""
Reference implementation for the B-tier task kb_l1_97.

Copied from the source record named in task.json, byte for byte, and then extended with the
tier metadata at the bottom. The B tier is graded on how the answer was
reached, so the problem file has to state the dispatch contract somewhere;
`kernel_static_checker.validate_library_kernel_static` consumes LIBRARY_POLICY
directly, and `eval.py` reads TIER to pick that rule set.

The reference is not the thing under test here, the dispatch is.
"""


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    return [Q, K, V]

def get_init_inputs():
    return []

# ============================================================================
# Task contract metadata
#
# The framework's problem format has no place for the A/B tier distinction, so the tier and the
# library policy travel with the problem file. Keep the two shared blocks below
# identical to `library_policy` in task.json; tests/test_task_packages.py fails
# if they drift.
# ============================================================================

TIER = "B_library"

LIBRARY_POLICY = {
    "allowed_libraries": [
        "libmusa",
        "libmudnn"
    ],
    "allowed_symbol_prefixes": [
        "musa",
        "mudnn"
    ],
    "required_trace_fields": [
        "case_id",
        "selected_path",
        "probe_status"
    ],
    "trace_env_var": "KB_DISPATCH_TRACE",
    "case_id_env_var": "KB_DISPATCH_CASE_ID",
    "dispatch_order": [
        "fused_library",
        "library_composition",
        "custom_fallback"
    ],
    "minimum_gap_cases": 3,
    # Only the reasons a hidden case of this task can reach; each is one of the
    # canonical failure_reasons in agent_reference/sdpa_forward_capabilities.json.
    # The reference calls scaled_dot_product_attention with no scale argument, so
    # the scale is always the library's own 1/sqrt(D), and there is no mask, so
    # `semantic_mismatch` is unreachable. The layout is contiguous BHSD and the
    # task is one SDPA call, so `layout_mismatch` and `multi_operator_required` do
    # not apply either. `unsupported_shape` is reachable because D is a case
    # parameter and the fused path's head-dimension window ends at 128. This is
    # below the two-category floor §4.3 asks for; see `admission.gap_reason_coverage`
    # in task.json.
    "required_gap_reasons": [
        "unsupported_shape"
    ]
}
