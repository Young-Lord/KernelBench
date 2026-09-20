"""
Reference implementation for the B-tier task kb_l3_31.

Copied from KernelBench/level3/31_VisionAttention.py, byte for byte, and then extended with the
tier metadata at the bottom. The B tier is graded on how the answer was
reached, so the problem file has to state the dispatch contract somewhere;
`kernel_static_checker.validate_library_kernel_static` consumes LIBRARY_POLICY
directly, and `eval.py` reads TIER to pick that rule set.

The reference is not the thing under test here, the dispatch is.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]

# ============================================================================
# Task contract metadata
#
# KernelBench has no place for the A/B tier distinction, so the tier and the
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
    # `unsupported_shape` is the one that applies: head_dim is embed_dim /
    # num_heads, both of which are case parameters, and the fused kernel on the
    # measured build declines anything above 128.
    #
    # This list said `layout_mismatch` until a device probe disproved it. The
    # reasoning was that the contract attends the (H*W, B, C) view of a
    # (B, C, H, W) image, so the library could not be called without a conversion.
    # Measured: the fused path accepted the contiguous, the transposed and the
    # strided views this task produces, and refused only a three-dimensional input,
    # which is a rank error rather than a layout the library declines, since
    # nn.MultiheadAttention always hands it four dimensions.
    #
    # `semantic_mismatch` is unreachable: the scale is the default 1/sqrt(D) and
    # there is no mask. `multi_operator_required` is unreachable: one attention op.
    # This is below the two-category floor §4.3 asks for; see
    # `admission.gap_reason_coverage` in task.json.
    "required_gap_reasons": [
        "unsupported_shape"
    ]
}
