"""
Reference implementation for the B-tier task kb_l3_44.

Copied from KernelBench/level3/44_MiniGPTBlock.py, byte for byte, and then extended with the
tier metadata at the bottom. The B tier is graded on how the answer was
reached, so the problem file has to state the dispatch contract somewhere;
`kernel_static_checker.validate_library_kernel_static` consumes LIBRARY_POLICY
directly, and `eval.py` reads TIER to pick that rule set.

The reference is not the thing under test here, the dispatch is.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Model(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]

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
    # canonical failure_reasons in agent_reference/transformer_block_capabilities.json.
    # `multi_operator_required`: the reference is a whole block, so no single fused
    # call serves it — the fused attention covers the core and the LayerNorms, MLP
    # projections, GELU and residuals have to be composed around it.
    # `unsupported_shape`: head_dim is n_embd / n_head, so the fused path's
    # head-dimension window can be stepped past. `semantic_mismatch` and
    # `layout_mismatch` do not apply: the scale is hard-coded to 1/sqrt(D), the mask
    # is plain causal so no row is empty, and the attention layout is contiguous BHSD.
    "required_gap_reasons": [
        "unsupported_shape",
        "multi_operator_required"
    ]
}
