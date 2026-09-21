"""
Reference implementation for the A-tier task kb_l1_97.

Copied verbatim from the source record named in task.json. The A tier is audited with the
framework's default rule set, which is the default when a problem
declares no TIER, so this file is the problem and nothing else. Do not
edit it: the submission is a separate module.

The module-level defaults below are the case-parameterisation surface.
The case driver appends overrides for them to instantiate each entry of
public_cases.json; the mapping from case field to variable name lives in
task.json under `case_parameters`.
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
