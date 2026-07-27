import torch
import torch.nn as nn

# torchada patches torch.utils.cpp_extension.load_inline to work transparently
# on Moore Threads MUSA GPUs — CUDA sources compile with mcc automatically.
# You can use either import style:
#
#   Option A (recommended — backed by torchada):
#     import torchada
#     from torch.utils.cpp_extension import load_inline
#
#   Option B (backward-compatible alias, same result):
#     from kernelbench.musa_extension import load_inline
#
from kernelbench.musa_extension import load_inline

# ── Kernel source ──────────────────────────────────────────────────────────
# With torchada, you can write kernels using either:
#   - #include <musa_runtime.h>   (MUSA-native, preferred for MUSA targets)
#   - #include <cuda_runtime.h>   (CUDA source — torchada auto-translates)
# Kernel launch syntax (<<<>>>) and all standard patterns work either way.
elementwise_add_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_musa(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_source,
    functions=["elementwise_add_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_musa(a, b)
