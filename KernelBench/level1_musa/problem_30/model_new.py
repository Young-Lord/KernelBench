import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void softsign_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor softsign_musa(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softsign_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

softsign_ext = load_inline(
    name="softsign",
    cpp_sources=softsign_source,
    functions=["softsign_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softsign_ext = softsign_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign_ext.softsign_musa(x)
