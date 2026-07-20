import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

selu_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void selu_kernel(const float* x, float* out, int size) {
    const float scale = 1.0507f;
    const float alpha = 1.6733f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val > 0.0f ? val : scale * alpha * (expf(val) - 1.0f);
    }
}

torch::Tensor selu_musa(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    selu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

selu_ext = load_inline(
    name="selu",
    cpp_sources=selu_source,
    functions=["selu_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.selu_ext = selu_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_ext.selu_musa(x)
