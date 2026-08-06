import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


softplus_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void softplus_kernel(
    const float* input,
    float* output,
    size_t element_count
) {
    for (
        size_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
        element_index < element_count;
        element_index += blockDim.x * gridDim.x
    ) {
        const float input_value = input[element_index];
        // Numerically stable softplus with beta=1, threshold=20 (as in PyTorch):
        // log1p(exp(x)) for x <= 20, identity for x > 20.
        output[element_index] = input_value > 20.0f
            ? input_value
            : log1pf(expf(input_value));
    }
}

torch::Tensor softplus_musa(torch::Tensor input) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    torch::Tensor output = torch::empty_like(input);
    const size_t element_count = input.numel();
    if (element_count == 0) {
        return output;
    }

    constexpr int threads_per_block = 256;
    int blocks_per_grid = static_cast<int>(
        (element_count + threads_per_block - 1) / threads_per_block
    );
    if (blocks_per_grid > 65535) {
        blocks_per_grid = 65535;
    }

    softplus_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        element_count
    );
    return output;
}
"""

softplus_extension = load_inline(
    name="kernelbench_level1_problem29_softplus_musa",
    cpp_sources=softplus_source,
    functions=["softplus_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softplus_extension = softplus_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.softplus_extension.softplus_musa(input)
