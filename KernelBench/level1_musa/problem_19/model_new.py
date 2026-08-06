import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


relu_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void relu_kernel(
    const float* input,
    float* output,
    size_t element_count
) {
    for (
        size_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
        element_index < element_count;
        element_index += blockDim.x * gridDim.x
    ) {
        output[element_index] = fmaxf(input[element_index], 0.0f);
    }
}

torch::Tensor relu_musa(torch::Tensor input) {
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

    relu_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        element_count
    );
    return output;
}
"""

relu_extension = load_inline(
    name="kernelbench_level1_problem19_relu_musa",
    cpp_sources=relu_source,
    functions=["relu_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu_extension = relu_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu_extension.relu_musa(input)
