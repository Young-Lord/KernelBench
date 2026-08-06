import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


swish_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void swish_kernel(
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
        const float sigmoid_value = input_value > 87.0f
            ? 1.0f
            : (input_value < -87.0f
                ? 0.0f
                : 1.0f / (1.0f + expf(-input_value)));
        output[element_index] = input_value * sigmoid_value;
    }
}

torch::Tensor swish_musa(torch::Tensor input) {
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

    swish_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        element_count
    );
    return output;
}
"""

swish_extension = load_inline(
    name="kernelbench_level1_problem25_swish_musa",
    cpp_sources=swish_source,
    functions=["swish_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.swish_extension = swish_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.swish_extension.swish_musa(input)
