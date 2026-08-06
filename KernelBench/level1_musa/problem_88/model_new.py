import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


new_gelu_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void new_gelu_kernel(
    const float* input,
    float* output,
    size_t element_count
) {
    // Constants from the minGPT reference implementation:
    // sqrt(2 / pi) and the cubic coefficient 0.044715.
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    constexpr float cubic_coefficient = 0.044715f;
    for (
        size_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
        element_index < element_count;
        element_index += blockDim.x * gridDim.x
    ) {
        const float input_value = input[element_index];
        const float inner = sqrt_two_over_pi * (
            input_value + cubic_coefficient * input_value * input_value * input_value
        );
        output[element_index] = 0.5f * input_value * (1.0f + tanhf(inner));
    }
}

torch::Tensor new_gelu_musa(torch::Tensor input) {
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

    new_gelu_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        element_count
    );
    return output;
}
"""

new_gelu_extension = load_inline(
    name="kernelbench_level1_problem88_new_gelu_musa",
    cpp_sources=new_gelu_source,
    functions=["new_gelu_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.new_gelu_extension = new_gelu_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.new_gelu_extension.new_gelu_musa(input)
