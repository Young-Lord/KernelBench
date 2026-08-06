import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


leaky_relu_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void leaky_relu_kernel(
    const float* input,
    float* output,
    float negative_slope,
    size_t element_count
) {
    for (
        size_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
        element_index < element_count;
        element_index += blockDim.x * gridDim.x
    ) {
        const float input_value = input[element_index];
        output[element_index] = input_value >= 0.0f
            ? input_value
            : negative_slope * input_value;
    }
}

torch::Tensor leaky_relu_musa(torch::Tensor input, double negative_slope) {
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

    leaky_relu_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(negative_slope),
        element_count
    );
    return output;
}
"""

leaky_relu_extension = load_inline(
    name="kernelbench_level1_problem20_leaky_relu_musa",
    cpp_sources=leaky_relu_source,
    functions=["leaky_relu_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_extension = leaky_relu_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_extension.leaky_relu_musa(
            input, self.negative_slope
        )
