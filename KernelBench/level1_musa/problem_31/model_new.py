import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


elu_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

__global__ void elu_kernel(
    const float* input,
    float* output,
    float alpha,
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
            : alpha * (expf(input_value) - 1.0f);
    }
}

torch::Tensor elu_musa(torch::Tensor input, double alpha) {
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

    elu_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(alpha),
        element_count
    );
    return output;
}
"""

elu_extension = load_inline(
    name="kernelbench_level1_problem31_elu_musa",
    cpp_sources=elu_source,
    functions=["elu_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.elu_extension = elu_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.elu_extension.elu_musa(input, self.alpha)
