import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

argmin_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cfloat>
#include <cstdint>
#include <vector>

__global__ void kernelbench_52_argmin_kernel(
    const float* input,
    int64_t* output,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t output_element_count) {
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_element_count) {
        return;
    }

    const int64_t outer_index = output_index / inner_size;
    const int64_t inner_index = output_index - outer_index * inner_size;

    float best_value = FLT_MAX;
    int64_t best_index = 0;
    for (int64_t reduce_index = 0; reduce_index < reduce_size; ++reduce_index) {
        const int64_t input_index =
            (outer_index * reduce_size + reduce_index) * inner_size + inner_index;
        const float value = input[input_index];
        // Strictly-less keeps the first minimal index, matching torch.argmin.
        if (value < best_value) {
            best_value = value;
            best_index = reduce_index;
        }
    }
    output[output_index] = best_index;
}

torch::Tensor kernelbench_52_argmin(torch::Tensor input, int64_t dimension) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    const int64_t dimension_count = input.dim();
    const int64_t normalized_dimension =
        dimension < 0 ? dimension + dimension_count : dimension;
    TORCH_CHECK(
        normalized_dimension >= 0 && normalized_dimension < dimension_count,
        "dimension out of range");

    int64_t outer_size = 1;
    int64_t inner_size = 1;
    for (int64_t axis = 0; axis < normalized_dimension; ++axis) {
        outer_size *= input.size(axis);
    }
    const int64_t reduce_size = input.size(normalized_dimension);
    TORCH_CHECK(reduce_size > 0, "argmin reduction dimension cannot be empty");
    for (int64_t axis = normalized_dimension + 1; axis < dimension_count; ++axis) {
        inner_size *= input.size(axis);
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(dimension_count - 1);
    for (int64_t axis = 0; axis < dimension_count; ++axis) {
        if (axis != normalized_dimension) {
            output_shape.push_back(input.size(axis));
        }
    }
    torch::Tensor output =
        torch::empty(output_shape, input.options().dtype(torch::kLong));

    const int64_t output_element_count = outer_size * inner_size;
    if (output_element_count > 0) {
        constexpr int threads_per_block = 256;
        const int64_t block_count =
            (output_element_count + threads_per_block - 1) / threads_per_block;
        TORCH_CHECK(block_count <= 2147483647LL, "argmin grid is too large");
        kernelbench_52_argmin_kernel
            <<<static_cast<unsigned int>(block_count), threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<int64_t>(),
                outer_size,
                reduce_size,
                inner_size,
                output_element_count);
    }
    return output;
}
"""

argmin_extension = load_inline(
    name="kernelbench_level1_problem52_argmin_musa",
    cpp_sources=argmin_source,
    functions=["kernelbench_52_argmin"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmin_extension.kernelbench_52_argmin(x, self.dim)
