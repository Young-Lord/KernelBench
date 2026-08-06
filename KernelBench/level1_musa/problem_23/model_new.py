import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


softmax_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>
#include <float.h>

__global__ void softmax_last_dim_kernel(
    const float* input,
    float* output,
    size_t row_count,
    size_t row_size
) {
    const size_t row = blockIdx.x;
    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;

    // Pass 1: find the row maximum for numerical stability.
    float row_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_max = fmaxf(row_max, row_input[i]);
    }

    __shared__ float shared_max[1024];
    shared_max[threadIdx.x] = row_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_max[threadIdx.x] = fmaxf(
                shared_max[threadIdx.x], shared_max[threadIdx.x + offset]
            );
        }
        __syncthreads();
    }
    row_max = shared_max[0];

    // Pass 2: sum exp(x - max); double accumulation keeps the large sum
    // accurate enough to match the PyTorch fp32 reference within tolerance.
    double row_sum = 0.0;
    for (size_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_sum += expf(row_input[i] - row_max);
    }

    __shared__ double shared_sum[1024];
    shared_sum[threadIdx.x] = row_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const double sum_exp = shared_sum[0];

    // Pass 3: write normalized probabilities.
    for (size_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_output[i] = static_cast<float>(
            expf(row_input[i] - row_max) / sum_exp
        );
    }
}

torch::Tensor softmax_last_dim_musa(torch::Tensor input) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");

    torch::Tensor output = torch::empty_like(input);
    const size_t row_size = static_cast<size_t>(input.size(-1));
    const size_t row_count = input.numel() / row_size;
    if (row_count == 0 || row_size == 0) {
        return output;
    }

    constexpr int threads_per_block = 256;

    softmax_last_dim_kernel<<<row_count, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        row_count,
        row_size
    );
    return output;
}
"""

softmax_extension = load_inline(
    name="kernelbench_level1_problem23_softmax_musa",
    cpp_sources=softmax_source,
    functions=["softmax_last_dim_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax_extension = softmax_extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reference applies softmax along dim=1 of a 2D (batch, features)
        # tensor, i.e. the last dimension.
        return self.softmax_extension.softmax_last_dim_musa(input)
