import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

// Frobenius norm: norm = sqrt(sum over ALL elements of x^2), y = x / norm.
// Two-phase: grid-stride partial sum-of-squares with atomicAdd into a global
// accumulator, then a grid-wide pass that multiplies by 1/norm.

__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void frobenius_partial_kernel(
    const float* input,
    float* total,
    int64_t numel) {
    extern __shared__ float shared_mem[];
    float* reduce_buf = shared_mem;

    float sum = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += (int64_t)gridDim.x * blockDim.x) {
        float val = input[i];
        sum += val * val;
    }
    sum = block_reduce_sum(sum, reduce_buf);
    if (threadIdx.x == 0) {
        atomicAdd(total, sum);
    }
}

__global__ void frobenius_apply_kernel(
    const float* input,
    float* output,
    const float* total,
    int64_t numel) {
    float inv_norm = rsqrtf(fmaxf(total[0], 1e-30f));
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += (int64_t)gridDim.x * blockDim.x) {
        output[i] = input[i] * inv_norm;
    }
}

torch::Tensor frobenius_norm_musa(torch::Tensor input) {
    const int64_t numel = input.numel();
    auto output = input.contiguous();

    auto total = torch::zeros({1}, input.options());

    const int block_size = 256;
    const int max_blocks = 1 << 20;
    int64_t needed_blocks = (numel + block_size - 1) / block_size;
    int num_blocks = (int)std::min<int64_t>(needed_blocks, max_blocks);
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    frobenius_partial_kernel<<<num_blocks, block_size, shared_bytes>>>(
        output.data_ptr<float>(),
        total.data_ptr<float>(),
        numel);
    frobenius_apply_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        total.data_ptr<float>(),
        numel);

    return output;
}
"""

frobenius_norm_ext = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_source,
    functions=["frobenius_norm_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.frobenius_norm = frobenius_norm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.frobenius_norm_musa(x)
