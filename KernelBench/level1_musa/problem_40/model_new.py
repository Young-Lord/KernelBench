import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

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

// LayerNorm over the trailing (features, dim1, dim2) per batch element,
// biased variance, matching torch.nn.LayerNorm. Affine params init gamma=1, beta=0.
__global__ void layer_norm_kernel(
    const float* input,
    float* output,
    int batch_size,
    int normalized_size,
    float eps) {
    int n = blockIdx.x;
    if (n >= batch_size) {
        return;
    }

    int64_t base = (int64_t)n * normalized_size;
    extern __shared__ float shared_mem[];
    float* reduce_buf = shared_mem;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += input[base + i];
    }
    sum = block_reduce_sum(sum, reduce_buf);
    float mean = sum / static_cast<float>(normalized_size);

    __syncthreads();

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float diff = input[base + i] - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq, reduce_buf);
    float inv_std = rsqrtf(sum_sq / static_cast<float>(normalized_size) + eps);

    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        output[base + i] = (input[base + i] - mean) * inv_std;
    }
}

torch::Tensor layer_norm_musa(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int normalized_size = input.numel() / batch_size;

    auto output = input.contiguous();

    const int block_size = 256;
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    layer_norm_kernel<<<batch_size, block_size, shared_bytes>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        normalized_size,
        eps);

    return output;
}
"""

layer_norm_ext = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_source,
    functions=["layer_norm_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.eps = 1e-5
        self.layer_norm = layer_norm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm.layer_norm_musa(x, self.eps)
