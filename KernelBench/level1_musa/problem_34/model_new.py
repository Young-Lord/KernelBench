import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

instance_norm_source = """
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

__global__ void instance_norm2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_features,
    int height,
    int width,
    float eps) {
    int nc = blockIdx.x;
    int n = nc / num_features;
    int c = nc % num_features;
    if (n >= batch_size) {
        return;
    }

    int spatial = height * width;
    int base = (n * num_features + c) * spatial;

    extern __shared__ float shared_mem[];
    float* reduce_buf = shared_mem;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        sum += input[base + i];
    }
    sum = block_reduce_sum(sum, reduce_buf);
    float mean = sum / static_cast<float>(spatial);

    __syncthreads();

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        float diff = input[base + i] - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq, reduce_buf);
    float inv_std = rsqrtf(sum_sq / static_cast<float>(spatial) + eps);

    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        output[base + i] = (input[base + i] - mean) * inv_std;
    }
}

torch::Tensor instance_norm2d_musa(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = input.contiguous();

    const int block_size = 256;
    const int num_blocks = batch_size * num_features;
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    instance_norm2d_kernel<<<num_blocks, block_size, shared_bytes>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        height,
        width,
        eps);

    return output;
}
"""

instance_norm_ext = load_inline(
    name="instance_norm2d",
    cpp_sources=instance_norm_source,
    functions=["instance_norm2d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.instance_norm = instance_norm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm2d_musa(x, self.eps)
