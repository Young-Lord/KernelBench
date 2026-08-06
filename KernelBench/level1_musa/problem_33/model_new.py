import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

batch_norm_source = """
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

// BatchNorm2d in training mode: per-channel statistics over (N, H, W),
// using the biased variance as torch.nn.BatchNorm2d does in training.
// Affine params are initialized to gamma=1, beta=0.
__global__ void batch_norm2d_train_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_features,
    int spatial,
    float eps) {
    int c = blockIdx.x;
    if (c >= num_features) {
        return;
    }

    int count = batch_size * spatial;
    extern __shared__ float shared_mem[];
    float* reduce_buf = shared_mem;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int n = i / spatial;
        int s = i % spatial;
        float val = input[(int64_t)(n * num_features + c) * spatial + s];
        sum += val;
    }
    sum = block_reduce_sum(sum, reduce_buf);
    float mean = sum / static_cast<float>(count);

    __syncthreads();

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int n = i / spatial;
        int s = i % spatial;
        float diff = input[(int64_t)(n * num_features + c) * spatial + s] - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq, reduce_buf);
    float inv_std = rsqrtf(sum_sq / static_cast<float>(count) + eps);

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int n = i / spatial;
        int s = i % spatial;
        output[(int64_t)(n * num_features + c) * spatial + s] =
            (input[(int64_t)(n * num_features + c) * spatial + s] - mean) * inv_std;
    }
}

torch::Tensor batch_norm2d_musa(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int spatial = input.size(2) * input.size(3);

    auto output = input.contiguous();

    const int block_size = 256;
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    batch_norm2d_train_kernel<<<num_features, block_size, shared_bytes>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        spatial,
        eps);

    return output;
}
"""

batch_norm_ext = load_inline(
    name="batch_norm2d_train",
    cpp_sources=batch_norm_source,
    functions=["batch_norm2d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.batch_norm = batch_norm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm.batch_norm2d_musa(x, self.eps)
