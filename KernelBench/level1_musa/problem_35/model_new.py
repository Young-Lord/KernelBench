import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

group_norm_source = """
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

// GroupNorm: per (batch, group) statistics over (channels_per_group, H, W),
// biased variance, matching torch.nn.GroupNorm. Affine params init gamma=1, beta=0.
__global__ void group_norm_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_features,
    int num_groups,
    int spatial,
    float eps) {
    int b = blockIdx.x / num_groups;
    int g = blockIdx.x % num_groups;
    int channels_per_group = num_features / num_groups;

    int count = channels_per_group * spatial;
    extern __shared__ float shared_mem[];
    float* reduce_buf = shared_mem;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int c_local = i / spatial;
        int s = i % spatial;
        int c = g * channels_per_group + c_local;
        float val = input[((int64_t)(b * num_features + c)) * spatial + s];
        sum += val;
    }
    sum = block_reduce_sum(sum, reduce_buf);
    float mean = sum / static_cast<float>(count);

    __syncthreads();

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int c_local = i / spatial;
        int s = i % spatial;
        int c = g * channels_per_group + c_local;
        float diff = input[((int64_t)(b * num_features + c)) * spatial + s] - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq, reduce_buf);
    float inv_std = rsqrtf(sum_sq / static_cast<float>(count) + eps);

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int c_local = i / spatial;
        int s = i % spatial;
        int c = g * channels_per_group + c_local;
        int64_t idx = ((int64_t)(b * num_features + c)) * spatial + s;
        output[idx] = (input[idx] - mean) * inv_std;
    }
}

torch::Tensor group_norm_musa(torch::Tensor input, int num_groups, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int spatial = input.size(2) * input.size(3);

    auto output = input.contiguous();

    const int block_size = 256;
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    group_norm_kernel<<<batch_size * num_groups, block_size, shared_bytes>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        num_groups,
        spatial,
        eps);

    return output;
}
"""

group_norm_ext = load_inline(
    name="group_norm",
    cpp_sources=group_norm_source,
    functions=["group_norm_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-5
        self.group_norm = group_norm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm.group_norm_musa(x, self.num_groups, self.eps)
