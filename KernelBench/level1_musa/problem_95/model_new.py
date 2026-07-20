import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

cross_entropy_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>
#include <cfloat>

__device__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void cross_entropy_kernel(
    const float* predictions,
    const int64_t* targets,
    float* out,
    int batch_size,
    int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const float* logits = predictions + batch_idx * num_classes;
    int64_t target = targets[batch_idx];

    extern __shared__ float shared_mem[];
    float* shared = shared_mem;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        local_max = fmaxf(local_max, logits[c]);
    }
    float max_logit = block_reduce_max(local_max, shared);
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        local_sum += expf(logits[c] - max_logit);
    }
    float sum_exp = block_reduce_sum(local_sum, shared);

    if (threadIdx.x == 0) {
        float log_sum_exp = max_logit + logf(sum_exp);
        float loss = log_sum_exp - logits[target];
        atomicAdd(out, loss);
    }
}

torch::Tensor cross_entropy_musa(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D");
    TORCH_CHECK(predictions.size(0) == targets.size(0), "batch size mismatch");

    int batch_size = static_cast<int>(predictions.size(0));
    int num_classes = static_cast<int>(predictions.size(1));

    auto out = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int shared_bytes = block_size * static_cast<int>(sizeof(float));

    cross_entropy_kernel<<<batch_size, block_size, shared_bytes>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        batch_size,
        num_classes);

    out /= static_cast<float>(batch_size);
    return out.squeeze();
}
"""

cross_entropy_ext = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_source,
    functions=["cross_entropy_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy_ext = cross_entropy_ext

    def forward(self, predictions, targets):
        return self.cross_entropy_ext.cross_entropy_musa(predictions, targets.long())
