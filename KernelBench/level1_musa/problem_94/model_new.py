import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

mse_loss_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>

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

__global__ void mse_loss_kernel(
    const float* predictions,
    const float* targets,
    float* out,
    int64_t element_count) {
    __shared__ float shared[256];

    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    float local_sum = 0.0f;
    for (int64_t i = index; i < element_count; i += stride) {
        float diff = predictions[i] - targets[i];
        local_sum += diff * diff;
    }
    float block_sum = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

torch::Tensor mse_loss_musa(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "shape mismatch");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");

    const int64_t element_count = predictions.numel();
    auto out = torch::zeros({}, predictions.options());

    if (element_count > 0) {
        constexpr int threads_per_block = 256;
        constexpr int block_count = 256;
        mse_loss_kernel<<<block_count, threads_per_block>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            out.data_ptr<float>(),
            element_count);
        out /= static_cast<double>(element_count);
    }
    return out;
}
"""

mse_loss_extension = load_inline(
    name="kernelbench_level1_problem94_mse_loss_musa",
    cpp_sources=mse_loss_source,
    functions=["mse_loss_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return mse_loss_extension.mse_loss_musa(predictions, targets)
