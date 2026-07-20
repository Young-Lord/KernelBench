import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

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

__global__ void hinge_loss_kernel(
    const float* predictions,
    const float* targets,
    float* out,
    int rows,
    int cols
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int n = rows * cols;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        int col = i % cols;
        float pred = predictions[i];
        float tgt = targets[col];
        float val = 1.0f - pred * tgt;
        sum += val > 0.0f ? val : 0.0f;
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (tid == 0) {
        atomicAdd(out, block_sum);
    }
}

torch::Tensor hinge_loss_musa(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D");

    int rows = static_cast<int>(predictions.size(0));
    int cols = static_cast<int>(predictions.size(1));

    auto out = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int num_blocks = 256;

    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols);

    out /= static_cast<float>(rows * cols);
    return out.squeeze();
}
"""

hinge_loss_ext = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_source,
    functions=["hinge_loss_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hinge_loss_ext = hinge_loss_ext

    def forward(self, predictions, targets):
        return self.hinge_loss_ext.hinge_loss_musa(predictions, targets)
