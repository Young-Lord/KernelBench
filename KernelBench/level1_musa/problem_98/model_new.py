import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

kl_div_source = """
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

__global__ void kl_div_kernel(
    const float* predictions,
    const float* targets,
    float* out,
    int batch_size,
    int dim
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int n = batch_size * dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float pred = predictions[i];
        float tgt = targets[i];
        sum += tgt * (logf(tgt) - logf(pred));
    }

    float block_sum = block_reduce_sum(sum, shared);
    if (tid == 0) {
        atomicAdd(out, block_sum);
    }
}

torch::Tensor kl_div_musa(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "shape mismatch");

    int batch_size = static_cast<int>(predictions.size(0));
    int dim = static_cast<int>(predictions.numel() / batch_size);

    auto out = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int num_blocks = 256;

    kl_div_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim);

    out /= static_cast<float>(batch_size);
    return out.squeeze();
}
"""

kl_div_ext = load_inline(
    name="kl_div_loss",
    cpp_sources=kl_div_source,
    functions=["kl_div_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl_div_ext = kl_div_ext

    def forward(self, predictions, targets):
        return self.kl_div_ext.kl_div_musa(predictions, targets)
