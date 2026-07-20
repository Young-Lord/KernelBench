import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

triplet_margin_source = """
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

__global__ void triplet_margin_kernel(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* out,
    int batch_size,
    int dim,
    float margin
) {
    __shared__ float shared_ap[256];
    __shared__ float shared_an[256];

    int b = blockIdx.x;
    if (b >= batch_size) {
        return;
    }

    int tid = threadIdx.x;
    int base = b * dim;

    float sq_ap = 0.0f;
    float sq_an = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float diff_ap = anchor[base + d] - positive[base + d];
        float diff_an = anchor[base + d] - negative[base + d];
        sq_ap += diff_ap * diff_ap;
        sq_an += diff_an * diff_an;
    }

    sq_ap = block_reduce_sum(sq_ap, shared_ap);
    sq_an = block_reduce_sum(sq_an, shared_an);

    if (tid == 0) {
        float d_ap = sqrtf(sq_ap);
        float d_an = sqrtf(sq_an);
        float loss = fmaxf(0.0f, d_ap - d_an + margin);
        atomicAdd(out, loss);
    }
}

torch::Tensor triplet_margin_musa(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    TORCH_CHECK(anchor.is_contiguous(), "anchor must be contiguous");
    TORCH_CHECK(positive.is_contiguous(), "positive must be contiguous");
    TORCH_CHECK(negative.is_contiguous(), "negative must be contiguous");

    int batch_size = static_cast<int>(anchor.size(0));
    int dim = static_cast<int>(anchor.size(1));

    auto out = torch::zeros({1}, anchor.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    triplet_margin_kernel<<<num_blocks, block_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim,
        margin);

    out /= static_cast<float>(batch_size);
    return out.squeeze();
}
"""

triplet_margin_ext = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_source,
    functions=["triplet_margin_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0) -> None:
        super().__init__()
        self.margin = margin
        self.triplet_margin_ext = triplet_margin_ext

    def forward(self, anchor, positive, negative):
        return self.triplet_margin_ext.triplet_margin_musa(
            anchor, positive, negative, self.margin
        )
