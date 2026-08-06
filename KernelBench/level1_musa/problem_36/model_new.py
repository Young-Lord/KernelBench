import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cmath>

// RMSNorm over dim=1 (num_features): y[n,c,h,w] = x[n,c,h,w] / sqrt(mean_c(x^2) + eps).
// One thread per (n, h, w) position; it loops over all channels.
__global__ void rms_norm_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_features,
    int spatial,
    float eps) {
    int64_t total_positions = (int64_t)batch_size * spatial;
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total_positions) {
        return;
    }

    int n = (int)(p / spatial);
    int64_t hw = p % spatial;

    float sum_sq = 0.0f;
    for (int c = 0; c < num_features; ++c) {
        float val = input[((int64_t)(n * num_features + c)) * spatial + hw];
        sum_sq += val * val;
    }
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(num_features) + eps);

    for (int c = 0; c < num_features; ++c) {
        int64_t idx = ((int64_t)(n * num_features + c)) * spatial + hw;
        output[idx] = input[idx] * inv_rms;
    }
}

torch::Tensor rms_norm_musa(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int spatial = input.size(2) * input.size(3);

    auto output = input.contiguous();

    const int block_size = 256;
    const int64_t total_positions = (int64_t)batch_size * spatial;
    const int num_blocks = (int)((total_positions + block_size - 1) / block_size);

    rms_norm_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        spatial,
        eps);

    return output;
}
"""

rms_norm_ext = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_source,
    functions=["rms_norm_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm.rms_norm_musa(x, self.eps)
