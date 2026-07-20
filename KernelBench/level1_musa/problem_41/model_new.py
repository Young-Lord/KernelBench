import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

maxpool1d_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cfloat>

__global__ void maxpool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_length;
    if (idx >= total) {
        return;
    }

    int out_pos = idx % output_length;
    int tmp = idx / output_length;
    int c = tmp % channels;
    int b = tmp / channels;

    int in_base = (b * channels + c) * input_length;
    float max_val = -FLT_MAX;

    for (int k = 0; k < kernel_size; ++k) {
        int in_idx = out_pos * stride + k * dilation - padding;
        float val = -FLT_MAX;
        if (in_idx >= 0 && in_idx < input_length) {
            val = input[in_base + in_idx];
        }
        max_val = fmaxf(max_val, val);
    }

    output[idx] = max_val;
}

torch::Tensor maxpool1d_musa(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length =
        (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    const int block_size = 256;
    const int total = batch_size * channels * output_length;
    const int num_blocks = (total + block_size - 1) / block_size;

    maxpool1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation);

    return output;
}
"""

maxpool1d_ext = load_inline(
    name="maxpool1d",
    cpp_sources=maxpool1d_source,
    functions=["maxpool1d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool1d = maxpool1d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool1d.maxpool1d_musa(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
