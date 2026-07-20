import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

avgpool1d_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void avgpool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding) {
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
    float sum = 0.0f;

    for (int k = 0; k < kernel_size; ++k) {
        int in_idx = out_pos * stride + k - padding;
        if (in_idx >= 0 && in_idx < input_length) {
            sum += input[in_base + in_idx];
        }
    }

    output[idx] = sum / static_cast<float>(kernel_size);
}

torch::Tensor avgpool1d_musa(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    const int block_size = 256;
    const int total = batch_size * channels * output_length;
    const int num_blocks = (total + block_size - 1) / block_size;

    avgpool1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding);

    return output;
}
"""

avgpool1d_ext = load_inline(
    name="avgpool1d",
    cpp_sources=avgpool1d_source,
    functions=["avgpool1d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avgpool1d = avgpool1d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool1d.avgpool1d_musa(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )
