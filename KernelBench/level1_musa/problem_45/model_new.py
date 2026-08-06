import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

avgpool2d_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>

__global__ void avgpool2d_kernel(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t padding_height,
    int64_t padding_width) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * channels * output_height * output_width;
    if (idx >= total) {
        return;
    }

    int64_t out_w = idx % output_width;
    int64_t tmp = idx / output_width;
    int64_t out_h = tmp % output_height;
    tmp /= output_height;
    int64_t c = tmp % channels;
    int64_t b = tmp / channels;

    const int64_t in_base = (b * channels + c) * input_height * input_width;
    float sum = 0.0f;

    for (int64_t kh = 0; kh < kernel_height; ++kh) {
        const int64_t in_h = out_h * stride_height + kh - padding_height;
        if (in_h < 0 || in_h >= input_height) {
            continue;
        }
        for (int64_t kw = 0; kw < kernel_width; ++kw) {
            const int64_t in_w = out_w * stride_width + kw - padding_width;
            if (in_w < 0 || in_w >= input_width) {
                continue;
            }
            sum += input[in_base + in_h * input_width + in_w];
        }
    }

    // count_include_pad=True (PyTorch default): always divide by the full window volume.
    output[idx] = sum / static_cast<float>(kernel_height * kernel_width);
}

torch::Tensor avgpool2d_musa(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    const int64_t batch_size = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t input_height = input.size(2);
    const int64_t input_width = input.size(3);

    const int64_t output_height =
        (input_height + 2 * padding - kernel_size) / stride + 1;
    const int64_t output_width =
        (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty(
        {batch_size, channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int64_t total = batch_size * channels * output_height * output_width;
    if (total > 0) {
        const int64_t block_count = (total + threads_per_block - 1) / threads_per_block;
        TORCH_CHECK(block_count <= 2147483647LL, "avgpool2d grid is too large");
        avgpool2d_kernel<<<static_cast<unsigned int>(block_count), threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding);
    }
    return output;
}
"""

avgpool2d_ext = load_inline(
    name="avgpool2d",
    cpp_sources=avgpool2d_source,
    functions=["avgpool2d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avgpool2d = avgpool2d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool2d.avgpool2d_musa(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )
