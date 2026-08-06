import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

maxpool2d_source = """
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cfloat>
#include <cstdint>

__global__ void maxpool2d_kernel(
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
    int64_t padding_width,
    int64_t dilation_height,
    int64_t dilation_width) {
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
    float max_val = -FLT_MAX;

    for (int64_t kh = 0; kh < kernel_height; ++kh) {
        const int64_t in_h = out_h * stride_height + kh * dilation_height - padding_height;
        if (in_h < 0 || in_h >= input_height) {
            continue;
        }
        for (int64_t kw = 0; kw < kernel_width; ++kw) {
            const int64_t in_w = out_w * stride_width + kw * dilation_width - padding_width;
            if (in_w < 0 || in_w >= input_width) {
                continue;
            }
            max_val = fmaxf(max_val, input[in_base + in_h * input_width + in_w]);
        }
    }

    output[idx] = max_val;
}

torch::Tensor maxpool2d_musa(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    const int64_t batch_size = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t input_height = input.size(2);
    const int64_t input_width = input.size(3);

    const int64_t output_height =
        (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int64_t output_width =
        (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty(
        {batch_size, channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int64_t total = batch_size * channels * output_height * output_width;
    if (total > 0) {
        const int64_t block_count = (total + threads_per_block - 1) / threads_per_block;
        TORCH_CHECK(block_count <= 2147483647LL, "maxpool2d grid is too large");
        maxpool2d_kernel<<<static_cast<unsigned int>(block_count), threads_per_block>>>(
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
            padding,
            dilation,
            dilation);
    }
    return output;
}
"""

maxpool2d_ext = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_source,
    functions=["maxpool2d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d.maxpool2d_musa(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
