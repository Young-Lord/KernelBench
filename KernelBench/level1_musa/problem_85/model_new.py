import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


conv2d_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t c_in,
    int64_t h_in,
    int64_t w_in,
    int64_t c_out,
    int64_t h_out,
    int64_t w_out,
    int64_t k_h,
    int64_t k_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    int64_t has_bias) {
    const int64_t c_in_per_group = c_in / groups;
    const int64_t c_out_per_group = c_out / groups;
    const int64_t total_outputs = batch_size * c_out * h_out * w_out;
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t flat_index =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat_index < total_outputs;
         flat_index += grid_stride) {
        int64_t rem = flat_index;
        const int64_t out_w = rem % w_out;
        rem /= w_out;
        const int64_t out_h = rem % h_out;
        rem /= h_out;
        const int64_t out_c = rem % c_out;
        rem /= c_out;
        const int64_t batch = rem;

        const int64_t group = out_c / c_out_per_group;
        const int64_t in_c_start = group * c_in_per_group;
        const int64_t in_c_end = in_c_start + c_in_per_group;

        float accumulator = has_bias ? bias[out_c] : 0.0f;
        const float* input_batch = input + batch * (c_in * h_in * w_in);
        for (int64_t in_c = in_c_start; in_c < in_c_end; ++in_c) {
            const float* input_channel = input_batch + in_c * (h_in * w_in);
            const int64_t in_c_local = in_c - in_c_start;
            const float* weight_slice =
                weight + ((out_c * c_in_per_group + in_c_local) * k_h) * k_w;
            for (int64_t k_i = 0; k_i < k_h; ++k_i) {
                const int64_t in_h = out_h * stride_h + k_i * dil_h - pad_h;
                if (in_h < 0 || in_h >= h_in) {
                    continue;
                }
                for (int64_t k_j = 0; k_j < k_w; ++k_j) {
                    const int64_t in_w = out_w * stride_w + k_j * dil_w - pad_w;
                    if (in_w < 0 || in_w >= w_in) {
                        continue;
                    }
                    accumulator +=
                        input_channel[in_h * w_in + in_w] *
                        weight_slice[k_i * k_w + k_j];
                }
            }
        }
        output[flat_index] = accumulator;
    }
}

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (O, C/groups, kH, kW)");

    const int64_t batch_size = input.size(0);
    const int64_t c_in = input.size(1);
    const int64_t h_in = input.size(2);
    const int64_t w_in = input.size(3);
    const int64_t c_out = weight.size(0);
    const int64_t k_h = weight.size(2);
    const int64_t k_w = weight.size(3);

    TORCH_CHECK(c_in % groups == 0, "input channels must be divisible by groups");
    TORCH_CHECK(c_out % groups == 0, "output channels must be divisible by groups");
    TORCH_CHECK(
        weight.size(1) == c_in / groups,
        "weight channel dim mismatch");

    const int64_t h_out = (h_in + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
    const int64_t w_out = (w_in + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(h_out > 0 && w_out > 0, "output spatial size must be positive");

    torch::Tensor output = torch::empty(
        {batch_size, c_out, h_out, w_out}, input.options());

    const int64_t has_bias = (bias.defined() && bias.numel() > 0) ? 1 : 0;
    const int64_t total_outputs = batch_size * c_out * h_out * w_out;
    if (total_outputs > 0) {
        constexpr int threads_per_block = 256;
        const int64_t max_blocks = 1048576LL;
        int64_t block_count = (total_outputs + threads_per_block - 1) / threads_per_block;
        if (block_count > max_blocks) {
            block_count = max_blocks;
        }
        conv2d_kernel<<<static_cast<unsigned int>(block_count), threads_per_block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size,
            c_in,
            h_in,
            w_in,
            c_out,
            h_out,
            w_out,
            k_h,
            k_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            groups,
            has_bias);
        const musaError_t launch_error = musaGetLastError();
        TORCH_CHECK(
            launch_error == musaSuccess,
            "conv2d kernel launch failed: ",
            musaGetErrorString(launch_error));
    }
    return output;
}
"""

conv2d_extension = load_inline(
    name="kernelbench_level1_problem85_conv2d_musa",
    cpp_sources=conv2d_source,
    functions=["conv2d_forward"],
    verbose=False,
)


def _pair(value):
    if isinstance(value, (tuple, list)):
        return value
    return (value, value)


def _bias_arg(bias, reference_tensor):
    if bias is None:
        return reference_tensor.new_empty(0)
    return bias.contiguous()


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv2d
        stride_h, stride_w = _pair(conv.stride)
        pad_h, pad_w = _pair(conv.padding)
        dil_h, dil_w = _pair(conv.dilation)
        return conv2d_extension.conv2d_forward(
            x.contiguous(),
            conv.weight.contiguous(),
            _bias_arg(conv.bias, x),
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            conv.groups,
        )
