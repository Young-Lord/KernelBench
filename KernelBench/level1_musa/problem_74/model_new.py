import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


convt1d_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>

__global__ void kernelbench_convt1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_length,
    int64_t out_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t ic_per_group,
    int64_t oc_per_group,
    int64_t output_element_count,
    int has_bias) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_element_count) {
    return;
  }

  int64_t remaining = output_index;
  const int64_t out_length_index = remaining % out_length;
  remaining /= out_length;
  const int64_t out_channel_index = remaining % out_channels;
  const int64_t batch_index = remaining / out_channels;

  const int64_t group_index = out_channel_index / oc_per_group;
  const int64_t out_channel_local =
      out_channel_index - group_index * oc_per_group;
  const int64_t in_channel_start = group_index * ic_per_group;

  float sum = has_bias ? bias[out_channel_index] : 0.0f;
  const int64_t input_batch_offset = batch_index * in_channels * in_length;
  const int64_t weight_channel_stride = oc_per_group * kernel_size;
  const int64_t weight_offset_base =
      in_channel_start * weight_channel_stride +
      out_channel_local * kernel_size;

  for (int64_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
    const int64_t numerator =
        out_length_index + padding - dilation * kernel_index;
    if (numerator % stride != 0) {
      continue;
    }
    const int64_t in_length_index = numerator / stride;
    if (in_length_index < 0 || in_length_index >= in_length) {
      continue;
    }
    const float* input_slice = input + input_batch_offset +
                               in_channel_start * in_length +
                               in_length_index;
    const float* weight_slice = weight + weight_offset_base + kernel_index;
    for (int64_t channel = 0; channel < ic_per_group; ++channel) {
      sum += input_slice[channel * in_length] *
             weight_slice[channel * weight_channel_stride];
    }
  }
  output[output_index] = sum;
}

torch::Tensor kernelbench_convt1d_musa(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation,
    int64_t groups,
    int64_t has_bias) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
              "weight must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 3, "input must be 3D");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D");

  const int64_t batch_size = input.size(0);
  const int64_t in_channels = input.size(1);
  const int64_t in_length = input.size(2);
  const int64_t oc_per_group = weight.size(1);
  const int64_t out_channels = oc_per_group * groups;
  const int64_t kernel_size = weight.size(2);

  TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
  TORCH_CHECK(in_channels % groups == 0,
              "in_channels must be divisible by groups");
  TORCH_CHECK(out_channels % groups == 0,
              "out_channels must be divisible by groups");

  const int64_t out_length =
      (in_length - 1) * stride - 2 * padding +
      dilation * (kernel_size - 1) + output_padding + 1;

  torch::Tensor output =
      torch::empty({batch_size, out_channels, out_length}, input.options());

  const int64_t ic_per_group = in_channels / groups;
  const int64_t output_element_count =
      batch_size * out_channels * out_length;
  if (output_element_count > 0) {
    constexpr int threads_per_block = 256;
    const int64_t block_count =
        (output_element_count + threads_per_block - 1) / threads_per_block;
    TORCH_CHECK(block_count <= 2147483647LL, "grid is too large");
    kernelbench_convt1d_kernel<<<static_cast<unsigned int>(block_count),
                                 threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        dilation,
        ic_per_group,
        oc_per_group,
        output_element_count,
        static_cast<int>(has_bias));
  }
  return output;
}

"""

convt1d_extension = load_inline(
    name="kernelbench_level1_problem74_convt1d_musa",
    cpp_sources=convt1d_source,
    functions=["kernelbench_convt1d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = 0
        self.dilation = dilation
        self.groups = 1
        self.has_bias = 1 if bias else 0

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.empty(0, device=x.device, dtype=x.dtype)
        return convt1d_extension.kernelbench_convt1d_musa(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
            self.has_bias,
        )
