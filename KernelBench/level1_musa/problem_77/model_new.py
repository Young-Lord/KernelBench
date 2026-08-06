import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline


convt3d_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>

__global__ void kernelbench_convt3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_depth,
    int64_t in_height,
    int64_t in_width,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t padding_depth,
    int64_t padding_height,
    int64_t padding_width,
    int64_t dilation_depth,
    int64_t dilation_height,
    int64_t dilation_width,
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
  const int64_t out_width_index = remaining % out_width;
  remaining /= out_width;
  const int64_t out_height_index = remaining % out_height;
  remaining /= out_height;
  const int64_t out_depth_index = remaining % out_depth;
  remaining /= out_depth;
  const int64_t out_channel_index = remaining % out_channels;
  const int64_t batch_index = remaining / out_channels;

  const int64_t group_index = out_channel_index / oc_per_group;
  const int64_t out_channel_local =
      out_channel_index - group_index * oc_per_group;
  const int64_t in_channel_start = group_index * ic_per_group;

  float sum = has_bias ? bias[out_channel_index] : 0.0f;
  const int64_t input_batch_offset =
      batch_index * in_channels * in_depth * in_height * in_width;
  const int64_t channel_stride_input = in_depth * in_height * in_width;
  const int64_t weight_channel_stride =
      oc_per_group * kernel_depth * kernel_height * kernel_width;
  const int64_t weight_offset_base =
      in_channel_start * weight_channel_stride +
      out_channel_local * kernel_depth * kernel_height * kernel_width;

  for (int64_t kernel_d = 0; kernel_d < kernel_depth; ++kernel_d) {
    const int64_t numerator_d =
        out_depth_index + padding_depth - dilation_depth * kernel_d;
    if (numerator_d % stride_depth != 0) {
      continue;
    }
    const int64_t in_depth_index = numerator_d / stride_depth;
    if (in_depth_index < 0 || in_depth_index >= in_depth) {
      continue;
    }
    for (int64_t kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
      const int64_t numerator_h =
          out_height_index + padding_height - dilation_height * kernel_h;
      if (numerator_h % stride_height != 0) {
        continue;
      }
      const int64_t in_height_index = numerator_h / stride_height;
      if (in_height_index < 0 || in_height_index >= in_height) {
        continue;
      }
      for (int64_t kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
        const int64_t numerator_w =
            out_width_index + padding_width - dilation_width * kernel_w;
        if (numerator_w % stride_width != 0) {
          continue;
        }
        const int64_t in_width_index = numerator_w / stride_width;
        if (in_width_index < 0 || in_width_index >= in_width) {
          continue;
        }
        const float* input_slice =
            input + input_batch_offset +
            in_channel_start * channel_stride_input +
            (in_depth_index * in_height + in_height_index) * in_width +
            in_width_index;
        const float* weight_slice =
            weight + weight_offset_base +
            (kernel_d * kernel_height + kernel_h) * kernel_width + kernel_w;
        for (int64_t channel = 0; channel < ic_per_group; ++channel) {
          sum += input_slice[channel * channel_stride_input] *
                 weight_slice[channel * weight_channel_stride];
        }
      }
    }
  }
  output[output_index] = sum;
}

torch::Tensor kernelbench_convt3d_musa(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t padding_depth,
    int64_t padding_height,
    int64_t padding_width,
    int64_t output_padding_depth,
    int64_t output_padding_height,
    int64_t output_padding_width,
    int64_t dilation_depth,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t groups,
    int64_t has_bias) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
              "weight must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 5, "input must be 5D");
  TORCH_CHECK(weight.dim() == 5, "weight must be 5D");

  const int64_t batch_size = input.size(0);
  const int64_t in_channels = input.size(1);
  const int64_t in_depth = input.size(2);
  const int64_t in_height = input.size(3);
  const int64_t in_width = input.size(4);
  const int64_t oc_per_group = weight.size(1);
  const int64_t out_channels = oc_per_group * groups;
  const int64_t kernel_depth = weight.size(2);
  const int64_t kernel_height = weight.size(3);
  const int64_t kernel_width = weight.size(4);

  TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
  TORCH_CHECK(in_channels % groups == 0,
              "in_channels must be divisible by groups");
  TORCH_CHECK(out_channels % groups == 0,
              "out_channels must be divisible by groups");

  const int64_t out_depth =
      (in_depth - 1) * stride_depth - 2 * padding_depth +
      dilation_depth * (kernel_depth - 1) + output_padding_depth + 1;
  const int64_t out_height =
      (in_height - 1) * stride_height - 2 * padding_height +
      dilation_height * (kernel_height - 1) + output_padding_height + 1;
  const int64_t out_width =
      (in_width - 1) * stride_width - 2 * padding_width +
      dilation_width * (kernel_width - 1) + output_padding_width + 1;

  torch::Tensor output =
      torch::empty({batch_size, out_channels, out_depth, out_height, out_width},
                   input.options());

  const int64_t ic_per_group = in_channels / groups;
  const int64_t output_element_count =
      batch_size * out_channels * out_depth * out_height * out_width;
  if (output_element_count > 0) {
    constexpr int threads_per_block = 256;
    const int64_t block_count =
        (output_element_count + threads_per_block - 1) / threads_per_block;
    TORCH_CHECK(block_count <= 2147483647LL, "grid is too large");
    kernelbench_convt3d_kernel<<<static_cast<unsigned int>(block_count),
                                 threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        ic_per_group,
        oc_per_group,
        output_element_count,
        static_cast<int>(has_bias));
  }
  return output;
}

"""

convt3d_extension = load_inline(
    name="kernelbench_level1_problem77_convt3d_musa",
    cpp_sources=convt3d_source,
    functions=["kernelbench_convt3d_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride_d, self.stride_h, self.stride_w = self._triple(stride)
        self.pad_d, self.pad_h, self.pad_w = self._triple(padding)
        self.out_pad_d, self.out_pad_h, self.out_pad_w = \
            self._triple(0)
        self.dil_d, self.dil_h, self.dil_w = self._triple(dilation)
        self.groups = 1
        self.has_bias = 1 if bias else 0

    @staticmethod
    def _triple(value):
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1]), int(value[2])
        return int(value), int(value), int(value)

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.empty(0, device=x.device, dtype=x.dtype)
        return convt3d_extension.kernelbench_convt3d_musa(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.pad_d,
            self.pad_h,
            self.pad_w,
            self.out_pad_d,
            self.out_pad_h,
            self.out_pad_w,
            self.dil_d,
            self.dil_h,
            self.dil_w,
            self.groups,
            self.has_bias,
        )
