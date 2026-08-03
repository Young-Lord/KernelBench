# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from datasets import load_dataset, Dataset
import pandas as pd
import json
from typing import Optional
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask

def get_sample(pytorch_module: str,
               dataset_id: str,
               level_id: str,
               pytorch_functional: Optional[str] = None,
               cuda_code: Optional[str] = None,
               extra_info: Optional[dict] = None) -> str:
    user_content = r'''You are an expert in PyTorch and CUDA programming. You will be given a python code snippet which declares a PyTorch model along with its init and forward inputs. The model is an instance of class `Model`. It will be created with arguments from `get_init_inputs()`. Then its `forward` function will be called with data from `get_inputs()`. Your task is to write a custom CUDA extension to accelerate the model `forward` function. Note that:

- Think carefully before providing your final answer.
- Provide only a single python code block in your final answer.
- Name your optimized model as `ModelNew`. Keep its `__init__` and `forward` function signature the same as `Model`. Keep the names of all submodules unchanged. Ensure the keys of model state dict are unchanged. Do not create any extra tensor parameter during model initialization.
- Inline the CUDA code within quotes and assign it to the `source` variable. Inline the C++ function definition into the `cpp_src` variable. Compile and load the extension using `torch.utils.cpp_extension.load_inline`.
- Carefully decide the kernel function signature and pass the correct arguments into your kernel.
- Do not perform extra initialization on parameters of any submodule. Keep them initialized by default.
- Implement all CUDA operators by yourself. Do not call any function from `torch` namespace except for allocating or initializing tensors. Do not call any function from `torch.nn.functional` namespace. Do not call the forward function of any submodule. You can only use the parameters and attributes of the submodule. For example, you should pass `self.linear.weight` and `self.linear.bias` as arguments to your CUDA kernel instead of directly running `self.linear(x)`.
- You can implement more than one kernel in the CUDA extension. If there are multiple operators within `forward` function, you must implement all of them no matter how many CUDA kernels are needed.

Here is an basic example:

===== Example 1 Start =====

Given PyTorch model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [2.0]
```

You should respond with:

[Your thinking process]

Then the final answer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = alpha * a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b, float alpha) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), alpha, size);

    return out;
}
"""

cpp_src = """
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b, float alpha);
"""

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return elementwise_add.elementwise_add_cuda(a, b, self.alpha)
```

===== Example 1 End =====

Here is another example with a `Conv2d` child module `self.conv2d`. You need to retain the name `conv2d` in your `ModelNew`. Remember not to call `self.conv2d(x)` within your optimized `forward` function. Use `self.conv2d.weight`, `self.conv2d.bias` and its other attributes instead.

===== Example 2 Start =====

Given PyTorch model:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5) # Asymmetric kernel
width = 256
height = 256
stride = 1
padding = (1, 2) # Asymmetric padding
dilation = (2, 1) # Asymmetric dilation

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
```

You should respond with:

[Your thinking process]

```python
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and C++ wrapper for 2D convolution
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int iH, int iW,
    int oH, int oW,
    int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * oH * oW;
    if (idx >= total_elements) {
        return;
    }

    int b = idx / (out_channels * oH * oW);
    int remainder = idx % (out_channels * oH * oW);
    int oc = remainder / (oH * oW);
    remainder = remainder % (oH * oW);
    int oh = remainder / oW;
    int ow = remainder % oW;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int ih = oh * stride_h - padding_h + kh * dilation_h;
                int iw = ow * stride_w - padding_w + kw * dilation_w;
                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                    int input_idx = b * (in_channels * iH * iW) + ic * (iH * iW) + ih * iW + iw;
                    int weight_idx = oc * (in_channels * kH * kW) + ic * (kH * kW) + kh * kW + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    output[idx] = sum;
}

torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int iH = input.size(2);
    int iW = input.size(3);

    int out_channels = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);

    int oH = (iH + 2 * padding_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int oW = (iW + 2 * padding_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch, out_channels, oH, oW}, input.options());

    int total_elements = output.numel();
    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        iH, iW,
        oH, oW,
        kH, kW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        bias.has_value()
    );

    return output;
}
"""

cpp_src = """
torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);
"""

# Load the CUDA extension
conv2d_cuda = load_inline(
    name='conv2d_cuda',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['conv2d_cuda_forward'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        return conv2d_cuda.conv2d_cuda_forward(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.conv2d.stride[0],
            self.conv2d.stride[1],
            self.conv2d.padding[0],
            self.conv2d.padding[1],
            self.conv2d.dilation[0],
            self.conv2d.dilation[1]
        )
```

===== Example 2 End =====

Here is the last example containing multiple operations. You must implement all the operations in the CUDA extension. You can choose to implement more than one kernel.

===== Example 3 Start =====

Given PyTorch model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.randn(512, 1024).cuda()
    return [x]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
```

You should respond with:

[Your thinking process]

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernels for linear and gelu computation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_forward_kernel(const float* x, const float* weight, const float* bias, float* output, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += x[row * K + k] * weight[col * K + k];
        }
        sum += bias[col];
        output[row * N + col] = sum;
    }
}

torch::Tensor linear_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);

    auto output = torch::zeros({M, N}, x.options());

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    linear_forward_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), M, N, K);

    return output;
}

__global__ void gelu_kernel(const float* x, float* out, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x_val + 0.044715f * x_val * x_val * x_val)));
        out[idx] = x_val * cdf;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int64_t size = x.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

cpp_src = """
torch::Tensor linear_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor gelu_cuda(torch::Tensor x);
"""

# Compile and load the CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['linear_forward_cuda', 'gelu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        x = fused_ops.linear_forward_cuda(x, self.fc1.weight, self.fc1.bias)
        x = fused_ops.gelu_cuda(x)
        x = fused_ops.linear_forward_cuda(x, self.fc2.weight, self.fc2.bias)
        return x
```

===== Example 3 End =====

Now you are given the below PyTorch model:
```python
__PYTORCH_MODULE__
```

Please respond with thinking process and your final answer.
'''.replace('__PYTORCH_MODULE__', pytorch_module.strip())

    output = {
        'prompt': [{
            'role': 'user',
            'content': user_content
        }],
        'data_source': f'{dataset_id}/level_{level_id}',
        'ability': 'code',
        'answer': '',
        'raw_problem': user_content,
        'level': level_id,
        'type': '',
        'reward_model': {
            'ground_truth':
                json.dumps({
                    'pytorch_module': pytorch_module,
                    'pytorch_functional': pytorch_functional or '',
                    'cuda_code': cuda_code or ''
                }),
            'style':
                'cuda-sandbox-v2'
        },
        'extra_info': extra_info or {}
    }
    return output

def convert_kernel_bench():
    dataset_id = 'ScalingIntelligence/KernelBench'
    dataset_full: dict[str, Dataset] = load_dataset(dataset_id)

    def convert_sample(sample):
        pytorch_module = sample['code']
        level_id = str(sample['level'])
        op_name = sample['name']
        problem_id = sample['problem_id']
        output = get_sample(pytorch_module=pytorch_module,
                            dataset_id=dataset_id,
                            level_id=level_id,
                            extra_info={
                                'op_name': op_name,
                                'problem_id': problem_id
                            })
        return pd.Series(output)

    for key, dataset in dataset_full.items():
        output_path = f"~/data/KernelBench/{key}.parquet"
        df = dataset.to_pandas()
        df = df.apply(convert_sample, axis=1)
        df.to_parquet(output_path)

if __name__ == '__main__':
    # eval dataset
    os.makedirs(os.path.expanduser('~/data/KernelBench'), exist_ok=True)
    convert_kernel_bench()
