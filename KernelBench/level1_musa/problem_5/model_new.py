import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

scalar_mul_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void scalar_mul_kernel(
    const float* input,
    float* output,
    float scalar,
    long long total
) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx] * scalar;
    }
}

torch::Tensor scalar_mul_musa(
    torch::Tensor input,
    double scalar
) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");

    const long long total = static_cast<long long>(input.numel());

    torch::Tensor output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = static_cast<int>((total + block_size - 1) / block_size);

    scalar_mul_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(scalar),
        total
    );

    const musaError_t launch_error = musaGetLastError();
    TORCH_CHECK(
        launch_error == musaSuccess,
        "scalar_mul_kernel launch failed: ",
        musaGetErrorString(launch_error)
    );

    return output;
}
"""

scalar_mul_extension = load_inline(
    name="kernelbench_level1_problem5_scalar_mul_musa",
    cpp_sources=scalar_mul_source,
    functions=["scalar_mul_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_mul_extension = scalar_mul_extension

    def forward(self, input: torch.Tensor, scalar: float) -> torch.Tensor:
        return self.scalar_mul_extension.scalar_mul_musa(input, scalar)
