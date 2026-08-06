import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

square_matmul_mnn_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void square_matmul_mnn_kernel(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_c,
    int matrix_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= matrix_size || column >= matrix_size) {
        return;
    }

    float sum = 0.0f;
    for (int inner_index = 0; inner_index < matrix_size; ++inner_index) {
        sum += matrix_a[row * matrix_size + inner_index]
            * matrix_b[inner_index * matrix_size + column];
    }

    matrix_c[row * matrix_size + column] = sum;
}

torch::Tensor square_matmul_mnn_musa(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b
) {
    TORCH_CHECK(matrix_a.scalar_type() == torch::kFloat32, "matrix_a must be float32");
    TORCH_CHECK(matrix_b.scalar_type() == torch::kFloat32, "matrix_b must be float32");
    TORCH_CHECK(matrix_a.is_contiguous(), "matrix_a must be contiguous");
    TORCH_CHECK(matrix_b.is_contiguous(), "matrix_b must be contiguous");
    TORCH_CHECK(matrix_a.device() == matrix_b.device(), "inputs must be on the same device");
    TORCH_CHECK(matrix_a.dim() == 2, "matrix_a must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "matrix_b must be 2D");
    TORCH_CHECK(matrix_a.size(0) == 4096 && matrix_a.size(1) == 4096,
                "matrix_a must have shape (4096, 4096)");
    TORCH_CHECK(matrix_b.size(0) == 4096 && matrix_b.size(1) == 4096,
                "matrix_b must have shape (4096, 4096)");

    constexpr int matrix_size = 4096;
    torch::Tensor matrix_c = torch::empty(
        {matrix_size, matrix_size},
        matrix_a.options()
    );

    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid(
        (matrix_size + threads_per_block.x - 1) / threads_per_block.x,
        (matrix_size + threads_per_block.y - 1) / threads_per_block.y
    );

    square_matmul_mnn_kernel<<<blocks_per_grid, threads_per_block>>>(
        matrix_a.data_ptr<float>(),
        matrix_b.data_ptr<float>(),
        matrix_c.data_ptr<float>(),
        matrix_size
    );

    const musaError_t launch_error = musaGetLastError();
    TORCH_CHECK(
        launch_error == musaSuccess,
        "square_matmul_mnn_kernel launch failed: ",
        musaGetErrorString(launch_error)
    );

    return matrix_c;
}
"""

square_matmul_mnn_extension = load_inline(
    name="kernelbench_level1_problem1_mnn_matmul_musa",
    cpp_sources=square_matmul_mnn_source,
    functions=["square_matmul_mnn_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.square_matmul_extension = square_matmul_mnn_extension

    def forward(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        return self.square_matmul_extension.square_matmul_mnn_musa(matrix_a, matrix_b)
