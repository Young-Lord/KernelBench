import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

batched_matmul_mnn_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void batched_matmul_mnn_kernel(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_c,
    int batch_size,
    int row_count,
    int column_count,
    int inner_size
) {
    const int batch_index = blockIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_index >= batch_size || row >= row_count || column >= column_count) {
        return;
    }

    float sum = 0.0f;
    const int matrix_a_batch_offset = batch_index * row_count * inner_size;
    const int matrix_b_batch_offset = batch_index * inner_size * column_count;
    for (int inner_index = 0; inner_index < inner_size; ++inner_index) {
        sum += matrix_a[matrix_a_batch_offset + row * inner_size + inner_index]
            * matrix_b[matrix_b_batch_offset + inner_index * column_count + column];
    }

    matrix_c[batch_index * row_count * column_count + row * column_count + column] = sum;
}

torch::Tensor batched_matmul_mnn_musa(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b
) {
    TORCH_CHECK(matrix_a.scalar_type() == torch::kFloat32, "matrix_a must be float32");
    TORCH_CHECK(matrix_b.scalar_type() == torch::kFloat32, "matrix_b must be float32");
    TORCH_CHECK(matrix_a.is_contiguous(), "matrix_a must be contiguous");
    TORCH_CHECK(matrix_b.is_contiguous(), "matrix_b must be contiguous");
    TORCH_CHECK(matrix_a.device() == matrix_b.device(), "inputs must be on the same device");
    TORCH_CHECK(matrix_a.dim() == 3, "matrix_a must be 3D");
    TORCH_CHECK(matrix_b.dim() == 3, "matrix_b must be 3D");
    TORCH_CHECK(
        matrix_a.size(0) == 128 && matrix_a.size(1) == 512 && matrix_a.size(2) == 1024,
        "matrix_a must have shape (128, 512, 1024)"
    );
    TORCH_CHECK(
        matrix_b.size(0) == 128 && matrix_b.size(1) == 1024 && matrix_b.size(2) == 2048,
        "matrix_b must have shape (128, 1024, 2048)"
    );

    constexpr int batch_size = 128;
    constexpr int row_count = 512;
    constexpr int inner_size = 1024;
    constexpr int column_count = 2048;
    torch::Tensor matrix_c = torch::empty(
        {batch_size, row_count, column_count},
        matrix_a.options()
    );

    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid(
        (column_count + threads_per_block.x - 1) / threads_per_block.x,
        (row_count + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    batched_matmul_mnn_kernel<<<blocks_per_grid, threads_per_block>>>(
        matrix_a.data_ptr<float>(),
        matrix_b.data_ptr<float>(),
        matrix_c.data_ptr<float>(),
        batch_size,
        row_count,
        column_count,
        inner_size
    );

    const musaError_t launch_error = musaGetLastError();
    TORCH_CHECK(
        launch_error == musaSuccess,
        "batched_matmul_mnn_kernel launch failed: ",
        musaGetErrorString(launch_error)
    );

    return matrix_c;
}
"""

batched_matmul_mnn_extension = load_inline(
    name="kernelbench_level1_problem3_mnn_batch_matmul_musa",
    cpp_sources=batched_matmul_mnn_source,
    functions=["batched_matmul_mnn_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batched_matmul_extension = batched_matmul_mnn_extension

    def forward(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul_extension.batched_matmul_mnn_musa(matrix_a, matrix_b)
