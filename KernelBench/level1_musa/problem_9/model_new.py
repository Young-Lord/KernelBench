import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

matmul_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void tall_skinny_matmul_kernel(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_c,
    int row_count,
    int inner_size,
    int column_count
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= row_count || column >= column_count) {
        return;
    }

    const long long a_row_offset = static_cast<long long>(row) * inner_size;
    const long long c_offset = static_cast<long long>(row) * column_count + column;

    float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int inner_index = 0;
    for (; inner_index + 7 < inner_size; inner_index += 8) {
        const float a0 = matrix_a[a_row_offset + inner_index];
        const float a1 = matrix_a[a_row_offset + inner_index + 1];
        const float a2 = matrix_a[a_row_offset + inner_index + 2];
        const float a3 = matrix_a[a_row_offset + inner_index + 3];
        const float a4 = matrix_a[a_row_offset + inner_index + 4];
        const float a5 = matrix_a[a_row_offset + inner_index + 5];
        const float a6 = matrix_a[a_row_offset + inner_index + 6];
        const float a7 = matrix_a[a_row_offset + inner_index + 7];
        const long long b_base = static_cast<long long>(inner_index) * column_count + column;
        accumulator[0] += a0 * matrix_b[b_base];
        accumulator[1] += a1 * matrix_b[b_base + column_count];
        accumulator[2] += a2 * matrix_b[b_base + 2 * column_count];
        accumulator[3] += a3 * matrix_b[b_base + 3 * column_count];
        accumulator[4] += a4 * matrix_b[b_base + 4 * column_count];
        accumulator[5] += a5 * matrix_b[b_base + 5 * column_count];
        accumulator[6] += a6 * matrix_b[b_base + 6 * column_count];
        accumulator[7] += a7 * matrix_b[b_base + 7 * column_count];
    }
    for (; inner_index < inner_size; ++inner_index) {
        accumulator[0] += matrix_a[a_row_offset + inner_index]
            * matrix_b[static_cast<long long>(inner_index) * column_count + column];
    }

    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum += accumulator[i];
    }
    matrix_c[c_offset] = sum;
}

torch::Tensor tall_skinny_matmul_musa(
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
    TORCH_CHECK(matrix_a.size(1) == matrix_b.size(0), "inner dimension mismatch");

    const int row_count = static_cast<int>(matrix_a.size(0));
    const int inner_size = static_cast<int>(matrix_a.size(1));
    const int column_count = static_cast<int>(matrix_b.size(1));

    torch::Tensor matrix_c = torch::empty(
        {row_count, column_count},
        matrix_a.options()
    );

    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid(
        (column_count + threads_per_block.x - 1) / threads_per_block.x,
        (row_count + threads_per_block.y - 1) / threads_per_block.y
    );

    tall_skinny_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(
        matrix_a.data_ptr<float>(),
        matrix_b.data_ptr<float>(),
        matrix_c.data_ptr<float>(),
        row_count,
        inner_size,
        column_count
    );

    const musaError_t launch_error = musaGetLastError();
    TORCH_CHECK(
        launch_error == musaSuccess,
        "tall_skinny_matmul_kernel launch failed: ",
        musaGetErrorString(launch_error)
    );

    return matrix_c;
}
"""

tall_skinny_matmul_extension = load_inline(
    name="kernelbench_level1_problem9_matmul_musa",
    cpp_sources=matmul_source,
    functions=["tall_skinny_matmul_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tall_skinny_matmul_extension = tall_skinny_matmul_extension

    def forward(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        return self.tall_skinny_matmul_extension.tall_skinny_matmul_musa(matrix_a, matrix_b)
