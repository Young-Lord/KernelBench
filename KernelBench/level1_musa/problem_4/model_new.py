import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

matvec_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void matvec_kernel(
    const float* matrix_a,
    const float* vector_b,
    float* vector_c,
    int row_count,
    int inner_size,
    int block_size
) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }
    const int tid = threadIdx.x;
    const long long a_row_offset = static_cast<long long>(row) * inner_size;

    float partial = 0.0f;
    for (int k = tid; k < inner_size; k += block_size) {
        partial += matrix_a[a_row_offset + k] * vector_b[k];
    }

    __shared__ float shared_sums[1024];
    shared_sums[tid] = partial;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        vector_c[row] = shared_sums[0];
    }
}

torch::Tensor matvec_musa(
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
    TORCH_CHECK(matrix_b.size(1) == 1, "matrix_b must have shape (K, 1)");
    TORCH_CHECK(matrix_a.size(1) == matrix_b.size(0), "inner dimension mismatch");

    const int row_count = static_cast<int>(matrix_a.size(0));
    const int inner_size = static_cast<int>(matrix_a.size(1));

    torch::Tensor vector_c = torch::empty(
        {row_count, 1},
        matrix_a.options()
    );

    const int block_size = 256;
    const int num_blocks = row_count;

    matvec_kernel<<<num_blocks, block_size>>>(
        matrix_a.data_ptr<float>(),
        matrix_b.data_ptr<float>(),
        vector_c.data_ptr<float>(),
        row_count,
        inner_size,
        block_size
    );

    const musaError_t launch_error = musaGetLastError();
    TORCH_CHECK(
        launch_error == musaSuccess,
        "matvec_kernel launch failed: ",
        musaGetErrorString(launch_error)
    );

    return vector_c;
}
"""

matvec_extension = load_inline(
    name="kernelbench_level1_problem4_matvec_musa",
    cpp_sources=matvec_source,
    functions=["matvec_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matvec_extension = matvec_extension

    def forward(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        return self.matvec_extension.matvec_musa(matrix_a, matrix_b)
