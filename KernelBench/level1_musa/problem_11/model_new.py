import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

einsum_4d_matmul_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void einsum_bijl_lk_kernel(
    const float* A,
    const float* B,
    float* C,
    int b,
    int i,
    int j,
    int l_dim,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * i * j * k;
    if (idx >= total) {
        return;
    }

    int k_idx = idx % k;
    int tmp = idx / k;
    int j_idx = tmp % j;
    tmp /= j;
    int i_idx = tmp % i;
    int b_idx = tmp / i;

    float sum = 0.0f;
    int a_base = ((b_idx * i + i_idx) * j + j_idx) * l_dim;
    for (int l = 0; l < l_dim; ++l) {
        sum += A[a_base + l] * B[l * k + k_idx];
    }
    C[idx] = sum;
}

torch::Tensor einsum_4d_matmul_musa(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "A and B inner dimension mismatch");

    int b = static_cast<int>(A.size(0));
    int i = static_cast<int>(A.size(1));
    int j = static_cast<int>(A.size(2));
    int l_dim = static_cast<int>(A.size(3));
    int k = static_cast<int>(B.size(1));

    auto C = torch::empty({b, i, j, k}, A.options());
    int total = b * i * j * k;

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    einsum_bijl_lk_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        b, i, j, l_dim, k);

    return C;
}
"""

einsum_4d_matmul = load_inline(
    name="einsum_4d_matmul",
    cpp_sources=einsum_4d_matmul_source,
    functions=["einsum_4d_matmul_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.einsum_4d_matmul = einsum_4d_matmul

    def forward(self, A, B):
        return self.einsum_4d_matmul.einsum_4d_matmul_musa(A, B)
