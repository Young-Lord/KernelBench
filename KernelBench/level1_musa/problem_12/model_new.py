import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void diag_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M;
    if (idx >= total) {
        return;
    }

    int n = idx / M;
    C[idx] = A[n] * B[idx];
}

torch::Tensor diag_matmul_musa(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 1, "A must be 1D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B row dimension mismatch");

    int N = static_cast<int>(A.size(0));
    int M = static_cast<int>(B.size(1));
    int total = N * M;

    auto C = torch::empty({N, M}, B.options());

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    diag_matmul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);

    return C;
}
"""

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_source,
    functions=["diag_matmul_musa"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_musa(A, B)
