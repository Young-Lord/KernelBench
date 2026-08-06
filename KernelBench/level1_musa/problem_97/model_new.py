import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

sdpa_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

__device__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// Scaled dot product attention: softmax(Q K^T / sqrt(head_dim)) V.
// Each block handles one query row of one (batch, head) pair.
__global__ void scaled_dot_product_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale) {
    const int bh = blockIdx.x;
    const int row = blockIdx.y;
    const int64_t pair_offset = static_cast<int64_t>(bh) * seq_len * head_dim;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;                  // seq_len floats
    float* q_row = shared_mem + seq_len;         // head_dim floats
    float* reduce_buf = shared_mem + seq_len + head_dim;  // blockDim floats

    const int tid = threadIdx.x;
    const int n = seq_len;
    const int d = head_dim;

    for (int i = tid; i < d; i += blockDim.x) {
        q_row[i] = Q[pair_offset + row * d + i];
    }
    __syncthreads();

    // scores[j] = (Q[row] . K[j]) * scale
    for (int j = tid; j < n; j += blockDim.x) {
        const float* k_j = K + pair_offset + j * d;
        float acc = 0.0f;
        for (int k = 0; k < d; ++k) {
            acc += q_row[k] * k_j[k];
        }
        scores[j] = acc * scale;
    }
    __syncthreads();

    // Row-wise softmax over scores.
    float local_max = -FLT_MAX;
    for (int j = tid; j < n; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    float row_max = block_reduce_max(local_max, reduce_buf);
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float exp_val = expf(scores[j] - row_max);
        scores[j] = exp_val;
        local_sum += exp_val;
    }
    float row_sum = block_reduce_sum(local_sum, reduce_buf);
    __syncthreads();

    for (int j = tid; j < n; j += blockDim.x) {
        scores[j] /= row_sum;
    }
    __syncthreads();

    // O[row, c] = sum_j scores[j] * V[j, c]
    for (int c = tid; c < d; c += blockDim.x) {
        const float* v_col = V + pair_offset + c;
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) {
            acc += scores[j] * v_col[j * d];
        }
        O[pair_offset + row * d + c] = acc;
    }
}

torch::Tensor scaled_dot_product_attention_musa(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "inputs must be 4D (B, H, S, E)");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");

    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seq_len = static_cast<int>(Q.size(2));
    const int head_dim = static_cast<int>(Q.size(3));
    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads &&
                K.size(2) == seq_len && K.size(3) == head_dim, "K shape mismatch");
    TORCH_CHECK(V.size(0) == batch && V.size(1) == heads &&
                V.size(2) == seq_len && V.size(3) == head_dim, "V shape mismatch");

    torch::Tensor output = torch::empty_like(Q);

    const dim3 grid(batch * heads, seq_len);
    constexpr int threads_per_block = 256;
    const size_t shared_bytes =
        static_cast<size_t>(seq_len + head_dim + threads_per_block) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    scaled_dot_product_attention_kernel<<<grid, threads_per_block, shared_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len,
        head_dim,
        scale);
    return output;
}
"""

sdpa_extension = load_inline(
    name="kernelbench_level1_problem97_sdpa_musa",
    cpp_sources=sdpa_source,
    functions=["scaled_dot_product_attention_musa"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return sdpa_extension.scaled_dot_product_attention_musa(Q, K, V)
