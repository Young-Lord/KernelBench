"""Level 3 / problem 31 VisionAttention MUSA implementation.

nn.MultiheadAttention is instantiated only to keep the exact reference weight
layout; the forward computes the packed-weights self-attention with the scaled
dot-product core on a hand-written online-softmax (FlashAttention-style) MUSA
kernel that streams the 16k-token key dimension in blocks (this sequence is far
too long for a full row of scores in shared memory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelbench.musa_extension import load_inline

FMHA_SOURCE = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

// Dense (non-causal) multi-head attention forward in online-softmax form, i.e.
// the FlashAttention-I structure used across the MUSA attention archive
// (MT-flashMLA online softmax, llama.cpp fattn, MATE FMHA).  One CTA owns one
// query row; the key/value dimension is swept in blocks of blockDim rows while
// a running (max, sum-of-exp, output) is maintained, so the full sequence never
// needs to fit in shared memory (required for the 16k-token attention here).
// Precondition: head_dim <= blockDim.x.

__device__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    return shared[0];
}

__global__ void fmha_dense_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch, int heads, int seq_len, int head_dim) {
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh % heads;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int T = seq_len;
    const int d = head_dim;
    const int BN = blockDim.x;

    extern __shared__ float shared_mem[];
    float* q_row = shared_mem;                  // d
    float* p_shared = shared_mem + d;           // BN (raw scores, then exp)
    float* reduce_buf = shared_mem + d + BN;    // BN

    const int64_t bh_offset = (static_cast<int64_t>(b) * heads + h) * T * d;
    const float* q_bh = Q + bh_offset;
    const float* k_bh = K + bh_offset;
    const float* v_bh = V + bh_offset;

    for (int i = tid; i < d; i += BN) q_row[i] = q_bh[row * d + i];
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)d);
    float m = -FLT_MAX;
    float l = 0.0f;
    float o_acc = 0.0f;   // only meaningful for tid < d

    const int n_chunks = (T + BN - 1) / BN;
    for (int ch = 0; ch < n_chunks; ++ch) {
        const int j = ch * BN + tid;
        float s = 0.0f;
        if (j < T) {
            const float* k_j = k_bh + j * d;
            for (int i = 0; i < d; ++i) s += q_row[i] * k_j[i];
            s *= scale;
        } else {
            s = -FLT_MAX;
        }
        p_shared[tid] = s;
        __syncthreads();

        const float chunk_max = block_reduce_max(p_shared[tid], reduce_buf);
        const float m_new = fmaxf(m, chunk_max);
        const float r = expf(m - m_new);
        l *= r;
        o_acc *= r;

        const float p = (j < T) ? expf(p_shared[tid] - m_new) : 0.0f;
        p_shared[tid] = p;
        __syncthreads();

        const float chunk_sum = block_reduce_sum(p_shared[tid], reduce_buf);
        l += chunk_sum;

        if (tid < d) {
            float acc = o_acc;
            for (int jj = 0; jj < BN; ++jj) {
                const int jg = ch * BN + jj;
                if (jg < T) {
                    const float w = p_shared[jj];
                    if (w != 0.0f) acc += w * v_bh[jg * d + tid];
                }
            }
            o_acc = acc;
        }
        m = m_new;
        __syncthreads();
    }

    if (tid < d) {
        O[bh_offset + row * d + tid] = l > 0.0f ? o_acc / l : 0.0f;
    }
}

torch::Tensor fmha_dense(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4, "(B,H,T,hs)");
    const int batch = Q.size(0), heads = Q.size(1), seq_len = Q.size(2), head_dim = Q.size(3);
    const int threads = 256;
    TORCH_CHECK(head_dim <= threads, "head_dim must be <= 256 for this kernel");
    torch::Tensor output = torch::empty_like(Q);
    const dim3 grid(batch * heads, seq_len);
    const size_t shared_bytes = static_cast<size_t>(head_dim + 2 * threads) * sizeof(float);
    fmha_dense_kernel<<<grid, threads, shared_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), batch, heads, seq_len, head_dim);
    return output;
}
"""

_fmha_ext = load_inline(
    name="level3_fmha_dense_musa_p31",
    cpp_sources=FMHA_SOURCE,
    functions=["fmha_dense"],
    verbose=False,
)


def _mha_self_attn(x, attn):
    """Packed-weights self attention for x: (L, N, E), returning attn output."""
    L, N, E = x.shape
    heads = attn.num_heads
    hs = E // heads
    w = attn.in_proj_weight
    b = attn.in_proj_bias
    qkv = F.linear(x, w, b)
    q, k, v = qkv.chunk(3, dim=-1)
    qh = q.view(L, N, heads, hs).permute(1, 2, 0, 3).contiguous()
    kh = k.view(L, N, heads, hs).permute(1, 2, 0, 3).contiguous()
    vh = v.view(L, N, heads, hs).permute(1, 2, 0, 3).contiguous()
    yh = _fmha_ext.fmha_dense(qh, kh, vh)  # (N, heads, L, hs)
    y = yh.permute(2, 0, 1, 3).contiguous().view(L, N, E)
    return F.linear(y, attn.out_proj.weight, attn.out_proj.bias)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (L, N, E)
        attn_output = _mha_self_attn(x, self.attn)
        x = self.norm(attn_output + x)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x
