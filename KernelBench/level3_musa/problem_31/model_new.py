"""Level 3 / problem 31 VisionAttention MUSA implementation (GEMM form).

nn.MultiheadAttention is instantiated only to keep the exact reference weight
layout; the forward computes the packed-weights self-attention on a hand
two-stage GEMM pipeline tuned on MTT S4000 / mp_22 for the (L=16384, d=32)
dense shape:
  qk  : S = scale * Q K^T          (M=N=T single-stage tiles, K dim = 32)
  soft: P = row_softmax(S)         per (bh, row)
  pv  : O = P V                    (row tile x 32 cols, K dim = T chunked)
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

#define TM 128
#define TN 128
#define THREADS 256
#define MT 8
#define NT 8
#define PAD (TM > TN ? TM : TN)  // unused placeholder

// ---- S[m][n] = sum_k Q[m][k] K[n][k] * scale ; single-stage (k over d) ----
__global__ void qk_tinyk_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    float* __restrict__ S, int T, int d, float scale) {
    const int bh = blockIdx.x;
    const int m0 = blockIdx.y * TM;
    const int n0 = blockIdx.z * TN;
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int tid = threadIdx.x;
    const int64_t bq = static_cast<int64_t>(bh) * T * d;
    const int PADK = d + 1;

    extern __shared__ float smem[];
    float* As = smem;                  // TM * PADK
    float* Bs = smem + TM * PADK;      // TN * PADK

    for (int idx = tid; idx < TM * (d / 4); idx += THREADS) {
        const int r = idx / (d / 4), c4 = (idx % (d / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(
            Q + bq + static_cast<int64_t>(m0 + r) * d + c4);
        As[r * PADK + c4] = v.x; As[r * PADK + c4 + 1] = v.y;
        As[r * PADK + c4 + 2] = v.z; As[r * PADK + c4 + 3] = v.w;
    }
    for (int idx = tid; idx < TN * (d / 4); idx += THREADS) {
        const int r = idx / (d / 4), c4 = (idx % (d / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(
            K + bq + static_cast<int64_t>(n0 + r) * d + c4);
        Bs[r * PADK + c4] = v.x; Bs[r * PADK + c4 + 1] = v.y;
        Bs[r * PADK + c4 + 2] = v.z; Bs[r * PADK + c4 + 3] = v.w;
    }
    __syncthreads();

    float acc[MT][NT];
    for (int a = 0; a < MT; ++a)
        for (int b = 0; b < NT; ++b) acc[a][b] = 0.0f;
    for (int k = 0; k < d; ++k) {
        for (int a = 0; a < MT; ++a) {
            const float av = As[(ty * MT + a) * PADK + k];
            for (int b = 0; b < NT; ++b)
                acc[a][b] += av * Bs[(tx * NT + b) * PADK + k];
        }
    }
    const int64_t bs = static_cast<int64_t>(bh) * T * T;
    for (int a = 0; a < MT; ++a)
        for (int b = 0; b < NT; ++b)
            S[bs + static_cast<int64_t>(m0 + ty * MT + a) * T + (n0 + tx * NT + b)] =
                acc[a][b] * scale;
}

// ---- P = softmax(S) row-wise ----
__global__ void softmax_kernel(const float* __restrict__ S, float* __restrict__ P, int T) {
    const int bh = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t off = (static_cast<int64_t>(bh) * T + row) * T;
    extern __shared__ float red[];
    float lm = -FLT_MAX;
    for (int j = tid; j < T; j += THREADS) lm = fmaxf(lm, S[off + j]);
    red[tid] = lm;
    __syncthreads();
    for (int st = THREADS / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();
    float ls = 0.0f;
    for (int j = tid; j < T; j += THREADS) ls += expf(S[off + j] - mx);
    red[tid] = ls;
    __syncthreads();
    for (int st = THREADS / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int j = tid; j < T; j += THREADS) P[off + j] = expf(S[off + j] - mx) * inv;
}

// ---- O = P V : M rows x d cols, K dim = T (chunked) ----
#define MV 128          // row tile
#define BK 32
#define PADK2 (BK + 1)
#define PADB (32 + 1)   // cols padded (d assumed 32 here)
__global__ void pv_smalln_kernel(
    const float* __restrict__ P, const float* __restrict__ V,
    float* __restrict__ O, int T, int d) {
    const int bh = blockIdx.x;
    const int m0 = blockIdx.y * MV;
    const int tid = threadIdx.x;
    const int row = m0 + (tid >> 1);      // 128 rows, 2 col-halves per row
    const int half = tid & 1;             // owns cols [half*16, half*16+16)
    const int64_t bp = static_cast<int64_t>(bh) * T * T;

    extern __shared__ float smem[];
    float* As = smem;                  // MV * PADK2
    float* Bs = smem + MV * PADK2;     // BK * PADB

    float acc[16];
    for (int c = 0; c < 16; ++c) acc[c] = 0.0f;

    for (int k0 = 0; k0 < T; k0 += BK) {
        for (int idx = tid; idx < MV * BK; idx += THREADS) {
            const int r = idx / BK, c = idx % BK;
            As[r * PADK2 + c] = P[bp + static_cast<int64_t>(m0 + r) * T + (k0 + c)];
        }
        for (int idx = tid; idx < BK * 32; idx += THREADS) {
            const int kk = idx / 32, c = idx % 32;
            Bs[kk * PADB + c] =
                V[static_cast<int64_t>(bh) * T * d +
                  static_cast<int64_t>(k0 + kk) * d + c];
        }
        __syncthreads();
        const float* arow = As + ((row - m0) * PADK2);
        for (int kk = 0; kk < BK; ++kk) {
            const float av = arow[kk];
            const float* brow = Bs + kk * PADB + half * 16;
            for (int c = 0; c < 16; ++c) acc[c] += av * brow[c];
        }
        __syncthreads();
    }
    float* orow = O + static_cast<int64_t>(bh) * T * d + static_cast<int64_t>(row) * d + half * 16;
    for (int c = 0; c < 16; ++c) orow[c] = acc[c];
}

torch::Tensor sdpa_gemm31(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "contig");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "f32");
    const int batch = Q.size(0), heads = Q.size(1), T = Q.size(2), d = Q.size(3);
    TORCH_CHECK(T % TM == 0 && d % 4 == 0, "tile div");
    const int bh = batch * heads;
    auto opts = Q.options();
    torch::Tensor S = torch::empty({bh, T, T}, opts);
    torch::Tensor P = torch::empty({bh, T, T}, opts);
    torch::Tensor O = torch::empty_like(Q);
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t sm_qk = static_cast<size_t>(2 * TM * (d + 1)) * sizeof(float);
    dim3 g1(bh, T / TM, T / TN);
    qk_tinyk_kernel<<<g1, THREADS, sm_qk>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(), T, d, scale);
    dim3 g2(bh, T);
    softmax_kernel<<<g2, THREADS, THREADS * sizeof(float)>>>(
        S.data_ptr<float>(), P.data_ptr<float>(), T);
    const size_t sm_pv = static_cast<size_t>(MV * PADK2 + BK * PADB) * sizeof(float);
    dim3 g3(bh, T / MV);
    pv_smalln_kernel<<<g3, THREADS, sm_pv>>>(
        P.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), T, d);
    return O;
}
"""

_fmha_ext = load_inline(
    name="level3_fmha_gemm_musa_p31",
    cpp_sources=FMHA_SOURCE,
    functions=["sdpa_gemm31"],
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
    yh = _fmha_ext.sdpa_gemm31(qh, kh, vh)  # dense softmax attention
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
