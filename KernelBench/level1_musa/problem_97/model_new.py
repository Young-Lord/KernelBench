"""Level 1 / problem 97 SDPA MUSA implementation (GEMM form).

Two register-tiled SIMT SGEMMs with a row-softmax between them, tuned for the
(32,32,512,1024) fp32 non-causal shape on MTT S4000 / mp_22:
  qk  : S = scale * Q K^T          (NT gemm, K dim = head_dim 1024)
  soft: P = row_softmax(S)         per (bh, row)
  pv  : O = P V                    (NN gemm, K dim = seq 512)
Each GEMM uses 64x64 output tiles, 256 threads (16x16) with 4x4 register
micro-tiles and K-dim chunks of 16 staged in shared memory.
"""

import torch
import torch.nn as nn

from kernelbench.musa_extension import load_inline

sdpa_source = r"""

#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

#define TM 64
#define TN 64
#define BK 16
#define THREADS 256           // 16 x 16
#define MT 4
#define NT 4
#define PADK (BK + 1)         // row stride when columns are kk
#define PADN (TN + 1)         // row stride when columns are n

// S[m][n] = scale * sum_k Q[m][k] * K[n][k];  Q (T,d), K (T,d) row-major
__global__ void qk_nt_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    float* __restrict__ S, int T, int d, float scale) {
    const int bh = blockIdx.x;
    const int m0 = blockIdx.y * TM;
    const int n0 = blockIdx.z * TN;
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int tid = threadIdx.x;
    const int64_t bq = static_cast<int64_t>(bh) * T * d;

    extern __shared__ float smem[];
    float* As = smem;                 // TM * PADK
    float* Bs = smem + TM * PADK;     // TN * PADK

    float acc[MT][NT];
    for (int a = 0; a < MT; ++a)
        for (int b = 0; b < NT; ++b) acc[a][b] = 0.0f;

    for (int k0 = 0; k0 < d; k0 += BK) {
        for (int idx = tid; idx < TM * BK; idx += THREADS) {
            const int r = idx / BK, c = idx % BK;
            As[r * PADK + c] = Q[bq + static_cast<int64_t>(m0 + r) * d + k0 + c];
        }
        for (int idx = tid; idx < TN * BK; idx += THREADS) {
            const int r = idx / BK, c = idx % BK;
            Bs[r * PADK + c] = K[bq + static_cast<int64_t>(n0 + r) * d + k0 + c];
        }
        __syncthreads();
        for (int kk = 0; kk < BK; ++kk) {
            for (int a = 0; a < MT; ++a) {
                const float av = As[(ty * MT + a) * PADK + kk];
                for (int b = 0; b < NT; ++b)
                    acc[a][b] += av * Bs[(tx * NT + b) * PADK + kk];
            }
        }
        __syncthreads();
    }

    const int64_t bs = static_cast<int64_t>(bh) * T * T;
    for (int a = 0; a < MT; ++a)
        for (int b = 0; b < NT; ++b)
            S[bs + static_cast<int64_t>(m0 + ty * MT + a) * T + (n0 + tx * NT + b)] =
                acc[a][b] * scale;
}

// P = softmax over the last dim of S, per row
__global__ void softmax_kernel(const float* __restrict__ S, float* __restrict__ P,
                               int T) {
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
    const float row_max = red[0];
    __syncthreads();
    float ls = 0.0f;
    for (int j = tid; j < T; j += THREADS) ls += expf(S[off + j] - row_max);
    red[tid] = ls;
    __syncthreads();
    for (int st = THREADS / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int j = tid; j < T; j += THREADS)
        P[off + j] = expf(S[off + j] - row_max) * inv;
}

// O[m][n] = sum_j P[m][j] * V[j][n];  P (T,T), V (T,d) row-major
__global__ void pv_nn_kernel(
    const float* __restrict__ P, const float* __restrict__ V,
    float* __restrict__ O, int T, int d) {
    const int bh = blockIdx.x;
    const int m0 = blockIdx.y * TM;
    const int n0 = blockIdx.z * TN;
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int tid = threadIdx.x;
    const int64_t bp = static_cast<int64_t>(bh) * T * T;

    extern __shared__ float smem[];
    float* As = smem;                 // TM * PADK
    float* Bs = smem + TM * PADK;     // BK * PADN

    float acc[MT][NT];
    for (int a = 0; a < MT; ++a)
        for (int b = 0; b < NT; ++b) acc[a][b] = 0.0f;

    for (int k0 = 0; k0 < T; k0 += BK) {
        for (int idx = tid; idx < TM * BK; idx += THREADS) {
            const int r = idx / BK, c = idx % BK;
            As[r * PADK + c] = P[bp + static_cast<int64_t>(m0 + r) * T + k0 + c];
        }
        for (int idx = tid; idx < BK * TN; idx += THREADS) {
            const int kk = idx / TN, c = idx % TN;
            Bs[kk * PADN + c] =
                V[static_cast<int64_t>(bh) * T * d +
                  static_cast<int64_t>(k0 + kk) * d + n0 + c];
        }
        __syncthreads();
        for (int kk = 0; kk < BK; ++kk) {
            for (int a = 0; a < MT; ++a) {
                const float av = As[(ty * MT + a) * PADK + kk];
                for (int b = 0; b < NT; ++b)
                    acc[a][b] += av * Bs[kk * PADN + (tx * NT + b)];
            }
        }
        __syncthreads();
    }

    const int64_t bo = static_cast<int64_t>(bh) * T * d;
    for (int a = 0; a < MT; ++a)
        for (int b = 0; b < NT; ++b)
            O[bo + static_cast<int64_t>(m0 + ty * MT + a) * d + (n0 + tx * NT + b)] =
                acc[a][b];
}

// ---- host ----
torch::Tensor sdpa_gemm(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    const int batch = Q.size(0), heads = Q.size(1);
    const int T = Q.size(2), d = Q.size(3);
    TORCH_CHECK(T % TM == 0 && d % TN == 0, "Tile divisibility");
    auto opts = Q.options();
    torch::Tensor S = torch::empty({batch * heads, T, T}, opts);
    torch::Tensor P = torch::empty({batch * heads, T, T}, opts);
    torch::Tensor O = torch::empty_like(Q);
    const int bh = batch * heads;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    const size_t sm_qk = static_cast<size_t>(TM * PADK + TN * PADK) * sizeof(float);
    const size_t sm_pv = static_cast<size_t>(TM * PADK + BK * PADN) * sizeof(float);
    dim3 g1(bh, T / TM, T / TN);
    qk_nt_kernel<<<g1, THREADS, sm_qk>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(), T, d, scale);
    const size_t sr = THREADS * sizeof(float);
    dim3 g2(bh, T);
    softmax_kernel<<<g2, THREADS, sr>>>(
        S.data_ptr<float>(), P.data_ptr<float>(), T);
    dim3 g3(bh, T / TM, d / TN);
    pv_nn_kernel<<<g3, THREADS, sm_pv>>>(
        P.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), T, d);
    return O;
}
"""

sdpa_extension = load_inline(
    name="level1_problem97_sdpa_gemm_musa",
    cpp_sources=sdpa_source,
    functions=["sdpa_gemm"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return sdpa_extension.sdpa_gemm(Q, K, V)
