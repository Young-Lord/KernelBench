"""Level 1 / problem 97 SDPA MUSA implementation.

Row-group band SDPA kernel tuned on MTT S4000 / mp_22 for the (32,32,512,1024)
fp32 non-causal shape:
  - one CTA owns BAND_R=8 query rows of one (batch, head) pair (256 threads,
    one warp per row; a lane holds columns lane+32*s of its row in registers);
  - warps walk the same key j, reading one coalesced K row (L1-shared across
    the band), so K DRAM traffic is cut ~8x vs one-row-per-CTA;
  - the V weighted sum keeps one register accumulator per row per thread and
    reads each V column once per key for all 8 rows (~8x less V traffic);
  - softmax weights for the band live in shared memory.
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

#define BAND_R 8
#define THREADS 256

__global__ void sdpa_band_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len, int head_dim, float scale) {
    const int bh = blockIdx.x;
    const int row0 = blockIdx.y * BAND_R;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;         // 0..7 -> row inside the band
    const int T = seq_len;
    const int d = head_dim;
    const int64_t po = static_cast<int64_t>(bh) * T * d;

    extern __shared__ float smem[];
    float* p_s = smem;                 // BAND_R * T softmax weights
    float* red = smem + BAND_R * T;    // THREADS partials

    const int row = row0 + warp;
    const float* qrow = Q + po + static_cast<int64_t>(row) * d;
    float qv[32];                      // lane owns columns lane+32*s
    for (int s = 0; s < 32; ++s) qv[s] = qrow[s * 32 + lane];

    // ---- raw scores: all warps walk the same key j (K row shared in L1) ----
    for (int j = 0; j < T; ++j) {
        const float* k_j = K + po + static_cast<int64_t>(j) * d;
        float part = 0.0f;
        for (int s = 0; s < 32; ++s) part += qv[s] * k_j[s * 32 + lane];
        red[tid] = part;
        __syncthreads();
        if (lane == 0) {
            float acc = 0.0f;
            for (int l = 0; l < 32; ++l) acc += red[warp * 32 + l];
            p_s[warp * T + j] = acc * scale;
        }
        __syncthreads();
    }

    // ---- per-row softmax in shared, done warp-by-warp ----
    const int r = warp;
    {
        float lm = -FLT_MAX;
        for (int j = lane; j < T; j += 32) lm = fmaxf(lm, p_s[r * T + j]);
        red[warp * 32 + lane] = lm;
        __syncthreads();
        for (int st = 16; st > 0; st >>= 1) {
            if (lane < st) red[warp * 32 + lane] =
                fmaxf(red[warp * 32 + lane], red[warp * 32 + lane + st]);
            __syncthreads();
        }
        const float row_max = red[warp * 32];
        __syncthreads();

        float ls = 0.0f;
        for (int j = lane; j < T; j += 32) ls += expf(p_s[r * T + j] - row_max);
        red[warp * 32 + lane] = ls;
        __syncthreads();
        for (int st = 16; st > 0; st >>= 1) {
            if (lane < st) red[warp * 32 + lane] += red[warp * 32 + lane + st];
            __syncthreads();
        }
        const float row_sum = red[warp * 32];
        __syncthreads();
        for (int j = lane; j < T; j += 32)
            p_s[r * T + j] = expf(p_s[r * T + j] - row_max) / row_sum;
        __syncthreads();
    }

    // ---- O: thread owns column c (coalesced), accumulates all R rows ----
    for (int cg = 0; cg < d / THREADS; ++cg) {
        const int c = cg * THREADS + tid;
        float acc[BAND_R];
        for (int rr = 0; rr < BAND_R; ++rr) acc[rr] = 0.0f;
        for (int j = 0; j < T; ++j) {
            const float v = V[po + static_cast<int64_t>(j) * d + c];
            for (int rr = 0; rr < BAND_R; ++rr) acc[rr] += p_s[rr * T + j] * v;
        }
        for (int rr = 0; rr < BAND_R; ++rr)
            O[po + static_cast<int64_t>(row0 + rr) * d + c] = acc[rr];
    }
}

torch::Tensor sdpa_band(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    const int batch = Q.size(0), heads = Q.size(1);
    const int seq_len = Q.size(2), head_dim = Q.size(3);
    TORCH_CHECK(seq_len % BAND_R == 0, "seq_len % 8 == 0");
    TORCH_CHECK(seq_len % 32 == 0, "seq_len % 32 == 0");
    TORCH_CHECK(head_dim % 32 == 0 && head_dim / 32 <= 32, "head_dim<=1024, %32==0");
    TORCH_CHECK(head_dim % THREADS == 0, "head_dim % 256 == 0");
    torch::Tensor output = torch::empty_like(Q);
    const dim3 grid(batch * heads, seq_len / BAND_R);
    const size_t shared_bytes = static_cast<size_t>(BAND_R * seq_len + THREADS) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    sdpa_band_kernel<<<grid, THREADS, shared_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), seq_len, head_dim, scale);
    return output;
}
"""

sdpa_extension = load_inline(
    name="level1_problem97_sdpa_band_musa",
    cpp_sources=sdpa_source,
    functions=["sdpa_band"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return sdpa_extension.sdpa_band(Q, K, V)
