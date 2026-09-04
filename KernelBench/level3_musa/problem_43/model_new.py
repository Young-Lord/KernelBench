import os
import math

import torch
import torch.nn as nn

from kernelbench.musa_extension import load_inline

# ---------------------------------------------------------------------------
# FMHA forward kernel (MUSA / MTT S4000, mp_22).
#
# Design reference (musa_attention_set archive, which is NOT co-located in the
# kernel source for self-containment):
#   - llama.cpp `fattn.cu` (ggml FlashAttention template): one CTA owns a query
#     row, scores are materialized in shared memory, online max/exp normalization
#     matches a numerically stable softmax.
#   - `level1/97` S4000 SDPA reference uses the same two-pass max-subtract
#     softmax + block tree reductions, which is proven to match `F.softmax`
#     within KernelBench fp32 tolerance on this card.
#
# This kernel keeps the causal upper triangle at -inf (softmax mode) exactly
# like the reference `masked_fill(self.bias == 0, -inf)`, and zeroes it in relu
# mode to reproduce `F.relu(masked_fill(...))`.  Each CTA handles one query row
# of one (batch, head) pair.  `head_dim` is runtime, so this works for the whole
# GPT family (hs = 64/96) as well as the vision problems (hs = 32/64).
# ---------------------------------------------------------------------------

FMHA_SOURCE = r"""#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

// Flash-style tiled forward attention (MUSA / MTT S4000, mp_22).
//
// One CTA owns a band of FMHA_TILE_BM query rows of a single (batch, head)
// pair and streams K/V through shared memory in FMHA_TILE_BN-key tiles, so the
// K/V tile is read once per band instead of once per query row and no full
// score row is ever materialized.
//
// Softmax mode (mode 0) is a true online softmax: running per-row max/sum with
// an O rescale every tile. ReLU mode (mode 1, DeepSeek-style relu attention)
// has no denominator and is just a streaming weighted sum of relu(logits).
// Causal masking only keeps keys j <= query row, matching the reference
// masked_fill(..., -inf) / relu(masked_fill(...)) semantics.
//
// Layout notes: shared arrays use a per-row padded stride sd = head_dim + 1
// (and BP = FMHA_TILE_BN + 1 for the score tile) so lanes reading different
// rows/keys of the same column do not land in the same shared bank
// (head_dim % 32 == 0 would otherwise serialize them). Each query row is owned
// by FMHA_TILE_ROW_LANES lanes; a lane always writes the same O columns, so the
// O accumulator needs no cross-lane traffic.

#define FMHA_TILE_BM 32
#define FMHA_TILE_ROW_LANES 32
#define FMHA_TILE_BN 32
#define FMHA_TILE_BP (FMHA_TILE_BN + 1)

__global__ void fmha_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch, int heads, int seq_len, int head_dim,
    float scale, int mode, int causal) {
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh % heads;
    const int q_band = blockIdx.y;
    const int tid = threadIdx.x;
    const int T = seq_len;
    const int d = head_dim;
    const int sd = d + 1;                       // padded per-row stride
    const int threads = FMHA_TILE_BM * FMHA_TILE_ROW_LANES;
    const int row0 = q_band * FMHA_TILE_BM;
    const int r = tid / FMHA_TILE_ROW_LANES;    // query row inside the band
    const int lane = tid % FMHA_TILE_ROW_LANES; // lane group of the row

    extern __shared__ float smem[];
    float* q_s = smem;                                 // BM * sd
    float* k_s = q_s + FMHA_TILE_BM * sd;              // BN * sd
    float* v_s = k_s + FMHA_TILE_BN * sd;              // BN * sd
    float* s_s = v_s + FMHA_TILE_BN * sd;              // BM * BP (scores / probs)
    float* o_s = s_s + FMHA_TILE_BM * FMHA_TILE_BP;    // BM * sd (accumulator)
    float* red = o_s + FMHA_TILE_BM * sd;              // BM * ROW_LANES

    const int64_t bh_offset = (static_cast<int64_t>(b) * heads + h) * T * d;
    const float* q_bh = Q + bh_offset;
    const float* k_bh = K + bh_offset;
    const float* v_bh = V + bh_offset;
    float* o_bh = O + bh_offset;

    const int row_global = row0 + r;

    for (int idx = tid; idx < FMHA_TILE_BM * d; idx += threads) {
        q_s[(idx / d) * sd + (idx % d)] = q_bh[(row0 + idx / d) * d + (idx % d)];
        o_s[(idx / d) * sd + (idx % d)] = 0.0f;
    }
    __syncthreads();

    // Keys a band can ever touch: causal bands stop at the last row of the
    // band, dense bands scan the whole sequence. Uniform across the block so
    // every thread takes the same barriers.
    const int kv_end = causal ? (row0 + FMHA_TILE_BM < T ? row0 + FMHA_TILE_BM : T)
                              : T;

    if (mode == 0) {
        float row_max = -FLT_MAX;
        float row_sum = 0.0f;
        for (int j0 = 0; j0 < kv_end; j0 += FMHA_TILE_BN) {
            const int tile_keys = (kv_end - j0) < FMHA_TILE_BN ? (kv_end - j0)
                                                               : FMHA_TILE_BN;
            // This row only attends to keys <= row_global.
            const int valid_n = causal && (row_global + 1 - j0) < tile_keys
                                    ? (row_global + 1 - j0) : tile_keys;

            for (int idx = tid; idx < tile_keys * d; idx += threads) {
                const int jj = idx / d;
                const int c = idx % d;
                k_s[jj * sd + c] = k_bh[(j0 + jj) * d + c];
                v_s[jj * sd + c] = v_bh[(j0 + jj) * d + c];
            }
            __syncthreads();

            float local_max = -FLT_MAX;
            const float* q_row = q_s + r * sd;
            for (int jj = lane; jj < valid_n; jj += FMHA_TILE_ROW_LANES) {
                const float* k_row = k_s + jj * sd;
                float acc = 0.0f;
                for (int i = 0; i < d; ++i) acc += q_row[i] * k_row[i];
                const float s = acc * scale;
                s_s[r * FMHA_TILE_BP + jj] = s;
                local_max = fmaxf(local_max, s);
            }

            red[r * FMHA_TILE_ROW_LANES + lane] = local_max;
            __syncthreads();
            for (int stride = FMHA_TILE_ROW_LANES / 2; stride > 0; stride >>= 1) {
                if (lane < stride)
                    red[r * FMHA_TILE_ROW_LANES + lane] = fmaxf(
                        red[r * FMHA_TILE_ROW_LANES + lane],
                        red[r * FMHA_TILE_ROW_LANES + lane + stride]);
                __syncthreads();
            }
            const float new_max = fmaxf(row_max, red[r * FMHA_TILE_ROW_LANES]);
            const float rescale = expf(row_max - new_max);  // 0 on the first tile
            row_max = new_max;

            float local_sum = 0.0f;
            for (int jj = lane; jj < valid_n; jj += FMHA_TILE_ROW_LANES) {
                const float p = expf(s_s[r * FMHA_TILE_BP + jj] - row_max);
                s_s[r * FMHA_TILE_BP + jj] = p;
                local_sum += p;
            }
            red[r * FMHA_TILE_ROW_LANES + lane] = local_sum;
            __syncthreads();
            for (int stride = FMHA_TILE_ROW_LANES / 2; stride > 0; stride >>= 1) {
                if (lane < stride)
                    red[r * FMHA_TILE_ROW_LANES + lane] +=
                        red[r * FMHA_TILE_ROW_LANES + lane + stride];
                __syncthreads();
            }
            const float tile_sum = red[r * FMHA_TILE_ROW_LANES];
            row_sum = row_sum * rescale + tile_sum;

            // Rescale the O accumulator and add this tile's contribution. Each
            // lane owns the columns c = lane (mod ROW_LANES).
            for (int c = lane; c < d; c += FMHA_TILE_ROW_LANES)
                o_s[r * sd + c] *= rescale;
            for (int jj = 0; jj < valid_n; ++jj) {
                const float p = s_s[r * FMHA_TILE_BP + jj];
                if (p == 0.0f) continue;
                const float* v_row = v_s + jj * sd;
                for (int c = lane; c < d; c += FMHA_TILE_ROW_LANES)
                    o_s[r * sd + c] += p * v_row[c];
            }
            __syncthreads();  // tile fully consumed before shared reuse
        }
        if (row_sum > 0.0f) {
            for (int c = lane; c < d; c += FMHA_TILE_ROW_LANES)
                o_bh[row_global * d + c] = o_s[r * sd + c] / row_sum;
        }
    } else {
        // ReLU attention: out += relu(scale * q.k) * v over the causal window.
        for (int j0 = 0; j0 < kv_end; j0 += FMHA_TILE_BN) {
            const int tile_keys = (kv_end - j0) < FMHA_TILE_BN ? (kv_end - j0)
                                                               : FMHA_TILE_BN;
            const int valid_n = causal && (row_global + 1 - j0) < tile_keys
                                    ? (row_global + 1 - j0) : tile_keys;

            for (int idx = tid; idx < tile_keys * d; idx += threads) {
                const int jj = idx / d;
                const int c = idx % d;
                k_s[jj * sd + c] = k_bh[(j0 + jj) * d + c];
                v_s[jj * sd + c] = v_bh[(j0 + jj) * d + c];
            }
            __syncthreads();

            const float* q_row = q_s + r * sd;
            for (int jj = lane; jj < valid_n; jj += FMHA_TILE_ROW_LANES) {
                const float* k_row = k_s + jj * sd;
                float acc = 0.0f;
                for (int i = 0; i < d; ++i) acc += q_row[i] * k_row[i];
                const float s = acc * scale;
                s_s[r * FMHA_TILE_BP + jj] = fmaxf(s, 0.0f);
            }
            __syncthreads();

            for (int jj = 0; jj < valid_n; ++jj) {
                const float p = s_s[r * FMHA_TILE_BP + jj];
                if (p == 0.0f) continue;
                const float* v_row = v_s + jj * sd;
                for (int c = lane; c < d; c += FMHA_TILE_ROW_LANES)
                    o_s[r * sd + c] += p * v_row[c];
            }
            __syncthreads();
        }
        for (int c = lane; c < d; c += FMHA_TILE_ROW_LANES)
            o_bh[row_global * d + c] = o_s[r * sd + c];
    }
}

torch::Tensor fmha_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int64_t mode,
    bool causal) {
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "inputs must be (B,H,T,hs)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "shape mismatch");

    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seq_len = static_cast<int>(Q.size(2));
    const int head_dim = static_cast<int>(Q.size(3));
    TORCH_CHECK(seq_len % FMHA_TILE_BM == 0, "seq_len must be a multiple of FMHA_TILE_BM");

    torch::Tensor output = torch::empty_like(Q);
    const int threads = FMHA_TILE_BM * FMHA_TILE_ROW_LANES;
    const dim3 grid(batch * heads, seq_len / FMHA_TILE_BM);
    const int sd = head_dim + 1;
    const size_t shared_bytes =
        (static_cast<size_t>(FMHA_TILE_BM) * sd * 2 +          // q_s + o_s
         static_cast<size_t>(2) * FMHA_TILE_BN * sd +          // k_s + v_s
         static_cast<size_t>(FMHA_TILE_BM) * FMHA_TILE_BP +    // s_s
         static_cast<size_t>(FMHA_TILE_BM) * FMHA_TILE_ROW_LANES) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    fmha_fwd_kernel<<<grid, threads, shared_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), batch, heads, seq_len, head_dim, scale,
        static_cast<int>(mode), static_cast<int>(causal));
    return output;
}
"""

_fmha_ext = load_inline(
    name="level3_fmha_musa",
    cpp_sources=FMHA_SOURCE,
    functions=["fmha_fwd"],
    verbose=False,
)


def _fmha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mode: int, causal: bool) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    return _fmha_ext.fmha_fwd(q, k, v, mode, causal)


def _causal_attention(x: torch.Tensor, n_head: int, causal: bool = True) -> torch.Tensor:
    """Split a (B,T,3C) fused projection into Q/K/V heads, run the FMHA kernel,
    then merge heads back to (B,T,C)."""
    B, T, C = x.shape
    embed = C // 3
    hs = embed // n_head
    q, k, v = x.split(embed, dim=2)
    qh = q.view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    kh = k.view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    vh = v.view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    yh = _fmha(qh, kh, vh, 0, causal)
    y = yh.permute(0, 2, 1, 3).contiguous().view(B, T, embed)
    return y


class ModelNew(nn.Module):
    """Vanilla causal multi-head masked self-attention (minGPT), with the
    attention itself running on a hand-written MUSA kernel."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3C)
        y = _causal_attention(qkv, self.n_head, causal=True)
        y = self.resid_dropout(self.c_proj(y))
        return y
