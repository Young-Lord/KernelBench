"""Level 3 / problem 44 MiniGPTBlock MUSA implementation.

The full GPT block runs on MUSA; the causal self-attention core is the same
hand-written FMHA kernel used for problem 43 (self-contained copy).
"""

import math

import torch
import torch.nn as nn

from kernelbench.musa_extension import load_inline

FMHA_SOURCE = r"""#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

// Tuned FMHA forward (MUSA / MTT S4000, mp_22, warp = 128).
//
// Instead of one CTA per query row (huge grid, redundant K/V traffic, one
// full-block tree reduction per row), each CTA owns a *band of BM query rows*
// of one (batch, head) and keeps their causal scores in shared memory:
//   - BM rows are reduced in lockstep by fixed 32-thread row groups, so the
//     whole softmax needs only O(log2(32)) barriers instead of several full
//     block reductions *per row*;
//   - the Q tile is read once for all BM rows and the score pass reuses it;
//   - masked (upper-triangle) entries are kept at -inf in softmax mode (the
//     reference semantics) and dropped to 0 in relu mode.
// Numerics remain fp32 max-subtraction softmax over the causal window.
// mode: 0 = softmax, 1 = relu. causal: restrict j <= row.

#define FMHA_BM 8
#define FMHA_GROUP 32

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
    const int band = blockIdx.y;
    const int tid = threadIdx.x;
    const int T = seq_len;
    const int d = head_dim;
    const int row0 = band * FMHA_BM;
    const int r = tid / FMHA_GROUP;        // 0..FMHA_BM-1 row inside band
    const int g = tid % FMHA_GROUP;        // 0..31 key/col lane for this row

    extern __shared__ float shared_mem[];
    float* q_tile = shared_mem;                 // FMHA_BM * d
    float* scores = q_tile + FMHA_BM * d;       // FMHA_BM * T
    float* red = scores + FMHA_BM * T;          // FMHA_BM * FMHA_GROUP

    const int64_t bh_offset = (static_cast<int64_t>(b) * heads + h) * T * d;
    const float* q_bh = Q + bh_offset;
    const float* k_bh = K + bh_offset;
    const float* v_bh = V + bh_offset;

    // Load this band's Q rows.
    for (int idx = tid; idx < FMHA_BM * d; idx += blockDim.x) {
        int rr = idx / d;
        q_tile[rr * d + (idx % d)] = q_bh[(row0 + rr) * d + (idx % d)];
    }
    __syncthreads();

    // scores[r][j] = scale * (q_row . k[j]); causal mask -> -FLT_MAX.
    {
        const int row = row0 + r;
        const float* qr = q_tile + r * d;
        for (int j = g; j < T; j += FMHA_GROUP) {
            if (causal && j > row) {
                scores[r * T + j] = -FLT_MAX;
            } else {
                const float* k_j = k_bh + j * d;
                float s = 0.0f;
                for (int i = 0; i < d; ++i) s += qr[i] * k_j[i];
                scores[r * T + j] = s * scale;
            }
        }
    }
    __syncthreads();

    if (mode == 0) {
        // Row-group max reduction.
        float local_max = -FLT_MAX;
        for (int j = g; j < T; j += FMHA_GROUP) {
            if (scores[r * T + j] > -FLT_MAX / 2.0f)
                local_max = fmaxf(local_max, scores[r * T + j]);
        }
        red[r * FMHA_GROUP + g] = local_max;
        __syncthreads();
        for (int stride = FMHA_GROUP / 2; stride > 0; stride >>= 1) {
            if (g < stride)
                red[r * FMHA_GROUP + g] = fmaxf(red[r * FMHA_GROUP + g],
                                                red[r * FMHA_GROUP + g + stride]);
            __syncthreads();
        }
        const float row_max = red[r * FMHA_GROUP];

        float local_sum = 0.0f;
        for (int j = g; j < T; j += FMHA_GROUP) {
            float e = scores[r * T + j] > -FLT_MAX / 2.0f
                          ? expf(scores[r * T + j] - row_max)
                          : 0.0f;
            scores[r * T + j] = e;
            local_sum += e;
        }
        red[r * FMHA_GROUP + g] = local_sum;
        __syncthreads();
        for (int stride = FMHA_GROUP / 2; stride > 0; stride >>= 1) {
            if (g < stride)
                red[r * FMHA_GROUP + g] += red[r * FMHA_GROUP + g + stride];
            __syncthreads();
        }
        const float row_sum = red[r * FMHA_GROUP];

        for (int j = g; j < T; j += FMHA_GROUP) {
            scores[r * T + j] = row_sum > 0.0f ? scores[r * T + j] / row_sum : 0.0f;
        }
        __syncthreads();
    } else {
        // ReLU attention: relu(masked scores) == 0 on the masked triangle.
        for (int j = g; j < T; j += FMHA_GROUP) {
            float s = scores[r * T + j];
            scores[r * T + j] = s > -FLT_MAX / 2.0f ? fmaxf(s, 0.0f) : 0.0f;
        }
        __syncthreads();
    }

    // Output: row group computes the output columns it owns.
    {
        const float* srow = scores + r * T;
        float* orow = O + bh_offset + (row0 + r) * d;
        for (int c = g; c < d; c += FMHA_GROUP) {
            float acc = 0.0f;
            for (int j = 0; j < T; ++j) {
                float w = srow[j];
                if (w != 0.0f) acc += w * v_bh[j * d + c];
            }
            orow[c] = acc;
        }
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

    torch::Tensor output = torch::empty_like(Q);
    const int threads = FMHA_BM * FMHA_GROUP;  // 256
    const dim3 grid(batch * heads, (seq_len + FMHA_BM - 1) / FMHA_BM);
    const size_t shared_bytes =
        (static_cast<size_t>(FMHA_BM) * (head_dim + seq_len) +
         FMHA_BM * FMHA_GROUP + FMHA_BM) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    fmha_fwd_kernel<<<grid, threads, shared_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), batch, heads, seq_len, head_dim, scale,
        static_cast<int>(mode), static_cast<int>(causal));
    return output;
}
"""

_fmha_ext = load_inline(
    name="level3_fmha_musa_p44",
    cpp_sources=FMHA_SOURCE,
    functions=["fmha_fwd"],
    verbose=False,
)


def _fmha(q, k, v, mode, causal):
    return _fmha_ext.fmha_fwd(q.contiguous(), k.contiguous(), v.contiguous(), mode, causal)


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Causal multi-head masked self-attention (minGPT) with the attention
    core (scores -> softmax -> weighted sum) executed by the MUSA kernel."""

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
        q, k, v = self.c_attn(x).split(C, dim=2)
        hs = C // self.n_head
        qh = q.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        kh = k.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        vh = v.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        y = _fmha(qh, kh, vh, 0, True)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ModelNew(nn.Module):
    """An unassuming Transformer block: LayerNorm + causal attention (MUSA
    kernel), LayerNorm + MLP."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
