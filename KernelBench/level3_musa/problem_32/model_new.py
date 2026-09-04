"""Level 3 / problem 32 ConvolutionalVisionTransformer MUSA implementation.

Module construction mirrors the PyTorch reference exactly (conv stem,
flatten projection, six independent transformer-encoder layers, [CLS] token,
classification head) so the random weights are identical under the eval seed.
The encoder self-attention core is replaced by the hand-written MUSA FMHA
kernel; every other op matches the reference op-for-op.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelbench.musa_extension import load_inline

FMHA_SOURCE = r"""#include <torch/extension.h>
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

// Fused causal/non-causal multi-head attention forward.
// Q/K/V are contiguous (B, H, T, hs); O is written as (B, H, T, hs).
// mode: 0 = softmax attention, 1 = relu attention (no normalization).
// causal: upper triangle is -inf (softmax) / 0 (relu).
__global__ void fmha_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    int mode,
    int causal) {
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh % heads;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int T = seq_len;
    const int d = head_dim;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;                // T floats
    float* q_row = shared_mem + T;             // d floats
    float* reduce_buf = shared_mem + T + d;    // blockDim floats

    const int64_t bh_offset = (static_cast<int64_t>(b) * heads + h) * T * d;
    const float* q_bh = Q + bh_offset;
    const float* k_bh = K + bh_offset;
    const float* v_bh = V + bh_offset;

    // Load this query row into shared memory.
    for (int i = tid; i < d; i += blockDim.x) {
        q_row[i] = q_bh[row * d + i];
    }
    __syncthreads();

    // scores[j] = (q_row . k[j]) * scale ; masked positions for this row.
    const int j_lim = causal ? (row + 1) : T;
    for (int j = tid; j < T; j += blockDim.x) {
        if (j < j_lim) {
            const float* k_j = k_bh + j * d;
            float acc = 0.0f;
            for (int i = 0; i < d; ++i) {
                acc += q_row[i] * k_j[i];
            }
            scores[j] = acc * scale;
        } else {
            scores[j] = -FLT_MAX;
        }
    }
    __syncthreads();

    if (mode == 0) {
        // Softmax with max subtraction over the unmasked (finite) scores.
        float local_max = -FLT_MAX;
        for (int j = tid; j < T; j += blockDim.x) {
            if (scores[j] > -FLT_MAX / 2.0f) {
                local_max = fmaxf(local_max, scores[j]);
            }
        }
        float row_max = block_reduce_max(local_max, reduce_buf);
        __syncthreads();

        float local_sum = 0.0f;
        for (int j = tid; j < T; j += blockDim.x) {
            float e = scores[j] > -FLT_MAX / 2.0f ? expf(scores[j] - row_max) : 0.0f;
            scores[j] = e;
            local_sum += e;
        }
        float row_sum = block_reduce_sum(local_sum, reduce_buf);
        __syncthreads();

        for (int j = tid; j < T; j += blockDim.x) {
            scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
        }
        __syncthreads();
    } else {
        // ReLU attention: relu(score), masked entries stay zero.
        for (int j = tid; j < T; j += blockDim.x) {
            scores[j] = scores[j] > -FLT_MAX / 2.0f ? fmaxf(scores[j], 0.0f) : 0.0f;
        }
        __syncthreads();
    }

    // O[row, c] = sum_j scores[j] * V[j, c]
    float* o_row = O + bh_offset + row * d;
    for (int c = tid; c < d; c += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < T; ++j) {
            float s = scores[j];
            if (s != 0.0f) {
                acc += s * v_bh[j * d + c];
            }
        }
        o_row[c] = acc;
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
    const dim3 grid(batch * heads, seq_len);
    const int threads = 256;
    const size_t shared_bytes =
        static_cast<size_t>(seq_len + head_dim + threads) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (seq_len + head_dim + threads > 48 * 1024 / (int)sizeof(float)) {
        TORCH_CHECK(false, "shared memory request exceeds 48KB");
    }

    fmha_fwd_kernel<<<grid, threads, shared_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), batch, heads, seq_len, head_dim, scale,
        static_cast<int>(mode), static_cast<int>(causal));
    return output;
}"""

_fmha_ext = load_inline(
    name="level3_fmha_musa_p32",
    cpp_sources=FMHA_SOURCE,
    functions=["fmha_fwd"],
    verbose=False,
)


def _fmha(q, k, v, causal):
    return _fmha_ext.fmha_fwd(q.contiguous(), k.contiguous(), v.contiguous(), 0, causal)


def _mha_attn(x, layer):
    """x is seq-first (T, N, E) as TransformerEncoderLayer sees it internally
    even when constructed with batch_first=True."""
    T, N, E = x.shape
    heads = layer.self_attn.num_heads
    hs = E // heads
    w = layer.self_attn.in_proj_weight
    b = layer.self_attn.in_proj_bias
    qkv = F.linear(x, w, b)
    q, k, v = qkv.chunk(3, dim=-1)
    qh = q.view(T, N, heads, hs).permute(1, 2, 0, 3).contiguous()
    kh = k.view(T, N, heads, hs).permute(1, 2, 0, 3).contiguous()
    vh = v.view(T, N, heads, hs).permute(1, 2, 0, 3).contiguous()
    yh = _fmha(qh, kh, vh, False)
    y = yh.permute(2, 0, 1, 3).contiguous().view(T, N, E)
    return F.linear(y, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)


def _encoder_layer(x, layer):
    x = x + layer.dropout1(_mha_attn(x, layer))
    x = layer.norm1(x)
    x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = x + layer.dropout2(x2)
    return layer.norm2(x)


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = self.linear_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)
        x = x.transpose(0, 1)  # seq-first for our layer math
        for layer in self.transformer_layers:
            x = _encoder_layer(x, layer)
        x = x.transpose(0, 1)
        return self.fc_out(x[:, 0])
