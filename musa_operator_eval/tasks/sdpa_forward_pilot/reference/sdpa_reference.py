"""Independent, blocked NumPy reference for the SDPA forward contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _visibility_mask(s_q: int, s_kv: int, *, causal: bool, window_left: int, window_right: int, explicit_mask: np.ndarray | None) -> np.ndarray:
    q_index = np.arange(s_q, dtype=np.int64)[:, None]
    kv_index = np.arange(s_kv, dtype=np.int64)[None, :]
    visible = np.ones((s_q, s_kv), dtype=bool)
    if causal:
        visible &= kv_index <= q_index
    if window_left != -1:
        visible &= kv_index >= q_index - window_left
    if window_right != -1:
        visible &= kv_index <= q_index + window_right
    if explicit_mask is not None:
        visible &= np.asarray(explicit_mask, dtype=bool)
    return visible


def sdpa_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | None = None,
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    explicit_mask: np.ndarray | None = None,
    query_block: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BHSD SDPA in float64 and return `(output, lse)`.

    Causal alignment is top-left. GQA is supported when H_q is an integer
    multiple of H_kv. Fully masked rows produce zero output and `-inf` LSE.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must use BHSD rank-4 tensors")
    b, h_q, s_q, d = q.shape
    if k.shape != v.shape or k.shape[0] != b or k.shape[3] != d:
        raise ValueError("k/v shapes must match and agree with q batch/head_dim")
    h_kv, s_kv = k.shape[1], k.shape[2]
    if h_q % h_kv:
        raise ValueError("H_q must be divisible by H_kv for GQA")
    if window_left < -1 or window_right < -1:
        raise ValueError("window bounds must be -1 or non-negative")
    scale_value = float(scale if scale is not None else 1.0 / math.sqrt(d))
    group = h_q // h_kv
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.repeat(np.asarray(k, dtype=np.float64), group, axis=1)
    v64 = np.repeat(np.asarray(v, dtype=np.float64), group, axis=1)
    visible = _visibility_mask(s_q, s_kv, causal=causal, window_left=window_left, window_right=window_right, explicit_mask=explicit_mask)
    output = np.zeros((b, h_q, s_q, d), dtype=np.float64)
    lse = np.full((b, h_q, s_q), -np.inf, dtype=np.float64)
    for start in range(0, s_q, query_block):
        stop = min(start + query_block, s_q)
        scores = np.einsum("bhqd,bhkd->bhqk", q64[:, :, start:stop], k64) * scale_value
        block_visible = visible[start:stop]
        scores = np.where(block_visible[None, None, :, :], scores, -np.inf)
        row_has_value = block_visible.any(axis=-1)
        safe_scores = np.where(row_has_value[None, None, :, None], scores, 0.0)
        row_max = safe_scores.max(axis=-1, keepdims=True)
        exponentials = np.where(block_visible[None, None, :, :], np.exp(safe_scores - row_max), 0.0)
        denominator = exponentials.sum(axis=-1, keepdims=True)
        probabilities = np.divide(exponentials, denominator, out=np.zeros_like(exponentials), where=denominator != 0)
        output[:, :, start:stop] = np.einsum("bhqk,bhkd->bhqd", probabilities, v64)
        block_lse = np.log(denominator[..., 0], where=denominator[..., 0] > 0, out=np.full_like(denominator[..., 0], -np.inf)) + row_max[..., 0]
        lse[:, :, start:stop] = np.where(row_has_value[None, None, :], block_lse, -np.inf)
    return output, lse
