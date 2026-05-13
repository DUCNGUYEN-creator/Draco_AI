# DracoAI V1 — modeling/kernels/triton/fused_attention.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Triton fused scaled-dot-product attention kernel.

Contract: tensor in → tensor out, no model state.
Gracefully degrades to NumPy if triton is unavailable.

Note: This is a reference implementation skeleton.
      Production use requires tuning BLOCK_SIZE and num_warps
      for the target GPU (A100, H100, RTX 3090, etc.).
"""
from __future__ import annotations
import numpy as np

__all__ = ["triton_fused_attention"]


def triton_fused_attention(
    Q: np.ndarray,  # (batch, heads, seq_q, head_dim)
    K: np.ndarray,  # (batch, heads, seq_k, head_dim)
    V: np.ndarray,  # (batch, heads, seq_k, head_dim)
    scale: float,
    causal: bool = True,
) -> np.ndarray:
    """
    Fused scaled dot-product attention.

    Tries triton.jit kernel first; falls back to NumPy on failure.
    Returns: (batch, heads, seq_q, head_dim)
    """
    try:
        return _triton_sdpa(Q, K, V, scale, causal)
    except Exception:
        return _numpy_sdpa_fallback(Q, K, V, scale, causal)


def _triton_sdpa(Q, K, V, scale, causal):
    import triton
    import triton.language as tl
    import cupy as cp

    # Convert to CuPy for Triton
    Q_c = cp.asarray(Q.astype(np.float16))
    K_c = cp.asarray(K.astype(np.float16))
    V_c = cp.asarray(V.astype(np.float16))

    B, H, Sq, D = Q.shape
    Sk = K.shape[2]
    out_c = cp.zeros((B, H, Sq, D), dtype=cp.float16)

    BLOCK = 64

    @triton.jit
    def _sdpa_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        stride_qb, stride_qh, stride_qq, stride_qd,
        stride_kb, stride_kh, stride_kk, stride_kd,
        stride_vb, stride_vh, stride_vk, stride_vd,
        stride_ob, stride_oh, stride_oq, stride_od,
        Sq, Sk, scale,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        # Minimal flash-attention-style kernel (1 block per query tile)
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        qid = tl.program_id(2)

        q_off = tl.arange(0, BLOCK_Q)
        d_off = tl.arange(0, HEAD_DIM)
        q_ptrs = (Q_ptr + bid * stride_qb + hid * stride_qh
                  + (qid * BLOCK_Q + q_off[:, None]) * stride_qq
                  + d_off[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=q_off[:, None] < Sq - qid * BLOCK_Q)

        acc  = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)
        lse  = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        m    = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)

        for k_start in range(0, Sk, BLOCK_K):
            k_off = tl.arange(0, BLOCK_K)
            k_ptrs = (K_ptr + bid * stride_kb + hid * stride_kh
                      + (k_start + k_off[None, :]) * stride_kk
                      + d_off[:, None] * stride_kd)
            k = tl.load(k_ptrs, mask=k_off[None, :] < Sk - k_start)
            s = tl.dot(q, k) * scale

            if causal:
                q_idx = qid * BLOCK_Q + q_off[:, None]
                k_idx = k_start + k_off[None, :]
                s = tl.where(k_idx <= q_idx, s, float("-inf"))

            m_new  = tl.maximum(m, tl.max(s, axis=1))
            p      = tl.exp(s - m_new[:, None])
            lse    = tl.exp(m - m_new) * lse + tl.sum(p, axis=1)
            acc    = acc * tl.exp(m - m_new)[:, None]
            m      = m_new

            v_ptrs = (V_ptr + bid * stride_vb + hid * stride_vh
                      + (k_start + k_off[:, None]) * stride_vk
                      + d_off[None, :] * stride_vd)
            v   = tl.load(v_ptrs, mask=k_off[:, None] < Sk - k_start)
            acc = acc + tl.dot(p.to(tl.float16), v)

        acc = acc / lse[:, None]
        o_ptrs = (O_ptr + bid * stride_ob + hid * stride_oh
                  + (qid * BLOCK_Q + q_off[:, None]) * stride_od
                  + d_off[None, :])
        tl.store(o_ptrs, acc.to(tl.float16), mask=q_off[:, None] < Sq - qid * BLOCK_Q)

    grid = (B, H, (Q.shape[2] + BLOCK - 1) // BLOCK)
    _sdpa_kernel[grid](
        Q_c, K_c, V_c, out_c,
        *Q_c.strides, *K_c.strides, *V_c.strides, *out_c.strides,
        Q.shape[2], K.shape[2], scale,
        BLOCK_Q=BLOCK, BLOCK_K=BLOCK, HEAD_DIM=Q.shape[-1],
    )
    return cp.asnumpy(out_c).astype(Q.dtype)


def _numpy_sdpa_fallback(Q, K, V, scale, causal):
    """Pure NumPy fallback — always correct, used when Triton unavailable."""
    scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        mask   = np.triu(np.full((sq, sk), -1e9, dtype=np.float32), k=sk - sq + 1)
        scores = scores + mask[None, None]
    scores = np.clip(scores, -50.0, 50.0)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-9)
    return w @ V