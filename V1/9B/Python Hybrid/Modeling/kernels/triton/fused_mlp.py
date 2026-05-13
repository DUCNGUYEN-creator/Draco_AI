# DracoAI V1 — modeling/kernels/triton/fused_mlp.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Triton fused SwiGLU MLP kernel.
gate = silu(x @ W_g)
output = (gate * (x @ W_u)) @ W_d
"""
from __future__ import annotations
import numpy as np

__all__ = ["triton_fused_swiglu"]


def triton_fused_swiglu(
    x:   np.ndarray,   # (batch_or_seq, d_model)
    W_g: np.ndarray,   # (d_model, d_ff)
    W_u: np.ndarray,   # (d_model, d_ff)
    W_d: np.ndarray,   # (d_ff, d_model)
) -> np.ndarray:
    """Fused SwiGLU forward. Falls back to NumPy if Triton unavailable."""
    try:
        return _triton_swiglu(x, W_g, W_u, W_d)
    except Exception:
        return _numpy_swiglu_fallback(x, W_g, W_u, W_d)


def _triton_swiglu(x, W_g, W_u, W_d):
    import triton
    import triton.language as tl
    import cupy as cp

    x_c   = cp.asarray(x.astype(np.float16))
    Wg_c  = cp.asarray(W_g.astype(np.float16))
    Wu_c  = cp.asarray(W_u.astype(np.float16))
    Wd_c  = cp.asarray(W_d.astype(np.float16))

    seq, d   = x.shape
    d_ff     = W_g.shape[1]
    gate_c   = cp.zeros((seq, d_ff), dtype=cp.float16)
    up_c     = cp.zeros((seq, d_ff), dtype=cp.float16)
    BLOCK    = 128

    @triton.jit
    def _gate_up_kernel(
        X_ptr, Wg_ptr, Wu_ptr, G_ptr, U_ptr,
        seq, d_in, d_ff,
        BLOCK_SEQ: tl.constexpr, BLOCK_FF: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK_FF + tl.arange(0, BLOCK_FF)
        d_off = tl.arange(0, d_in) if d_in <= 256 else tl.arange(0, 256)

        x_row = tl.load(X_ptr + row * d_in + d_off, mask=d_off < d_in)
        wg_col = tl.load(Wg_ptr + d_off[:, None] * d_ff + col[None, :],
                         mask=(d_off[:, None] < d_in) & (col[None, :] < d_ff))
        wu_col = tl.load(Wu_ptr + d_off[:, None] * d_ff + col[None, :],
                         mask=(d_off[:, None] < d_in) & (col[None, :] < d_ff))

        g_val = tl.sum(x_row[:, None] * wg_col, axis=0)
        u_val = tl.sum(x_row[:, None] * wu_col, axis=0)
        # SiLU: g * sigmoid(g)
        g_act = g_val * tl.sigmoid(g_val.to(tl.float32)).to(tl.float16)

        tl.store(G_ptr + row * d_ff + col, g_act, mask=col < d_ff)
        tl.store(U_ptr + row * d_ff + col, u_val, mask=col < d_ff)

    grid = (seq, (d_ff + BLOCK - 1) // BLOCK)
    _gate_up_kernel[grid](x_c, Wg_c, Wu_c, gate_c, up_c,
                          seq, d, d_ff,
                          BLOCK_SEQ=1, BLOCK_FF=BLOCK)

    fused_c = gate_c * up_c
    out_c   = fused_c @ Wd_c
    return cp.asnumpy(out_c).astype(x.dtype)


def _numpy_swiglu_fallback(x, W_g, W_u, W_d):
    gate = x @ W_g
    gate = gate / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
    return (gate * (x @ W_u)) @ W_d