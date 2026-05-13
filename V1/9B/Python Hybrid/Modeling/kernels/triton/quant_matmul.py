# DracoAI V1 — modeling/kernels/triton/quant_matmul.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Triton INT8 / INT4 weight-only matmul kernels.
Activations are float16; weights are int8 or packed int4.
"""
from __future__ import annotations
import numpy as np

__all__ = ["triton_int8_matmul", "triton_int4_matmul"]


def triton_int8_matmul(
    x:     np.ndarray,   # (..., in_feat) float16/float32
    W_q:   np.ndarray,   # (out_feat, in_feat) int8
    scale: np.ndarray,   # (out_feat,) float32
) -> np.ndarray:
    """
    INT8 weight-only matmul: output = x @ (W_q * scale[:, None]).T
    Falls back to NumPy dequantize if Triton unavailable.
    """
    try:
        return _triton_i8mm(x, W_q, scale)
    except Exception:
        W_f = W_q.astype(np.float32) * scale[:, None]
        return x.astype(np.float32) @ W_f.T


def triton_int4_matmul(
    x:     np.ndarray,   # (..., usable) float16/float32
    W_q:   np.ndarray,   # (out_feat, n_packed) uint8
    scale: np.ndarray,   # (out_feat, n_groups) float32
    zero:  np.ndarray,   # (out_feat, n_groups) float32
    group_size: int = 128,
) -> np.ndarray:
    """
    INT4 weight-only matmul with asymmetric per-group dequantization.
    Falls back to NumPy if Triton unavailable.
    """
    try:
        return _triton_i4mm(x, W_q, scale, zero, group_size)
    except Exception:
        return _numpy_i4mm_fallback(x, W_q, scale, zero, group_size)


def _triton_i8mm(x, W_q, scale):
    """Minimal Triton INT8 kernel — uses tl.dot with int8 inputs."""
    import triton
    import triton.language as tl
    import cupy as cp

    x_c = cp.asarray(x.reshape(-1, x.shape[-1]).astype(np.float16))
    Wq_c = cp.asarray(W_q)
    sc_c = cp.asarray(scale)
    M, K = x_c.shape
    N    = W_q.shape[0]
    out_c = cp.zeros((M, N), dtype=cp.float32)
    BLOCK = 64

    @triton.jit
    def _i8mm_kernel(
        X, W, S, O,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            rk = k + tl.arange(0, BLOCK_K)
            x_tile  = tl.load(X + rm[:, None] * K + rk[None, :],
                               mask=(rm[:, None] < M) & (rk[None, :] < K))
            w_tile  = tl.load(W + rn[:, None] * K + rk[None, :],
                               mask=(rn[:, None] < N) & (rk[None, :] < K))
            acc    += tl.dot(x_tile, w_tile.T.to(tl.float16))
        s_tile = tl.load(S + rn, mask=rn < N)
        acc    = acc * s_tile[None, :]
        tl.store(O + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    grid = ((M + BLOCK - 1) // BLOCK, (N + BLOCK - 1) // BLOCK)
    _i8mm_kernel[grid](x_c, Wq_c, sc_c, out_c, M, N, K,
                       BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK)
    result = cp.asnumpy(out_c).astype(x.dtype)
    return result.reshape(*x.shape[:-1], N)


def _triton_i4mm(x, W_q, scale, zero, group_size):
    # Dequantize on GPU then matmul — simpler than a full int4 kernel
    import cupy as cp
    Wq_c = cp.asarray(W_q)
    lo   = (Wq_c & 0x0F).astype(cp.float32)
    hi   = ((Wq_c >> 4) & 0x0F).astype(cp.float32)
    n_packed = Wq_c.shape[1]
    W_r = cp.zeros((W_q.shape[0], n_packed * 2), dtype=cp.float32)
    W_r[:, 0::2] = lo; W_r[:, 1::2] = hi
    n_groups = scale.shape[1]
    usable   = n_groups * group_size
    W_r      = W_r[:, :usable].reshape(W_q.shape[0], n_groups, group_size)
    sc_c     = cp.asarray(scale); z_c = cp.asarray(zero)
    W_f      = (W_r * sc_c[:, :, None] + z_c[:, :, None]).reshape(W_q.shape[0], usable)
    x_c      = cp.asarray(x.reshape(-1, x.shape[-1]).astype(np.float32))
    out_c    = x_c @ W_f.T
    result   = cp.asnumpy(out_c).astype(x.dtype)
    return result.reshape(*x.shape[:-1], W_q.shape[0])


def _numpy_i4mm_fallback(x, W_q, scale, zero, group_size):
    n_groups = scale.shape[1]
    out      = W_q.shape[0]
    usable   = n_groups * group_size
    lo = (W_q & 0x0F).astype(np.float32)
    hi = ((W_q >> 4) & 0x0F).astype(np.float32)
    n_packed = W_q.shape[1]
    W_r = np.empty((out, n_packed * 2), dtype=np.float32)
    W_r[:, 0::2] = lo; W_r[:, 1::2] = hi
    W_r  = W_r[:, :usable].reshape(out, n_groups, group_size)
    W_f  = (W_r * scale[:, :, None] + zero[:, :, None]).reshape(out, usable)
    return x.astype(np.float32) @ W_f.T