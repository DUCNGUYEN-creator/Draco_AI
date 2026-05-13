# DracoAI V1 — modeling/kernels/numba/quant_matmul.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Numba JIT INT8 / INT4 weight-only matmul — CPU parallel.
"""
from __future__ import annotations
import numpy as np

__all__ = ["numba_int8_matmul", "numba_int4_matmul"]


def numba_int8_matmul(
    x:     np.ndarray,
    W_q:   np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    try:
        return _numba_i8mm(x, W_q, scale)
    except Exception:
        W_f = W_q.astype(np.float32) * scale[:, None]
        return x.astype(np.float32) @ W_f.T


def numba_int4_matmul(
    x:     np.ndarray,
    W_q:   np.ndarray,
    scale: np.ndarray,
    zero:  np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    try:
        return _numba_i4mm(x, W_q, scale, zero, group_size)
    except Exception:
        return _numpy_i4mm(x, W_q, scale, zero, group_size)


def _numba_i8mm(x, W_q, scale):
    import numba as nb

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _kernel(x, W_q, scale):
        M, K = x.shape
        N    = W_q.shape[0]
        out  = np.zeros((M, N), dtype=np.float32)
        for i in nb.prange(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += x[i, k] * W_q[j, k]
                out[i, j] = s * scale[j]
        return out

    x32 = x.reshape(-1, x.shape[-1]).astype(np.float32)
    result = _kernel(x32, W_q, scale).astype(x.dtype)
    return result.reshape(*x.shape[:-1], W_q.shape[0])


def _numba_i4mm(x, W_q, scale, zero, group_size):
    W_f = _dequant_i4(W_q, scale, zero, group_size)
    return x.astype(np.float32) @ W_f.T


def _dequant_i4(W_q, scale, zero, group_size):
    n_groups = scale.shape[1]
    out      = W_q.shape[0]
    usable   = n_groups * group_size
    lo = (W_q & 0x0F).astype(np.float32)
    hi = ((W_q >> 4) & 0x0F).astype(np.float32)
    n_packed = W_q.shape[1]
    W_r = np.empty((out, n_packed * 2), dtype=np.float32)
    W_r[:, 0::2] = lo; W_r[:, 1::2] = hi
    W_r  = W_r[:, :usable].reshape(out, n_groups, group_size)
    return (W_r * scale[:, :, None] + zero[:, :, None]).reshape(out, usable)


def _numpy_i4mm(x, W_q, scale, zero, group_size):
    return x.astype(np.float32) @ _dequant_i4(W_q, scale, zero, group_size).T