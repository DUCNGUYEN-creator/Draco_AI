# DracoAI V1 — modeling/kernels/numba/fused_mlp.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Numba JIT fused SwiGLU — CPU fallback accelerator.
Faster than pure NumPy on multi-core CPUs via parallelised loops.
"""
from __future__ import annotations
import numpy as np

__all__ = ["numba_fused_swiglu"]


def numba_fused_swiglu(
    x:   np.ndarray,
    W_g: np.ndarray,
    W_u: np.ndarray,
    W_d: np.ndarray,
) -> np.ndarray:
    try:
        return _numba_swiglu(x, W_g, W_u, W_d)
    except Exception:
        gate = x @ W_g
        gate = gate / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
        return (gate * (x @ W_u)) @ W_d


def _numba_swiglu(x, W_g, W_u, W_d):
    import numba as nb
    import math

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _kernel(x, W_g, W_u, W_d):
        seq, d   = x.shape
        d_ff     = W_g.shape[1]
        d_out    = W_d.shape[1]
        gate_up  = np.empty((seq, d_ff), dtype=np.float32)

        for i in nb.prange(seq):
            for j in range(d_ff):
                g = 0.0; u = 0.0
                for k in range(d):
                    g += x[i, k] * W_g[k, j]
                    u += x[i, k] * W_u[k, j]
                # SiLU: g * sigmoid(g)
                g_act    = g / (1.0 + math.exp(-min(max(g, -50.0), 50.0)))
                gate_up[i, j] = g_act * u

        out = np.zeros((seq, d_out), dtype=np.float32)
        for i in nb.prange(seq):
            for j in range(d_out):
                s = 0.0
                for k in range(d_ff):
                    s += gate_up[i, k] * W_d[k, j]
                out[i, j] = s
        return out

    x32   = x.astype(np.float32)
    Wg32  = W_g.astype(np.float32)
    Wu32  = W_u.astype(np.float32)
    Wd32  = W_d.astype(np.float32)
    return _kernel(x32, Wg32, Wu32, Wd32).astype(x.dtype)