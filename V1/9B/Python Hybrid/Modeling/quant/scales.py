# DracoAI V1 — modeling/quant/scales.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Scale / zero-point computation utilities.
Pure functions — no model state, no QuantizedLinear dependency.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "compute_int8_scale", "compute_int4_scale_zero",
    "dequantize_int8", "dequantize_int4",
]


def compute_int8_scale(W: np.ndarray) -> np.ndarray:
    """Per-output-channel symmetric INT8 scale. W: (out, in) → scale: (out,)"""
    return (np.abs(W).max(axis=1) / 127.0 + 1e-9).astype(np.float32)


def compute_int4_scale_zero(
    W: np.ndarray,
    group_size: int = 128,
) -> tuple:
    """
    Asymmetric per-group INT4 scale and zero.
    W: (out, in) → scale (out, n_groups), zero (out, n_groups)
    """
    out, in_feat = W.shape
    n_groups     = in_feat // group_size
    if n_groups == 0:
        raise ValueError(f"group_size={group_size} > in_feat={in_feat}")
    usable   = n_groups * group_size
    W_use    = W[:, :usable].reshape(out, n_groups, group_size).astype(np.float32)
    w_min    = W_use.min(axis=-1)
    w_max    = W_use.max(axis=-1)
    scale    = ((w_max - w_min) / 15.0 + 1e-9).astype(np.float32)
    zero     = w_min.astype(np.float32)
    return scale, zero


def dequantize_int8(W_q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """W_q: (out, in) int8, scale: (out,) → (out, in) float32"""
    return W_q.astype(np.float32) * scale[:, None]


def dequantize_int4(
    W_q:        np.ndarray,
    scale:      np.ndarray,
    zero:       np.ndarray,
    group_size: int = 128,
) -> np.ndarray:
    """
    W_q: (out, n_packed) uint8 (lo=even nibbles, hi=odd nibbles)
    scale: (out, n_groups), zero: (out, n_groups)
    Returns: (out, usable) float32
    """
    n_groups = scale.shape[1]
    out      = W_q.shape[0]
    usable   = n_groups * group_size

    lo = (W_q & 0x0F).astype(np.float32)
    hi = ((W_q >> 4) & 0x0F).astype(np.float32)
    n_packed = W_q.shape[1]
    W_r = np.empty((out, n_packed * 2), dtype=np.float32)
    W_r[:, 0::2] = lo
    W_r[:, 1::2] = hi
    W_r = W_r[:, :usable].reshape(out, n_groups, group_size)
    return (W_r * scale[:, :, None] + zero[:, :, None]).reshape(out, usable)