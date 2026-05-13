# DracoAI V1 — modeling/ops/attention_ops.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""RoPE, safe softmax, causal mask — pure numeric ops."""
from __future__ import annotations
import numpy as np
from ..constants import ROPE_THETA, SOFTMAX_EPS, LOGIT_CLIP

__all__ = ["rope_freqs", "apply_rope", "safe_softmax", "causal_mask_bias"]


def rope_freqs(head_dim: int, base: float = ROPE_THETA) -> np.ndarray:
    i = np.arange(0, head_dim, 2, dtype=np.float32)
    return 1.0 / (base ** (i / head_dim))


def apply_rope(x: np.ndarray, freqs: np.ndarray, offset: int = 0) -> np.ndarray:
    seq  = x.shape[-2]
    hdim = x.shape[-1]
    pos    = np.arange(offset, offset + seq, dtype=np.float32)
    angles = np.outer(pos, freqs)
    cos = np.cos(angles).astype(x.dtype)
    sin = np.sin(angles).astype(x.dtype)
    extra = x.ndim - 2
    shape = (1,) * extra + (seq, hdim // 2)
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    x1 = x[..., :hdim // 2]
    x2 = x[..., hdim // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def safe_softmax(x: np.ndarray, axis: int = -1,
                 clip: float = LOGIT_CLIP) -> np.ndarray:
    x = np.clip(x, -clip, clip)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + SOFTMAX_EPS)


def causal_mask_bias(seq: int, dtype=np.float32) -> np.ndarray:
    return np.triu(np.full((seq, seq), -1e9, dtype=dtype), 1)