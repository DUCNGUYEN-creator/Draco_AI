# DracoAI V1 — modeling/kv_cache/kv_quant.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
KV-Quantization utilities — standalone module for Key-Value cache compression.

Provides symmetric per-head INT8 quantization for KV cache tensors.
This module contains the pure functions used by KVCache when
use_kv_quant=True, extracted here for independent testing and reuse.

Compression math
────────────────
For each (head, position) pair, we store:
  • INT8 quantized values: q = round(clip(x / scale, -127, 127))
  • float16 scale:         scale = max(|x|) / 127 + eps

Dequantization: x_approx = q * scale
Max relative error: ≈ 1/127 ≈ 0.8% per element.
Typical mean relative error: < 0.3% for normal attention weights.

Memory savings vs float16:
  INT8 values: 1 byte per element
  float16 scale: 2 bytes per head per position (amortised across head_dim)
  Break-even: head_dim > 4 (always true in practice, typically 64-128)
  Effective bits per weight: 8 + 16/head_dim ≈ 8.25 bits (vs 16 for fp16)
  → ~48% memory reduction.

Versus float32:
  Effective bits: 8.25 vs 32 → ~74% reduction.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = [
    "kv_quantize", "kv_dequantize",
    "kv_quantize_batch", "kv_dequantize_batch",
    "kv_memory_bytes",
]

_EPS = np.float16(1e-5)


# ── Per-vector quantization ───────────────────────────────────────────────────

def kv_quantize(
    x: np.ndarray,   # (..., head_dim) float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-vector INT8 quantization.

    Parameters
    ----------
    x : (..., head_dim) float32

    Returns
    -------
    q     : (..., head_dim) int8    — quantized values
    scale : (..., 1)        float16 — per-vector scale
    """
    scale_f32 = (np.abs(x).max(axis=-1, keepdims=True) / 127.0 + float(_EPS)
                 ).astype(np.float32)
    scale = scale_f32.astype(np.float16)
    q = np.clip(
        np.round(x / scale_f32), -127, 127
    ).astype(np.int8)
    return q, scale


def kv_dequantize(
    q:     np.ndarray,   # (..., head_dim) int8
    scale: np.ndarray,   # (..., 1) float16
) -> np.ndarray:
    """
    Dequantize INT8 KV values back to float32.

    Returns (..., head_dim) float32.
    """
    return q.astype(np.float32) * scale.astype(np.float32)


# ── Batch quantization for entire KV slabs ───────────────────────────────────

def kv_quantize_batch(
    KV: np.ndarray,   # (n_layers, batch, n_kv_heads, window, head_dim) float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize an entire KV slab at once.

    Designed to compress a KVCache buffer from float16 to INT8 for
    serialization or offloading.  The scale has shape
    (n_layers, batch, n_kv_heads, window, 1) float16.

    Returns
    -------
    q     : same shape as KV, dtype int8
    scale : (n_layers, batch, n_kv_heads, window, 1) float16
    """
    x32   = KV.astype(np.float32)
    scale_f32 = (np.abs(x32).max(axis=-1, keepdims=True) / 127.0 + float(_EPS)
                 ).astype(np.float32)
    scale = scale_f32.astype(np.float16)
    q = np.clip(np.round(x32 / scale_f32), -127, 127).astype(np.int8)
    return q, scale


def kv_dequantize_batch(
    q:     np.ndarray,   # (..., head_dim) int8
    scale: np.ndarray,   # (..., 1) float16
) -> np.ndarray:
    """Dequantize an entire quantized KV slab back to float32."""
    return q.astype(np.float32) * scale.astype(np.float32)


# ── Memory estimation ─────────────────────────────────────────────────────────

def kv_memory_bytes(
    n_layers:   int,
    max_batch:  int,
    n_kv_heads: int,
    window:     int,
    head_dim:   int,
    quantized:  bool = False,
    dtype:      np.dtype = np.float16,
) -> dict:
    """
    Estimate memory usage for K and V buffers combined.

    Returns a dict with 'float_bytes', 'quant_bytes', and 'savings_pct'.
    """
    float_bytes = (n_layers * max_batch * n_kv_heads * window * head_dim
                   * 2 * np.dtype(dtype).itemsize)
    # INT8 values + float16 scale per slot
    quant_val_bytes   = n_layers * max_batch * n_kv_heads * window * head_dim * 2 * 1
    quant_scale_bytes = n_layers * max_batch * n_kv_heads * window * 1 * 2 * 2
    quant_bytes = quant_val_bytes + quant_scale_bytes
    savings_pct = (1 - quant_bytes / max(float_bytes, 1)) * 100

    return {
        "float_bytes": float_bytes,
        "float_gb":    float_bytes / 1024**3,
        "quant_bytes": quant_bytes,
        "quant_gb":    quant_bytes / 1024**3,
        "savings_pct": round(savings_pct, 1),
    }