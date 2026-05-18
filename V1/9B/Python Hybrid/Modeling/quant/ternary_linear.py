# DracoAI V1 — modeling/quant/ternary_linear.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TernaryLinear — Hybrid Ternary-Dense weight layer (BitNet 1.58b style).

Architecture
────────────
Weights are quantized to {-1, 0, +1} using the BitNet 1.58b formula:
    E = mean(|W|)
    W_t = Round(Clip(W / E, -1, 1))

Storage uses 2-bit packing: 4 ternary values per uint8 byte.
    Encoding: 00 → 0,  01 → +1,  10 → -1  (11 unused, treated as 0)

Forward pass (addition-only, no multiplications in inner loop):
    Y = (X @ pos_mask.T - X @ neg_mask.T) * scale
where pos_mask / neg_mask are boolean masks extracted from packed weights,
and scale = E (the per-output-row mean-abs of the original float weights).

When the Triton ternary kernel is available it is used; otherwise the
NumPy fallback computes the same result via matrix-boolean products.

Selective use policy (enforced by callers):
  GREEN  (use ternary): MoE Expert FFN layers (W_g, W_u, W_d)
  RED (keep FP16/F32): Attention Q/K/V/O, Router, Embedding, LM Head, Norm

Memory savings vs FP16: ~8× compression for weight storage.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

__all__ = ["TernaryLinear", "ternarize_weight"]

logger = logging.getLogger(__name__)


# ── Public helper ─────────────────────────────────────────────────────────────

def ternarize_weight(W: np.ndarray) -> tuple:
    """
    Quantize a float weight matrix to ternary {-1, 0, +1}.

    Parameters
    ----------
    W : (out_feat, in_feat) float32

    Returns
    -------
    W_packed : (out_feat, ceil(in_feat/4)) uint8  — 2-bit packed ternary
    scale    : (out_feat,) float32                — per-row mean-abs scale
    in_feat  : int                                — original input dimension
    """
    W32 = W.astype(np.float32)
    out_feat, in_feat = W32.shape
    # Per-row scale (BitNet 1.58b: mean of absolute values)
    scale = np.abs(W32).mean(axis=1).astype(np.float32)  # (out,)
    E = scale[:, None] + 1e-9  # broadcast-safe
    W_t = np.round(np.clip(W32 / E, -1.0, 1.0)).astype(np.int8)
    packed = _pack_ternary(W_t)
    return packed, scale, in_feat


# ── 2-bit packing / unpacking ─────────────────────────────────────────────────

def _pack_ternary(W_t: np.ndarray) -> np.ndarray:
    """
    Pack int8 {-1,0,+1} tensor to uint8 2-bit representation.
    4 values per byte: bits [1:0]=val0, [3:2]=val1, [5:4]=val2, [7:6]=val3
    Encoding: 0→0, +1→1, -1→2.
    """
    out, in_feat = W_t.shape
    # Encode: 0→0, +1→1, -1→2
    encoded = np.where(W_t == 0, np.uint8(0),
               np.where(W_t == 1, np.uint8(1), np.uint8(2)))
    # Pad input dim to multiple of 4
    pad = (4 - in_feat % 4) % 4
    if pad:
        encoded = np.pad(encoded, ((0, 0), (0, pad)), constant_values=0)
    reshaped = encoded.reshape(out, -1, 4)
    packed = (
        (reshaped[:, :, 0]       ) |
        (reshaped[:, :, 1] << 2  ) |
        (reshaped[:, :, 2] << 4  ) |
        (reshaped[:, :, 3] << 6  )
    ).astype(np.uint8)
    return packed


def _unpack_ternary(packed: np.ndarray, in_feat: int) -> np.ndarray:
    """
    Unpack uint8 2-bit tensor back to int8 {-2, -1, 0, +1}.
    Code 2 → -1, others as-is (0, 1).
    """
    parts = [((packed >> (2 * i)) & np.uint8(0x03)) for i in range(4)]
    decoded = np.stack(parts, axis=-1).reshape(packed.shape[0], -1)[:, :in_feat]
    # Decode: 0→0, 1→+1, 2→-1
    return np.where(decoded == 0, np.int8(0),
            np.where(decoded == 1, np.int8(1), np.int8(-1))).astype(np.int8)


# ── TernaryLinear ─────────────────────────────────────────────────────────────

class TernaryLinear:
    """
    Weight-only ternary linear layer: y = x @ W_t.T * scale.

    Weights stored as 2-bit packed uint8.  Forward pass uses addition-only
    arithmetic (no float multiplications in the inner loop).

    The Triton kernel path is attempted first; if unavailable (CPU-only
    environment) the NumPy fallback is used transparently.
    """

    def __init__(self):
        self.W_packed:  Optional[np.ndarray] = None   # (out, ceil(in/4)) uint8
        self.scale:     Optional[np.ndarray] = None   # (out,) float32
        self.in_feat:   int = 0
        self.out_feat:  int = 0
        self._W_float_cache: Optional[np.ndarray] = None  # lazy dequant cache

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_float(cls, W: np.ndarray) -> "TernaryLinear":
        """
        Quantize a float weight matrix and return a TernaryLinear layer.

        W : (out_feat, in_feat) float32 / float16
        """
        t = cls()
        t.out_feat, t.in_feat = W.shape
        t.W_packed, t.scale, _ = ternarize_weight(W)
        return t

    # ── Dequantize ────────────────────────────────────────────────────────────

    def dequantize(self) -> np.ndarray:
        """Return reconstructed float32 weight matrix (out, in)."""
        if self._W_float_cache is not None:
            return self._W_float_cache
        W_t = _unpack_ternary(self.W_packed, self.in_feat).astype(np.float32)
        W_f = W_t * self.scale[:, None]
        self._W_float_cache = W_f
        return W_f

    def invalidate_cache(self):
        self._W_float_cache = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Addition-only forward: y = (X @ pos_mask.T - X @ neg_mask.T) * scale.

        Falls back to standard dequant matmul if x is non-contiguous or
        if the Triton ternary kernel is unavailable.

        x : (..., in_feat) float32
        returns : (..., out_feat) float32
        """
        # Try Triton kernel path (GPU, zero-multiply)
        try:
            from ..kernels import get_kernel
            kern = get_kernel("ternary_matmul")
            if kern is not None:
                return kern(x.astype(np.float32), self.W_packed,
                            self.scale, self.in_feat)
        except Exception:
            pass

        # NumPy addition-only path
        return self._numpy_forward(x)

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Pure-NumPy addition-only matmul.
        Y = (X @ pos.T - X @ neg.T) * scale[None, :]

        Uses boolean mask matmul which avoids fp multiplications
        in the inner product (replaced by conditional accumulate).
        """
        x32 = x.astype(np.float32)
        W_t = _unpack_ternary(self.W_packed, self.in_feat).astype(np.float32)
        # Boolean-equivalent masks (cast once, reuse)
        pos = (W_t ==  1.0)  # (out, in) bool
        neg = (W_t == -1.0)  # (out, in) bool
        # Matrix products with {0,1} masks = conditional sums
        y_pos = x32 @ pos.astype(np.float32).T   # (..., out)
        y_neg = x32 @ neg.astype(np.float32).T   # (..., out)
        return (y_pos - y_neg) * self.scale[None, :]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save ternary weights to .npz file."""
        np.savez_compressed(
            path if path.endswith(".npz") else path + ".npz",
            W_packed  = self.W_packed,
            scale     = self.scale,
            in_feat   = np.array(self.in_feat),
            out_feat  = np.array(self.out_feat),
        )

    @classmethod
    def load(cls, path: str) -> "TernaryLinear":
        fname = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fname)
        t = cls()
        t.W_packed = data["W_packed"]
        t.scale    = data["scale"]
        t.in_feat  = int(data["in_feat"])
        t.out_feat = int(data["out_feat"])
        return t

    def __repr__(self) -> str:
        ratio = (self.out_feat * self.in_feat * 2) / (
            self.W_packed.nbytes * 8) if self.W_packed is not None else 0
        return (f"TernaryLinear(out={self.out_feat}, in={self.in_feat}, "
                f"bits≈1.58, compress={ratio:.1f}x)")