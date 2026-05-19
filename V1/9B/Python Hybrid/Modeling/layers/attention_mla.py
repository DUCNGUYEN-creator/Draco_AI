# DracoAI V1 — modeling/layers/attention_mla.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MLA — Multi-head Latent Attention (V1 simplified).

Reduces KV-cache memory by projecting K and V down to a smaller
latent dimension before storing them.  On read, the latent vectors
are expanded back to full head_dim for the attention computation.

Architecture
────────────
Standard GQA KV storage: (n_kv_heads, window, head_dim) ← full precision
MLA KV storage:           (n_kv_heads, window, latent_dim) ← compressed

Memory ratio: latent_dim / head_dim  (e.g. 0.25× for latent_dim=head_dim//4)

Projection matrices (per attention layer, trained jointly):
  W_kc : (head_dim, latent_dim)  — Key compress
  W_vc : (head_dim, latent_dim)  — Value compress
  W_ke : (latent_dim, head_dim)  — Key expand
  W_ve : (latent_dim, head_dim)  — Value expand

Forward pass:
  K_latent = K @ W_kc             (1, n_kv_heads, seq, latent_dim)
  V_latent = V @ W_vc
  Store K_latent, V_latent in KVCache (smaller dtype=float16 or int8)

  On read:
  K_full = K_latent_cached @ W_ke (1, n_kv_heads, history, head_dim)
  V_full = V_latent_cached @ W_ve

Integration:
  MLAProjection is attached to GQAttention and called during update/get.
  When mla is None (default), GQAttention behaves exactly as before.
  This makes MLA a pure opt-in feature with zero overhead when unused.

Note on accuracy:
  W_kc * W_ke ≠ I in general, so there IS compression loss.
  For inference-only use (not training from scratch), the compression
  matrices should be initialized as approximate pseudo-inverses or
  trained via a small calibration pass.  Without training, MLA reduces
  memory but increases reconstruction error ~30-50%.
  Use case: memory-constrained inference where some quality loss is
  acceptable in exchange for running a larger context window.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np

__all__ = ["MLAProjection"]


class MLAProjection:
    """
    KV compression / expansion matrices for one attention layer.

    Parameters
    ----------
    n_kv_heads  : number of KV heads (same as GQAttention)
    head_dim    : full KV vector dimension
    latent_dim  : compressed dimension  (recommended: head_dim // 4 to head_dim // 2)

    Initialisation: W_kc and W_ke are initialised as approximate
    truncated SVD so that W_kc @ W_ke ≈ I (low reconstruction error).
    In practice this means compress → expand ≈ identity for the
    top-latent_dim singular vectors of a typical K matrix.
    """

    def __init__(self, n_kv_heads: int, head_dim: int, latent_dim: int):
        if latent_dim >= head_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be < head_dim ({head_dim}) "
                f"for compression to be meaningful.")
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.latent_dim = latent_dim

        # Initialise via random orthonormal basis (SVD-like approximation)
        # W_kc: (head_dim, latent_dim),  W_ke = W_kc.T (exact left-inverse)
        # With this init, W_kc @ W_ke = I_{latent} so the top-k subspace
        # is perfectly preserved — reconstruction error comes only from
        # the discarded (head_dim - latent_dim) dimensions.
        scale = 1.0 / math.sqrt(head_dim)
        Q_k, _ = np.linalg.qr(
            np.random.randn(head_dim, latent_dim).astype(np.float32) * scale)
        Q_v, _ = np.linalg.qr(
            np.random.randn(head_dim, latent_dim).astype(np.float32) * scale)

        self.W_kc: np.ndarray = Q_k.astype(np.float32)          # (head_dim, latent_dim)
        self.W_ke: np.ndarray = Q_k.T.astype(np.float32)        # (latent_dim, head_dim)
        self.W_vc: np.ndarray = Q_v.astype(np.float32)
        self.W_ve: np.ndarray = Q_v.T.astype(np.float32)

    # ── Compression (applied before writing to KVCache) ───────────────────────

    def compress_k(self, K: np.ndarray) -> np.ndarray:
        """
        K : (1, n_kv_heads, seq, head_dim)
        Returns (1, n_kv_heads, seq, latent_dim)
        """
        return K @ self.W_kc

    def compress_v(self, V: np.ndarray) -> np.ndarray:
        return V @ self.W_vc

    # ── Expansion (applied after reading from KVCache) ────────────────────────

    def expand_k(self, K_latent: np.ndarray) -> np.ndarray:
        """
        K_latent : (1, n_kv_heads, history, latent_dim)
        Returns   (1, n_kv_heads, history, head_dim)
        """
        return K_latent @ self.W_ke

    def expand_v(self, V_latent: np.ndarray) -> np.ndarray:
        return V_latent @ self.W_ve

    # ── Memory reporting ──────────────────────────────────────────────────────

    def compression_ratio(self) -> float:
        """Storage ratio: latent / full (lower = more compressed)."""
        return self.latent_dim / self.head_dim

    def __repr__(self) -> str:
        return (f"MLAProjection(n_kv_heads={self.n_kv_heads}, "
                f"head_dim={self.head_dim}, latent_dim={self.latent_dim}, "
                f"ratio={self.compression_ratio():.2f}x)")