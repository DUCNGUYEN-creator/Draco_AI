# DracoAI V1 — modeling/config.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI V1 — Model Configuration.

Constants are defined ONCE in constants.py and re-exported here for
backward compatibility.  config.py owns only ModelConfig.

COMPUTE_DTYPE is NOT redefined here.  It is imported from dtypes and
re-exported so that callers importing from config see the live value
managed by dtypes.get_compute_dtype() / set_compute_dtype().

FIXES (this revision):
  ✅ FIX-COMPUTE-DTYPE-SINGLE-SOURCE : removed local _detect_compute_dtype()
     call and local COMPUTE_DTYPE assignment.  Now delegates entirely to
     dtypes.get_compute_dtype().  This guarantees set_compute_dtype() updates
     are visible to all importers regardless of which module they import from.
"""
from __future__ import annotations
import numpy as np

# ── Re-export all constants from the single source of truth ───────────
from .constants import (
    SINK_TOKENS, SPEC_THRESH, DEFAULT_TEMP, DEFAULT_TOP_P,
    MOE_NOISE_SCALE, MOE_TOP_K, ROPE_THETA,
    LOGIT_CLIP, SOFTMAX_EPS, NORM_EPS,
    KVCACHE_WARN_GB, HEALTH_SAT_THRESH, HEALTH_MEM_WARN_MB,
    HEALTH_COLLAPSE_THRESH, WAL_FLUSH_INTERVAL,
    KERNEL_NUMPY, KERNEL_TRITON, KERNEL_NUMBA,
)

# ── COMPUTE_DTYPE: single source in dtypes.py ─────────────────────────
# Re-export get_compute_dtype so callers can do:
#   from .config import get_compute_dtype
# and always receive the live dtype, even after set_compute_dtype().
from .dtypes import get_compute_dtype, set_compute_dtype

# For backward-compatible attribute access (e.g. `config.COMPUTE_DTYPE`),
# expose the current value at import time.  For guaranteed freshness after
# set_compute_dtype(), callers must use get_compute_dtype() directly.
COMPUTE_DTYPE: np.dtype = get_compute_dtype()


class ModelConfig:
    __slots__ = (
        "d_model", "n_layers", "n_heads", "n_kv_heads",
        "head_dim", "d_ff", "n_experts", "vocab_size", "window", "rope_theta",
    )

    def __init__(
        self,
        d_model:    int   = 128,
        n_layers:   int   = 4,
        n_heads:    int   = 4,
        n_kv_heads: int   = 2,
        head_dim:   int   = 32,
        d_ff:       int   = 512,
        n_experts:  int   = 8,
        vocab_size: int   = 151936,
        window:     int   = 1024,
        rope_theta: float = ROPE_THETA,
    ):
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be positive and even, got {head_dim}")
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.d_ff       = d_ff
        self.n_experts  = n_experts
        self.vocab_size = vocab_size
        self.window     = window
        self.rope_theta = rope_theta

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            d_model    = d.get("d_model",    128),
            n_layers   = d.get("n_layers",   4),
            n_heads    = d.get("n_heads",    4),
            n_kv_heads = d.get("n_kv_heads", 2),
            head_dim   = d.get("head_dim",   32),
            d_ff       = d.get("d_ff",       512),
            n_experts  = d.get("n_experts",  8),
            vocab_size = d.get("vocab_size", 151936),
            window     = d.get("window",     1024),
            rope_theta = d.get("rope_theta", ROPE_THETA),
        )

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"ModelConfig(d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, d_ff={self.d_ff}, "
            f"n_experts={self.n_experts}, vocab={self.vocab_size}, "
            f"window={self.window})"
        )


__all__ = [
    # ModelConfig
    "ModelConfig", "COMPUTE_DTYPE", "get_compute_dtype", "set_compute_dtype",
    # Re-exported constants (full set for wildcard imports)
    "SINK_TOKENS", "SPEC_THRESH", "DEFAULT_TEMP", "DEFAULT_TOP_P",
    "MOE_NOISE_SCALE", "MOE_TOP_K", "ROPE_THETA",
    "LOGIT_CLIP", "SOFTMAX_EPS", "NORM_EPS",
    "KVCACHE_WARN_GB", "HEALTH_SAT_THRESH", "HEALTH_MEM_WARN_MB",
    "HEALTH_COLLAPSE_THRESH", "WAL_FLUSH_INTERVAL",
    "KERNEL_NUMPY", "KERNEL_TRITON", "KERNEL_NUMBA",
]