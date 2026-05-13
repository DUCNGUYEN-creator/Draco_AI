# DracoAI V1 — modeling/constants.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Immutable compile-time constants.
NO runtime state, NO imports from upper layers.
All values are plain Python literals — safe to import from anywhere.

This is the SINGLE source of truth for all named constants.
config.py re-exports a subset for backward compatibility but never
redefines them.
"""

# ── KV cache ──────────────────────────────────────────────────────────
SINK_TOKENS: int = 4          # Attention-sink slots never evicted

# ── Speculative decoding ──────────────────────────────────────────────
SPEC_THRESH: float = 0.80     # MTP accept confidence threshold

# ── Sampling defaults ─────────────────────────────────────────────────
DEFAULT_TEMP:  float = 0.7
DEFAULT_TOP_P: float = 0.9

# ── MoE routing ───────────────────────────────────────────────────────
MOE_NOISE_SCALE: float = 0.05   # Gumbel noise scale for router diversity
MOE_TOP_K:       int   = 2      # Default top-k routing

# ── RoPE ──────────────────────────────────────────────────────────────
ROPE_THETA: float = 10_000.0

# ── Numerical stability ───────────────────────────────────────────────
LOGIT_CLIP:   float = 50.0
SOFTMAX_EPS:  float = 1e-9
NORM_EPS:     float = 1e-6

# ── Memory thresholds ─────────────────────────────────────────────────
KVCACHE_WARN_GB:        float = 4.0
HEALTH_SAT_THRESH:      float = 45.0
HEALTH_MEM_WARN_MB:     float = 12_000.0
HEALTH_COLLAPSE_THRESH: float = 0.9

# ── Generation checkpointing ──────────────────────────────────────────
WAL_FLUSH_INTERVAL: int = 16

# ── Kernel registry keys ──────────────────────────────────────────────
KERNEL_NUMPY  = "numpy"
KERNEL_TRITON = "triton"
KERNEL_NUMBA  = "numba"

__all__ = [
    "SINK_TOKENS", "SPEC_THRESH", "DEFAULT_TEMP", "DEFAULT_TOP_P",
    "MOE_NOISE_SCALE", "MOE_TOP_K", "ROPE_THETA",
    "LOGIT_CLIP", "SOFTMAX_EPS", "NORM_EPS",
    "KVCACHE_WARN_GB", "HEALTH_SAT_THRESH", "HEALTH_MEM_WARN_MB",
    "HEALTH_COLLAPSE_THRESH", "WAL_FLUSH_INTERVAL",
    "KERNEL_NUMPY", "KERNEL_TRITON", "KERNEL_NUMBA",
]