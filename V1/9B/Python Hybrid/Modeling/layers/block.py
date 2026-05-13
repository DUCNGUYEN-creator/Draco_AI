# DracoAI V1 — modeling/layers/block.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TransformerBlock — single transformer layer: GQA + MoE FFN with pre-norm.

This is the SOLE definition of TransformerBlock.
transformer.py imports from here; it does NOT redefine the class.

Deep Engram integration: the engram parameter is forwarded from the
TransformerBlock.forward() signature down into GQAttention.forward(),
enabling the three-tier hierarchical memory to blend with exact attention
inside the attention kernel.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .attention       import GQAttention
from .moe             import MoELayer
from ..ops.tensor_ops import rms_norm

if TYPE_CHECKING:
    from ..kv_cache.kv_cache     import KVCache
    from ..kv_cache.engram_cache import EngramCache
    from ..runtime.tensor_pool   import TensorPool

__all__ = ["TransformerBlock"]


class TransformerBlock:
    """Pre-norm GQA + MoE FFN block.

    Parameters
    ──────────
    engram (forward arg): Optional EngramCache. When provided, GQAttention
        blends exact sliding-window attention with engram cross-attention.
        Passed per-call so it can be enabled/disabled per generation step
        without reconstructing the block.
    """

    def __init__(self, layer_idx: int, d_model: int, n_heads: int,
                 n_kv_heads: int, head_dim: int, d_ff: int,
                 n_experts: int = 8, rope_theta: float = 10000.0):
        self.layer_idx = layer_idx
        self.attn  = GQAttention(d_model, n_heads, n_kv_heads, head_dim, rope_theta)
        self.moe   = MoELayer(d_model, d_ff, n_experts)
        self.norm1 = np.ones(d_model, dtype=np.float32)
        self.norm2 = np.ones(d_model, dtype=np.float32)

    def forward(
        self,
        x:           np.ndarray,
        cache:       "KVCache",
        add_noise:   bool                   = True,
        intent_bias: Optional[np.ndarray]   = None,
        snap:        Optional[dict]          = None,
        batch_idx:   int                     = 0,
        pool:        Optional["TensorPool"]  = None,
        engram:      Optional["EngramCache"] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        x: (1, seq, d_model) → (output, aux_dict)

        Both attention and MoE route to the best available kernel
        (Triton → Numba → NumPy) internally. No platform branching here.

        engram: when provided, deep Engram cross-attention is blended with
            exact sliding-window attention inside GQAttention.forward().
        """
        h      = rms_norm(x, self.norm1)
        h      = self.attn.forward(
            h, cache, self.layer_idx,
            snap      = snap,
            batch_idx = batch_idx,
            pool      = pool,
            engram    = engram,      # ← deep Engram integration
        )
        x      = x + h
        h, aux = self.moe.forward(rms_norm(x, self.norm2),
                                  add_noise=add_noise, intent_bias=intent_bias)
        return x + h, aux