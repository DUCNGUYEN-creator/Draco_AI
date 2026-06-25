# DracoAI V1 — modeling/layers/block.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TransformerBlock — single transformer layer: GQA + MoE FFN with pre-norm.

This is the SOLE definition of TransformerBlock.
transformer.py imports from here; it does NOT redefine the class.

NEW in this revision:
  ✅ FEAT-HYBRID-ATTENTION-BLOCK : Optional HybridAttentionConfig and
     MLAProjection parameters.  When hybrid_config is set, the block
     queries config.is_global(layer_idx) each forward call to decide
     whether to pass is_global=True to GQAttention (full-history mode).
     When mla is set, K/V are compressed before storage and expanded
     after retrieval — reducing KV-cache footprint transparently.

     Both features are fully backward-compatible:
       • hybrid_config=None → all layers behave as local (original behavior)
       • mla=None           → no compression (original behavior)

FIXES retained:
  ✅ Deep Engram integration: engram forwarded to GQAttention.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .attention       import GQAttention
from .moe             import MoELayer
from ..ops.tensor_ops import rms_norm

if TYPE_CHECKING:
    from ..kv_cache.kv_cache          import KVCache
    from ..kv_cache.engram_cache      import EngramCache
    from ..runtime.tensor_pool        import TensorPool
    from ..layers.attention_mla       import MLAProjection
    from ..layers.hybrid_attention    import HybridAttentionConfig

__all__ = ["TransformerBlock"]


class TransformerBlock:
    """Pre-norm GQA + MoE FFN block.

    Parameters (constructor)
    ────────────────────────
    hybrid_config : HybridAttentionConfig — decides whether this layer
        uses full-history (global) or sliding-window (local) attention.
        When None, all layers use sliding-window (default, original behavior).
    mla           : MLAProjection — compresses K/V before storage and
        expands after retrieval.  When None, full head_dim KV is stored.

    Parameters (forward call)
    ─────────────────────────
    engram : Optional EngramCache. When provided, GQAttention blends exact
        sliding-window attention with engram cross-attention.
    """

    def __init__(
        self,
        layer_idx:      int,
        d_model:        int,
        n_heads:        int,
        n_kv_heads:     int,
        head_dim:       int,
        d_ff:           int,
        n_experts:      int   = 8,
        rope_theta:     float = 10000.0,
        ternary_experts: bool = False,
        sparsity_thresh: float = 0.0,
        hybrid_config:  Optional["HybridAttentionConfig"] = None,
        mla:            Optional["MLAProjection"] = None,
    ):
        self.layer_idx      = layer_idx
        self._hybrid_config = hybrid_config
        self._mla           = mla

        self.attn  = GQAttention(
            d_model, n_heads, n_kv_heads, head_dim, rope_theta,
            sparsity_thresh=sparsity_thresh,
        )
        self.moe   = MoELayer(
            d_model, d_ff, n_experts,
            ternary_experts=ternary_experts,
        )
        self.norm1 = np.ones(d_model, dtype=np.float32)
        self.norm2 = np.ones(d_model, dtype=np.float32)

    # ── Config accessors ──────────────────────────────────────────────────────

    def set_hybrid_config(self, config: Optional["HybridAttentionConfig"]) -> None:
        """Attach or replace the hybrid attention configuration."""
        self._hybrid_config = config

    def set_mla(self, mla: Optional["MLAProjection"]) -> None:
        """Attach or remove the MLA projection."""
        self._mla = mla

    @property
    def is_global_layer(self) -> bool:
        """True if this block uses full-history attention."""
        if self._hybrid_config is None:
            return False
        return self._hybrid_config.is_global(self.layer_idx)

    # ── Forward ───────────────────────────────────────────────────────────────

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
        rope_offset: Optional[int]           = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        x: (1, seq, d_model) → (output, aux_dict)

        rope_offset: RoPE positional offset, captured ONCE at the top of
            DracoTransformerV1.forward() before any layer advances _cache_pos.
            Ensures all layers use the same positional encoding for the same
            token sequence. If None, falls back to cache.get_pos().
        """
        _is_global = (
            self._hybrid_config.is_global(self.layer_idx)
            if self._hybrid_config is not None
            else False
        )

        h      = rms_norm(x, self.norm1)
        h      = self.attn.forward(
            h, cache, self.layer_idx,
            snap        = snap,
            batch_idx   = batch_idx,
            pool        = pool,
            engram      = engram,
            mla         = self._mla,
            is_global   = _is_global,
            rope_offset = rope_offset,
        )
        x      = x + h
        h, aux = self.moe.forward(rms_norm(x, self.norm2),
                                  add_noise=add_noise, intent_bias=intent_bias)
        return x + h, aux