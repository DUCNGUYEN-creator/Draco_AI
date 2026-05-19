# DracoAI V1 — modeling/layers/hybrid_attention.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Hybrid Attention Scheduler — Global Full-Attention + Local SWA.

Architecture ("Landmark" structure)
────────────────────────────────────
Layers are divided into two categories:

  Local layers  : Standard SWA (Sliding Window Attention) — cheap,
                  each token attends only to the last `window` tokens.
                  Default mode; uses the existing KVCache ring buffer.

  Global layers : Full-Attention — attends to the ENTIRE history.
                  Implemented by bypassing the ring-buffer eviction:
                  the global layer's KVCache stores all tokens (up to
                  a separate `global_window` limit, default = 4×window).
                  These layers act as "knowledge relay stations" that
                  prevent information loss across long contexts.

Engram integration:
  Global layer KV is the highest-quality source for Engram compression.
  When _try_commit_block() runs, it preferentially extracts blocks from
  global layers (if available) and falls back to the mean across all
  layers.  This is handled transparently in transformer.py by passing
  `use_global_for_engram=True` to the HybridAttentionConfig.

Usage in TransformerBlock:
  block.attn is still a GQAttention instance.
  The HybridAttentionConfig is consulted ONCE per forward call to decide
  whether the layer runs local (SWA via ring buffer) or global (all-history).
  No code change is needed in GQAttention itself — the difference is in
  which KVCache slots are passed and how cache.get() is called.

Default global layer placement:
  global_layers = [0, n_layers // 2, n_layers - 1]
  (First, middle, last — three "relay stations".)
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

__all__ = ["HybridAttentionConfig", "build_default_global_layers"]


def build_default_global_layers(n_layers: int) -> List[int]:
    """
    Default placement: first, middle, and last layer as global.
    For n_layers <= 2, all layers are global.
    """
    if n_layers <= 2:
        return list(range(n_layers))
    mid = n_layers // 2
    candidates = sorted({0, mid, n_layers - 1})
    return candidates


class HybridAttentionConfig:
    """
    Configuration object that classifies each layer as local or global.

    Passed to TransformerBlock (and forwarded to GQAttention) so that
    the attention kernel knows which KVCache budget to use.

    Parameters
    ----------
    n_layers      : Total number of transformer layers.
    global_layers : List of layer indices that use full-attention.
                    Default: first + middle + last (three relay stations).
    global_window : Max history tokens for global layers (0 = unlimited).
                    Setting this prevents OOM when running very long contexts.
    """

    def __init__(
        self,
        n_layers:      int,
        global_layers: Optional[List[int]] = None,
        global_window: int = 0,
    ):
        self.n_layers      = n_layers
        self.global_window = global_window

        if global_layers is None:
            self.global_layers: List[int] = build_default_global_layers(n_layers)
        else:
            self.global_layers = sorted(set(
                i for i in global_layers if 0 <= i < n_layers
            ))

        self._global_set = frozenset(self.global_layers)

    def is_global(self, layer_idx: int) -> bool:
        """True if layer_idx should use full-attention (global mode)."""
        return layer_idx in self._global_set

    def is_local(self, layer_idx: int) -> bool:
        return not self.is_global(layer_idx)

    def global_kv_limit(self, layer_idx: int, current_pos: int) -> int:
        """
        How many KV slots a global layer should attend to.

        Returns current_pos (all history) when global_window == 0,
        or min(current_pos, global_window) when a limit is set.
        """
        if not self.is_global(layer_idx):
            raise ValueError(f"layer_idx={layer_idx} is a local layer")
        if self.global_window <= 0:
            return current_pos
        return min(current_pos, self.global_window)

    def best_engram_layer(self) -> int:
        """
        Return the most informative global layer index for Engram compression.

        The LAST global layer (closest to the output) typically has the
        most compressed, semantically rich representations.
        """
        return self.global_layers[-1] if self.global_layers else 0

    def summary(self) -> dict:
        return {
            "n_layers":      self.n_layers,
            "global_layers": self.global_layers,
            "local_layers":  [i for i in range(self.n_layers)
                              if i not in self._global_set],
            "global_window": self.global_window,
        }

    def __repr__(self) -> str:
        return (f"HybridAttentionConfig(global={self.global_layers}, "
                f"local=[{self.n_layers - len(self.global_layers)} layers], "
                f"global_window={self.global_window})")