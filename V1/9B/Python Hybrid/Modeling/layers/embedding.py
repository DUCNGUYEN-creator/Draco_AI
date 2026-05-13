# DracoAI V1 — modeling/layers/embedding.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Token embedding lookup + weight-tying helper."""
from __future__ import annotations
import numpy as np

__all__ = ["Embedding"]


class Embedding:
    """
    Vocabulary embedding table.
    weight: (vocab_size, d_model).
    Shared with lm_head by default (weight tying).
    """

    def __init__(self, vocab_size: int, d_model: int, dtype: np.dtype = np.float32):
        scale        = 1.0 / (d_model ** 0.5)
        self.weight  = (np.random.randn(vocab_size, d_model) * scale).astype(dtype)

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        ids = np.clip(ids, 0, self.weight.shape[0] - 1)
        return self.weight[ids]

    def __repr__(self) -> str:
        v, d = self.weight.shape
        return f"Embedding(vocab={v}, d_model={d}, dtype={self.weight.dtype})"