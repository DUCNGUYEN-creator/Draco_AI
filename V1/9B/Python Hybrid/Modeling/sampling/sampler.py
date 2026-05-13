# DracoAI V1 — modeling/sampling/sampler.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Sampler — static methods, no instance state."""
from __future__ import annotations
import numpy as np
from typing import Tuple
from ..constants import DEFAULT_TEMP, DEFAULT_TOP_P, LOGIT_CLIP
from .mirostat import mirostat_v2 as _mirostat_v2

__all__ = ["Sampler"]


class Sampler:
    @staticmethod
    def mirostat_v2(logits: np.ndarray, mu: float,
                    tau: float = 5.0, eta: float = 0.1) -> Tuple[int, float]:
        return _mirostat_v2(logits, mu, tau, eta)

    @staticmethod
    def topk_topp(logits: np.ndarray, temp: float = DEFAULT_TEMP,
                  top_p: float = DEFAULT_TOP_P, top_k: int = 50,
                  min_p: float = 0.0) -> int:
        logits = np.clip(logits / max(temp, 1e-6), -LOGIT_CLIP, LOGIT_CLIP)
        if top_k > 0 and top_k < len(logits):
            kth    = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < kth, -1e9, logits)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-9
        if min_p > 0.0:
            max_prob = float(probs.max())
            probs[probs < min_p * max_prob] = 0.0
            p_sum = probs.sum()
            probs = probs / p_sum if p_sum > 1e-9 else np.full_like(
                probs, 1.0 / len(probs))
        idx    = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[idx])
        cut    = int(np.searchsorted(cumsum, top_p)) + 1
        probs_trunc = np.zeros_like(probs)
        probs_trunc[idx[:cut]] = probs[idx[:cut]]
        p_sum = probs_trunc.sum()
        if p_sum < 1e-9:
            probs_trunc[idx[0]] = 1.0; p_sum = 1.0
        probs_trunc /= p_sum
        bad = ~np.isfinite(probs_trunc)
        if bad.any():
            probs_trunc[bad] = 0.0
            s = probs_trunc.sum()
            if s < 1e-9:
                probs_trunc[idx[0]] = 1.0
            else:
                probs_trunc /= s
        return int(np.random.choice(len(probs_trunc), p=probs_trunc))

    @staticmethod
    def argmax(logits: np.ndarray) -> int:
        return int(np.argmax(logits))