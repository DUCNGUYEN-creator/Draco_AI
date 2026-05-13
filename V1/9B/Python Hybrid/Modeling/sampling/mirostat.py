# DracoAI V1 — modeling/sampling/mirostat.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Mirostat v2 adaptive sampler.

mu update uses MINUS sign (correct negative feedback per Basu 2020):
    mu ← mu - eta * (surprise - tau)
"""
from __future__ import annotations
import numpy as np
from ..constants import LOGIT_CLIP

__all__ = ["mirostat_v2"]


def mirostat_v2(
    logits: np.ndarray,
    mu:     float,
    tau:    float = 5.0,
    eta:    float = 0.1,
) -> tuple:
    """
    Returns (chosen_id: int, new_mu: float).
    """
    logits = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum() + 1e-9
    bad = ~np.isfinite(probs)
    if bad.any():
        probs[bad] = 0.0
        s = probs.sum()
        probs[:] = (probs / s) if s > 1e-9 else (1.0 / len(probs))

    idx          = np.argsort(probs)[::-1]
    probs_sorted = probs[idx]
    surprises    = -np.log2(probs_sorted + 1e-9)

    raw_cutoff = int(np.searchsorted(surprises, mu))
    cutoff     = max(1, min(raw_cutoff, len(probs_sorted)))
    trunc      = probs_sorted[:cutoff].copy()
    t_sum      = trunc.sum()
    if t_sum < 1e-9:
        trunc = np.ones(1); t_sum = 1.0
    trunc /= t_sum

    chosen_local = int(np.random.choice(len(trunc), p=trunc))
    chosen_id    = int(idx[chosen_local])
    surprise     = float(-np.log2(probs[chosen_id] + 1e-9))
    new_mu       = max(0.1, mu - eta * (surprise - tau))
    return chosen_id, new_mu