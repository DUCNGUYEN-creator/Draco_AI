# DracoAI V1 — modeling/sampling/penalties.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Token-level penalty utilities — decoupled from Sampler / DracoTransformerV1.
All functions return a MODIFIED COPY (do not mutate the input array).
"""
from __future__ import annotations
import math
from typing import Dict
import numpy as np

__all__ = ["apply_repetition_penalty", "apply_frequency_penalty", "apply_presence_penalty"]


def apply_repetition_penalty(
    logits:    np.ndarray,
    freq:      Dict[int, int],
    pos:       Dict[int, int],
    n_pos:     int,
    rep_alpha: float = 0.5,
) -> np.ndarray:
    """
    Soft repetition penalty: logit[t] -= alpha * log(1 + freq[t]) / dist
    where dist = n_pos - last_pos[t] + 1 (recent tokens penalised more).
    Returns a modified copy.
    """
    logits = logits.copy()
    for tid, cnt in freq.items():
        if cnt > 0:
            dist = n_pos - pos.get(tid, 0) + 1
            logits[tid] -= rep_alpha * math.log(1 + cnt) / dist
    return logits


def apply_frequency_penalty(
    logits:  np.ndarray,
    freq:    Dict[int, int],
    penalty: float = 0.1,
) -> np.ndarray:
    """logit[t] -= penalty * count[t]. Returns a modified copy."""
    logits = logits.copy()
    for tid, cnt in freq.items():
        if cnt > 0:
            logits[tid] -= penalty * cnt
    return logits


def apply_presence_penalty(
    logits:  np.ndarray,
    freq:    Dict[int, int],
    penalty: float = 0.1,
) -> np.ndarray:
    """Flat discount for any token that has appeared at least once. Returns a modified copy."""
    logits = logits.copy()
    for tid, cnt in freq.items():
        if cnt > 0:
            logits[tid] -= penalty
    return logits