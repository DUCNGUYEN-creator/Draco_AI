# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Small numeric helpers shared across calibration/fusion/metrics —
kept dependency-free (no numpy requirement) so thinking_engine installs
anywhere transformer_v1's heavier stack is unavailable."""

from __future__ import annotations

import math
from typing import Iterable, List


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-clamp(x, -20.0, 20.0)))


def logit(p: float, eps: float = 1e-6) -> float:
    p = clamp(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def stddev(xs: Iterable[float]) -> float:
    return math.sqrt(variance(xs))


def entropy(probs: Iterable[float], eps: float = 1e-12) -> float:
    """Shannon entropy in nats."""
    total = 0.0
    for p in probs:
        if p > eps:
            total -= p * math.log(p)
    return total


def softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(xs)] * len(xs)
