# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Normalization helpers — e.g. making an expert-boost dict sum to 1.0."""

from __future__ import annotations

from typing import Dict, TypeVar

K = TypeVar("K")


def normalize_dict(raw: Dict[K, float]) -> Dict[K, float]:
    total = sum(raw.values())
    if total <= 0:
        n = max(len(raw), 1)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}


def min_max_normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))
