# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Histogram + percentile helpers for inspecting score distributions in
benchmarks/* reports."""

from __future__ import annotations

from typing import Dict, List


def histogram(values: List[float], n_bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> Dict[int, int]:
    counts = {i: 0 for i in range(n_bins)}
    width = (hi - lo) / n_bins if hi > lo else 1.0
    for v in values:
        idx = min(int((v - lo) / width), n_bins - 1) if width > 0 else 0
        idx = max(0, idx)
        counts[idx] += 1
    return counts


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)
