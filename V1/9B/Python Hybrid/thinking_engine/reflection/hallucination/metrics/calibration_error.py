# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Expected / Maximum Calibration Error — the standard metrics for
judging whether a calibrator's output probabilities match empirical
frequency."""

from __future__ import annotations

from typing import List, Tuple

from .ece import compute_ece_bins


def expected_calibration_error(predictions: List[float], labels: List[int], n_bins: int = 10) -> float:
    bins = compute_ece_bins(predictions, labels, n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    return sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / total


def maximum_calibration_error(predictions: List[float], labels: List[int], n_bins: int = 10) -> float:
    bins = compute_ece_bins(predictions, labels, n_bins)
    gaps = [abs(b["confidence"] - b["accuracy"]) for b in bins if b["count"] > 0]
    return max(gaps) if gaps else 0.0
