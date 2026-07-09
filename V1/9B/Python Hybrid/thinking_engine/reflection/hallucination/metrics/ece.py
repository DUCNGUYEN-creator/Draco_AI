# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Bucketing helper shared by expected_calibration_error and
maximum_calibration_error — bins predictions into n equal-width
confidence buckets and computes per-bucket accuracy vs mean confidence."""

from __future__ import annotations

from typing import Dict, List


def compute_ece_bins(predictions: List[float], labels: List[int], n_bins: int = 10) -> List[Dict[str, float]]:
    bins = [{"count": 0, "sum_conf": 0.0, "sum_correct": 0} for _ in range(n_bins)]
    for p, y in zip(predictions, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx]["count"] += 1
        bins[idx]["sum_conf"] += p
        bins[idx]["sum_correct"] += y

    result = []
    for b in bins:
        count = b["count"]
        result.append(
            {
                "count": count,
                "confidence": b["sum_conf"] / count if count else 0.0,
                "accuracy": b["sum_correct"] / count if count else 0.0,
            }
        )
    return result
