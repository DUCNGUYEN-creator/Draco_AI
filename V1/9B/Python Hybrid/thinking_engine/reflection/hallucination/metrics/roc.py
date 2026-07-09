# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""ROC curve + AUC — used by benchmarks/verifier.py to judge a
verifier's discriminative power independent of its calibration quality
(a verifier can have perfect AUC but terrible calibration, or vice
versa — ECE and AUC measure genuinely different things)."""

from __future__ import annotations

from typing import List, Tuple


def roc_curve_points(predictions: List[float], labels: List[int]) -> List[Tuple[float, float]]:
    """Returns [(false_positive_rate, true_positive_rate), ...] sorted by
    descending threshold, starting at (0,0) and ending at (1,1)."""
    pairs = sorted(zip(predictions, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)]  # degenerate — no discrimination possible

    points = [(0.0, 0.0)]
    tp, fp = 0, 0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        points.append((fp / n_neg, tp / n_pos))
    return points


def auc_from_points(points: List[Tuple[float, float]]) -> float:
    """Trapezoidal-rule integration of the ROC curve."""
    if len(points) < 2:
        return 0.5
    points = sorted(points)
    auc = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc
