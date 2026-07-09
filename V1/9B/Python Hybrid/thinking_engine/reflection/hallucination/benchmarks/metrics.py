# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""MetricsBenchmark — validates metric implementations against known
ground-truth values (e.g. AUC of a perfect classifier = 1.0, Brier
score of perfect predictions = 0.0, entropy of uniform = log(n))."""

from __future__ import annotations

import math
from typing import Dict

from ..metrics import auc_from_points, brier_score, normalized_entropy, roc_curve_points


class MetricsBenchmark:
    def run(self) -> Dict[str, object]:
        perfect_preds = [0.99, 0.99, 0.01, 0.01]
        perfect_labels = [1, 1, 0, 0]
        worst_preds = [0.01, 0.01, 0.99, 0.99]

        pts_perfect = roc_curve_points(perfect_preds, perfect_labels)
        pts_worst = roc_curve_points(worst_preds, perfect_labels)

        return {
            "auc_perfect_classifier": round(auc_from_points(pts_perfect), 4),
            "auc_worst_classifier": round(auc_from_points(pts_worst), 4),
            "brier_perfect": round(brier_score(perfect_preds, perfect_labels), 4),
            "brier_worst": round(brier_score(worst_preds, perfect_labels), 4),
            "entropy_uniform_4": round(normalized_entropy([0.25]*4), 4),
            "entropy_degenerate": round(normalized_entropy([1.0, 0.0, 0.0, 0.0]), 4),
            "checks": {
                "auc_perfect==1": abs(auc_from_points(pts_perfect) - 1.0) < 0.01,
                "brier_perfect~0": brier_score(perfect_preds, perfect_labels) < 0.01,
                "entropy_uniform==1": abs(normalized_entropy([0.25]*4) - 1.0) < 0.001,
                "entropy_degenerate==0": normalized_entropy([1.0, 0.0, 0.0, 0.0]) == 0.0,
            },
        }
