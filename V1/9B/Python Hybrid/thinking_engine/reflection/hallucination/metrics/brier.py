# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Brier score — mean squared error between predicted probability and
binary outcome. Simpler than ECE (no binning needed), penalizes
overconfident wrong predictions more heavily than ECE does."""

from __future__ import annotations

from typing import List

from ....utils.math import mean


def brier_score(predictions: List[float], labels: List[int]) -> float:
    if not predictions:
        return 0.0
    return mean((p - y) ** 2 for p, y in zip(predictions, labels))
