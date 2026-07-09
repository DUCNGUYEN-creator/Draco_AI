# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Mean/weighted confidence helpers over a batch of verification results."""

from __future__ import annotations

from typing import List

from ....utils.math import mean


def mean_confidence(verification_results: List[dict]) -> float:
    return mean(r.get("confidence", 0.0) for r in verification_results)


def confidence_weighted_score(verification_results: List[dict]) -> float:
    total_w = sum(r.get("confidence", 0.0) for r in verification_results)
    if total_w <= 0:
        return mean(r.get("score", 0.5) for r in verification_results)
    return sum(r.get("score", 0.5) * r.get("confidence", 0.0) for r in verification_results) / total_w
