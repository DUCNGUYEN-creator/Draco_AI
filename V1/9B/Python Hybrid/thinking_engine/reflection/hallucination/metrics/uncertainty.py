# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Predictive entropy of a fused risk score — treats the fused score as
a Bernoulli probability and computes the binary entropy, a single-number
"how uncertain was the final verdict itself" summary distinct from
analyzers.uncertainty.UncertaintyAnalyzer (which measures DISAGREEMENT
across verifiers, not the final fused score's own uncertainty)."""

from __future__ import annotations

import math


def predictive_entropy(probability: float, eps: float = 1e-12) -> float:
    p = max(eps, min(1.0 - eps, probability))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))
