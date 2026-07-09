# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BalancedStrategy
===================
6 verifiers — the DEFAULT strategy (config.HallucinationConfig.strategy
default = "balanced"). Covers the most common, highest-value failure
modes without paying for every verifier on every request.
"""

from __future__ import annotations

from typing import List


class BalancedStrategy:
    name = "balanced"
    verifier_names: List[str] = [
        "retrieval",
        "contradiction",
        "consistency",
        "numerical",
        "reasoning",
        "tool",
    ]
    fusion_method: str = "noisy_or"
    calibration_method: str = "platt"
