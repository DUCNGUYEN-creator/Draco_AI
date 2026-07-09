# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.strategy
=====================================================
Pre-configured verifier sets + fusion/calibration method choices, per
the architecture's own design:
    Fast     -> 2 verifiers
    Balanced -> 6 verifiers
    Paranoid -> 12 verifiers (this engine has 9 built-ins, so Paranoid
                runs all 9 plus headroom for future verifiers without
                needing to change here)

config.HallucinationConfig.strategy selects one of these by name;
pipeline/assessment_pipeline.py reads .verifier_names /
.fusion_method / .calibration_method off whichever is selected.
"""

from .fast import FastStrategy
from .balanced import BalancedStrategy
from .paranoid import ParanoidStrategy
from .custom import CustomStrategy

__all__ = ["FastStrategy", "BalancedStrategy", "ParanoidStrategy", "CustomStrategy"]


def get_strategy(name: str, **kwargs):
    if name == "fast":
        return FastStrategy()
    if name == "balanced":
        return BalancedStrategy()
    if name == "paranoid":
        return ParanoidStrategy()
    if name == "custom":
        return CustomStrategy(**kwargs)
    return BalancedStrategy()  # safe, well-rounded default
