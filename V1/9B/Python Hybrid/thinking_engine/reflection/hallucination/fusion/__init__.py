# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.fusion
===================================================
Combines multiple (decorrelated, calibrated) verifier failure-
probabilities into ONE risk_score for a claim. This is the "↓ Fusion"
step right before Risk/Report. Every method here takes a list of
(probability, weight) pairs — weight typically = verifier confidence *
correlation discount from correlation.reducer.EvidenceReducer — and
returns a FusionResult.
"""

from .base import BaseFusionStrategy
from .weighted import WeightedAverageFusion
from .noisy_or import NoisyOrFusion
from .bayesian import BayesianFusion
from .dempster_shafer import DempsterShaferFusion
from .logistic import LogisticFusion
from .ensemble import FusionEnsemble
from .selector import FusionSelector

__all__ = [
    "BaseFusionStrategy",
    "WeightedAverageFusion",
    "NoisyOrFusion",
    "BayesianFusion",
    "DempsterShaferFusion",
    "LogisticFusion",
    "FusionEnsemble",
    "FusionSelector",
]
