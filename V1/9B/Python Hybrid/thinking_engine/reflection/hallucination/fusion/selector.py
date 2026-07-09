# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FusionSelector
=================
Resolves config.HallucinationConfig.fusion_method into a concrete
strategy instance — registry/factory.py's verifier-side counterpart for
the fusion layer specifically.
"""

from __future__ import annotations

from .base import BaseFusionStrategy
from .bayesian import BayesianFusion
from .dempster_shafer import DempsterShaferFusion
from .ensemble import FusionEnsemble
from .logistic import LogisticFusion
from .noisy_or import NoisyOrFusion
from .weighted import WeightedAverageFusion

_REGISTRY = {
    "weighted": WeightedAverageFusion,
    "noisy_or": NoisyOrFusion,
    "bayesian": BayesianFusion,
    "dempster_shafer": DempsterShaferFusion,
    "logistic": LogisticFusion,
}


class FusionSelector:
    def select(self, method: str) -> BaseFusionStrategy:
        if method == "ensemble":
            return FusionEnsemble()
        cls = _REGISTRY.get(method)
        if cls is None:
            return NoisyOrFusion()  # safe, conservative default (per config docstring rationale)
        return cls()
