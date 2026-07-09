# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BayesianBeliefUpdater
========================
Maintains a belief probability per (subj, obj) edge and updates it via
Bayes' rule as new evidence arrives. Ported 1:1 from engine_v1.py's
``BayesianBeliefUpdater``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

from ..utils.probability import bayes_update


class BayesianBeliefUpdater:
    def __init__(self) -> None:
        self._beliefs: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.7)

    def update(self, subj: str, obj: str, evidence_supports: bool) -> None:
        prior = self._beliefs[(subj, obj)]
        if evidence_supports:
            likelihood_h, likelihood_nh = 0.9, 0.1
        else:
            likelihood_h, likelihood_nh = 0.1, 0.9
        self._beliefs[(subj, obj)] = bayes_update(prior, likelihood_h, likelihood_nh)

    def get_belief(self, subj: str, obj: str) -> float:
        return round(self._beliefs[(subj, obj)], 3)

    def is_uncertain(self, subj: str, obj: str, threshold: float = 0.4) -> bool:
        return self._beliefs[(subj, obj)] < threshold
