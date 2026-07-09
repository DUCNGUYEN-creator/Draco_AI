# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""BaseFusionStrategy — shared contract for every fusion method."""

from __future__ import annotations

import abc
from typing import Dict, List, Tuple

from ..models.fusion import FusionResult


class BaseFusionStrategy(abc.ABC):
    method_name: str = "base"

    @abc.abstractmethod
    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        """``signals`` is a list of (verifier_name, failure_probability,
        weight) triples — weight in [0, 1], already decorrelated by
        correlation/reducer.py. Returns a FusionResult with fused_score
        in [0, 1] representing overall hallucination risk."""

    def _empty_result(self) -> FusionResult:
        return FusionResult(method=self.method_name, fused_score=0.0, n_signals_used=0)
