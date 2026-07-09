# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FusionPipeline — Stage 5: "Fusion"
======================================
Resolves the configured fusion method via FusionFactory and runs it
over the (already decorrelated, via CorrelationPipeline) signal list,
producing one FusionResult per claim.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..factory.fusion_factory import FusionFactory
from ..models.fusion import FusionResult


class FusionPipeline:
    def __init__(self, factory: Optional[FusionFactory] = None) -> None:
        self.factory = factory or FusionFactory()

    def fuse(self, signals: List[Tuple[str, float, float]], method: str) -> FusionResult:
        strategy = self.factory.get(method)
        return strategy.fuse(signals)
