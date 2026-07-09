# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Evaluator
===========
New addition. Runs a mini evaluation harness over the ExperienceBuffer
to compute the engine's current average quality score per intent, which
the pipeline uses to decide whether to escalate strategy (e.g. switch
from "balanced" to "paranoid" Hallucination strategy) for a given
intent type where quality has been trending down.
"""

from __future__ import annotations

from typing import Any, Dict

from .experience import ExperienceBuffer
from .statistics import LearningStats


class Evaluator:
    def __init__(self, buffer: ExperienceBuffer, stats: LearningStats) -> None:
        self._buffer = buffer
        self._stats = stats

    def run(self, sample_size: int = 100) -> Dict[str, Any]:
        samples = self._buffer.sample(sample_size)
        for s in samples:
            intent = s.get("metadata", {}).get("intent_type", "unknown")
            self._stats.record(intent, s.get("rating", 0.5))
        return self._stats.summary()

    def should_escalate_strategy(self, intent_type: str, threshold: float = 0.5) -> bool:
        """True if avg_rating for this intent is below threshold — suggest
        switching to a stricter Hallucination strategy for this type."""
        return self._stats.avg_rating(intent_type) < threshold
