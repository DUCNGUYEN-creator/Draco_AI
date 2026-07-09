# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
LearningStats
===============
Accumulates per-intent and per-rating-bucket statistics for the
learning subsystem — used by Evaluator to compute aggregate quality
metrics (e.g. average rating per intent type over last N turns).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class LearningStats:
    def __init__(self) -> None:
        self._ratings: Dict[str, List[float]] = defaultdict(list)

    def record(self, intent_type: str, rating: float) -> None:
        self._ratings[intent_type].append(rating)

    def avg_rating(self, intent_type: str) -> float:
        r = self._ratings.get(intent_type, [])
        return sum(r) / len(r) if r else 0.5

    def summary(self) -> Dict[str, Any]:
        return {
            intent: {
                "n": len(ratings),
                "avg": round(sum(ratings) / len(ratings), 3),
                "min": round(min(ratings), 3),
                "max": round(max(ratings), 3),
            }
            for intent, ratings in self._ratings.items()
        }
