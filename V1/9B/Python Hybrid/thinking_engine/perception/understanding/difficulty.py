# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DifficultyScorer
==================
Scores 0.0 (trivial) -> 1.0 (very hard), used to auto-route to System 2.
Ported 1:1 from engine_v1.py's ``DifficultyScorer``.
"""

from __future__ import annotations

from ...constants import (
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_COMPARISON,
    INTENT_LOGIC,
    INTENT_MATH,
    INTENT_WHY,
)

_HARD_KEYWORDS = [
    "prove", "chứng minh", "implement", "refactor", "optimize", "debug",
    "compare", "so sánh", "tại sao", "phân tích", "analyze", "tổng hợp",
    "synthesize", "thiết kế", "design", "architecture", "kiến trúc",
]


class DifficultyScorer:
    def score(self, query: str, intent: dict) -> float:
        base = 0.2
        itype = intent.get("intent", INTENT_CHAT)
        if itype in (INTENT_MATH, INTENT_LOGIC, INTENT_CODE):
            base += 0.3
        elif itype in (INTENT_COMPARISON, INTENT_WHY):
            base += 0.2
        entity_count = len(intent.get("entities", []))
        base += min(entity_count * 0.05, 0.2)
        ql = query.lower()
        kw_hits = sum(1 for k in _HARD_KEYWORDS if k in ql)
        base += min(kw_hits * 0.08, 0.24)
        wc = intent.get("word_count", 5)
        if wc >= 20:
            base += 0.1
        return min(base, 1.0)
