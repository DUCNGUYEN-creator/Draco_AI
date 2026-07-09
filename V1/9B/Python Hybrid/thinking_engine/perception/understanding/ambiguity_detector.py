# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
AmbiguityDetector
====================
New addition (marked NEW in the proposed architecture). Flags queries
that are too underspecified to answer confidently — feeds into the
ActiveLearningLoop's clarification-question trigger alongside raw
confidence so we don't only rely on per-intent base confidence.
"""

from __future__ import annotations

_VAGUE_REFERENTS = ["nó", "cái này", "cái đó", "this", "that", "it"]
_VAGUE_QUANTIFIERS = ["một số", "vài", "some", "several", "a few"]


class AmbiguityDetector:
    def score(self, query: str, intent: dict) -> float:
        """0.0 (clear) .. 1.0 (very ambiguous)."""
        ql = query.lower()
        score = 0.0
        wc = intent.get("word_count", len(query.split()))
        if wc <= 2:
            score += 0.4
        if any(r in ql for r in _VAGUE_REFERENTS) and not intent.get("entities"):
            score += 0.3
        if any(q in ql for q in _VAGUE_QUANTIFIERS):
            score += 0.15
        if not intent.get("entities") and intent.get("intent") not in ("chat",):
            score += 0.15
        return min(score, 1.0)

    def is_ambiguous(self, query: str, intent: dict, threshold: float = 0.5) -> bool:
        return self.score(query, intent) >= threshold
