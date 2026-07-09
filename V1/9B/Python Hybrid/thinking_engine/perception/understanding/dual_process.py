# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DualProcessDecider (System 1 / System 2)
===========================================
Ported 1:1 from engine_v1.py's ``DualProcessDecider``.
"""

from __future__ import annotations

from ...constants import (
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_COMPARISON,
    INTENT_LOGIC,
    INTENT_MATH,
    INTENT_MEMORY,
)


class DualProcessDecider:
    FAST_INTENTS = {INTENT_CHAT, INTENT_MEMORY}
    SLOW_INTENTS = {INTENT_MATH, INTENT_LOGIC, INTENT_CODE, INTENT_COMPARISON}

    def decide_mode(self, intent: dict, query: str) -> str:
        """Fast-path for simple queries. INTENT_CHAT with word_count <= 3
        always returns "fast" (skips ToT, Debate, Self-Consistency)."""
        itype = intent["intent"]
        wc = intent.get("word_count", 5)
        if itype in self.FAST_INTENTS or wc <= 3:
            return "fast"
        if itype in self.SLOW_INTENTS or wc >= 15:
            return "slow"
        return "fast"

    @staticmethod
    def is_simple_chat(intent: dict) -> bool:
        """True if the query can use the early-exit path:
        INTENT_CHAT + word_count <= 3."""
        return intent.get("intent") == INTENT_CHAT and intent.get("word_count", 99) <= 3
