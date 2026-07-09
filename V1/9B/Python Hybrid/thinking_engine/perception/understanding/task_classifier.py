# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TaskClassifier
================
Coarser-grained classifier than IntentDetector: buckets a query into
one of a handful of *task families* (reasoning / generation / retrieval
/ conversation / tool_use). Planning and Routing consult this instead of
re-deriving task family from raw intent each time.
"""

from __future__ import annotations

from ...constants import (
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_COMPARISON,
    INTENT_CREATIVE,
    INTENT_FACTUAL,
    INTENT_HOW_TO,
    INTENT_LOGIC,
    INTENT_MATH,
    INTENT_MEMORY,
    INTENT_WHY,
)

TASK_REASONING = "reasoning"
TASK_GENERATION = "generation"
TASK_RETRIEVAL = "retrieval"
TASK_CONVERSATION = "conversation"

_MAP = {
    INTENT_MATH: TASK_REASONING,
    INTENT_LOGIC: TASK_REASONING,
    INTENT_CODE: TASK_GENERATION,
    INTENT_CREATIVE: TASK_GENERATION,
    INTENT_FACTUAL: TASK_RETRIEVAL,
    INTENT_HOW_TO: TASK_RETRIEVAL,
    INTENT_WHY: TASK_REASONING,
    INTENT_COMPARISON: TASK_REASONING,
    INTENT_CHAT: TASK_CONVERSATION,
    INTENT_MEMORY: TASK_RETRIEVAL,
}


class TaskClassifier:
    def classify(self, intent: dict) -> str:
        return _MAP.get(intent.get("intent"), TASK_CONVERSATION)
