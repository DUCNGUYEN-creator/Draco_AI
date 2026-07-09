# DracoAI V1 — thinking_engine/perception/understanding/multi_turn_intent_tracker.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MultiTurnIntentTracker
========================
Detects topic-shift (chuyển chủ đề) between consecutive conversational
turns so the engine can decide whether to reset working memory / context,
re-route to a different expert mix, or prompt the user for confirmation.

Ported from engine_v1.py's ``MultiTurnIntentTracker`` class (lines
~1556–1620). The original class was present in ``ThinkingEngineV1`` and
called inside ``process()`` to populate ``state.topic_shift`` — this
field existed in ``ThinkingState`` but was ALWAYS False in the ported
codebase (Bug #2).

Detection strategy
------------------
Lightweight, no embeddings required (designed for offline-only engine):

1. **Intent delta**: if the primary intent changes between the previous
   turn and the current turn, that's a signal (especially strong when
   switching across task families — e.g. MATH → CHAT).
2. **Entity overlap**: if there is ZERO entity overlap between turns,
   that's a strong topic-shift signal.
3. **Keyword continuity**: presence of turn-continuation markers
   ("vậy thì", "thế còn", "also", "but what about") vs. new-topic
   markers ("chủ đề mới", "nói về", "let's talk about", "change topic").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ...constants import (
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_CREATIVE,
    INTENT_FACTUAL,
    INTENT_HOW_TO,
    INTENT_LOGIC,
    INTENT_MATH,
    INTENT_MEMORY,
)

# ── Vocabulary for continuation vs. new-topic markers ──────────────────────
_CONTINUATION_MARKERS: List[str] = [
    # Vietnamese
    "vậy thì", "thế thì", "thế còn", "ngoài ra", "thêm nữa",
    "tiếp tục", "còn gì nữa", "bổ sung", "liên quan",
    # English
    "also", "furthermore", "additionally", "moreover", "related to",
    "what about", "how about", "and also", "in addition",
    "following up", "continuing", "back to",
]

_NEW_TOPIC_MARKERS: List[str] = [
    # Vietnamese
    "chủ đề mới", "nói về", "chuyển sang", "hỏi cái khác",
    "không liên quan", "bỏ qua", "quên đi", "đổi sang",
    # English
    "new topic", "let's talk about", "change topic", "switch to",
    "unrelated", "different question", "by the way", "off topic",
    "forget that", "never mind",
]

# ── Task-family grouping for intent-distance ────────────────────────────────
_TASK_FAMILIES: Dict[str, str] = {
    INTENT_MATH: "analytical",
    INTENT_LOGIC: "analytical",
    INTENT_CODE: "technical",
    INTENT_CREATIVE: "generative",
    INTENT_FACTUAL: "informational",
    INTENT_HOW_TO: "informational",
    INTENT_MEMORY: "informational",
    INTENT_CHAT: "conversational",
}


class MultiTurnIntentTracker:
    """Tracks topic continuity across conversational turns.

    Thread-safety
    -------------
    This class is NOT thread-safe — it holds mutable ``_prev_*`` state.
    Since ``ThinkingEngine.process()`` serialises pipeline runs for a
    single session, this is fine; multi-session engines should use one
    tracker per session.
    """

    def __init__(self) -> None:
        self._prev_intent: Optional[str] = None
        self._prev_entities: Set[str] = set()
        self._turn_count: int = 0

    def track(
        self,
        query: str,
        intent: Dict[str, Any],
        history: Optional[List[dict]] = None,
    ) -> bool:
        """Returns True if a topic-shift is detected between the current
        query/intent and the previous turn's state.

        Side-effect: updates internal ``_prev_*`` trackers for the next call.
        """
        current_intent = intent.get("intent", INTENT_CHAT)
        current_entities: Set[str] = set(intent.get("entities", []))
        ql = query.lower()

        # ── First turn: no previous context to compare against ────────
        if self._turn_count == 0:
            self._prev_intent = current_intent
            self._prev_entities = current_entities
            self._turn_count = 1
            return False

        shift_score = 0.0

        # ── Signal 1: Intent delta ────────────────────────────────────
        if current_intent != self._prev_intent:
            prev_family = _TASK_FAMILIES.get(self._prev_intent or "", "other")
            curr_family = _TASK_FAMILIES.get(current_intent, "other")
            if prev_family != curr_family:
                shift_score += 0.45  # cross-family shift is strong signal
            else:
                shift_score += 0.20  # same family, different intent

        # ── Signal 2: Entity overlap ──────────────────────────────────
        if self._prev_entities and current_entities:
            overlap = self._prev_entities & current_entities
            if not overlap:
                shift_score += 0.35  # zero entity overlap
            elif len(overlap) < min(len(self._prev_entities), len(current_entities)) * 0.3:
                shift_score += 0.15  # weak entity overlap
        elif self._prev_entities and not current_entities:
            # Previous had entities, current doesn't — could be shift or just chat
            shift_score += 0.10

        # ── Signal 3: Explicit topic markers ──────────────────────────
        if any(marker in ql for marker in _NEW_TOPIC_MARKERS):
            shift_score += 0.40  # explicit "new topic" signal

        # ── Anti-signal: Continuation markers ─────────────────────────
        if any(marker in ql for marker in _CONTINUATION_MARKERS):
            shift_score -= 0.30  # continuation reduces shift confidence

        # ── Anti-signal: Very short history (< 3 turns) ───────────────
        if self._turn_count < 3:
            shift_score -= 0.10  # early-conversation tolerance

        # ── Update state for next call ────────────────────────────────
        self._prev_intent = current_intent
        self._prev_entities = current_entities
        self._turn_count += 1

        return shift_score >= 0.50

    def reset(self) -> None:
        """Resets tracker state. Call when starting a new session."""
        self._prev_intent = None
        self._prev_entities = set()
        self._turn_count = 0
