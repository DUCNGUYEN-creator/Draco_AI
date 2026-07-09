# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ContextWindowManager
=======================
Monitors estimated token count and collapses older messages into a
summary placeholder when the budget is exceeded. Ported 1:1 from
engine_v1.py's ``ContextWindowManager``, including the CTX-MGR-FIX
ordering (token check runs before the message-count guard).
"""

from __future__ import annotations

from typing import List

from .summarization import HistorySummarizer


class ContextWindowManager:
    def __init__(self, max_tokens: int = 2800):
        self.max_tokens = max_tokens
        self._summarizer = HistorySummarizer()

    @staticmethod
    def _est_tokens(messages: List[dict]) -> int:
        """Rough estimate: chars / 4."""
        return sum(len(m.get("content", "")) // 4 for m in messages)

    def manage(self, messages: List[dict]) -> List[dict]:
        """Keeps messages[0] (system) + last 4 messages; the middle is
        replaced with a summary system message. The token check runs
        BEFORE the message-count guard so a small number of very long
        messages is still correctly trimmed."""
        if self._est_tokens(messages) <= self.max_tokens:
            return messages
        if len(messages) <= 5:
            # Can't trim further without losing system prompt or last exchange
            return messages
        old_msgs = messages[1:-4]
        summary = self._summarizer.summarize(old_msgs)
        return (
            [messages[0]]
            + [{"role": "system", "content": f"[Conversation summary] {summary}"}]
            + messages[-4:]
        )
