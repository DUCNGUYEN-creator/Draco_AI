# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ContextualPromptRewriter (CPR)
================================
Resolves anaphora/ellipsis against recent conversation history — e.g.
"nó" / "cái đó" / "it" / "that" referring to the previous turn's topic.
Ported from engine_v1.py's ``ContextualPromptRewriter``.
"""

from __future__ import annotations

import re
from typing import List


class ContextualPromptRewriter:
    _PRONOUN_PATTERNS = [
        r"\bnó\b", r"\bcái đó\b", r"\bviệc đó\b", r"\bđiều đó\b",
        r"\bit\b", r"\bthat\b", r"\bthis\b",
    ]

    def rewrite(self, question: str, history: List[dict]) -> str:
        """If the question contains a bare anaphoric pronoun and we have
        history, append a contextual hint referencing the last user turn.
        Returns the (possibly) rewritten question unchanged in length-class
        but enriched with a parenthetical context hint.
        """
        if not history:
            return question
        ql = question.lower()
        if not any(re.search(p, ql) for p in self._PRONOUN_PATTERNS):
            return question
        last_user_turn = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_turn = msg.get("content", "")
                break
        if not last_user_turn:
            return question
        hint = last_user_turn[:60].strip()
        if not hint:
            return question
        return f"{question} (context: {hint})"
