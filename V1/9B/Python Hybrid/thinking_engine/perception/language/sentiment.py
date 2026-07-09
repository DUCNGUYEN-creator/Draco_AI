# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SentimentAnalyzer
====================
Keyword-based positive/negative/neutral classifier — ported from the
sentiment branch embedded in engine_v1.py's ``IntentDetector.detect()``.
Kept as its own class so it can be swapped for an embedding-based
model later without touching IntentDetector.
"""

from __future__ import annotations

_POSITIVE = ["hay", "tốt", "tuyệt", "great", "love", "thích", "good", "awesome"]
_NEGATIVE = [
    "tệ", "xấu", "dở", "ghét", "bad", "wrong", "horrible", "bực", "chán",
    "tức", "khó chịu", "frustrat", "annoying",
]


class SentimentAnalyzer:
    def analyze(self, text: str) -> str:
        tl = text.lower()
        if any(w in tl for w in _POSITIVE):
            return "positive"
        if any(w in tl for w in _NEGATIVE):
            return "negative"
        return "neutral"
