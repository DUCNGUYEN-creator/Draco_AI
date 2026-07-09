# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
UncertaintyQuantifier
========================
Tags individual sentences in an answer with [confidence:X] based on
hedge-word / certainty-word heuristics. Ported 1:1 from engine_v1.py's
``UncertaintyQuantifier``.
"""

from __future__ import annotations

import re
from typing import List

_HEDGE_WORDS = [
    "có thể", "maybe", "perhaps", "possibly", "không chắc", "i think",
    "tôi nghĩ", "dường như", "it seems", "might", "could be", "probably",
]
_CERTAIN_WORDS = [
    "chắc chắn", "definitely", "clearly", "obviously", "always", "luôn",
    "proven", "đã được chứng minh",
]


class UncertaintyQuantifier:
    def tag(self, answer: str, base_confidence: float = 0.75) -> str:
        if not answer or len(answer.strip()) < 10:
            return answer
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        tagged: List[str] = []
        for sent in sentences:
            sl = sent.lower()
            conf = base_confidence
            hedges = sum(1 for w in _HEDGE_WORDS if w in sl)
            certains = sum(1 for w in _CERTAIN_WORDS if w in sl)
            conf -= hedges * 0.08
            conf += certains * 0.05
            conf = round(max(0.1, min(1.0, conf)), 2)
            if conf < 0.5:
                tagged.append(f"[confidence:{conf}] {sent}")
            else:
                tagged.append(sent)
        return " ".join(tagged)

    def overall_confidence(self, answer: str, base: float) -> float:
        sl = answer.lower()
        hedges = sum(1 for w in _HEDGE_WORDS if w in sl)
        certains = sum(1 for w in _CERTAIN_WORDS if w in sl)
        return round(max(0.1, min(1.0, base - hedges * 0.05 + certains * 0.03)), 2)
