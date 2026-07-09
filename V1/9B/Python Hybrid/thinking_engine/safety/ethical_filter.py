# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EthicalFilter
===============
Pre-output safety gate. Scores a candidate answer for ethical issues:
0.0 = completely safe, 1.0 = highly problematic. Triggers a rewrite
request if score > threshold. Ported 1:1 from engine_v1.py's
``EthicalFilter``.
"""

from __future__ import annotations

from typing import List

_UNSAFE_KEYWORDS: List[str] = [
    "giết", "tự tử", "chế tạo bom", "vũ khí", "kích động",
    "kill", "suicide", "bomb", "weapon", "discriminat", "hate speech",
    "phân biệt chủng tộc", "khủng bố", "terrorist",
]
_BIAS_KEYWORDS: List[str] = [
    "tất cả người", "all women", "all men", "all asians",
    "người việt đều", "người tây đều",
]


class EthicalFilter:
    _UNSAFE_KEYWORDS = _UNSAFE_KEYWORDS
    _BIAS_KEYWORDS = _BIAS_KEYWORDS

    def score(self, text: str) -> float:
        tl = text.lower()
        score = 0.0
        for kw in self._UNSAFE_KEYWORDS:
            if kw in tl:
                score += 0.25
        for kw in self._BIAS_KEYWORDS:
            if kw in tl:
                score += 0.1
        return min(score, 1.0)

    def is_safe(self, text: str, threshold: float = 0.3) -> bool:
        return self.score(text) < threshold

    def build_rewrite_instruction(self) -> str:
        return (
            "[SAFETY] Câu trả lời trước vi phạm hướng dẫn an toàn. "
            "Hãy viết lại theo cách an toàn, không gây tổn thương, "
            "trung lập và tôn trọng tất cả mọi người."
        )
