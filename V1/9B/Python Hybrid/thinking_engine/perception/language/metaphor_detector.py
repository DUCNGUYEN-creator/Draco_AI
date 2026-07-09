# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MetaphorDetector
==================
Detects figurative language (ẩn dụ) in user queries and returns a
literal-translation hint for the reasoning pipeline. Ported 1:1 from
engine_v1.py's ``MetaphorDetector``.
"""

from __future__ import annotations

import re
from typing import Optional

_METAPHOR_PATTERNS = [
    r"như\s+\w+",
    r"giống\s+như",
    r"is\s+like",
    r"as\s+\w+\s+as",
]

_COMMON_METAPHORS = {
    "mớ bòng bong": "very tangled / complicated",
    "đầu óc trống rỗng": "mind is blank",
    "trái tim tan vỡ": "heartbroken",
    "bão tố": "turbulent situation",
}


class MetaphorDetector:
    def detect(self, text: str) -> Optional[str]:
        tl = text.lower()
        for phrase, meaning in _COMMON_METAPHORS.items():
            if phrase in tl:
                return f"[METAPHOR DETECTED] '{phrase}' → literal meaning: {meaning}"
        for pat in _METAPHOR_PATTERNS:
            if re.search(pat, tl):
                return "[METAPHOR DETECTED] figurative language present — interpret carefully"
        return None
