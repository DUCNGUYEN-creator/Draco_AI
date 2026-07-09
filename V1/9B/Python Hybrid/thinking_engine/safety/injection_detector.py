# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
InjectionDetector
====================
Detects (without necessarily sanitizing) prompt-injection ATTEMPTS in
external content, returning a structured report instead of mutated
text — useful for logging/telemetry ("how often are we seeing
injection attempts") separately from PromptSanitizer's mutate-in-place
behaviour.
"""

from __future__ import annotations

import re
from typing import Dict, List

_INJECTION_PATTERNS = [
    r"<\|.*?\|>",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"<<SYS>>.*?<</SYS>>",
    r"ignore (all|previous) instructions",
    r"bỏ qua (mọi|tất cả) hướng dẫn (trước|trên)",
    r"you are now",
    r"act as if",
]


class InjectionDetector:
    def detect(self, text: str) -> Dict[str, object]:
        hits: List[str] = []
        for pat in _INJECTION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE | re.DOTALL):
                hits.append(pat)
        return {
            "is_suspicious": len(hits) > 0,
            "matched_patterns": hits,
            "risk_score": min(len(hits) * 0.3, 1.0),
        }
