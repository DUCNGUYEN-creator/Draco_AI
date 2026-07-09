# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
InstructionChainParser
========================
Detects and splits sequential multi-step instructions such as
"Đầu tiên ... sau đó ... rồi ..." / "First ... then ... next ...".
Ported from engine_v1.py's ``InstructionChainParser``.
"""

from __future__ import annotations

import re
from typing import List

_CHAIN_MARKERS = [
    "đầu tiên", "sau đó", "rồi", "tiếp theo", "cuối cùng",
    "first", "then", "next", "after that", "finally",
]


class InstructionChainParser:
    def is_chain(self, text: str) -> bool:
        tl = text.lower()
        hits = sum(1 for m in _CHAIN_MARKERS if m in tl)
        return hits >= 2

    def parse(self, text: str) -> List[str]:
        """Split on chain markers, keeping the remainder of each segment
        as one step. Falls back to comma/semicolon split if markers
        don't yield at least 2 non-empty steps."""
        pattern = r"(?:" + "|".join(re.escape(m) for m in _CHAIN_MARKERS) + r")[,:]?\s*"
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        steps = [p.strip(" .,;") for p in parts if p.strip(" .,;")]
        if len(steps) < 2:
            steps = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
        return steps
