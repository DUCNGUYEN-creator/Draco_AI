# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ProblemDecomposer
====================
New addition. Splits a compound question ("X và Y", "A or B", "X, Y,
and Z") into independent sub-questions that can be reasoned about (and
later verified) separately — distinct from planning/goal_decomposer.py
(which decomposes a *goal* into *sequential steps*): this decomposes a
*question* into *parallel, independent sub-questions*.
"""

from __future__ import annotations

import re
from typing import List

_SPLIT_PATTERN = re.compile(r"\s+(?:và|and|,)\s+|\s+(?:hay|or)\s+", re.IGNORECASE)


class ProblemDecomposer:
    def decompose(self, question: str, max_parts: int = 4) -> List[str]:
        parts = [p.strip(" ?.!") for p in _SPLIT_PATTERN.split(question) if p.strip(" ?.!")]
        if len(parts) < 2:
            return [question]
        return parts[:max_parts]

    def is_compound(self, question: str) -> bool:
        return len(self.decompose(question)) > 1
