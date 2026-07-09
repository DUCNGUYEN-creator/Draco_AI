# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TripleExtractor
=================
Extracts (subject, relation, object, weight) triples from free text via
regex patterns. Ported from the pattern table embedded in engine_v1.py's
``KnowledgeGraph.extract_and_add_triples`` — split out so the patterns
can be unit-tested/extended without touching KnowledgeGraph itself.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_PATTERNS: List[Tuple[str, str, float]] = [
    (r"(\w[\w\s]{1,20})\s+là\s+([\w][\w\s]{1,20})", "là", 0.8),
    (r"(\w[\w\s]{1,20})\s+is\s+([\w][\w\s]{1,20})", "is", 0.8),
    (r"(\w[\w\s]{1,20})\s+gây ra\s+([\w][\w\s]{1,20})", "causes", 0.7),
    (r"(\w[\w\s]{1,20})\s+causes?\s+([\w][\w\s]{1,20})", "causes", 0.7),
    (r"(\w[\w\s]{1,20})\s+thuộc\s+([\w][\w\s]{1,20})", "belongs_to", 0.75),
    (r"(\w[\w\s]{1,20})\s+dùng để\s+([\w][\w\s]{1,20})", "used_for", 0.7),
]


class TripleExtractor:
    PATTERNS = _PATTERNS

    def extract(self, text: str, conf: float = 0.6) -> List[Tuple[str, str, str, float]]:
        results: List[Tuple[str, str, str, float]] = []
        for pat, rel, base_w in self.PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                subj = m.group(1).strip()[:30]
                obj = m.group(2).strip()[:30]
                if len(subj) < 2 or len(obj) < 2:
                    continue
                results.append((subj, rel, obj, base_w * conf))
        return results
