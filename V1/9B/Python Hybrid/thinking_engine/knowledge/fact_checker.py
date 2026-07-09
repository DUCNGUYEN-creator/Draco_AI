# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FactConsistencyChecker
=========================
Extracts (subject, relation, object) triples from a model answer and
cross-checks them against the KnowledgeGraph, flagging contradictions.
Ported 1:1 from engine_v1.py's ``FactConsistencyChecker``. This is one
of the Infrastructure-layer signals consumed by the Verification
layer's hallucination/verifiers/retrieval.py and consistency.py.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from .knowledge_graph import KnowledgeGraph

_TRIPLE_PATTERNS = [
    r"([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯ][a-zàáâãèéêìíòóôõùúăđơư]+)\s+"
    r"(is|was|sinh năm|có|thuộc|là)\s+"
    r"([0-9]{4}|[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯ][a-zàáâãèéêìíòóôõùúăđơư]+)",
]


class FactConsistencyChecker:
    def check(self, answer: str, kg: "KnowledgeGraph") -> List[str]:
        """Returns a list of contradiction strings (empty = no issues)."""
        issues: List[str] = []
        for pattern in _TRIPLE_PATTERNS:
            for m in re.finditer(pattern, answer):
                subj, rel, obj = m.group(1), m.group(2), m.group(3)
                if subj in kg.g and obj not in kg.g.get(subj, {}):
                    strong_nbs = {nb for nb, w in kg.g[subj].items() if w > 0.7}
                    if strong_nbs and obj not in strong_nbs:
                        issues.append(
                            f"Fact conflict: '{subj} {rel} {obj}' "
                            f"not found in KG (known: {list(strong_nbs)[:3]})"
                        )
        return issues
