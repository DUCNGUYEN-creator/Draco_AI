# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
HypothesisTester
==================
Qualitative H0 test via KG edge weights + entity extraction. Ported 1:1
from engine_v1.py's ``HypothesisTester``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph


class HypothesisTester:
    def test(self, hypothesis: str, kg: "KnowledgeGraph") -> dict:
        """Returns {hypothesis, support_strength, verdict, evidence}."""
        entities = re.findall(r"\b[A-Z][a-z]+\b", hypothesis)
        support = 0.0
        evidence: List[str] = []

        for i, ea in enumerate(entities):
            for eb in entities[i + 1 :]:
                w = kg.g.get(ea, {}).get(eb, 0.0)
                if w > 0:
                    support += w
                    evidence.append(f"KG edge '{ea}→{eb}' weight={w:.2f}")

        support = min(support, 1.0)
        verdict = "support" if support > 0.5 else ("weak" if support > 0.2 else "reject")
        return {
            "hypothesis": hypothesis,
            "support_strength": round(support, 3),
            "verdict": verdict,
            "evidence": evidence,
        }
