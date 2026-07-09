# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
InductiveReasoner
====================
New addition. Generalizes from a list of specific KG-edge observations
to a candidate general rule ("most X are Y"), with a support ratio.
Complements HypothesisTester (which tests a single stated hypothesis)
by *proposing* a hypothesis from observed pattern frequency instead.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph


class InductiveReasoner:
    def generalize(self, kg: "KnowledgeGraph", concepts: List[str], hops: int = 1) -> List[Tuple[str, float]]:
        """Given several concepts, find neighbours common across the
        majority of them — a simple frequentist generalization. Returns
        [(candidate_generalization, support_ratio), ...] sorted descending."""
        if not concepts:
            return []
        counter: Counter = Counter()
        for c in concepts:
            for nb in kg.related(c, hops=hops):
                counter[nb] += 1
        n = len(concepts)
        results = [(nb, count / n) for nb, count in counter.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def induce_rule(self, kg: "KnowledgeGraph", concepts: List[str], min_support: float = 0.6) -> str:
        generalizations = self.generalize(kg, concepts)
        strong = [g for g, s in generalizations if s >= min_support]
        if not strong:
            return "[INDUCTION] No generalization reached the support threshold."
        return (
            f"[INDUCTION] Observed concepts {concepts} commonly relate to: "
            f"{', '.join(strong[:3])} (support >= {min_support:.0%})."
        )
