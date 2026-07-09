# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
AnalogicalMapper
==================
KG-grounded analogy solver: A:B :: C:?. Ported 1:1 from engine_v1.py's
``AnalogicalMapper``, including the ANALOGY-FIX (concept_a guides
candidate selection via the A-B structural signature) and the
weight-threshold guard against low-confidence analogies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph

_ANALOGY_MIN_WEIGHT = 0.3


class AnalogicalMapper:
    def find_analogy(
        self,
        kg: "KnowledgeGraph",
        concept_a: str,
        concept_b: str,
        concept_c: str,
    ) -> Optional[str]:
        """A:B :: C:? — find X such that C->X mirrors A->B structurally."""
        related_a = set(kg.related(concept_a, hops=2).keys())
        related_b = set(kg.related(concept_b, hops=2).keys())
        ab_signature = related_a & related_b

        related_c = set(kg.related(concept_c, hops=2).keys())

        candidates = related_c & ab_signature if ab_signature else set()
        if not candidates:
            candidates = related_c & related_b
        if not candidates:
            candidates = set(kg.related(concept_c, hops=1).keys())
        if not candidates:
            return None

        best = max(candidates, key=lambda n: kg.g.get(concept_c, {}).get(n, 0.0))
        best_weight = kg.g.get(concept_c, {}).get(best, 0.0)
        if best_weight < _ANALOGY_MIN_WEIGHT:
            return None
        return best

    def describe_analogy(self, a: str, b: str, c: str, x: Optional[str]) -> str:
        if x is None:
            return f"[ANALOGY] {a}:{b} :: {c}:? — No analogy found in knowledge graph."
        return f"[ANALOGY] {a}:{b} :: {c}:{x} — {c} relates to {x} as {a} relates to {b}."
