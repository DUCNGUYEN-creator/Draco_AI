# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DeductiveReasoner
====================
New addition. Applies formal modus-ponens-style chaining over KG edges:
given "A relates to B" and "B relates to C" (both above a confidence
threshold), deduces "A transitively relates to C" with combined
confidence = w(A,B) * w(B,C). This is a STRICT logical step generator
for reasoning — it never judges whether the deduced claim is actually
TRUE in the world (that's the Verification layer's job).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph


class DeductiveReasoner:
    def chain(
        self, kg: "KnowledgeGraph", a: str, b: str, c: str, min_weight: float = 0.3
    ) -> Optional[Tuple[float, str]]:
        """Deduce A -> C via the intermediate B. Returns (confidence, explanation) or None."""
        w_ab = kg.g.get(a, {}).get(b, 0.0)
        w_bc = kg.g.get(b, {}).get(c, 0.0)
        if w_ab < min_weight or w_bc < min_weight:
            return None
        confidence = w_ab * w_bc
        explanation = (
            f"[DEDUCTION] {a} relates to {b} (w={w_ab:.2f}), "
            f"{b} relates to {c} (w={w_bc:.2f}) "
            f"=> {a} transitively relates to {c} (confidence={confidence:.2f})."
        )
        return confidence, explanation

    def find_chains(
        self, kg: "KnowledgeGraph", a: str, c: str, min_weight: float = 0.3
    ) -> List[Tuple[float, str]]:
        """Search all intermediate B nodes connecting A to C."""
        results: List[Tuple[float, str]] = []
        for b in kg.g.get(a, {}):
            if b == c:
                continue
            chained = self.chain(kg, a, b, c, min_weight)
            if chained:
                results.append(chained)
        results.sort(key=lambda x: x[0], reverse=True)
        return results
