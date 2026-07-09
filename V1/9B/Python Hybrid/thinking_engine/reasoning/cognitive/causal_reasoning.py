# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CausalReasoner
================
New addition. Distinguishes mere correlation (a KG edge exists) from a
claimed causal relationship (edge created via a "causes"/"gây ra"
relation specifically — see knowledge/graph_extractor.py's relation
labels) and produces a structured causal-chain explanation. Used by the
DifficultyScorer-routed "why" intent path alongside AbductionEngine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph

_CAUSAL_RELATIONS = {"causes", "gây ra"}


class CausalReasoner:
    def explain_cause(self, kg: "KnowledgeGraph", cause: str, effect: str) -> Optional[str]:
        """Looks for an explicit causal triple (cause, "causes"/"gây ra",
        effect) recorded by graph_extractor.TripleExtractor; falls back
        to a plain-edge mention with a lower-confidence caveat."""
        for subj, rel, obj, w in getattr(kg, "_triples", []):
            if subj == cause and obj == effect and rel in _CAUSAL_RELATIONS:
                return f"[CAUSAL] '{cause}' causes '{effect}' (recorded relation, w={w:.2f})."
        w = kg.g.get(cause, {}).get(effect, 0.0)
        if w > 0:
            return (
                f"[CAUSAL — UNCONFIRMED] '{cause}' and '{effect}' are related "
                f"(w={w:.2f}) but no explicit causal relation was recorded — "
                f"this may be correlation, not causation."
            )
        return None

    def causal_chain(self, kg: "KnowledgeGraph", cause: str, effect: str, max_hops: int = 4) -> List[str]:
        """Best-effort multi-hop causal path via BFS over the graph,
        labelled as a correlation chain unless every hop is an explicit
        causal relation."""
        path = kg.bfs(cause, effect)
        if not path or len(path) - 1 > max_hops:
            return []
        causal_triples = {
            (s, o) for s, rel, o, _ in getattr(kg, "_triples", []) if rel in _CAUSAL_RELATIONS
        }
        all_causal = all((path[i], path[i + 1]) in causal_triples for i in range(len(path) - 1))
        label = "CAUSAL CHAIN" if all_causal else "ASSOCIATIVE CHAIN (not all hops causal)"
        return [f"[{label}]"] + path
