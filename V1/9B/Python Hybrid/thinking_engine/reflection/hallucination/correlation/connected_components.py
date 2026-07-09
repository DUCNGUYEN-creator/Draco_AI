# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConnectedComponentsClusterer
===============================
Finds TRANSITIVE correlation clusters via Union-Find over a
CorrelationGraph's edges — catches chains of near-duplicate evidence
(A~B~C) that GreedyClusterer's single-pass visitation order might split
into two groups depending on which item it happens to visit first.
"""

from __future__ import annotations

from typing import Any, List

from ....utils.graph import UnionFind
from ..models.correlation import CorrelationGroup
from .base import BaseCorrelator
from .graph import CorrelationGraph
from .similarity import SimilarityScorer


class ConnectedComponentsClusterer(BaseCorrelator):
    def __init__(self, threshold: float = 0.82, scorer: SimilarityScorer | None = None) -> None:
        self.threshold = threshold
        self.scorer = scorer or SimilarityScorer()
        self._graph_builder = CorrelationGraph(self.scorer)

    def correlate(self, items: List[str]) -> List[CorrelationGroup]:
        n = len(items)
        if n == 0:
            return []
        adj = self._graph_builder.build(items, self.threshold)
        uf = UnionFind(range(n))
        for i, neighbors in adj.items():
            for j in neighbors:
                uf.union(i, j)

        components = uf.components()
        groups: List[CorrelationGroup] = []
        for comp in components:
            members = sorted(comp)
            rep = members[0]
            # Min similarity across all edges actually inside this
            # component — reports the WEAKEST link, a conservative
            # estimate of how tightly correlated the whole group really is.
            sims = [adj[a].get(b) for a in members for b in members if b in adj.get(a, {})]
            sims = [s for s in sims if s is not None]
            groups.append(
                CorrelationGroup(
                    member_indices=members,
                    representative_index=rep,
                    similarity=min(sims) if sims else 1.0,
                )
            )
        return groups
