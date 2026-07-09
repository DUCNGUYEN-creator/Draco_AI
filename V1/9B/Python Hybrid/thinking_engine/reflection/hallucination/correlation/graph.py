# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CorrelationGraph
===================
Builds an explicit similarity GRAPH (edge = similarity above threshold)
over a set of evidence texts — the structure ConnectedComponentsClusterer
operates on. Kept separate from GreedyClusterer because GreedyClusterer
is single-linkage-by-construction (whoever it visits first "claims" a
group), while building the full graph first lets
ConnectedComponentsClusterer find TRANSITIVE clusters (A~B, B~C, but
A!~C directly — still one cluster) that greedy clustering can miss
depending on iteration order.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .similarity import SimilarityScorer


class CorrelationGraph:
    def __init__(self, scorer: SimilarityScorer | None = None) -> None:
        self.scorer = scorer or SimilarityScorer()

    def build(self, items: List[str], threshold: float = 0.82) -> Dict[int, Dict[int, float]]:
        n = len(items)
        adj: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.scorer.jaccard(items[i], items[j])
                if sim >= threshold:
                    adj[i][j] = sim
                    adj[j][i] = sim
        return adj
