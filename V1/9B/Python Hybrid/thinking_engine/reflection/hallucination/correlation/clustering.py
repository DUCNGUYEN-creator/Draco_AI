# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
GreedyClusterer
==================
O(n^2) greedy single-linkage clustering of evidence texts by Jaccard
similarity — simple and predictable, the right choice for the small
evidence-bundle sizes (typically <20 items per claim) this engine deals
with. ConnectedComponentsClusterer (graph.py-backed) is the alternative
implementation when correlation needs to be transitive across more than
pairwise comparisons.
"""

from __future__ import annotations

from typing import Any, List

from ..models.correlation import CorrelationGroup
from .base import BaseCorrelator
from .similarity import SimilarityScorer


class GreedyClusterer(BaseCorrelator):
    def __init__(self, threshold: float = 0.82, scorer: SimilarityScorer | None = None) -> None:
        self.threshold = threshold
        self.scorer = scorer or SimilarityScorer()

    def correlate(self, items: List[str]) -> List[CorrelationGroup]:
        groups: List[CorrelationGroup] = []
        assigned = [False] * len(items)

        for i, text_i in enumerate(items):
            if assigned[i]:
                continue
            group = CorrelationGroup(member_indices=[i], representative_index=i, similarity=1.0)
            assigned[i] = True
            for j in range(i + 1, len(items)):
                if assigned[j]:
                    continue
                sim = self.scorer.jaccard(text_i, items[j])
                if sim >= self.threshold:
                    group.member_indices.append(j)
                    group.similarity = min(group.similarity, sim)
                    assigned[j] = True
            groups.append(group)
        return groups
