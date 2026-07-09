# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""CorrelationBenchmark — verifies deduplication actually reduces item
count on near-duplicate inputs, and that ConnectedComponents finds
transitive clusters that GreedyClusterer might miss."""

from __future__ import annotations

from typing import Dict

from ..correlation.clustering import GreedyClusterer
from ..correlation.connected_components import ConnectedComponentsClusterer


class CorrelationBenchmark:
    def run(self) -> Dict[str, object]:
        items = [
            "DracoAI uses Mixture of Experts architecture",
            "DracoAI architecture uses Mixture of Experts",  # near-duplicate of 0
            "B C D E F G H",                                 # overlaps with item 2 (chain link)
            "C D E F G H I",                                 # overlaps with item 1 above (transitive)
            "totally unrelated sentence about cats",
        ]
        gc = GreedyClusterer(threshold=0.35)
        cc = ConnectedComponentsClusterer(threshold=0.35)
        gc_groups = gc.correlate(items)
        cc_groups = cc.correlate(items)
        return {
            "n_items": len(items),
            "greedy_n_groups": len(gc_groups),
            "cc_n_groups": len(cc_groups),
            "dedup_reduces_items": len(gc_groups) < len(items),
            "cc_found_transitive_chain": any(len(g.member_indices) >= 3 for g in cc_groups),
        }
