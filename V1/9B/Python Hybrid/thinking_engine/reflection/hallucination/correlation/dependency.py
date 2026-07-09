# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierDependencyGraph
==========================
Correlation-layer wrapper around
analyzers.dependency.DependencyAnalyzer's known-pairs table, expressed
as an explicit weighted graph (rather than a flat pair-list) so
fusion/* can ask richer questions like "what's the TOTAL correlation
mass touching verifier X" instead of only "is X paired with Y".
"""

from __future__ import annotations

from typing import Dict, List

from ..analyzers.dependency import _KNOWN_DEPENDENT_PAIRS


class VerifierDependencyGraph:
    def build(self, present_verifiers: List[str]) -> Dict[str, Dict[str, float]]:
        present = set(present_verifiers)
        adj: Dict[str, Dict[str, float]] = {v: {} for v in present}
        for a, b in _KNOWN_DEPENDENT_PAIRS:
            if a in present and b in present:
                # Fixed correlation weight for known-dependent pairs — a
                # richer version could vary this per pair, but a flat
                # weight keeps the discount predictable and easy to reason about.
                adj[a][b] = 0.6
                adj[b][a] = 0.6
        return adj

    def total_correlation_mass(self, adj: Dict[str, Dict[str, float]], verifier: str) -> float:
        return sum(adj.get(verifier, {}).values())
