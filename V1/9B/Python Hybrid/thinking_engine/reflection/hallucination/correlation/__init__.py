# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.correlation
========================================================
Detects when multiple pieces of evidence or multiple verifier signals
are NOT independent (near-duplicate passages, structurally-dependent
verifier pairs) so fusion/* doesn't double-count correlated signals as
if they were independent confirmations. This is the "↓ Correlation"
step in the canonical Evidence → Verification → Calibration →
Correlation → Fusion → Risk → Report pipeline.
"""

from .base import BaseCorrelator
from .similarity import SimilarityScorer
from .deduplication import EvidenceDeduplicator
from .dependency import VerifierDependencyGraph
from .clustering import GreedyClusterer
from .connected_components import ConnectedComponentsClusterer
from .graph import CorrelationGraph
from .reducer import EvidenceReducer

__all__ = [
    "BaseCorrelator",
    "SimilarityScorer",
    "EvidenceDeduplicator",
    "VerifierDependencyGraph",
    "GreedyClusterer",
    "ConnectedComponentsClusterer",
    "CorrelationGraph",
    "EvidenceReducer",
]
