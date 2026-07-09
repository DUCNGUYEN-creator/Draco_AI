# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
build_consensus_report
========================
New addition. Produces a structured (non-string) summary of a debate
round — which experts agreed, which dissented, and the meaningful
shared-keyword set — so the Verification layer can later assess
"how contested was this answer" as one hallucination-risk signal
(reflection/hallucination/verifiers/consistency.py) without re-parsing
the Arbitrator's free-text output.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .expert import EXPERT_NAMES
from .voting import ConsensusChecker


def build_consensus_report(thoughts: Dict[int, str], threshold: float = 0.75) -> Dict[str, Any]:
    checker = ConsensusChecker()
    consensus_reached = checker.check(thoughts.values(), threshold)

    kw_sets = {eid: set(t.lower().split()) for eid, t in thoughts.items()}
    all_sets = list(kw_sets.values())
    shared = set.intersection(*all_sets) if all_sets else set()
    meaningful_shared = sorted(w for w in shared if len(w) > 3)

    dissenters: List[int] = []
    for eid, words in kw_sets.items():
        overlap = len(words & shared)
        if overlap < max(1, len(shared) * 0.3):
            dissenters.append(eid)

    return {
        "consensus_reached": consensus_reached,
        "n_experts": len(thoughts),
        "shared_keywords": meaningful_shared,
        "dissenting_experts": [EXPERT_NAMES.get(e, f"Expert{e}") for e in dissenters],
    }
