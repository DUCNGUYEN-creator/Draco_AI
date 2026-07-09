# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DependencyAnalyzer
=====================
Flags when two verifiers' signals are NOT independent — e.g.
ToolVerifier and NumericalVerifier both effectively re-check the same
arithmetic fact when a calculator tool was used for a math claim.
Counting both as independent "votes" during fusion would overstate
confidence (double-counting). This analyzer surfaces such pairs so
fusion/* or correlation/* can down-weight one of them.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# Known structurally-dependent verifier pairs in THIS engine's design —
# both ultimately derive their signal from the same underlying fact
# source, so their scores are correlated, not independent.
_KNOWN_DEPENDENT_PAIRS: Set[Tuple[str, str]] = {
    ("numerical", "tool"),
    ("symbolic", "reasoning"),
    ("retrieval", "citation"),
}


class DependencyAnalyzer:
    def find_dependent_pairs(self, verification_results: List[dict]) -> List[Tuple[str, str]]:
        present = {r.get("verifier") for r in verification_results if r.get("confidence", 0.0) > 0.1}
        found = []
        for a, b in _KNOWN_DEPENDENT_PAIRS:
            if a in present and b in present:
                found.append((a, b))
        return found

    def independence_discount(self, verification_results: List[dict]) -> float:
        """Returns a multiplier in (0, 1] for how much to discount the
        combined evidence weight due to detected dependencies — more
        dependent pairs found => more discount (signals overlap more)."""
        pairs = self.find_dependent_pairs(verification_results)
        return max(0.5, 1.0 - 0.15 * len(pairs))
