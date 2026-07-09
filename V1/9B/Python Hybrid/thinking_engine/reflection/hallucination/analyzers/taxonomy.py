# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
HallucinationTaxonomy
========================
Classifies WHICH KIND of hallucination a low-scoring claim represents,
based on which verifier(s) flagged it. Standard taxonomy from the
hallucination-detection literature, mapped onto this engine's specific
verifier set.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class HallucinationType(str, Enum):
    UNSUPPORTED = "unsupported"              # RetrievalVerifier: no evidence at all
    CONTRADICTORY = "contradictory"          # ContradictionVerifier: actively conflicts
    SELF_INCONSISTENT = "self_inconsistent"  # ConsistencyVerifier: disagrees with own other paths
    NUMERICAL_ERROR = "numerical_error"      # NumericalVerifier: wrong arithmetic
    LOGICAL_ERROR = "logical_error"          # SymbolicVerifier: invalid formal logic claim
    FABRICATED_CITATION = "fabricated_citation"  # CitationVerifier: cites nonexistent source
    PLAN_DEVIATION = "plan_deviation"        # PlannerVerifier: ignores committed plan
    TOOL_MISMATCH = "tool_mismatch"          # ToolVerifier: contradicts actual tool output
    NON_SEQUITUR = "non_sequitur"            # ReasoningVerifier: doesn't follow from reasoning trace
    UNKNOWN = "unknown"


_VERIFIER_TO_TYPE: Dict[str, HallucinationType] = {
    "retrieval": HallucinationType.UNSUPPORTED,
    "contradiction": HallucinationType.CONTRADICTORY,
    "consistency": HallucinationType.SELF_INCONSISTENT,
    "numerical": HallucinationType.NUMERICAL_ERROR,
    "symbolic": HallucinationType.LOGICAL_ERROR,
    "citation": HallucinationType.FABRICATED_CITATION,
    "planner": HallucinationType.PLAN_DEVIATION,
    "tool": HallucinationType.TOOL_MISMATCH,
    "reasoning": HallucinationType.NON_SEQUITUR,
}

_FAILURE_THRESHOLD = 0.4  # a verifier "flags" a claim when its score drops below this


class HallucinationTaxonomy:
    def classify(self, verification_results: List[dict]) -> List[HallucinationType]:
        types: List[HallucinationType] = []
        for r in verification_results:
            if r.get("score", 1.0) < _FAILURE_THRESHOLD and r.get("confidence", 0.0) > 0.3:
                t = _VERIFIER_TO_TYPE.get(r.get("verifier", ""), HallucinationType.UNKNOWN)
                if t not in types:
                    types.append(t)
        return types or ([] if not verification_results else [])
