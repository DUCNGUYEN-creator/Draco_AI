# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SeverityAnalyzer
===================
Scores how SEVERE a detected hallucination is, independent of how
LIKELY it is. A confidently-wrong numerical claim in a financial report
is more severe than a vague hedge that turns out unsupported. Severity
is a multiplier applied downstream (risk/report.py), not a replacement
for the fused probability.
"""

from __future__ import annotations

from typing import Dict, List

from .taxonomy import HallucinationType

# Domain-informed severity weights — numerical/tool mismatches are most
# severe (concrete, checkable, high user-trust impact); plan deviation
# is least severe (often just stylistic drift, not factual harm).
_SEVERITY_WEIGHTS: Dict[HallucinationType, float] = {
    HallucinationType.NUMERICAL_ERROR: 0.95,
    HallucinationType.TOOL_MISMATCH: 0.95,
    HallucinationType.FABRICATED_CITATION: 0.9,
    HallucinationType.CONTRADICTORY: 0.85,
    HallucinationType.LOGICAL_ERROR: 0.8,
    HallucinationType.UNSUPPORTED: 0.6,
    HallucinationType.NON_SEQUITUR: 0.55,
    HallucinationType.SELF_INCONSISTENT: 0.5,
    HallucinationType.PLAN_DEVIATION: 0.3,
    HallucinationType.UNKNOWN: 0.5,
}


class SeverityAnalyzer:
    def severity_for_types(self, types: List[HallucinationType]) -> float:
        if not types:
            return 0.0
        return max(_SEVERITY_WEIGHTS.get(t, 0.5) for t in types)

    def severity_breakdown(self, types: List[HallucinationType]) -> Dict[str, float]:
        return {t.value: _SEVERITY_WEIGHTS.get(t, 0.5) for t in types}
