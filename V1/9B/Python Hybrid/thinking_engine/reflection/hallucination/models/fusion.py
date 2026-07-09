# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""FusionResult — the combined risk estimate after merging (deduplicated,
decorrelated) verifier signals into one number."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FusionResult:
    method: str
    fused_score: float  # 0.0 (safe) .. 1.0 (high hallucination risk)
    per_verifier_contribution: Dict[str, float] = field(default_factory=dict)
    n_signals_used: int = 0
    n_signals_deduplicated: int = 0
    notes: List[str] = field(default_factory=list)
