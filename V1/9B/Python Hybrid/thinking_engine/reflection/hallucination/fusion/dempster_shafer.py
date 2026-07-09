# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DempsterShaferFusion
=======================
Dempster-Shafer evidence combination over a simplified 2-element frame
{Hallucination, NotHallucination}, with explicit mass-on-Uncertainty
for low-confidence verifiers — the right tool when verifiers can
genuinely ABSTAIN (assign mass to "I don't know") rather than being
forced to vote 0.5 like the other fusion methods treat abstention.
Captures epistemic uncertainty separately from aleatory disagreement,
which the report.py can surface as "verifiers mostly didn't have an
opinion" vs "verifiers actively disagreed".
"""

from __future__ import annotations

from typing import List, Tuple

from ..models.fusion import FusionResult
from .base import BaseFusionStrategy


def _combine_two(m1: dict, m2: dict) -> dict:
    """Dempster's rule of combination for two basic probability
    assignments over {H, NH, U} (Hallucination / NotHallucination /
    Uncertain-frame-wide)."""
    keys = ("H", "NH", "U")
    raw = {k: 0.0 for k in keys}
    conflict = 0.0
    for k1 in keys:
        for k2 in keys:
            product = m1[k1] * m2[k2]
            if k1 == k2:
                raw[k1] += product
            elif "U" in (k1, k2):
                # U combined with anything yields that anything (U is
                # the "don't know" element — agrees with whatever the
                # more specific hypothesis says).
                target = k1 if k2 == "U" else k2
                raw[target] += product
            else:
                conflict += product  # H vs NH = direct conflict, discarded then renormalized
    norm = 1.0 - conflict
    if norm <= 1e-9:
        return {"H": 0.5, "NH": 0.5, "U": 0.0}
    return {k: v / norm for k, v in raw.items()}


class DempsterShaferFusion(BaseFusionStrategy):
    method_name = "dempster_shafer"

    def _to_mass(self, p: float, weight: float) -> dict:
        # weight controls how much mass goes to H/NH vs staying Uncertain —
        # a low-weight (low-confidence/discounted) verifier mostly votes "U".
        mass_h = p * weight
        mass_nh = (1.0 - p) * weight
        mass_u = 1.0 - weight
        return {"H": mass_h, "NH": mass_nh, "U": mass_u}

    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        if not signals:
            return self._empty_result()

        combined = {"H": 0.0, "NH": 0.0, "U": 1.0}  # start fully uncertain
        contributions = {}
        for name, p, w in signals:
            mass = self._to_mass(p, w)
            before_h = combined["H"]
            combined = _combine_two(combined, mass)
            contributions[name] = round(combined["H"] - before_h, 4)

        return FusionResult(
            method=self.method_name,
            fused_score=combined["H"],
            per_verifier_contribution=contributions,
            n_signals_used=len(signals),
            notes=[f"residual_uncertainty={combined['U']:.4f}"],
        )
