# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ParanoidStrategy
===================
Runs every registered verifier — for high-stakes claims (financial,
medical, legal-adjacent content; or whenever CoverageAnalyzer/strategy
escalation decides the balanced set didn't reach sufficient coverage).
Uses the slower-but-more-robust ensemble fusion + calibration methods
since latency is no longer the binding constraint at this tier.
"""

from __future__ import annotations

from typing import List

from ..registry.verifier_registry import VerifierRegistry


class ParanoidStrategy:
    name = "paranoid"
    fusion_method: str = "ensemble"
    calibration_method: str = "ensemble"

    @property
    def verifier_names(self) -> List[str]:
        # Pull every currently-registered verifier name rather than a
        # hardcoded list — Paranoid should automatically pick up any new
        # verifier registered via registry.VerifierRegistry.register(),
        # without this file needing an edit (the whole point of the
        # plugin registry design).
        return VerifierRegistry().available()
