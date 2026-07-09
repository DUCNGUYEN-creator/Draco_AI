# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CustomStrategy
================
User/caller-specified verifier set + fusion/calibration methods —
escape hatch for callers who know exactly which signals they want
(e.g. a code-review-focused deployment that only cares about
"numerical" + "symbolic" + "tool" verifiers).
"""

from __future__ import annotations

from typing import List


class CustomStrategy:
    name = "custom"

    def __init__(
        self,
        verifier_names: List[str] | None = None,
        fusion_method: str = "weighted",
        calibration_method: str = "platt",
    ) -> None:
        self.verifier_names = verifier_names or ["retrieval", "contradiction"]
        self.fusion_method = fusion_method
        self.calibration_method = calibration_method
