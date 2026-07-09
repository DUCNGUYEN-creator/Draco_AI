# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PolicyChecker
================
Aggregates EthicalFilter + InjectionDetector into one pass/fail policy
decision for a piece of text (either user input or model output),
giving Reflection a single call instead of orchestrating both safety
checks itself.
"""

from __future__ import annotations

from typing import Any, Dict

from .ethical_filter import EthicalFilter
from .injection_detector import InjectionDetector


class PolicyChecker:
    def __init__(self) -> None:
        self.ethical_filter = EthicalFilter()
        self.injection_detector = InjectionDetector()

    def check(self, text: str) -> Dict[str, Any]:
        ethical_score = self.ethical_filter.score(text)
        injection_report = self.injection_detector.detect(text)
        passed = ethical_score < 0.3 and not injection_report["is_suspicious"]
        return {
            "passed": passed,
            "ethical_score": ethical_score,
            "injection": injection_report,
        }
