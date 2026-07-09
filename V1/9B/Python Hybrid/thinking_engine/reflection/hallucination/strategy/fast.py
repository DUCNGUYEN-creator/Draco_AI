# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FastStrategy
==============
2 verifiers — the cheapest meaningful check: RetrievalVerifier (catches
the most common failure mode, "no evidence at all") and
ContradictionVerifier (catches the most damaging failure mode, "actively
wrong"). Use for low-latency paths (e.g. INTENT_CHAT, simple factual
lookups) where running the full ensemble would dominate response time.
"""

from __future__ import annotations

from typing import List


class FastStrategy:
    name = "fast"
    verifier_names: List[str] = ["retrieval", "contradiction"]
    fusion_method: str = "noisy_or"
    calibration_method: str = "temperature"  # fastest-converging, lowest sample requirement
