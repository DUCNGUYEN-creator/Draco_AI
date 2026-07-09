# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ForgettingMechanism
======================
Ebbinghaus-style LTM decay + spaced-repetition via access_count. Ported
1:1 from engine_v1.py's ``ForgettingMechanism``.
"""

from __future__ import annotations

import math
import time
from typing import List


class ForgettingMechanism:
    def __init__(self, decay_rate: float = 0.1, forget_threshold: float = 0.05):
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold

    def tick(self, facts: List[dict]) -> List[dict]:
        """Run one decay cycle. Returns surviving facts (importance above
        threshold). Call after every N turns or periodically."""
        now = time.time()
        alive = []
        for f in facts:
            elapsed = (now - f.get("last_access", now)) / 3600.0  # hours
            decay = math.exp(-self.decay_rate * elapsed)
            f["importance"] = f.get("importance", 1.0) * decay
            if f["importance"] >= self.forget_threshold:
                alive.append(f)
        return alive

    def access(self, fact: dict) -> None:
        """Record that a fact was accessed — boosts importance (spaced repetition)."""
        fact["access_count"] = fact.get("access_count", 0) + 1
        fact["last_access"] = time.time()
        fact["importance"] = min(1.0, fact.get("importance", 0.5) + 0.15)
