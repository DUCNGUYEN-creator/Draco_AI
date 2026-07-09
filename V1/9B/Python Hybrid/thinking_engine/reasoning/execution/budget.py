# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ReasoningBudget
=================
A simple step/time budget object threaded through the reasoning loop so
no sub-system can run unboundedly — formalizes the various hardcoded
caps engine_v1.py sprinkled through MCTS (max_rollout_depth=10), debate
(max_rounds=3), and goal decomposition (rollout_depth=20) into one
inspectable object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ReasoningBudget:
    max_steps: int = 50
    max_seconds: float = 30.0
    _steps_used: int = 0
    _start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.time()
        self._steps_used = 0

    def consume(self, steps: int = 1) -> None:
        self._steps_used += steps

    def exhausted(self) -> bool:
        if self._steps_used >= self.max_steps:
            return True
        if self._start_time and (time.time() - self._start_time) >= self.max_seconds:
            return True
        return False

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self._steps_used)
