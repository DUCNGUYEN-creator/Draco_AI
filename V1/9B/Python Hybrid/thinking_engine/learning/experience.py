# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExperienceBuffer
==================
Rolling buffer of (question, answer, rating) triples used by Evaluator
and future RL/RLHF-style fine-tuning loops. Analogous to a replay
buffer in deep RL — not consumed by the per-request hot path.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class ExperienceBuffer:
    def __init__(self, max_size: int = 10000) -> None:
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=max_size)

    def add(self, experience: Dict[str, Any]) -> None:
        self._buffer.append(experience)

    def sample(self, n: int) -> List[Dict[str, Any]]:
        buf = list(self._buffer)
        import random
        return random.sample(buf, min(n, len(buf)))

    def __len__(self) -> int:
        return len(self._buffer)
