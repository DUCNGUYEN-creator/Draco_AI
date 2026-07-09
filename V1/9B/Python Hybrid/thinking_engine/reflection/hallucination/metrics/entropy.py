# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Shannon entropy helpers — raw nats and normalized [0,1] forms."""

from __future__ import annotations

import math
from typing import Iterable, List

from ....utils.math import entropy as _entropy


def shannon_entropy(probs: Iterable[float]) -> float:
    return _entropy(probs)


def normalized_entropy(probs: List[float]) -> float:
    """Entropy divided by max possible entropy (log(n)) — comparable
    across distributions of different sizes, unlike raw nats."""
    n = len(probs)
    if n <= 1:
        return 0.0
    max_ent = math.log(n)
    if max_ent <= 0:
        return 0.0
    return min(_entropy(probs) / max_ent, 1.0)
