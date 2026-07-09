# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""CorrelationGroup — a cluster of evidence/verifications deemed
non-independent (near-duplicate or causally linked), used to avoid
double-counting correlated signals during fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CorrelationGroup:
    member_indices: List[int] = field(default_factory=list)
    representative_index: int = -1
    similarity: float = 0.0

    def size(self) -> int:
        return len(self.member_indices)

    def is_singleton(self) -> bool:
        return self.size() <= 1
