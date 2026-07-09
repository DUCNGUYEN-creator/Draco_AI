# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""RunningStats — Welford's online mean/variance accumulator, used by
metrics/drift.py and benchmarks/* to track verifier-score distributions
over time without storing every sample."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return self.variance ** 0.5
