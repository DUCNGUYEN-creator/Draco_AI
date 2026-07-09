# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""A tiny context-manager timer used for telemetry latency measurement."""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    """Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000.0


def now_ts() -> float:
    return time.time()
