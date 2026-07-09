# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
LongTermMemory
=================
Persists important facts across sessions, applying ForgettingMechanism
decay on each tick(). This is the "ltm_facts" concept threaded through
engine_v1.py's ``process()`` and ``recursive_critique()`` — formalized
here as its own addressable store instead of a bare List[dict] the
caller has to manage themselves.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from .forgetting import ForgettingMechanism


class LongTermMemory:
    def __init__(self, decay_rate: float = 0.1, forget_threshold: float = 0.05) -> None:
        self._facts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._forgetting = ForgettingMechanism(decay_rate, forget_threshold)

    def remember(self, key: str, value: Any, importance: float = 1.0) -> None:
        with self._lock:
            for f in self._facts:
                if f.get("key") == key:
                    f["value"] = value
                    f["importance"] = max(f.get("importance", 0.0), importance)
                    f["last_access"] = time.time()
                    return
            self._facts.append(
                {
                    "key": key,
                    "value": value,
                    "importance": importance,
                    "access_count": 0,
                    "last_access": time.time(),
                }
            )

    def tick(self) -> None:
        """Run one Ebbinghaus decay cycle, dropping facts below threshold."""
        with self._lock:
            self._facts = self._forgetting.tick(self._facts)

    def access(self, key: str) -> Any:
        with self._lock:
            for f in self._facts:
                if f.get("key") == key:
                    self._forgetting.access(f)
                    return f.get("value")
        return None

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(f) for f in self._facts]
