# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Thread-safety helpers — including the ``LockedBridge`` proxy ported
verbatim (in behaviour) from engine_v1.py's ``_LockedBridge``.
"""

from __future__ import annotations

import threading
from typing import Any, List


class LockedBridge:
    """Transparent proxy that serializes all bridge.generate() calls via a
    shared threading.Lock.

    WHY: NumPy-backend transformers mutate KVCache in-place during
    generate(). When the ThreadPoolExecutor reasoning stage runs several
    LLM-calling tasks concurrently (ToT, Council, GoalDecomposer,
    PlanDecomposer, ...) they all receive the same bridge instance and
    would race on the same cache without this proxy.
    """

    def __init__(self, bridge: Any, lock: threading.Lock) -> None:
        object.__setattr__(self, "_bridge", bridge)
        object.__setattr__(self, "_lock", lock)

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 256, **kwargs: Any) -> Any:
        lock = object.__getattribute__(self, "_lock")
        bridge = object.__getattribute__(self, "_bridge")
        with lock:
            return bridge.generate(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)

    def __getattr__(self, name: str) -> Any:
        bridge = object.__getattribute__(self, "_bridge")
        return getattr(bridge, name)

    def __repr__(self) -> str:
        bridge = object.__getattribute__(self, "_bridge")
        return f"LockedBridge({bridge!r})"


class NamedLocks:
    """A small registry of named locks — avoids each subsystem declaring
    its own ad-hoc threading.Lock() and forgetting to share it."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get(self, name: str) -> threading.Lock:
        with self._registry_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]
