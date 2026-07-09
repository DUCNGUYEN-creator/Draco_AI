# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
LRUCache
==========
Thread-safe, TTL-aware Least-Recently-Used cache built on
collections.OrderedDict — the generic primitive every other cache/*.py
module wraps with a domain-specific key scheme.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    def __init__(self, max_size: int = 512, ttl_seconds: float = 600.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, tuple]" = OrderedDict()  # key -> (value, inserted_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, inserted_at = entry
            if (time.time() - inserted_at) > self.ttl_seconds:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            if len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
