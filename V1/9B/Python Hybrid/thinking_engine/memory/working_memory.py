# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
WorkingMemory
===============
The short-lived, in-process scratch buffer for the *current* turn — the
active conversation window the engine is reasoning over right now.
Distinct from LongTermMemory (persisted, decayed, recalled across
sessions): WorkingMemory simply caps how many recent turns are held in
RAM, mirroring engine_v1.py's "managed messages" concept but as its own
addressable component instead of being baked into ContextWindowManager.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List


class WorkingMemory:
    def __init__(self, max_messages: int = 30) -> None:
        self.max_messages = max_messages
        self._buffer: Deque[Dict] = deque(maxlen=max_messages)
        self._lock = threading.Lock()

    def append(self, message: Dict) -> None:
        with self._lock:
            self._buffer.append(message)

    def extend(self, messages: List[Dict]) -> None:
        with self._lock:
            self._buffer.extend(messages)

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
