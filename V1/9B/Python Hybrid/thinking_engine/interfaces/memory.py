# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.memory — contract for any memory backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    def add(self, key: str, value: Any, importance: float = 1.0) -> None:
        ...

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...

    def forget(self, key: str) -> None:
        ...

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        ...
