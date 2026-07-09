# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.knowledge — contract for graph/RAG knowledge backends."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class KnowledgeStore(Protocol):
    def add(self, a: str, b: str, w: float = 1.0) -> None:
        ...

    def bfs(self, src: str, dst: str) -> Optional[List[str]]:
        ...

    def related(self, concept: str, hops: int = 2) -> Dict[str, int]:
        ...
