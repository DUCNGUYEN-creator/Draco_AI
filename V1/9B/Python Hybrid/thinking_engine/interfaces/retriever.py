# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.retriever — contract for RAG-style retrieval augmenters."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def is_applicable(self, intent: Dict[str, Any]) -> bool:
        ...

    def retrieve(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...
