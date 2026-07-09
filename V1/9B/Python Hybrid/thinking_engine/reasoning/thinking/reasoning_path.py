# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ReasoningPathComputer
========================
KG path reasoning — ported from the
``ThinkingEngineV1._compute_reasoning_path`` helper. Tries A* first
(prefers strongly-related edges), falls back to BFS, then to a
2-hop neighbourhood listing when only one entity is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph


class ReasoningPathComputer:
    def compute(self, kg: "KnowledgeGraph", entities: List[str]) -> List[str]:
        """Read-only — safe to call concurrently with other readers as
        long as no concurrent KG write is in flight (callers should
        await any pending KG-extraction future first; see
        reasoning/execution/controller.py for the locking pattern)."""
        if len(entities) >= 2:
            path, _ = kg.astar(entities[0], entities[1])
            if not path:
                path = kg.bfs(entities[0], entities[1])
            return path if path else []
        elif entities:
            return list(kg.related(entities[0], hops=2).keys())[:4]
        return []
