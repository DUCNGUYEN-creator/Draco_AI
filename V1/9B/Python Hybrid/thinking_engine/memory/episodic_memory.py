# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EpisodicMemory
================
Stores discrete "episodes" — (timestamp, question, answer, metadata)
records of complete past interactions, as opposed to WorkingMemory's
raw message buffer or SemanticMemory's fact triples. Supports simple
recency + substring-match retrieval; designed to be backed later by a
real vector store without changing this class's public surface.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    def __init__(self, max_episodes: int = 2000) -> None:
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        self._lock = threading.Lock()

    def record(self, question: str, answer: str, **metadata: Any) -> Episode:
        ep = Episode(question=question, answer=answer, metadata=metadata)
        with self._lock:
            self._episodes.append(ep)
            if len(self._episodes) > self.max_episodes:
                self._episodes = self._episodes[-self.max_episodes :]
        return ep

    def search(self, query: str, top_k: int = 5) -> List[Episode]:
        """Cheap substring/word-overlap relevance scoring — replace with a
        real embedding search when one is wired up."""
        ql = set(query.lower().split())
        with self._lock:
            episodes = list(self._episodes)
        scored = []
        for ep in episodes:
            words = set((ep.question + " " + ep.answer).lower().split())
            overlap = len(ql & words)
            if overlap > 0:
                scored.append((overlap, ep))
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def recent(self, n: int = 5) -> List[Episode]:
        with self._lock:
            return self._episodes[-n:]
