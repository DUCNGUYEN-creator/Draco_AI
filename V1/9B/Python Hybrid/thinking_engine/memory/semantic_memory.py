# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SemanticMemory
=================
Holds extracted (subject, relation, object, confidence) facts —
conceptually adjacent to the KnowledgeGraph in knowledge/, but scoped
to *user-specific* learned facts (e.g. "user's project is called
DracoAI") rather than the engine's general-purpose world graph.
Kept as a flat, queryable fact-store rather than a graph because
user facts are usually queried by subject, not traversed multi-hop.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Tuple


class SemanticMemory:
    def __init__(self) -> None:
        self._facts: Dict[str, List[Tuple[str, str, float, float]]] = {}
        # subject -> list of (relation, obj, confidence, last_seen_ts)
        self._lock = threading.Lock()

    def add_fact(self, subject: str, relation: str, obj: str, confidence: float = 0.8) -> None:
        with self._lock:
            self._facts.setdefault(subject.lower(), [])
            entries = self._facts[subject.lower()]
            for i, (rel, o, _, _) in enumerate(entries):
                if rel == relation and o == obj:
                    entries[i] = (relation, obj, confidence, time.time())
                    return
            entries.append((relation, obj, confidence, time.time()))

    def get_facts(self, subject: str) -> List[Tuple[str, str, float, float]]:
        with self._lock:
            return list(self._facts.get(subject.lower(), []))

    def all_subjects(self) -> List[str]:
        with self._lock:
            return list(self._facts.keys())
