# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SourceManager
===============
Tracks the provenance (origin) of every document/triple that entered
the engine — RAG passage, user-uploaded file, conversation memory, or
the static init_default() KG seed. Hallucination/verifiers/retrieval.py
consults this to weight evidence by source trustworthiness.
"""

from __future__ import annotations

from typing import Dict, Optional

SOURCE_RAG = "rag"
SOURCE_MEMORY = "memory"
SOURCE_USER_UPLOAD = "user_upload"
SOURCE_SEED_KG = "seed_kg"
SOURCE_TOOL = "tool"

_DEFAULT_TRUST: Dict[str, float] = {
    SOURCE_RAG: 0.7,
    SOURCE_MEMORY: 0.6,
    SOURCE_USER_UPLOAD: 0.85,
    SOURCE_SEED_KG: 0.9,
    SOURCE_TOOL: 0.95,
}


class SourceManager:
    def __init__(self) -> None:
        self._trust = dict(_DEFAULT_TRUST)
        self._provenance: Dict[str, str] = {}

    def register(self, item_id: str, source: str) -> None:
        self._provenance[item_id] = source

    def trust_score(self, item_id: str) -> float:
        source = self._provenance.get(item_id)
        if source is None:
            return 0.5  # unknown provenance — neutral trust
        return self._trust.get(source, 0.5)

    def set_trust(self, source: str, score: float) -> None:
        self._trust[source] = max(0.0, min(1.0, score))
