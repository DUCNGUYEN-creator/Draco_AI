# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CitationTracker
==================
Tracks which retrieved documents were actually used in a response so
the Hallucination verification layer's verifiers/citation.py can later
check that every cited claim maps back to a real, retrieved source.
"""

from __future__ import annotations

from typing import Dict, List

from ..utils.hashing import short_hash


class CitationTracker:
    def __init__(self) -> None:
        self._registry: Dict[str, dict] = {}

    def register(self, doc: dict) -> str:
        cid = short_hash(doc.get("text", "")[:200])
        self._registry[cid] = doc
        return cid

    def get(self, citation_id: str) -> dict:
        return self._registry.get(citation_id, {})

    def format_all(self, docs: List[dict]) -> List[str]:
        out = []
        for d in docs:
            cid = short_hash(d.get("text", "")[:200])
            out.append(f"[{cid}] {d.get('text', '')[:80]}")
        return out
