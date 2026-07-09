# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TemporalKnowledgeGraph
=========================
Extends KnowledgeGraph with valid_from / valid_to metadata per triple,
enabling "before/after X year" consistency checks. Ported 1:1 from
engine_v1.py's ``TemporalKnowledgeGraph``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .knowledge_graph import KnowledgeGraph


class TemporalKnowledgeGraph(KnowledgeGraph):
    def __init__(self) -> None:
        super().__init__()
        # triple_hash -> {"valid_from": str|None, "valid_to": str|None}
        self._temporal_attrs: Dict[str, Dict[str, Optional[str]]] = {}

    def add_temporal(
        self,
        subj: str,
        rel: str,
        obj: str,
        w: float = 1.0,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> None:
        key = self._triple_key(subj, rel, obj)
        self._temporal_attrs[key] = {"valid_from": valid_from, "valid_to": valid_to}
        self.add(subj, obj, w)
        if key not in self._triple_hashes:
            self._triple_hashes.add(key)
            self._triples.append((subj, rel, obj, w))

    def is_valid_at(self, subj: str, rel: str, obj: str, year: int) -> Optional[bool]:
        """True if valid at given year, False if out of range, None if unknown."""
        key = self._triple_key(subj, rel, obj)
        attrs = self._temporal_attrs.get(key)
        if attrs is None:
            return None
        vf = attrs.get("valid_from")
        vt = attrs.get("valid_to")
        try:
            if vf and int(vf) > year:
                return False
            if vt and int(vt) < year:
                return False
        except (ValueError, TypeError):
            return None
        return True

    def check_temporal_consistency(self, answer: str) -> List[str]:
        """Scan answer for years and flag if they contradict known triples."""
        issues: List[str] = []
        year_matches = re.findall(r"\b(1\d{3}|20\d{2})\b", answer)
        for ys in year_matches:
            year = int(ys)
            if year < 1000 or year > 2100:
                continue
            for subj, rel, obj, _ in self._triples[:50]:  # cap scan
                result = self.is_valid_at(subj, rel, obj, year)
                if result is False:
                    issues.append(f"Temporal conflict: '{subj} {rel} {obj}' not valid in {year}")
        return issues
