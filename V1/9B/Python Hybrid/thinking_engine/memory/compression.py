# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MemoryCompressor
===================
Deduplicates and shrinks a list of memory-candidate dicts before they
reach MemoryReranker — drops near-duplicate text entries (same first-N
chars) to keep reranking cheap on large candidate sets.
"""

from __future__ import annotations

from typing import List


class MemoryCompressor:
    def compress(self, candidates: List[dict], dedup_prefix_len: int = 40, max_items: int = 50) -> List[dict]:
        seen = set()
        out: List[dict] = []
        for c in candidates:
            key = c.get("text", "")[:dedup_prefix_len].lower().strip()
            if key and key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= max_items:
                break
        return out
