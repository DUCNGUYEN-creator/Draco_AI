# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
KnowledgeReranker
====================
Lightweight relevance reranker for RAG documents — distinct from
memory/memory_reranker.py (which reranks conversational memories using
intent keywords + recency). This one scores word-overlap relevance
against the query, suitable for retrieved knowledge passages.
"""

from __future__ import annotations

from typing import List


class KnowledgeReranker:
    def rerank(self, docs: List[dict], query: str, top_k: int = 3) -> List[dict]:
        if not docs:
            return []
        ql = set(query.lower().split())
        scored = []
        for d in docs:
            text = d.get("text", "").lower()
            overlap = len(ql & set(text.split()))
            base_score = d.get("score", 0.0)
            final = base_score * 0.6 + min(overlap * 0.1, 0.4)
            nd = dict(d)
            nd["rerank_score"] = final
            scored.append(nd)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def format_context(self, docs: List[dict], max_chars: int = 800) -> str:
        parts: List[str] = []
        total = 0
        for d in docs:
            t = d.get("text", "")
            if not t:
                continue
            parts.append(t[:200])
            total += len(t[:200])
            if total > max_chars:
                break
        return "\n".join(parts)
