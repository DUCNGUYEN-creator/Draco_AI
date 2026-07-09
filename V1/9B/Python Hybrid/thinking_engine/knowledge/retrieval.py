# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
RetrievalAugmenter (RAG hook)
==============================
Documented stub for INTENT_FACTUAL / INTENT_HOW_TO retrieval-augmented
generation. Ported 1:1 from engine_v1.py's ``RetrievalAugmenter``.

To activate real RAG, replace ``retrieve()``'s body with:
    vec  = embedder.encode(query)
    docs = vector_store.search(vec, top_k)
    return [{"text": d.text, "score": d.score, "ts": d.timestamp} for d in docs]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ..constants import INTENT_FACTUAL, INTENT_HOW_TO

if TYPE_CHECKING:  # pragma: no cover
    from ..memory.memory_reranker import MemoryReranker

_TRIGGER_INTENTS = {INTENT_FACTUAL, INTENT_HOW_TO}


class RetrievalAugmenter:
    def is_applicable(self, intent: Dict[str, Any]) -> bool:
        return intent.get("intent") in _TRIGGER_INTENTS

    def retrieve(self, query: str, intent: Dict[str, Any], top_k: int = 3) -> List[dict]:
        """Stub retriever — returns [] until a real vector store is
        connected. Empty-list return is safe; augment_memory_summary
        handles it gracefully."""
        return []

    def augment_memory_summary(
        self,
        base_summary: str,
        retrieved: List[dict],
        reranker: "MemoryReranker",
        query: str,
        intent: Dict[str, Any],
    ) -> str:
        if not retrieved:
            return base_summary
        reranked = reranker.rerank(retrieved, query, intent, top_k=3)
        rag_text = reranker.format_for_prompt(reranked, max_chars=300)
        if not rag_text:
            return base_summary
        return (base_summary + " [RAG] " + rag_text).strip()
