# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
RAGPipeline
=============
Orchestrates retrieve -> rerank -> cite for a single query, wrapping
RetrievalAugmenter + KnowledgeReranker + CitationTracker into one call.
New addition (the original engine_v1.py inlined this orchestration
directly in ``ThinkingEngineV1.process()``); formalized here so Planning
and Reasoning stages can invoke RAG without reaching into the engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .citation import CitationTracker
from .reranker import KnowledgeReranker
from .retrieval import RetrievalAugmenter


class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[RetrievalAugmenter] = None,
        reranker: Optional[KnowledgeReranker] = None,
        citations: Optional[CitationTracker] = None,
    ) -> None:
        self.retriever = retriever or RetrievalAugmenter()
        self.reranker = reranker or KnowledgeReranker()
        self.citations = citations or CitationTracker()

    def run(self, query: str, intent: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        if not self.retriever.is_applicable(intent):
            return {"applicable": False, "docs": [], "context_text": "", "citations": []}
        docs = self.retriever.retrieve(query, intent, top_k=top_k)
        reranked = self.reranker.rerank(docs, query, top_k=top_k)
        for d in reranked:
            self.citations.register(d)
        context_text = self.reranker.format_context(reranked)
        return {
            "applicable": True,
            "docs": reranked,
            "context_text": context_text,
            "citations": self.citations.format_all(reranked),
        }
