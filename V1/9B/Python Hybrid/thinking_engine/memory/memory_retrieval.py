# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MemoryRetrieval
==================
Stage-2 orchestrator of the pipeline ("Memory Retrieval"). Combines
WorkingMemory, EpisodicMemory, SemanticMemory and a caller-supplied list
of raw memory_candidates, then hands everything to MemoryReranker +
MemoryCompressor to produce the single ``full_memory`` string that gets
spliced into the prompt. This is the formalized version of the ad-hoc
rerank/format logic that lived inline inside engine_v1.py's
``ThinkingEngineV1.process()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..perception.prompt.sanitizer import PromptSanitizer
from .compression import MemoryCompressor
from .episodic_memory import EpisodicMemory
from .memory_reranker import MemoryReranker
from .semantic_memory import SemanticMemory
from .working_memory import WorkingMemory


class MemoryRetrieval:
    def __init__(
        self,
        reranker: Optional[MemoryReranker] = None,
        compressor: Optional[MemoryCompressor] = None,
        sanitizer: Optional[PromptSanitizer] = None,
        working_memory: Optional[WorkingMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
    ) -> None:
        self.reranker = reranker or MemoryReranker()
        self.compressor = compressor or MemoryCompressor()
        self.sanitizer = sanitizer or PromptSanitizer()
        self.working_memory = working_memory or WorkingMemory()
        self.episodic_memory = episodic_memory or EpisodicMemory()
        self.semantic_memory = semantic_memory or SemanticMemory()

    def retrieve(
        self,
        query: str,
        intent: Dict[str, Any],
        memory_candidates: Optional[List[dict]] = None,
        memory_summary: str = "",
        top_k: int = 3,
    ) -> str:
        """Returns the combined, sanitized memory_summary string ready to
        splice into the prompt — mirrors the inline logic previously in
        ThinkingEngineV1.process()'s memory-rerank block."""
        memory_summary = self.sanitizer.sanitize(memory_summary)

        reranked_memory = ""
        if memory_candidates and isinstance(memory_candidates, list):
            compacted = self.compressor.compress(memory_candidates)
            safe_candidates = []
            for c in compacted:
                sc = dict(c)
                sc["text"] = self.sanitizer.sanitize(sc.get("text", ""))
                safe_candidates.append(sc)
            reranked = self.reranker.rerank(safe_candidates, query, intent, top_k=top_k)
            reranked_memory = self.reranker.format_for_prompt(reranked)

        # Episodic recall — surfaces a relevant past Q/A pair if highly overlapping
        episodes = self.episodic_memory.search(query, top_k=1)
        episodic_note = ""
        if episodes:
            ep = episodes[0]
            episodic_note = f"[EPISODE] Q: {ep.question[:80]} A: {ep.answer[:120]}"

        parts = [p for p in [memory_summary, reranked_memory, episodic_note] if p]
        return " | ".join(parts).strip(" |")
