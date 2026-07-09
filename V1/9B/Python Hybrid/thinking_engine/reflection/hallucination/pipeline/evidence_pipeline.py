# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EvidencePipeline — Stage 1: "Evidence"
=========================================
Gathers an EvidenceBundle for a claim from whatever sources the
caller's context provides (KG path, RAG docs, memory snippets, tool
results) and checks/populates EvidenceCache. This is the ONLY stage
that touches raw Infrastructure-layer objects (KnowledgeGraph,
RAGPipeline output, etc.) — every later stage operates purely on the
already-assembled EvidenceBundle / VerificationResult / FusionResult
dataclasses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..cache.evidence_cache import EvidenceCache
from ..models.evidence import Evidence, EvidenceBundle
from ..models.enums import EvidenceType


class EvidencePipeline:
    def __init__(self, cache: Optional[EvidenceCache] = None) -> None:
        self.cache = cache or EvidenceCache()

    def gather(self, claim: str, context: Dict[str, Any]) -> EvidenceBundle:
        cached = self.cache.get(claim)
        if cached is not None:
            return cached

        bundle = EvidenceBundle(claim=claim)

        # KG reasoning_path entries — high-trust, structured evidence.
        for node in context.get("reasoning_path", []) or []:
            bundle.add(Evidence(text=str(node), source_type=EvidenceType.KNOWLEDGE_GRAPH, trust_score=0.85))

        # RAG-retrieved documents.
        for doc in context.get("rag_docs", []) or []:
            bundle.add(
                Evidence(
                    text=doc.get("text", ""),
                    source_type=EvidenceType.RAG_DOCUMENT,
                    trust_score=doc.get("rerank_score", 0.6),
                    metadata={"source_id": doc.get("source_id")},
                )
            )

        # Memory snippets.
        memory_text = context.get("memory_summary", "")
        if memory_text:
            bundle.add(Evidence(text=memory_text, source_type=EvidenceType.MEMORY, trust_score=0.55))

        # Tool results — empirically derived, highest default trust.
        for r in context.get("tool_results", []) or []:
            if r.get("ok"):
                bundle.add(
                    Evidence(
                        text=str(r.get("output", "")),
                        source_type=EvidenceType.TOOL_RESULT,
                        trust_score=0.95,
                        metadata={"tool": r.get("tool")},
                    )
                )

        self.cache.set(claim, bundle)
        return bundle
