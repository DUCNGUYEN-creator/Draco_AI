# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MemoryReranker
================
Re-scores raw memory-search candidates by blending semantic similarity,
intent-keyword match, and recency. Ported 1:1 from engine_v1.py's
``MemoryReranker`` — including the RERANK-FIX fallback when the
threshold filters every candidate out, and the c.copy() fix that
prevents in-place mutation of caller-owned candidate dicts.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List

from ..constants import INTENT_CHAT, INTENT_CODE, INTENT_FACTUAL, INTENT_LOGIC, INTENT_MATH

_INTENT_KW: Dict[str, List[str]] = {
    INTENT_CODE: ["code", "def ", "class ", "import", "function", "python", "error", "bug", "return", "```"],
    INTENT_MATH: ["=", "tính", "số", "kết quả", "giải", "phương trình", "math", "calculate"],
    INTENT_LOGIC: ["vì", "nếu", "thì", "suy ra", "kết luận", "logic"],
    INTENT_FACTUAL: ["là", "có", "tại", "khi", "where", "when", "what"],
}


class MemoryReranker:
    INTENT_KW = _INTENT_KW

    def rerank(
        self,
        candidates: List[dict],
        query: str,
        intent: dict,
        top_k: int = 3,
        threshold: float = 0.1,
    ) -> List[dict]:
        itype = intent.get("intent", INTENT_CHAT)
        kws = self.INTENT_KW.get(itype, [])
        now = time.time()
        scored: List[dict] = []
        for c in candidates:
            nc = c.copy()  # never mutate caller-owned dicts
            text = nc.get("text", "").lower()
            semantic = nc.get("score", 0.0)
            intent_m = sum(1 for k in kws if k in text) / max(len(kws), 1)
            if intent_m > 0:
                intent_m = min(intent_m * 1.5, 1.0)
            age = (now - nc.get("ts", now)) / 86400.0
            recency = math.exp(-age / 7.0)
            final = semantic * 0.4 + intent_m * 0.4 + recency * 0.2
            if final >= threshold:
                nc["rerank_score"] = final
                scored.append(nc)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = scored[:top_k]

        # Fallback: if threshold filtered everything, return top_k by raw score
        if not result and candidates:
            fallback = [c.copy() for c in candidates]
            fallback.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            for c in fallback[:top_k]:
                c.setdefault("rerank_score", c.get("score", 0.0))
            result = fallback[:top_k]

        return result

    def format_for_prompt(self, memories: List[dict], max_chars: int = 500) -> str:
        parts: List[str] = []
        total = 0
        for m in memories:
            t = m.get("text", "")
            if not t:
                continue
            if len(t) > 150:
                t = t[:147] + "..."
            parts.append(t)
            total += len(t)
            if total > max_chars:
                break
        return " | ".join(parts)
