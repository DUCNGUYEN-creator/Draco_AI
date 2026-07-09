# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
HistorySummarizer
====================
Production hook for collapsing old conversation turns into a compact
summary. Ported from the ``_summarize_history`` stub embedded inside
engine_v1.py's ``ContextWindowManager`` — split out so a real LLM-backed
summarizer can be swapped in without touching ContextWindowManager.
"""

from __future__ import annotations

from typing import Any, List, Optional


class HistorySummarizer:
    def __init__(self, bridge: Any = None, tokenizer: Any = None) -> None:
        self.bridge = bridge
        self.tokenizer = tokenizer

    def summarize(self, old_msgs: List[dict], max_chars: int = 600) -> str:
        """Stub: concatenate first 120 chars of each message's content.
        PRODUCTION HOOK: when bridge+tokenizer are connected, replace this
        with a real LLM summarization call."""
        if self.bridge is not None and self.tokenizer is not None:
            llm_summary = self._summarize_with_llm(old_msgs, max_chars)
            if llm_summary:
                return llm_summary
        parts: List[str] = []
        for m in old_msgs:
            role = m.get("role", "?")
            content = m.get("content", "")[:120]
            parts.append(f"[{role}]: {content}")
        return " | ".join(parts)[:max_chars]

    def _summarize_with_llm(self, old_msgs: List[dict], max_chars: int) -> Optional[str]:
        if not hasattr(self.bridge, "is_connected") or not self.bridge.is_connected():
            return None
        try:
            text = "\n".join(f"{m.get('role','?')}: {m.get('content','')}" for m in old_msgs)
            prompt = f"Summarize this conversation concisely:\n{text}\n\nSummary:"
            ids = self.tokenizer.encode(prompt, add_bos=True)
            out_ids = self.bridge.generate(ids, max_new_tokens=120)
            if not out_ids:
                return None
            return self.tokenizer.decode(out_ids)[:max_chars]
        except Exception:
            return None
