# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ResponseBuilder
=================
Converts token IDs produced by the LLM bridge back into a structured
response dict, stripping ChatML scaffolding and extracting any
<tool_call> blocks for ToolCallParser.

This is the "Output" stage of the pipeline, running AFTER Reflection
has potentially modified the messages with [REFINED] or rewrite
instructions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class ResponseBuilder:
    # ChatML control tags to strip from generated text
    _STOP_RE = re.compile(r"<\|im_end\|>.*", re.DOTALL)
    _IM_START_RE = re.compile(r"<\|im_start\|>[a-z]+\n?")
    _FINAL_ANSWER_RE = re.compile(r"\[FINAL ANSWER\]\s*", re.IGNORECASE)

    def build(
        self,
        token_ids: List[int],
        tokenizer: Any,
        generation_request: Any = None,
    ) -> Dict[str, Any]:
        if not token_ids or tokenizer is None:
            return {"text": "", "tool_calls": [], "token_count": 0}

        raw = tokenizer.decode(token_ids)
        text = self._STOP_RE.sub("", raw)
        text = self._IM_START_RE.sub("", text)
        text = self._FINAL_ANSWER_RE.sub("", text)
        text = text.strip()

        # Extract tool calls before returning so the caller can route
        # them to ToolExecutor without re-running the decoder
        tool_call_texts = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)

        return {
            "text": text,
            "tool_call_texts": tool_call_texts,
            "token_count": len(token_ids),
        }
