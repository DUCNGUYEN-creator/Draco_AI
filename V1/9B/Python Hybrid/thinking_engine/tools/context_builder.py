# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolContextBuilder
=====================
Builds the "[TOOLS AVAILABLE]" system-prompt injection AND the
post-execution tool-result re-injection string. Split from
engine_v1.py's ``ToolCallingFramework.build_tool_injection`` /
``build_tool_context``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .registry import ToolRegistry


class ToolContextBuilder:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def build_tool_injection(self, intent: Dict[str, Any], query: str) -> str:
        """Returns a string to inject into the system prompt describing
        available tools. The model is expected to emit
        <tool_call>{"name": ..., "args": ...}</tool_call>."""
        relevant = self.registry.relevant_for(intent)
        if not relevant:
            return ""
        lines = ["[TOOLS AVAILABLE — use <tool_call>{...}</tool_call> syntax]"]
        for t in relevant:
            lines.append(f"  • {t['name']}: {t['description']}")
        return "\n".join(lines)

    @staticmethod
    def build_tool_context(results: List[Dict[str, Any]]) -> str:
        """Format tool results as a string to inject back into the next
        LLM message (as a user-turn tool_result block)."""
        if not results:
            return ""
        lines = ["[TOOL RESULTS]"]
        for r in results:
            status = "OK" if r.get("ok") else "FAILED"
            lines.append(f"  • {r.get('tool', '?')} [{status}]: {r.get('output', '')}")
        return "\n".join(lines)
