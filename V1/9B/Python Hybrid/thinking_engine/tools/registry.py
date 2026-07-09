# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolRegistry
==============
Holds the DEFAULT_TOOLS table (name/description/triggers/keywords) and
decides when a tool should be injected — split out from
engine_v1.py's ``ToolCallingFramework`` so new tools can be registered
without touching the parser/executor.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..constants import INTENT_CHAT, INTENT_CODE, INTENT_FACTUAL, INTENT_HOW_TO, INTENT_MATH

DEFAULT_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Input: {'expr': '2+2*3'}",
        "triggers": [INTENT_MATH],
        "keywords": ["tính", "calculate", "compute", "=", "solve"],
    },
    {
        "name": "code_runner",
        "description": "Run a Python code snippet safely. Input: {'code': '...'}",
        "triggers": [INTENT_CODE],
        "keywords": ["chạy", "run", "execute", "test", "demo"],
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Input: {'query': '...'}",
        "triggers": [INTENT_FACTUAL, INTENT_HOW_TO],
        "keywords": ["tìm kiếm", "search", "latest", "mới nhất", "tra cứu"],
    },
]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: List[Dict[str, Any]] = [dict(t) for t in DEFAULT_TOOLS]
        self._handlers: Dict[str, Any] = {}

    def register(self, tool_def: Dict[str, Any], handler: Any = None) -> None:
        self._tools.append(tool_def)
        if handler is not None:
            self._handlers[tool_def["name"]] = handler

    def get_handler(self, name: str) -> Any:
        return self._handlers.get(name)

    def all_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools)

    def relevant_for(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        itype = intent.get("intent", INTENT_CHAT)
        return [t for t in self._tools if itype in t.get("triggers", [])]

    def should_use_tools(self, intent: Dict[str, Any], query: str) -> bool:
        ql = query.lower()
        for tool in self.relevant_for(intent):
            if any(kw in ql for kw in tool.get("keywords", [])):
                return True
        return False
