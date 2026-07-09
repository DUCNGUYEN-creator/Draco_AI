# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolCallValidator
====================
New addition. Validates a parsed tool-call dict against the registry
BEFORE execution — rejects unknown tool names and missing required
args early, instead of letting ToolExecutor's per-tool branch silently
fall through to the "(unknown tool)" stub. Helps the Hallucination
verifiers/tool.py distinguish "tool call was well-formed but tool
itself failed" from "tool call was malformed".
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .registry import ToolRegistry

_REQUIRED_ARGS = {
    "calculator": ["expr"],
    "code_runner": ["code"],
    "web_search": ["query"],
}


class ToolCallValidator:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def validate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        if call.get("parse_error"):
            return False, "Tool call JSON could not be parsed"
        name = call.get("name")
        known_names = {t["name"] for t in self.registry.all_tools()}
        if name not in known_names:
            return False, f"Unknown tool: {name!r}"
        args = call.get("args", {})
        if not isinstance(args, dict):
            return False, "args must be a JSON object"
        for required in _REQUIRED_ARGS.get(name, []):
            if required not in args:
                return False, f"Missing required arg '{required}' for tool '{name}'"
        return True, ""

    def validate_all(self, calls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, str]]:
        return [(c, *self.validate(c)) for c in calls]
