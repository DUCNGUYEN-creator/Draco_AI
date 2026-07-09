# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolExecutor
==============
Executes parsed/validated tool calls. Ported 1:1 from engine_v1.py's
``ToolCallingFramework._call_tool`` / ``execute_tool_calls`` —
calculator uses SafeASTEvaluator (no eval() risk); code_runner and
web_search remain documented stubs (PRODUCTION HOOK markers preserved)
until a real sandbox / search backend is wired up.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .registry import ToolRegistry
from .sandbox import SafeASTEvaluator
from .validator import ToolCallValidator


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        validator: ToolCallValidator | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.validator = validator or ToolCallValidator(self.registry)
        self._ast_eval = SafeASTEvaluator()

    def _call_tool(self, name: str, args: dict) -> Dict[str, Any]:
        custom_handler = self.registry.get_handler(name)
        if custom_handler is not None:
            return custom_handler(args)

        if name == "calculator":
            expr = str(args.get("expr", "0"))
            result = self._ast_eval.evaluate(expr)
            return {"tool": name, "input": expr, "output": result, "ok": not result.startswith("Error")}
        if name == "code_runner":
            # PRODUCTION HOOK: run in sandbox (e.g. restrictedpython, subprocess jail)
            code = args.get("code", "")
            return {"tool": name, "input": code[:200], "output": "(stub — sandbox not connected)", "ok": False}
        if name == "web_search":
            # PRODUCTION HOOK: connect to a real search backend
            query = args.get("query", "")
            return {"tool": name, "input": query, "output": "(stub — search not connected)", "ok": False}
        return {"tool": name, "input": str(args), "output": "(unknown tool)", "ok": False}

    def execute(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute parsed tool calls, validating each first. Returns a
        structured list of results: {tool, input, output, ok}."""
        if not calls:
            return []
        results: List[Dict[str, Any]] = []
        for call in calls:
            ok, reason = self.validator.validate(call)
            if not ok:
                results.append(
                    {
                        "tool": call.get("name", "parse_error"),
                        "input": call.get("raw", str(call.get("args", "")))[:80],
                        "output": reason,
                        "ok": False,
                    }
                )
                continue
            name = call.get("name", "unknown")
            args = call.get("args", {})
            results.append(self._call_tool(name, args))
        return results
