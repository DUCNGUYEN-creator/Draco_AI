# DracoAI V1 — thinking_engine/tools/tool_crafter.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolCrafter
=============
Zero-shot "tool" synthesis hook — when the engine detects INTENT_CODE
(or any tool-triggering intent) but NO registered tool matches the query,
ToolCrafter generates a best-effort code-stub "tool" that can be surfaced
to the caller or executed in a sandbox.

Ported from engine_v1.py's ``ToolCrafter`` class:

    Original API (engine_v1.py):
        is_applicable(intent, query, available_tools) -> bool
        craft(query) -> dict  # returns {"name": ..., "code": ..., "args": ...}

    BUG #3 — Previous ported version:
        craft_stub(name, args) -> dict  # completely wrong API
        - Assumed the caller already KNEW the tool name and args
        - Was the OPPOSITE of zero-shot (required fully specified input)
        - Did not check intent or available tools at all

    Fixed version (this file):
        is_applicable(intent, query, available_tools) -> bool
        craft(query, intent) -> dict

Architecture
------------
The crafter does NOT actually generate real executable code (that would
need a code-generation model). Instead, it constructs a structured
descriptor that:
1. Identifies what kind of tool operation the user likely wants
2. Generates a plausible function signature / parameter schema
3. Marks the descriptor as ``status: "zero_shot"`` so the executor
   and UI know this is an unverified auto-generated tool proposal

The ``state.zero_shot_tool`` field receives the output, and the
ToolExecutor can optionally attempt to execute simple cases.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..constants import INTENT_CODE, INTENT_MATH

# ── Zero-shot tool templates keyed by detected operation type ──────────────
_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "compute": {
        "name": "zero_shot_calculator",
        "description": "Auto-generated: evaluate a mathematical/numerical expression",
        "args_schema": {"expr": "str"},
    },
    "transform": {
        "name": "zero_shot_transformer",
        "description": "Auto-generated: transform / convert data",
        "args_schema": {"input": "str", "format": "str"},
    },
    "search": {
        "name": "zero_shot_search",
        "description": "Auto-generated: search for information",
        "args_schema": {"query": "str"},
    },
    "generate": {
        "name": "zero_shot_generator",
        "description": "Auto-generated: generate code / content",
        "args_schema": {"prompt": "str", "language": "str"},
    },
}

# ── Operation-detection keywords ──────────────────────────────────────────
_OP_KEYWORDS: Dict[str, List[str]] = {
    "compute": [
        "tính", "tính toán", "calculate", "compute", "evaluate",
        "bao nhiêu", "sum", "total", "average", "mean",
    ],
    "transform": [
        "chuyển đổi", "convert", "transform", "format", "parse",
        "encode", "decode", "serialize", "mã hóa", "giải mã",
    ],
    "search": [
        "tìm", "tìm kiếm", "search", "lookup", "find", "query",
        "tra cứu", "tra", "fetch", "get",
    ],
    "generate": [
        "viết", "tạo", "generate", "create", "write", "build",
        "implement", "code", "lập trình", "sinh mã",
    ],
}


class ToolCrafter:
    """Zero-shot tool synthesis for unmatched intent-query pairs."""

    def is_applicable(
        self,
        intent: Dict[str, Any],
        query: str,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Returns True when the intent suggests tool-use but no registered
        tool matches the query — exactly the condition under which
        engine_v1.py's original ``ToolCrafter.is_applicable()`` returned True.

        Parameters
        ----------
        intent          : the intent dict from IntentDetector.detect()
        query           : the user's (rewritten) query
        available_tools : list of tool dicts from ToolRegistry.relevant_for()
                          — if empty/None and intent is tool-triggering,
                          this method returns True
        """
        itype = intent.get("intent", "")
        # Only trigger for tool-heavy intents
        if itype not in (INTENT_CODE, INTENT_MATH):
            return False
        # If there are relevant registered tools, no need for zero-shot
        if available_tools:
            return False
        return True

    def _detect_operation(self, query: str) -> str:
        """Returns the best-matching operation type from _OP_KEYWORDS."""
        ql = query.lower()
        best_op = "generate"  # default for code intent
        best_score = 0
        for op, keywords in _OP_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in ql)
            if score > best_score:
                best_score = score
                best_op = op
        return best_op

    def _extract_args(self, query: str) -> Dict[str, str]:
        """Extracts plausible argument values from the query using simple
        heuristics (quoted strings, code blocks, numeric expressions)."""
        args: Dict[str, str] = {}

        # Extract quoted strings as potential arguments
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted:
            args["input"] = quoted[0]

        # Extract code blocks
        code_blocks = re.findall(r'```\w*\n?(.*?)```', query, re.DOTALL)
        if code_blocks:
            args["code"] = code_blocks[0].strip()

        # Extract mathematical expressions
        math_exprs = re.findall(r'(\d+[\s]*[+\-*/^%][\s]*\d[\d\s+\-*/^%]*)', query)
        if math_exprs:
            args["expr"] = math_exprs[0].strip()

        # Fallback: use the full query as the prompt
        if not args:
            args["prompt"] = query

        return args

    def craft(self, query: str, intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates a zero-shot tool descriptor for an unmatched query.

        Returns a dict with:
            name     : str — the generated tool name
            args     : dict — extracted / inferred arguments
            code     : str — a placeholder code stub
            status   : "zero_shot" — always this value
            note     : str — human-readable explanation
            template : str — which template was used

        This is stored in ``ThinkingState.zero_shot_tool`` and surfaced
        through the engine output so the caller/UI can decide whether to
        actually execute it.
        """
        op = self._detect_operation(query)
        template = _TEMPLATES.get(op, _TEMPLATES["generate"])
        args = self._extract_args(query)

        # Generate a simple code stub
        func_name = f"zero_shot_{op}"
        param_list = ", ".join(args.keys())
        code_stub = (
            f"def {func_name}({param_list}):\n"
            f"    \"\"\"Auto-generated zero-shot tool for: {query[:80]}...\"\"\"\n"
            f"    # PRODUCTION HOOK: implement actual logic\n"
            f"    return {{'status': 'stub', 'input': locals()}}\n"
        )

        return {
            "name": template["name"],
            "args": args,
            "code": code_stub,
            "status": "zero_shot",
            "template": op,
            "note": (
                f"No registered tool matched this {intent.get('intent', 'unknown') if intent else 'unknown'}-intent "
                f"query. Auto-generated a '{op}' tool stub. "
                f"This is a zero-shot proposal — no code was executed."
            ),
        }
