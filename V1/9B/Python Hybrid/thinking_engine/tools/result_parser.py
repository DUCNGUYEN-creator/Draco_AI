# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolResultParser
===================
Post-processes raw tool-execution results — e.g. attempts to coerce a
calculator's string output back to a Python number for downstream
numeric verifiers (reflection/hallucination/verifiers/numerical.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ToolResultParser:
    def as_number(self, result: Dict[str, Any]) -> Optional[float]:
        if not result.get("ok"):
            return None
        try:
            return float(result.get("output", ""))
        except (TypeError, ValueError):
            return None

    def summarize(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        parts = []
        for r in results:
            status = "✓" if r.get("ok") else "✗"
            parts.append(f"{status} {r.get('tool')}: {r.get('output')}")
        return " | ".join(parts)
