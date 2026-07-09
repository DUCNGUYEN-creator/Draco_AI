# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolCallParser
================
Extracts <tool_call>...</tool_call> blocks from model output. Ported
1:1 from engine_v1.py's ``ToolCallingFramework.parse_tool_calls`` —
including the markdown-fence-stripping and trailing-comma-fixing
robustness fixes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


class ToolCallParser:
    def parse(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            raw = m.group(1).strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
            raw = re.sub(r",(\s*[}\]])", r"\1", raw)  # trailing-comma fix
            try:
                parsed = json.loads(raw)
                calls.append(parsed)
            except Exception:
                calls.append({"raw": raw, "parse_error": True})
        return calls
