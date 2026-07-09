# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PromptParser
=============
Small companion to PromptCompiler: parses compiled message blocks back
into structured sections (e.g. to inspect what a [PLAN] block contained
during debugging/tests). Not present as a standalone class in the
original engine_v1.py but split out here to keep compiler.py focused
purely on *building* prompts, not *reading* them back.
"""

from __future__ import annotations

import re
from typing import Dict, List


class PromptParser:
    _SECTION_RE = re.compile(r"\[([A-Z][A-Z _-]*)\]")

    def split_sections(self, text: str) -> Dict[str, str]:
        """Split a compiled [PLAN]...[FINAL ANSWER] block into a dict of
        {section_name: content}. Order of appearance is preserved via
        Python 3.7+ dict ordering."""
        matches = list(self._SECTION_RE.finditer(text))
        sections: Dict[str, str] = {}
        for i, m in enumerate(matches):
            name = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[name] = text[start:end].strip()
        return sections

    def extract_messages_by_role(self, messages: List[dict], role: str) -> List[str]:
        return [m.get("content", "") for m in messages if m.get("role") == role]
