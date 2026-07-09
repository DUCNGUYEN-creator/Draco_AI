# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PromptSanitizer
=================
Anti-injection guard for ANY external content (RAG passages, tool
output, retrieved memory) before it is spliced into a prompt. Ported
1:1 from engine_v1.py's ``PromptSanitizer`` — same patterns, same
HTML-entity-unescape-before-regex fix (SANITIZER-V2).
"""

from __future__ import annotations

import html
import re


class PromptSanitizer:
    """Sanitize external content before injecting into the prompt.
    Blocks control-token injection attempts such as <|im_start|>,
    [SYSTEM], <<SYS>>...<</SYS>>, [/INST], and HTML-entity-encoded
    bypass variants (e.g. &lt;|im_start|&gt;).
    """

    _DANGER_PATTERNS = [
        (r"<\|.*?\|>", "[BLOCKED]"),
        (r"\[SYSTEM\]", "[BLOCKED]"),
        (r"\[INST\]", "[BLOCKED]"),
        (r"<<SYS>>.*?<</SYS>>", "[BLOCKED]"),
        (r"<\|im_start\|>", "[BLOCKED]"),
        (r"<\|im_end\|>", "[BLOCKED]"),
        (r"<\|system\|>", "[BLOCKED]"),
        (r"\[/INST\]", "[BLOCKED]"),
    ]

    def sanitize(self, text: str) -> str:
        if not text:
            return text
        # Decode HTML entities first so encoded bypass variants
        # (&lt;|im_start|&gt;) are caught by the regexes below.
        text = html.unescape(text)
        for pattern, replacement in self._DANGER_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)
        return text
