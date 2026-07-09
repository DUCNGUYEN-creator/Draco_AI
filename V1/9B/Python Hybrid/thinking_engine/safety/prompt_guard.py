# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PromptGuard
=============
Thin safety-layer wrapper around perception.prompt.sanitizer.PromptSanitizer
— re-exposed here so safety/ is the single place callers look for "is
this input/external content safe to use" without reaching into
perception internals (which is conceptually about *understanding*
text, not *guarding against* malicious text).
"""

from __future__ import annotations

from ..perception.prompt.sanitizer import PromptSanitizer


class PromptGuard:
    def __init__(self) -> None:
        self._sanitizer = PromptSanitizer()

    def guard(self, text: str) -> str:
        return self._sanitizer.sanitize(text)
