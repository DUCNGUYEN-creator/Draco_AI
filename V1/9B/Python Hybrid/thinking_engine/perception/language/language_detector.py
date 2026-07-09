# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""LanguageDetector — thin wrapper around utils.text.detect_lang, kept as
its own class so perception/__init__.py exposes a stable, swappable
component (e.g. to later plug in a real fastText/langid model)."""

from __future__ import annotations

from ...utils.text import detect_lang


class LanguageDetector:
    def detect(self, text: str) -> str:
        return detect_lang(text)
