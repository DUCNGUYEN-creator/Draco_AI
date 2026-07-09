# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""EntityExtractor — capitalized-token heuristic NER, ported from the
entity-extraction regex embedded in engine_v1.py's IntentDetector.detect()."""

from __future__ import annotations

from typing import List

from ...utils.text import extract_capitalized_entities


class EntityExtractor:
    def extract(self, text: str, limit: int = 5) -> List[str]:
        return extract_capitalized_entities(text, limit=limit)
