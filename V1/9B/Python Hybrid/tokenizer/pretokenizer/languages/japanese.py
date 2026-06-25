# DracoAI V1 — tokenizer/pretokenizer/languages/japanese.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Japanese pre-tokenizer (Kanji + Kana + Latin mixed)."""
from typing import List
from ..families.cjk_family import split_cjk

def split_japanese(text: str) -> List[str]:
    """Segment Japanese text (delegates to CJK family splitter)."""
    return split_cjk(text)
