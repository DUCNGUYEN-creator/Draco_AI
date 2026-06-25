# DracoAI V1 — tokenizer/pretokenizer/languages/korean.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Korean pre-tokenizer (Hangul syllables + Jamo)."""
from typing import List
from ..families.cjk_family import split_cjk

def split_korean(text: str) -> List[str]:
    return split_cjk(text)
