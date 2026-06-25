# DracoAI V1 — tokenizer/pretokenizer/languages/russian.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Russian pre-tokenizer (Cyrillic)."""
from typing import List
from ..families.slavic import split_slavic

def split_russian(text: str) -> List[str]:
    return split_slavic(text)
