# DracoAI V1 — tokenizer/pretokenizer/languages/spanish.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Spanish pre-tokenizer (Latin-script)."""
from typing import List
from ..families.latin import split_latin

def split_spanish(text: str) -> List[str]:
    return split_latin(text)
