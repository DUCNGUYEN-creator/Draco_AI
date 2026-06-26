# DracoAI V1 — tokenizer/pretokenizer/languages/mongolian.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Mongolian pre-tokenizer (Cyrillic script, agglutinative morphology)."""
from typing import List
from ..families.agglutinative import split_agglutinative

def split_mongolian(text: str) -> List[str]:
    return split_agglutinative(text)
