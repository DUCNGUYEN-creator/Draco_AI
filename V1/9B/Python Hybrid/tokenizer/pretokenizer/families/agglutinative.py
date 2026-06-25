# DracoAI V1 — tokenizer/pretokenizer/families/agglutinative.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Agglutinative-language pre-tokenizer (Turkish, Finnish, Hungarian,
Mongolian, Kazakh, Uzbek, Estonian, etc.).

Agglutinative languages concatenate many morphological suffixes onto
root words, producing very long tokens that BPE handles poorly if the
word is kept intact.  This splitter:
  1. Splits on whitespace / punctuation boundaries (standard).
  2. Applies a soft maximum segment length to prevent single tokens
     from being excessively long before BPE (reduces merge table bloat).
"""

import re
from typing import List

_MAX_WORD_LEN = 30   # soft limit; longer segments are further sub-split

try:
    import regex as _re
    _AGG_PATTERN = _re.compile(
        r"[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*)+"
        r"|\p{N}{1,3}"
        r"|[^\s\p{L}\p{N}]++"
        r"|\s+",
    )
except ImportError:
    _AGG_PATTERN = re.compile(
        r"(?:[^\W\d_][\u0300-\u036F]*)+"
        r"|\d{1,3}"
        r"|\s+"
        r"|[^\w\s]",
        re.UNICODE,
    )


def split_agglutinative(text: str) -> List[str]:
    """
    Segment agglutinative-language text.

    Words longer than *_MAX_WORD_LEN* are split at vowel boundaries
    to approximate morpheme boundaries (rough heuristic; BPE will
    learn the correct boundaries from training data).
    """
    results: List[str] = []
    for m in _AGG_PATTERN.finditer(text):
        word = m.group()
        if not word:
            continue
        if len(word) <= _MAX_WORD_LEN or not word.isalpha():
            results.append(word)
        else:
            # Rough sub-split at character-group boundaries
            chunk_size = _MAX_WORD_LEN // 2
            for i in range(0, len(word), chunk_size):
                results.append(word[i:i + chunk_size])
    return results
