# DracoAI V1 — tokenizer/pretokenizer/families/cjk_family.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
CJK family pre-tokenizer (Chinese, Japanese, Korean).

Each CJK / Kana / Hangul character is its own segment.
Non-CJK runs (Latin, digits, spaces, punctuation) are segmented with
the SOTA word-piece pattern.  Fullwidth characters are normalised
to halfwidth before segmentation.
"""

import re
from typing import List

from ...unicode.width import fullwidth_to_halfwidth

# CJK character class regex (Python re is sufficient for range matching)
_CJK_RE = re.compile(
    r"[\u4E00-\u9FFF"        # CJK Unified Ideographs
    r"\u3400-\u4DBF"         # CJK Extension A
    r"\uF900-\uFAFF"         # CJK Compatibility Ideographs
    r"\u3040-\u309F"         # Hiragana
    r"\u30A0-\u30FF"         # Katakana
    r"\uAC00-\uD7AF"         # Hangul syllables
    r"\u1100-\u11FF]",       # Hangul Jamo
    re.UNICODE,
)

try:
    import regex as _re
    _NON_CJK_PATTERN = _re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*)+"
        r"|\p{N}{1,3}"
        r"|[^\s\p{L}\p{N}]++"
        r"|\s+",
    )
except ImportError:
    _NON_CJK_PATTERN = re.compile(
        r"(?:[^\W\d_][\u0300-\u036F]*)+"
        r"|\d{1,3}"
        r"|\s+"
        r"|[^\w\s]",
        re.UNICODE,
    )


def is_cjk_char(c: str) -> bool:
    return bool(_CJK_RE.match(c))


def split_cjk(text: str) -> List[str]:
    """
    Segment CJK/Kana/Hangul text into pre-tokenizer units.

    CJK characters → individual segments.
    Non-CJK runs → SOTA word-piece segmentation.
    """
    text = fullwidth_to_halfwidth(text)
    segments: List[str] = []
    i, n = 0, len(text)

    while i < n:
        c = text[i]
        if is_cjk_char(c):
            segments.append(c)
            i += 1
        else:
            j = i + 1
            while j < n and not is_cjk_char(text[j]):
                j += 1
            run = text[i:j]
            for m in _NON_CJK_PATTERN.finditer(run):
                w = m.group()
                if w:
                    segments.append(w)
            i = j

    return [s for s in segments if s]
