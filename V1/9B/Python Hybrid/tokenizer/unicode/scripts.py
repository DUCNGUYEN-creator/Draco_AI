# DracoAI V1 — tokenizer/unicode/scripts.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Unicode Script Detection
=============================================
Identifies writing scripts for multilingual pre-tokenisation routing.

Supported scripts (detection order):
  Emoji, CJK (Han), Hiragana, Katakana, Hangul, Arabic, Hebrew,
  Cyrillic, Devanagari, Thai, Lao, Khmer, Myanmar, Tibetan,
  Georgian, Armenian, Ethiopic, Vietnamese (Latin subtype), Latin,
  Digit, Whitespace, Punctuation, Other.
"""

import unicodedata
from typing import Dict, List, Tuple

from ..constants import (
    CJK_RANGES, EMOJI_RANGES,
    HIRAGANA_RANGE, KATAKANA_RANGE, HANGUL_RANGE, HANGUL_JAMO,
    ARABIC_RANGE, CYRILLIC_RANGE, DEVANAGARI_RANGE, HEBREW_RANGE,
    THAI_RANGE, LAO_RANGE, KHMER_RANGE, MYANMAR_RANGE, TIBETAN_RANGE,
    GEORGIAN_RANGE, ARMENIAN_RANGE, ETHIOPIC_RANGE,
    SCRIPTIO_CONTINUA_RANGES, VIET_CHARS,
)

# ── Script labels ─────────────────────────────────────────────────────
SCRIPT_EMOJI            = "Emoji"
SCRIPT_CJK              = "CJK"
SCRIPT_HIRAGANA         = "Hiragana"
SCRIPT_KATAKANA         = "Katakana"
SCRIPT_HANGUL           = "Hangul"
SCRIPT_ARABIC           = "Arabic"
SCRIPT_HEBREW           = "Hebrew"
SCRIPT_CYRILLIC         = "Cyrillic"
SCRIPT_DEVANAGARI       = "Devanagari"
SCRIPT_THAI             = "Thai"
SCRIPT_LAO              = "Lao"
SCRIPT_KHMER            = "Khmer"
SCRIPT_MYANMAR          = "Myanmar"
SCRIPT_TIBETAN          = "Tibetan"
SCRIPT_GEORGIAN         = "Georgian"
SCRIPT_ARMENIAN         = "Armenian"
SCRIPT_ETHIOPIC         = "Ethiopic"
SCRIPT_VIETNAMESE       = "Vietnamese"
SCRIPT_LATIN            = "Latin"
SCRIPT_DIGIT            = "Digit"
SCRIPT_WHITESPACE       = "Whitespace"
SCRIPT_PUNCTUATION      = "Punctuation"
SCRIPT_OTHER            = "Other"

_SCRIPTIO_CONTINUA_SCRIPTS = {
    SCRIPT_THAI, SCRIPT_LAO, SCRIPT_KHMER, SCRIPT_MYANMAR, SCRIPT_TIBETAN,
}


def _in_range(cp: int, ranges: list) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_of(c: str) -> str:
    """Return the script label for a single character *c*."""
    cp = ord(c)

    # Fast path: ASCII
    if cp < 128:
        if c.isdigit():   return SCRIPT_DIGIT
        if c.isalpha():   return SCRIPT_LATIN
        if c.isspace():   return SCRIPT_WHITESPACE
        return SCRIPT_PUNCTUATION

    if c.isspace():   return SCRIPT_WHITESPACE
    if c.isdigit():   return SCRIPT_DIGIT

    # Emoji (check before CJK — some ranges overlap)
    if _in_range(cp, EMOJI_RANGES):
        return SCRIPT_EMOJI

    # CJK Han
    if _in_range(cp, CJK_RANGES):
        return SCRIPT_CJK

    # Japanese kana
    if HIRAGANA_RANGE[0] <= cp <= HIRAGANA_RANGE[1]: return SCRIPT_HIRAGANA
    if KATAKANA_RANGE[0] <= cp <= KATAKANA_RANGE[1]: return SCRIPT_KATAKANA

    # Korean
    if HANGUL_RANGE[0] <= cp <= HANGUL_RANGE[1]: return SCRIPT_HANGUL
    if HANGUL_JAMO[0]  <= cp <= HANGUL_JAMO[1]:  return SCRIPT_HANGUL

    # Semitic
    if ARABIC_RANGE[0] <= cp <= ARABIC_RANGE[1]: return SCRIPT_ARABIC
    if HEBREW_RANGE[0] <= cp <= HEBREW_RANGE[1]: return SCRIPT_HEBREW

    # Cyrillic
    if CYRILLIC_RANGE[0] <= cp <= CYRILLIC_RANGE[1]: return SCRIPT_CYRILLIC

    # Indic
    if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]: return SCRIPT_DEVANAGARI

    # Scriptio continua (word-boundary-less scripts)
    if THAI_RANGE[0]    <= cp <= THAI_RANGE[1]:    return SCRIPT_THAI
    if LAO_RANGE[0]     <= cp <= LAO_RANGE[1]:     return SCRIPT_LAO
    if KHMER_RANGE[0]   <= cp <= KHMER_RANGE[1]:   return SCRIPT_KHMER
    if MYANMAR_RANGE[0] <= cp <= MYANMAR_RANGE[1]: return SCRIPT_MYANMAR
    if TIBETAN_RANGE[0] <= cp <= TIBETAN_RANGE[1]: return SCRIPT_TIBETAN

    # Other identifiable scripts
    if GEORGIAN_RANGE[0]  <= cp <= GEORGIAN_RANGE[1]:  return SCRIPT_GEORGIAN
    if ARMENIAN_RANGE[0]  <= cp <= ARMENIAN_RANGE[1]:  return SCRIPT_ARMENIAN
    if ETHIOPIC_RANGE[0]  <= cp <= ETHIOPIC_RANGE[1]:  return SCRIPT_ETHIOPIC

    # Vietnamese precomposed diacritics (Latin Extended Additional)
    if c in VIET_CHARS: return SCRIPT_VIETNAMESE

    if c.isalpha(): return SCRIPT_LATIN

    return SCRIPT_OTHER


def is_scriptio_continua(c: str) -> bool:
    """
    Return True if *c* belongs to a scriptio-continua script
    (Thai, Lao, Khmer, Myanmar, Tibetan).

    These scripts lack whitespace word boundaries and require
    character-level splitting before BPE to prevent over-tokenisation.
    """
    return script_of(c) in _SCRIPTIO_CONTINUA_SCRIPTS


def is_cjk(c: str) -> bool:
    return script_of(c) == SCRIPT_CJK


def is_japanese_kana(c: str) -> bool:
    return script_of(c) in (SCRIPT_HIRAGANA, SCRIPT_KATAKANA)


def is_hangul(c: str) -> bool:
    return script_of(c) == SCRIPT_HANGUL


def is_arabic(c: str) -> bool:
    return script_of(c) == SCRIPT_ARABIC


def is_thai(c: str) -> bool:
    return script_of(c) == SCRIPT_THAI


def detect_dominant_script(text: str) -> str:
    """Return the most frequent script in *text* (ignoring ws/punct/digits)."""
    skip  = {SCRIPT_WHITESPACE, SCRIPT_PUNCTUATION, SCRIPT_DIGIT}
    counts: Dict[str, int] = {}
    for c in text:
        s = script_of(c)
        if s not in skip:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else SCRIPT_LATIN


def script_segments(text: str) -> List[Tuple[str, str]]:
    """Split *text* into contiguous runs sharing the same script."""
    if not text:
        return []
    segs: List[Tuple[str, str]] = []
    cur_s = script_of(text[0])
    cur_c = text[0]
    for c in text[1:]:
        s = script_of(c)
        if s == cur_s:
            cur_c += c
        else:
            segs.append((cur_s, cur_c))
            cur_s, cur_c = s, c
    segs.append((cur_s, cur_c))
    return segs
