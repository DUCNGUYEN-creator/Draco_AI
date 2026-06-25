# DracoAI V1 — tokenizer/unicode/bidi.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Unicode Bidirectional (BiDi) Utilities
===========================================================
Direction detection for RTL scripts (Arabic, Hebrew, etc.) and
mixed-direction text handling.

For actual BiDi visual reordering, optionally installs python-bidi:
    pip install python-bidi
"""

import unicodedata
from typing import List, Tuple

_RTL_BIDI_CLASSES = frozenset({"R", "AL", "RLE", "RLO", "RLI"})

_ARABIC_PRES_A = (0xFB50, 0xFDFF)
_ARABIC_PRES_B = (0xFE70, 0xFEFF)


def bidi_class(c: str) -> str:
    """Return the Unicode Bidi_Class of *c*."""
    try:
        return unicodedata.bidirectional(c)
    except Exception:
        return "L"


def is_rtl_char(c: str) -> bool:
    """Return True if *c* has a right-to-left bidi class."""
    return bidi_class(c) in _RTL_BIDI_CLASSES


def is_arabic_presentation_form(c: str) -> bool:
    """Return True if *c* is an Arabic presentation-form glyph."""
    cp = ord(c)
    return ((_ARABIC_PRES_A[0] <= cp <= _ARABIC_PRES_A[1]) or
            (_ARABIC_PRES_B[0] <= cp <= _ARABIC_PRES_B[1]))


def contains_rtl(text: str) -> bool:
    """Return True if *text* contains at least one RTL character."""
    return any(is_rtl_char(c) for c in text)


def has_mixed_direction(text: str) -> bool:
    """Return True if *text* has both LTR and RTL alphabetic characters."""
    has_ltr = any(not is_rtl_char(c) and c.isalpha() for c in text)
    has_rtl = any(is_rtl_char(c) for c in text)
    return has_ltr and has_rtl


def base_direction(text: str) -> str:
    """Return "rtl" if the dominant direction is RTL, else "ltr"."""
    rtl_count = sum(1 for c in text if is_rtl_char(c))
    ltr_count = sum(1 for c in text if not is_rtl_char(c) and c.isalpha())
    return "rtl" if rtl_count > ltr_count else "ltr"


def split_by_direction(text: str) -> List[Tuple[str, str]]:
    """
    Split *text* into direction-homogeneous runs.

    Returns
    -------
    List[Tuple[str, str]]
        List of ("ltr" | "rtl" | "neutral", segment_text).
    """
    if not text:
        return []

    def direction(c: str) -> str:
        if is_rtl_char(c): return "rtl"
        if c.isalpha() or c.isdigit(): return "ltr"
        return "neutral"

    segs: List[Tuple[str, str]] = []
    cur_d = direction(text[0])
    cur_c = text[0]

    for c in text[1:]:
        d = direction(c)
        eff = d if d != "neutral" else cur_d
        if eff == cur_d:
            cur_c += c
        else:
            segs.append((cur_d, cur_c))
            cur_d, cur_c = eff, c

    segs.append((cur_d, cur_c))
    return segs


def logical_to_visual(text: str) -> str:
    """
    Apply Unicode BiDi algorithm to produce visually-ordered text.

    Uses ``python-bidi`` when installed; returns *text* unchanged
    as a fallback (logical order is correct for tokenizer input—visual
    reordering is only needed for display).

    Install: pip install python-bidi
    """
    try:
        from bidi.algorithm import get_display  # type: ignore
        return get_display(text)
    except ImportError:
        return text