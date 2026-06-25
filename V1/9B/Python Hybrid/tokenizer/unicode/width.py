# DracoAI V1 — tokenizer/unicode/width.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Fullwidth ↔ halfwidth conversion and display-width helpers."""

import unicodedata


def fullwidth_to_halfwidth(text: str) -> str:
    """Convert fullwidth ASCII (U+FF01–FF5E) and ideographic space to halfwidth."""
    result = []
    for c in text:
        cp = ord(c)
        if 0xFF01 <= cp <= 0xFF5E:
            result.append(chr(cp - 0xFEE0))
        elif cp == 0x3000:
            result.append(" ")
        else:
            result.append(c)
    return "".join(result)


def normalise_width(text: str) -> str:
    """Full NFKC width normalisation (fullwidth→halfwidth, ligatures, etc.)."""
    return unicodedata.normalize("NFKC", text)


def is_fullwidth(c: str) -> bool:
    cp = ord(c)
    return 0xFF01 <= cp <= 0xFF60 or cp == 0x3000


def display_width(text: str) -> int:
    """Estimate terminal display width in columns (CJK/fullwidth = 2, else 1)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)