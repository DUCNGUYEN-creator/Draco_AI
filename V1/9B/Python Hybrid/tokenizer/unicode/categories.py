# DracoAI V1 — tokenizer/unicode/categories.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Unicode general category helpers."""

import unicodedata
from ..constants import VIET_CHARS


def category(c: str) -> str:
    return unicodedata.category(c)

def is_letter(c: str) -> bool:
    return unicodedata.category(c).startswith("L")

def is_combining(c: str) -> bool:
    return unicodedata.category(c) in ("Mn", "Mc", "Me")

def is_digit(c: str) -> bool:
    return unicodedata.category(c) == "Nd"

def is_punctuation(c: str) -> bool:
    return unicodedata.category(c).startswith("P")

def is_symbol(c: str) -> bool:
    return unicodedata.category(c).startswith("S")

def is_separator(c: str) -> bool:
    return unicodedata.category(c).startswith("Z")

def is_control(c: str) -> bool:
    return unicodedata.category(c) == "Cc"

def is_whitespace(c: str) -> bool:
    return c.isspace()

def is_vietnamese_diacritic(c: str) -> bool:
    return c in VIET_CHARS

def char_type(c: str) -> str:
    if c.isdigit():            return "NUM"
    if c in VIET_CHARS:        return "VIET_TONE"
    if c.isalpha():            return "ALPHA"
    if is_punctuation(c):      return "PUNCT"
    if c == " ":               return "SPACE"
    if c == "\n":              return "NEWLINE"
    if c.isspace():            return "SPACE"
    if is_control(c):          return "CONTROL"
    if is_symbol(c):           return "SYMBOL"
    return "OTHER"

def analyze_word(w: str) -> dict:
    types = [char_type(c) for c in w]
    return {
        "text": w, "length": len(w), "char_types": types,
        "has_viet":  any(t == "VIET_TONE" for t in types),
        "has_num":   any(t == "NUM" for t in types),
        "is_upper":  w.isupper(), "is_title": w.istitle(),
        "is_alpha":  all(c.isalpha() for c in w),
        "is_mixed":  len(set(types) - {"SPACE"}) > 1,
    }
