# DracoAI V1 — tokenizer/unicode/script_rules.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Script-level Tokenisation Rules
====================================================
Maps each Unicode script to its recommended pre-tokenisation strategy
and provides helpers for the pretokenizer routing layer.

Strategy codes
--------------
"char"      Each character is its own segment (CJK, Kana, Hangul,
            Scriptio-Continua).
"word"      Standard word/regex splitting (Latin, Cyrillic, Greek,
            Armenian, Georgian, Ethiopic, etc.).
"arabic"    Word-boundary splitting respecting Arabic letter forms.
"thai"      Library-assisted word segmentation (pythainlp) with
            char-level fallback.
"emoji"     Treat the entire ZWJ sequence as a single atom (handled
            upstream by emoji.split_preserving_emoji).
"""

from typing import Dict

from .scripts import (
    SCRIPT_EMOJI, SCRIPT_CJK, SCRIPT_HIRAGANA, SCRIPT_KATAKANA,
    SCRIPT_HANGUL, SCRIPT_ARABIC, SCRIPT_HEBREW, SCRIPT_CYRILLIC,
    SCRIPT_DEVANAGARI, SCRIPT_THAI, SCRIPT_LAO, SCRIPT_KHMER,
    SCRIPT_MYANMAR, SCRIPT_TIBETAN, SCRIPT_GEORGIAN, SCRIPT_ARMENIAN,
    SCRIPT_ETHIOPIC, SCRIPT_VIETNAMESE, SCRIPT_LATIN,
    SCRIPT_DIGIT, SCRIPT_WHITESPACE, SCRIPT_PUNCTUATION, SCRIPT_OTHER,
)

STRATEGY_CHAR       = "char"
STRATEGY_WORD       = "word"
STRATEGY_ARABIC     = "arabic"
STRATEGY_THAI       = "thai"
STRATEGY_EMOJI      = "emoji"

# Default routing table
SCRIPT_STRATEGY: Dict[str, str] = {
    SCRIPT_EMOJI:      STRATEGY_EMOJI,
    SCRIPT_CJK:        STRATEGY_CHAR,
    SCRIPT_HIRAGANA:   STRATEGY_CHAR,
    SCRIPT_KATAKANA:   STRATEGY_CHAR,
    SCRIPT_HANGUL:     STRATEGY_CHAR,
    SCRIPT_THAI:       STRATEGY_THAI,
    SCRIPT_LAO:        STRATEGY_CHAR,   # scriptio continua → char
    SCRIPT_KHMER:      STRATEGY_CHAR,
    SCRIPT_MYANMAR:    STRATEGY_CHAR,
    SCRIPT_TIBETAN:    STRATEGY_CHAR,
    SCRIPT_ARABIC:     STRATEGY_ARABIC,
    SCRIPT_HEBREW:     STRATEGY_ARABIC, # similar word-boundary rules
    SCRIPT_DEVANAGARI: STRATEGY_WORD,
    SCRIPT_CYRILLIC:   STRATEGY_WORD,
    SCRIPT_GEORGIAN:   STRATEGY_WORD,
    SCRIPT_ARMENIAN:   STRATEGY_WORD,
    SCRIPT_ETHIOPIC:   STRATEGY_WORD,
    SCRIPT_VIETNAMESE: STRATEGY_WORD,
    SCRIPT_LATIN:      STRATEGY_WORD,
    SCRIPT_DIGIT:      STRATEGY_WORD,
    SCRIPT_WHITESPACE: STRATEGY_WORD,
    SCRIPT_PUNCTUATION:STRATEGY_WORD,
    SCRIPT_OTHER:      STRATEGY_CHAR,   # safe fallback for unknown scripts
}


def strategy_for(script: str) -> str:
    """Return the tokenisation strategy for *script*."""
    return SCRIPT_STRATEGY.get(script, STRATEGY_WORD)


def requires_char_split(script: str) -> bool:
    """Return True if *script* should be split character-by-character."""
    return strategy_for(script) == STRATEGY_CHAR


def requires_thai_split(script: str) -> bool:
    """Return True if *script* uses dictionary-based Thai segmentation."""
    return strategy_for(script) == STRATEGY_THAI
