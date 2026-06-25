# DracoAI V1 — tokenizer/pretokenizer/splitter.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Master Pre-tokenizer Splitter
==================================================
Language-aware text segmentation applied *before* BPE.

Pipeline per text segment (special tokens already removed upstream):
  1. Anchor emoji ZWJ sequences as indivisible atoms.
  2. Detect script of first non-space character -> routing strategy.
  3. Dispatch to the appropriate family/language splitter.
  4. For scriptio-continua scripts (Thai/Lao/Khmer/Myanmar/Tibetan):
     delegate to Thai dict-splitter or char-level split (when
     enable_scriptio_continua_split is True; otherwise word-split).
  5. For CJK / Kana / Hangul: char-level split (mirrors Qwen).
  6. For all other scripts: SOTA Latin-family regex.
"""

from typing import List

from ..unicode.emoji import split_preserving_emoji, contains_emoji
from ..unicode.scripts import (
    script_of, detect_dominant_script,
    SCRIPT_CJK, SCRIPT_HIRAGANA, SCRIPT_KATAKANA, SCRIPT_HANGUL,
    SCRIPT_ARABIC, SCRIPT_HEBREW,
    SCRIPT_CYRILLIC, SCRIPT_DEVANAGARI,
    SCRIPT_THAI, SCRIPT_LAO, SCRIPT_KHMER, SCRIPT_MYANMAR, SCRIPT_TIBETAN,
    SCRIPT_EMOJI,
)
from ..unicode.script_rules import strategy_for, STRATEGY_CHAR, STRATEGY_THAI

from .families.cjk_family import split_cjk
from .families.latin import split_latin
from .families.slavic import split_slavic
from .families.indic import split_indic
from .families.semitic import split_semitic
from .families.agglutinative import split_agglutinative
from .languages.thai import split_thai

_CJK_SCRIPTS     = {SCRIPT_CJK, SCRIPT_HIRAGANA, SCRIPT_KATAKANA, SCRIPT_HANGUL}
_SC_SCRIPTS      = {SCRIPT_LAO, SCRIPT_KHMER, SCRIPT_MYANMAR, SCRIPT_TIBETAN}
_SLAVIC_SCRIPTS  = {SCRIPT_CYRILLIC}
_SEMITIC_SCRIPTS = {SCRIPT_ARABIC, SCRIPT_HEBREW}
_INDIC_SCRIPTS   = {SCRIPT_DEVANAGARI}


def _route(
    text: str,
    enable_cjk_split:               bool,
    enable_thai_split:              bool,
    enable_scriptio_continua_split: bool,
) -> List[str]:
    """Route a single text segment (no emoji, no special tokens) to a splitter."""
    if not text:
        return []

    dom = detect_dominant_script(text)

    if dom in _CJK_SCRIPTS:
        if enable_cjk_split:
            return split_cjk(text)
        # CJK split disabled -> treat as Latin word-split
        return split_latin(text)

    if dom == SCRIPT_THAI:
        if enable_thai_split:
            return split_thai(text)
        if enable_scriptio_continua_split:
            return list(text)
        return split_latin(text)

    if dom in _SC_SCRIPTS:
        # Lao, Khmer, Myanmar, Tibetan: char-split when enabled, word-split otherwise
        if enable_scriptio_continua_split:
            return list(text)
        return split_latin(text)

    if dom in _SLAVIC_SCRIPTS:
        return split_slavic(text)

    if dom in _SEMITIC_SCRIPTS:
        return split_semitic(text)

    if dom in _INDIC_SCRIPTS:
        return split_indic(text)

    # Default: Latin / Vietnamese / mixed-script -> SOTA Latin regex
    return split_latin(text)


def pretokenize(
    text:                          str,
    enable_cjk_split:              bool = True,
    enable_thai_split:             bool = False,
    enable_scriptio_continua_split: bool = True,
) -> List[str]:
    """
    Master pre-tokenizer entry point.

    Parameters
    ----------
    text : str
        NFC-normalised text, special tokens already stripped.
    enable_cjk_split : bool
        If True (default), CJK/Kana/Hangul characters are segmented
        individually before BPE.
    enable_thai_split : bool
        If True, use library-based Thai word segmentation (requires
        pythainlp or thai_segmenter).  Char-level fallback otherwise.
    enable_scriptio_continua_split : bool
        If True (default), Lao/Khmer/Myanmar/Tibetan are character-level
        split.  If False, word-split is applied (same as Latin).

    Returns
    -------
    List[str]
        Non-empty segments ready for byte-encoding and BPE.
    """
    if not text:
        return []

    segments: List[str] = []

    # Step 1: Protect emoji ZWJ sequences as atomic units.
    # split_preserving_emoji returns either emoji sequences or plain text runs.
    parts = split_preserving_emoji(text) if contains_emoji(text) else [text]

    for part in parts:
        if not part:
            continue

        # Check if this part is itself an emoji sequence (from split_preserving_emoji).
        # An emoji atom should be passed through without further segmentation.
        if contains_emoji(part) and detect_dominant_script(part) == SCRIPT_EMOJI:
            segments.append(part)
            continue

        # Steps 2-6: Script-aware routing for non-emoji text.
        seg_results = _route(
            part,
            enable_cjk_split=enable_cjk_split,
            enable_thai_split=enable_thai_split,
            enable_scriptio_continua_split=enable_scriptio_continua_split,
        )
        segments.extend(seg_results)

    return [s for s in segments if s]
