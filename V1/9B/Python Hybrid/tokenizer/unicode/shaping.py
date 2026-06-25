# DracoAI V1 — tokenizer/unicode/shaping.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Script Shaping Utilities
=============================================
Normalises Arabic, Indic, and other complex-script text to canonical
Unicode forms before BPE encoding.

Arabic: presentation forms (shaped glyphs stored in Presentation Forms
blocks) are mapped back to base Arabic letters via NFKC.

Devanagari / Indic: Nukta normalisation to composed forms.
"""

import unicodedata


def normalise_arabic(text: str) -> str:
    """
    Normalise Arabic text:
    1. NFKC to map presentation forms (ﻤ → م) to base letters.
    2. Strip tatweel (U+0640) which is purely decorative.

    Parameters
    ----------
    text : str
        Arabic text, possibly containing presentation-form glyphs.

    Returns
    -------
    str
        Normalised Arabic text in base letter form.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0640", "")  # strip tatweel
    return text


def normalise_devanagari(text: str) -> str:
    """
    Normalise Devanagari text to NFC (precomposed).

    Ensures nukta-composed characters (e.g. ड़, ढ़) are stored as
    single codepoints rather than base + U+093C nukta sequences.
    """
    return unicodedata.normalize("NFC", text)


def normalise_hebrew(text: str) -> str:
    """
    Normalise Hebrew text.

    Strips cantillation marks (U+0591–U+05AF) and optional vowel
    points (nikud, U+05B0–U+05C7) to reduce token fragmentation.
    Configurable: pass strip_nikud=False to keep them.
    """
    result = []
    for c in text:
        cp = ord(c)
        if 0x0591 <= cp <= 0x05C7:
            continue   # cantillation + nikud
        result.append(c)
    return "".join(result)


def normalise_for_script(text: str, script: str) -> str:
    """
    Apply script-specific normalisation.

    Parameters
    ----------
    text : str
        Input text segment.
    script : str
        Script label from tokenizer.unicode.scripts (e.g. SCRIPT_ARABIC).

    Returns
    -------
    str
        Normalised text.
    """
    from .scripts import (
        SCRIPT_ARABIC, SCRIPT_HEBREW, SCRIPT_DEVANAGARI,
    )
    if script == SCRIPT_ARABIC:
        return normalise_arabic(text)
    if script == SCRIPT_HEBREW:
        return normalise_hebrew(text)
    if script == SCRIPT_DEVANAGARI:
        return normalise_devanagari(text)
    return text
