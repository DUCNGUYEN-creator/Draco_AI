# DracoAI V1 — tokenizer/unicode/ansi.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — ANSI Escape Sequence Utilities
====================================================
Strip or detect ANSI terminal escape sequences that may appear in
code outputs, log snippets, or terminal-copied text.
"""

import re

# CSI sequences: ESC [ <params> <final-byte>  (e.g. \x1b[31m, \x1b[0m)
# OSC sequences: ESC ] <string> BEL           (e.g. \x1b]0;title\x07)
_ANSI_CSI_RE = re.compile(r"\x1b\[[\x20-\x3F]*[\x40-\x7E]")
_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Other ESC sequences: ESC <non-[ char>  (two-char sequences like ESC M, ESC =)
# Deliberately narrow: only matches ESC followed by exactly one non-bracket letter.
# This avoids over-matching content that happens to follow an ESC byte.
_ANSI_2CHAR_RE = re.compile(r"\x1b[^\x1b\[]")


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from *text*.

    Handles CSI (colour/cursor), OSC (title strings), and simple
    two-character ESC sequences.  Applied in order from most-specific
    to least-specific to avoid partial matches.
    """
    text = _ANSI_CSI_RE.sub("", text)
    text = _ANSI_OSC_RE.sub("", text)
    text = _ANSI_2CHAR_RE.sub("", text)
    return text


def contains_ansi(text: str) -> bool:
    """Return True if *text* contains ANSI escape sequences."""
    return bool(
        _ANSI_CSI_RE.search(text) or
        _ANSI_OSC_RE.search(text) or
        _ANSI_2CHAR_RE.search(text)
    )


def strip_control_chars(text: str) -> str:
    """
    Remove C0 / C1 control characters (U+0000–U+001F, U+007F–U+009F)
    except tab (U+0009), newline (U+000A), and carriage return (U+000D),
    which are meaningful whitespace.
    """
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)