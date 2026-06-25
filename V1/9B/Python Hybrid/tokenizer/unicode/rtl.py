# DracoAI V1 — tokenizer/unicode/rtl.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — RTL compatibility shim
===========================================
Thin re-export from unicode/bidi.py for backwards compatibility.
New code should import directly from tokenizer.unicode.bidi.
"""
from .bidi import (  # noqa: F401
    is_rtl_char,
    contains_rtl,
    has_mixed_direction,
    base_direction,
    split_by_direction,
    logical_to_visual,
)