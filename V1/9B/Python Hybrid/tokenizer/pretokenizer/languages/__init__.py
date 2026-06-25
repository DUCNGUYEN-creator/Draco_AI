# DracoAI V1 — tokenizer/pretokenizer/languages/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Per-language pre-tokenizer dispatch."""

from .vietnamese import split_vietnamese
from .thai       import split_thai, thai_available
from .japanese   import split_japanese
from .korean     import split_korean
from .russian    import split_russian
from .spanish    import split_spanish
from .mongolian  import split_mongolian

__all__ = [
    "split_vietnamese", "split_thai", "thai_available",
    "split_japanese", "split_korean",
    "split_russian", "split_spanish", "split_mongolian",
]