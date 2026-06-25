# DracoAI V1 — tokenizer/unicode/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""DracoAI Tokenizer — unicode sub-package."""

from .normalize    import normalize, nfc, nfkc, nfd
from .grapheme     import (iter_graphemes, grapheme_len,
                                            split_graphemes, is_safe_to_yield,
                                            safe_flush_point)
from .emoji        import (iter_emoji_sequences, contains_emoji,
                                            split_preserving_emoji)
from .scripts      import (script_of, detect_dominant_script,
                                            script_segments, is_scriptio_continua)
from .script_rules import (strategy_for, requires_char_split,
                                            requires_thai_split)
from .categories   import char_type, analyze_word
from .width        import fullwidth_to_halfwidth, normalise_width
from .bidi         import is_rtl_char, contains_rtl
from .ansi         import strip_ansi, strip_control_chars
from .shaping      import normalise_for_script

__all__ = [
    "normalize", "nfc", "nfkc", "nfd",
    "iter_graphemes", "grapheme_len", "split_graphemes",
    "is_safe_to_yield", "safe_flush_point",
    "iter_emoji_sequences", "contains_emoji", "split_preserving_emoji",
    "script_of", "detect_dominant_script", "script_segments", "is_scriptio_continua",
    "strategy_for", "requires_char_split", "requires_thai_split",
    "char_type", "analyze_word",
    "fullwidth_to_halfwidth", "normalise_width",
    "is_rtl_char", "contains_rtl",
    "strip_ansi", "strip_control_chars",
    "normalise_for_script",
]