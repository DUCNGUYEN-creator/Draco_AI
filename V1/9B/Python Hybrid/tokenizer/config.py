# DracoAI V1 — tokenizer/config.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Configuration
====================================
TokenizerConfig is the single runtime-configuration object for the entire
tokenizer package.  Pass an instance to BPETokenizer.__init__() to
override any default behaviour.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TokenizerConfig:
    """
    Runtime configuration for DracoAI BPETokenizer.

    Parameters
    ----------
    model_name : str
        Stored in saved checkpoints for identification.
    strict_utf8 : bool
        True  -> raise UnicodeDecodeError on invalid bytes.
        False (default) -> replace with U+FFFD.
    grapheme_flush : bool
        True (default) -> stream_decode flushes on grapheme-cluster
        boundaries (safe for emoji ZWJ sequences and Vietnamese diacritics).
        False -> flush on any FLUSH_CHARS character (legacy mode).
    normalisation_form : str
        Unicode normalisation before encoding: "NFC" (default), "NFKC",
        "NFD", or "NFKD".
    enable_cjk_split : bool
        True (default) -> CJK / Kana characters are individually segmented
        before BPE, matching Qwen pre-tokenizer behaviour.
    enable_thai_split : bool
        True  -> use dictionary-based Thai word segmentation when pythainlp
        or thai_segmenter is installed; otherwise falls back to char-level.
        False (default) -> character-level split for Thai / Scriptio-Continua.
    enable_scriptio_continua_split : bool
        True (default) -> Lao, Khmer, Myanmar, Tibetan are split at
        character level (like CJK) to avoid BPE over-tokenisation.
        False -> treat these scripts as word-boundary scripts (Latin-style).
    add_bos_by_default : bool
        Prepend BOS in encode() when the caller does not explicitly specify
        add_bos.  False (default).
    add_eos_by_default : bool
        Append EOS in encode() when the caller does not explicitly specify
        add_eos.  False (default).
    word_cache_size : int
        Maximum number of word-level BPE results to memoize (LRU).
        0 disables the cache.  Default: 30_000.
    max_input_length : int
        Truncate input text beyond this length to prevent DoS.
    max_consecutive_spaces : int
        Collapse runs of spaces longer than this to prevent DoS.
    max_combining_marks : int
        Strip excess combining marks (Zalgo protection).
    extra_special_tokens : Dict[str, int]
        Additional special tokens beyond the built-in Qwen set.
        Keys are token strings; values are explicit IDs (must not
        conflict with built-ins or QWEN_BASE_END range).
    """

    model_name:                     str            = "Qwen3.5-9B-Instruct"
    strict_utf8:                    bool           = False
    grapheme_flush:                 bool           = True
    normalisation_form:             str            = "NFC"
    enable_cjk_split:               bool           = True
    enable_thai_split:              bool           = False
    enable_scriptio_continua_split: bool           = True
    add_bos_by_default:             bool           = False
    add_eos_by_default:             bool           = False
    word_cache_size:                int            = 30_000
    max_input_length:               int            = 200_000
    max_consecutive_spaces:         int            = 32
    max_combining_marks:            int            = 16
    extra_special_tokens:           Dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError for invalid combinations."""
        valid_nf = {"NFC", "NFKC", "NFD", "NFKD"}
        if self.normalisation_form not in valid_nf:
            raise ValueError(
                f"normalisation_form must be one of {valid_nf}, "
                f"got {self.normalisation_form!r}"
            )
        if self.word_cache_size < 0:
            raise ValueError("word_cache_size must be >= 0")
        if self.max_input_length < 1:
            raise ValueError("max_input_length must be >= 1")
        if self.max_consecutive_spaces < 0:
            raise ValueError("max_consecutive_spaces must be >= 0")
        if self.max_combining_marks < 0:
            raise ValueError("max_combining_marks must be >= 0")


DEFAULT_CONFIG = TokenizerConfig()