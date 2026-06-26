# DracoAI V1 — tokenizer/__init.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI V1 — Tokenizer Package
================================
Qwen 3.5 9B-compatible BPE tokenizer with full multilingual,
emoji, streaming, and grapheme-aware decode support.
"""

from .tokenizer import BPETokenizer
from .config    import TokenizerConfig, DEFAULT_CONFIG
from .constants import (
    SPECIAL_TOKENS, SPECIAL_ID_TO_NAME, QWEN_BASE_END,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
)

__all__ = [
    "BPETokenizer",
    "TokenizerConfig",
    "DEFAULT_CONFIG",
    "SPECIAL_TOKENS",
    "SPECIAL_ID_TO_NAME",
    "QWEN_BASE_END",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "UNK_TOKEN",
]