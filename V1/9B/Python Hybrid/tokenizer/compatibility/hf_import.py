# DracoAI V1 — tokenizer/compatibility/hf_import.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — HuggingFace Import Bridge
==============================================
Thin wrapper around load_from_hf_json for general HF tokenizer.json files.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tokenizer import BPETokenizer


def import_hf_tokenizer(tok: "BPETokenizer", tokenizer_json_path: str) -> None:
    """
    Import merge rules and added tokens from a HuggingFace tokenizer.json.

    Parameters
    ----------
    tok : BPETokenizer
        Target tokenizer instance.
    tokenizer_json_path : str
        Path to the ``tokenizer.json`` file from any HF BPE tokenizer.
    """
    from ..vocab.serialization import load_from_hf_json
    load_from_hf_json(tok, tokenizer_json_path)
