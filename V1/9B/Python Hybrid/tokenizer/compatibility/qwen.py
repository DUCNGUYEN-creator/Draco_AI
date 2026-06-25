# DracoAI V1 — tokenizer/compatibility/qwen.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Qwen Compatibility Layer
=============================================
Import / export helpers for Qwen 3.5 9B Instruct tokenizer format.
Wraps the serialization module's load_from_hf_json for convenience.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tokenizer import BPETokenizer


def load_qwen_tokenizer_json(tok: "BPETokenizer", path: str) -> None:
    """
    Load a Qwen-compatible ``tokenizer.json`` file into *tok*.

    Parameters
    ----------
    tok : BPETokenizer
        Target tokenizer instance to populate.
    path : str
        Path to the Qwen ``tokenizer.json`` checkpoint file.
    """
    from ..vocab.serialization import load_from_hf_json
    load_from_hf_json(tok, path)


def export_qwen_tokenizer_json(tok: "BPETokenizer") -> dict:
    """
    Export the tokenizer's merge rules in a Qwen-compatible JSON structure.

    Returns a dict that can be written to ``tokenizer.json``.

    Parameters
    ----------
    tok : BPETokenizer
        Source tokenizer.

    Returns
    -------
    dict
        HuggingFace-compatible tokenizer.json structure.
    """
    from ..constants import SPECIAL_TOKENS, QWEN_BASE_END

    merges = []
    for (a, b), _ in sorted(
        tok._merge_engine.merges.items(),
        key=lambda kv: (tok._merge_engine.get_rank(kv[0])
                        if tok._merge_engine.get_rank(kv[0]) is not None
                        else float("inf")),
    ):
        ba = tok._token_to_bytes(a)
        bb = tok._token_to_bytes(b)
        try:
            left_str  = ba.decode("utf-8")
            right_str = bb.decode("utf-8")
            merges.append(f"{left_str} {right_str}")
        except UnicodeDecodeError:
            pass  # skip non-UTF-8 byte pairs

    added_tokens = []
    for tid, byt in sorted(tok._ext_vocab.id_to_bytes.items()):
        try:
            content = byt.decode("utf-8")
        except UnicodeDecodeError:
            content = byt.decode("latin-1")
        added_tokens.append({
            "id":        tid,
            "content":   content,
            "special":   True,
            "added":     True,
        })

    return {
        "version": "1.0",
        "model": {
            "type":   "BPE",
            "merges": merges,
        },
        "added_tokens": added_tokens,
    }
