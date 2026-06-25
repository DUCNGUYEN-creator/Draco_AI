# DracoAI V1 — tokenizer/vocab/serialization.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Serialisation
===================================
Save and load tokenizer state to/from disk.

Format: JSON file ``tokenizer_draco.json`` inside a directory.

Merge rules are stored as [[a, b, merged_id], ...] in rank order.

Extension vocab bytes are stored as base64-encoded strings so that
tokens containing non-Latin-1 bytes (e.g. emoji) survive the roundtrip
without corruption.  Older checkpoints that used latin-1 encoding are
also read correctly via a fallback path.
"""

import base64
import json
import os
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..tokenizer import BPETokenizer


_CONFIG_FILENAME = "tokenizer_draco.json"
_FORMAT_VERSION  = 3   # bump when storage format changes


def _config_field_names(tok: "BPETokenizer") -> set:
    """Return the dataclass field names supported by tok.config."""
    return {field.name for field in fields(type(tok.config))}


def _apply_saved_config(tok: "BPETokenizer", config: Dict[str, Any]) -> None:
    """Apply persisted TokenizerConfig values to an existing tokenizer."""
    payload = config.get("tokenizer_config")
    if isinstance(payload, dict):
        valid_fields = _config_field_names(tok)
        for key, value in payload.items():
            if key in valid_fields:
                setattr(tok.config, key, value)
    else:
        # Legacy v1/v2 checkpoints stored only these top-level values.
        if "model" in config:
            tok.config.model_name = config["model"]
        tok.config.strict_utf8 = config.get("strict_utf8", False)

    tok.config.validate()

    # The special-token regex is built from config.extra_special_tokens at
    # tokenizer construction time, so rebuild it after loading config.
    from .special_tokens import SpecialTokenRegistry

    tok._special = SpecialTokenRegistry(extra=tok.config.extra_special_tokens)


def save(tok: "BPETokenizer", path: str) -> None:
    """
    Save tokenizer state to *path* directory.

    Parameters
    ----------
    tok : BPETokenizer
        The tokenizer instance to serialise.
    path : str
        Directory path (created if it does not exist).
    """
    os.makedirs(path, exist_ok=True)

    # Merges in rank order so they can be loaded and reconstructed correctly.
    merges_list: List[List[int]] = []
    for pair, merged_id in tok._merge_engine._ranks.items():
        a, b = pair
        merges_list.append([a, b, merged_id])

    # Extension vocab: store raw bytes as base64 to safely roundtrip
    # any byte sequence including non-Latin-1 (emoji tokens, etc.).
    ext_vocab: Dict[str, str] = {
        str(tid): base64.b64encode(byt).decode("ascii")
        for tid, byt in tok._ext_vocab.id_to_bytes.items()
    }

    config: Dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "model":          tok.config.model_name,
        "tokenizer_config": asdict(tok.config),
        "merges":         merges_list,
        "ext_vocab":      ext_vocab,
        "strict_utf8":    tok.config.strict_utf8,
    }

    filepath = os.path.join(path, _CONFIG_FILENAME)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load(tok: "BPETokenizer", path: str, load_config: bool = True) -> None:
    """
    Load tokenizer state from *path* directory into *tok* in-place.

    Parameters
    ----------
    tok : BPETokenizer
        The tokenizer instance to populate.
    path : str
        Directory containing ``tokenizer_draco.json``.
    load_config : bool
        If True, persisted TokenizerConfig values are applied. If False,
        the existing tokenizer config is preserved.
    """
    cfg_path = os.path.join(path, _CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        return

    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    fmt_version = config.get("format_version", 1)

    if load_config:
        _apply_saved_config(tok, config)

    rules = [((a, b), c) for a, b, c in config.get("merges", [])]
    tok._merge_engine.load(rules)

    # Rebuild vocab bytes in rule order to ensure merge byte dependencies are met.
    for (a, b), merged_id in rules:
        ba = tok._token_to_bytes(a)
        bb = tok._token_to_bytes(b)
        tok._vocab.add(merged_id, ba + bb)

    # Extension vocab: v2 uses base64; v1 used latin-1 (fallback).
    for k_str, v_enc in config.get("ext_vocab", {}).items():
        token_id = int(k_str)
        if fmt_version >= 2:
            try:
                raw_bytes = base64.b64decode(v_enc.encode("ascii"))
            except Exception:
                # Graceful fallback if somehow stored as latin-1
                raw_bytes = v_enc.encode("latin-1")
        else:
            # Legacy v1 format: value is the token string stored as latin-1
            # decoded bytes.  Re-encode as latin-1 to recover raw bytes.
            raw_bytes = v_enc.encode("latin-1")

        # Reconstruct the token string from raw bytes (UTF-8).
        try:
            token_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = raw_bytes.decode("latin-1")

        tok._ext_vocab.add_with_id(token_str, token_id)

    tok._nl_cache = None
    tok._chatml.invalidate_cache()


def _gpt2_bytes_to_unicode() -> dict:
    """
    Build the GPT-2 byte-to-unicode mapping used by Qwen/GPT tokenizers.

    GPT-2 maps raw bytes (0-255) to unicode characters so the vocabulary
    consists only of printable characters (no raw control bytes).  This
    lets BPE operate on a character-level vocabulary without byte strings.

    Inverse mapping (unicode char -> raw byte) is needed when loading
    Qwen tokenizer.json merge rules back into our byte-level BPE.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("\xa1"), ord("\xac") + 1)) +
        list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


# Built once at module load time
_UNICODE_TO_BYTE: dict = _gpt2_bytes_to_unicode()


def _vocab_id_for(raw_vocab: Dict[str, Any], token_str: str) -> Optional[int]:
    """Return the integer HF vocab ID for token_str when available."""
    token_id = raw_vocab.get(token_str)
    return token_id if isinstance(token_id, int) else None


def _can_register_vocab_id(tok: "BPETokenizer", token_id: int, token_bytes: bytes) -> bool:
    """Return True if token_id can safely represent token_bytes."""
    from ..constants import SPECIAL_ID_TO_NAME, QWEN_BASE_END

    if token_id in SPECIAL_ID_TO_NAME or token_id >= QWEN_BASE_END:
        return False
    existing = tok._vocab.get_bytes(token_id)
    return existing is None or existing == token_bytes


def _ensure_vocab_token(
    tok: "BPETokenizer",
    token_str: str,
    token_bytes: bytes,
    raw_vocab: Dict[str, Any],
) -> int:
    """Resolve/register an internal token ID for imported HF token bytes."""
    existing_id = tok._vocab.get_id(token_bytes)
    if existing_id is not None:
        return existing_id

    vocab_id = _vocab_id_for(raw_vocab, token_str)
    if vocab_id is not None and _can_register_vocab_id(tok, vocab_id, token_bytes):
        tok._vocab.add(vocab_id, token_bytes)
        return vocab_id

    token_id = tok._vocab.max_id() + 1
    tok._vocab.add(token_id, token_bytes)
    return token_id


def _gpt2_token_to_bytes(token_str: str) -> bytes:
    """
    Decode a GPT-2/Qwen vocabulary token string back to raw bytes.

    Qwen tokens use the GPT-2 byte-level encoding where each raw byte
    is represented as a specific unicode character.  For example:
      - 'G' with combining char (U+0120) -> 0x20 (space)
      - Corresponding char for 0x0A (newline)
      - 'hello' -> b'hello' (ASCII stays as-is)

    Parameters
    ----------
    token_str : str
        A token string from the Qwen vocabulary.

    Returns
    -------
    bytes
        The corresponding raw byte sequence.
    """
    result = []
    for c in token_str:
        if c in _UNICODE_TO_BYTE:
            result.append(_UNICODE_TO_BYTE[c])
        else:
            # Fallback: encode as UTF-8 (for tokens that are plain ASCII/unicode)
            result.extend(c.encode("utf-8"))
    return bytes(result)


def load_from_hf_json(tok: "BPETokenizer", path: str) -> None:
    """
    Load merge rules from a HuggingFace ``tokenizer.json`` file.

    Handles both:
    - GPT-2/Qwen style: tokens encoded as GPT-2 byte-level unicode chars
      (Ġ for space, Ċ for newline, etc.)
    - Plain UTF-8 style: tokens stored as raw UTF-8 strings

    Parameters
    ----------
    tok : BPETokenizer
        The tokenizer instance to populate.
    path : str
        Path to a HuggingFace ``tokenizer.json`` file.
    """
    from ..constants import SPECIAL_ID_TO_NAME, QWEN_BASE_END

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Extension / added tokens
    for entry in data.get("added_tokens", []):
        tid     = entry["id"]
        content = entry["content"]
        if tid not in SPECIAL_ID_TO_NAME and tid >= QWEN_BASE_END:
            tok._ext_vocab.add_with_id(content, tid)

    # Determine if this uses GPT-2 byte-level encoding by checking the
    # model type or by sampling the vocabulary.
    model      = data.get("model", {})
    model_type = model.get("type", "").lower()
    raw_vocab  = model.get("vocab", {})

    # Detect GPT-2 byte-level encoding: Qwen/GPT-2 models have 'Ġ' (U+0120) in vocab.
    use_gpt2_encoding = (
        model_type in ("bpe",) and
        any("\u0120" in k or "\u010a" in k for k in raw_vocab)
    )

    # Merge rules
    raw_merges  = model.get("merges", [])
    merge_rules = []

    for pair_str in raw_merges:
        parts = pair_str.split(" ", 1)
        if len(parts) != 2:
            continue
        left_str, right_str = parts[0], parts[1]

        if use_gpt2_encoding:
            left_b  = _gpt2_token_to_bytes(left_str)
            right_b = _gpt2_token_to_bytes(right_str)
        else:
            left_b  = left_str.encode("utf-8")
            right_b = right_str.encode("utf-8")

        lid = _ensure_vocab_token(tok, left_str, left_b, raw_vocab)
        rid = _ensure_vocab_token(tok, right_str, right_b, raw_vocab)

        merged_str = left_str + right_str
        merged_b   = left_b + right_b
        merged_id  = _ensure_vocab_token(tok, merged_str, merged_b, raw_vocab)

        merge_rules.append(((lid, rid), merged_id))

    tok._merge_engine.load(merge_rules)
    tok._nl_cache = None
    tok._chatml.invalidate_cache()
