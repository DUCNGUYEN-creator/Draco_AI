# DracoAI V1 — tokenizer/chatml/template.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — ChatML Template Engine
===========================================
Encodes conversation messages in ChatML format:

    <|im_start|>role
    content<|im_end|>
    <|im_start|>assistant
    (generation starts here)

Compatible with Qwen 3.5 9B Instruct checkpoint expectations.
"""

from typing import Callable, Dict, List, Optional

from ..constants import SPECIAL_TOKENS, BOS_TOKEN, EOS_TOKEN
from .roles import VALID_ROLES, normalize_role
from .formatter import format_system_prompt


class ChatMLTemplate:
    """
    Encodes a list of chat messages into token IDs.

    Parameters
    ----------
    encode_fn : Callable[[str], List[int]]
        The tokenizer's encode() function (without BOS/EOS, NFC-normalised).
    encode_bytes_fn : Callable[[bytes], List[int]]
        Encode raw bytes directly (used for the newline b"\\n").
    """

    def __init__(
        self,
        encode_fn:       Callable[[str], List[int]],
        encode_bytes_fn: Callable[[bytes], List[int]],
    ) -> None:
        self._encode       = encode_fn
        self._encode_bytes = encode_bytes_fn
        self._nl_cache:   Optional[List[int]] = None

        self._im_start = SPECIAL_TOKENS[BOS_TOKEN]
        self._im_end   = SPECIAL_TOKENS[EOS_TOKEN]

    def _newline_ids(self) -> List[int]:
        """Return cached newline token sequence."""
        if self._nl_cache is None:
            self._nl_cache = self._encode_bytes(b"\n")
        return self._nl_cache

    def invalidate_cache(self) -> None:
        """Call when the merge table changes to force newline recompute."""
        self._nl_cache = None

    def encode(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        """
        Encode a list of ChatML messages to token IDs.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Each dict must have "role" and "content" keys.
        add_generation_prompt : bool
            If True (default), append the assistant turn opener.
        system_prompt : Optional[str]
            If provided, prepended as a system message before all others.

        Returns
        -------
        List[int]
            Flat list of token IDs.
        """
        ids:    List[int] = []
        nl     = self._newline_ids()

        # Optional system message prepended
        if system_prompt is not None:
            messages = [{"role": "system", "content": format_system_prompt(system_prompt)}] + list(messages)

        for msg in messages:
            role    = normalize_role(msg.get("role", "user"))
            content = msg.get("content", "")

            ids.append(self._im_start)
            ids.extend(self._encode(role))
            ids.extend(nl)
            ids.extend(self._encode(content))
            ids.append(self._im_end)
            ids.extend(nl)

        if add_generation_prompt:
            ids.append(self._im_start)
            ids.extend(self._encode("assistant"))
            ids.extend(nl)

        return ids

    def decode_messages(
        self,
        ids: List[int],
        decode_fn: Callable[[List[int]], str],
    ) -> List[Dict[str, str]]:
        """
        Reverse a token ID sequence back into message dicts.

        Splits on <|im_start|> / <|im_end|> boundaries.

        Parameters
        ----------
        ids : List[int]
            Token IDs previously produced by encode().
        decode_fn : Callable[[List[int]], str]
            The tokenizer's decode() function (skip_special=False).

        Returns
        -------
        List[Dict[str, str]]
            List of {"role": ..., "content": ...} dicts.
        """
        messages: List[Dict[str, str]] = []
        current:  List[int] = []
        in_msg = False

        for tid in ids:
            if tid == self._im_start:
                in_msg  = True
                current = []
            elif tid == self._im_end:
                if in_msg and current:
                    raw  = decode_fn(current).strip()
                    # First line is the role; rest is content
                    lines = raw.split("\n", 1)
                    role    = lines[0].strip() if lines else "user"
                    content = lines[1].strip() if len(lines) > 1 else ""
                    messages.append({"role": role, "content": content})
                in_msg  = False
                current = []
            elif in_msg:
                current.append(tid)

        return messages
