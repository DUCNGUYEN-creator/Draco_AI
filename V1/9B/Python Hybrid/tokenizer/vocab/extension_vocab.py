# DracoAI V1 — tokenizer/vocab/extension_vocab.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Extension Vocabulary
=========================================
Manages custom tokens added above the QWEN_BASE_END boundary (151936).

Extension tokens are assigned IDs starting at QWEN_BASE_END, skipping
any already-used special token IDs.  The registry is idempotent: adding
the same token string twice returns the same ID.
"""

import re
from typing import Dict, List, Optional, Set

from ..constants import QWEN_BASE_END, SPECIAL_ID_TO_NAME


class ExtensionVocab:
    """
    Registry for custom extension tokens.

    Token IDs are allocated above QWEN_BASE_END and do not overlap with
    the built-in special token IDs (151643–151652).
    """

    def __init__(self) -> None:
        self._id_to_bytes: Dict[int, bytes]  = {}
        self._bytes_to_id: Dict[bytes, int]  = {}
        self._str_to_id: Dict[str, int]      = {}
        self._reserved: Set[int]             = set(SPECIAL_ID_TO_NAME.keys())
        self._next_candidate: int            = QWEN_BASE_END
        self._pattern: Optional[re.Pattern]  = None

    def _alloc_id(self) -> int:
        """Find and return the next free extension ID."""
        used = set(self._id_to_bytes.keys()) | self._reserved
        while self._next_candidate in used:
            self._next_candidate += 1
        allocated               = self._next_candidate
        self._next_candidate   += 1
        return allocated

    def add(self, token_str: str) -> int:
        """
        Register *token_str* as an extension token.

        Idempotent: returns the existing ID if already registered.

        Parameters
        ----------
        token_str : str
            The token string to register (e.g. "<draco_special>").

        Returns
        -------
        int
            The token ID (>= QWEN_BASE_END).
        """
        if not token_str:
            raise ValueError("Extension token string must not be empty")

        existing = self._str_to_id.get(token_str)
        if existing is not None:
            return existing

        b = token_str.encode("utf-8")
        if b in self._bytes_to_id:
            token_id = self._bytes_to_id[b]
            self._str_to_id[token_str] = token_id
            self._rebuild_pattern()
            return token_id

        token_id                   = self._alloc_id()
        self._id_to_bytes[token_id] = b
        self._bytes_to_id[b]       = token_id
        self._str_to_id[token_str] = token_id
        self._rebuild_pattern()
        return token_id

    def add_with_id(self, token_str: str, token_id: int) -> None:
        """
        Register *token_str* with an explicit *token_id*.

        Used when loading a saved checkpoint that already has assigned IDs.

        Raises
        ------
        ValueError
            If *token_id* conflicts with a built-in special token.
        """
        if token_id in self._reserved:
            raise ValueError(
                f"Token ID {token_id} is reserved for a special token; "
                f"cannot register extension token {token_str!r}"
            )
        if not token_str:
            raise ValueError("Extension token string must not be empty")
        b = token_str.encode("utf-8")
        existing_bytes = self._id_to_bytes.get(token_id)
        if existing_bytes is not None and existing_bytes != b:
            raise ValueError(
                f"Token ID {token_id} is already used by a different "
                "extension token"
            )
        existing_id = self._bytes_to_id.get(b)
        if existing_id is not None and existing_id != token_id:
            raise ValueError(
                f"Extension token {token_str!r} is already registered as "
                f"ID {existing_id}; cannot remap it to {token_id}"
            )
        self._id_to_bytes[token_id] = b
        self._bytes_to_id[b]        = token_id
        self._str_to_id[token_str]  = token_id
        if token_id >= self._next_candidate:
            self._next_candidate = token_id + 1
        self._rebuild_pattern()

    def _rebuild_pattern(self) -> None:
        """Rebuild the compiled split regex from current extension tokens."""
        if not self._str_to_id:
            self._pattern = None
            return
        sorted_tokens = sorted(self._str_to_id.keys(), key=len, reverse=True)
        self._pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
        )

    # ── Lookups ───────────────────────────────────────────────────────

    def get_bytes(self, token_id: int) -> Optional[bytes]:
        """Return bytes for *token_id*, or None."""
        return self._id_to_bytes.get(token_id)

    def get_id(self, token_bytes: bytes) -> Optional[int]:
        """Return ID for *token_bytes*, or None."""
        return self._bytes_to_id.get(token_bytes)

    def id_of(self, token_str: str) -> Optional[int]:
        """Return ID for *token_str*, or None."""
        return self._str_to_id.get(token_str)

    def is_token_str(self, token_str: str) -> bool:
        """Return True if *token_str* is a registered extension token."""
        return token_str in self._str_to_id

    def split(self, text: str) -> List[str]:
        """Split text on registered extension tokens, preserving tokens."""
        if self._pattern is None:
            return [text]
        return [part for part in self._pattern.split(text) if part]

    def contains_id(self, token_id: int) -> bool:
        return token_id in self._id_to_bytes

    def contains_bytes(self, token_bytes: bytes) -> bool:
        return token_bytes in self._bytes_to_id

    def __len__(self) -> int:
        return len(self._id_to_bytes)

    @property
    def id_to_bytes(self) -> Dict[int, bytes]:
        return dict(self._id_to_bytes)

    @property
    def str_to_id(self) -> Dict[str, int]:
        return dict(self._str_to_id)
