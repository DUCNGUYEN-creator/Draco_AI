# DracoAI V1 — tokenizer/vocab/vocab.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Core Vocabulary
=====================================
Manages the base byte vocabulary (IDs 0–255), BPE-merged tokens
(IDs 256+), and provides fast token↔bytes lookups.
"""

from typing import Dict, Optional


class Vocabulary:
    """
    Bidirectional token-to-bytes mapping for the base vocabulary.

    IDs 0–255 correspond to single bytes.
    IDs 256+ are BPE-merged tokens (bytes concatenated from their
    constituent sub-tokens).
    """

    def __init__(self) -> None:
        self._id_to_bytes: Dict[int, bytes] = {}
        self._bytes_to_id: Dict[bytes, int] = {}

        # Populate byte tokens 0–255
        for i in range(256):
            b = bytes([i])
            self._id_to_bytes[i] = b
            self._bytes_to_id[b] = i

    # ── Mutations ─────────────────────────────────────────────────────

    def add(self, token_id: int, token_bytes: bytes) -> None:
        """
        Register a token.

        If *token_id* is already present, the bytes mapping is updated
        (useful for re-loading checkpoints that recompute merged bytes).
        """
        old_bytes = self._id_to_bytes.get(token_id)
        if old_bytes is not None and old_bytes != token_bytes:
            self._bytes_to_id.pop(old_bytes, None)

        old_id = self._bytes_to_id.get(token_bytes)
        if old_id is not None and old_id != token_id:
            self._id_to_bytes.pop(old_id, None)

        self._id_to_bytes[token_id] = token_bytes
        self._bytes_to_id[token_bytes] = token_id

    def remove(self, token_id: int) -> None:
        """Remove a token by ID (does NOT remove base byte tokens 0–255)."""
        if token_id < 256:
            return  # never remove base bytes
        b = self._id_to_bytes.pop(token_id, None)
        if b is not None:
            self._bytes_to_id.pop(b, None)

    # ── Lookups ───────────────────────────────────────────────────────

    def get_bytes(self, token_id: int) -> Optional[bytes]:
        """Return the bytes for *token_id*, or None if not found."""
        return self._id_to_bytes.get(token_id)

    def get_id(self, token_bytes: bytes) -> Optional[int]:
        """Return the token ID for *token_bytes*, or None if not found."""
        return self._bytes_to_id.get(token_bytes)

    def contains_id(self, token_id: int) -> bool:
        """Return True if *token_id* is in the vocabulary."""
        return token_id in self._id_to_bytes

    def contains_bytes(self, token_bytes: bytes) -> bool:
        """Return True if *token_bytes* maps to a known token."""
        return token_bytes in self._bytes_to_id

    # ── Inspection ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._id_to_bytes)

    def max_id(self) -> int:
        """Return the highest registered token ID."""
        return max(self._id_to_bytes.keys(), default=0)

    @property
    def id_to_bytes(self) -> Dict[int, bytes]:
        """Read-only view of the id→bytes dict (for serialisation)."""
        return dict(self._id_to_bytes)

    @property
    def bytes_to_id(self) -> Dict[bytes, int]:
        """Read-only view of the bytes→id dict."""
        return dict(self._bytes_to_id)
