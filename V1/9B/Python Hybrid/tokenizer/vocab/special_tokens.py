# DracoAI V1 — tokenizer/vocab/special_tokens.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Special Token Registry
===========================================
Manages the set of special tokens, their IDs, and their regex pattern
for splitting text during encoding.
"""

import re
from typing import Dict, Optional

from ..constants import SPECIAL_TOKENS, SPECIAL_ID_TO_NAME


class SpecialTokenRegistry:
    """
    Maintains the special token table and a compiled split pattern.

    Allows runtime addition of new special tokens (e.g. tool-specific
    tokens) beyond the base Qwen set.
    """

    def __init__(self, extra: Optional[Dict[str, int]] = None) -> None:
        # Start from the base Qwen special tokens
        self._str_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self._id_to_str: Dict[int, str] = dict(SPECIAL_ID_TO_NAME)
        self._pattern: Optional[re.Pattern] = None

        if extra:
            # Validate all extra tokens before registering any, to fail fast
            # on conflicts without leaving the registry in a partial state.
            for token_str, token_id in extra.items():
                existing_id = self._str_to_id.get(token_str)
                if existing_id is not None and existing_id != token_id:
                    raise ValueError(
                        f"Token string {token_str!r} is already registered as "
                        f"ID {existing_id}; cannot remap it to {token_id}"
                    )
                existing = self._id_to_str.get(token_id)
                if existing is not None and existing != token_str:
                    raise ValueError(
                        f"Token ID {token_id} is already used by {existing!r}; "
                        f"cannot register {token_str!r}"
                    )
            # Batch-add all tokens, then rebuild pattern once.
            for token_str, token_id in extra.items():
                self._str_to_id[token_str] = token_id
                self._id_to_str[token_id]  = token_str

        self._rebuild_pattern()

    def register(self, token_str: str, token_id: int) -> None:
        """
        Register a new special token.

        Parameters
        ----------
        token_str : str
            The token string (e.g. "<|custom|>").
        token_id : int
            The token ID (must not conflict with existing tokens).

        Raises
        ------
        ValueError
            If *token_id* is already used by a different token string.
        """
        existing = self._id_to_str.get(token_id)
        if existing is not None and existing != token_str:
            raise ValueError(
                f"Token ID {token_id} is already used by {existing!r}; "
                f"cannot register {token_str!r}"
            )
        existing_id = self._str_to_id.get(token_str)
        if existing_id is not None and existing_id != token_id:
            raise ValueError(
                f"Token string {token_str!r} is already registered as "
                f"ID {existing_id}; cannot remap it to {token_id}"
            )
        self._str_to_id[token_str] = token_id
        self._id_to_str[token_id] = token_str
        self._rebuild_pattern()

    def _rebuild_pattern(self) -> None:
        """Rebuild the compiled split regex from current token set."""
        # Sort longest-first so longer tokens match before their prefixes.
        sorted_tokens = sorted(self._str_to_id.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in sorted_tokens)
        self._pattern = re.compile(f"({pattern})")

    # ── Lookups ───────────────────────────────────────────────────────

    def id_of(self, token_str: str) -> Optional[int]:
        """Return the ID for *token_str*, or None."""
        return self._str_to_id.get(token_str)

    def str_of(self, token_id: int) -> Optional[str]:
        """Return the token string for *token_id*, or None."""
        return self._id_to_str.get(token_id)

    def is_special(self, token_id: int) -> bool:
        """Return True if *token_id* is a special token."""
        return token_id in self._id_to_str

    def is_special_str(self, token_str: str) -> bool:
        """Return True if *token_str* is a registered special token."""
        return token_str in self._str_to_id

    @property
    def split_pattern(self) -> re.Pattern:
        """Compiled regex for splitting text on special tokens."""
        if self._pattern is None:
            self._rebuild_pattern()
        return self._pattern  # type: ignore[return-value]

    @property
    def str_to_id(self) -> Dict[str, int]:
        return dict(self._str_to_id)

    @property
    def id_to_str(self) -> Dict[int, str]:
        return dict(self._id_to_str)

    def __len__(self) -> int:
        return len(self._str_to_id)
