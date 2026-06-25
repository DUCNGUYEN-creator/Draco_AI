# DracoAI V1 — tokenizer/chatml/roles.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — ChatML Roles
==================================
Valid role names and normalisation helpers for ChatML messages.
"""

from typing import FrozenSet

# Standard ChatML roles supported by Qwen 3.5 9B Instruct
VALID_ROLES: FrozenSet[str] = frozenset({
    "system",
    "user",
    "assistant",
    "tool",        # function / tool response role
    "function",    # alternative tool role name
})

# Aliases that should be normalised to canonical roles
_ROLE_ALIASES = {
    "human":     "user",
    "model":     "assistant",
    "bot":       "assistant",
    "ai":        "assistant",
    "gpt":       "assistant",
    "claude":    "assistant",
    "draco":     "assistant",
}


def normalize_role(role: str) -> str:
    """
    Normalise a role string to a canonical ChatML role.

    Unknown roles are returned as-is (allows custom roles for
    multi-agent setups).

    Parameters
    ----------
    role : str
        Raw role string from the message dict.

    Returns
    -------
    str
        Canonical role string.
    """
    r = role.strip().lower()
    return _ROLE_ALIASES.get(r, r)


def is_valid_role(role: str) -> bool:
    """Return True if *role* is a standard ChatML role."""
    return normalize_role(role) in VALID_ROLES