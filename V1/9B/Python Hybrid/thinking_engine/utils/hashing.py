# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Stable hashing helpers used by KG dedup, evidence cache keys, etc."""

from __future__ import annotations

import hashlib


def stable_hash(*parts: str) -> str:
    """MD5 of lower-cased, pipe-joined parts. Deterministic across runs
    (unlike Python's salted built-in hash()), which matters for caching
    and for the KnowledgeGraph's triple-dedup key."""
    joined = "|".join(p.lower() for p in parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
