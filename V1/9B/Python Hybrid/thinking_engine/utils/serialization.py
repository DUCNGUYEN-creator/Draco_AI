# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""JSON-safe (de)serialization helpers used by hallucination cache/registry."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def to_json(obj: Any) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    return json.dumps(obj, default=str, ensure_ascii=False)


def from_json(text: str) -> Any:
    return json.loads(text)
