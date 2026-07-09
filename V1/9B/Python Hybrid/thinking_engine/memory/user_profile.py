# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
UserProfileManager
====================
Per-user preference store. Ported 1:1 from engine_v1.py's
``UserProfileManager``.
"""

from __future__ import annotations

from typing import Any, Dict


class UserProfileManager:
    """
    Profile fields:
        tone: "formal" | "casual" | "humorous"
        preferred_lang: "vi" | "en"
        expertise_level: "beginner" | "intermediate" | "expert"
        favorite_intents: List[str]
        creativity_override: float | None
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, dict] = {}

    def get_or_create(self, user_id: str) -> dict:
        if user_id not in self._profiles:
            self._profiles[user_id] = {
                "tone": "casual",
                "preferred_lang": "vi",
                "expertise_level": "intermediate",
                "favorite_intents": [],
                "creativity_override": None,
            }
        return self._profiles[user_id]

    def update(self, user_id: str, **kwargs: Any) -> None:
        profile = self.get_or_create(user_id)
        for k, v in kwargs.items():
            if k in profile:
                profile[k] = v

    def apply_to_intent(self, user_id: str, intent: dict) -> dict:
        """Returns a NEW intent dict (does not mutate caller's dict)."""
        if user_id not in self._profiles:
            return intent
        p = self._profiles[user_id]
        intent = dict(intent)
        if p["creativity_override"] is not None:
            intent["creativity"] = p["creativity_override"]
        if p["preferred_lang"]:
            intent["preferred_lang"] = p["preferred_lang"]
        return intent
