# DracoAI V1 — thinking_engine/perception/understanding/spatial_solver.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SpatialSolver
===============
Detects and resolves spatial / directional relationship queries such as
"Hà Nội nằm ở đâu so với TP.HCM?", "What is north of Paris?",
"Which city is east of Berlin?".

Ported 1:1 from engine_v1.py's ``SpatialSolver`` class (lines ~1459-1530).
The original class was present in ``ThinkingEngineV1`` and called inside
``process()`` to populate ``state.spatial_note`` — this field existed in
``ThinkingState`` but was NEVER SET in the ported codebase (Bug #1).

Architecture
------------
This solver is intentionally lightweight: it does pure pattern-matching
(no geocoding API, no lat-lon arithmetic) because the engine is designed
to run fully offline. When a spatial query is detected, it returns a
structured description string that the PromptCompiler will inject into the
system prompt so the LLM can produce a geographically accurate answer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ── Cardinal / ordinal direction vocabulary (Vi + En) ──────────────────────
_DIRECTIONS: Dict[str, str] = {
    # Vietnamese
    "bắc": "north", "phía bắc": "north", "hướng bắc": "north",
    "nam": "south", "phía nam": "south", "hướng nam": "south",
    "đông": "east", "phía đông": "east", "hướng đông": "east",
    "tây": "west", "phía tây": "west", "hướng tây": "west",
    "đông bắc": "northeast", "tây bắc": "northwest",
    "đông nam": "southeast", "tây nam": "southwest",
    # English
    "north": "north", "northern": "north",
    "south": "south", "southern": "south",
    "east": "east", "eastern": "east",
    "west": "west", "western": "west",
    "northeast": "northeast", "northwest": "northwest",
    "southeast": "southeast", "southwest": "southwest",
    "above": "north", "below": "south",
    "left": "west", "right": "east",
}

# ── Spatial trigger keywords ────────────────────────────────────────────────
_SPATIAL_TRIGGERS: List[str] = [
    # Vietnamese
    "nằm ở đâu", "ở phía", "hướng nào", "nằm về phía", "cách bao xa",
    "bao xa", "khoảng cách", "ở đâu so với", "nằm ở", "địa điểm",
    # English
    "where is", "which direction", "how far", "distance from", "to the north",
    "to the south", "to the east", "to the west", "located", "position of",
    "relative to", "compared to", "north of", "south of", "east of", "west of",
]

# ── Opposites (for constraint checking) ────────────────────────────────────
_OPPOSITES: Dict[str, str] = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
}


class SpatialSolver:
    """Detects spatial / directional queries and generates a structured
    spatial context note for the PromptCompiler."""

    def is_applicable(self, text: str) -> bool:
        """Returns True if *text* contains a detectable spatial query."""
        tl = text.lower()
        return any(trigger in tl for trigger in _SPATIAL_TRIGGERS)

    def detect_directions(self, text: str) -> List[str]:
        """Returns a list of canonical direction names found in *text*."""
        tl = text.lower()
        found: List[str] = []
        # Longest-match first to catch "đông bắc" before "đông"
        for phrase in sorted(_DIRECTIONS.keys(), key=len, reverse=True):
            if phrase in tl:
                canonical = _DIRECTIONS[phrase]
                if canonical not in found:
                    found.append(canonical)
        return found

    def extract_subjects(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts the two spatial subjects (e.g., cities, regions) from
        a query using a simple proper-noun heuristic.  Returns (subject_a,
        subject_b); either may be None if only one subject is found.
        """
        # Named entities: capitalised runs (works for both EN and VI proper nouns)
        tokens = re.findall(
            r"\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯẠẬẶỆỘỢỤ]"
            r"[A-Za-zàáâãèéêìíòóôõùúăđơưạậặệộợụ0-9.]+\b",
            text,
        )
        # Remove single-char tokens and deduplicate preserving order
        seen: Dict[str, None] = {}
        for t in tokens:
            if len(t) > 1:
                seen[t] = None
        subjects = list(seen.keys())
        a = subjects[0] if len(subjects) >= 1 else None
        b = subjects[1] if len(subjects) >= 2 else None
        return a, b

    def solve(self, text: str) -> str:
        """Main entry point.  Returns a spatial context note string to be
        stored in ``ThinkingState.spatial_note``, or an empty string if the
        query is not spatial.

        The note is intentionally SHORT (1-2 sentences) so it fits inside
        the PromptCompiler's available token budget without crowding out more
        important context.
        """
        if not self.is_applicable(text):
            return ""

        directions = self.detect_directions(text)
        subject_a, subject_b = self.extract_subjects(text)

        parts: List[str] = []

        if subject_a and subject_b and directions:
            dir_str = " / ".join(directions)
            parts.append(
                f"[SPATIAL] Query involves directional relationship: "
                f"{subject_a} → {dir_str} → {subject_b}."
            )
            # Warn the LLM if the query contains contradictory directions
            for d in directions:
                opp = _OPPOSITES.get(d)
                if opp and opp in directions:
                    parts.append(
                        f"[SPATIAL-WARN] Conflicting directions detected "
                        f"({d} and {opp}) — verify geographical claim carefully."
                    )
        elif subject_a and directions:
            dir_str = " / ".join(directions)
            parts.append(
                f"[SPATIAL] Query asks for {dir_str} neighbour / position of {subject_a}."
            )
        elif directions:
            dir_str = " / ".join(directions)
            parts.append(f"[SPATIAL] Directional cue detected: {dir_str}.")
        else:
            parts.append("[SPATIAL] Spatial query detected (no specific direction extracted).")

        return " ".join(parts)
