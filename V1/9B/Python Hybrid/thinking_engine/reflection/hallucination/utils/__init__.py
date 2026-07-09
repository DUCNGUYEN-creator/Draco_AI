# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.utils
================================================
Hallucination-specific utility re-exports. Most actual math/probability/
graph utilities live in thinking_engine.utils (shared engine-wide);
this sub-package re-exports only the subset specifically used inside
reflection.hallucination to give a clean local import path for code
that stays strictly within this package boundary.
"""

from ....utils.math import clamp, entropy, mean, sigmoid, softmax
from ....utils.probability import bayes_update, noisy_or
from ....utils.hashing import short_hash, stable_hash
from ....utils.timer import Timer, now_ts
from ....utils.graph import UnionFind
from ....utils.serialization import from_json, to_json
from ....utils.validator import require_in_range, require_non_empty, require_one_of

__all__ = [
    "clamp", "entropy", "mean", "sigmoid", "softmax",
    "bayes_update", "noisy_or",
    "short_hash", "stable_hash",
    "Timer", "now_ts",
    "UnionFind",
    "from_json", "to_json",
    "require_in_range", "require_non_empty", "require_one_of",
]
