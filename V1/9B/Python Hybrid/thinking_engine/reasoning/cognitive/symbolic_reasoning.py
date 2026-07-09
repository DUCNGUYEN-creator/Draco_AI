# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SymbolicReasoner
===================
New addition. Thin, safe wrapper around SafeASTEvaluator-style symbolic
manipulation for simple propositional logic ("A and B", "not A or B")
— evaluates truth tables for small numbers of boolean variables so
formal-logic questions can be checked exactly rather than only via
keyword heuristics (ChainOfThoughtVerifier's contradiction-pair regexes).
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"\bnot\b|\band\b|\bor\b|\(|\)|[A-Za-z_][A-Za-z0-9_]*")

# Trailing natural-language predicate phrases ("is always true", "luôn đúng",
# "is a tautology", ...) that may follow the actual boolean expression in a
# claim like "A or not A is always true". Stripped before parsing so only the
# pure propositional-logic portion is ever passed to eval().
_PREDICATE_SUFFIX_RE = re.compile(
    r"\s+(?:is\s+(?:always|never)?\s*(?:true|false)|"
    r"luôn\s+(?:luôn\s+)?(?:đúng|sai)|"
    r"is\s+a\s+tautology|is\s+a\s+contradiction).*$",
    re.IGNORECASE,
)

_LOGIC_KEYWORDS = {"not", "and", "or"}

# English filler/reserved words that must never be treated as a boolean
# variable name. Compared case-insensitively, EXCEPT that a single
# uppercase letter (A, B, C, ...) is always treated as a genuine logic
# variable even if its lowercase form collides with a filler word (e.g.
# "A" the variable vs "a" the indefinite article) — multi-letter
# lowercase fillers like "is"/"always"/"true" are unambiguous and always
# filtered.
_RESERVED_FILLER_WORDS = {
    "is", "an", "the", "always", "never", "true", "false",
    "tautology", "contradiction", "luôn", "đúng", "sai",
}
_SINGLE_LETTER_FILLER = {"a"}  # only ambiguous with the variable name "A"


def _is_reserved_filler(token: str) -> bool:
    tl = token.lower()
    if tl in _LOGIC_KEYWORDS:
        return True
    if len(token) == 1 and token.isalpha():
        # Single-letter tokens are always treated as logic variables
        # (A, B, x, ...), never as the article "a" — case is irrelevant
        # here precisely because we want to KEEP single letters.
        return False
    return tl in _RESERVED_FILLER_WORDS


class SymbolicReasoner:
    @staticmethod
    def _strip_predicate_suffix(expr: str) -> str:
        return _PREDICATE_SUFFIX_RE.sub("", expr).strip()

    def extract_variables(self, expr: str) -> List[str]:
        """Tokenizes on the ORIGINAL (not lower-cased) text so that
        single-letter variable names like 'A' are never confused with
        the lower-case article 'a' — only the final variable list is
        normalized for downstream substitution."""
        core = self._strip_predicate_suffix(expr)
        tokens = _TOKEN_RE.findall(core)
        seen: List[str] = []
        seen_lower: set = set()
        for t in tokens:
            if t in ("(", ")"):
                continue
            if _is_reserved_filler(t):
                continue
            key = t.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                seen.append(key)
        return seen

    def evaluate(self, expr: str, assignment: Dict[str, bool]) -> bool:
        """Evaluates a propositional expression using Python's own boolean
        operators after substituting variable names with True/False
        literals — restricted to and/or/not/parentheses/identifiers only,
        so this never executes arbitrary code (same safety posture as
        tools/SafeASTEvaluator). Any trailing natural-language predicate
        phrase (e.g. "is always true") is stripped first so only the pure
        boolean expression is ever evaluated.
        """
        core = self._strip_predicate_suffix(expr)
        safe_expr = core.lower()
        for name in sorted(assignment, key=len, reverse=True):
            safe_expr = re.sub(rf"\b{re.escape(name.lower())}\b", str(assignment[name]), safe_expr)
        if not re.fullmatch(r"[\sA-Za-z()_]+", safe_expr.replace("True", "").replace("False", "")):
            raise ValueError("Unsupported tokens in expression")
        return bool(eval(safe_expr, {"__builtins__": {}}, {}))  # noqa: S307 — input is regex-validated

    def truth_table(self, expr: str) -> List[Tuple[Dict[str, bool], bool]]:
        variables = self.extract_variables(expr)
        rows = []
        for combo in itertools.product([False, True], repeat=len(variables)):
            assignment = dict(zip(variables, combo))
            rows.append((assignment, self.evaluate(expr, assignment)))
        return rows

    def is_tautology(self, expr: str) -> bool:
        return all(result for _, result in self.truth_table(expr))

    def is_contradiction(self, expr: str) -> bool:
        return all(not result for _, result in self.truth_table(expr))
