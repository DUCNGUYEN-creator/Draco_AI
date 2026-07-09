# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SelfDebugger
==============
New addition. For INTENT_CODE reasoning specifically: takes a code
snippet + the SafeASTEvaluator-style execution result (or an
externally-run test result) and produces a structured "what went
wrong" note the reasoning loop can feed back into the next generation
attempt — the code-specific analogue of RecursiveReflectionLoop.
"""

from __future__ import annotations

import re
from typing import List, Optional


class SelfDebugger:
    _ERROR_PATTERNS = [
        (r"SyntaxError", "Syntax error — check brackets, indentation, colons."),
        (r"NameError", "Undefined name — check variable/function spelling and scope."),
        (r"TypeError", "Type mismatch — check argument types and conversions."),
        (r"IndexError", "Index out of range — check loop bounds / list length."),
        (r"KeyError", "Missing dict key — check key existence before access."),
        (r"ZeroDivisionError", "Division by zero — add a guard before dividing."),
        (r"RecursionError", "Infinite/too-deep recursion — check base case."),
    ]

    def diagnose(self, error_text: str) -> str:
        for pattern, hint in self._ERROR_PATTERNS:
            if re.search(pattern, error_text):
                return f"[SELF-DEBUG] {pattern}: {hint}"
        return f"[SELF-DEBUG] Unrecognized error — raw: {error_text[:200]}"

    def build_fix_prompt(self, code: str, error_text: str) -> str:
        diagnosis = self.diagnose(error_text)
        return (
            f"[CODE]\n{code}\n\n[ERROR]\n{error_text[:300]}\n\n"
            f"[DIAGNOSIS]\n{diagnosis}\n\n[TASK] Fix the code above.\n[FIXED CODE]"
        )
