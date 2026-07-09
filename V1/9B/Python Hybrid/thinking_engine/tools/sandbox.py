# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SafeASTEvaluator
===================
Evaluates simple math expressions using Python's AST — no eval() risk,
no __builtins__ bypass, no DoS via large exponents. Ported 1:1 from
engine_v1.py's ``SafeASTEvaluator``.

Supported: +, -, *, /, **, %, //, unary minus, integers, floats.
Max expression length: 200 chars. Max AST node count: 64.
Max exponent value: 1000 (guards against 999999**999999 CPU hang).
"""

from __future__ import annotations

import ast
from typing import Optional


class SafeASTEvaluator:
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.FloorDiv, ast.USub, ast.UAdd,
    )
    _MAX_LEN = 200
    _MAX_NODES = 64
    _MAX_EXPONENT = 1000  # DoS guard: 999999**999999 would hang CPU indefinitely

    def _count_nodes(self, node: ast.AST) -> int:
        return 1 + sum(self._count_nodes(c) for c in ast.iter_child_nodes(node))

    def _check_pow_safety(self, tree: ast.AST) -> Optional[str]:
        """Walk AST and reject any Pow whose exponent constant exceeds _MAX_EXPONENT."""
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exp_node = node.right
                if isinstance(exp_node, ast.UnaryOp) and isinstance(exp_node.op, ast.USub):
                    exp_node = exp_node.operand
                if isinstance(exp_node, ast.Constant) and isinstance(exp_node.value, (int, float)):
                    if abs(exp_node.value) > self._MAX_EXPONENT:
                        return (
                            f"Error: exponent {exp_node.value} exceeds max allowed "
                            f"({self._MAX_EXPONENT}) — operation refused (DoS guard)"
                        )
        return None

    def evaluate(self, expr: str) -> str:
        """Returns a string result or an error message."""
        expr = expr.strip()
        if len(expr) > self._MAX_LEN:
            return f"Error: expression too long (max {self._MAX_LEN} chars)"
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"Syntax error: {e}"
        if self._count_nodes(tree) > self._MAX_NODES:
            return f"Error: expression too complex (max {self._MAX_NODES} AST nodes)"
        for node in ast.walk(tree):
            if not isinstance(node, self._ALLOWED_NODES):
                return f"Error: unsupported operation ({type(node).__name__})"
        pow_err = self._check_pow_safety(tree)
        if pow_err:
            return pow_err
        try:
            result = eval(  # noqa: S307 — AST-validated, no builtins, expression-only
                compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}
            )
            return str(result)
        except ZeroDivisionError:
            return "Error: division by zero"
        except OverflowError:
            return "Error: result too large (overflow)"
        except Exception as e:
            return f"Error: {e}"
