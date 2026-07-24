"""
tools/calculator.py
───────────────────
A safe math evaluation tool — demonstrates tool use beyond search.

Concepts from document:
  - Tool categories: Code execution tools (safe sandboxed eval)
  - Restricting tool input: we whitelist allowed characters for safety
  - Tool error handling: returns structured error string, never raises
"""

from __future__ import annotations

import ast
import math
import operator as op


# Whitelisted operations — safe subset of Python math
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def safe_calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: A math expression string, e.g. "sqrt(2) * pi" or "2**10"

    Returns:
        The result as a string, or an error message.

    Security: Uses AST parsing — no exec() or eval() with untrusted globals.
    """
    try:
        expression = expression.strip()
        # Replace known constant names
        for name, val in [("pi", str(math.pi)), ("e", str(math.e))]:
            expression = expression.replace(name, str(val))

        # Replace function calls with safe versions
        for fname in ALLOWED_FUNCTIONS:
            if fname + "(" in expression:
                expression = expression.replace(
                    fname + "(", f"__fn_{fname}("
                )

        # Build a safe namespace
        safe_globals = {f"__fn_{k}": v for k, v in ALLOWED_FUNCTIONS.items()}

        result = _eval_ast(ast.parse(expression, mode="eval").body, safe_globals)
        return f"Result: {result}"

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Math error: {e}"
    except Exception as e:
        return f"Could not compute '{expression}': {e}"


def _eval_ast(node, safe_globals: dict):
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        left = _eval_ast(node.left, safe_globals)
        right = _eval_ast(node.right, safe_globals)
        return ALLOWED_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand, safe_globals)
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name and func_name in safe_globals:
            args = [_eval_ast(a, safe_globals) for a in node.args]
            return safe_globals[func_name](*args)
        raise ValueError(f"Function '{func_name}' not allowed")
    elif isinstance(node, ast.Name):
        if node.id in safe_globals:
            return safe_globals[node.id]
        raise ValueError(f"Name '{node.id}' not allowed")
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")
