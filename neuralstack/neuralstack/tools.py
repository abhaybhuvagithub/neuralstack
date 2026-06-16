"""tools.py — local tools the agent can call.

Each tool has (1) a JSON schema Claude sees, and (2) a Python function that
actually runs when Claude requests it. `search_knowledge_base` is the RAG
entry point: it is wired to the vector store in agent.py.

This is also where you would register additional local capabilities the
agent should have (database lookups, internal APIs, etc.).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List


# --- tool JSON schemas (the "menu" Claude is shown) ----------------------

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the indexed document knowledge base for passages relevant to "
        "a natural-language query. Use this whenever the user asks something "
        "that might be answered by the ingested documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
            "k": {"type": "integer", "description": "How many passages to return (default 3)."},
        },
        "required": ["query"],
    },
}

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression, e.g. '1200 * 0.12'.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [SEARCH_TOOL, CALCULATOR_TOOL]


# --- safe calculator implementation --------------------------------------

import ast
import operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an arithmetic AST without using Python's eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:  # noqa: BLE001 - report any failure to the model
        return f"error: {exc}"


# Map of tool name -> callable. search_knowledge_base is injected at runtime
# in agent.py because it needs the live vector store.
LOCAL_TOOL_FNS: Dict[str, Callable[..., str]] = {
    "calculator": lambda expression: calculator(expression),
}
