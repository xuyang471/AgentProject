from __future__ import annotations

import ast
import math
import re


ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}

ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def normalize_expression(expression: str) -> str:
    normalized = str(expression or "").strip()
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("×", "*")
    normalized = re.sub(r"(?<=[\d\)])\s*[xX]\s*(?=[\d\(])", "*", normalized)
    normalized = normalized.replace("÷", "/")
    normalized = normalized.replace("，", ",")
    normalized = normalized.replace("％", "%")
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized)
    normalized = re.sub(r"[^0-9A-Za-z_\.\,\+\-\*\/\(\)%\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数值常量。")

    if isinstance(node, ast.Num):
        return node.n

    if isinstance(node, ast.BinOp):
        operator = ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("表达式中包含不支持的运算符。")
        return operator(_evaluate_node(node.left), _evaluate_node(node.right))

    if isinstance(node, ast.UnaryOp):
        operator = ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("表达式中包含不支持的一元运算。")
        return operator(_evaluate_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只允许调用受限数学函数。")
        func = ALLOWED_FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"不支持函数：{node.func.id}")
        args = [_evaluate_node(arg) for arg in node.args]
        return func(*args)

    raise ValueError("表达式结构不受支持。")


def calculate_expression(expression: str):
    normalized = normalize_expression(expression)
    if not normalized:
        raise ValueError("计算表达式不能为空。")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("表达式格式不合法。") from exc

    result = _evaluate_node(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result
