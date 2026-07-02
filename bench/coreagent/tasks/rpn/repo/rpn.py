"""rpn — a reverse-Polish-notation calculator. See spec.md.

Implement each component. Stub bodies raise NotImplementedError so the test
suite fails until you do.
"""

from __future__ import annotations


def tokenize(expr: str) -> list[str]:
    raise NotImplementedError


def evaluate(tokens: list[str]) -> float:
    raise NotImplementedError


def evaluate_expr(expr: str) -> float:
    raise NotImplementedError


class RPNCalculator:
    def __init__(self) -> None:
        raise NotImplementedError

    def push(self, expr: str) -> float:
        raise NotImplementedError

    def last(self) -> float | None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
