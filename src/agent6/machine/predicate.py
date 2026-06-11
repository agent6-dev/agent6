# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Restricted, non-Turing-complete predicate language for `branch` states.

A predicate is parsed with :func:`ast.parse` in ``mode="eval"`` and then
walked against a strict allow-list of node types. Anything outside the
allow-list, function calls beyond a tiny fixed set, Python attribute
access, comprehensions, lambdas, arithmetic, is rejected at
``machine check`` time. The evaluator never ``eval``/``exec``s, never
calls ``getattr``, and never resolves arbitrary Python names: an
``Attribute`` chain is reinterpreted as *data* navigation into a record
value (an ordered dict lookup), and a bare ``Name`` must be a declared
blackboard variable.

This module is intentionally dependency-free (stdlib ``ast`` only) so the
security-critical allow-list can be audited in isolation.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "ALLOWED_FUNCTIONS",
    "Predicate",
    "PredicateError",
    "Reference",
    "evaluate",
    "parse_predicate",
]

# The only callable names a predicate may invoke. Each is a fixed-arity,
# pure builtin re-implemented by the evaluator; we never call the Python
# builtin via the name.
ALLOWED_FUNCTIONS = frozenset({"len"})

_ALLOWED_COMPARISONS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)


class PredicateError(Exception):
    """Raised when a predicate is not a valid, allow-listed expression."""


@dataclass(frozen=True, slots=True)
class Reference:
    """A blackboard reference: a root variable plus zero or more record
    field navigations. ``verdict.confidence`` is ``Reference("verdict",
    ("confidence",))``."""

    root: str
    path: tuple[str, ...]

    @property
    def dotted(self) -> str:
        return ".".join((self.root, *self.path))


@dataclass(frozen=True, slots=True)
class Predicate:
    """A parsed, allow-list-validated predicate.

    ``references`` is every blackboard reference the predicate reads, in
    source order, so a caller can type-check each against the declared
    variables and record schemas.
    """

    source: str
    tree: ast.Expression
    references: tuple[Reference, ...]


def parse_predicate(source: str) -> Predicate:
    """Parse and allow-list-validate *source*.

    Raises :class:`PredicateError` on a syntax error or any node outside
    the allow-list. Does not type-check references, that is the caller's
    job, since types live in the machine model.
    """
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise PredicateError(f"not a valid expression: {source!r} ({exc.msg})") from exc
    references: list[Reference] = []
    _check(tree.body, references)
    return Predicate(source=source, tree=tree, references=tuple(references))


def _check(node: ast.expr, references: list[Reference]) -> None:  # noqa: PLR0911, PLR0912
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise PredicateError(f"unsupported boolean operator: {type(node.op).__name__}")
        for value in node.values:
            _check(value, references)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise PredicateError(f"unsupported unary operator: {type(node.op).__name__}")
        _check(node.operand, references)
        return
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(op, _ALLOWED_COMPARISONS):
                raise PredicateError(f"unsupported comparison: {type(op).__name__}")
        _check(node.left, references)
        for comparator in node.comparators:
            _check(comparator, references)
        return
    if isinstance(node, ast.Call):
        _check_call(node, references)
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            if not isinstance(element, ast.Constant):
                raise PredicateError("list/tuple literals may contain only constants")
            _check(element, references)
        return
    if isinstance(node, ast.Constant):
        if node.value is not None and not isinstance(node.value, (str, int, float, bool)):
            raise PredicateError(f"unsupported literal: {node.value!r}")
        return
    if isinstance(node, (ast.Name, ast.Attribute)):
        references.append(_as_reference(node))
        return
    raise PredicateError(f"unsupported syntax: {type(node).__name__}")


def _check_call(node: ast.Call, references: list[Reference]) -> None:
    func = node.func
    if not isinstance(func, ast.Name) or func.id not in ALLOWED_FUNCTIONS:
        name = func.id if isinstance(func, ast.Name) else type(func).__name__
        raise PredicateError(f"calls are restricted to {sorted(ALLOWED_FUNCTIONS)}; got {name!r}")
    if node.keywords:
        raise PredicateError(f"{func.id}() takes no keyword arguments")
    if len(node.args) != 1:
        raise PredicateError(f"{func.id}() takes exactly one argument")
    arg = node.args[0]
    if isinstance(arg, ast.Starred):
        raise PredicateError(f"{func.id}() does not accept starred arguments")
    _check(arg, references)


def _as_reference(node: ast.expr) -> Reference:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        if not isinstance(current.ctx, ast.Load):
            raise PredicateError("attribute assignment is not allowed")
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise PredicateError(f"unsupported reference: {type(current).__name__}")
    if not isinstance(current.ctx, ast.Load):
        raise PredicateError("name assignment is not allowed")
    parts.append(current.id)
    parts.reverse()
    return Reference(root=parts[0], path=tuple(parts[1:]))


def evaluate(predicate: Predicate, blackboard: Mapping[str, object]) -> bool:
    """Evaluate *predicate* against *blackboard*, returning a bool.

    Pure: navigates record values as ordered dict lookups, never touching
    the host environment. Raises :class:`PredicateError` if a reference
    cannot be resolved against the blackboard at runtime.
    """
    return bool(_eval(predicate.tree.body, blackboard))


def _eval(node: ast.expr, blackboard: Mapping[str, object]) -> object:  # noqa: PLR0911, PLR0912
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: object = True
            for value in node.values:
                result = _eval(value, blackboard)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = _eval(value, blackboard)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, blackboard)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -_as_number(operand)
        return +_as_number(operand)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, blackboard)
    if isinstance(node, ast.Call):
        # Allow-list guarantees this is len() with exactly one argument.
        value = _eval(node.args[0], blackboard)
        if not isinstance(value, (str, list, tuple, dict, bytes)):
            raise PredicateError(f"len() argument has no length: {value!r}")
        return len(value)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(element, blackboard) for element in node.elts]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _resolve(_as_reference(node), blackboard)
    raise PredicateError(f"unsupported syntax: {type(node).__name__}")


def _eval_compare(node: ast.Compare, blackboard: Mapping[str, object]) -> bool:
    left = _eval(node.left, blackboard)
    for op, comparator_node in zip(node.ops, node.comparators, strict=True):
        right = _eval(comparator_node, blackboard)
        if not _compare(op, left, right):
            return False
        left = right
    return True


def _compare(op: ast.cmpop, left: object, right: object) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.In):
        return _contains(right, left)
    if isinstance(op, ast.NotIn):
        return not _contains(right, left)
    return _order(op, left, right)


def _order(op: ast.cmpop, left: object, right: object) -> bool:
    lhs = _as_number(left) if _is_number(left) and _is_number(right) else left
    rhs = _as_number(right) if _is_number(left) and _is_number(right) else right
    try:
        if isinstance(op, ast.Lt):
            return lhs < rhs  # type: ignore[operator]
        if isinstance(op, ast.LtE):
            return lhs <= rhs  # type: ignore[operator]
        if isinstance(op, ast.Gt):
            return lhs > rhs  # type: ignore[operator]
        return lhs >= rhs  # type: ignore[operator]
    except TypeError as exc:
        raise PredicateError(f"cannot order {left!r} and {right!r}") from exc


def _contains(container: object, item: object) -> bool:
    if isinstance(container, str):
        if not isinstance(item, str):
            raise PredicateError(f"`in` on a string requires a string, got {item!r}")
        return item in container
    if not isinstance(container, (list, tuple, dict)):
        raise PredicateError(f"value is not a container for `in`: {container!r}")
    return item in container


def _resolve(reference: Reference, blackboard: Mapping[str, object]) -> object:
    if reference.root not in blackboard:
        raise PredicateError(f"unknown reference: {reference.root!r}")
    value = blackboard[reference.root]
    for key in reference.path:
        if not isinstance(value, Mapping):
            raise PredicateError(f"cannot navigate into non-record value at {reference.dotted!r}")
        if key not in value:
            raise PredicateError(f"missing field {key!r} in {reference.dotted!r}")
        value = value[key]
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredicateError(f"expected a number, got {value!r}")
    return float(value)
