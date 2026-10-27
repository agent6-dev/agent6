# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every multi-line src docstring has the Google sections its function needs.

ruff requires none of them: no rule asks for a missing `Args` section (D417 checks only one
that is present), and DOC201 and DOC402 accept a summary starting with "Return" or "Yield" as
the section. A one-line docstring is exempt: its summary says it all.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib
from collections.abc import Iterator

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "agent6"
# A property reads as an attribute; an abstract method is a stub its overrides document.
_EXEMPT_DECORATORS = frozenset({"abc.abstractmethod", "functools.cached_property", "property"})
_SECTION = re.compile(r"^(Args|Returns|Yields):$", re.MULTILINE)
_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _functions(parent: ast.AST) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, bool]]:
    for node in ast.iter_child_nodes(parent):
        if isinstance(node, _FUNCTION_TYPES):
            yield node, isinstance(parent, ast.ClassDef)
        yield from _functions(node)


def _own_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Return the function's body nodes, without those of nested functions and classes."""
    nodes: list[ast.AST] = []
    stack: list[ast.AST] = [
        n for n in fn.body if not isinstance(n, (*_FUNCTION_TYPES, ast.ClassDef))
    ]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, (*_FUNCTION_TYPES, ast.Lambda, ast.ClassDef))
        )
    return nodes


def _has_value(node: ast.Return | ast.Yield) -> bool:
    value = node.value
    return value is not None and not (isinstance(value, ast.Constant) and value.value is None)


def _needed(fn: ast.FunctionDef | ast.AsyncFunctionDef, method: bool) -> set[str]:
    """Return the sections the signature and body call for."""
    args = fn.args
    params = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if method and params and "staticmethod" not in {ast.unparse(d) for d in fn.decorator_list}:
        params = params[1:]
    params += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
    nodes = _own_nodes(fn)
    generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in nodes)
    returns_none = fn.returns is not None and ast.unparse(fn.returns) in {"None", "NoReturn"}
    needed: set[str] = set()
    # As in D417, a parameter with a leading underscore needs no entry.
    if any(not p.startswith("_") for p in params):
        needed.add("Args")
    if not (generator or returns_none) and any(
        isinstance(n, ast.Return) and _has_value(n) for n in nodes
    ):
        needed.add("Returns")
    if any(
        isinstance(n, ast.YieldFrom) or (isinstance(n, ast.Yield) and _has_value(n)) for n in nodes
    ):
        needed.add("Yields")
    return needed


def test_a_multi_line_docstring_has_the_sections_its_function_needs() -> None:
    missing: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        for fn, method in _functions(ast.parse(path.read_text(encoding="utf-8"))):
            doc = ast.get_docstring(fn)
            if doc is None or "\n" not in doc.strip():
                continue
            if _EXEMPT_DECORATORS & {ast.unparse(d) for d in fn.decorator_list}:
                continue
            absent = _needed(fn, method) - set(_SECTION.findall(doc))
            if absent:
                where = path.relative_to(_SRC.parents[1])
                missing.append(f"{where}:{fn.lineno} {fn.name}: {', '.join(sorted(absent))}")
    assert not missing, f"{len(missing)} docstrings miss a section:\n" + "\n".join(missing)


def test_ruff_requires_imperative_summaries() -> None:
    """The Google convention disables D401 unless its exact code is selected."""
    config = tomllib.loads((_SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    lint = config["tool"]["ruff"]["lint"]
    assert "D401" in lint["select"]
    assert "D401" not in lint["ignore"]


def _parsed(source: str) -> ast.FunctionDef:
    fn = ast.parse(source).body[0]
    assert isinstance(fn, ast.FunctionDef)
    return fn


def test_a_def_in_a_function_body_keeps_its_returns_and_yields_to_itself() -> None:
    """A def directly in a function's body leaked its `return` and `yield` into the outer one."""
    assert _needed(_parsed("def f():\n    def g():\n        return 1\n    g()\n"), False) == set()
    gen = _parsed("def f():\n    def g():\n        yield 1\n    return g\n")
    assert _needed(gen, False) == {"Returns"}
