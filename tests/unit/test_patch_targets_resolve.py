# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every patch target in the suite resolves the way monkeypatch resolves it.

A module reads the names it uses at call time, so a test that patches a name on the module
that merely imported it patches nothing; pyright cannot see a string target, so this pin walks
every `monkeypatch.setattr`, `mock.patch.object` and `mock.patch("dotted.name")` in the tests.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import types

_TESTS = pathlib.Path(__file__).resolve().parents[1]


def _resolve(dotted: str) -> object | None:
    """Import the longest module prefix, then walk the rest by attribute."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def _targets(tree: ast.Module) -> list[tuple[int, list[str], str, ast.expr | None]]:
    """The (line, modules, name, value) of every patch call, string targets included.

    A name bound by several imports in one file (one per function) lists every module.
    """
    aliases: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                aliases.setdefault(a.asname or a.name, []).append(f"{node.module}.{a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                aliases.setdefault(a.asname or a.name.split(".")[0], []).append(a.name)
    out: list[tuple[int, list[str], str, ast.expr | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0] if node.args else None
        second = node.args[1] if len(node.args) > 1 else None
        value = node.args[2] if len(node.args) > 2 else None
        if (
            fname in ("setattr", "object")
            and isinstance(first, ast.Name)
            and isinstance(second, ast.Constant)
            and isinstance(second.value, str)
            and first.id in aliases
        ):
            out.append((node.lineno, aliases[first.id], second.value, value))
        elif (
            fname in ("patch", "setattr")
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and "." in first.value
        ):
            module, _, name = first.value.rpartition(".")
            out.append((node.lineno, [module], name, second))
    return out


def _own_module(obj: object) -> bool:
    """An agent6 or tests module: a stdlib module patched with a fake is a stand-in by design."""
    return isinstance(obj, types.ModuleType) and obj.__name__.startswith(("agent6.", "tests."))


def _stand_in(value: ast.expr | None, tree: ast.Module) -> bool:
    """A `types.SimpleNamespace(...)` value, or a name bound to one, stands in for a module."""
    if isinstance(value, ast.Call) and getattr(value.func, "attr", "") == "SimpleNamespace":
        return True
    if isinstance(value, ast.Name):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == value.id for t in node.targets)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "SimpleNamespace"
            ):
                return True
    return False


def test_every_patch_target_resolves() -> None:
    """A patch names an attribute the target module binds, and never replaces a module binding."""
    missing: list[str] = []
    for path in sorted(_TESTS.rglob("*.py")):
        if path == pathlib.Path(__file__):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, modules, name, value in _targets(tree):
            targets = [
                t
                for m in modules
                if m.startswith(("agent6.", "tests."))
                if isinstance(t := _resolve(m), types.ModuleType)
            ]
            if not targets:
                continue  # a class or object attribute, checked by pyright
            names = "|".join(dict.fromkeys(modules))
            where = f"{path.relative_to(_TESTS.parent)}:{line}: {names}.{name}"
            if not any(hasattr(t, name) for t in targets):
                missing.append(f"{where} does not exist")
            elif all(_own_module(getattr(t, name, None)) for t in targets) and not _stand_in(
                value, tree
            ):
                missing.append(f"{where} is a module; patch the name inside it")
    assert not missing, "\n".join(missing)
