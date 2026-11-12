# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The module-style import rule, read from the tree.

`from agent6.pkg.module import name` is refused everywhere: a leaf module's members are reached
as `module.name`. A package's `__init__.py` re-exports names from its modules with a trailing
`# re-export`; `pyproject.toml`'s `banned-from` carries the same rule for external modules.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _banned_from() -> list[str]:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["tool"]["ruff"]["lint"]["flake8-import-conventions"]["banned-from"]


def _leaf_modules() -> set[str]:
    src = _ROOT / "src" / "agent6"
    return {
        "agent6." + str(p.relative_to(src)).replace("/", ".")[:-3]
        for p in src.rglob("*.py")
        if p.name != "__init__.py" and "/jail/" not in str(p)
    }


def test_no_name_is_imported_from_a_leaf_module() -> None:
    """A member import from an agent6 leaf module is refused unless the line says `# re-export`.

    The 392-entry `banned-from` list that once pinned this went stale with every module added
    or moved; the tree is the list.
    """
    leaves = _leaf_modules()
    violations: list[str] = []
    for root in ("src", "tests"):
        for path in sorted((_ROOT / root).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                if node.module in leaves and "# re-export" not in lines[node.lineno - 1]:
                    names = ", ".join(alias.name for alias in node.names)
                    where = f"{path.relative_to(_ROOT)}:{node.lineno}"
                    violations.append(f"{where}: from {node.module} import {names}")
    assert not violations, "\n".join(violations)


def test_the_external_list_holds_no_agent6_module() -> None:
    """The tree rule above owns agent6's modules; the list names only external ones."""
    assert not [m for m in _banned_from() if m.startswith("agent6")]


def test_the_list_is_sorted_and_keeps_the_typing_exceptions() -> None:
    banned = _banned_from()
    assert banned == sorted(set(banned))
    assert not {"typing", "collections.abc", "__future__"} & set(banned)


def test_standard_library_imports_are_not_aliased() -> None:
    """Standard-library imports keep their module names in source and tests."""
    violations: list[str] = []
    for root in ("src", "tests"):
        for path in sorted((_ROOT / root).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for alias in node.names:
                    module = node.module if isinstance(node, ast.ImportFrom) else alias.name
                    if alias.asname and module and module.split(".")[0] in sys.stdlib_module_names:
                        violations.append(
                            f"{path.relative_to(_ROOT)}:{node.lineno}: "
                            f"{module} imports {alias.name} as {alias.asname}"
                        )
    assert not violations, "\n".join(violations)
