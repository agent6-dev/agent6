# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The module-style import rule's list is the tree.

`banned-from` names every leaf module of agent6, so `from agent6.pkg.module
import name` is refused everywhere and `from agent6.pkg import module` is the
way; a module added without its entry would slip through.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


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


def test_every_leaf_module_is_listed_and_nothing_else_of_agent6() -> None:
    listed = {m for m in _banned_from() if m.startswith("agent6.")}
    assert listed == _leaf_modules()


def test_the_list_is_sorted_and_keeps_the_typing_exceptions() -> None:
    banned = _banned_from()
    assert banned == sorted(set(banned))
    assert not {"typing", "collections.abc", "__future__"} & set(banned)
