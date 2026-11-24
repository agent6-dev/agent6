# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The app layer writes through the injected `Reporter`, read from the tree."""

from __future__ import annotations

import ast
import pathlib

_APP = pathlib.Path(__file__).resolve().parents[2] / "src" / "agent6" / "app"


def test_no_app_module_prints_directly() -> None:
    """A `print` in `app/` outside `reporter.py` bypasses the front-end's two channels."""
    direct = [
        f"{path.relative_to(_APP)}:{node.lineno}"
        for path in sorted(_APP.rglob("*.py"))
        if path.name != "reporter.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "print"
    ]
    assert not direct, direct
