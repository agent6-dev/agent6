# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every `python -m agent6...` host spawn passes `-P`.

`python -m` prepends the current directory to `sys.path`, so a model-planted
top-level `agent6/` package in the workspace would shadow the installed one and
execute on the host, outside the jail, on `import agent6`. `-P` (safe path)
keeps cwd off `sys.path`. This pins the invariant across the tree, so a new
spawn site that forgets it fails here rather than in the field.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent6


def _is_sys_executable(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _python_m_agent6_spawns() -> list[tuple[str, int, list[str]]]:
    """(file, lineno, string args) for every argv list literal that runs
    `sys.executable ... -m agent6...`."""
    src = Path(agent6.__file__).resolve().parent
    found: list[tuple[str, int, list[str]]] = []
    for py in src.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.List) or not node.elts:
                continue
            if not _is_sys_executable(node.elts[0]):
                continue
            strs = [
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if "-m" in strs and any(s.startswith("agent6") for s in strs[strs.index("-m") + 1 :]):
                found.append((py.relative_to(src).as_posix(), node.lineno, strs))
    return found


def test_every_python_m_agent6_spawn_uses_safe_path() -> None:
    spawns = _python_m_agent6_spawns()
    # A zero match means the finder broke, not that the tree is clean.
    assert len(spawns) >= 2, f"expected the machine-agent + tui co-processes, found {spawns}"
    for file, lineno, strs in spawns:
        assert "-P" in strs, f"{file}:{lineno} spawns `python -m agent6` without -P"
        assert strs.index("-P") < strs.index("-m"), f"{file}:{lineno} -P must precede -m"
