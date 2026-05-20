# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.tools.code_index`.

The LSP tests need `pyright-langserver` on PATH and are gated by the
`needs_pyright` marker; they auto-skip otherwise. The outline tests are
pure-Python and always run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent6.tools.code_index import LspClient, outline

_HAS_PYRIGHT = shutil.which("pyright-langserver") is not None
needs_pyright = pytest.mark.skipif(not _HAS_PYRIGHT, reason="pyright-langserver not on PATH")


# ---------------------------------------------------------------------------
# Outline (no LSP)
# ---------------------------------------------------------------------------


def test_outline_returns_top_level_defs_and_classes(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text(
        "import os\n"
        "\n"
        "def alpha():\n"
        "    def nested():\n"
        "        pass\n"
        "    return nested\n"
        "\n"
        "async def beta(x):\n"
        "    return x\n"
        "\n"
        "class Gamma:\n"
        "    def method(self): ...\n",
        encoding="utf-8",
    )
    entries = outline(src)
    names = [(e.kind, e.name, e.line) for e in entries]
    assert names == [
        ("def", "alpha", 2),
        ("def", "beta", 7),
        ("class", "Gamma", 10),
    ]


def test_outline_skips_non_python(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("def not_python(): pass\n", encoding="utf-8")
    assert outline(src) == []


def test_outline_handles_empty_file(tmp_path: Path) -> None:
    src = tmp_path / "empty.py"
    src.write_text("", encoding="utf-8")
    assert outline(src) == []


# ---------------------------------------------------------------------------
# LSP
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # Tiny but realistic: `target.py` defines `foo`; `caller.py` imports and
    # calls it. Definition jumps from the call site to the def line; references
    # query at the def returns the call site.
    (tmp_path / "target.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    (tmp_path / "caller.py").write_text(
        "from target import foo\n\n\ndef main():\n    return foo()\n", encoding="utf-8"
    )
    # Make sure pyright treats this as a project root.
    (tmp_path / "pyrightconfig.json").write_text("{}\n", encoding="utf-8")
    return tmp_path


@needs_pyright
def test_lsp_definition_jumps_to_target(repo: Path) -> None:
    client = LspClient.start(repo)
    try:
        # `foo()` is at line 4, column 11 in caller.py (0-indexed).
        locs = client.definition(repo / "caller.py", line=4, character=11)
    finally:
        client.shutdown()
    assert locs, "expected at least one definition location"
    assert any(loc.path == (repo / "target.py").resolve() and loc.line == 0 for loc in locs)


@needs_pyright
def test_lsp_references_finds_call_site(repo: Path) -> None:
    client = LspClient.start(repo)
    try:
        # Open both files so pyright has them in the workspace model before
        # the references query runs.
        client.definition(repo / "caller.py", line=4, character=11)
        # `def foo` is at line 0, column 4 in target.py.
        locs = client.references(repo / "target.py", line=0, character=4, include_decl=False)
    finally:
        client.shutdown()
    # The call site `foo()` in caller.py should be there.
    paths = {loc.path for loc in locs}
    assert (repo / "caller.py").resolve() in paths
