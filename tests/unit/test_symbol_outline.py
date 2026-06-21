# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the symbol-outline repo-priors block."""

from __future__ import annotations

from pathlib import Path

from agent6.tools.index import Symbol
from agent6.workflows._symbol_outline import (
    SYMBOL_OUTLINE_MAX_FILES as _SYMBOL_OUTLINE_MAX_FILES,
)
from agent6.workflows._symbol_outline import (
    SYMBOL_OUTLINE_MAX_PER_FILE as _SYMBOL_OUTLINE_MAX_PER_FILE,
)
from agent6.workflows._symbol_outline import (
    build_symbol_outline_block as _build_symbol_outline_block,
)


def test_outline_block_empty_when_no_outlines(tmp_path: Path) -> None:
    assert _build_symbol_outline_block({}, root=tmp_path) == ""


def test_outline_block_renders_simple_outline(tmp_path: Path) -> None:
    f = tmp_path / "pkg" / "mod.py"
    f.parent.mkdir(parents=True)
    f.touch()
    syms = [
        Symbol(name="Foo", kind="class", path=f, line=0, col=0),
        Symbol(name="bar", kind="function", path=f, line=10, col=0),
    ]
    out = _build_symbol_outline_block({f: syms}, root=tmp_path)
    assert "pkg/mod.py:" in out
    # Line numbers are rendered 1-based in the block.
    assert "class Foo:1" in out
    assert "function bar:11" in out


def test_outline_block_caps_per_file(tmp_path: Path) -> None:
    f = tmp_path / "huge.py"
    f.touch()
    syms = [
        Symbol(name=f"fn{i}", kind="function", path=f, line=i, col=0)
        for i in range(_SYMBOL_OUTLINE_MAX_PER_FILE + 7)
    ]
    out = _build_symbol_outline_block({f: syms}, root=tmp_path)
    assert "... (+7 more)" in out
    # First _SYMBOL_OUTLINE_MAX_PER_FILE function names are listed.
    for i in range(_SYMBOL_OUTLINE_MAX_PER_FILE):
        assert f"function fn{i}:" in out


def test_outline_block_drops_paths_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere.py"
    syms = [Symbol(name="x", kind="function", path=outside, line=0, col=0)]
    out = _build_symbol_outline_block({outside: syms}, root=tmp_path)
    assert out == ""


def test_outline_block_caps_total_file_count(tmp_path: Path) -> None:
    outlines: dict[Path, list[Symbol]] = {}
    n = _SYMBOL_OUTLINE_MAX_FILES + 5
    for i in range(n):
        f = tmp_path / f"f{i:03d}.py"
        f.touch()
        outlines[f] = [Symbol(name="x", kind="function", path=f, line=0, col=0)]
    out = _build_symbol_outline_block(outlines, root=tmp_path)
    assert "more files" in out
