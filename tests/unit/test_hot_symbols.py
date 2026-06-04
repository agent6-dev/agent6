# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for hot-symbol enumeration."""

from __future__ import annotations

from pathlib import Path

from agent6.tools.index import SymbolIndex


def test_hot_symbols_finds_cross_file_referenced_symbol(tmp_path: Path) -> None:
    """A function called from multiple files should rank high."""
    (tmp_path / "a.py").write_text(
        "def shared_func(x):\n    return x + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "from a import shared_func\n\nresult = shared_func(2)\n",
        encoding="utf-8",
    )
    (tmp_path / "c.py").write_text(
        "from a import shared_func\n\nresult = shared_func(3)\n",
        encoding="utf-8",
    )
    idx = SymbolIndex(tmp_path)
    hot = idx.hot_symbols(min_files_referenced=2, max_symbols=10)
    names = [t[0] for t in hot]
    assert "shared_func" in names
    # shared_func should report at least 3 referenced files (a, b, c).
    entry = next(t for t in hot if t[0] == "shared_func")
    _name, kind, path, _line, n_files = entry
    assert kind == "function"
    assert path == "a.py"
    assert n_files >= 3


def test_hot_symbols_respects_min_files_referenced(tmp_path: Path) -> None:
    """Symbols only referenced in their own file should not appear."""
    (tmp_path / "only_a.py").write_text(
        "def local_helper():\n    return 1\n\nresult = local_helper()\n",
        encoding="utf-8",
    )
    idx = SymbolIndex(tmp_path)
    hot = idx.hot_symbols(min_files_referenced=2)
    names = [t[0] for t in hot]
    assert "local_helper" not in names


def test_hot_symbols_excludes_external_names(tmp_path: Path) -> None:
    """References without a corresponding def in the index (e.g. stdlib
    names) should be skipped - the planner can't action them."""
    (tmp_path / "a.py").write_text(
        "import os\nprint(os.path.join('a', 'b'))\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "import os\nprint(os.environ)\n",
        encoding="utf-8",
    )
    idx = SymbolIndex(tmp_path)
    hot = idx.hot_symbols(min_files_referenced=1)
    names = [t[0] for t in hot]
    # 'os' is referenced 2x across 2 files but has NO def in the index;
    # must be excluded.
    assert "os" not in names


def test_hot_symbols_caps_results(tmp_path: Path) -> None:
    """max_symbols truncates the result to top-K."""
    # Create 5 cross-file-referenced functions.
    for i in range(5):
        (tmp_path / f"def_{i}.py").write_text(
            f"def func_{i}():\n    return {i}\n",
            encoding="utf-8",
        )
    (tmp_path / "uses_all.py").write_text(
        "\n".join(f"from def_{i} import func_{i}" for i in range(5))
        + "\n"
        + "\n".join(f"x = func_{i}()" for i in range(5))
        + "\n",
        encoding="utf-8",
    )
    idx = SymbolIndex(tmp_path)
    hot = idx.hot_symbols(min_files_referenced=2, max_symbols=3)
    assert len(hot) == 3


def test_hot_symbols_returns_empty_on_no_qualifying_symbols(tmp_path: Path) -> None:
    """A repo with only file-local symbols returns []."""
    (tmp_path / "only.py").write_text("x = 1\n", encoding="utf-8")
    idx = SymbolIndex(tmp_path)
    assert idx.hot_symbols(min_files_referenced=2) == []
