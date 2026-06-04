# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for `agent6.tools.index.SymbolIndex`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent6.tools.index import Reference, Symbol, SymbolIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def py_project(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        textwrap.dedent(
            """\
            def foo(x):
                return x

            class Bar:
                def baz(self):
                    return foo(1)

            CONST = foo(2)
            """
        )
    )
    (tmp_path / "b.py").write_text(
        textwrap.dedent(
            """\
            from a import foo, Bar

            def caller():
                # foo is called here too
                return foo(Bar().baz())
            """
        )
    )
    # An excluded path that must not pollute the index.
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("def should_not_appear(): pass\n")
    return tmp_path


# ---------------------------------------------------------------------------
# outline()
# ---------------------------------------------------------------------------


def test_outline_python_top_level_and_nested(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    syms = idx.outline(py_project / "a.py")
    by_name = {(s.name, s.kind): s for s in syms}
    assert ("foo", "function") in by_name
    assert ("Bar", "class") in by_name
    # nested method is captured by the function_definition rule
    assert ("baz", "function") in by_name
    # Source-order
    assert [s.name for s in syms] == ["foo", "Bar", "baz"]


def test_outline_returns_empty_for_unknown_extension(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("# hello\n")
    idx = SymbolIndex(tmp_path)
    assert idx.outline(tmp_path / "x.md") == []


def test_outline_returns_empty_for_missing_file(tmp_path: Path) -> None:
    idx = SymbolIndex(tmp_path)
    assert idx.outline(tmp_path / "nope.py") == []


# ---------------------------------------------------------------------------
# find_definition()
# ---------------------------------------------------------------------------


def test_find_definition_locates_single_def(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    defs = idx.find_definition("Bar")
    assert len(defs) == 1
    assert defs[0].kind == "class"
    assert defs[0].path == (py_project / "a.py").resolve()
    assert defs[0].line == 3


def test_find_definition_returns_empty_for_unknown_name(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    assert idx.find_definition("definitely_does_not_exist") == []


def test_find_definition_skips_excluded_dirs(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    assert idx.find_definition("should_not_appear") == []


# ---------------------------------------------------------------------------
# find_references()
# ---------------------------------------------------------------------------


def test_find_references_filters_comments_and_strings(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        textwrap.dedent(
            """\
            def foo():
                return 1

            # foo in a comment must not count
            s = "foo in a string must not count either"
            foo()
            """
        )
    )
    idx = SymbolIndex(tmp_path)
    refs = idx.find_references("foo")
    # Two: definition site + the call.
    assert len(refs) == 2
    assert all(isinstance(r, Reference) for r in refs)
    lines = sorted(r.line for r in refs)
    assert lines == [0, 5]


def test_find_references_spans_files(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    refs = idx.find_references("foo")
    paths = {r.path for r in refs}
    assert (py_project / "a.py").resolve() in paths
    assert (py_project / "b.py").resolve() in paths


# ---------------------------------------------------------------------------
# Incremental invalidation
# ---------------------------------------------------------------------------


def test_mark_changed_picks_up_new_symbols(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    assert idx.find_definition("brand_new") == []
    target = py_project / "a.py"
    target.write_text(target.read_text() + "\ndef brand_new():\n    return 0\n")
    idx.mark_changed(target)
    defs = idx.find_definition("brand_new")
    assert len(defs) == 1
    assert defs[0].kind == "function"


def test_mark_deleted_drops_file_from_index(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    # Prime the index
    assert idx.find_definition("Bar")
    (py_project / "a.py").unlink()
    idx.mark_deleted(py_project / "a.py")
    assert idx.find_definition("Bar") == []


def test_mark_changed_on_deleted_file_silently_removes(py_project: Path) -> None:
    idx = SymbolIndex(py_project)
    assert idx.find_definition("Bar")
    target = py_project / "a.py"
    target.unlink()
    # Caller uses mark_changed for both edits and deletes by mistake; the
    # next refresh should notice the file is gone and drop it.
    idx.mark_changed(target)
    assert idx.find_definition("Bar") == []


def test_lazy_initial_scan_is_deferred_until_first_query(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def x(): pass\n")
    idx = SymbolIndex(tmp_path)
    # Before any query, internal caches are empty.
    assert not idx._symbols  # pyright: ignore[reportPrivateUsage]
    idx.outline(tmp_path / "a.py")
    assert idx._symbols  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Multi-language sanity
# ---------------------------------------------------------------------------


def test_rust_outline(tmp_path: Path) -> None:
    (tmp_path / "lib.rs").write_text(
        textwrap.dedent(
            """\
            pub fn greet(name: &str) -> String {
                format!("hi {name}")
            }

            pub struct Greeter {
                pub prefix: String,
            }

            pub trait Greet {
                fn greet(&self) -> String;
            }
            """
        )
    )
    idx = SymbolIndex(tmp_path)
    syms = idx.outline(tmp_path / "lib.rs")
    by_kind = {(s.name, s.kind) for s in syms}
    assert ("greet", "function") in by_kind
    assert ("Greeter", "struct") in by_kind
    assert ("Greet", "trait") in by_kind


def test_typescript_outline(tmp_path: Path) -> None:
    (tmp_path / "a.ts").write_text(
        textwrap.dedent(
            """\
            export function greet(name: string): string {
                return `hi ${name}`;
            }

            export class Greeter {
                greet(name: string): string {
                    return greet(name);
                }
            }

            export interface Named {
                name: string;
            }

            export type Maybe<T> = T | null;
            """
        )
    )
    idx = SymbolIndex(tmp_path)
    syms = idx.outline(tmp_path / "a.ts")
    by_kind = {(s.name, s.kind) for s in syms}
    assert ("greet", "function") in by_kind
    assert ("Greeter", "class") in by_kind
    assert ("greet", "method") in by_kind
    assert ("Named", "interface") in by_kind
    assert ("Maybe", "type") in by_kind


# ---------------------------------------------------------------------------
# Symbol value type
# ---------------------------------------------------------------------------


def test_symbol_is_frozen_dataclass() -> None:
    s = Symbol(name="x", kind="function", path=Path("/tmp/x.py"), line=0, col=0)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        s.name = "y"  # type: ignore[misc]
